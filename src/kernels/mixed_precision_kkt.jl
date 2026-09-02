#=====================================================================
    Guarded mixed-precision KKT factorization for dense extended-precision
    SDPs (`BigFloat` and fixed-width types such as `Float64x4`).

    The expensive Schur and equality-complement factorizations are carried
    out in Float64. Every residual, correction accumulation, and acceptance
    check remains in the requested target arithmetic. Conservative
    conditioning and refinement-count guards reject systems for which
    Float64 iterative refinement is unlikely to reach the working precision.
    A failed guard or stalled refinement may first promote the preconditioner
    to an extension-provided intermediate arithmetic. Acceptance still uses
    target-precision residuals; failure at that rung lazily falls back to the
    native extended-precision factorization.

    Automatic mode is the default for BigFloat and fixed-width extended
    arithmetic, but the crossover and numerical guards reject it unless a
    clear benefit is predicted. It never handles ordinary Float64,
    fixed-count refinement, or rank-deficient equality complements.
    Singleton-local `2x2` block-arrow systems use their separate Float64x4
    reduced-panel preconditioner when the MultiFloats extension is available.
=====================================================================#

const MIXED_KKT_PREDICTOR_RESIDUAL_LIMIT = 1.0e-8
const MIXED_KKT_CONTRACTION_SAFETY = 8.0
const MIXED_KKT_MINIMUM_AUTO_DIMENSION = 256
const MIXED_KKT_FALLBACK_COOLDOWN = 2
const MIXED_KKT_MAX_DYNAMIC_FALLBACKS = 2
const MIXED_KKT_MAX_STATIC_REJECTIONS = 3
const MIXED_KKT_FLOAT64_REGULARIZATION_ATTEMPTS = 8

"""
    mixed_intermediate_arithmetic(T)

Return an optional arithmetic type between `Float64` and `T` for a guarded
KKT fallback. Extensions may specialize this hook; the core package never
loads an optional arithmetic dependency merely to construct the workspace.
"""
mixed_intermediate_arithmetic(::Type) = Nothing

struct IntermediateCholeskyFactor{M<:AbstractMatrix}
    L::M
end

mutable struct IntermediatePrecisionKKTWorkspace{U}
    S::Matrix{U}
    Btil::Matrix{U}
    equality_scale::Vector{U}
    Q::Matrix{U}
    r::Vector{U}
    p::Vector{U}
    dx::Vector{U}
    dy::Vector{U}
    # Narrowed from `Any` (TASK-P0-TYPED-CORE): `_blocked_intermediate_cholesky!`
    # returns `IntermediateCholeskyFactor`, `_factor_float64_preconditioner!`
    # returns `LinearAlgebra.Cholesky`, both exposing `.L`; `nothing` is the
    # pre-factorization state.
    Sfactor::Union{Nothing,IntermediateCholeskyFactor,LinearAlgebra.Cholesky}
    Qfactor::Union{Nothing,IntermediateCholeskyFactor,LinearAlgebra.Cholesky}
    thread_count::Int
end

mutable struct MixedPrecisionKKTWorkspace
    mode::Symbol
    active::Bool
    attempted::Bool
    fell_back::Bool
    disabled::Bool
    reason::Symbol
    factor_attempt_count::Int
    dynamic_fallback_count::Int
    static_rejection_count::Int
    last_static_rejection::Symbol
    cooldown_remaining::Int
    condition_estimate::Float64
    predicted_refinement_steps::Int
    predictor_refinement_steps::Int
    float64_regularization_attempts::Int
    intermediate_factor_attempts::Int
    intermediate_refinement_steps::Int
    intermediate_factor_seconds::Float64
    intermediate_solve_seconds::Float64
    native_regularization_attempts::Int
    S64::Matrix{Float64}
    Btil64::Matrix{Float64}
    equality_scale64::Vector{Float64}
    Q64::Matrix{Float64}
    r64::Vector{Float64}
    p64::Vector{Float64}
    dx64::Vector{Float64}
    dy64::Vector{Float64}
    Sfactor::Union{Nothing,IntermediateCholeskyFactor,LinearAlgebra.Cholesky}
    Qfactor::Union{Nothing,IntermediateCholeskyFactor,LinearAlgebra.Cholesky}
    intermediate::Union{Nothing,IntermediatePrecisionKKTWorkspace}
    intermediate_active::Bool
end

function _mixed_precision_kkt_diagnostics(ws)
    ws.mixed_precision === nothing ||
        return _mixed_precision_kkt_diagnostics(ws.mixed_precision)
    arrow = ws.arrow
    if arrow !== nothing && arrow.mixed_source_cons !== nothing
        active = arrow.mixed_reduced_ready
        fell_back = arrow.mixed_reduced_fallback_count > 0
        return (
            available=true,
            backend=:float64x4_reduced_arrow,
            mode=arrow.mixed_reduced_mode,
            active=active,
            attempted=arrow.mixed_reduced_attempt_count > 0,
            fell_back=fell_back,
            disabled=!arrow.mixed_reduced_enabled,
            reason=arrow.mixed_reduced_reason,
            factor_attempt_count=arrow.mixed_reduced_attempt_count,
            dynamic_fallback_count=arrow.mixed_reduced_fallback_count,
            static_rejection_count=0,
            cooldown_remaining=0,
            condition_estimate=NaN,
            predicted_refinement_steps=0,
            predictor_refinement_steps=0,
            float64_regularization_attempts=0,
            intermediate_factor_attempts=0,
            intermediate_refinement_steps=0,
            intermediate_factor_seconds=0.0,
            intermediate_solve_seconds=0.0,
            intermediate_arithmetic=Nothing,
            intermediate_storage_bytes=0,
            native_regularization_attempts=0,
            threads=arrow.mixed_reduced_threads,
        )
    end
    return (available=false,)
end

function _mixed_precision_kkt_diagnostics(
    mixed::MixedPrecisionKKTWorkspace,
)
    intermediate = mixed.intermediate
    intermediate_arithmetic =
        intermediate === nothing ?
        Nothing : eltype(intermediate.S)
    intermediate_storage_bytes =
        intermediate === nothing ?
        0 :
        _intermediate_precision_storage_bytes(
            intermediate_arithmetic,
            size(intermediate.S, 1),
            size(intermediate.Q, 1),
        )
    return (
        available=true,
        backend=mixed.intermediate_active ?
                :intermediate_dense : :float64_dense,
        mode=mixed.mode,
        active=mixed.active,
        attempted=mixed.attempted,
        fell_back=mixed.fell_back,
        disabled=mixed.disabled,
        reason=mixed.reason,
        factor_attempt_count=mixed.factor_attempt_count,
        dynamic_fallback_count=mixed.dynamic_fallback_count,
        static_rejection_count=mixed.static_rejection_count,
        cooldown_remaining=mixed.cooldown_remaining,
        condition_estimate=mixed.condition_estimate,
        predicted_refinement_steps=mixed.predicted_refinement_steps,
        predictor_refinement_steps=mixed.predictor_refinement_steps,
        float64_regularization_attempts=
            mixed.float64_regularization_attempts,
        intermediate_factor_attempts=
            mixed.intermediate_factor_attempts,
        intermediate_refinement_steps=
            mixed.intermediate_refinement_steps,
        intermediate_factor_seconds=
            mixed.intermediate_factor_seconds,
        intermediate_solve_seconds=
            mixed.intermediate_solve_seconds,
        intermediate_arithmetic,
        intermediate_storage_bytes,
        native_regularization_attempts=
            mixed.native_regularization_attempts,
    )
end

function _mixed_precision_storage_bytes(m::Int, n::Int)
    elements =
        m * m + m * n + n * n +
        3m + 3n
    return Base.checked_mul(elements, sizeof(Float64))
end

function _mixed_precision_workspace_decision(
    prob::SDPProblem{T},
    mode::Symbol,
    memory_fraction::Float64;
    available_memory_bytes::Integer=_available_memory_bytes(),
) where {T}
    arithmetic = _arithmetic_class(T)
    arithmetic in (:bigfloat, :fixed_extended) || return (
        enabled=false,
        reason=:unsupported_arithmetic,
        required_bytes=0,
        memory_limit_bytes=0,
    )
    mode === :off && return (
        enabled=false,
        reason=:disabled,
        required_bytes=0,
        memory_limit_bytes=0,
    )
    prob.structure.schur_backend === :dense_cholesky || return (
        enabled=false,
        reason=:unsupported_schur_backend,
        required_bytes=0,
        memory_limit_bytes=0,
    )
    _, m, n, _ = prob.dims
    m > 0 || return (
        enabled=false,
        reason=:empty_system,
        required_bytes=0,
        memory_limit_bytes=0,
    )
    required = try
        _mixed_precision_storage_bytes(m, n)
    catch error
        error isa OverflowError || rethrow()
        return (
            enabled=false,
            reason=:storage_overflow,
            required_bytes=typemax(Int),
            memory_limit_bytes=0,
        )
    end
    available = Int(available_memory_bytes)
    memory_limit = available > 0 ?
                   floor(Int, available * memory_fraction) : 0
    (memory_limit > 0 && required <= memory_limit) || return (
        enabled=false,
        reason=available > 0 ? :memory_budget : :memory_unknown,
        required_bytes=required,
        memory_limit_bytes=memory_limit,
    )
    mode === :auto && m < MIXED_KKT_MINIMUM_AUTO_DIMENSION && return (
        enabled=false,
        reason=:below_auto_dimension,
        required_bytes=required,
        memory_limit_bytes=memory_limit,
    )
    return (
        enabled=true,
        reason=:selected,
        required_bytes=required,
        memory_limit_bytes=memory_limit,
    )
end

function _mixed_precision_workspace(
    prob::SDPProblem{T},
    mode::Symbol,
    memory_fraction::Float64;
    decision=nothing,
) where {T}
    resolved = decision === nothing ?
               _mixed_precision_workspace_decision(
                   prob,
                   mode,
                   memory_fraction,
               ) : decision
    resolved.enabled || return nothing
    arithmetic = _arithmetic_class(T)
    arithmetic in (:bigfloat, :fixed_extended) || error(
        "planned mixed-precision route has unsupported arithmetic $T",
    )
    mode === :off && error(
        "planned mixed-precision route cannot use mode=:off",
    )
    prob.structure.schur_backend === :dense_cholesky || error(
        "planned mixed-precision route requires a dense Schur backend",
    )
    L, m, n, k = prob.dims
    m > 0 || error("planned mixed-precision route has an empty system")

    return MixedPrecisionKKTWorkspace(
        mode,
        false,
        false,
        false,
        false,
        :not_attempted,
        0,
        0,
        0,
        :none,
        0,
        Inf,
        typemax(Int),
        0,
        0,
        0,
        0,
        0.0,
        0.0,
        0,
        zeros(Float64, m, m),
        zeros(Float64, m, n),
        zeros(Float64, n),
        zeros(Float64, n, n),
        zeros(Float64, m),
        zeros(Float64, n),
        zeros(Float64, m),
        zeros(Float64, n),
        nothing,
        nothing,
        nothing,
        false,
    )
end

function _record_static_mixed_rejection!(
    mixed::MixedPrecisionKKTWorkspace,
    reason::Symbol;
    permanent::Bool=false,
)
    mixed.active = false
    mixed.fell_back = true
    if mixed.last_static_rejection === reason
        mixed.static_rejection_count += 1
    else
        mixed.static_rejection_count = 1
        mixed.last_static_rejection = reason
    end
    if permanent ||
       mixed.static_rejection_count >= MIXED_KKT_MAX_STATIC_REJECTIONS
        mixed.disabled = true
        mixed.cooldown_remaining = 0
        mixed.reason = :disabled_after_repeated_static_rejection
    else
        mixed.cooldown_remaining = MIXED_KKT_FALLBACK_COOLDOWN
        mixed.reason = reason
    end
    return false
end

@inline function _copy_float64_checked!(
    destination::AbstractArray{Float64},
    source::AbstractArray{T},
) where {T}
    @inbounds for index in eachindex(destination, source)
        value = Float64(source[index])
        isfinite(value) || return false
        destination[index] = value
    end
    return true
end

@inline function _copy_extended_owned!(
    destination::AbstractVector{T},
    source::AbstractVector{Float64},
) where {T}
    @inbounds for index in eachindex(destination, source)
        destination[index] = T(source[index])
    end
    return destination
end

@inline function _copy_extended_owned!(
    destination::AbstractVector{BigFloat},
    source::AbstractVector{Float64},
)
    @inbounds for index in eachindex(destination, source)
        _mpfr_set_float64!(destination[index], source[index])
    end
    return destination
end

function _triangular_condition_estimate(
    factor::LinearAlgebra.Cholesky{Float64,<:AbstractMatrix{Float64}},
)
    # LAPACK `xTRCON` estimates reciprocal triangular condition numbers in
    # O(n²) work and O(n) scratch; it does not form an inverse. Since
    # A = L*L', cond₁(A) <= cond₁(L)*cond∞(L). Using both estimates is a
    # conservative guard and remains negligible beside the O(n³) Cholesky.
    reciprocal_one, reciprocal_infinity = try
        (
            LinearAlgebra.LAPACK.trcon!('O', 'L', 'N', factor.factors),
            LinearAlgebra.LAPACK.trcon!('I', 'L', 'N', factor.factors),
        )
    catch exception
        _recoverable(exception) || rethrow()
        (0.0, 0.0)
    end
    estimate =
        reciprocal_one > 0.0 && reciprocal_infinity > 0.0 ?
        inv(reciprocal_one * reciprocal_infinity) :
        Inf
    return isfinite(estimate) ? estimate : Inf
end

function _predicted_mixed_refinement_steps(
    ::Type{T},
    condition_estimate::Float64,
    relative_tolerance::T,
) where {T}
    contraction = min(
        0.95,
        condition_estimate *
        eps(Float64) *
        MIXED_KKT_CONTRACTION_SAFETY,
    )
    (isfinite(contraction) && 0.0 < contraction < 1.0) ||
        return typemax(Int)
    relative_tolerance > zero(T) || return typemax(Int)
    # Work in BigFloat so subnormal/underflow limits in Float64 do not turn a
    # valid high-precision target into zero.
    estimate = ceil(
        Int,
        log(BigFloat(relative_tolerance)) / log(BigFloat(contraction)),
    )
    return max(0, estimate)
end

function _mixed_refinement_relative_tolerance(
    opts::SolverOptions{T},
) where {T}
    opts.refine_tol > zero(T) && return opts.refine_tol
    arithmetic_floor = T(REFINE_DEFAULT_TOL_ULPS) * eps(T)
    solve_tolerance = min(
        opts.ϵ_gap,
        opts.ϵ_primal,
        opts.ϵ_dual,
    )
    # Driving a Float64 preconditioner to every bit of Float64x4 or BigFloat
    # is unnecessary when the requested outer certificate is much looser.
    # Squaring the requested tolerance leaves a substantial accuracy margin
    # while avoiding a late native O(n^3) fallback caused only by an
    # over-demanding refinement target. An explicit refine_tol still wins.
    requested_target = solve_tolerance * solve_tolerance
    return max(arithmetic_floor, requested_target)
end

function _factor_float64_preconditioner!(
    destination::Matrix{Float64},
    source::AbstractMatrix,
)
    _copy_float64_checked!(destination, source) ||
        return (factor=nothing, attempts=0, reason=:nonfinite_conversion)
    factor = LinearAlgebra.cholesky!(
        Symmetric(destination, :L);
        check=false,
    )
    LinearAlgebra.issuccess(factor) &&
        return (factor=factor, attempts=0, reason=:success)

    regularization = eps(Float64)
    for attempt in 1:MIXED_KKT_FLOAT64_REGULARIZATION_ATTEMPTS
        _copy_float64_checked!(destination, source) ||
            return (
                factor=nothing,
                attempts=attempt - 1,
                reason=:nonfinite_conversion,
            )
        @inbounds for index in axes(destination, 1)
            diagonal = destination[index, index]
            destination[index, index] +=
                regularization * max(abs(diagonal), 1.0)
        end
        factor = LinearAlgebra.cholesky!(
            Symmetric(destination, :L);
            check=false,
        )
        LinearAlgebra.issuccess(factor) &&
            return (
                factor=factor,
                attempts=attempt,
                reason=:success,
            )
        regularization *= 10.0
    end
    return (
        factor=nothing,
        attempts=MIXED_KKT_FLOAT64_REGULARIZATION_ATTEMPTS,
        reason=:not_positive_definite,
    )
end

function _factor_float64_equality_preconditioner!(
    mixed::MixedPrecisionKKTWorkspace,
)
    LinearAlgebra.mul!(
        mixed.Q64,
        transpose(mixed.Btil64),
        mixed.Btil64,
    )
    factor = LinearAlgebra.cholesky!(
        Symmetric(mixed.Q64, :L);
        check=false,
    )
    LinearAlgebra.issuccess(factor) &&
        return (factor=factor, attempts=0)

    regularization = eps(Float64)
    for attempt in 1:MIXED_KKT_FLOAT64_REGULARIZATION_ATTEMPTS
        LinearAlgebra.mul!(
            mixed.Q64,
            transpose(mixed.Btil64),
            mixed.Btil64,
        )
        @inbounds for index in axes(mixed.Q64, 1)
            diagonal = mixed.Q64[index, index]
            mixed.Q64[index, index] +=
                regularization * max(abs(diagonal), 1.0)
        end
        factor = LinearAlgebra.cholesky!(
            Symmetric(mixed.Q64, :L);
            check=false,
        )
        LinearAlgebra.issuccess(factor) &&
            return (factor=factor, attempts=attempt)
        regularization *= 10.0
    end
    return (
        factor=nothing,
        attempts=MIXED_KKT_FLOAT64_REGULARIZATION_ATTEMPTS,
    )
end

function _intermediate_precision_storage_bytes(
    ::Type{U},
    m::Int,
    n::Int,
) where {U}
    elements = m * m + m * n + n * n + 2m + 3n
    return Base.checked_mul(elements, sizeof(U))
end

function _intermediate_precision_workspace(
    ::Type{U},
    m::Int,
    n::Int,
    memory_fraction::Float64,
) where {U}
    isbitstype(U) || return nothing
    required = try
        _intermediate_precision_storage_bytes(U, m, n)
    catch error
        error isa OverflowError || rethrow()
        return nothing
    end
    available = _available_memory_bytes()
    memory_limit = available > 0 ?
                   floor(Int, available * memory_fraction) : 0
    (memory_limit > 0 && required <= memory_limit) || return nothing
    return IntermediatePrecisionKKTWorkspace{U}(
        alloc_zeros(U, m, m),
        alloc_zeros(U, m, n),
        alloc_zeros(U, n),
        alloc_zeros(U, n, n),
        alloc_zeros(U, m),
        alloc_zeros(U, n),
        alloc_zeros(U, m),
        alloc_zeros(U, n),
        nothing,
        nothing,
        1,
    )
end

@inline function _copy_intermediate_checked!(
    destination::AbstractArray{U},
    source::AbstractArray,
) where {U}
    @inbounds for index in eachindex(destination, source)
        value = U(source[index])
        isfinite(value) || return false
        destination[index] = value
    end
    return true
end

function _copy_intermediate_checked!(
    destination::AbstractArray{U},
    source::AbstractArray,
    thread_count::Int,
) where {U}
    length(destination) == length(source) ||
        throw(DimensionMismatch("intermediate-precision copy dimensions differ"))
    count = length(destination)
    workers = min(
        max(thread_count, 1),
        Threads.nthreads(),
        cld(count, 262_144),
    )
    workers <= 1 &&
        return _copy_intermediate_checked!(destination, source)
    valid = fill(UInt8(1), workers)
    @sync for worker in 1:workers
        first = (worker - 1) * count ÷ workers + 1
        last = worker * count ÷ workers
        Threads.@spawn begin
            worker_valid = true
            @inbounds for index in first:last
                value = U(source[index])
                if !isfinite(value)
                    worker_valid = false
                    break
                end
                destination[index] = value
            end
            valid[worker] = worker_valid ? UInt8(1) : UInt8(0)
        end
    end
    return all(!iszero, valid)
end

function _factor_intermediate_panel!(
    matrix::AbstractMatrix{U},
    first::Int,
    last::Int,
) where {U}
    @inbounds for column in first:last
        diagonal = matrix[column, column]
        for index in first:(column - 1)
            value = matrix[column, index]
            diagonal -= value * value
        end
        diagonal > zero(U) || return false
        pivot = sqrt(diagonal)
        inverse_pivot = inv(pivot)
        if last < size(matrix, 1)
            # Only the lower triangle is defined by this factorization. Cache
            # the panel reciprocals in disjoint upper-triangle scratch so the
            # following triangular solve remains allocation-free. The first
            # pivot uses the first cell immediately to the right of the panel;
            # every later pivot uses its upper-triangle slot in the first
            # panel row.
            if column == first
                matrix[first, last + 1] = inverse_pivot
            else
                matrix[first, column] = inverse_pivot
            end
        end
        matrix[column, column] = pivot
        for row in (column + 1):last
            value = matrix[row, column]
            for index in first:(column - 1)
                value -=
                    matrix[row, index] *
                    matrix[column, index]
            end
            matrix[row, column] = value * inverse_pivot
        end
    end
    return true
end

function _solve_intermediate_panel_rows!(
    matrix::AbstractMatrix{U},
    panel_first::Int,
    panel_last::Int,
    row_first::Int,
    row_last::Int,
) where {U}
    @inbounds for row in row_first:row_last
        for column in panel_first:panel_last
            value = matrix[row, column]
            for index in panel_first:(column - 1)
                value -=
                    matrix[row, index] *
                    matrix[column, index]
            end
            inverse_pivot = column == panel_first ?
                matrix[panel_first, panel_last + 1] :
                matrix[panel_first, column]
            matrix[row, column] = value * inverse_pivot
        end
    end
    return nothing
end

function _solve_intermediate_panel!(
    matrix::AbstractMatrix{U},
    panel_first::Int,
    panel_last::Int,
    thread_count::Int,
) where {U}
    first_row = panel_last + 1
    last_row = size(matrix, 1)
    first_row > last_row && return matrix
    rows = last_row - first_row + 1
    workers = min(
        max(thread_count, 1),
        Threads.nthreads(),
        cld(rows, 64),
    )
    if workers <= 1
        _solve_intermediate_panel_rows!(
            matrix,
            panel_first,
            panel_last,
            first_row,
            last_row,
        )
        return matrix
    end
    @sync for worker in 1:workers
        row_first =
            first_row + (worker - 1) * rows ÷ workers
        row_last =
            first_row + worker * rows ÷ workers - 1
        Threads.@spawn _solve_intermediate_panel_rows!(
            matrix,
            panel_first,
            panel_last,
            row_first,
            row_last,
        )
    end
    return matrix
end

function _blocked_cholesky_lower!(
    matrix::Matrix{U},
    thread_count::Int,
    panel_size::Int,
) where {U}
    dimension = size(matrix, 1)
    panel_size > 0 || throw(ArgumentError("panel size must be positive"))
    config = ExtendedPrecisionBLAS.KernelConfig(
        row_tile=64,
        column_tile=16,
        micro_tile=2,
    )
    for panel_first in 1:panel_size:dimension
        panel_last =
            min(panel_first + panel_size - 1, dimension)
        _factor_intermediate_panel!(
            matrix,
            panel_first,
            panel_last,
        ) || return false
        _solve_intermediate_panel!(
            matrix,
            panel_first,
            panel_last,
            thread_count,
        )
        panel_last == dimension && continue
        trailing = view(
            matrix,
            (panel_last + 1):dimension,
            (panel_last + 1):dimension,
        )
        panel = transpose(
            view(
                matrix,
                (panel_last + 1):dimension,
                panel_first:panel_last,
            ),
        )
        ExtendedPrecisionBLAS.syrk!(
            trailing,
            panel,
            -one(U),
            one(U),
            config,
            thread_count,
        )
    end
    return true
end

function _blocked_intermediate_cholesky!(
    matrix::Matrix{U},
    thread_count::Int,
    panel_size::Int,
) where {U}
    _blocked_cholesky_lower!(
        matrix,
        thread_count,
        panel_size,
    ) || return nothing
    return IntermediateCholeskyFactor(matrix)
end

_blocked_intermediate_cholesky!(
    matrix::Matrix{U},
    thread_count::Int,
) where {U} =
    _blocked_intermediate_cholesky!(matrix, thread_count, 64)

function _intermediate_trsm!(
    lower_factor::AbstractMatrix{U},
    right_hand_sides::AbstractMatrix{U},
    thread_count::Int,
) where {U}
    columns = size(right_hand_sides, 2)
    workers = min(
        max(thread_count, 1),
        Threads.nthreads(),
        cld(columns, 8),
    )
    if workers <= 1
        LinearAlgebra.ldiv!(
            LowerTriangular(lower_factor),
            right_hand_sides,
        )
        return right_hand_sides
    end
    @sync for worker in 1:workers
        first = (worker - 1) * columns ÷ workers + 1
        last = worker * columns ÷ workers
        Threads.@spawn LinearAlgebra.ldiv!(
            LowerTriangular(lower_factor),
            view(right_hand_sides, :, first:last),
        )
    end
    return right_hand_sides
end

function _factor_intermediate_once!(
    destination::Matrix{U},
    thread_count::Int,
) where {U}
    if size(destination, 1) >= 512 &&
       thread_count > 1 &&
       ExtendedPrecisionBLAS.arithmetic_family(U) ===
       :fixed_extended
        return _blocked_intermediate_cholesky!(
            destination,
            thread_count,
        )
    end
    factor = LinearAlgebra.cholesky!(
        Symmetric(destination, :L);
        check=false,
    )
    return LinearAlgebra.issuccess(factor) ? factor : nothing
end

function _factor_intermediate_matrix!(
    destination::Matrix{U},
    source::AbstractMatrix,
    thread_count::Int,
) where {U}
    _copy_intermediate_checked!(
        destination,
        source,
        thread_count,
    ) ||
        return nothing
    factor = _factor_intermediate_once!(
        destination,
        thread_count,
    )
    factor === nothing || return factor

    regularization = eps(U)
    for _ in 1:6
        _copy_intermediate_checked!(
            destination,
            source,
            thread_count,
        ) ||
            return nothing
        @inbounds for index in axes(destination, 1)
            diagonal = destination[index, index]
            destination[index, index] +=
                regularization * max(abs(diagonal), one(U))
        end
        factor = _factor_intermediate_once!(
            destination,
            thread_count,
        )
        factor === nothing || return factor
        regularization *= U(10)
    end
    return nothing
end

function _try_factor_intermediate_kkt!(
    mixed::MixedPrecisionKKTWorkspace,
    ws,
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    U = mixed_intermediate_arithmetic(T)
    U === Nothing && return false
    intermediate = mixed.intermediate
    if intermediate === nothing
        intermediate = _intermediate_precision_workspace(
            U,
            prob.dims.m,
            prob.dims.n,
            opts.mixed_precision_memory_fraction,
        )
        intermediate === nothing && return false
        mixed.intermediate = intermediate
    end
    intermediate isa IntermediatePrecisionKKTWorkspace{U} ||
        return false

    mixed.intermediate_factor_attempts += 1
    started = time_ns()
    schur_factor = _factor_intermediate_matrix!(
        intermediate.S,
        ws.S,
        opts.threads,
    )
    if schur_factor === nothing
        mixed.intermediate_factor_seconds +=
            _elapsed_seconds(started)
        return false
    end

    equality_factor = nothing
    n = prob.dims.n
    if n > 0
        _copy_intermediate_checked!(
            intermediate.Btil,
            prob.B,
            opts.threads,
        ) || return false
        _intermediate_trsm!(
            schur_factor.L,
            intermediate.Btil,
            opts.threads,
        )
        _normalize_equality_panel_columns!(
            intermediate.Btil,
            intermediate.equality_scale,
        )
        ExtendedPrecisionBLAS.syrk!(
            intermediate.Q,
            intermediate.Btil,
            one(U),
            zero(U),
            ExtendedPrecisionBLAS.KernelConfig(
                row_tile=64,
                column_tile=16,
                micro_tile=2,
            ),
            opts.threads,
        )
        equality_factor = LinearAlgebra.cholesky!(
            Symmetric(intermediate.Q, :L);
            check=false,
        )
        if !LinearAlgebra.issuccess(equality_factor)
            mixed.intermediate_factor_seconds +=
                _elapsed_seconds(started)
            return false
        end
    end

    intermediate.Sfactor = schur_factor
    intermediate.Qfactor = equality_factor
    intermediate.thread_count = min(
        max(opts.threads, 1),
        Threads.nthreads(),
    )
    mixed.intermediate_factor_seconds += _elapsed_seconds(started)
    mixed.intermediate_active = true
    mixed.active = true
    mixed.reason = :intermediate_active
    if opts.verbosity >= 1
        @info(
            "Mixed-precision KKT promoted its preconditioner before native fallback.",
            arithmetic=U,
            factor_seconds=mixed.intermediate_factor_seconds,
            factor_attempts=mixed.intermediate_factor_attempts,
        )
        flush(stderr)
    end
    return true
end

function _try_factor_mixed_kkt!(
    mixed::MixedPrecisionKKTWorkspace,
    ws,
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    mixed.active = false
    mixed.intermediate_active = false
    mixed.native_regularization_attempts = 0
    if mixed.disabled
        return false
    end
    if mixed.cooldown_remaining > 0
        mixed.cooldown_remaining -= 1
        mixed.reason = :fallback_cooldown
        return false
    end
    mixed.attempted = true
    mixed.factor_attempt_count += 1
    mixed.reason = :attempting
    # `:on` is an expert request to measure the actual correction behavior for
    # fixed-width extended arithmetic. The triangular condition estimate is a
    # deliberately conservative upper bound and becomes wildly pessimistic on
    # Task_Low08 even while one target-precision correction per direction is
    # sufficient. Keep the estimate diagnostic, but let the predictor residual
    # guard and monotone refinement checks make the safety decision. Automatic
    # mode and BigFloat retain the static condition cutoff.
    measured_trial =
        mixed.mode === :on &&
        _arithmetic_class(T) === :fixed_extended

    opts.refine_policy === :fixed && begin
        return _record_static_mixed_rejection!(
            mixed,
            :fixed_refinement_policy;
            permanent=true,
        )
    end
    ws.arrow === nothing || begin
        return _record_static_mixed_rejection!(
            mixed,
            :block_arrow_system;
            permanent=true,
        )
    end
    schur_result = _factor_float64_preconditioner!(
        mixed.S64,
        ws.S,
    )
    schur_result.reason === :nonfinite_conversion && begin
        return _record_static_mixed_rejection!(
            mixed,
            :nonfinite_conversion,
        )
    end
    mixed.float64_regularization_attempts += schur_result.attempts
    schur_factor = schur_result.factor
    schur_factor === nothing && begin
        return _record_static_mixed_rejection!(
            mixed,
            :float64_schur_not_positive_definite,
        )
    end
    condition_estimate = _triangular_condition_estimate(schur_factor)
    condition_estimate <= opts.mixed_precision_condition_limit ||
        measured_trial || begin
        mixed.condition_estimate = condition_estimate
        return _record_static_mixed_rejection!(
            mixed,
            :condition_limit,
        )
    end

    n = prob.dims.n
    equality_factor = nothing
    if n > 0
        _copy_float64_checked!(mixed.Btil64, prob.B) || begin
            return _record_static_mixed_rejection!(
                mixed,
                :nonfinite_conversion,
            )
        end
        LinearAlgebra.ldiv!(
            LowerTriangular(schur_factor.L),
            mixed.Btil64,
        )
        _normalize_equality_panel_columns!(
            mixed.Btil64,
            mixed.equality_scale64,
        )
        equality_result =
            _factor_float64_equality_preconditioner!(mixed)
        mixed.float64_regularization_attempts +=
            equality_result.attempts
        equality_factor = equality_result.factor
        equality_factor === nothing && begin
            return _record_static_mixed_rejection!(
                mixed,
                :float64_equality_complement_not_positive_definite,
            )
        end
        equality_condition = _triangular_condition_estimate(equality_factor)
        condition_estimate = max(condition_estimate, equality_condition)
        condition_estimate <= opts.mixed_precision_condition_limit ||
            measured_trial || begin
            mixed.condition_estimate = condition_estimate
            return _record_static_mixed_rejection!(
                mixed,
                :condition_limit,
            )
        end
    end

    relative_tolerance =
        _mixed_refinement_relative_tolerance(opts)
    predicted_steps = _predicted_mixed_refinement_steps(
        T,
        condition_estimate,
        relative_tolerance,
    )
    predicted_steps <= opts.mixed_precision_refine_max_steps ||
        measured_trial || begin
        mixed.condition_estimate = condition_estimate
        mixed.predicted_refinement_steps = predicted_steps
        return _record_static_mixed_rejection!(
            mixed,
            :refinement_budget,
        )
    end

    mixed.Sfactor = schur_factor
    mixed.Qfactor = equality_factor
    mixed.condition_estimate = condition_estimate
    mixed.predicted_refinement_steps = predicted_steps
    mixed.static_rejection_count = 0
    mixed.last_static_rejection = :none
    mixed.active = true
    mixed.reason = :active
    return true
end

function _solve_mixed_kkt!(
    mixed::MixedPrecisionKKTWorkspace,
    n::Int,
    r::AbstractVector{T},
    p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T}
    if mixed.intermediate_active
        started = time_ns()
        result = _solve_intermediate_kkt!(
            mixed.intermediate,
            n,
            r,
            p_rhs,
            dx_out,
            dy_out,
        )
        mixed.intermediate_solve_seconds +=
            _elapsed_seconds(started)
        return result
    end
    _copy_float64_checked!(mixed.r64, r) ||
        throw(ArgumentError("mixed-precision KKT right-hand side overflowed Float64"))
    LinearAlgebra.ldiv!(
        LowerTriangular(mixed.Sfactor.L),
        mixed.r64,
    )

    if n > 0
        _copy_float64_checked!(mixed.p64, p_rhs) ||
            throw(ArgumentError("mixed-precision equality right-hand side overflowed Float64"))
        @inbounds for index in eachindex(mixed.p64, mixed.equality_scale64)
            mixed.p64[index] *= mixed.equality_scale64[index]
        end
        LinearAlgebra.mul!(
            mixed.dy64,
            transpose(mixed.Btil64),
            mixed.r64,
        )
        @inbounds for index in eachindex(mixed.dy64)
            mixed.dy64[index] =
                mixed.p64[index] - mixed.dy64[index]
        end
        LinearAlgebra.ldiv!(mixed.Qfactor, mixed.dy64)
        LinearAlgebra.mul!(mixed.dx64, mixed.Btil64, mixed.dy64)
        @inbounds for index in eachindex(mixed.dx64)
            mixed.dx64[index] += mixed.r64[index]
        end
        LinearAlgebra.ldiv!(
            UpperTriangular(transpose(mixed.Sfactor.L)),
            mixed.dx64,
        )
        _recover_original_equality_multiplier!(
            mixed.dy64,
            mixed.equality_scale64,
        )
        _copy_extended_owned!(dy_out, mixed.dy64)
    else
        copyto!(mixed.dx64, mixed.r64)
        LinearAlgebra.ldiv!(
            UpperTriangular(transpose(mixed.Sfactor.L)),
            mixed.dx64,
        )
    end
    _copy_extended_owned!(dx_out, mixed.dx64)
    return dx_out, dy_out
end

function _solve_intermediate_kkt!(
    intermediate::IntermediatePrecisionKKTWorkspace{U},
    n::Int,
    r::AbstractVector{T},
    p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T,U}
    _copy_intermediate_checked!(intermediate.r, r) ||
        throw(ArgumentError("intermediate-precision KKT right-hand side overflowed"))
    LinearAlgebra.ldiv!(
        LowerTriangular(intermediate.Sfactor.L),
        intermediate.r,
    )

    if n > 0
        _copy_intermediate_checked!(intermediate.p, p_rhs) ||
            throw(ArgumentError("intermediate-precision equality right-hand side overflowed"))
        @inbounds for index in eachindex(
            intermediate.p,
            intermediate.equality_scale,
        )
            intermediate.p[index] *= intermediate.equality_scale[index]
        end
        _intermediate_transpose_matvec!(
            intermediate.dy,
            intermediate.Btil,
            intermediate.r,
            intermediate.thread_count,
        )
        @inbounds for index in eachindex(intermediate.dy)
            intermediate.dy[index] =
                intermediate.p[index] - intermediate.dy[index]
        end
        LinearAlgebra.ldiv!(
            intermediate.Qfactor,
            intermediate.dy,
        )
        _intermediate_matvec!(
            intermediate.dx,
            intermediate.Btil,
            intermediate.dy,
            intermediate.thread_count,
        )
        @inbounds for index in eachindex(intermediate.dx)
            intermediate.dx[index] += intermediate.r[index]
        end
        LinearAlgebra.ldiv!(
            UpperTriangular(transpose(intermediate.Sfactor.L)),
            intermediate.dx,
        )
        _recover_original_equality_multiplier!(
            intermediate.dy,
            intermediate.equality_scale,
        )
        @inbounds for index in eachindex(dy_out, intermediate.dy)
            dy_out[index] = T(intermediate.dy[index])
        end
    else
        copyto!(intermediate.dx, intermediate.r)
        LinearAlgebra.ldiv!(
            UpperTriangular(transpose(intermediate.Sfactor.L)),
            intermediate.dx,
        )
    end
    @inbounds for index in eachindex(dx_out, intermediate.dx)
        dx_out[index] = T(intermediate.dx[index])
    end
    return dx_out, dy_out
end

function _intermediate_transpose_matvec_range!(
    output::AbstractVector{U},
    matrix::AbstractMatrix{U},
    vector::AbstractVector{U},
    first::Int,
    last::Int,
) where {U}
    @inbounds for column in first:last
        accumulator = zero(U)
        for row in axes(matrix, 1)
            accumulator +=
                matrix[row, column] * vector[row]
        end
        output[column] = accumulator
    end
    return nothing
end

function _intermediate_transpose_matvec!(
    output::AbstractVector{U},
    matrix::AbstractMatrix{U},
    vector::AbstractVector{U},
    thread_count::Int,
) where {U}
    columns = size(matrix, 2)
    workers = min(
        max(thread_count, 1),
        Threads.nthreads(),
        cld(columns, 16),
    )
    if workers <= 1
        _intermediate_transpose_matvec_range!(
            output,
            matrix,
            vector,
            1,
            columns,
        )
        return output
    end
    @sync for worker in 1:workers
        first = (worker - 1) * columns ÷ workers + 1
        last = worker * columns ÷ workers
        Threads.@spawn _intermediate_transpose_matvec_range!(
            output,
            matrix,
            vector,
            first,
            last,
        )
    end
    return output
end

function _intermediate_matvec_range!(
    output::AbstractVector{U},
    matrix::AbstractMatrix{U},
    vector::AbstractVector{U},
    first::Int,
    last::Int,
) where {U}
    @inbounds for row in first:last
        accumulator = zero(U)
        for column in axes(matrix, 2)
            accumulator +=
                matrix[row, column] * vector[column]
        end
        output[row] = accumulator
    end
    return nothing
end

function _intermediate_matvec!(
    output::AbstractVector{U},
    matrix::AbstractMatrix{U},
    vector::AbstractVector{U},
    thread_count::Int,
) where {U}
    rows = size(matrix, 1)
    workers = min(
        max(thread_count, 1),
        Threads.nthreads(),
        cld(rows, 128),
    )
    if workers <= 1
        _intermediate_matvec_range!(
            output,
            matrix,
            vector,
            1,
            rows,
        )
        return output
    end
    @sync for worker in 1:workers
        first = (worker - 1) * rows ÷ workers + 1
        last = worker * rows ÷ workers
        Threads.@spawn _intermediate_matvec_range!(
            output,
            matrix,
            vector,
            first,
            last,
        )
    end
    return output
end

function _activate_native_extended_kkt!(
    ws,
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    reason::Symbol,
) where {T}
    mixed = ws.mixed_precision
    mixed === nothing && return _factor_dense_kkt_native!(ws, prob, opts)
    opts.verbosity >= 1 && @warn(
        "Mixed-precision KKT correction rejected; recomputing with the native target-precision factorization.",
        reason=reason,
        condition_estimate=mixed.condition_estimate,
        predicted_refinement_steps=mixed.predicted_refinement_steps,
        predictor_refinement_steps=mixed.predictor_refinement_steps,
        float64_regularization_attempts=
            mixed.float64_regularization_attempts,
        intermediate_factor_attempts=
            mixed.intermediate_factor_attempts,
        intermediate_refinement_steps=
            mixed.intermediate_refinement_steps,
    )
    opts.verbosity >= 1 && flush(stderr)
    mixed.active = false
    mixed.intermediate_active = false
    mixed.fell_back = true
    if mixed.disabled
        # Preserve the evidence that disabled the path. In particular, a
        # repeated static rejection must not later be mislabeled as a dynamic
        # refinement fallback.
        factor = _factor_dense_kkt_native!(ws, prob, opts)
        mixed.native_regularization_attempts = factor.reg_attempts
        return factor
    end
    if reason in (
        :predictor_residual_guard,
        :right_hand_side_conversion,
        :refinement_stalled,
    )
        mixed.dynamic_fallback_count += 1
        if mixed.dynamic_fallback_count >=
           MIXED_KKT_MAX_DYNAMIC_FALLBACKS
            mixed.disabled = true
            mixed.cooldown_remaining = 0
            mixed.reason = :disabled_after_repeated_fallback
        else
            mixed.cooldown_remaining = MIXED_KKT_FALLBACK_COOLDOWN
            mixed.reason = reason
        end
    else
        mixed.reason = reason
    end
    factor = _factor_dense_kkt_native!(ws, prob, opts)
    mixed.native_regularization_attempts = factor.reg_attempts
    return factor
end

function _mixed_kkt_relative_residual(
    ws,
    prob::SDPProblem{T},
    r::AbstractVector{T},
) where {T}
    residual = _kkt_direction_residual!(ws, prob, r)
    scale = max(
        knrmInf(r),
        prob.dims.n > 0 ? knrmInf(ws.p) : zero(T),
        one(T),
    )
    return residual / scale
end

function _refine_mixed_predictor_to_guard!(
    ws,
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    r::AbstractVector{T},
) where {T}
    mixed = ws.mixed_precision::MixedPrecisionKKTWorkspace
    scale = max(
        knrmInf(r),
        prob.dims.n > 0 ? knrmInf(ws.p) : zero(T),
        one(T),
    )
    target = T(MIXED_KKT_PREDICTOR_RESIDUAL_LIMIT) * scale
    residual = _kkt_direction_residual!(ws, prob, r)
    residual <= target && return true
    maximum_steps = min(
        mixed.predicted_refinement_steps,
        opts.mixed_precision_refine_max_steps,
    )
    for _ in 1:maximum_steps
        copy_owned!(ws.dx_best, ws.dx)
        prob.dims.n > 0 && copy_owned!(ws.dy_best, ws.dy)
        correction_ok = try
            _apply_kkt_correction!(ws, prob)
            true
        catch error
            error isa ArgumentError || rethrow()
            false
        end
        corrected = correction_ok ?
                    _kkt_direction_residual!(ws, prob, r) :
                    T(Inf)
        if !isfinite(corrected) || corrected >= residual
            copy_owned!(ws.dx, ws.dx_best)
            prob.dims.n > 0 && copy_owned!(ws.dy, ws.dy_best)
            return false
        end
        residual = corrected
        mixed.predictor_refinement_steps += 1
        residual <= target && return true
    end
    return false
end

function _solve_mixed_kkt_guarded!(
    ws,
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    r::AbstractVector{T},
) where {T}
    mixed = ws.mixed_precision
    if mixed === nothing || !mixed.active
        _solve_kkt_owned!(
            ws,
            prob.dims.n,
            r,
            ws.p,
            ws.dx,
            ws.dy,
        )
        return true
    end
    low_solve_ok = try
        _solve_kkt_owned!(
            ws,
            prob.dims.n,
            r,
            ws.p,
            ws.dx,
            ws.dy,
        )
        true
    catch error
        error isa ArgumentError || rethrow()
        false
    end
    if !low_solve_ok
        factor = _activate_native_extended_kkt!(
            ws,
            prob,
            opts,
            :right_hand_side_conversion,
        )
        factor.ok || return false
        _solve_kkt_owned!(
            ws,
            prob.dims.n,
            r,
            ws.p,
            ws.dx,
            ws.dy,
        )
        return true
    end
    relative_residual = _mixed_kkt_relative_residual(ws, prob, r)
    if isfinite(relative_residual) &&
       relative_residual <= T(MIXED_KKT_PREDICTOR_RESIDUAL_LIMIT)
        return true
    end
    if isfinite(relative_residual) &&
       _refine_mixed_predictor_to_guard!(ws, prob, opts, r)
        return true
    end
    if !mixed.intermediate_active &&
       _try_factor_intermediate_kkt!(mixed, ws, prob, opts)
        predictor_steps_before =
            mixed.predictor_refinement_steps
        intermediate_ok = try
            _solve_kkt_owned!(
                ws,
                prob.dims.n,
                r,
                ws.p,
                ws.dx,
                ws.dy,
            )
            intermediate_residual =
                _mixed_kkt_relative_residual(ws, prob, r)
            isfinite(intermediate_residual) &&
                (
                    intermediate_residual <=
                    T(MIXED_KKT_PREDICTOR_RESIDUAL_LIMIT) ||
                    _refine_mixed_predictor_to_guard!(
                        ws,
                        prob,
                        opts,
                        r,
                    )
                )
        catch error
            error isa ArgumentError || rethrow()
            false
        end
        mixed.intermediate_refinement_steps +=
            mixed.predictor_refinement_steps -
            predictor_steps_before
        intermediate_ok && return true
        mixed.intermediate_active = false
    end
    factor = _activate_native_extended_kkt!(
        ws,
        prob,
        opts,
        :predictor_residual_guard,
    )
    factor.ok || return false
    _solve_kkt_owned!(
        ws,
        prob.dims.n,
        r,
        ws.p,
        ws.dx,
        ws.dy,
    )
    return true
end

function _refine_with_active_intermediate!(
    ws,
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    r::AbstractVector{T},
    absolute_tolerance::T,
) where {T}
    intermediate_steps = 0
    residual = T(Inf)
    success = try
        _solve_kkt_owned!(
            ws,
            prob.dims.n,
            r,
            ws.p,
            ws.dx,
            ws.dy,
        )
        residual = _kkt_direction_residual!(ws, prob, r)
        for _ in 1:opts.mixed_precision_refine_max_steps
            residual <= absolute_tolerance && break
            copy_owned!(ws.dx_best, ws.dx)
            prob.dims.n > 0 &&
                copy_owned!(ws.dy_best, ws.dy)
            _apply_kkt_correction!(ws, prob)
            corrected =
                _kkt_direction_residual!(ws, prob, r)
            if !isfinite(corrected) ||
               corrected >= residual
                copy_owned!(ws.dx, ws.dx_best)
                prob.dims.n > 0 &&
                    copy_owned!(ws.dy, ws.dy_best)
                break
            end
            intermediate_steps += 1
            residual = corrected
        end
        residual <= absolute_tolerance
    catch error
        error isa ArgumentError || rethrow()
        residual = T(Inf)
        false
    end
    return success, intermediate_steps, residual
end

function _refine_mixed_direction!(
    ws,
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    r::AbstractVector{T},
) where {T}
    mixed = ws.mixed_precision::MixedPrecisionKKTWorkspace
    scale = max(
        knrmInf(r),
        prob.dims.n > 0 ? knrmInf(ws.p) : zero(T),
        one(T),
    )
    relative_tolerance =
        _mixed_refinement_relative_tolerance(opts)
    absolute_tolerance = relative_tolerance * scale
    residual = _kkt_direction_residual!(ws, prob, r)
    steps = 0

    for _ in 1:opts.mixed_precision_refine_max_steps
        residual <= absolute_tolerance && return (steps, residual)
        copy_owned!(ws.dx_best, ws.dx)
        prob.dims.n > 0 && copy_owned!(ws.dy_best, ws.dy)
        correction_ok = try
            _apply_kkt_correction!(ws, prob)
            true
        catch error
            error isa ArgumentError || rethrow()
            false
        end
        corrected = correction_ok ?
                    _kkt_direction_residual!(ws, prob, r) :
                    T(Inf)
        if !isfinite(corrected) || corrected >= residual
            copy_owned!(ws.dx, ws.dx_best)
            prob.dims.n > 0 && copy_owned!(ws.dy, ws.dy_best)
            break
        end
        steps += 1
        residual = corrected
    end

    if !mixed.intermediate_active &&
       _try_factor_intermediate_kkt!(mixed, ws, prob, opts)
        intermediate_ok, intermediate_steps, residual =
            _refine_with_active_intermediate!(
                ws,
                prob,
                opts,
                r,
                absolute_tolerance,
            )
        mixed.intermediate_refinement_steps +=
            intermediate_steps
        intermediate_ok &&
            return (steps + intermediate_steps, residual)
        mixed.intermediate_active = false
    end

    # Not reaching the requested residual target is never silently accepted.
    # Re-factor in native precision and recompute the direction from the
    # original right-hand side. This preserves the numerical behavior of the
    # established path at the cost of an unsuccessful low-precision attempt.
    factor = _activate_native_extended_kkt!(
        ws,
        prob,
        opts,
        :refinement_stalled,
    )
    factor.ok || return (steps, residual)
    _solve_kkt_owned!(
        ws,
        prob.dims.n,
        r,
        ws.p,
        ws.dx,
        ws.dy,
    )
    native_steps, residual = refine_direction!(ws, prob, opts, r)
    return (steps + native_steps, residual)
end
