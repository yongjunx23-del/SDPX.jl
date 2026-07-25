#=====================================================================
    Opt-in mixed-precision KKT factorization for dense extended-precision
    SDPs (`BigFloat` and fixed-width types such as `Float64x4`).

    The expensive Schur and equality-complement factorizations are carried
    out in Float64. Every residual, correction accumulation, and acceptance
    check remains in the requested target arithmetic. Conservative
    conditioning and refinement-count guards reject systems for which
    Float64 iterative refinement is unlikely to reach the working precision.
    A failed guard or stalled refinement lazily falls back to the native
    extended-precision factorization.

    This path is deliberately disabled by default. It never handles ordinary
    Float64, block-arrow systems, fixed-count refinement, or rank-deficient
    equality complements.
=====================================================================#

const MIXED_KKT_PREDICTOR_RESIDUAL_LIMIT = 1.0e-8
const MIXED_KKT_CONTRACTION_SAFETY = 8.0
const MIXED_KKT_MINIMUM_AUTO_DIMENSION = 256
const MIXED_KKT_FALLBACK_COOLDOWN = 2
const MIXED_KKT_MAX_DYNAMIC_FALLBACKS = 2
const MIXED_KKT_MAX_STATIC_REJECTIONS = 3

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
    native_regularization_attempts::Int
    S64::Matrix{Float64}
    Btil64::Matrix{Float64}
    Q64::Matrix{Float64}
    r64::Vector{Float64}
    p64::Vector{Float64}
    dx64::Vector{Float64}
    dy64::Vector{Float64}
    Sfactor::Any
    Qfactor::Any
end

_mixed_precision_kkt_diagnostics(ws) =
    ws.mixed_precision === nothing ?
    (available=false,) :
    _mixed_precision_kkt_diagnostics(ws.mixed_precision)

function _mixed_precision_kkt_diagnostics(
    mixed::MixedPrecisionKKTWorkspace,
)
    return (
        available=true,
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
        native_regularization_attempts=
            mixed.native_regularization_attempts,
    )
end

function _mixed_precision_storage_bytes(m::Int, n::Int)
    elements =
        m * m + m * n + n * n +
        3m + 2n
    return Base.checked_mul(elements, sizeof(Float64))
end

function _mixed_precision_workspace(
    prob::SDPProblem{T},
    mode::Symbol,
    memory_fraction::Float64,
) where {T}
    arithmetic = _arithmetic_class(T)
    arithmetic in (:bigfloat, :fixed_extended) || return nothing
    mode === :off && return nothing
    prob.structure.schur_backend === :dense_cholesky || return nothing
    L, m, n, k = prob.dims
    m > 0 || return nothing

    required = try
        _mixed_precision_storage_bytes(m, n)
    catch error
        error isa OverflowError || rethrow()
        return nothing
    end
    available = _available_memory_bytes()
    memory_limit = available > 0 ?
                   floor(Int, available * memory_fraction) : 0
    # Optional acceleration may not allocate from an unknown or inadequate
    # memory budget. `:on` requests an attempt; it does not override safety.
    (memory_limit > 0 && required <= memory_limit) || return nothing
    mode === :auto && m < MIXED_KKT_MINIMUM_AUTO_DIMENSION && return nothing

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
        zeros(Float64, m, m),
        zeros(Float64, m, n),
        zeros(Float64, n, n),
        zeros(Float64, m),
        zeros(Float64, n),
        zeros(Float64, m),
        zeros(Float64, n),
        nothing,
        nothing,
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
        converted = BigFloat(source[index])
        MA.operate_to!(destination[index], copy, converted)
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
    catch
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

function _try_factor_mixed_kkt!(
    mixed::MixedPrecisionKKTWorkspace,
    ws,
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    mixed.active = false
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
    _copy_float64_checked!(mixed.S64, ws.S) || begin
        return _record_static_mixed_rejection!(
            mixed,
            :nonfinite_conversion,
        )
    end

    schur_factor = LinearAlgebra.cholesky!(
        Symmetric(mixed.S64, :L);
        check=false,
    )
    LinearAlgebra.issuccess(schur_factor) || begin
        return _record_static_mixed_rejection!(
            mixed,
            :float64_schur_not_positive_definite,
        )
    end
    condition_estimate = _triangular_condition_estimate(schur_factor)
    condition_estimate <= opts.mixed_precision_condition_limit || begin
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
        LinearAlgebra.mul!(
            mixed.Q64,
            transpose(mixed.Btil64),
            mixed.Btil64,
        )
        equality_factor = LinearAlgebra.cholesky!(
            Symmetric(mixed.Q64, :L);
            check=false,
        )
        LinearAlgebra.issuccess(equality_factor) || begin
            return _record_static_mixed_rejection!(
                mixed,
                :float64_equality_complement_not_positive_definite,
            )
        end
        equality_condition = _triangular_condition_estimate(equality_factor)
        condition_estimate = max(condition_estimate, equality_condition)
        condition_estimate <= opts.mixed_precision_condition_limit || begin
            mixed.condition_estimate = condition_estimate
            return _record_static_mixed_rejection!(
                mixed,
                :condition_limit,
            )
        end
    end

    relative_tolerance = opts.refine_tol > zero(T) ?
                         opts.refine_tol :
                         T(REFINE_DEFAULT_TOL_ULPS) * eps(T)
    predicted_steps = _predicted_mixed_refinement_steps(
        T,
        condition_estimate,
        relative_tolerance,
    )
    predicted_steps <= opts.mixed_precision_refine_max_steps || begin
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
    _copy_float64_checked!(mixed.r64, r) ||
        throw(ArgumentError("mixed-precision KKT right-hand side overflowed Float64"))
    LinearAlgebra.ldiv!(
        LowerTriangular(mixed.Sfactor.L),
        mixed.r64,
    )

    if n > 0
        _copy_float64_checked!(mixed.p64, p_rhs) ||
            throw(ArgumentError("mixed-precision equality right-hand side overflowed Float64"))
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

function _activate_native_extended_kkt!(
    ws,
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    reason::Symbol,
) where {T}
    mixed = ws.mixed_precision
    mixed === nothing && return _factor_dense_kkt_native!(ws, prob, opts)
    mixed.active = false
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
    relative_tolerance = opts.refine_tol > zero(T) ?
                         opts.refine_tol :
                         T(REFINE_DEFAULT_TOL_ULPS) * eps(T)
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
