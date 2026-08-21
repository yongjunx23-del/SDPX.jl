#=====================================================================
    Main loop (§1.6 merges cold-start/warm-start/findFeasible into one
    `solve!`), termination (§5.1), restart repair (§5.2), checkpointing
    (§5.5).
=====================================================================#

function dual_objective(prob::SDPProblem{T}, y, Y) where {T}
    d = zero(T)
    for l in 1:prob.dims.L
        d += kdot(prob.C[l], Y[l])
    end
    prob.dims.n > 0 && (d += LinearAlgebra.dot(prob.b, y))
    return d
end

function print_header(opts::SolverOptions)
    opts.verbosity >= 1 || return
    println("iter\tprimal obj\tdual obj\tgap\t\tprimal res\tdual res\tprimal step\tdual step\ttime (s)")
    println("="^131)
    flush(stdout)
end

function print_iter(opts::SolverOptions{T}, iter, pObj, dObj, gap, p_res, d_res, tX=nothing, tY=nothing, dt=nothing) where {T}
    opts.verbosity >= 1 || return
    pf, df, gf, pr, dr = Float64(pObj), Float64(dObj), Float64(gap), Float64(p_res), Float64(d_res)
    if tX === nothing
        @printf "%d\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n" iter pf df gf pr dr
    else
        @printf "%d\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n" iter pf df gf pr dr Float64(tX) Float64(tY) dt
    end
    flush(stdout)
end

"""Classify a failed Newton step for the solve-level termination record.

`newton_step!` deliberately keeps its historical human-readable `reason`
strings (they are part of the returned message), while the typed result also
needs a stable machine-readable reason and stage.  Keep this mapping here at
the loop boundary so the numerical kernel and its successful trajectory are
unchanged.  The prefix checks are intentionally conservative: an unknown
future detail is reported as the generic `:newton_breakdown` rather than
silently falling back to `:none`.
"""
@inline function _sdp_newton_termination_metadata(detail)
    text = String(detail)
    if startswith(text, "Cholesky factorization")
        return (:cone_factorization_failed, :newton_factorization)
    elseif startswith(text, "pivoted LDLT factorization") ||
           startswith(text, "Schur complement not positive definite")
        return (:kkt_factorization_failed, :newton_factorization)
    elseif startswith(text, "Native extended-precision fallback")
        return (:direction_solve_failed, :newton_direction)
    elseif startswith(text, "final structured KKT direction residual")
        return (:direction_residual_exceeded, :newton_refinement)
    end
    return (:newton_breakdown, :newton_step)
end

@inline function _sdp_cold_start_kkt_formulation(ws::Workspace)
    ws.executed_backend === :mixed_precision &&
        return :dense_normal_equations
    return kkt_formulation_from_backend(ws.executed_backend)
end

@inline function _sdp_cold_start_factorization(ws::Workspace)
    ws.augmented !== nothing && return :pivoted_symmetric_ldlt
    ws.executed_backend === :dense_cholesky && return :cholesky
    ws.executed_backend === :mixed_precision &&
        return :mixed_precision_cholesky
    ws.executed_backend === :sparse_schur_cholesky &&
        return :sparse_cholesky
    ws.executed_backend === :block_arrow && return :block_cholesky
    return :not_executed
end

"""
    _release_iteration_memory!()

Run an explicit full collection and, on glibc-based Linux systems, return free
allocator pages to the operating system. This is intentionally reachable only
through `SolverOptions.force_gc`: large sparse factorizations and dense
multi-right-hand-side solves can leave several GiB in allocator arenas after
an iteration, while ordinary solves are faster when Julia chooses its own GC
schedule.
"""
function _release_iteration_memory!()
    GC.gc(true)
    if Sys.islinux()
        try
            ccall(:malloc_trim, Cint, (Csize_t,), 0)
        catch exception
            _recoverable(exception) || rethrow()
        end
    end
    return nothing
end

# ---- checkpointing (§5.5) ----

function save_checkpoint(path::AbstractString, ::Type{T}, x, X, y, Y, μ, iter, restarts, dims) where {T}
    isempty(path) && return
    cp = Checkpoint{T}(CHECKPOINT_FORMAT_VERSION, x, X, y, Y, μ, iter, restarts, dims)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        Serialization.serialize(io, cp)
    end
    mv(tmp, path; force=true)
    return nothing
end

"""
    save_checkpoint_jld2(path, T, x, X, y, Y, μ, iter, restarts, dims)
    load_checkpoint_jld2(path, T)

JLD2-backed checkpoint I/O (§5.5) — only available once the `JLD2`
package extension is loaded (`using JLD2`); more portable across
Julia versions than the default Serialization-based
[`save_checkpoint`](@ref)/[`load_checkpoint`](@ref).
"""
function save_checkpoint_jld2 end
function load_checkpoint_jld2 end

function load_checkpoint(path::AbstractString, ::Type{T}) where {T}
    cp = open(path, "r") do io
        Serialization.deserialize(io)
    end
    cp isa Checkpoint{T} || throw(ArgumentError("checkpoint at $path is not a Checkpoint{$T} (got $(typeof(cp)))"))
    cp.format_version == CHECKPOINT_FORMAT_VERSION ||
        throw(ArgumentError("checkpoint format version $(cp.format_version) unsupported (expected $CHECKPOINT_FORMAT_VERSION)"))
    return cp
end

# ---- main loop ----

function _replace_solver_options(
    options::SolverOptions{T};
    kwargs...,
) where {T}
    names = fieldnames(typeof(options))
    values = NamedTuple{names}(Tuple(getfield(options, field) for field in names))
    return SolverOptions{T}(; merge(values, (; kwargs...))...)
end

function _reround_solver_options(
    options::SolverOptions{BigFloat},
    bits::Int;
    kwargs...,
)
    names = fieldnames(typeof(options))
    values = map(names) do field
        value = getfield(options, field)
        value isa BigFloat ?
        BigFloat(value; precision=bits) : value
    end
    typed = NamedTuple{names}(Tuple(values))
    return SolverOptions{BigFloat}(; merge(typed, (; kwargs...))...)
end

"""
    recommended_adaptive_sigma_max(beta, requested)

Return the expert override when it is positive, otherwise select the guarded
automatic centering cap. Every solver path uses the generic `0.5` cap;
`adaptive_sigma_max` remains the inspectable expert override.
"""
@inline function recommended_adaptive_sigma_max(
    beta::T,
    requested::T,
) where {T}
    requested > zero(T) && return max(requested, beta)
    return max(T(1) / T(2), beta)
end

"""
    recommended_parameters(prob, opts) -> NamedTuple

    Resolve the automatic SDP cold-start parameters. This numeric resolver is
    invoked only once per SDP solve, on `solve_prob` after equilibration/Ruiz (or
    the identity scaling stage). The dedicated LP path has a provenance-only
    post-scaling resolver after its geometric data scaling; both paths construct
    the actual automatic initial point from an affine KKT solve.

    The automatic SDP path no longer derives an Ω from the cone data: a fully
    cold `:auto` solve starts from the identity-metric KKT point instead of a
    scaled multiple of the cone identity, so the resolver is a pure
    compatibility hint and returns the raw requested `Ωp`/`Ωd` untouched.
    Fixed-width users that rely on the old data-scale floor keep full control
    through `parameter_policy = :fixed`, which never calls this rule and uses
    the exact supplied options. No structure, cone type, size, or arithmetic
    branch influences the selection. Because this resolver runs only in the
    final transformed coordinates, it reports `:post_scaling_mehrotra`; the
    immutable plan records the deferred policy identity
    `:automatic_mehrotra`.
"""
function recommended_parameters(
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    return (
        β=opts.β,
        γ=opts.γ,
        Ωp=opts.Ωp,
        Ωd=opts.Ωd,
        predictor=opts.predictor,
        parameter_strategy=opts.parameter_strategy,
        profile=:post_scaling_mehrotra,
    )
end

"""
    _equality_factor_diagnostics(workspace, equality_count)

Return the equality-system method and numerical-rank verdict from the last
Newton factorization. This is intentionally O(n), allocation-free, and kept
out of the iteration hot path. A stopped solve can therefore distinguish an
uncertified objective caused by equality rank loss from ordinary slow
convergence.
"""
function _equality_factor_diagnostics(
    workspace::Workspace{T},
    equality_count::Int,
) where {T}
    equality_count == 0 &&
        return (
            available=true,
            method=:none,
            rank=0,
            dimension=0,
            rank_deficient=false,
            quality=one(T),
            gram_kernel=:none,
        )
    if workspace.augmented !== nothing
        augmented = workspace.augmented::DenseAugmentedKKTWorkspace{T}
        factor = augmented.factor
        factor === nothing && return let inertia = augmented.inertia,
            zero_count = inertia === nothing ? equality_count : Int(inertia[3]),
            numerical_rank = max(equality_count - zero_count, 0)
            (
                available=inertia !== nothing,
                factor_available=false,
                method=:augmented_ldlt,
                rank=numerical_rank,
                dimension=equality_count,
                rank_deficient=augmented.rank_deficient,
                quality=zero(T),
                gram_kernel=:not_formed_augmented,
                inertia,
                factor_diagnostics=augmented.factor_diagnostics,
                regularization=augmented.regularization,
            )
        end
        inertia = augmented.inertia
        inertia === nothing && return (
            available=false,
            factor_available=true,
            method=:augmented_ldlt,
            rank=0,
            dimension=equality_count,
            rank_deficient=true,
            quality=zero(T),
            gram_kernel=:not_formed_augmented,
            inertia=nothing,
            factor_diagnostics=augmented.factor_diagnostics,
            regularization=augmented.regularization,
        )
        numerical_rank = max(equality_count - Int(inertia[3]), 0)
        return (
            available=true,
            method=:augmented_ldlt,
            rank=numerical_rank,
            dimension=equality_count,
            rank_deficient=numerical_rank < equality_count,
            quality=one(T),
            gram_kernel=:not_formed_augmented,
            inertia,
            pivot_blocks=augmented.pivot_blocks,
            permutation=augmented.permutation,
            factor_diagnostics=augmented.factor_diagnostics,
            regularization=augmented.regularization,
        )
    end
    mixed = workspace.mixed_precision
    if mixed !== nothing && mixed.active
        factor = mixed.intermediate_active ?
                 mixed.intermediate.Qfactor : mixed.Qfactor
        if factor !== nothing
            lower = factor isa IntermediateCholeskyFactor ?
                    factor.L : factor.factors
            return (
                available=true,
                method=mixed.intermediate_active ?
                       :mixed_intermediate_normal_equations :
                       :mixed_float64_normal_equations,
                rank=equality_count,
                dimension=equality_count,
                rank_deficient=false,
                quality=_ingest_owned_scalar(
                    T,
                    _cholesky_diagonal_quality(lower),
                ),
                gram_kernel=workspace.equality_gram_kernel,
            )
        end
    end
    factor = workspace.Qchol
    factor === nothing &&
        return (
            available=false,
            method=:unavailable,
            rank=0,
            dimension=equality_count,
            rank_deficient=false,
            quality=zero(T),
            gram_kernel=workspace.equality_gram_kernel,
        )
    if factor isa EqualityQRFactor{T}
        qr_factor = factor::EqualityQRFactor{T}
        return (
            available=true,
            method=:rank_revealing_qr,
            rank=qr_factor.rank,
            dimension=equality_count,
            rank_deficient=qr_factor.rank < equality_count,
            quality=qr_factor.quality,
            gram_kernel=workspace.equality_gram_kernel,
        )
    elseif factor isa LinearAlgebra.CholeskyPivoted
        rank = factor.rank
        return (
            available=true,
            method=:pivoted_normal_equations,
            rank=rank,
            dimension=equality_count,
            rank_deficient=rank < equality_count,
            quality=rank < equality_count ?
                    zero(T) :
                    _cholesky_diagonal_quality(
                        view(factor.L, 1:rank, 1:rank),
                    ),
            gram_kernel=workspace.equality_gram_kernel,
        )
    elseif factor isa LegacyLACholeskyFactor
        return (
            available=true,
            method=:normal_equations,
            rank=equality_count,
            dimension=equality_count,
            rank_deficient=false,
            quality=_cholesky_diagonal_quality(factor.factors),
            gram_kernel=workspace.equality_gram_kernel,
        )
    elseif factor isa AbstractLACholeskyFactor{T}
        return (
            available=true,
            method=:la_backend_normal_equations,
            rank=equality_count,
            dimension=equality_count,
            rank_deficient=false,
            quality=_cholesky_diagonal_quality(factor.factors),
            gram_kernel=workspace.equality_gram_kernel,
        )
    end
    return (
        available=true,
        method=:normal_equations,
        rank=equality_count,
        dimension=equality_count,
        rank_deficient=false,
        quality=_cholesky_diagonal_quality(factor.factors),
        gram_kernel=workspace.equality_gram_kernel,
    )
end

mutable struct BestIterateWorkspace{T}
    valid::Bool
    x::Vector{T}
    X::Vector{Matrix{T}}
    y::Vector{T}
    Y::Vector{Matrix{T}}
    pObj::T
    dObj::T
    gap_rel::T
    p_res::T
    d_res::T
    iter::Int
end

function BestIterateWorkspace(x, X, y, Y)
    T = eltype(x)
    return BestIterateWorkspace{T}(
        false,
        alloc_zeros(T, length(x)),
        [alloc_zeros(T, size(block)...) for block in X],
        alloc_zeros(T, length(y)),
        [alloc_zeros(T, size(block)...) for block in Y],
        zero(T),
        zero(T),
        T(Inf),
        T(Inf),
        T(Inf),
        0,
    )
end

function _store_best_iterate!(
    best::BestIterateWorkspace{T},
    ws::Workspace{T},
    x,
    X,
    y,
    Y,
    pObj::T,
    dObj::T,
    gap_rel::T,
    p_res::T,
    d_res::T,
    iter::Int,
) where {T}
    copy_owned!(best.x, x)
    copy_owned!(best.y, y)
    threaded =
        ws.thread_count > 1 &&
        length(X) > 1 &&
        thread_safe_arithmetic(T) &&
        sum(length, ws.block_bins; init=0) >= 256
    if threaded
        @sync for bin in ws.block_bins
            isempty(bin) && continue
            Threads.@spawn begin
                @inbounds for block in bin
                    copy_owned!(best.X[block], X[block])
                    copy_owned!(best.Y[block], Y[block])
                end
            end
        end
    else
        @inbounds for block in eachindex(X)
            copy_owned!(best.X[block], X[block])
            copy_owned!(best.Y[block], Y[block])
        end
    end
    best.pObj = pObj
    best.dObj = dObj
    best.gap_rel = gap_rel
    best.p_res = p_res
    best.d_res = d_res
    best.iter = iter
    best.valid = true
    return best
end

function _scaled_identity(
    ::Type{T},
    scale::T,
    dimension::Int,
) where {T}
    return Matrix{T}(scale * I, dimension, dimension)
end

function _scaled_identity(
    ::Type{BigFloat},
    scale::BigFloat,
    dimension::Int,
)
    matrix = alloc_zeros(BigFloat, dimension, dimension)
    @inbounds for index in 1:dimension
        MA.operate_to!(matrix[index, index], copy, scale)
    end
    return matrix
end

"""
    _tolerance_precision_diagnostic(T, tolerance)

Estimate the significand width needed by a requested tolerance without
narrowing extended-precision values through `Float64`. The result is used only
for diagnostics; it never changes solver tolerances.
"""
function _tolerance_precision_diagnostic(
    ::Type{T},
    tolerance::T,
) where {T}
    threshold = T(100) * eps(T)
    needed_bits = tolerance <= zero(T) ?
                  typemax(Int) :
                  _nonnegative_int_saturating(
                      -log2(tolerance),
                      RoundUp,
                  )
    return (
        warn=tolerance < threshold,
        needed_bits=needed_bits,
        threshold=threshold,
    )
end

function _validate_warm_start(
    prob::SDPProblem;
    x0=nothing,
    X0=nothing,
    y0=nothing,
    Y0=nothing,
    resume::AbstractString="",
    accepted_y_lengths=(prob.dims.n,),
)
    supplied = (x0 !== nothing, X0 !== nothing, y0 !== nothing, Y0 !== nothing)
    !isempty(resume) && any(supplied) &&
        throw(ArgumentError(
            "resume cannot be combined with x0, X0, y0, or Y0; " *
            "the checkpoint already provides the complete iterate",
        ))

    if x0 !== nothing
        x0 isa AbstractVector ||
            throw(ArgumentError("x0 must be a vector"))
        length(x0) == prob.dims.m ||
            throw(DimensionMismatch(
                "x0 has length $(length(x0)); expected $(prob.dims.m)",
            ))
    end
    if y0 !== nothing
        y0 isa AbstractVector ||
            throw(ArgumentError("y0 must be a vector"))
        expected = unique(Int[value for value in accepted_y_lengths])
        length(y0) in expected ||
            throw(DimensionMismatch(
                "y0 has length $(length(y0)); expected " *
                join(expected, " or "),
            ))
    end

    (X0 === nothing) == (Y0 === nothing) ||
        throw(ArgumentError(
            "X0 and Y0 must be supplied together for an SDP warm start",
        ))
    X0 === nothing && return nothing
    X0 isa Union{AbstractVector,Tuple} ||
        throw(ArgumentError("X0 must be a vector or tuple of PSD blocks"))
    Y0 isa Union{AbstractVector,Tuple} ||
        throw(ArgumentError("Y0 must be a vector or tuple of PSD blocks"))
    length(X0) == prob.dims.L ||
        throw(DimensionMismatch(
            "X0 contains $(length(X0)) blocks; expected $(prob.dims.L)",
        ))
    length(Y0) == prob.dims.L ||
        throw(DimensionMismatch(
            "Y0 contains $(length(Y0)) blocks; expected $(prob.dims.L)",
        ))
    @inbounds for block in 1:prob.dims.L
        expected = (prob.dims.k[block], prob.dims.k[block])
        X0[block] isa AbstractMatrix ||
            throw(ArgumentError("X0[$block] must be a matrix"))
        Y0[block] isa AbstractMatrix ||
            throw(ArgumentError("Y0[$block] must be a matrix"))
        size(X0[block]) == expected ||
            throw(DimensionMismatch(
                "X0[$block] has size $(size(X0[block])); expected $expected",
            ))
        size(Y0[block]) == expected ||
            throw(DimensionMismatch(
                "Y0[$block] has size $(size(Y0[block])); expected $expected",
            ))
    end
    return nothing
end

function _sdp_setup_time_limit_result(
    prob::SDPProblem{T},
    elapsed::Float64,
    ;
    executed=nothing,
) where {T}
    termination = executed === nothing ?
                  (reason=:time_limit, stage=:sdp_setup) :
                  (reason=:time_limit, stage=:sdp_setup, executed)
    return SDPResult{T}(
        TimeLimit,
        "Time limit exceeded before SDP iterations began.",
        alloc_zeros(T, prob.dims.m),
        [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k],
        alloc_zeros(T, prob.dims.n),
        [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k],
        zero(T),
        zero(T),
        T(Inf),
        T(Inf),
        T(Inf),
        0,
        0,
        0,
        (total=elapsed,),
        NamedTuple[],
        nothing,
        termination,
    )
end

function _equilibrate_warm_start(
    ::Type{T},
    eq::Equilibration{T},
    x0,
    X0,
    y0,
    Y0,
) where {T}
    scaled_x0 = if x0 === nothing
        nothing
    else
        values = _owned_array_copy(T, x0)
        @inbounds for index in eachindex(values, eq.s)
            values[index] *= eq.s[index]
        end
        values
    end
    scaled_X0 = if X0 === nothing
        nothing
    else
        [
            begin
                matrix = _owned_array_copy(T, X0[block])
                diagonal = eq.E[block]
                @inbounds for column in axes(matrix, 2),
                              row in axes(matrix, 1)
                    matrix[row, column] *=
                        diagonal[row] * diagonal[column]
                end
                matrix
            end
            for block in eachindex(X0)
        ]
    end
    scaled_Y0 = if Y0 === nothing
        nothing
    else
        [
            begin
                matrix = _owned_array_copy(T, Y0[block])
                diagonal = eq.E[block]
                @inbounds for column in axes(matrix, 2),
                              row in axes(matrix, 1)
                    matrix[row, column] /=
                        diagonal[row] * diagonal[column]
                end
                # Objective normalisation divides `c` by `objective_scale`,
                # which scales the dual by its inverse. `unequilibrate`
                # multiplies `y`/`Y` back, so mapping a warm start *into* the
                # equilibrated space must divide by the same factor. Omitting it
                # makes the round-trip asymmetric and returns duals inflated by
                # exactly `objective_scale`.
                eq.objective_scale == one(T) ||
                    (matrix ./= eq.objective_scale)
                matrix
            end
            for block in eachindex(Y0)
        ]
    end
    scaled_y0 = (y0 === nothing || eq.objective_scale == one(T)) ? y0 :
                _owned_array_copy(T, y0) ./ eq.objective_scale
    return scaled_x0, scaled_X0, scaled_y0, scaled_Y0
end

"""
    _kkt_cold_start_initialization(ws, prob, opts) -> NamedTuple

Phase-2 identity-metric KKT cold start for a fully cold `:auto` SDP solve.
The cone identity is substituted for the initial point before any iterate is
formed, the identity-metric Schur complement `H = A'A` is assembled through
the current threaded Schur route and factored exactly once from the immutable
plan (`select_backend`/`factorize!`), and the same accepted factor serves the
two planned direction solves:

    Hx − Bq = A·C,  B'x = b   (primal RHS), and
    Hv − Bq = c,    B'v = 0   (dual RHS),

giving `x`, `Y = A(v)`, `y = -q`, and `X = A(x) - C`.  Each direction is
checked against the original, unregularized identity-metric KKT operator.  If
its normalized residual exceeds the cold-start gate, the existing structured
refinement seam reuses the same accepted factor for a bounded number of
correction solves.  This is required for a compatible rank-deficient Gram
whose factor-side regularization leaves an `O(sqrt(eps(T)))` residual, and for
an accepted mixed-precision factor whose first solve is intentionally lower
precision.  It never selects another provider or formulation; any fallback is
limited to the immutable plan's existing mixed-precision contract and remains
visible in the initialization record.

Pre-shift cone residuals are then computed with the current residual kernel
and must be finite and within
`max(sqrt(eps(T)), ϵ_primal, ϵ_dual)` after normalization; otherwise the
caller reports `NumericalBreakdown` at stage `:sdp_initialization` with no Ω
fallback. Each block is then pushed strictly into the PSD interior by
`_cold_start_psd_shift!` and the global identity pre-centering shifts from
`_cold_start_centering_shifts` are applied. The returned record carries the
required provenance (policy/path, formulation/provider/factorization,
pre-shift residuals, largest shifts, post margins and κ normalized by the
total PSD degree, factor count `1`, RHS count `2`, fallback reason, and
regularization attempts) without touching the Newton timing/counter totals.
"""
function _kkt_cold_start_initialization(
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    L, m, n, k = prob.dims
    cons = prob.cons
    total_psd_degree = sum(k; init=0)
    degree_denominator = max(total_psd_degree, 1)

    # ---- identity-metric Schur: X = Y = I, one factorization ----
    identity = [_scaled_identity(T, one(T), dimension) for dimension in k]
    factor_blocks!(ws, identity, identity)
    parallel_blas = ws.thread_count > 1 ? 1 : blas_threads()
    schur_blas = schur_blas_threads(
        ws,
        prob,
        cons,
        parallel_blas,
        blas_threads(),
    )
    _with_blas_threads(schur_blas) do
        threaded_schur_build!(ws, prob, cons, identity, identity)
    end
    backend = select_backend(ws)
    kkt = _with_blas_threads(_kkt_blas_threads(m)) do
        factorize!(backend, ws, prob, opts)
    end
    if !kkt.ok
        record = (
            ok=false,
            reason=:identity_kkt_factorization,
            policy=:auto,
            initialization_policy=:kkt_cold_start,
            path=:kkt_cold_start,
            # No factor was accepted; regularization_attempts records the
            # provider/formulation retries that preceded this failure.
            factor_count=0,
            rhs_solves=0,
            kkt_backend=ws.executed_backend,
            kkt_formulation=_sdp_cold_start_kkt_formulation(ws),
            la_backend=ws.executed_la_backend,
            la_provider=ws.executed_la_provider,
            la_ownership=ws.executed_la_ownership,
            factorization=_sdp_cold_start_factorization(ws),
            fallback_reason=ws.backend_fallback_reason,
            regularization_attempts=kkt.reg_attempts,
            total_psd_degree=total_psd_degree,
        )
        return (
            ok=false,
            reason=:identity_kkt_factorization,
            record=record,
        )
    end

    # ---- two planned solves and bounded corrections, same accepted factor ----
    residual_threshold = max(
        sqrt(eps(T)),
        opts.ϵ_primal,
        opts.ϵ_dual,
    )
    mixed = ws.mixed_precision
    guard_refinement_before = mixed === nothing ? 0 :
        mixed.predictor_refinement_steps
    dynamic_fallback_before = mixed === nothing ? 0 :
        mixed.dynamic_fallback_count
    native_regularization_before = mixed === nothing ? 0 :
        mixed.native_regularization_attempts
    primary_rhs_solves = 0
    cold_refinement_steps = 0

    function solve_cold_rhs!(
        rhs::AbstractVector{T},
        equality_rhs::AbstractVector{T},
    )
        copy_owned!(ws.p, equality_rhs)
        solved = solve_direction!(backend, ws, prob, opts, rhs)
        primary_rhs_solves += 1
        solved || return (
            ok=false,
            initial_residual=T(Inf),
            final_residual=T(Inf),
            normalized_residual=T(Inf),
            refinement_steps=0,
        )
        initial_residual = _kkt_direction_residual!(ws, prob, rhs)
        scale = max(
            knrmInf(rhs),
            n > 0 ? knrmInf(equality_rhs) : zero(T),
            one(T),
        )
        normalized_residual = initial_residual / scale
        refinement_steps = 0
        final_residual = initial_residual
        if isfinite(normalized_residual) &&
           normalized_residual > residual_threshold
            refinement_steps, final_residual =
                refine!(backend, ws, prob, opts, rhs)
            cold_refinement_steps += refinement_steps
            # `refine!` reports the last accepted residual. Recompute through
            # the original structured operator so this diagnostic and the
            # later cone residual gate have one authoritative value.
            final_residual = _kkt_direction_residual!(ws, prob, rhs)
            normalized_residual = final_residual / scale
        end
        return (
            ok=isfinite(normalized_residual) &&
                normalized_residual <= residual_threshold,
            initial_residual,
            final_residual,
            normalized_residual,
            refinement_steps,
        )
    end

    primal_rhs = alloc_zeros(T, m)
    @inbounds for l in 1:L
        accumulate_v_owned!(primal_rhs, cons, l, prob.C[l], one(T))
    end
    primal_solve = solve_cold_rhs!(primal_rhs, prob.b)
    if !primal_solve.ok
        record = (
            ok=false,
            reason=:kkt_cold_start_solve,
            policy=:auto,
            initialization_policy=:kkt_cold_start,
            path=:kkt_cold_start,
            factor_count=1,
            rhs_solves=primary_rhs_solves,
            base_rhs_solves=primary_rhs_solves,
            cold_refinement_steps=cold_refinement_steps,
            guard_refinement_steps=mixed === nothing ? 0 :
                mixed.predictor_refinement_steps - guard_refinement_before,
            dynamic_fallback_factorizations=mixed === nothing ? 0 :
                mixed.dynamic_fallback_count - dynamic_fallback_before,
            native_regularization_attempts=mixed === nothing ? 0 :
                max(
                    mixed.native_regularization_attempts -
                    native_regularization_before,
                    0,
                ),
            kkt_backend=ws.executed_backend,
            kkt_formulation=_sdp_cold_start_kkt_formulation(ws),
            la_backend=ws.executed_la_backend,
            la_provider=ws.executed_la_provider,
            la_ownership=ws.executed_la_ownership,
            factorization=_sdp_cold_start_factorization(ws),
            fallback_reason=ws.backend_fallback_reason,
            regularization_attempts=kkt.reg_attempts,
            total_psd_degree=total_psd_degree,
            primal_kkt_initial_residual=primal_solve.initial_residual,
            primal_kkt_final_residual=primal_solve.final_residual,
        )
        return (ok=false, reason=:kkt_cold_start_solve, record=record)
    end
    x = _owned_array_copy(T, ws.dx)
    primal_q = _owned_array_copy(T, ws.dy)

    dual_rhs = _owned_array_copy(T, prob.c)
    zero_equality_rhs = alloc_zeros(T, n)
    dual_solve = solve_cold_rhs!(dual_rhs, zero_equality_rhs)
    if !dual_solve.ok
        record = (
            ok=false,
            reason=:kkt_cold_start_solve,
            policy=:auto,
            initialization_policy=:kkt_cold_start,
            path=:kkt_cold_start,
            factor_count=1,
            rhs_solves=primary_rhs_solves,
            base_rhs_solves=primary_rhs_solves,
            cold_refinement_steps=cold_refinement_steps,
            guard_refinement_steps=mixed === nothing ? 0 :
                mixed.predictor_refinement_steps - guard_refinement_before,
            dynamic_fallback_factorizations=mixed === nothing ? 0 :
                mixed.dynamic_fallback_count - dynamic_fallback_before,
            native_regularization_attempts=mixed === nothing ? 0 :
                max(
                    mixed.native_regularization_attempts -
                    native_regularization_before,
                    0,
                ),
            kkt_backend=ws.executed_backend,
            kkt_formulation=_sdp_cold_start_kkt_formulation(ws),
            la_backend=ws.executed_la_backend,
            la_provider=ws.executed_la_provider,
            la_ownership=ws.executed_la_ownership,
            factorization=_sdp_cold_start_factorization(ws),
            fallback_reason=ws.backend_fallback_reason,
            regularization_attempts=kkt.reg_attempts,
            total_psd_degree=total_psd_degree,
            primal_kkt_initial_residual=primal_solve.initial_residual,
            primal_kkt_final_residual=primal_solve.final_residual,
            dual_kkt_initial_residual=dual_solve.initial_residual,
            dual_kkt_final_residual=dual_solve.final_residual,
        )
        return (ok=false, reason=:kkt_cold_start_solve, record=record)
    end
    v = _owned_array_copy(T, ws.dx)
    q = _owned_array_copy(T, ws.dy)

    X = [
        begin
            block = alloc_zeros(T, dimension, dimension)
            buildP_owned!(block, cons, l, x)
            kaxpby_owned!(-one(T), prob.C[l], one(T), block)
            block
        end
        for (l, dimension) in pairs(k)
    ]
    Y = [
        begin
            block = alloc_zeros(T, dimension, dimension)
            buildP_owned!(block, cons, l, v)
            block
        end
        for (l, dimension) in pairs(k)
    ]
    y = _owned_array_copy(T, q)
    @inbounds for index in eachindex(y)
        y[index] = -y[index]
    end

    # ---- pre-shift residuals through the current residual kernel ----
    placeholder_μ = alloc_zeros(T, L)
    p_res, d_res, _ = threaded_compute_residuals!(
        ws,
        prob,
        x,
        X,
        y,
        Y,
        placeholder_μ,
        opts,
    )
    scale_p = one(T) + max(
        L > 0 ? maximum(l -> knrmInf(prob.C[l]), 1:L) : zero(T),
        n > 0 ? knrmInf(prob.b) : zero(T),
    )
    scale_d = one(T) + knrmInf(prob.c)
    normalized_primal = p_res / scale_p
    normalized_dual = d_res / scale_d
    residuals_finite =
        isfinite(normalized_primal) && isfinite(normalized_dual)
    residuals_ok = residuals_finite &&
        normalized_primal <= residual_threshold &&
        normalized_dual <= residual_threshold
    if !residuals_ok
        record = (
            ok=false,
            reason=:kkt_cold_start_residuals,
            policy=:auto,
            initialization_policy=:kkt_cold_start,
            path=:kkt_cold_start,
            factor_count=1,
            rhs_solves=2,
            base_rhs_solves=2,
            cold_refinement_steps=cold_refinement_steps,
            guard_refinement_steps=mixed === nothing ? 0 :
                mixed.predictor_refinement_steps - guard_refinement_before,
            dynamic_fallback_factorizations=mixed === nothing ? 0 :
                mixed.dynamic_fallback_count - dynamic_fallback_before,
            native_regularization_attempts=mixed === nothing ? 0 :
                max(
                    mixed.native_regularization_attempts -
                    native_regularization_before,
                    0,
                ),
            kkt_backend=ws.executed_backend,
            kkt_formulation=_sdp_cold_start_kkt_formulation(ws),
            la_backend=ws.executed_la_backend,
            la_provider=ws.executed_la_provider,
            la_ownership=ws.executed_la_ownership,
            factorization=_sdp_cold_start_factorization(ws),
            fallback_reason=ws.backend_fallback_reason,
            regularization_attempts=kkt.reg_attempts,
            total_psd_degree=total_psd_degree,
            pre_shift_primal_residual=p_res,
            pre_shift_dual_residual=d_res,
            normalized_primal_residual=normalized_primal,
            normalized_dual_residual=normalized_dual,
            residual_threshold=residual_threshold,
            primal_kkt_initial_residual=primal_solve.initial_residual,
            primal_kkt_final_residual=primal_solve.final_residual,
            dual_kkt_initial_residual=dual_solve.initial_residual,
            dual_kkt_final_residual=dual_solve.final_residual,
        )
        return (
            ok=false,
            reason=:kkt_cold_start_residuals,
            record=record,
        )
    end

    # ---- strictly interior PSD shifts, then global pre-centering ----
    primal_shifts = Vector{T}(undef, L)
    dual_shifts = Vector{T}(undef, L)
    primal_margins = Vector{T}(undef, L)
    dual_margins = Vector{T}(undef, L)
    @inbounds for l in 1:L
        primal_ok, primal_shifts[l], primal_margins[l], _ =
            _cold_start_psd_shift!(X[l])
        dual_ok, dual_shifts[l], dual_margins[l], _ =
            _cold_start_psd_shift!(Y[l])
        if !(primal_ok && dual_ok)
            record = (
                ok=false,
                reason=:cold_start_psd_shift,
                policy=:auto,
                initialization_policy=:kkt_cold_start,
                path=:kkt_cold_start,
                factor_count=1,
                rhs_solves=2,
                base_rhs_solves=2,
                cold_refinement_steps=cold_refinement_steps,
                guard_refinement_steps=mixed === nothing ? 0 :
                    mixed.predictor_refinement_steps - guard_refinement_before,
                dynamic_fallback_factorizations=mixed === nothing ? 0 :
                    mixed.dynamic_fallback_count - dynamic_fallback_before,
                native_regularization_attempts=mixed === nothing ? 0 :
                    max(
                        mixed.native_regularization_attempts -
                        native_regularization_before,
                        0,
                    ),
                kkt_backend=ws.executed_backend,
                kkt_formulation=_sdp_cold_start_kkt_formulation(ws),
                la_backend=ws.executed_la_backend,
                la_provider=ws.executed_la_provider,
                la_ownership=ws.executed_la_ownership,
                factorization=_sdp_cold_start_factorization(ws),
                fallback_reason=ws.backend_fallback_reason,
                regularization_attempts=kkt.reg_attempts,
                total_psd_degree=total_psd_degree,
                pre_shift_primal_residual=p_res,
                pre_shift_dual_residual=d_res,
                normalized_primal_residual=normalized_primal,
                normalized_dual_residual=normalized_dual,
                residual_threshold=residual_threshold,
                block=Int(l),
            )
            return (
                ok=false,
                reason=:cold_start_psd_shift,
                record=record,
            )
        end
    end

    kappa_before = sum(
        block -> kdot(X[block], Y[block]),
        1:L;
        init=zero(T),
    )
    primal_mass = sum(block -> tr(X[block]), 1:L; init=zero(T))
    dual_mass = sum(block -> tr(Y[block]), 1:L; init=zero(T))
    floor_ok, primal_mass_floor, dual_mass_floor =
        _cold_start_identity_mass_shifts(
            primal_mass,
            dual_mass,
            total_psd_degree,
        )
    if !floor_ok
        record = (
            ok=false,
            reason=:cold_start_identity_mass_floor,
            policy=:auto,
            initialization_policy=:kkt_cold_start,
            path=:kkt_cold_start,
            factor_count=1,
            rhs_solves=2,
            base_rhs_solves=2,
            cold_refinement_steps=cold_refinement_steps,
            guard_refinement_steps=mixed === nothing ? 0 :
                mixed.predictor_refinement_steps - guard_refinement_before,
            dynamic_fallback_factorizations=mixed === nothing ? 0 :
                mixed.dynamic_fallback_count - dynamic_fallback_before,
            native_regularization_attempts=mixed === nothing ? 0 :
                max(
                    mixed.native_regularization_attempts -
                    native_regularization_before,
                    0,
                ),
            kkt_backend=ws.executed_backend,
            kkt_formulation=_sdp_cold_start_kkt_formulation(ws),
            la_backend=ws.executed_la_backend,
            la_provider=ws.executed_la_provider,
            la_ownership=ws.executed_la_ownership,
            factorization=_sdp_cold_start_factorization(ws),
            fallback_reason=ws.backend_fallback_reason,
            regularization_attempts=kkt.reg_attempts,
            total_psd_degree=total_psd_degree,
            pre_shift_primal_residual=p_res,
            pre_shift_dual_residual=d_res,
            normalized_primal_residual=normalized_primal,
            normalized_dual_residual=normalized_dual,
            residual_threshold=residual_threshold,
            kappa_before=kappa_before / degree_denominator,
            primal_mass,
            dual_mass,
        )
        return (
            ok=false,
            reason=:cold_start_identity_mass_floor,
            record=record,
        )
    end
    if primal_mass_floor > zero(T)
        @inbounds for l in 1:L
            _cold_start_add_psd_identity!(X[l], primal_mass_floor)
        end
    end
    if dual_mass_floor > zero(T)
        @inbounds for l in 1:L
            _cold_start_add_psd_identity!(Y[l], dual_mass_floor)
        end
    end
    kappa_after_mass_floor = sum(
        block -> kdot(X[block], Y[block]),
        1:L;
        init=zero(T),
    )
    primal_mass_after_floor = sum(
        block -> tr(X[block]),
        1:L;
        init=zero(T),
    )
    dual_mass_after_floor = sum(
        block -> tr(Y[block]),
        1:L;
        init=zero(T),
    )
    centered, primal_centering, dual_centering =
        _cold_start_centering_shifts(
            kappa_after_mass_floor,
            primal_mass_after_floor,
            dual_mass_after_floor,
        )
    if !centered
        record = (
            ok=false,
            reason=:cold_start_centering_shifts,
            policy=:auto,
            initialization_policy=:kkt_cold_start,
            path=:kkt_cold_start,
            factor_count=1,
            rhs_solves=2,
            base_rhs_solves=2,
            cold_refinement_steps=cold_refinement_steps,
            guard_refinement_steps=mixed === nothing ? 0 :
                mixed.predictor_refinement_steps - guard_refinement_before,
            dynamic_fallback_factorizations=mixed === nothing ? 0 :
                mixed.dynamic_fallback_count - dynamic_fallback_before,
            native_regularization_attempts=mixed === nothing ? 0 :
                max(
                    mixed.native_regularization_attempts -
                    native_regularization_before,
                    0,
                ),
            kkt_backend=ws.executed_backend,
            kkt_formulation=_sdp_cold_start_kkt_formulation(ws),
            la_backend=ws.executed_la_backend,
            la_provider=ws.executed_la_provider,
            la_ownership=ws.executed_la_ownership,
            factorization=_sdp_cold_start_factorization(ws),
            fallback_reason=ws.backend_fallback_reason,
            regularization_attempts=kkt.reg_attempts,
            total_psd_degree=total_psd_degree,
            pre_shift_primal_residual=p_res,
            pre_shift_dual_residual=d_res,
            normalized_primal_residual=normalized_primal,
            normalized_dual_residual=normalized_dual,
            residual_threshold=residual_threshold,
            kappa_before=kappa_before / degree_denominator,
            kappa_after_mass_floor=
                kappa_after_mass_floor / degree_denominator,
            primal_mass=primal_mass_after_floor,
            dual_mass=dual_mass_after_floor,
            primal_mass_floor_shift=primal_mass_floor,
            dual_mass_floor_shift=dual_mass_floor,
        )
        return (
            ok=false,
            reason=:cold_start_centering_shifts,
            record=record,
        )
    end
    if primal_centering > zero(T)
        @inbounds for l in 1:L
            _cold_start_add_psd_identity!(X[l], primal_centering)
        end
    end
    if dual_centering > zero(T)
        @inbounds for l in 1:L
            _cold_start_add_psd_identity!(Y[l], dual_centering)
        end
    end
    @inbounds for l in 1:L
        _, _, primal_margins[l], _ = _cold_start_psd_shift!(X[l])
        _, _, dual_margins[l], _ = _cold_start_psd_shift!(Y[l])
    end
    kappa_after = sum(
        block -> kdot(X[block], Y[block]),
        1:L;
        init=zero(T),
    )
    μ = [
        opts.β * kdot(X[l], Y[l]) / k[l]
        for l in 1:L
    ]

    record = (
        ok=true,
        reason=:none,
        policy=:auto,
        initialization_policy=:kkt_cold_start,
        path=:kkt_cold_start,
        factor_count=1,
        rhs_solves=2,
        base_rhs_solves=2,
        cold_refinement_steps=cold_refinement_steps,
        guard_refinement_steps=mixed === nothing ? 0 :
            mixed.predictor_refinement_steps - guard_refinement_before,
        dynamic_fallback_factorizations=mixed === nothing ? 0 :
            mixed.dynamic_fallback_count - dynamic_fallback_before,
        native_regularization_attempts=mixed === nothing ? 0 :
            max(
                mixed.native_regularization_attempts -
                native_regularization_before,
                0,
            ),
        kkt_backend=ws.executed_backend,
        kkt_formulation=_sdp_cold_start_kkt_formulation(ws),
        la_backend=ws.executed_la_backend,
        la_provider=ws.executed_la_provider,
        la_ownership=ws.executed_la_ownership,
        factorization=_sdp_cold_start_factorization(ws),
        fallback_reason=ws.backend_fallback_reason,
        regularization_attempts=kkt.reg_attempts,
        total_psd_degree=total_psd_degree,
        pre_shift_primal_residual=p_res,
        pre_shift_dual_residual=d_res,
        normalized_primal_residual=normalized_primal,
        normalized_dual_residual=normalized_dual,
        residual_threshold=residual_threshold,
        primal_kkt_initial_residual=primal_solve.initial_residual,
        primal_kkt_final_residual=primal_solve.final_residual,
        dual_kkt_initial_residual=dual_solve.initial_residual,
        dual_kkt_final_residual=dual_solve.final_residual,
        largest_primal_shift=L == 0 ? zero(T) :
            maximum(primal_shifts) +
            max(primal_mass_floor, zero(T)) +
            max(primal_centering, zero(T)),
        largest_dual_shift=L == 0 ? zero(T) :
            maximum(dual_shifts) +
            max(dual_mass_floor, zero(T)) +
            max(dual_centering, zero(T)),
        primal_margin=L == 0 ? one(T) :
                      minimum(primal_margins),
        dual_margin=L == 0 ? one(T) :
                     minimum(dual_margins),
        kappa_before=kappa_before / degree_denominator,
        kappa_after_mass_floor=
            kappa_after_mass_floor / degree_denominator,
        kappa_after=kappa_after / degree_denominator,
        complementarity_before=kappa_before / degree_denominator,
        complementarity_after=kappa_after / degree_denominator,
        primal_mass_before=primal_mass,
        dual_mass_before=dual_mass,
        primal_mass_after_floor=primal_mass_after_floor,
        dual_mass_after_floor=dual_mass_after_floor,
        primal_mass_floor_shift=primal_mass_floor,
        dual_mass_floor_shift=dual_mass_floor,
        primal_centering_shift=primal_centering,
        dual_centering_shift=dual_centering,
    )
    return (
        ok=true,
        x=x,
        X=X,
        y=y,
        Y=Y,
        μ=μ,
        record=record,
    )
end

function _accepted_sdp_trial_residuals!(
    ws::Workspace{T},
    block_primal_residual::T,
    primal_step::T,
    dual_step::T,
) where {T}
    primal_scale = one(T) - primal_step
    dual_scale = one(T) - dual_step
    block_residual_after = abs(primal_scale) * block_primal_residual

    primal_residual_after = block_residual_after
    if !isempty(ws.p)
        # p+ = (1-tX)p + tX*rho_p
        kaxpby_owned!(primal_step, ws.ρp, primal_scale, ws.p)
        primal_residual_after = max(
            primal_residual_after,
            knrmInf(ws.p),
        )
    end

    # d+ = (1-tY)d - tY*rho_r
    kaxpby_owned!(-dual_step, ws.ρr, dual_scale, ws.d)
    dual_residual_after = knrmInf(ws.d)
    return primal_residual_after, dual_residual_after
end

"""
    _solve_sdp_core!(prob::SDPProblem{T}, opts::SolverOptions{T}=SolverOptions{T}();
           x0=nothing, X0=nothing, y0=nothing, Y0=nothing, resume="") -> SDPResult{T}

The one loop serving cold start, warm start (`x0,X0,y0,Y0`),
`OPTIMIZE`/`FEASIBILITY` mode, and resume-from-checkpoint (§1.6, §5.5)
— replacing the four near-duplicate `sdp`/`findFeasible` bodies in the
original. See `compat.jl` for the legacy `sdp`/`findFeasible` API that
wraps this.
"""
function _solve_sdp_core!(prob::SDPProblem{T}, opts::SolverOptions{T}=SolverOptions{T}();
    x0=nothing, X0=nothing, y0=nothing, Y0=nothing,
    resume::AbstractString="", deadline::Float64=Inf,
    apply_equilibration::Bool=false,
    execution_plan::Union{Nothing,ExecutionPlan}=nothing) where {T}

    core_started = time()
    core_started_ns = time_ns()
    opts.parameter_policy in (:fixed, :auto) ||
        throw(ArgumentError("parameter_policy must be :fixed or :auto"))
    opts.parameter_strategy in (:fixed, :adaptive) ||
        throw(ArgumentError("parameter_strategy must be :fixed or :adaptive"))
    zero(T) < opts.β < one(T) ||
        throw(ArgumentError("β must be strictly between zero and one"))
    zero(T) < opts.γ < one(T) ||
        throw(ArgumentError("γ must be strictly between zero and one"))
    opts.step_rule in (:backtrack, :fraction_to_boundary, :auto) ||
        throw(ArgumentError(
            "step_rule must be :backtrack, :fraction_to_boundary, or :auto",
        ))
    opts.extended_precision_blas in (:off, :auto, :on) ||
        throw(ArgumentError(
            "extended_precision_blas must be :off, :auto, or :on",
        ))
    0.0 <= opts.extended_precision_memory_fraction <= 1.0 ||
        throw(ArgumentError(
            "extended_precision_memory_fraction must be between zero and one",
        ))
    _validate_warm_start(
        prob;
        x0=x0,
        X0=X0,
        y0=y0,
        Y0=Y0,
        resume=resume,
    )
    # The original automatic policy is captured before the resolver replaces
    # the options below. A fully cold solve (`:auto` policy with no warm start
    # and no resume) takes the Phase-2 identity-metric KKT initialization;
    # every other path keeps its current point construction exactly.
    fully_cold = opts.parameter_policy === :auto &&
                 x0 === nothing &&
                 X0 === nothing &&
                 y0 === nothing &&
                 Y0 === nothing &&
                 isempty(resume)
    time() >= deadline &&
        return _sdp_setup_time_limit_result(
            prob,
            time() - core_started,
            executed=(
                solver=:sdp,
                parameter_profile=:not_resolved,
                parameter_source=:not_resolved,
                parameter_resolution_count=0,
                stage=:not_resolved,
            ),
        )
    validation_finished_ns = time_ns()

    if T === BigFloat
        check_precision_consistency(prob, opts.precision_bits, opts.verbosity)
        opts.convert_inputs && (prob = reround(prob, opts.precision_bits))
    end
    precision_preparation_finished_ns = time_ns()

    eq = nothing
    solve_prob = prob
    if apply_equilibration
        solve_prob, eq = equilibrate(prob)
    end
    equilibration_finished_ns = time_ns()

    termination_gap_tolerance = opts.ϵ_gap
    if eq !== nothing && eq.objective_scale > one(T)
        # Objective equilibration divides both primal and dual objectives by
        # `objective_scale`. When the original optimum is close to zero, both
        # relative-gap denominators are one, so an internally accepted gap is
        # multiplied by that scale after `unequilibrate`. This used to let the
        # core report `Optimal` before the original-coordinate certificate was
        # accurate enough (a 1e-10 internal gap could return as 7.7e-9).
        #
        # The ratio between original and scaled relative gaps is never larger
        # than max(1, objective_scale), including nonzero large objectives.
        # Tighten only the acceptance threshold. Feeding the stricter value
        # into stagnation and adaptive-control heuristics can make a flat early
        # gap dominate their progress merit and stop a solve that later
        # recovers once the gap starts moving. The controller should continue
        # to interpret the accuracy requested by the user; only a prospective
        # success must satisfy the conservative scaled threshold.
        termination_gap_tolerance =
            opts.ϵ_gap / eq.objective_scale
    end

    # Executed provenance: the automatic resolver runs exactly once, on the
    # post-scaling problem (after equilibration/Ruiz or identity scaling). It
    # resolves controller parameters only; the automatic initial point is the
    # affine KKT cold start below and does not depend on Ωp/Ωd.
    # The plan identity is deferred (`:automatic_mehrotra`); the executed
    # record reports the post-scaling resolution. Fixed policy never invokes
    # the resolver and records the exact user options as `:user_fixed`.
    parameter_resolution_count = opts.parameter_policy === :auto ? 1 : 0
    parameter_source = opts.parameter_policy === :auto ?
                       :post_scaling_mehrotra : :user_fixed
    executed_parameters = if opts.parameter_policy === :auto
        selected = recommended_parameters(solve_prob, opts)
        adaptive_sigma_max = selected.parameter_strategy === :adaptive ?
                             recommended_adaptive_sigma_max(
                                 selected.β,
                                 opts.adaptive_sigma_max,
                             ) :
                             zero(T)
        opts.verbosity >= 1 && println(
            "SDPX auto parameters: profile=$(selected.profile), " *
            "beta=$(Float64(selected.β)), gamma=$(Float64(selected.γ)), " *
            "omega_p=$(Float64(selected.Ωp)), omega_d=$(Float64(selected.Ωd)), " *
            "predictor=$(selected.predictor), strategy=$(selected.parameter_strategy), " *
            "adaptive_sigma_max=$(Float64(adaptive_sigma_max))",
        )
        opts = _replace_solver_options(
            opts;
            β=selected.β,
            γ=selected.γ,
            Ωp=selected.Ωp,
            Ωd=selected.Ωd,
            predictor=selected.predictor,
            parameter_strategy=selected.parameter_strategy,
            adaptive_sigma_max,
            parameter_policy=:fixed,
        )
        (
            beta=selected.β,
            gamma=selected.γ,
            omega_p=selected.Ωp,
            omega_d=selected.Ωd,
            predictor=selected.predictor,
            strategy=selected.parameter_strategy,
            adaptive_sigma_max,
            profile=:post_scaling_mehrotra,
        )
    else
        (
            beta=opts.β,
            gamma=opts.γ,
            omega_p=opts.Ωp,
            omega_d=opts.Ωd,
            predictor=opts.predictor,
            strategy=opts.parameter_strategy,
            adaptive_sigma_max=opts.adaptive_sigma_max,
            profile=:user_fixed,
        )
    end
    executed_parameter_record = (
        parameter_profile=executed_parameters.profile,
        executed_parameters=(
            beta=executed_parameters.beta,
            gamma=executed_parameters.gamma,
            omega_p=executed_parameters.omega_p,
            omega_d=executed_parameters.omega_d,
            predictor=executed_parameters.predictor,
            strategy=executed_parameters.strategy,
            adaptive_sigma_max=executed_parameters.adaptive_sigma_max,
        ),
        parameter_source,
        parameter_resolution_count,
        stage=parameter_resolution_count == 1 ?
            :post_scaling : :not_applicable,
    )
    if eq !== nothing && isempty(resume)
        x0, X0, y0, Y0 = _equilibrate_warm_start(
            T,
            eq,
            x0,
            X0,
            y0,
            Y0,
        )
    end
    parameter_selection_finished_ns = time_ns()

    L, m, n, k = solve_prob.dims
    ws = Workspace(
        solve_prob;
        extended_precision_blas=opts.extended_precision_blas,
        extended_precision_memory_fraction=
            opts.extended_precision_memory_fraction,
        mixed_precision_kkt=opts.mixed_precision_kkt,
        mixed_precision_memory_fraction=
            opts.mixed_precision_memory_fraction,
        equality_solver=opts.equality_solver,
        thread_count=opts.threads,
        execution_plan=execution_plan,
    )
    workspace_finished_ns = time_ns()
    time() >= deadline &&
        return _sdp_setup_time_limit_result(
            solve_prob,
            time() - core_started,
            executed=merge((solver=:sdp,), executed_parameter_record),
        )

    local x::Vector{T}, y::Vector{T}, X::Vector{Matrix{T}}, Y::Vector{Matrix{T}}, μ::Vector{T}
    local iter::Int, restarts::Int
    local initialization_record = nothing

    if !isempty(resume)
        cp = load_checkpoint(resume, T)
        cp.dims == solve_prob.dims || throw(ArgumentError("checkpoint dims $(cp.dims) do not match problem dims $(solve_prob.dims)"))
        _validate_warm_start(
            solve_prob;
            x0=cp.x,
            X0=cp.X,
            y0=cp.y,
            Y0=cp.Y,
        )
        x = _owned_array_copy(T, cp.x)
        X = [_owned_array_copy(T, block) for block in cp.X]
        y = _owned_array_copy(T, cp.y)
        Y = [_owned_array_copy(T, block) for block in cp.Y]
        μ = _owned_array_copy(T, cp.μ)
        iter, restarts = cp.iter, cp.restarts
    else
        if fully_cold
            cold_start = _kkt_cold_start_initialization(
                ws,
                solve_prob,
                opts,
            )
            if !cold_start.ok
                return SDPResult{T}(
                    NumericalBreakdown,
                    "Cold-start KKT initialization failed " *
                    "($(cold_start.reason)).",
                    alloc_zeros(T, m),
                    [
                        alloc_zeros(T, dimension, dimension)
                        for dimension in k
                    ],
                    alloc_zeros(T, n),
                    [
                        alloc_zeros(T, dimension, dimension)
                        for dimension in k
                    ],
                    zero(T),
                    zero(T),
                    T(Inf),
                    T(Inf),
                    T(Inf),
                    0,
                    0,
                    0,
                    nothing,
                    NamedTuple[],
                    nothing,
                    (
                        reason=cold_start.reason,
                        stage=:sdp_initialization,
                        initialization=cold_start.record,
                        executed=merge(
                            (solver=:sdp,),
                            executed_parameter_record,
                            (
                                stage=:sdp_initialization,
                                initialization=cold_start.record,
                            ),
                        ),
                    ),
                )
            end
            x = cold_start.x
            X = cold_start.X
            y = cold_start.y
            Y = cold_start.Y
            μ = cold_start.μ
            initialization_record = cold_start.record
        else
            x = x0 === nothing ? alloc_zeros(T, m) : _owned_array_copy(T, x0)
            y = y0 === nothing ? alloc_zeros(T, n) : _owned_array_copy(T, y0)
            if X0 === nothing
                scales = initial_block_scales(solve_prob, opts)
                X = [
                    _scaled_identity(T, opts.Ωp * scales[l], k[l])
                    for l in 1:L
                ]
                Y = [
                    _scaled_identity(T, opts.Ωd * scales[l], k[l])
                    for l in 1:L
                ]
            else
                X = [_owned_array_copy(T, X0[l]) for l in 1:L]
                Y = [_owned_array_copy(T, Y0[l]) for l in 1:L]
                (all(l -> isposdef(X[l]), 1:L) && all(l -> isposdef(Y[l]), 1:L)) ||
                    return SDPResult{T}(
                        NumericalBreakdown,
                        "initial X0/Y0 must be positive definite",
                        x,
                        X,
                        y,
                        Y,
                        zero(T),
                        zero(T),
                        T(Inf),
                        T(Inf),
                        T(Inf),
                        0,
                        0,
                        0,
                        nothing,
                        NamedTuple[],
                        nothing,
                        (
                            reason=:invalid_warm_start,
                            stage=:sdp_initialization,
                            executed=merge(
                                (solver=:sdp,),
                                executed_parameter_record,
                            ),
                        ),
                    )
            end
            μ = [opts.β * kdot(X[l], Y[l]) / k[l] for l in 1:L]
        end
        iter = 0
        restarts = 0
    end
    initialization_finished_ns = time_ns()

    total_reg = 0
    centering_attempts = 0
    last_refine_steps = 0
    total_refine_steps = 0
    last_refine_residual = zero(T)
    t_start = core_started
    parameter_controller = AdaptiveIPMController(opts)
    phase_residual = 0.0
    phase_schur = 0.0
    phase_kkt = 0.0
    phase_predictor = 0.0
    phase_corrector = 0.0
    phase_kkt_schur_copy = 0.0
    phase_kkt_schur_factorization = 0.0
    phase_kkt_constraint_triangular_solve = 0.0
    phase_kkt_equality_gram = 0.0
    phase_kkt_equality_factorization = 0.0
    phase_kkt_other = 0.0
    phase_predictor_rhs = 0.0
    phase_predictor_linear_solve = 0.0
    phase_predictor_direction_recovery = 0.0
    phase_complementarity_analysis = 0.0
    phase_corrector_rhs = 0.0
    phase_corrector_linear_solve = 0.0
    phase_refinement = 0.0
    phase_corrector_direction_recovery = 0.0
    phase_line_search = 0.0
    phase_update = 0.0
    phase_best_iterate = 0.0
    phase_objective_and_targets = 0.0

    pObj = LinearAlgebra.dot(solve_prob.c, x)
    dObj = dual_objective(solve_prob, y, Y)
    p_res, d_res, _ = threaded_compute_residuals!(
        ws,
        solve_prob,
        x,
        X,
        y,
        Y,
        μ,
        opts,
    )

    scale_p = 1 + max(L > 0 ? maximum(l -> knrmInf(solve_prob.C[l]), 1:L) : zero(T),
        n > 0 ? knrmInf(solve_prob.b) : zero(T))
    scale_d = 1 + knrmInf(solve_prob.c)

    # §4.1 tolerance hygiene: warn once, up front, if the requested gap is at or
    # below the type's own working precision — verified during development that
    # asking Float64x2 (eps≈1e-31) for ϵ_gap=1e-25 grinds the line search down to
    # noise and exhausts restarts, exactly the silent-iterMax-burn this catches.
    precision_diagnostic = _tolerance_precision_diagnostic(T, opts.ϵ_gap)
    if opts.verbosity >= 1 && precision_diagnostic.warn
        @warn "ϵ_gap=$(opts.ϵ_gap) needs roughly $(precision_diagnostic.needed_bits) significand bits; " *
              "T=$T has $(sig_bits(T)). This tolerance may be unreachable; " *
              "the solver will return a non-optimal status with precision-floor " *
              "or stagnation diagnostics. Loosen ϵ_gap or use a higher-precision T."
    end

    print_header(opts)
    print_iter(opts, iter, pObj, dObj, pObj - dObj, p_res, d_res)
    initial_residual_finished_ns = time_ns()

    status = NotStarted
    message = ""
    # Structured terminal metadata for this iterative core.  Successful SDP
    # results intentionally retain the historical `(:none, :none)` pair;
    # every non-success break below assigns an explicit reason and stage.
    termination_reason = :none
    termination_stage = :none

    # Best-iterate retention. An interior-point run can reach a good point and
    # then wander away from it: when one side of the KKT system has been driven
    # deep into tolerance while the other diverges (primal residual at roundoff
    # scale while the dual residual grows past ~1), the restart rule rescales
    # the collapsed side and the previously good iterate is lost. Reporting the
    # *last* point in that situation returns a worse answer than the solver
    # actually found, and the objective from such a run is not meaningful.
    #
    # The merit is the largest of the three scaled quantities the termination
    # test uses, so "best" means "closest to satisfying the stopping criteria",
    # not merely smallest gap (which a wildly infeasible point can fake). Only
    # non-`Optimal` exits fall back to it — a converged run already ends on its
    # best point, so this can never change a successful solve.
    best_merit = T(Inf)
    best_iterate = BestIterateWorkspace(x, X, y, Y)
    stagnation = StagnationDetector{T}(opts.stall_iterations, opts.stall_tolerance)
    initial_merit = T(Inf)   # merit at the first iterate, for progress detection
    current_complementarity =
        sum(block -> kdot(X[block], Y[block]), 1:L; init=zero(T))

    while true
        gap = pObj - dObj
        gap_rel = abs(gap) / max(one(T), (abs(pObj) + abs(dObj)) / 2)
        term_ok, gap_ok = if opts.termination === :legacy
            (p_res <= opts.ϵ_primal && d_res <= opts.ϵ_dual),
            (zero(T) <= gap <= termination_gap_tolerance)
        else
            (p_res / scale_p <= opts.ϵ_primal &&
             d_res / scale_d <= opts.ϵ_dual),
            (gap_rel <= termination_gap_tolerance)
        end
        # Residuals between accepted steps are updated from the exact affine
        # residual recurrence below. Before issuing a success certificate,
        # recompute them from the current iterate so accumulated roundoff can
        # never turn an estimate into a false `Optimal` status.
        #
        # Do not perform this full original-coordinate block scan merely
        # because feasibility is already inside tolerance. On large sparse
        # problems feasibility can arrive many iterations before the gap; the
        # old `if term_ok` therefore rebuilt every residual on each of those
        # iterations even though an optimal certificate was impossible.
        certificate_candidate = if opts.mode === OPTIMIZE
            term_ok && gap_ok
        elseif opts.mode === FEASIBILITY
            term_ok && (pObj < zero(T) || dObj >= zero(T))
        else
            term_ok
        end
        if certificate_candidate
            p_res, d_res = solution_residuals(
                solve_prob,
                x,
                X,
                y,
                Y,
            )
            term_ok = if opts.termination === :legacy
                p_res <= opts.ϵ_primal && d_res <= opts.ϵ_dual
            else
                p_res / scale_p <= opts.ϵ_primal &&
                d_res / scale_d <= opts.ϵ_dual
            end
        end

        # Progress is judged by `StagnationDetector` (see `stagnation.jl`): all
        # four convergence metrics, each normalised by the tolerance requested
        # for it, tracked over a rolling window. `merit <= 1` means converged.
        # The accepted-step update already computed this scalar while it owned
        # each block. Reusing it avoids another complete block scan at the top
        # of every iteration. Restarts refresh the cache after rescaling.
        complementarity = current_complementarity
        objective_scale = max(abs(pObj), abs(dObj))
        merit = stagnation_merit(stagnation, opts, p_res, d_res, gap_rel,
            complementarity, scale_p, scale_d, objective_scale)
        floor_reached = at_precision_floor(p_res, d_res, gap_rel, scale_p, scale_d)
        stagnated = observe!(stagnation, merit, floor_reached, opts.iter_max - iter)
        isfinite(merit) && !isfinite(initial_merit) && (initial_merit = merit)
        if isfinite(merit) && merit < best_merit
            best_merit = merit
            best_iterate_started = time_ns()
            _store_best_iterate!(
                best_iterate,
                ws,
                x,
                X,
                y,
                Y,
                pObj,
                dObj,
                gap_rel,
                p_res,
                d_res,
                iter,
            )
            phase_best_iterate +=
                (time_ns() - best_iterate_started) / 1.0e9
        end

        if term_ok
            if opts.mode === OPTIMIZE && gap_ok
                status, message = Optimal, "Optimal"
                break
            elseif opts.mode === FEASIBILITY
                if pObj < zero(T)
                    status, message = FeasibleCert, "Feasible"
                    termination_reason = :feasible_certificate
                    termination_stage = :termination_check
                    break
                elseif dObj >= zero(T)
                    status, message = InfeasibleCert, "Infeasible"
                    termination_reason = :infeasible_certificate
                    termination_stage = :termination_check
                    break
                end
            end
        end
        if iter >= opts.iter_max
            status, message = IterLimit, "Cannot reach optimality (feasibility) within $(opts.iter_max) iterations."
            termination_reason = :iteration_limit
            termination_stage = :termination_check
            break
        end
        # Precision-exhaustion stop. Once the scaled merit has not improved for
        # `stall_iterations` consecutive iterations the working precision is
        # spent: further iterations do not converge, and the restart rule can
        # actively destroy the best iterate found — rescaling the collapsed
        # side of an already near-converged KKT system makes the merit degrade
        # again rather than converge. Reporting `Stalled` with the retained
        # best iterate is both faster and more honest than grinding to
        # `IterLimit`/`MaxRestartsExceeded`.
        if stagnated
            # Distinguish "the arithmetic ran out" from "the algorithm stopped
            # making progress". Only the first is fixed by a wider type, and
            # saying so is the difference between an actionable result and a
            # bare `Stalled`.
            status = stagnation.reason === :precision_floor ? InsufficientPrecision : Stalled
            message = stagnation_message(stagnation, opts.ϵ_gap)
            termination_reason = stagnation.reason === :precision_floor ?
                                 :precision_floor : :stagnation
            termination_stage = :stagnation_check
            break
        end
        if time() >= deadline || time() - t_start > opts.max_time
            status, message = TimeLimit, "Time limit ($(opts.max_time)s) exceeded after $iter iterations."
            termination_reason = :time_limit
            termination_stage = :termination_check
            break
        end
        if opts.callback !== nothing
            # `refine_*` describe the *previous* iteration's KKT solve: the
            # callback runs at the top of the loop, before this iteration's
            # direction exists. They are the diagnostic for whether a stalling
            # solve is limited by the accuracy of the linear solve or by the
            # algorithm — a residual at round-off means regularization and extra
            # precision cannot help.
            cbstate = (
                iter=iter,
                pObj=_diagnostic_scalar_copy(pObj),
                dObj=_diagnostic_scalar_copy(dObj),
                gap=_diagnostic_scalar_copy(gap),
                gap_rel=_diagnostic_scalar_copy(gap_rel),
                p_res=_diagnostic_scalar_copy(p_res),
                d_res=_diagnostic_scalar_copy(d_res),
                complementarity=
                    _diagnostic_scalar_copy(complementarity),
                termination_merit=_diagnostic_scalar_copy(merit),
                μ=_owned_array_copy(T, μ),
                restarts=restarts,
                refine_steps=last_refine_steps,
                refine_residual=
                    _diagnostic_scalar_copy(last_refine_residual),
            )
            if opts.callback(cbstate) === true
                status, message = UserStopped, "Stopped by callback after $iter iterations."
                termination_reason = :user_stopped
                termination_stage = :termination_check
                break
            end
        end

        t1 = time()
        iteration_options = controller_options(opts, parameter_controller)
        result = newton_step!(
            ws,
            solve_prob,
            iteration_options,
            x,
            X,
            y,
            Y,
            μ,
            parameter_controller=parameter_controller,
            iteration=iter + 1,
            relative_gap=gap_rel,
            primal_scale=scale_p,
            dual_scale=scale_d,
        )
        if result.status === :breakdown
            status, message = NumericalBreakdown, result.reason
            termination_reason, termination_stage =
                _sdp_newton_termination_metadata(result.reason)
            break
        end
        p_res, d_res = result.p_res, result.d_res
        last_refine_steps = result.refine_steps
        total_refine_steps += result.refine_steps
        last_refine_residual = result.refine_residual
        total_reg += result.reg_attempts
        phase_residual += result.phase_times.residual_and_block_factor
        phase_schur += result.phase_times.schur_assembly
        phase_kkt += result.phase_times.kkt_factorization
        phase_predictor += result.phase_times.predictor
        phase_corrector += result.phase_times.corrector
        phase_kkt_schur_copy +=
            result.phase_times.kkt_schur_copy
        phase_kkt_schur_factorization +=
            result.phase_times.kkt_schur_factorization
        phase_kkt_constraint_triangular_solve +=
            result.phase_times.kkt_constraint_triangular_solve
        phase_kkt_equality_gram +=
            result.phase_times.kkt_equality_gram
        phase_kkt_equality_factorization +=
            result.phase_times.kkt_equality_factorization
        phase_kkt_other += result.phase_times.kkt_other
        phase_predictor_rhs += result.phase_times.predictor_rhs
        phase_predictor_linear_solve +=
            result.phase_times.predictor_linear_solve
        phase_predictor_direction_recovery +=
            result.phase_times.predictor_direction_recovery
        phase_complementarity_analysis +=
            result.phase_times.complementarity_analysis
        phase_corrector_rhs += result.phase_times.corrector_rhs
        phase_corrector_linear_solve +=
            result.phase_times.corrector_linear_solve
        phase_refinement += result.phase_times.refinement
        phase_corrector_direction_recovery +=
            result.phase_times.corrector_direction_recovery

        line_search_started = time_ns()
        selected_parameters = result.iteration_parameters
        tX, tY = threaded_line_search!(
            ws,
            X,
            Y,
            selected_parameters.primal_fraction_to_boundary,
            selected_parameters.dual_fraction_to_boundary,
            selected_parameters.backtracking_factor,
            selected_parameters.minimum_step,
            opts.step_rule,
            opts.parameter_strategy === :adaptive ? sqrt(eps(T)) : zero(T),
        )
        selected_step_rule = resolved_step_rule(ws, opts.step_rule)
        backtracking_count =
            estimate_backtracking_count(
                tX,
                selected_parameters.backtracking_factor,
                selected_step_rule,
            ) +
            estimate_backtracking_count(
                tY,
                selected_parameters.backtracking_factor,
                selected_step_rule,
            )
        phase_line_search += (time_ns() - line_search_started) / 1.0e9

        # A step collapsing on a side that is *already feasible* is not a
        # failure — that side has nowhere useful left to go, so a tiny step
        # there is the expected outcome, and the other side can still make
        # progress. Primal and dual step lengths are independent in this
        # method, so treating `tX < min_step || tY < min_step` as fatal ends
        # perfectly healthy solves: when the primal residual sits at roundoff
        # scale (`p_res ≈ 1e-47`) the primal step `tX` duly collapses while the
        # duality gap is still ~1e-3, and stopping there discards a solve that
        # can still make progress.
        #
        # Only a stuck side that still has work to do is a real collapse.
        primal_feasible = p_res / scale_p <= opts.ϵ_primal
        dual_feasible = d_res / scale_d <= opts.ϵ_dual
        x_stuck = tX < selected_parameters.minimum_step
        y_stuck = tY < selected_parameters.minimum_step
        if x_stuck && primal_feasible && !y_stuck
            tX = zero(T)          # freeze a converged primal, keep going
            x_stuck = false
        end
        if y_stuck && dual_feasible && !x_stuck
            tY = zero(T)
            y_stuck = false
        end
        if x_stuck || y_stuck
            # A lower-precision preconditioner can pass the
            # target-arithmetic residual guard yet produce a direction too
            # inaccurate to preserve a useful cone-interior step. Retry from
            # the unchanged iterate with the native factorization before
            # recentering, rescaling, or declaring a stall.
            if x_stuck &&
               y_stuck &&
               ws.mixed_precision !== nothing
                mixed =
                    ws.mixed_precision::MixedPrecisionKKTWorkspace
                if mixed.active
                    mixed.active = false
                    mixed.intermediate_active = false
                    mixed.fell_back = true
                    mixed.disabled = true
                    mixed.cooldown_remaining = 0
                    mixed.dynamic_fallback_count += 1
                    mixed.reason = :outer_step_collapse
                    opts.verbosity >= 1 && println(
                        "Both cone steps collapsed with a mixed KKT " *
                        "preconditioner; retrying from the unchanged " *
                        "iterate with native $(T) factorization.",
                    )
                    continue
                end
            end
            # Before treating this as a scaling problem worth restarting, check
            # whether we're actually at the IPM's numerical convergence tail: gap
            # and residuals already close to the requested tolerance, with the
            # step collapsing because there's no more room to push further at this
            # working precision — not because Ωp/Ωd were badly chosen. Rescaling
            # does not fix that case. Found empirically, not designed in advance:
            # cross-checking against Clarabel.jl on a well-posed instance, SDPX
            # reached the same optimum to 7 digits by iteration 20 (gap=8.7e-9
            # against a default ϵ_gap=1e-10), then burned all 10 restarts trying
            # to close the last order of magnitude, ending in a misleading
            # `MaxRestartsExceeded` for a run that had, for practical purposes,
            # already converged. `pObj`/`dObj` here are still the last *accepted*
            # iterate's values (updated later in this loop only on acceptance),
            # so this checks "how close were we when the step collapsed."
            if opts.mode === OPTIMIZE
                near_tol = T(1000)
                gap_rel_now = abs(pObj - dObj) / max(one(T), (abs(pObj) + abs(dObj)) / 2)
                p_res_rel_now = p_res / scale_p
                d_res_rel_now = d_res / scale_d
                if gap_rel_now < near_tol * opts.ϵ_gap && p_res_rel_now < near_tol * opts.ϵ_primal &&
                   d_res_rel_now < near_tol * opts.ϵ_dual
                    status, message = Stalled,
                    "Step size collapsed within $(Float64(near_tol))×, but not within, " *
                    "the requested tolerance. Returning the best iterate without an " *
                    "Optimal certificate; use a wider arithmetic type or loosen the tolerance."
                    termination_reason = :step_collapse_near_tolerance
                    termination_stage = :line_search
                    break
                end
            end
            # §4.2: cap the escalation for dynamic-range-limited types so repeated
            # restarts can't overflow straight to NaN the way an unconditional
            # ×omega_step would (BigFloat/Float64 have enough range that opts.max_omega
            # already bounds this sensibly; fixed-width types need a tighter, type-aware cap).
            effective_omega_step = dynamic_range_limited(T) ?
                                    min(
                                        selected_parameters.restart_scale,
                                        sqrt(floatmax(T)),
                                    ) :
                                    selected_parameters.restart_scale
            # A restart exists to repair a badly *scaled* starting point: it
            # multiplies the collapsed side by `omega_step` (1e5 by default) and
            # re-centres. That is the right response early, when the iterate is
            # still far from the central path.
            #
            # It is the wrong response once the solve has clearly made progress.
            # A step collapsing then means the working precision is exhausted,
            # and rescaling by 1e5 destroys the good iterate instead of saving
            # it. The plain stall counter never fires then because a restart
            # does not increment `iter`, so the restart budget is spent first
            # and the run ends in `MaxRestartsExceeded` with the best point
            # discarded.
            #
            # So: if the merit has improved by orders of magnitude since the
            # start, treat a collapsed step as precision exhaustion and stop
            # with the retained best iterate.
            # Two conditions, not one. Requiring only "improved a lot since the
            # start" is a false positive on badly *scaled* problems: a solve
            # that starts far from the central path can improve its merit by
            # several orders of magnitude in the first few iterations purely by
            # shrinking huge initial residuals, and would then be declared
            # stalled while still converging. Precision exhaustion means the
            # iterate is genuinely *near* a solution, so also require the merit
            # to be small in absolute terms.
            # `merit` is now normalised by the requested tolerances, so
            # `merit <= 1` *is* convergence and "genuinely near a solution"
            # means within a small multiple of 1 — not the old absolute 1e-4,
            # which under the new scaling would mean 10000x better than asked.
            made_real_progress = isfinite(initial_merit) &&
                                 best_merit < initial_merit * T(1e-3) &&
                                 best_merit < T(NEAR_SOLUTION_MERIT)
            if opts.stall_iterations > 0 && made_real_progress
                status, message = InsufficientPrecision,
                "Step size collapsed after the solve had already converged by a " *
                "factor of $(round(Float64(initial_merit / max(best_merit, eps(T))), sigdigits=3)) " *
                "in the scaled merit; this is precision exhaustion at $(T), not bad " *
                "initial scaling, so restarting would discard the best iterate. " *
                "Use a wider arithmetic type (for example Float64x2) or loosen the tolerance."
                termination_reason = :precision_floor
                termination_stage = :line_search
                break
            end
            # Before either giving up or rescaling: try *recentering*.
            #
            # A step can collapse for a reason neither branch below addresses.
            # The primal can sit at `p_res ≈ 1e-47` (exactly feasible) with a
            # KKT residual ~1e-48 — far above `eps` of the working type, so the
            # direction is accurate and the precision is nowhere near exhausted
            # — and yet the step falls under `min_step = 1e-10` while the
            # duality gap is still ~1e-3. What has happened is that the iterate
            # has run into the boundary of the PSD cone far from the optimum —
            # too little centering, not bad scaling and not lost precision.
            #
            # The repair for that is to aim the next direction back at the
            # central path by raising β, which is cheap and non-destructive: it
            # keeps the iterate and only changes the target. Rescaling by 1e5
            # (below) would throw away a feasible primal, and stopping would
            # discard a solve that can still make progress.
            if centering_attempts < opts.max_centering &&
               parameter_controller.beta < T(CENTERING_BETA_MAX)
                centering_attempts += 1
                # Move the controller's β, not just μ. μ is recomputed from
                # `parameter_controller.beta` at the end of every accepted
                # iteration, so writing μ alone would be undone immediately and
                # buy exactly one centered step.
                parameter_controller.beta = min(
                    parameter_controller.beta * T(CENTERING_BETA_STEP),
                    T(CENTERING_BETA_MAX))
                effective_beta = parameter_controller.beta
                for l in 1:L
                    μ[l] = effective_beta * kdot(X[l], Y[l]) / k[l]
                end
                opts.verbosity >= 1 && println(
                    "Step size too small, but both the direction and the residuals are " *
                    "healthy — recentering $centering_attempts/$(opts.max_centering) " *
                    "with β ← $(Float64(effective_beta)) instead of rescaling.")
                continue
            end
            # Rescale only a side that is *actually* infeasible. The collapsed
            # step tells you which side stopped moving, not which side is badly
            # scaled, and those are different questions: an exactly feasible
            # side (primal residual ~1e-48) can still see its step collapse
            # while the other residual crawls down, and both steps then
            # collapse together. Rescaling that feasible side by 1e5 destroys a
            # perfectly good iterate — the residual walks from ~1e-48 up
            # through 1e+8, 1e+13, 1e+18, 1e+23 over successive restarts,
            # converging on nothing.
            rescale_X = x_stuck && !primal_feasible
            rescale_Y = y_stuck && !dual_feasible
            if opts.restart && restarts < opts.max_restarts && (rescale_X || rescale_Y)
                restarts += 1
                if rescale_X
                    for l in 1:L
                        X[l] .*= effective_omega_step
                    end
                end
                if rescale_Y
                    for l in 1:L
                        Y[l] .*= effective_omega_step
                    end
                end
                current_complementarity = zero(T)
                for l in 1:L
                    block_complementarity = kdot(X[l], Y[l])
                    current_complementarity += block_complementarity
                    μ[l] =
                        parameter_controller.beta *
                        block_complementarity /
                        k[l]
                end
                opts.verbosity >= 1 &&
                    println("Step size too small! Restart $restarts/$(opts.max_restarts): rescaling the collapsed side(s) by ×$(Float64(effective_omega_step)).")
                continue
            elseif opts.restart && restarts < opts.max_restarts
                # Both collapsed sides are already feasible, so there is nothing
                # for a rescale to repair; the step is limited by centering or
                # precision, not by the starting scale.
                status, message = Stalled,
                "Step size collapsed while both residuals were already within " *
                "tolerance (p_res=$(Float64(p_res)), d_res=$(Float64(d_res))); " *
                "rescaling would only discard the converged iterate."
                termination_reason = :step_collapse_feasible
                termination_stage = :line_search
                break
            else
                status = restarts >= opts.max_restarts ? MaxRestartsExceeded : NumericalBreakdown
                message = restarts >= opts.max_restarts ?
                           "Step size collapsed after using up max_restarts=$(opts.max_restarts) rescue attempts." :
                           "Step size collapsed below min_step and restart=false."
                termination_reason = restarts >= opts.max_restarts ?
                                     :max_restarts : :step_collapse
                termination_stage = :line_search
                break
            end
        end

        update_started = time_ns()
        complementarity_after, finite_blocks = threaded_update_blocks!(
            ws,
            X,
            Y,
            tX,
            tY,
        )
        trial_combine_owned!(x, x, tX, ws.dx)
        n > 0 && trial_combine_owned!(y, y, tY, ws.dy)

        if !(
            finite_blocks &&
            all(isfinite, x) &&
            all(isfinite, y) &&
            all(isfinite, μ)
        )
            status, message = NumericalBreakdown,
            "non-finite primal or dual iterate detected" *
            (dynamic_range_limited(T) ? " ($T's dynamic range exceeded)" : "") *
            " — rescale (scaling=:equilibrate), loosen Ωp/Ωd/omega_step, or use a wider-range T"
            termination_reason = :nonfinite_iterate
            termination_stage = :iterate_update
            break
        end

        primal_residual_after, dual_residual_after =
            _accepted_sdp_trial_residuals!(
                ws,
                result.block_primal_residual,
                tX,
                tY,
            )

        record_and_update!(
            parameter_controller;
            iteration=iter + 1,
            predictor_quality=result.predictor_quality,
            complementarity_before=result.complementarity,
            complementarity_after=complementarity_after,
            primal_residual=p_res,
            dual_residual=d_res,
            primal_step=tX,
            dual_step=tY,
            primal_residual_after=primal_residual_after,
            dual_residual_after=dual_residual_after,
            backtracking_count=backtracking_count,
            affine_primal_step=result.affine_primal_step,
            affine_dual_step=result.affine_dual_step,
            mu_before=result.mu,
            mu_affine=result.mu_aff,
            relative_gap=gap_rel,
            regularization=result.regularization,
            refinement_count=result.refine_steps,
            refinement_residual=result.refine_residual,
            factorization_quality=result.factorization_quality,
            primal_psd_margin=result.primal_psd_margin,
            dual_psd_margin=result.dual_psd_margin,
            precision_floor=result.iteration_diagnostics.precision_floor,
            selected_parameters=selected_parameters,
        )
        phase_update += (time_ns() - update_started) / 1.0e9
        current_complementarity = complementarity_after

        # `threaded_update_blocks!` cached every post-step `dot(X_l, Y_l)` in
        # `ws.block_norms`. Consume those values before the dual-objective
        # evaluation reuses the same scratch vector for `dot(C_l, Y_l)`.
        objective_and_targets_started = time_ns()
        threaded_update_mu!(
            ws,
            μ,
            parameter_controller.beta,
            k,
            complementarity_after,
            parameter_controller.strategy === :adaptive &&
            !parameter_controller.fallback,
        )
        if !all(isfinite, μ)
            status, message = NumericalBreakdown,
            "non-finite complementarity target detected"
            termination_reason = :nonfinite_complementarity
            termination_stage = :target_update
            break
        end

        pObj = LinearAlgebra.dot(solve_prob.c, x)
        dObj = threaded_dual_objective(ws, solve_prob, y, Y)
        if !isfinite(pObj) || !isfinite(dObj)
            status, message = NumericalBreakdown,
            "non-finite primal or dual objective detected"
            termination_reason = :nonfinite_objective
            termination_stage = :objective_update
            break
        end
        phase_objective_and_targets +=
            (time_ns() - objective_and_targets_started) / 1.0e9
        iter += 1

        # The accepted affine residuals include the final inexact-direction
        # terms recorded by the unregularized structured KKT operator.  The
        # vector update above is exact and O(m+n); prospective success and the
        # final returned point are still recomputed/certified independently in
        # original coordinates.
        p_res = primal_residual_after
        d_res = dual_residual_after
        t2 = time()
        print_iter(opts, iter, pObj, dObj, pObj - dObj, p_res, d_res, tX, tY, t2 - t1)

        if opts.checkpoint_every > 0 && !isempty(opts.checkpoint_path) && iter % opts.checkpoint_every == 0
            save_checkpoint(opts.checkpoint_path, T, x, X, y, Y, μ, iter, restarts, solve_prob.dims)
        end
        opts.force_gc && _release_iteration_memory!()
    end

    # A future guard added to the loop must not silently regress to the old
    # uninformative `:none` record.  All current non-success branches assign a
    # more specific reason above; this fail-closed fallback keeps the metadata
    # contract explicit if a new status is introduced later.
    if status !== Optimal && termination_reason === :none
        termination_reason = status === FeasibleCert ?
                             :feasible_certificate :
                             status === InfeasibleCert ?
                             :infeasible_certificate :
                             status === IterLimit ?
                             :iteration_limit :
                             status === TimeLimit ?
                             :time_limit :
                             status === UserStopped ?
                             :user_stopped : :nonoptimal_exit
        termination_stage = :termination_check
    end

    # On a non-optimal exit, hand back the best point actually visited rather
    # than wherever the iteration happened to stop. Restricted to cases where
    # the retained point is meaningfully better than the final one, so ordinary
    # `IterLimit` runs that were still improving are reported as-is.
    if status !== Optimal && status !== FeasibleCert && status !== InfeasibleCert &&
       best_iterate.valid
        final_merit = max(p_res / scale_p, d_res / scale_d,
            abs(pObj - dObj) / max(one(T), (abs(pObj) + abs(dObj)) / 2))
        if !isfinite(final_merit) || best_merit < final_merit / 2
            best = best_iterate
            x, X, y, Y = best.x, best.X, best.y, best.Y
            pObj, dObj = best.pObj, best.dObj
            p_res, d_res = best.p_res, best.d_res
            message *= " Returned the best iterate found (iteration $(best.iter), " *
                       "scaled merit $(round(Float64(best_merit), sigdigits=3)) versus " *
                       "$(round(Float64(final_merit), sigdigits=3)) at exit)."
            opts.verbosity >= 1 && println(
                "Reporting best iterate from iteration $(best.iter) " *
                "(scaled merit $(round(Float64(best_merit), sigdigits=3)))."
            )
        end
    end

    finalization_started_ns = time_ns()
    if eq !== nothing
        x, X, y, Y = unequilibrate(eq, x, X, y, Y)
        pObj = LinearAlgebra.dot(prob.c, x)
        dObj = dual_objective(prob, y, Y)
    end
    # `X` is an affine slack, not an independent model variable:
    #
    #     X_l = sum_i A_i^(l) * x_i - C_l.
    #
    # Reconstruct it from the original data before certification. A tiny
    # residual in equilibrated coordinates can be amplified substantially by
    # the inverse block congruence, even when the returned `x` defines a
    # numerically sound original-coordinate PSD matrix. Reusing that amplified
    # workspace slack would therefore report a misleading primal residual.
    # Rebuilding changes neither `x` nor the objective or iteration trajectory;
    # it makes the returned slack satisfy the original affine definition
    # exactly, after which the PSD certificate tests the matrix that `x`
    # actually defines.
    @inbounds for block in eachindex(X)
        buildP_owned!(X[block], prob.cons, block, x)
        kaxpby_owned!(
            -one(T),
            prob.C[block],
            one(T),
            X[block],
        )
    end
    # Always report certificates in the same (original) coordinates as the
    # returned iterate, including non-optimal exits and unequilibrated solves.
    p_res, d_res = solution_residuals(prob, x, X, y, Y)
    finalization_finished_ns = time_ns()

    elapsed = time() - t_start
    phase_setup_validation =
        (validation_finished_ns - core_started_ns) / 1.0e9
    phase_precision_preparation =
        (precision_preparation_finished_ns - validation_finished_ns) /
        1.0e9
    phase_equilibration =
        (equilibration_finished_ns - precision_preparation_finished_ns) /
        1.0e9
    phase_parameter_selection =
        (parameter_selection_finished_ns - equilibration_finished_ns) /
        1.0e9
    phase_workspace_setup =
        (workspace_finished_ns - parameter_selection_finished_ns) /
        1.0e9
    phase_initialization =
        (initialization_finished_ns - workspace_finished_ns) / 1.0e9
    phase_initial_residual =
        (initial_residual_finished_ns - initialization_finished_ns) /
        1.0e9
    phase_finalization =
        (finalization_finished_ns - finalization_started_ns) / 1.0e9
    accounted = (
        phase_setup_validation +
        phase_precision_preparation +
        phase_equilibration +
        phase_parameter_selection +
        phase_workspace_setup +
        phase_initialization +
        phase_initial_residual +
        phase_residual +
        phase_schur +
        phase_kkt +
        phase_predictor +
        phase_corrector +
        phase_line_search +
        phase_update +
        phase_best_iterate +
        phase_objective_and_targets +
        phase_finalization
    )
    timings = opts.timing ? (
        total=elapsed,
        setup_validation=phase_setup_validation,
        precision_preparation=phase_precision_preparation,
        equilibration=phase_equilibration,
        parameter_selection=phase_parameter_selection,
        workspace_setup=phase_workspace_setup,
        initialization=phase_initialization,
        initial_residual=phase_initial_residual,
        residual_and_block_factor=phase_residual,
        schur_assembly=phase_schur,
        kkt_factorization=phase_kkt,
        predictor=phase_predictor,
        corrector=phase_corrector,
        kkt_schur_copy=phase_kkt_schur_copy,
        kkt_schur_factorization=phase_kkt_schur_factorization,
        kkt_constraint_triangular_solve=
            phase_kkt_constraint_triangular_solve,
        kkt_equality_gram=phase_kkt_equality_gram,
        kkt_equality_factorization=
            phase_kkt_equality_factorization,
        kkt_other=phase_kkt_other,
        predictor_rhs=phase_predictor_rhs,
        predictor_linear_solve=phase_predictor_linear_solve,
        predictor_direction_recovery=
            phase_predictor_direction_recovery,
        complementarity_analysis=phase_complementarity_analysis,
        corrector_rhs=phase_corrector_rhs,
        corrector_linear_solve=phase_corrector_linear_solve,
        refinement=phase_refinement,
        corrector_direction_recovery=
            phase_corrector_direction_recovery,
        line_search=phase_line_search,
        update=phase_update,
        best_iterate=phase_best_iterate,
        objective_and_targets=phase_objective_and_targets,
        finalization=phase_finalization,
        other=max(0.0, elapsed - accounted),
    ) : nothing
    gap_rel_final = abs(pObj - dObj) / max(one(T), (abs(pObj) + abs(dObj)) / 2)
    equality_diagnostics =
        _equality_factor_diagnostics(ws, solve_prob.dims.n)

    return SDPResult{T}(
        status,
        message,
        x,
        X,
        y,
        Y,
        pObj,
        dObj,
        gap_rel_final,
        p_res,
        d_res,
        iter,
        restarts,
        total_reg,
        timings,
        parameter_controller.history,
        nothing,
        (
            reason=termination_reason,
            stage=termination_stage,
            # Keep the detector's raw signal available to diagnostics users;
            # `reason` above is the stable terminal-path classification.
            stagnation_reason=stagnation.reason,
            merit=best_merit,
            rate=stagnation.rate,
            projected_iterations=stagnation.projected,
            window=stagnation.window,
            total_refinement_steps=total_refine_steps,
            equilibration=(
                enabled=eq !== nothing,
                adaptive=true,
                passes=eq === nothing ? Int[] : copy(eq.ruiz_passes),
            ),
            mixed_precision_kkt=
                _mixed_precision_kkt_diagnostics(ws),
            sparse_schur_backend=
                ws.sparse_kkt === nothing ? nothing :
                let diagnostics = sparse_schur_diagnostics(ws, solve_prob),
                    sparse_workspace = ws.sparse_kkt
                    merge(
                        diagnostics,
                        (
                            # Preserve the historical termination keys used by
                            # diagnostics consumers while exposing the richer
                            # frozen-pattern metrics above.
                            schur_nnz=diagnostics.structural_nnz,
                            regularization=sparse_workspace isa
                                           GenericSparseSchurSDPWorkspace ?
                                           sparse_workspace.regularization :
                                           nothing,
                            equality_requires_pivoting=
                                sparse_workspace isa GenericSparseSchurSDPWorkspace ?
                                sparse_workspace.equality_requires_pivoting :
                                false,
                        ),
                    )
                end,
            equality_system=equality_diagnostics,
            augmented_kkt=
                ws.augmented === nothing ?
                (available=false,) :
                let augmented =
                        ws.augmented::DenseAugmentedKKTWorkspace{T},
                    factor = augmented.factor
                    factor === nothing ?
                    (
                        available=false,
                        factorization=:pivoted_symmetric_ldlt,
                        regularization=augmented.regularization,
                        rank_deficient=augmented.rank_deficient,
                        inertia=augmented.inertia,
                        factor_diagnostics=augmented.factor_diagnostics,
                    ) :
                    (
                        available=true,
                        dimension=size(augmented.matrix, 1),
                        factorization=:pivoted_symmetric_ldlt,
                        factor_kind=la_factor_kind(factor),
                        factor_provider=la_factor_provider_identity(
                            la_factor_provider(factor),
                        ),
                        factor_precision=augmented.factor_precision,
                        inertia=augmented.inertia,
                        pivot_blocks=augmented.pivot_blocks,
                        permutation=augmented.permutation,
                        factor_diagnostics=augmented.factor_diagnostics,
                        regularization=augmented.regularization,
                    )
                end,
            executed=(
                solver=:sdp,
                parameter_profile=executed_parameters.profile,
                executed_parameters=(
                    beta=executed_parameters.beta,
                    gamma=executed_parameters.gamma,
                    omega_p=executed_parameters.omega_p,
                    omega_d=executed_parameters.omega_d,
                    predictor=executed_parameters.predictor,
                    strategy=executed_parameters.strategy,
                    adaptive_sigma_max=
                        executed_parameters.adaptive_sigma_max,
                ),
                parameter_source,
                parameter_resolution_count,
                stage=parameter_resolution_count == 1 ?
                    :post_scaling : :not_applicable,
                kkt=ws.executed_backend,
                planned_backend=planned_backend_name(ws),
                executed_backend=ws.executed_backend,
                kkt_formulation=kkt_formulation_from_backend(
                    ws.executed_backend,
                ),
                fallback_reason=ws.backend_fallback_reason,
                la_backend=ws.executed_la_backend,
                la_provider=ws.executed_la_provider,
                la_ownership=ws.executed_la_ownership,
                la_fallback_reason=ws.la_fallback_reason,
                la_factorization=if ws.augmented !== nothing
                    :pivoted_symmetric_ldlt
                elseif ws.executed_backend === :dense_cholesky
                    :cholesky
                elseif ws.executed_backend === :mixed_precision
                    :mixed_precision_cholesky
                elseif ws.executed_backend === :sparse_schur_cholesky
                    :sparse_cholesky
                elseif ws.executed_backend === :block_arrow
                    :block_cholesky
                else
                    :not_executed
                end,
                la_regularization=
                    ws.augmented === nothing ?
                    nothing :
                    (ws.augmented::DenseAugmentedKKTWorkspace{T}).regularization,
                factor_diagnostics=
                    ws.augmented === nothing ?
                    nothing :
                    (ws.augmented::DenseAugmentedKKTWorkspace{T}).factor_diagnostics,
                equality=equality_diagnostics.method,
                effective_threads=ws.thread_count,
                fine_grained_block_tasks=length(ws.block_bins),
                fine_grained_block_partition=
                    fine_grained_block_partition(
                        T,
                        ws.arrow !== nothing &&
                        (ws.arrow::ArrowWorkspace{T}).reduced_panel_enabled,
                        prob.dims.k,
                        length(ws.block_bins),
                    ),
                schur_threads=
                    ws.arrow !== nothing &&
                    (ws.arrow::ArrowWorkspace{T}).reduced_panel_enabled ?
                    reduced_arrow_worker_count(
                        T,
                        ws.thread_count,
                        length(ws.blk),
                        length((ws.arrow::ArrowWorkspace{T}).global_ids),
                    ) :
                    ws.thread_count,
                factor_threads=
                    ws.arrow !== nothing &&
                    (ws.arrow::ArrowWorkspace{T}).reduced_panel_enabled ?
                    reduced_arrow_factor_worker_count(
                        T,
                        ws.thread_count,
                        length((ws.arrow::ArrowWorkspace{T}).global_ids),
                    ) : nothing,
                arrow_linear_solve=
                    ws.arrow !== nothing &&
                    (ws.arrow::ArrowWorkspace{T}).reduced_panel_enabled ?
                    reduced_arrow_simd_solve(T) ?
                    :multifloatvec4_simd_singleton :
                    :scalar_singleton : nothing,
                initialization=initialization_record,
            ),
        ),
    )
end

"""
    block_norm_stats(prob) -> (norms, gmean, maxnorm, spread)

Per-block `‖C_l‖∞` plus summary statistics for the explicit expert
`omega_scaling=:per_block` mode. Automatic KKT initialization does not call
this helper. Zero or non-finite norms are replaced by one so a block with no
constant term does not collapse the geometric mean.
"""
function block_norm_stats(prob::SDPProblem{T}) where {T}
    L = prob.dims.L
    norms = Vector{T}(undef, L)
    @inbounds for l in 1:L
        nl = knrmInf(prob.C[l])
        norms[l] = (nl > zero(T) && isfinite(nl)) ? nl : one(T)
    end
    lo, hi = extrema(norms)
    gmean = exp(sum(log, norms) / L)
    (isfinite(gmean) && gmean > zero(T)) || (gmean = one(T))
    return (norms=norms, gmean=gmean, maxnorm=hi, spread=hi / lo)
end

"""A tolerance-normalised merit below this counts as "near a solution": within
this factor of the tolerance the user actually asked for."""
const NEAR_SOLUTION_MERIT = 100

"""Each recentering attempt multiplies β by this factor."""
const CENTERING_BETA_STEP = 4

"""Recentering never pushes β past this, which is already heavy centering."""
const CENTERING_BETA_MAX = 0.5

"""
    initial_block_scales(prob, opts) -> Vector{T}

Expert fixed-policy multipliers for `X_l = Ωp·s_l·I`, `Y_l = Ωd·s_l·I`.

`:scalar` (and `:auto`) return all ones — the classical single-Ω start.
`:per_block` returns `s_l = ‖C_l‖∞ / geomean(‖C‖∞)`, giving `X_l ≈ ‖C_l‖∞·I`
when paired with `Ω = geomean`.

`:auto` resolves to `:scalar` only for compatibility with explicit fixed-policy
initialization. Automatic KKT initialization bypasses this helper entirely.
`:per_block` is retained solely as an expert fixed-policy option.
"""
function initial_block_scales(prob::SDPProblem{T}, opts::SolverOptions{T}) where {T}
    mode = opts.omega_scaling
    (mode === :scalar || mode === :auto) && return ones(T, prob.dims.L)
    mode === :per_block ||
        throw(ArgumentError("omega_scaling must be :scalar, :per_block, or :auto, got $(mode)"))

    stats = block_norm_stats(prob)
    norms = stats.norms
    @inbounds for l in eachindex(norms)
        norms[l] /= stats.gmean
    end
    return norms
end

"""
    solve!(prob, options=SolverOptions{T}(); warm-start keywords...) -> SDPResult

Run the automatic optimization pipeline. The pipeline classifies the model,
removes dependent equalities, chooses scaling and kernels, dispatches pure
`1×1`-cone models to the dedicated LP solver, and otherwise preserves the
existing SDP implementation.
"""

function solve!(
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}();
    x0=nothing,
    X0=nothing,
    y0=nothing,
    Y0=nothing,
    resume::AbstractString="",
    _prepared_data=nothing,
) where {T}
    return _solve_pipeline!(
        prob,
        opts;
        x0=x0,
        X0=X0,
        y0=y0,
        Y0=Y0,
        resume=resume,
        _prepared_data=_prepared_data,
    )
end

"""
    adaptive_working_precision_bits(problem, options) -> Int

Choose the first precision for a staged BigFloat solve. The requested
`precision_bits` remains a hard upper bound and fallback precision. The guard
combines the smallest requested tolerance with 96 safety bits and a
dimension-dependent term, rounds upward to a 32-bit boundary, and never drops
below `minimum_working_precision_bits`.

The deliberately large guard is appropriate for an interior-point Newton
system: this selector is intended to skip obviously unnecessary MPFR limbs,
not to estimate the fewest bits with which a problem might happen to solve.
"""
function adaptive_working_precision_bits(
    prob::SDPProblem{BigFloat},
    opts::SolverOptions{BigFloat},
)
    requested = opts.precision_bits
    opts.working_precision_policy === :auto || return requested
    tolerances = (opts.ϵ_gap, opts.ϵ_primal, opts.ϵ_dual)
    any(iszero, tolerances) && return requested
    tolerance = min(tolerances...)
    isfinite(tolerance) && tolerance > zero(BigFloat) ||
        return requested
    accuracy_bits = max(0, ceil(Int, -log2(tolerance)))
    problem_dimension = max(
        prob.dims.m,
        prob.dims.n,
        sum(prob.dims.k),
        2,
    )
    dimension_guard = ceil(Int, log2(problem_dimension))
    guarded_bits = accuracy_bits + 96 + dimension_guard
    rounded_bits = 32 * cld(guarded_bits, 32)
    minimum_bits = min(
        opts.minimum_working_precision_bits,
        requested,
    )
    return clamp(rounded_bits, minimum_bits, requested)
end

@inline _working_precision_success(status::SolveStatus) =
    status in (
        Optimal,
        FeasibleCert,
        InfeasibleCert,
        PrimalInfeasible,
        DualInfeasible,
    )

function _record_working_precision!(
    result::SDPResult,
    message::String,
)
    result.diagnostics === nothing ||
        push!(result.diagnostics.warnings, message)
    return result
end

"""Build the immutable pre-execution ladder authority for a BigFloat solve.
The ladder is built unconditionally before the first rung executes. The
selector semantics are unchanged: `:fixed` and resume use exactly the
requested rung; `:auto` selects at most two rungs (the adaptive lower choice
plus the requested upper rung)."""
function _build_precision_ladder_plan(
    prob::SDPProblem{BigFloat},
    opts::SolverOptions{BigFloat};
    resume::AbstractString="",
)
    requested_bits = opts.precision_bits
    resume_bypass = !isempty(resume)
    selected_bits = resume_bypass ?
                    requested_bits :
                    adaptive_working_precision_bits(prob, opts)
    rungs = if selected_bits == requested_bits
        (
            PrecisionAttemptSpec(1, requested_bits, :requested),
        )
    else
        (
            PrecisionAttemptSpec(1, selected_bits, :selected),
            PrecisionAttemptSpec(2, requested_bits, :requested),
        )
    end
    selection_reason = resume_bypass ? :resume :
                       opts.working_precision_policy === :fixed ? :fixed :
                       selected_bits < requested_bits ? :adaptive_lower :
                       :adaptive_requested
    return PrecisionLadderPlan(
        opts.working_precision_policy,
        requested_bits,
        selected_bits,
        min(opts.minimum_working_precision_bits, requested_bits),
        resume_bypass,
        selection_reason,
        rungs,
        (
            AlmostOptimal,
            InsufficientPrecision,
            Stalled,
            NumericalBreakdown,
            NumericalFailure,
            MaxRestartsExceeded,
        ),
        :shared_wall_clock,
    )
end

"""The ladder's own retry decision for a completed rung, driven exclusively
by the ladder plan's retry-eligibility set and the shared remaining budget —
never by a separate authority."""
@inline function _ladder_retry_decision(
    plan::PrecisionLadderPlan,
    rung::Int,
    status::SolveStatus,
    remaining_budget_seconds::Float64,
)
    # There is no next rung: the ladder stops regardless of the outcome.
    rung >= length(plan.rungs) && return :terminal
    _working_precision_success(status) && return :success
    status in plan.retry_statuses || return :ineligible_status
    remaining_budget_seconds > 0 || return :no_time
    return :retry
end

"""Patch the just-completed rung's report with the orchestrator's
authoritative retry decision and shared remaining budget. The report built
inside `_attach_diagnostics` records the same rule provisionally; the
orchestrator owns the final gate and the true remaining wall-clock budget
(which includes inter-rung overhead such as the release/GC step)."""
function _patch_ladder_report!(
    context::PrecisionLadderContext,
    decision::Symbol,
    remaining_budget_seconds::Float64,
)
    old = context.reports[end]
    facts = old.facts
    context.reports[end] = PrecisionAttemptReport(
        old.spec,
        old.child_plan,
        old.record,
        PrecisionAttemptScalarFacts(
            facts.status,
            facts.termination_reason,
            facts.elapsed_seconds,
            facts.success,
            decision,
            remaining_budget_seconds,
        ),
    )
    return context
end

"""Flatten the per-rung A0 records into the final diagnostics. `result` is
the final rung's result; its diagnostics already carry the full ladder report
(every rung, in order) and the final child plan. `attempts` is replaced with
the flattened per-rung records; every other diagnostics field is preserved."""
function _merge_ladder_result(
    result::SDPResult{T},
    ladder_context::PrecisionLadderContext,
) where {T}
    # Diagnostics-disabled runs never build attempt records; there is nothing
    # to merge and the result already carries the final rung's payload.
    result.diagnostics === nothing && return result
    attempts = Tuple(
        report.record for report in ladder_context.reports
    )
    diagnostics = result.diagnostics
    merged_diagnostics = SolveDiagnostics(
        diagnostics.classification,
        diagnostics.plan,
        diagnostics.presolve,
        diagnostics.timings,
        diagnostics.memory,
        diagnostics.selected_algorithms,
        diagnostics.parameter_history,
        diagnostics.warnings,
        diagnostics.termination,
        attempts,
        PrecisionLadderReport(
            ladder_context.plan,
            Tuple(copy(ladder_context.reports)),
        ),
    )
    return SDPResult{T}(
        result.status,
        result.message,
        result.x,
        result.X,
        result.y,
        result.Y,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.restarts,
        result.regularizations,
        result.timings,
        result.parameter_history,
        merged_diagnostics,
        result.termination,
    )
end

function solve!(
    prob::SDPProblem{BigFloat},
    opts::SolverOptions{BigFloat}=SolverOptions{BigFloat}();
    x0=nothing,
    X0=nothing,
    y0=nothing,
    Y0=nothing,
    resume::AbstractString="",
    _prepared_data=nothing,
)
    _validate_solver_options(opts)
    ladder_plan = _build_precision_ladder_plan(
        prob,
        opts;
        resume=resume,
    )
    requested_precision = ladder_plan.requested_bits
    selected_precision = ladder_plan.selected_bits
    ladder_started_ns = time_ns()
    ladder_context = PrecisionLadderContext(
        ladder_plan,
        1,
        1,
        ladder_plan.rungs[1].bits,
        ladder_started_ns,
        isfinite(opts.max_time) ? opts.max_time : Inf,
        PrecisionAttemptReport[],
    )

    function run_at_precision(run_options, bits, context)
        reusable_prepared_data =
            _prepared_data !== nothing &&
            get(_prepared_data, :precision_bits, 0) == bits ?
            _prepared_data : nothing
        run = () -> _solve_pipeline!(
            prob,
            run_options;
            x0=x0,
            X0=X0,
            y0=y0,
            Y0=Y0,
            resume=resume,
            _prepared_data=reusable_prepared_data,
            ladder_context=context,
        )
        return Base.precision(BigFloat) == bits ?
               run() :
               setprecision(run, BigFloat, bits)
    end

    first_spec = ladder_plan.rungs[1]
    first_options = if first_spec.bits == requested_precision
        opts
    else
        setprecision(BigFloat, first_spec.bits) do
            _reround_solver_options(
                opts,
                first_spec.bits;
                precision_bits=first_spec.bits,
                working_precision_policy=:fixed,
            )
        end
    end
    lower_result = run_at_precision(
        first_options,
        first_spec.bits,
        ladder_context,
    )
    if opts.working_precision_policy === :auto && !isempty(resume)
        _record_working_precision!(
            lower_result,
            "Adaptive working precision was bypassed while resuming a " *
            "checkpoint; the requested $(requested_precision)-bit " *
            "precision was used.",
        )
        return _merge_ladder_result(lower_result, ladder_context)
    end
    length(ladder_plan.rungs) == 1 &&
        return _merge_ladder_result(lower_result, ladder_context)

    elapsed = (time_ns() - ladder_started_ns) / 1.0e9
    remaining_time = isfinite(opts.max_time) ?
                     max(0.0, opts.max_time - elapsed) :
                     Inf
    decision = _ladder_retry_decision(
        ladder_plan,
        1,
        lower_result.status,
        remaining_time,
    )
    # Diagnostics-enabled runs carry the same rule inside the rung report;
    # the orchestrator's authoritative clock and shared-budget gate may
    # refine the provisional decision (inter-rung overhead), so the report is
    # patched with the final decision and remaining budget. Diagnostics-
    # disabled runs never build reports; the decision above is all they need.
    isempty(ladder_context.reports) ||
        _patch_ladder_report!(ladder_context, decision, remaining_time)

    if decision === :success
        message = opts.certification ?
                  "Adaptive working precision selected " *
                  "$(selected_precision) of $(requested_precision) " *
                  "requested bits; the result passed original-coordinate " *
                  "certification without a retry." :
                  "Adaptive working precision selected " *
                  "$(selected_precision) of $(requested_precision) " *
                  "requested bits; final certification was disabled."
        _record_working_precision!(lower_result, message)
        return _merge_ladder_result(lower_result, ladder_context)
    end

    if decision !== :retry
        _record_working_precision!(
            lower_result,
            "Adaptive working precision selected $(selected_precision) of " *
            "$(requested_precision) requested bits, but the run ended with " *
            "$(lower_result.status); the fallback was not eligible or its " *
            "time budget was exhausted.",
        )
        return _merge_ladder_result(lower_result, ladder_context)
    end

    lower_status = lower_result.status
    lower_result = nothing
    # A fallback is already an exceptional, expensive path. Release the
    # lower-precision iterate before allocating the requested-precision
    # workspace so the retry does not retain two full SDP solutions.
    GC.gc()
    fallback_options = _replace_solver_options(
        opts;
        working_precision_policy=:fixed,
        max_time=remaining_time,
    )
    fallback_context = PrecisionLadderContext(
        ladder_plan,
        2,
        2,
        requested_precision,
        time_ns(),
        remaining_time,
        ladder_context.reports,
    )
    fallback_result = run_at_precision(
        fallback_options,
        requested_precision,
        fallback_context,
    )
    _record_working_precision!(
        fallback_result,
        "Adaptive working precision first tried $(selected_precision) bits " *
        "and ended with $(lower_status); SDPX retried at the requested " *
        "$(requested_precision)-bit precision.",
    )
    return _merge_ladder_result(fallback_result, fallback_context)
end

function _solve_pipeline!(
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}();
    x0=nothing,
    X0=nothing,
    y0=nothing,
    Y0=nothing,
    resume::AbstractString="",
    _prepared_data=nothing,
    ladder_context::Union{Nothing,PrecisionLadderContext}=nothing,
) where {T}
    _validate_solver_options(opts)
    pipeline_started = time()
    pipeline_preprocess_seconds = 0.0
    pipeline_equality_presolve_seconds = 0.0
    pipeline_structural_analysis_seconds = 0.0
    pipeline_execution_planning_seconds = 0.0
    pipeline_certification_seconds = 0.0
    deadline = isfinite(opts.max_time) ?
               pipeline_started + opts.max_time :
               Inf
    if _prepared_data !== nothing
        get(_prepared_data, :precision_bits, 0) == _preprocess_precision_bits(T) ||
            throw(PreparedStructureMismatch(
                :arithmetic_precision_changed,
                "prepared preprocessing data was built at a different " *
                "arithmetic precision",
            ))
    end
    preprocess_started = time_ns()
    preprocessed = _prepared_data === nothing ?
                   preprocess(prob, opts) :
                   _prepared_data.preprocessed
    pipeline_preprocess_seconds = _prepared_data === nothing ?
                                  (time_ns() - preprocess_started) / 1.0e9 :
                                  0.0
    equality_presolve_started = time_ns()
    reduced, equality_map, equality_report = if _prepared_data === nothing
        presolve_equalities(preprocessed.problem, opts)
    else
        (
            _prepared_data.reduced,
            _prepared_data.equality_map,
            _prepared_data.equality_report,
        )
    end
    pipeline_equality_presolve_seconds = _prepared_data === nothing ?
        (time_ns() - equality_presolve_started) / 1.0e9 : 0.0
    report = _merge_presolve_reports(
        preprocessed,
        equality_map,
        equality_report,
        reduced,
    )
    # Finalize the structural plan against the model that will actually be
    # factorized. The automatic initial point remains solve-local and is built
    # only after scaling; planning sees no Ω heuristic or benchmark profile.
    # Presolve still changes the diagnostic equality count.
    planning_problem = report.inconsistent ? prob : reduced
    prepared_plan = _prepared_data === nothing ? nothing :
                    get(_prepared_data, :execution_plan, nothing)
    reuse_prepared_plan = !report.inconsistent &&
                          prepared_plan !== nothing &&
                          get(_prepared_data, :precision_bits, 0) ==
                              _preprocess_precision_bits(T)
    plan::ExecutionPlan = if reuse_prepared_plan
        prepared_plan::ExecutionPlan
    else
        structural_analysis_started = time_ns()
        execution_route = resolve_execution_route(
            AutoPlanner(),
            planning_problem,
            opts,
            equality_evidence=equality_map.planning_evidence,
        )
        pipeline_structural_analysis_seconds +=
            (time_ns() - structural_analysis_started) / 1.0e9
        execution_planning_started = time_ns()
        built = build_execution_plan(
            AutoPlanner(),
            planning_problem,
            execution_route;
            chordal_estimate=preprocessed.plan === nothing ?
                             nothing : preprocessed.plan.chordal,
        )
        pipeline_execution_planning_seconds +=
            (time_ns() - execution_planning_started) / 1.0e9
        built
    end
    pipeline_timings = () -> opts.timing ? (
        preprocess=pipeline_preprocess_seconds,
        equality_presolve=pipeline_equality_presolve_seconds,
        structural_analysis=pipeline_structural_analysis_seconds,
        execution_planning=pipeline_execution_planning_seconds,
        certification=pipeline_certification_seconds,
    ) : NamedTuple()
    warnings = String[]
    if opts.threads > Base.Threads.nthreads()
        push!(
            warnings,
            "Requested $(opts.threads) threads, but Julia started with " *
            "$(Base.Threads.nthreads()); the solve uses $(plan.threads).",
        )
    end
    if T === BigFloat && opts.threads > 1
        if plan.schedule === :mixed_arrow_contiguous_blocks
            push!(
                warnings,
                "Native BigFloat kernels remain serial; the guarded " *
                "Float64x4 reduced-arrow panel may use $(plan.threads) " *
                "threads, while BigFloat residual and refinement phases " *
                "preserve strict scalar ownership.",
            )
        elseif plan.schedule === :reduced_arrow_contiguous_blocks
            push!(
                warnings,
                "Native BigFloat parallelism is restricted to the " *
                "ownership-safe reduced-arrow panel and disjoint Schur " *
                "tiles; residual, direction, and refinement phases remain " *
                "serial.",
            )
        elseif plan.schedule === :owned_bigfloat_equality_tiles
            push!(
                warnings,
                "Native BigFloat uses exclusive local-block ownership and " *
                "disjoint equality Gram/GEMV tiles across " *
                "$(plan.threads) threads. Other scalar phases retain their " *
                "ownership-safe scheduling and may scale less strongly.",
            )
        elseif plan.schedule === :lp_bigfloat_panels
            push!(
                warnings,
                "Native BigFloat LP assembly uses exclusive panel-row and " *
                "Schur-tile ownership across $(plan.threads) threads; " *
                "predictor, residual, and refinement phases remain serial.",
            )
        else
            push!(
                warnings,
                "BigFloat solver kernels are serial because the current " *
                "mutable-scalar workspaces require strict ownership and " *
                "aliasing guarantees.",
            )
        end
    end
    if plan.classification.cone === :socp
        if plan.classification.maximum_block_size <= 2
            push!(
                warnings,
                "Detected Lorentz-compatible 2x2 structure. SDPX is using " *
                "the semidefinite primal-dual route with the exact " *
                "Q3-to-S_+^2 structural mapping and specialized scalar " *
                "2x2 kernels.",
            )
        else
            push!(
                warnings,
                "Detected exact SOC-arrow PSD structure. SDPX is using the " *
                "semidefinite primal-dual route; the NativeSOC backend is " *
                "available only through the public Model/ConicProblem API.",
            )
        end
    end
    _validate_warm_start(
        prob;
        x0=x0,
        X0=X0,
        y0=y0,
        Y0=Y0,
        resume=resume,
        accepted_y_lengths=(
            preprocessed.reconstruction.original_equalities,
            equality_map.original_count,
            length(equality_map.keep),
        ),
    )
    if time() >= deadline
        return _time_limit_pipeline_result(
            prob,
            report,
            plan,
            time() - pipeline_started,
            warnings,
            opts.diagnostics,
            opts.max_time,
            opts.certification,
            pipeline_timings(),
            ladder_context,
        )
    end
    report.inconsistent &&
        return _inconsistent_presolve_result(
            prob,
            report,
            plan,
            opts,
            pipeline_timings(),
            ladder_context,
        )

    preprocessed_warm_start = _transform_preprocess_warm_start(
        preprocessed.reconstruction;
        x0=x0,
        X0=X0,
        y0=y0,
        Y0=Y0,
    )

    # The dedicated LP route stores each scalar inequality dual as a 1×1 PSD
    # block at the generic warm-start boundary.  Preserve the preprocessing
    # block map above, then unwrap those owned scalar blocks immediately before
    # calling solve_lp!, whose native z0 entry uses row-vector coordinates.
    lp_preprocessed_z0 = if plan.algorithm !== :lp_primal_dual ||
                            preprocessed_warm_start.Y0 === nothing
        nothing
    else
        blocks = preprocessed_warm_start.Y0
        blocks isa Union{AbstractVector,Tuple} || throw(ArgumentError(
            "LP Y0 warm start must be a vector or tuple of 1×1 blocks",
        ))
        values = Vector{T}(undef, length(blocks))
        @inbounds for index in eachindex(values)
            block = blocks[index]
            block isa AbstractMatrix && size(block) == (1, 1) || throw(
                ArgumentError(
                    "LP Y0 warm start block $index must be a 1×1 matrix",
                ),
            )
            values[index] = _owned_array_copy(T, block)[1, 1]
        end
        values
    end

    reduced_y0 = if preprocessed_warm_start.y0 === nothing
        nothing
    else
        supplied = preprocessed_warm_start.y0
        if length(supplied) == equality_map.original_count
            reduced_y = alloc_zeros(T, length(equality_map.keep))
            kmul_owned!(
                reduced_y,
                equality_map.multiplier_map,
                supplied,
            )
        elseif length(supplied) == length(equality_map.keep)
            supplied
        else
            throw(ArgumentError(
                "y0 has length $(length(supplied)); expected " *
                "$(equality_map.original_count) in original equality coordinates " *
                "or $(length(equality_map.keep)) after presolve",
            ))
        end
    end

    if opts.verbosity >= 2
        println(
            "SDPX execution plan: cone=$(plan.classification.cone), " *
            "solver=$(plan.algorithm), scaling=$(plan.scaling), " *
            "kkt=$(plan.kkt_backend), " *
            "kernel=$(plan.gram_kernel), scheduling=$(plan.schedule), " *
            "threads=$(plan.threads)",
        )
        report.removed_dependent_equalities > 0 && println(
            "SDPX presolve: removed $(report.removed_dependent_equalities) " *
            "dependent equality constraints.",
        )
    end

    result = nothing
    redundant_rows = 0
    workspace_bytes = 0
    if plan.algorithm === :lp_primal_dual
        isempty(resume) ||
            throw(ArgumentError("resume checkpoints are not supported by the dedicated LP path"))
        # Public LP cone-dual starts arrive as scalar 1×1 Y0 blocks.  The
        # generic preprocessing stage has already mapped and reduced those
        # blocks above; only the resulting scalar vector is passed to the
        # dedicated core below.
        # The same precision hygiene the SDP core performs (this file,
        # `_solve_sdp_core!`): warn when BigFloat inputs carry fewer bits than
        # the requested working precision, and normalize stored precision when
        # asked. The dedicated LP path used to bypass both, so a 128-bit-input
        # LP inside a 256-bit solve proceeded without a word. (Review P2.7.)
        if T === BigFloat
            check_precision_consistency(reduced, opts.precision_bits, opts.verbosity)
            opts.convert_inputs && (reduced = reround(reduced, opts.precision_bits))
        end
        # Neutral plan: the plan parameters are user hints only. The LP
        # resolver runs once inside `solve_lp!` after `_scale_lp!`; the core
        # must not substitute plan values into the options.
        lp_options = opts
        result, redundant_rows, workspace_bytes = solve_lp!(
            reduced,
            lp_options,
            plan;
            x0=preprocessed_warm_start.x0,
            y0=reduced_y0,
            z0=lp_preprocessed_z0,
            deadline=deadline,
        )
    else
        # Pre-flight against the memory actually available. Nothing compared
        # the workspace size against anything before this, so a model too large
        # for the machine simply ran until an allocation failed, with no
        # indication of which dimension caused it.
        #
        # Deliberately *not* checked against `plan.memory_budget_bytes`: that
        # field is `available × extended_precision_memory_fraction`, a budget
        # for extended-precision buffers rather than for the whole workspace.
        # The workspace floor of a large block-arrow solve can exceed that 10%
        # budget comfortably while still running fine, so comparing the two
        # would warn on a workload that works.
        #
        # Uses the O(1) floor rather than the full estimate, which walks every
        # coefficient and would cost more than the solve it precedes.
        #
        # The estimate must match the KKT route the plan actually chose. The
        # dense floor is an `m x m` figure; the block-arrow route never forms
        # that matrix, and applying the dense model to it can overstate the
        # requirement by three orders of magnitude for a solve that fits in
        # memory. A warning wrong by three orders of magnitude drives users off
        # runs that fit, so each route is estimated on its own terms and a
        # route with no estimate stays silent.
        let route = plan.kkt_backend,
            floor_bytes = route === :block_arrow ?
                          arrow_workspace_floor_bytes(T, reduced, plan.threads) :
                          route === :sparse_schur_cholesky ?
                          estimate_sdp_workspace_bytes(
                              reduced,
                              plan.threads,
                          ) :
                          route === :dense_augmented_ldlt ?
                          dense_augmented_workspace_floor_bytes(
                              T,
                              reduced.dims.m,
                              reduced.dims.n,
                              reduced.dims.L,
                              plan.threads,
                          ) : dense_workspace_floor_bytes(
                              T,
                              reduced.dims.m,
                              reduced.dims.n,
                              reduced.dims.L,
                              plan.threads,
                          ),
            available = _available_memory_bytes()

            if floor_bytes > 0 && available > 0 && floor_bytes > available
                push!(
                    warnings,
                    "The $(
                        route === :block_arrow ? "block-arrow" :
                        route === :sparse_schur_cholesky ?
                        "sparse Schur" :
                        "dense"
                    ) " *
                    "workspace needs at least " *
                    "$(round(floor_bytes / 2^30; digits=2)) GiB but only " *
                    "$(round(available / 2^30; digits=2)) GiB is available; " *
                    "the solve may exhaust memory. Reduce the thread count or " *
                    "the precision, or move to a larger machine.",
                )
            end
        end
        core_options = _replace_solver_options(
            opts;
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            threads=plan.threads,
        )
        result = _solve_sdp_core!(
            reduced,
            core_options;
            x0=preprocessed_warm_start.x0,
            X0=preprocessed_warm_start.X0,
            y0=reduced_y0,
            Y0=preprocessed_warm_start.Y0,
            resume=resume,
            deadline=deadline,
            apply_equilibration=plan.scaling === :sdp_ruiz,
            execution_plan=plan,
        )
        # Keep diagnostics out of the hot path: recursively traversing every
        # sparse coefficient object can cost much more than a warmed solve.
        workspace_bytes = plan.kkt_backend === :dense_augmented_ldlt ?
                          estimate_dense_augmented_workspace_bytes(
                              reduced,
                              plan.threads,
                          ) : estimate_sdp_workspace_bytes(
                              reduced,
                              plan.threads,
                          )
    end
    if redundant_rows > 0
        report = PresolveReport(
            report.original_equalities,
            report.reduced_equalities,
            report.removed_dependent_equalities,
            report.removed_zero_equalities,
            redundant_rows,
            report.inconsistent,
            report.equality_keep,
            report.elapsed,
            report.preprocessing,
        )
    end
    result = _restore_equalities(result, equality_map)
    result = reconstruct(
        preprocessed.reconstruction,
        prob,
        result,
    )
    certification_started = time_ns()
    diagnosable_failure = result.status in (
        Stalled,
        IterLimit,
        NumericalBreakdown,
        MaxRestartsExceeded,
        InsufficientPrecision,
        NumericalFailure,
    )
    if opts.mode === OPTIMIZE && diagnosable_failure
        if opts.certification
            result, _, infeasibility_message =
                certify_optimize_infeasibility(prob, result, opts)
            infeasibility_message === nothing ||
                push!(warnings, infeasibility_message)
        end
    end
    result, certificate, certificate_warning = if result.status === TimeLimit
        (
            result,
            opts.certification ?
            (available=false, reason=:time_limit) :
            (available=false, reason=:certification_disabled),
            nothing,
        )
    else
        certify_final_result(prob, result, opts)
    end
    certificate_warning === nothing ||
        push!(warnings, certificate_warning)
    pipeline_certification_seconds +=
        (time_ns() - certification_started) / 1.0e9
    equality_diagnostics = get(
        result.termination,
        :equality_system,
        (available=false,),
    )
    if equality_diagnostics.available &&
       equality_diagnostics.rank_deficient
        push!(
            warnings,
            "The final equality Newton system has numerical rank " *
            "$(equality_diagnostics.rank) of " *
            "$(equality_diagnostics.dimension) under " *
            "$(equality_diagnostics.method). Treat a stopped objective as " *
            "uncertified and reduce the equality basis or use a wider " *
            "arithmetic type.",
        )
    end
    if !(result.status in (
        Optimal,
        FeasibleCert,
        InfeasibleCert,
        PrimalInfeasible,
        DualInfeasible,
    ))
        push!(
            warnings,
            "The reported primal and dual objectives belong to the best " *
            "available iterate; status $(result.status) does not certify " *
            "either value as an optimum or rigorous bound.",
        )
    end
    if any(row -> row.fallback, result.parameter_history)
        push!(
            warnings,
            "Adaptive parameter control detected instability and reverted to the fixed defaults.",
        )
    end
    mixed_kkt = get(
        result.termination,
        :mixed_precision_kkt,
        (available=false,),
    )
    if mixed_kkt.available &&
       (mixed_kkt.fell_back || mixed_kkt.disabled)
        push!(
            warnings,
            "Mixed-precision KKT used the native extended-precision " *
            "fallback (reason=$(mixed_kkt.reason), attempts=" *
            "$(mixed_kkt.factor_attempt_count), dynamic_fallbacks=" *
            "$(mixed_kkt.dynamic_fallback_count), static_rejections=" *
            "$(mixed_kkt.static_rejection_count)).",
        )
    end
    return _attach_diagnostics(
        result,
        plan,
        report,
        time() - pipeline_started,
        warnings,
        workspace_bytes,
        opts.diagnostics,
        (reason=:none,),
        certificate,
        pipeline_timings(),
        ladder_context,
    )
end

"""
    solve(prob; tolerance=1e-8, maximum_iterations=200, time_limit=Inf,
          threads=Threads.nthreads(), precision=nothing, verbosity=1,
          diagnostics=true, warm_start=nothing, kwargs...)

Simplified public interface. Low-level interior-point values remain available
through `SolverOptions` for expert use; typical callers only need these common
controls.
"""
function solve(
    prob::SDPProblem{T};
    tolerance::Real=1e-8,
    maximum_iterations::Int=200,
    time_limit::Real=Inf,
    threads::Int=Base.Threads.nthreads(),
    precision::Union{Nothing,Integer}=nothing,
    verbosity::Int=1,
    diagnostics::Bool=true,
    timing::Bool=true,
    warm_start=nothing,
    presolve::Union{Bool,Symbol}=:auto,
    presolve_bounds::Bool=true,
    presolve_fixed_variables::Bool=true,
    presolve_zero_constraints::Bool=true,
    presolve_duplicate_constraints::Bool=true,
    presolve_dependent_equalities::Bool=true,
    scaling::Symbol=:auto,
    formulation::Symbol=:auto,
    algorithm::Symbol=:auto,
    parameter_strategy::Symbol=:adaptive,
    working_precision_policy::Symbol=:auto,
    minimum_working_precision_bits::Int=192,
) where {T}
    if T === BigFloat &&
       precision !== nothing &&
       Base.precision(BigFloat) != Int(precision)
        requested_precision = Int(precision)
        requested_precision > 0 ||
            throw(ArgumentError("precision must be a positive bit count"))
        return setprecision(BigFloat, requested_precision) do
            solve(
                prob;
                tolerance=tolerance,
                maximum_iterations=maximum_iterations,
                time_limit=time_limit,
                threads=threads,
                precision=nothing,
                verbosity=verbosity,
                diagnostics=diagnostics,
                timing=timing,
                warm_start=warm_start,
                presolve=presolve,
                presolve_bounds=presolve_bounds,
                presolve_fixed_variables=presolve_fixed_variables,
                presolve_zero_constraints=presolve_zero_constraints,
                presolve_duplicate_constraints=
                    presolve_duplicate_constraints,
                presolve_dependent_equalities=
                    presolve_dependent_equalities,
                scaling=scaling,
                formulation=formulation,
                algorithm=algorithm,
                parameter_strategy=parameter_strategy,
                working_precision_policy=working_precision_policy,
                minimum_working_precision_bits=
                    minimum_working_precision_bits,
            )
        end
    end
    frontend_started = time_ns()
    precision_bits = precision === nothing ?
                     (T === BigFloat ? Base.precision(BigFloat) : sig_bits(T)) :
                     Int(precision)
    tolerance_t = T(tolerance)
    options = SolverOptions{T}(
        ϵ_gap=tolerance_t,
        ϵ_primal=tolerance_t,
        ϵ_dual=tolerance_t,
        iter_max=maximum_iterations,
        max_time=Float64(time_limit),
        threads=threads,
        precision_bits=precision_bits,
        verbosity=verbosity,
        diagnostics=diagnostics,
        timing=timing,
        presolve=presolve,
        presolve_bounds=presolve_bounds,
        presolve_fixed_variables=presolve_fixed_variables,
        presolve_zero_constraints=presolve_zero_constraints,
        presolve_duplicate_constraints=presolve_duplicate_constraints,
        presolve_dependent_equalities=presolve_dependent_equalities,
        scaling=scaling,
        formulation=formulation,
        algorithm=algorithm,
        parameter_strategy=parameter_strategy,
        working_precision_policy=working_precision_policy,
        minimum_working_precision_bits=minimum_working_precision_bits,
        parameter_policy=:auto,
    )
    frontend_seconds = (time_ns() - frontend_started) / 1.0e9
    result = if warm_start === nothing
        solve!(prob, options)
    else
        warm_start isa NamedTuple ||
            throw(ArgumentError("warm_start must be a NamedTuple such as (; x0, X0, y0, Y0)"))
        solve!(prob, options; warm_start...)
    end
    return _with_frontend_timing(result, frontend_seconds, timing)
end

function _resolve_precision_type(precision, c, A, C, B, b)
    precision === nothing && return infer_eltype(c, A, C, B, b)
    precision === :float64 && return Float64
    precision === :bigfloat && return BigFloat
    precision isa Type && return _require_supported_arithmetic_type(precision)
    throw(ArgumentError(
        "precision must be nothing, :float64, :bigfloat, or a supported MultiFloats type",
    ))
end

"""
    solve(c, A, C, B, b; precision=nothing, tolerance=1e-8, ...)

One-call interface that ingests and solves a model. `precision` may be
`:float64`, `:bigfloat`, or a concrete extended type such as `Float64x4`.
Use `solve!(problem, SolverOptions(...))` for expert interior-point controls.
"""
function solve(
    c,
    A,
    C,
    B,
    b;
    tolerance::Real=1e-8,
    maximum_iterations::Int=200,
    time_limit::Real=Inf,
    threads::Int=Base.Threads.nthreads(),
    precision=nothing,
    precision_bits::Int=256,
    verbosity::Int=1,
    diagnostics::Bool=true,
    timing::Bool=true,
    warm_start=nothing,
    presolve::Union{Bool,Symbol}=:auto,
    presolve_bounds::Bool=true,
    presolve_fixed_variables::Bool=true,
    presolve_zero_constraints::Bool=true,
    presolve_duplicate_constraints::Bool=true,
    presolve_dependent_equalities::Bool=true,
    scaling::Symbol=:auto,
    formulation::Symbol=:auto,
    algorithm::Symbol=:auto,
    parameter_strategy::Symbol=:adaptive,
    working_precision_policy::Symbol=:auto,
    minimum_working_precision_bits::Int=192,
    sparse::Union{Bool,Symbol}=:auto,
)
    T = _resolve_precision_type(precision, c, A, C, B, b)
    run = function ()
        api_started = time()
        problem = ingest(
            c,
            A,
            C,
            B,
            b;
            T=T,
            sparse=sparse,
            verbosity=verbosity,
        )
        remaining_time = isfinite(time_limit) ?
                         max(
                             0.0,
                             Float64(time_limit) - (time() - api_started),
                         ) :
                         Inf
        frontend_seconds = time() - api_started
        result = solve(
            problem;
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            time_limit=remaining_time,
            threads=threads,
            precision=precision_bits,
            verbosity=verbosity,
            diagnostics=diagnostics,
            timing=timing,
            warm_start=warm_start,
            presolve=presolve,
            presolve_bounds=presolve_bounds,
            presolve_fixed_variables=presolve_fixed_variables,
            presolve_zero_constraints=presolve_zero_constraints,
            presolve_duplicate_constraints=
                presolve_duplicate_constraints,
            presolve_dependent_equalities=
                presolve_dependent_equalities,
            scaling=scaling,
            formulation=formulation,
            algorithm=algorithm,
            parameter_strategy=parameter_strategy,
            working_precision_policy=working_precision_policy,
            minimum_working_precision_bits=
                minimum_working_precision_bits,
        )
        return _with_frontend_timing(result, frontend_seconds, timing)
    end
    return T === BigFloat ? setprecision(run, BigFloat, precision_bits) : run()
end
