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
    lp_initial_scale_indicator(prob) -> T

Estimate how far the origin is from the LP constraint hyperplanes, using

`max_i |h_i| / ‖G_i‖∞` and `max_j |b_j| / ‖B_j‖∞`.

Unlike a raw right-hand-side norm, this diagnostic is invariant to positive
rescaling of an individual constraint. It is used only by the zero-probe
parameter selector: aggressive LP centering is reliable when the feasible
set is reasonably close to the origin, while a conservative profile is safer
for very distant Float64 starts. A nonzero right-hand side with a zero
coefficient row returns `Inf`; presolve subsequently reports the underlying
inconsistency.
"""
function lp_initial_scale_indicator(prob::SDPProblem{T}) where {T}
    all(==(1), prob.dims.k) ||
        throw(ArgumentError(
            "lp_initial_scale_indicator requires a pure 1x1-cone LP",
        ))
    largest = zero(T)
    if prob.cons isa DenseCons{T}
        panels = (prob.cons::DenseCons{T}).Av
        @inbounds for block in eachindex(panels)
            coefficient = maximum(abs, panels[block]; init=zero(T))
            rhs = abs(prob.C[block][1, 1])
            ratio = coefficient > zero(T) ?
                    rhs / coefficient :
                    (rhs > zero(T) ? T(Inf) : zero(T))
            largest = max(largest, ratio)
        end
    else
        cons = prob.cons::SparseCons{T}
        @inbounds for block in eachindex(cons.Asp)
            coefficient = zero(T)
            for variable in cons.active[block]
                coefficient = max(
                    coefficient,
                    abs(cons.Asp[block][variable][1, 1]),
                )
            end
            rhs = abs(prob.C[block][1, 1])
            ratio = coefficient > zero(T) ?
                    rhs / coefficient :
                    (rhs > zero(T) ? T(Inf) : zero(T))
            largest = max(largest, ratio)
        end
    end
    @inbounds for equality in axes(prob.B, 2)
        coefficient = maximum(
            abs,
            view(prob.B, :, equality);
            init=zero(T),
        )
        rhs = abs(prob.b[equality])
        ratio = coefficient > zero(T) ?
                rhs / coefficient :
                (rhs > zero(T) ? T(Inf) : zero(T))
        largest = max(largest, ratio)
    end
    return largest
end

# The aggressive LP profile was robust below this row-scale-invariant distance
# in the deterministic validation suite. Every arithmetic type falls back
# above it: a wider exponent range prevents overflow, but does not remove the
# globalization difficulty of a very distant infeasible start.
const LP_AGGRESSIVE_START_SCALE_LIMIT = 1_000

@inline function _large_lattice_dense_schur_profile(
    variables::Int,
    equalities::Int,
    blocks::Int,
    coefficient_density::Float64,
    schur_density::Float64,
)
    return variables >= 4_000 &&
           equalities >= 100 &&
           blocks >= 16 &&
           coefficient_density <= 0.005 &&
           schur_density >= 0.75
end

"""
    recommended_adaptive_sigma_max(profile, beta, requested)

Return the expert override when it is positive, otherwise select the guarded
automatic centering cap for a structural parameter profile. The large dense
lattice profile has a separately validated low-beta trajectory; allowing the
generic controller to jump from `0.075` to `0.5` caused 27 backtracking trials
and an additional Task_Low08 iteration. A controlled same-node sweep selected
`0.2`: it retained the 28-iteration trajectory, minimized backtracking among
the accurate capped runs, and improved the PSD certificate. Retain the generic
`0.5` cap elsewhere.
"""
@inline function recommended_adaptive_sigma_max(
    profile::Symbol,
    beta::T,
    requested::T,
) where {T}
    requested > zero(T) && return max(requested, beta)
    return profile === :large_lattice_dense_schur ?
           max(T(1) / T(5), beta) :
           max(T(1) / T(2), beta)
end

"""
    recommended_parameters(prob, opts) -> NamedTuple

Choose a zero-probe parameter profile from problem structure and requested
arithmetic. The current profiles are calibrated for sparse block-arrow SDPs
with many `2x2` blocks. General problems retain the supplied parameters.
"""
function recommended_parameters(
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    cons = prob.cons
    if all(==(1), prob.dims.k)
        if opts.algorithm === :sdp
            return (
                β=opts.β,
                γ=opts.γ,
                Ωp=opts.Ωp,
                Ωd=opts.Ωd,
                predictor=opts.predictor,
                parameter_strategy=opts.parameter_strategy,
                profile=:lp_general_conic,
            )
        end
        initial_scale = lp_initial_scale_indicator(prob)
        aggressive =
            isfinite(initial_scale) &&
            initial_scale <= T(LP_AGGRESSIVE_START_SCALE_LIMIT)
        return (
            β=aggressive ? T(1) / T(50) : opts.β,
            γ=aggressive ? T(99) / T(100) : opts.γ,
            Ωp=opts.Ωp,
            Ωd=opts.Ωd,
            predictor=opts.predictor,
            parameter_strategy=opts.parameter_strategy,
            profile=aggressive ?
                    :lp_mehrotra_fast_start :
                    :lp_mehrotra_conservative_start,
        )
    end
    if prob.structure.profile ===
       :sparse_coefficients_dense_psd_dense_schur &&
        prob.dims.m >= 1_000 &&
       prob.dims.n > 0
        # Task_Low08-like systems occupy a narrower regime than the general
        # large-equality profile: thousands of variables, several dense PSD
        # blocks, extremely sparse coefficients, and an almost-dense Schur
        # complement. A 2026-07 cluster sweep around the old (0.1, 0.85)
        # profile found (0.075, 0.8) to be the only nearby setting that was
        # both robust and faster: 24 versus 27 iterations, with a valid
        # 4.27e-7 relative gap at a 1e-6 request. Nearby beta/gamma pairs that
        # stalled are deliberately excluded by the structural gate rather
        # than generalized to every equality-constrained SDP.
        lattice_like = _large_lattice_dense_schur_profile(
            prob.dims.m,
            prob.dims.n,
            prob.dims.L,
            prob.structure.coefficient_density,
            prob.structure.schur_density,
        )
        return (
            β=lattice_like ? T(3) / T(40) : T(1) / T(10),
            γ=lattice_like ? T(4) / T(5) : T(17) / T(20),
            Ωp=T(100),
            Ωd=T(1) / T(1_000),
            predictor=:sdpb,
            parameter_strategy=opts.parameter_strategy,
            profile=lattice_like ?
                    :large_lattice_dense_schur :
                    :large_equality_dense_schur,
        )
    end
    if cons isa SparseCons{T} && prob.dims.n == 0 && all(<=(2), prob.dims.k)
        max_active = maximum(length, cons.active; init=0)
        beta, gamma, profile = if max_active <= 6
            (T(1) / T(10), T(17) / T(20), :small_arrow_2x2)
        elseif max_active <= 14
            (T(1) / T(10), T(4) / T(5), :medium_arrow_2x2)
        elseif max_active <= WIDE_ARROW_ACTIVE_LIMIT
            # The medium J=32/K=4 CSDR dual has 144 shared variables plus
            # one local variable per block. The old catch-all "large" profile
            # selected β=0.01 and stalled; β=0.1 converged reliably across the
            # Ω/γ sweep. Keep the genuinely large 385-active-variable case on
            # its separately validated low-β profile.
            (T(1) / T(10), T(17) / T(20), :wide_arrow_2x2)
        else
            # Beyond the separately calibrated wide-arrow range. The CSDR
            # 80/4/40/100 model has 385 active variables per block, and the
            # old `(0.4, 0.7)` setting did not converge on it at all; a sweep
            # found `(0.01, 0.85)` reaching the correct basin.
            (T(1) / T(100), T(17) / T(20), :large_arrow_2x2)
        end
        if T === BigFloat &&
           opts.ϵ_gap < T(1) / T(10_000_000_000)
            beta = T(1) / T(10)
            gamma = min(gamma, T(3) / T(4))
            profile = :high_accuracy_bigfloat_2x2
        end
        # Scale the initial point to the data instead of pinning it at 10.
        # `X = Ω·I` has to be commensurate with `C`, since the initial primal
        # residual is `‖Ω·I − C‖`: on the CSDR model `max‖C_l‖∞ = 116.6`, so
        # Ω=10 starts far outside the region where the Newton step is usable —
        # measured step sizes collapse to 1e-3 and the primal objective runs
        # away to 1e13. A sweep confirmed Ω=100 recovers the correct basin on
        # exactly that instance, and Ω tracking `max‖C_l‖∞` reproduces it
        # without hard-coding the case.
        # An earlier revision set this multiplier to 1, reasoning that Ω=100
        # worked on the CSDR instance whose `max‖C_l‖∞` happened to be 116.6,
        # so "Ω tracks `max‖C_l‖∞`" reproduced it without hard-coding. That was
        # a coincidence of one instance. A CSDR model with `max‖C_l‖∞ = 35.4`
        # gets Ω=35 from the same rule and does not converge.
        #
        # Swept across three CSDR instances with independently known Clarabel
        # optima and the dense lattice benchmark, iterations and status:
        #
        #   multiplier   s15         s20              s25         Task_Low08
        #            1   22 Optimal  94 Stalled       81 Optimal  27 Optimal
        #           10   26 Optimal  36 Optimal       46 Optimal  27 Optimal
        #          100   29 Optimal  81 Stalled       53 Optimal  27 Optimal
        #
        # Ten is the only value that solves all four. Note the behaviour is not
        # monotone — 100 is worse than 10 on `s20` — so this cannot be inferred
        # from a single instance in either direction, which is how the previous
        # value was arrived at. The dense lattice result is unchanged to every
        # digit, so this is not a trade against it. Keep 10 as the floor for
        # small-data models.
        stats = block_norm_stats(prob)
        wide_small_data =
            profile === :wide_arrow_2x2 &&
            stats.maxnorm <= T(WIDE_ARROW_SMALL_DATA_NORM_LIMIT)
        omega = if wide_small_data
            # The response is sharply non-monotone on the medium canonical
            # model: Ω=25 and Ω=30 converge, while the unrounded 5*maxnorm
            # value Ω≈27.56 stalls. The lower grid point Ω=25 needs 41
            # iterations versus 46 at Ω=30, so quantize down to the faster
            # validated point while retaining the floor at 10.
            max(
                T(10),
                T(WIDE_ARROW_OMEGA_MULTIPLIER) * floor(stats.maxnorm),
            )
        else
            max(T(10), T(OMEGA_DATA_MULTIPLIER) * stats.maxnorm)
        end
        return (
            β=beta,
            γ=gamma,
            Ωp=omega,
            Ωd=omega,
            predictor=opts.predictor,
            # Deliberately NOT :adaptive. Enabling the β/γ controller here did
            # look like a large win, but that measurement was taken while the
            # solve was terminating prematurely; re-running it against a correct
            # Ω and correct termination reverses the result — 47 iterations to
            # gap 3.08e-04 with the fixed parameters, against 33 iterations to
            # 6.08e-04 with the controller. It also costs accuracy on the dense
            # lattice benchmark (gap 6.29e-07 → 5.07e-04).
            parameter_strategy=opts.parameter_strategy,
            profile,
        )
    end
    # Scale the initial point to the data. `X = Ω·I` has to be commensurate
    # with `C`, because the initial dual residual is `‖C_l − Ω·I‖`; a fixed
    # Ω=1 against data of magnitude 1e7 leaves the solve stranded at its
    # starting residual. Measured on the badly-scaled benchmark generator
    # (`max‖C_l‖∞ = 1.6e7`): Ω=1 stalls at iteration 15 with `gap_rel = 2.0`
    # and `dObj = -1.5e12`, while Ω = max‖C_l‖∞ converges to `Optimal` in 19
    # iterations at `gap_rel = 1.9e-11` with a valid certificate. Equilibration
    # does not substitute for it — it was measured and left the solve stalled.
    #
    # Taking the max with `opts.Ωp` makes this a no-op for data already at unit
    # scale (where `max‖C_l‖∞ ≈ 1`), so well-scaled models keep their previous
    # behaviour and a user-supplied larger Ω is still honoured.
    stats = block_norm_stats(prob)
    return (
        β=opts.β,
        γ=opts.γ,
        Ωp=max(opts.Ωp, stats.maxnorm),
        Ωd=max(opts.Ωd, stats.maxnorm),
        predictor=opts.predictor,
        parameter_strategy=opts.parameter_strategy,
        profile=:general_adaptive,
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
    elseif factor isa BigFloatCholeskyFactor
        return (
            available=true,
            method=:normal_equations,
            rank=equality_count,
            dimension=equality_count,
            rank_deficient=false,
            quality=_cholesky_diagonal_quality(factor.L),
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
) where {T}
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
        (reason=:time_limit, stage=:sdp_setup),
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
    time() >= deadline &&
        return _sdp_setup_time_limit_result(
            prob,
            time() - core_started,
        )
    validation_finished_ns = time_ns()

    if T === BigFloat
        check_precision_consistency(prob, opts.precision_bits, opts.verbosity)
        opts.convert_inputs && (prob = reround(prob, opts.precision_bits))
    end
    precision_preparation_finished_ns = time_ns()

    eq = nothing
    solve_prob = prob
    if opts.equilibrate
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
        # accurate enough (J200/K2: 1e-10 internally became 7.7e-9 on return).
        #
        # The ratio between original and scaled relative gaps is never larger
        # than max(1, objective_scale), including nonzero large objectives.
        # Tighten only the acceptance threshold. Feeding the stricter value
        # into stagnation and adaptive-control heuristics can make a flat early
        # gap dominate their progress merit and stop a solve that later
        # recovers (observed on J200/K2 at iteration 18). The controller should
        # continue to interpret the accuracy requested by the user; only a
        # prospective success must satisfy the conservative scaled threshold.
        termination_gap_tolerance =
            opts.ϵ_gap / eq.objective_scale
    end

    # Parameter selection must see the problem that will actually be solved.
    # Equilibration changes the data scale by orders of magnitude, and the
    # initial point `X = Ω·I` is chosen *from* that scale — picking Ω on the
    # original problem and then solving the equilibrated one applies an Ω that
    # can be wrong by the whole equilibration factor. Observed on the
    # badly-scaled benchmark generator: Ω selected from `max‖C_l‖∞ = 1.6e7` and
    # applied to an equilibrated problem of unit scale drove the primal residual
    # to 8.7e+15, where the same solve without equilibration converged.
    if opts.parameter_policy === :auto
        selected = recommended_parameters(solve_prob, opts)
        adaptive_sigma_max = selected.parameter_strategy === :adaptive ?
                             recommended_adaptive_sigma_max(
                                 selected.profile,
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
    end
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
        )

    local x::Vector{T}, y::Vector{T}, X::Vector{Matrix{T}}, Y::Vector{Matrix{T}}, μ::Vector{T}
    local iter::Int, restarts::Int

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
                return SDPResult{T}(NumericalBreakdown, "initial X0/Y0 must be positive definite",
                    x, X, y, Y, zero(T), zero(T), T(Inf), T(Inf), T(Inf), 0, 0, 0, nothing)
        end
        μ = [opts.β * kdot(X[l], Y[l]) / k[l] for l in 1:L]
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

    # Best-iterate retention. An interior-point run can reach a good point and
    # then wander away from it: on the `Task_Low08` lattice benchmark at
    # `ϵ=1e-8` the primal residual reaches 1.4e-12 around iteration 55 while the
    # dual residual diverges to ~1.8, after which the restart rule rescales the
    # collapsed side and the iterate is lost. Reporting the *last* point in that
    # situation returns a worse answer than the solver actually found, and the
    # objective from such a run is not meaningful.
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
        # because feasibility is already inside tolerance. On large CSDR
        # models feasibility can arrive dozens of iterations before the gap;
        # the old `if term_ok` therefore rebuilt every residual on each of
        # those iterations even though an optimal certificate was impossible.
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
                    break
                elseif dObj >= zero(T)
                    status, message = InfeasibleCert, "Infeasible"
                    break
                end
            end
        end
        if iter >= opts.iter_max
            status, message = IterLimit, "Cannot reach optimality (feasibility) within $(opts.iter_max) iterations."
            break
        end
        # Precision-exhaustion stop. Once the scaled merit has not improved for
        # `stall_iterations` consecutive iterations the working precision is
        # spent: further iterations do not converge, and the restart rule can
        # actively destroy the best iterate found (observed on `Task_Low08` at
        # ϵ=1e-8, where the best point is reached near iteration 30 and the run
        # then degrades through iteration 55 plus restarts). Reporting `Stalled`
        # with the retained best iterate is both faster and more honest than
        # grinding to `IterLimit`/`MaxRestartsExceeded`.
        if stagnated
            # Distinguish "the arithmetic ran out" from "the algorithm stopped
            # making progress". Only the first is fixed by a wider type, and
            # saying so is the difference between an actionable result and a
            # bare `Stalled`.
            status = stagnation.reason === :precision_floor ? InsufficientPrecision : Stalled
            message = stagnation_message(stagnation, opts.ϵ_gap)
            break
        end
        if time() >= deadline || time() - t_start > opts.max_time
            status, message = TimeLimit, "Time limit ($(opts.max_time)s) exceeded after $iter iterations."
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
        # perfectly healthy solves: on the CSDR sparse model the primal reaches
        # `p_res ≈ 1e-47` by iteration 2 and stays there, `tX` duly collapses,
        # and the run stopped at iteration 27 with the *dual* gap still at
        # 9e-4 — the same place whether or not stall detection was enabled.
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
            # it — on `Task_Low08` at ϵ=1e-8 the best point (scaled merit 2.2e-7)
            # is reached near iteration 30 and then thrown away by exactly this
            # escalation, ending in `MaxRestartsExceeded` after 78 s. The plain
            # stall counter never fires there because a restart does not
            # increment `iter`, so the restart budget is spent first.
            #
            # So: if the merit has improved by orders of magnitude since the
            # start, treat a collapsed step as precision exhaustion and stop
            # with the retained best iterate.
            # Two conditions, not one. Requiring only "improved a lot since the
            # start" is a false positive on badly *scaled* problems: the CSDR
            # sparse model improves its merit by ~1000x in the first few
            # iterations purely by shrinking huge initial residuals, and would
            # then be declared stalled at iteration 16 while still converging.
            # Precision exhaustion means the iterate is genuinely *near* a
            # solution, so also require the merit to be small in absolute terms.
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
                break
            end
            # Before either giving up or rescaling: try *recentering*.
            #
            # A step can collapse for a reason neither branch below addresses.
            # On the CSDR sparse model the primal sits at `p_res ≈ 1e-47`
            # (exactly feasible), the KKT residual is ~1e-48 against
            # `eps(Float64x4) = 2.4e-63` (so the direction is accurate and the
            # precision is nowhere near exhausted), and yet the step falls under
            # `min_step = 1e-10` while the duality gap is still ~1e-3. What has
            # happened is that the iterate has run into the boundary of the PSD
            # cone far from the optimum — too little centering, not bad scaling
            # and not lost precision.
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
            # scaled, and those are different questions: on the CSDR sparse
            # model the primal residual sits at 1e-48 (exactly feasible) from
            # iteration 2 onward while the dual residual crawls down, and both
            # steps then collapse together. Rescaling X by 1e5 there destroys a
            # perfectly good primal iterate — the observed trace walks p_res
            # from 1e-48 up through 1e+8, 1e+13, 1e+18, 1e+23 over five
            # restarts, converging on nothing.
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
                break
            else
                status = restarts >= opts.max_restarts ? MaxRestartsExceeded : NumericalBreakdown
                message = restarts >= opts.max_restarts ?
                           "Step size collapsed after using up max_restarts=$(opts.max_restarts) rescue attempts." :
                           "Step size collapsed below min_step and restart=false."
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
            " — rescale (equilibrate=true), loosen Ωp/Ωd/omega_step, or use a wider-range T"
            break
        end

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
            break
        end

        pObj = LinearAlgebra.dot(solve_prob.c, x)
        dObj = threaded_dual_objective(ws, solve_prob, y, Y)
        if !isfinite(pObj) || !isfinite(dObj)
            status, message = NumericalBreakdown,
            "non-finite primal or dual objective detected"
            break
        end
        phase_objective_and_targets +=
            (time_ns() - objective_and_targets_started) / 1.0e9
        iter += 1

        # Newton feasibility equations are affine:
        #   P⁺ = (1-tX)P and d⁺ = (1-tY)d.
        # Carry the residual norms to the accepted iterate without another full
        # contraction. A direct recomputation still certifies any prospective
        # `Optimal` status and the final returned point.
        p_res *= abs(one(T) - tX)
        d_res *= abs(one(T) - tY)
        t2 = time()
        print_iter(opts, iter, pObj, dObj, pObj - dObj, p_res, d_res, tX, tY, t2 - t1)

        if opts.checkpoint_every > 0 && !isempty(opts.checkpoint_path) && iter % opts.checkpoint_every == 0
            save_checkpoint(opts.checkpoint_path, T, x, X, y, Y, μ, iter, restarts, solve_prob.dims)
        end
        opts.force_gc && _release_iteration_memory!()
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
            reason=stagnation.reason,
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
                let sparse_workspace =
                        ws.sparse_kkt::SparseSchurSDPWorkspace,
                    backend = sparse_workspace.backend
                    backend === nothing ||
                    backend.factorization === nothing ?
                    (available=false,) :
                    merge(
                            statistics(backend),
                            (
                                available=true,
                                factor_nonzeros=
                                    nnz(backend.factorization),
                                regularization=
                                    sparse_workspace.regularization,
                                equality_requires_pivoting=
                                    sparse_workspace.equality_requires_pivoting,
                            ),
                        )
                end,
            equality_system=equality_diagnostics,
            executed=(
                solver=:sdp,
                kkt=ws.executed_backend,
                planned_backend=planned_backend_name(ws),
                executed_backend=ws.executed_backend,
                fallback_reason=ws.backend_fallback_reason,
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
            ),
        ),
    )
end

"""
    block_norm_stats(prob) -> (norms, gmean, maxnorm, spread)

Per-block `‖C_l‖∞` plus the summary statistics the initial-point scaling rules
key off. Zero or non-finite norms are replaced by one so a block with no
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

"""Default multiple of `max‖C_l‖∞` used for `X = Ω·I`.

Ten. Chosen by sweeping four problems that each have an independently known
answer — three CSDR instances against Clarabel optima and the dense lattice
benchmark — rather than by fitting one of them; see the sweep table at the use
site in `recommended_parameters`. The separately classified moderate-data
wide-arrow regime uses [`WIDE_ARROW_OMEGA_MULTIPLIER`](@ref) instead.

The two previous values were each fitted to a single instance and each failed
elsewhere. Three was fitted against runs that were terminating prematurely. One
replaced it on the reasoning that Ω=100 worked on a CSDR model whose
`max‖C_l‖∞` was 116.6, so a multiplier of 1 reproduced that value without
hard-coding it — but a CSDR model with `max‖C_l‖∞ = 35.4` then gets Ω=35 and
does not converge.

The response surface is not monotone: 100 is worse than 10 on that same
instance. So no single problem can identify this constant, in either direction,
and changing it needs the whole sweep re-run rather than one benchmark
improved."""
const OMEGA_DATA_MULTIPLIER = 10

"""Largest active set assigned to the separately calibrated wide-arrow start.

This keeps the 145-active-variable medium CSDR model out of the genuinely
large 385-active-variable regime while leaving substantial distance from both
measurements instead of keying on an exact benchmark dimension.
"""
const WIDE_ARROW_ACTIVE_LIMIT = 256

"""Use the moderate-data wide-arrow scale only below this block norm."""
const WIDE_ARROW_SMALL_DATA_NORM_LIMIT = 10

"""Initial-point multiplier for a wide arrow whose block data are below 10.

The fixed J=32/K=4 canonical model converges at Ω=25, 30, 40, and 50, while
Ω=20, the unrounded `5·maxnorm≈27.6`, and the old automatic Ω≈55 stall.
The rule rounds `maxnorm` down before applying this multiplier, selecting the
validated Ω=25 point. It needs 41 iterations versus 46 at Ω=30. The floor at
10 remains in force.
"""
const WIDE_ARROW_OMEGA_MULTIPLIER = 5

"""A tolerance-normalised merit below this counts as "near a solution": within
this factor of the tolerance the user actually asked for."""
const NEAR_SOLUTION_MERIT = 100

"""Each recentering attempt multiplies β by this factor."""
const CENTERING_BETA_STEP = 4

"""Recentering never pushes β past this, which is already heavy centering."""
const CENTERING_BETA_MAX = 0.5

"""
    initial_block_scales(prob, opts) -> Vector{T}

Per-block multipliers for the initial point `X_l = Ωp·s_l·I`, `Y_l = Ωd·s_l·I`.

`:scalar` (and `:auto`) return all ones — the classical single-Ω start.
`:per_block` returns `s_l = ‖C_l‖∞ / geomean(‖C‖∞)`, giving `X_l ≈ ‖C_l‖∞·I`
when paired with `Ω = geomean`.

`:auto` deliberately does **not** select `:per_block`, because it was measured
and is worse. The reasoning that motivated it — the initial dual residual on
block `l` is `‖C_l − Ωd·s_l·I‖`, so a scalar Ωd cannot suit a model whose block
norms vary widely — turns out to be the wrong criterion. What the initial
point actually has to do is dominate the data. The final CSDR sweep selected a
scalar Ω matching the largest block norm, while per-block scaling, which
shrinks small blocks to their own tiny norms, diverged.

It is kept as an explicit option because the diagnosis may still be right for
models whose *solution* scales with the block data, but it is not the default
and should not be enabled without measuring.
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
) where {T}
    return _solve_pipeline!(
        prob,
        opts;
        x0=x0,
        X0=X0,
        y0=y0,
        Y0=Y0,
        resume=resume,
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

@inline _working_precision_retry(status::SolveStatus) =
    status in (
        AlmostOptimal,
        InsufficientPrecision,
        Stalled,
        NumericalBreakdown,
        NumericalFailure,
        MaxRestartsExceeded,
    )

function _record_working_precision!(
    result::SDPResult,
    message::String,
)
    result.diagnostics === nothing ||
        push!(result.diagnostics.warnings, message)
    return result
end

function solve!(
    prob::SDPProblem{BigFloat},
    opts::SolverOptions{BigFloat}=SolverOptions{BigFloat}();
    x0=nothing,
    X0=nothing,
    y0=nothing,
    Y0=nothing,
    resume::AbstractString="",
)
    _validate_solver_options(opts)
    requested_precision = opts.precision_bits
    selected_precision =
        isempty(resume) ?
        adaptive_working_precision_bits(prob, opts) :
        requested_precision
    started = time()

    function run_at_precision(run_options, bits)
        run = () -> _solve_pipeline!(
            prob,
            run_options;
            x0=x0,
            X0=X0,
            y0=y0,
            Y0=Y0,
            resume=resume,
        )
        return Base.precision(BigFloat) == bits ?
               run() :
               setprecision(run, BigFloat, bits)
    end

    if selected_precision == requested_precision
        result = run_at_precision(opts, requested_precision)
        if opts.working_precision_policy === :auto && !isempty(resume)
            _record_working_precision!(
                result,
                "Adaptive working precision was bypassed while resuming a " *
                "checkpoint; the requested $(requested_precision)-bit " *
                "precision was used.",
            )
        end
        return result
    end

    lower_options = setprecision(BigFloat, selected_precision) do
        _reround_solver_options(
            opts,
            selected_precision;
            precision_bits=selected_precision,
            working_precision_policy=:fixed,
        )
    end
    lower_result = run_at_precision(lower_options, selected_precision)
    if _working_precision_success(lower_result.status)
        return _record_working_precision!(
            lower_result,
            "Adaptive working precision selected $(selected_precision) of " *
            "$(requested_precision) requested bits; the result passed " *
            "original-coordinate certification without a retry.",
        )
    end

    elapsed = time() - started
    remaining_time = isfinite(opts.max_time) ?
                     max(0.0, opts.max_time - elapsed) :
                     Inf
    if !_working_precision_retry(lower_result.status) ||
       remaining_time <= 0
        return _record_working_precision!(
            lower_result,
            "Adaptive working precision selected $(selected_precision) of " *
            "$(requested_precision) requested bits, but the run ended with " *
            "$(lower_result.status); the fallback was not eligible or its " *
            "time budget was exhausted.",
        )
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
    fallback_result =
        run_at_precision(fallback_options, requested_precision)
    return _record_working_precision!(
        fallback_result,
        "Adaptive working precision first tried $(selected_precision) bits " *
        "and ended with $(lower_status); SDPX retried at the requested " *
        "$(requested_precision)-bit precision.",
    )
end

function _solve_pipeline!(
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}();
    x0=nothing,
    X0=nothing,
    y0=nothing,
    Y0=nothing,
    resume::AbstractString="",
) where {T}
    _validate_solver_options(opts)
    pipeline_started = time()
    deadline = isfinite(opts.max_time) ?
               pipeline_started + opts.max_time :
               Inf
    preprocessed = preprocess(prob, opts)
    reduced, equality_map, equality_report =
        presolve_equalities(preprocessed.problem, opts)
    report = _merge_presolve_reports(
        preprocessed,
        equality_map,
        equality_report,
        reduced,
    )
    # Finalize the plan against the model that will actually be factorized.
    # In particular, equality presolve can change the selected parameter profile
    # and the diagnostic equality count.
    planning_problem = report.inconsistent ? prob : reduced
    execution_route = resolve_execution_route(
        AutoPlanner(),
        planning_problem,
        opts,
    )
    plan = build_execution_plan(
        AutoPlanner(),
        planning_problem,
        execution_route,
    )
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
        if plan.algorithm === :socp_fixed_trace_q3
            push!(
                warnings,
                "Selected the compact fixed-trace Q3 backend. Cone state, " *
                "local Newton metrics, and boundary steps remain in Lorentz " *
                "coordinates; PSD matrices are materialized only for the " *
                "final compatibility certificate.",
            )
        elseif plan.algorithm === :socp_psd2
            push!(
                warnings,
                "Detected Lorentz-compatible 2x2 structure. SDPX is using " *
                "the exact Q3-to-S_+^2 isomorphism and specialized scalar " *
                "2x2 kernels; the general-dimensional Lorentz NT backend " *
                "remains experimental.",
            )
        elseif plan.classification.maximum_block_size <= 2
            push!(
                warnings,
                "The model is exactly Q3/S_+^2 representable, but " *
                "algorithm=:sdp selected the semidefinite reference path.",
            )
        else
            push!(
                warnings,
                "Detected exact SOC-arrow PSD structure. General-dimensional " *
                "cones still use the semidefinite lift because the compact " *
                "Lorentz NT backend has not passed its promotion gates.",
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
        )
    end
    report.inconsistent &&
        return _inconsistent_presolve_result(
            prob,
            report,
            plan,
            opts,
        )

    preprocessed_warm_start = _transform_preprocess_warm_start(
        preprocessed.reconstruction;
        x0=x0,
        X0=X0,
        y0=y0,
        Y0=Y0,
    )

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
        X0 === nothing && Y0 === nothing ||
            push!(
                warnings,
                "Matrix-valued X0/Y0 warm starts are ignored by the dedicated LP path; " *
                "provide x0/y0.",
            )
        # The same precision hygiene the SDP core performs (this file,
        # `_solve_sdp_core!`): warn when BigFloat inputs carry fewer bits than
        # the requested working precision, and normalize stored precision when
        # asked. The dedicated LP path used to bypass both, so a 128-bit-input
        # LP inside a 256-bit solve proceeded without a word. (Review P2.7.)
        if T === BigFloat
            check_precision_consistency(reduced, opts.precision_bits, opts.verbosity)
            opts.convert_inputs && (reduced = reround(reduced, opts.precision_bits))
        end
        lp_options = opts.parameter_policy === :auto ?
                     _replace_solver_options(
                         opts;
                         β=plan.parameters.beta,
                         γ=plan.parameters.gamma,
                         Ωp=plan.parameters.omega_p,
                         Ωd=plan.parameters.omega_d,
                         predictor=plan.parameters.predictor,
                         parameter_policy=:fixed,
                     ) :
                     opts
        result, redundant_rows, workspace_bytes = solve_lp!(
            reduced,
            lp_options,
            plan;
            x0=preprocessed_warm_start.x0,
            y0=reduced_y0,
            deadline=deadline,
        )
    elseif plan.algorithm === :socp_fixed_trace_q3
        workspace_bytes = estimate_fixed_trace_q3_workspace_bytes(
            T,
            reduced,
            plan.threads;
            q3_gram_strategy=opts.q3_gram_strategy,
        )
        available = _available_memory_bytes()
        if available > 0 && workspace_bytes > available
            push!(
                warnings,
                "The compact Q3 workspace needs approximately " *
                "$(round(workspace_bytes / 2^30; digits=2)) GiB but only " *
                "$(round(available / 2^30; digits=2)) GiB is available.",
            )
        end
        reduced_rejection = _fixed_trace_q3_rejection(reduced)
        unsupported_start =
            preprocessed_warm_start.x0 !== nothing ||
            preprocessed_warm_start.X0 !== nothing ||
            reduced_y0 !== nothing ||
            preprocessed_warm_start.Y0 !== nothing ||
            !isempty(resume)
        q3_options = _replace_solver_options(
            opts;
            algorithm=:socp,
            presolve=false,
            scaling=:none,
            equilibrate=false,
            threads=plan.threads,
        )
        native_result = if reduced_rejection !== :eligible
            push!(
                warnings,
                "Presolve changed the fixed-trace Q3 structure " *
                "(reason=$reduced_rejection); this solve used the exact " *
                "PSD2 fallback.",
            )
            nothing
        elseif unsupported_start
            push!(
                warnings,
                "The first native fixed-trace Q3 implementation does not " *
                "consume matrix warm starts or checkpoints; this solve used " *
                "the exact PSD2 fallback.",
            )
            nothing
        else
            _solve_fixed_trace_q3_core!(
                reduced,
                q3_options;
                deadline=deadline,
            )
        end
        native_certificate = if native_result !== nothing &&
                                native_result.status === Optimal
            result_certificate(reduced, native_result, q3_options)
        else
            nothing
        end
        native_certificate_valid = native_certificate === nothing ||
                                   native_certificate.valid
        if !native_certificate_valid
            push!(
                warnings,
                "Native fixed-trace Q3 failed its reduced original-coordinate " *
                "certificate ($(native_certificate.failures)); SDPX will use " *
                "the PSD2 reference fallback.",
            )
        end
        fallback = native_result === nothing ||
                   !(native_result.status in (Optimal, TimeLimit, UserStopped)) ||
                   !native_certificate_valid
        if fallback && native_result === nothing && time() >= deadline
            return _time_limit_pipeline_result(
                prob,
                report,
                plan,
                time() - pipeline_started,
                warnings,
                opts.diagnostics,
                opts.max_time,
            )
        end
        if fallback && time() < deadline
            reason = native_result === nothing ?
                     (
                         reduced_rejection === :eligible ?
                         :unsupported_start : reduced_rejection
                     ) : native_result.status
            push!(
                warnings,
                "Native fixed-trace Q3 did not produce a promoted result " *
                "(reason=$reason); SDPX reran the unchanged PSD2 reference " *
                "path for numerical safety.",
            )
            native_result = nothing
            T === BigFloat && GC.gc()
            fallback_options = _replace_solver_options(
                opts;
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
                equilibrate=false,
                threads=plan.threads,
            )
            fallback_route = resolve_execution_route(
                AutoPlanner(),
                reduced,
                fallback_options,
            )
            result = _solve_sdp_core!(
                reduced,
                fallback_options;
                x0=preprocessed_warm_start.x0,
                X0=preprocessed_warm_start.X0,
                y0=reduced_y0,
                Y0=preprocessed_warm_start.Y0,
                resume=resume,
                deadline=deadline,
                execution_plan=build_execution_plan(
                    AutoPlanner(),
                    reduced,
                    fallback_route,
                ),
            )
            workspace_bytes = max(
                workspace_bytes,
                estimate_sdp_workspace_bytes(reduced, plan.threads),
            )
        else
            result = native_result
        end
    else
        # Pre-flight against the memory actually available. Nothing compared
        # the workspace size against anything before this, so a model too large
        # for the machine simply ran until an allocation failed, with no
        # indication of which dimension caused it.
        #
        # Deliberately *not* checked against `plan.memory_budget_bytes`: that
        # field is `available × extended_precision_memory_fraction`, a budget
        # for extended-precision buffers rather than for the whole workspace.
        # The lattice benchmark's floor is 2.8 GiB against a 10% budget it
        # exceeds comfortably while still running fine, so comparing the two
        # would warn on a workload that works.
        #
        # Uses the O(1) floor rather than the full estimate, which walks every
        # coefficient and would cost more than the solve it precedes.
        #
        # The estimate must match the KKT route the plan actually chose. The
        # dense floor is an `m x m` figure; the block-arrow route never forms
        # that matrix, and applying the dense model to it overstated the CSDR
        # 200/2/10/400 requirement as 3,218 GiB for a solve that runs in about
        # 5 GiB. A warning wrong by three orders of magnitude drives users off
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
                          dense_workspace_floor_bytes(
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
            equilibrate=plan.scaling === :sdp_ruiz,
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
            execution_plan=plan,
        )
        # Keep diagnostics out of the hot path: recursively traversing every
        # sparse coefficient object can cost much more than a warmed solve.
        workspace_bytes = estimate_sdp_workspace_bytes(
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
    diagnosable_failure = result.status in (
        Stalled,
        IterLimit,
        NumericalBreakdown,
        MaxRestartsExceeded,
        InsufficientPrecision,
        NumericalFailure,
    )
    if opts.mode === OPTIMIZE && diagnosable_failure
        result, _, infeasibility_message =
            certify_optimize_infeasibility(prob, result, opts)
        infeasibility_message === nothing ||
            push!(warnings, infeasibility_message)
    end
    result, certificate, certificate_warning = if result.status === TimeLimit
        (
            result,
            (available=false, reason=:time_limit),
            nothing,
        )
    else
        certify_final_result(prob, result, opts)
    end
    certificate_warning === nothing ||
        push!(warnings, certificate_warning)
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
    chordal_decomposition::Symbol=:auto,
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
                chordal_decomposition=chordal_decomposition,
                algorithm=algorithm,
                parameter_strategy=parameter_strategy,
                working_precision_policy=working_precision_policy,
                minimum_working_precision_bits=
                    minimum_working_precision_bits,
            )
        end
    end
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
        chordal_decomposition=chordal_decomposition,
        algorithm=algorithm,
        parameter_strategy=parameter_strategy,
        working_precision_policy=working_precision_policy,
        minimum_working_precision_bits=minimum_working_precision_bits,
        parameter_policy=:auto,
    )
    if warm_start === nothing
        return solve!(prob, options)
    end
    warm_start isa NamedTuple ||
        throw(ArgumentError("warm_start must be a NamedTuple such as (; x0, X0, y0, Y0)"))
    return solve!(prob, options; warm_start...)
end

function _resolve_precision_type(precision, c, A, C, B, b)
    precision === nothing && return infer_eltype(c, A, C, B, b)
    precision === :float64 && return Float64
    precision === :bigfloat && return BigFloat
    precision isa Type && precision <: AbstractFloat && return precision
    throw(ArgumentError(
        "precision must be nothing, :float64, :bigfloat, or an AbstractFloat type",
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
    chordal_decomposition::Symbol=:auto,
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
        return solve(
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
            chordal_decomposition=chordal_decomposition,
            algorithm=algorithm,
            parameter_strategy=parameter_strategy,
            working_precision_policy=working_precision_policy,
            minimum_working_precision_bits=
                minimum_working_precision_bits,
        )
    end
    return T === BigFloat ? setprecision(run, BigFloat, precision_bits) : run()
end
