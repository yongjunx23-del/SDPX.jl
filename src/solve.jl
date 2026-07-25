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

function _finite_iterate(x, X, y, Y, μ)
    return all(isfinite, x) &&
           all(isfinite, y) &&
           all(isfinite, μ) &&
           all(block -> all(isfinite, block), X) &&
           all(block -> all(isfinite, block), Y)
end

function print_header(opts::SolverOptions)
    opts.verbosity >= 1 || return
    println("iter\tprimal obj\tdual obj\tgap\t\tprimal res\tdual res\tprimal step\tdual step\ttime (s)")
    println("="^131)
end

function print_iter(opts::SolverOptions{T}, iter, pObj, dObj, gap, p_res, d_res, tX=nothing, tY=nothing, dt=nothing) where {T}
    opts.verbosity >= 1 || return
    pf, df, gf, pr, dr = Float64(pObj), Float64(dObj), Float64(gap), Float64(p_res), Float64(d_res)
    if tX === nothing
        @printf "%d\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n" iter pf df gf pr dr
    else
        @printf "%d\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n" iter pf df gf pr dr Float64(tX) Float64(tY) dt
    end
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
        return (
            β=opts.β,
            γ=opts.γ,
            Ωp=opts.Ωp,
            Ωd=opts.Ωd,
            predictor=opts.predictor,
            parameter_strategy=opts.parameter_strategy,
            profile=:lp_mehrotra,
        )
    end
    if prob.structure.profile ===
       :sparse_coefficients_dense_psd_dense_schur &&
       prob.dims.m >= 1_000 &&
       prob.dims.n > 0
        return (
            β=T(0.1),
            γ=T(0.85),
            Ωp=T(100),
            Ωd=T(0.001),
            predictor=:sdpb,
            parameter_strategy=opts.parameter_strategy,
            profile=:large_equality_dense_schur,
        )
    end
    if cons isa SparseCons{T} && prob.dims.n == 0 && all(<=(2), prob.dims.k)
        max_active = maximum(length, cons.active; init=0)
        beta, gamma, profile = if max_active <= 6
            (T(0.1), T(0.85), :small_arrow_2x2)
        elseif max_active <= 14
            (T(0.1), T(0.8), :medium_arrow_2x2)
        else
            # Beyond the range the fixed profiles were calibrated on
            # (thresholds top out at 14 active variables per block). The CSDR
            # 80/4/40/100 model has 385, and the old `(0.4, 0.7)` setting did
            # not converge on it at all; a sweep found `(0.01, 0.85)` reaching
            # the correct basin.
            (T(0.01), T(0.85), :large_arrow_2x2)
        end
        if T === BigFloat && opts.ϵ_gap < T(1e-10)
            beta = T(0.1)
            gamma = min(gamma, T(0.75))
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
        # The final sweep found that matching the largest block norm is a safer
        # general start than the earlier fitted 3× multiplier. Keep 10 as the
        # floor for small-data models.
        stats = block_norm_stats(prob)
        omega = max(T(10), T(OMEGA_DATA_MULTIPLIER) * stats.maxnorm)
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
    return (
        β=opts.β,
        γ=opts.γ,
        Ωp=opts.Ωp,
        Ωd=opts.Ωd,
        predictor=opts.predictor,
        parameter_strategy=opts.parameter_strategy,
        profile=:general_fixed,
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
    @inbounds for block in eachindex(X)
        copy_owned!(best.X[block], X[block])
        copy_owned!(best.Y[block], Y[block])
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
                matrix
            end
            for block in eachindex(Y0)
        ]
    end
    return scaled_x0, scaled_X0, y0, scaled_Y0
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
    resume::AbstractString="", deadline::Float64=Inf) where {T}

    core_started = time()
    opts.parameter_policy in (:fixed, :auto) ||
        throw(ArgumentError("parameter_policy must be :fixed or :auto"))
    opts.parameter_strategy in (:fixed, :adaptive) ||
        throw(ArgumentError("parameter_strategy must be :fixed or :adaptive"))
    if opts.parameter_policy === :auto
        selected = recommended_parameters(prob, opts)
        opts.verbosity >= 1 && println(
            "SDPX auto parameters: profile=$(selected.profile), " *
            "beta=$(Float64(selected.β)), gamma=$(Float64(selected.γ)), " *
            "omega_p=$(Float64(selected.Ωp)), omega_d=$(Float64(selected.Ωd)), " *
            "predictor=$(selected.predictor), strategy=$(selected.parameter_strategy)",
        )
        opts = _replace_solver_options(
            opts;
            β=selected.β,
            γ=selected.γ,
            Ωp=selected.Ωp,
            Ωd=selected.Ωd,
            predictor=selected.predictor,
            parameter_strategy=selected.parameter_strategy,
            parameter_policy=:fixed,
        )
    end
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

    if T === BigFloat
        check_precision_consistency(prob, opts.precision_bits, opts.verbosity)
        opts.convert_inputs && (prob = reround(prob, opts.precision_bits))
    end

    eq = nothing
    solve_prob = prob
    if opts.equilibrate
        solve_prob, eq = equilibrate(prob)
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

    L, m, n, k = solve_prob.dims
    ws = Workspace(
        solve_prob;
        extended_precision_blas=opts.extended_precision_blas,
        extended_precision_memory_fraction=
            opts.extended_precision_memory_fraction,
        thread_count=opts.threads,
    )
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

    total_reg = 0
    centering_attempts = 0
    last_refine_steps = 0
    last_refine_residual = zero(T)
    t_start = core_started
    parameter_controller = AdaptiveIPMController(opts)
    phase_residual = 0.0
    phase_schur = 0.0
    phase_kkt = 0.0
    phase_predictor = 0.0
    phase_corrector = 0.0
    phase_line_search = 0.0
    phase_update = 0.0

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

    while true
        gap = pObj - dObj
        gap_rel = abs(gap) / max(one(T), (abs(pObj) + abs(dObj)) / 2)
        term_ok, gap_ok = if opts.termination === :legacy
            (p_res <= opts.ϵ_primal && d_res <= opts.ϵ_dual), (zero(T) <= gap <= opts.ϵ_gap)
        else
            (p_res / scale_p <= opts.ϵ_primal && d_res / scale_d <= opts.ϵ_dual), (gap_rel <= opts.ϵ_gap)
        end
        # Residuals between accepted steps are updated from the exact affine
        # residual recurrence below. Before issuing a success certificate,
        # recompute them from the current iterate so accumulated roundoff can
        # never turn an estimate into a false `Optimal` status.
        if term_ok
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
        complementarity = sum(l -> kdot(X[l], Y[l]), 1:L; init=zero(T))
        objective_scale = max(abs(pObj), abs(dObj))
        merit = stagnation_merit(stagnation, opts, p_res, d_res, gap_rel,
            complementarity, scale_p, scale_d, objective_scale)
        floor_reached = at_precision_floor(p_res, d_res, gap_rel, scale_p, scale_d)
        stagnated = observe!(stagnation, merit, floor_reached, opts.iter_max - iter)
        isfinite(merit) && !isfinite(initial_merit) && (initial_merit = merit)
        if isfinite(merit) && merit < best_merit
            best_merit = merit
            _store_best_iterate!(
                best_iterate,
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
            status = Stalled
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
                p_res=_diagnostic_scalar_copy(p_res),
                d_res=_diagnostic_scalar_copy(d_res),
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
        )
        if result.status === :breakdown
            status, message = NumericalBreakdown, result.reason
            break
        end
        p_res, d_res = result.p_res, result.d_res
        last_refine_steps = result.refine_steps
        last_refine_residual = result.refine_residual
        total_reg += result.reg_attempts
        phase_residual += result.phase_times.residual_and_block_factor
        phase_schur += result.phase_times.schur_assembly
        phase_kkt += result.phase_times.kkt_factorization
        phase_predictor += result.phase_times.predictor
        phase_corrector += result.phase_times.corrector

        line_search_started = time_ns()
        tX, tY = threaded_line_search!(
            ws,
            X,
            Y,
            iteration_options.γ,
            opts.min_step,
            opts.step_rule,
        )
        selected_step_rule = resolved_step_rule(ws, opts.step_rule)
        backtracking_count =
            estimate_backtracking_count(
                tX,
                iteration_options.γ,
                selected_step_rule,
            ) +
            estimate_backtracking_count(
                tY,
                iteration_options.γ,
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
        x_stuck = tX < opts.min_step
        y_stuck = tY < opts.min_step
        if x_stuck && primal_feasible && !(y_stuck && !dual_feasible)
            tX = zero(T)          # freeze a converged primal, keep going
            x_stuck = false
        end
        if y_stuck && dual_feasible && !x_stuck
            tY = zero(T)
            y_stuck = false
        end
        if x_stuck || y_stuck
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
                                    min(opts.omega_step, sqrt(floatmax(T))) : opts.omega_step
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
                status, message = Stalled,
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
                for l in 1:L
                    μ[l] = opts.β * kdot(X[l], Y[l]) / k[l]
                end
                opts.verbosity >= 1 &&
                    println("Step size too small! Restart $restarts/$(opts.max_restarts): rescaling the collapsed side(s) by ×$(Float64(opts.omega_step)).")
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
        for l in 1:L
            trial_combine!(X[l], X[l], tX, ws.blk[l].dX)
            trial_combine!(Y[l], Y[l], tY, ws.blk[l].dY)
        end
        trial_combine!(x, x, tX, ws.dx)
        n > 0 && trial_combine!(y, y, tY, ws.dy)

        if !_finite_iterate(x, X, y, Y, μ)
            status, message = NumericalBreakdown,
            "non-finite primal or dual iterate detected" *
            (dynamic_range_limited(T) ? " ($T's dynamic range exceeded)" : "") *
            " — rescale (equilibrate=true), loosen Ωp/Ωd/omega_step, or use a wider-range T"
            break
        end

        complementarity_after = zero(T)
        @inbounds for l in 1:L
            complementarity_after += kdot(X[l], Y[l])
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
        )
        phase_update += (time_ns() - update_started) / 1.0e9

        pObj = LinearAlgebra.dot(solve_prob.c, x)
        dObj = dual_objective(solve_prob, y, Y)
        if !isfinite(pObj) || !isfinite(dObj)
            status, message = NumericalBreakdown,
            "non-finite primal or dual objective detected"
            break
        end
        iter += 1

        for l in 1:L
            μ[l] = parameter_controller.beta * kdot(X[l], Y[l]) / k[l]
        end
        if !all(isfinite, μ)
            status, message = NumericalBreakdown,
            "non-finite complementarity target detected"
            break
        end

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

    if eq !== nothing
        x, X, y, Y = unequilibrate(eq, x, X, y, Y)
        pObj = LinearAlgebra.dot(prob.c, x)
        dObj = dual_objective(prob, y, Y)
    end
    # Always report certificates in the same (original) coordinates as the
    # returned iterate, including non-optimal exits and unequilibrated solves.
    p_res, d_res = solution_residuals(prob, x, X, y, Y)

    elapsed = time() - t_start
    timings = opts.timing ? (
        total=elapsed,
        residual_and_block_factor=phase_residual,
        schur_assembly=phase_schur,
        kkt_factorization=phase_kkt,
        predictor=phase_predictor,
        corrector=phase_corrector,
        line_search=phase_line_search,
        update=phase_update,
    ) : nothing
    gap_rel_final = abs(pObj - dObj) / max(one(T), (abs(pObj) + abs(dObj)) / 2)

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

"""Initial `X = Ω·I` is set to this multiple of `max‖C_l‖∞`.

1, i.e. Ω simply matches the largest block norm. An earlier value of 3 was
fitted against runs that were terminating prematurely, and re-measuring once
that was fixed showed 1 is clearly better on the CSDR sparse model: 47
iterations to gap 3.08e-04, against 35 iterations to 7.08e-03 at 3."""
const OMEGA_DATA_MULTIPLIER = 1

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

function solve!(
    prob::SDPProblem{BigFloat},
    opts::SolverOptions{BigFloat}=SolverOptions{BigFloat}();
    x0=nothing,
    X0=nothing,
    y0=nothing,
    Y0=nothing,
    resume::AbstractString="",
)
    requested_precision = opts.precision_bits
    requested_precision > 0 ||
        throw(ArgumentError("precision_bits must be positive"))
    run = () -> _solve_pipeline!(
        prob,
        opts;
        x0=x0,
        X0=X0,
        y0=y0,
        Y0=Y0,
        resume=resume,
    )
    return Base.precision(BigFloat) == requested_precision ?
           run() :
           setprecision(run, BigFloat, requested_precision)
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
    reduced, equality_map, report = presolve_equalities(prob, opts)
    # Finalize the plan against the model that will actually be factorized.
    # In particular, equality presolve can change the selected parameter profile
    # and the diagnostic equality count.
    plan = build_execution_plan(report.inconsistent ? prob : reduced, opts)
    warnings = String[]
    if opts.threads > Base.Threads.nthreads()
        push!(
            warnings,
            "Requested $(opts.threads) threads, but Julia started with " *
            "$(Base.Threads.nthreads()); the solve uses $(plan.threads).",
        )
    end
    if T === BigFloat && opts.threads > 1
        push!(
            warnings,
            "BigFloat solver kernels are serial because the current mutable-scalar workspaces require strict ownership and aliasing guarantees.",
        )
    end
    if plan.classification.cone === :socp
        push!(
            warnings,
            "Detected exact SOC-arrow PSD structure. The current " *
            ":socp_psd_lift path solves the semidefinite lift; native SOCP " *
            "scaling and Newton systems are not implemented.",
        )
    end
    _validate_warm_start(
        prob;
        x0=x0,
        X0=X0,
        y0=y0,
        Y0=Y0,
        resume=resume,
        accepted_y_lengths=(
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
            opts.diagnostics,
        )

    reduced_y0 = if y0 === nothing
        nothing
    else
        supplied = _owned_array_copy(T, y0)
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
        result, redundant_rows, workspace_bytes = solve_lp!(
            reduced,
            opts,
            plan;
            x0=x0,
            y0=reduced_y0,
            deadline=deadline,
        )
    else
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
            x0=x0,
            X0=X0,
            y0=reduced_y0,
            Y0=Y0,
            resume=resume,
            deadline=deadline,
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
        )
    end
    result = _restore_equalities(result, equality_map)
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
    if any(row -> row.fallback, result.parameter_history)
        push!(
            warnings,
            "Adaptive parameter control detected instability and reverted to the fixed defaults.",
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
    warm_start=nothing,
    presolve::Bool=true,
    scaling::Symbol=:auto,
    algorithm::Symbol=:auto,
    parameter_strategy::Symbol=:fixed,
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
                warm_start=warm_start,
                presolve=presolve,
                scaling=scaling,
                algorithm=algorithm,
                parameter_strategy=parameter_strategy,
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
        presolve=presolve,
        scaling=scaling,
        algorithm=algorithm,
        parameter_strategy=parameter_strategy,
        parameter_policy=:auto,
        timing=true,
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
    warm_start=nothing,
    presolve::Bool=true,
    scaling::Symbol=:auto,
    algorithm::Symbol=:auto,
    parameter_strategy::Symbol=:fixed,
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
            warm_start=warm_start,
            presolve=presolve,
            scaling=scaling,
            algorithm=algorithm,
            parameter_strategy=parameter_strategy,
        )
    end
    return T === BigFloat ? setprecision(run, BigFloat, precision_bits) : run()
end
