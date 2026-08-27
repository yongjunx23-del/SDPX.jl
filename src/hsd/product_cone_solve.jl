#=====================================================================#
# Internal symmetric product-cone HSD solve loop.
#
# This file is deliberately not wired to the public/MOI route.  It drives
# `product_hsd_step!` and promotes a terminal status only after one of the
# original-coordinate certificate verifiers succeeds.  There is no legacy
# solve, PSD lift, projected-gradient ray search, or other fallback here.
#=====================================================================#

"""Typed terminal status of the internal symmetric product-HSD loop."""
@enum ProductHSDSolveStatus::UInt8 begin
    ProductHSDOptimal
    ProductHSDPrimalInfeasible
    ProductHSDDualInfeasible
    ProductHSDMaxIterations
    ProductHSDSingular
    ProductHSDBreakdown
    ProductHSDRankAmbiguous
    ProductHSDTimeLimit
end

"""Typed, inspectable reason for a product-HSD terminal status."""
@enum ProductHSDSolveReason::UInt8 begin
    ProductHSDVerifiedInitialPoint
    ProductHSDVerifiedAcceptedStep
    ProductHSDVerifiedTerminalNewtonTrial
    ProductHSDIterationLimitReached
    ProductHSDSingularKKTReason
    ProductHSDLineSearchBreakdown
    ProductHSDDirectionBreakdown
    ProductHSDUnverifiedZeroComplementarity
    ProductHSDRankAmbiguousSetup
    ProductHSDRankRayVerificationFailed
    ProductHSDKKTInitializationFailed
    ProductHSDTimeLimitReached
end

"""
    ProductHSDSolveResult{T}

Cold-path result of [`product_hsd_solve!`](@ref).  `x`, `s`, and `y` are in
original coordinates: all three contain the recovered optimum, only `y`
contains a normalized primal-infeasibility ray, and `x`/`s` contain a
normalized dual-infeasibility ray.  The `hsd_*` buffers preserve the exact
canonical homogeneous point which produced the status, so callers can load
it into a fresh state and invoke the strict verifier a second time.

For a verified terminal Newton trial, the result owns the trial buffers while
the mutable input state is restored to its last runtime-consistent accepted
iterate.  `terminal_alpha > 0` records that case.  For an optimal result,
`normalized_residual` is the scale-invariant recovered residual used by
[`verify_optimal!`](@ref), namely the normalized homogeneous residual divided
by `tau`; it therefore reports the quantity which actually passed `tol`.
"""
struct ProductHSDSolveResult{T}
    status::ProductHSDSolveStatus
    reason::ProductHSDSolveReason
    last_step::HSDStepCode
    iterations::Int
    factorizations::Int
    terminal_alpha::T
    tau::T
    kappa::T
    mu::T
    normalized_residual::T
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    hsd_x::Vector{T}
    hsd_s::Vector{T}
    hsd_y::Vector{T}
end

@inline function _product_hsd_make_result(
    state::ProductConeHSDState{T},
    status::ProductHSDSolveStatus,
    reason::ProductHSDSolveReason,
    last_step::HSDStepCode,
    terminal_alpha::T,
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
) where {T}
    base = state.base
    hsd_residual!(base)
    normalized_residual = hsd_normalized_residual(base)
    if status === ProductHSDOptimal
        # Optimality was promoted only after `verify_optimal!` proved tau > 0
        # and checked this recovered, homogeneous-scale-invariant residual.
        # Reporting the unscaled embedding residual here made a certified solve
        # appear looser whenever tau > 1 (the Exp release point has tau ≈ 2.29).
        normalized_residual /= base.tau
    end
    return ProductHSDSolveResult{T}(
        status,
        reason,
        last_step,
        base.record.iterations,
        product_hsd_factor_count(state),
        terminal_alpha,
        base.tau,
        base.kappa,
        base.mu,
        normalized_residual,
        copy(x_original),
        copy(s_original),
        copy(y_original),
        copy(base.x),
        copy(base.s),
        copy(base.y),
    )
end

"""Run only the strict original-coordinate certificate gates."""
function _product_hsd_verified_result(
    state::ProductConeHSDState{T},
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    tol::T,
    reason::ProductHSDSolveReason,
    last_step::HSDStepCode,
    terminal_alpha::T=zero(T);
    check_optimal::Bool=true,
) where {T}
    canonical = state.base.canonical
    if check_optimal && verify_optimal!(
        canonical, state.base, x_original, s_original, y_original; tol=tol,
    )
        return _product_hsd_make_result(
            state, ProductHSDOptimal, reason, last_step, terminal_alpha,
            x_original, s_original, y_original,
        )
    end
    if verify_primal_infeasibility!(
        canonical, state.base, y_original; tol=tol,
    )
        return _product_hsd_make_result(
            state, ProductHSDPrimalInfeasible, reason, last_step,
            terminal_alpha, x_original, s_original, y_original,
        )
    end
    if verify_dual_infeasibility!(
        canonical, state.base, x_original, s_original; tol=tol,
    )
        return _product_hsd_make_result(
            state, ProductHSDDualInfeasible, reason, last_step,
            terminal_alpha, x_original, s_original, y_original,
        )
    end
    return nothing
end

@inline function _product_hsd_refinement_maxabs(values::AbstractVector{T}) where {T}
    magnitude_max = zero(T)
    @inbounds for value in values
        magnitude = abs(value)
        magnitude > magnitude_max && (magnitude_max = magnitude)
    end
    return magnitude_max
end

@inline function _product_hsd_refinement_scale(values::AbstractVector{T}) where {T}
    return max(one(T), _product_hsd_refinement_maxabs(values))
end

"""
Try one cold-path affine certificate refinement of an accepted HSD iterate.
The cone variables are held fixed while the primal variable is
corrected in `A*dx = -rP`; the dual is then corrected in the joint
stationarity/gap equations `[A'; b']*dy = [-rD; -gap]`. This removes the
homogeneous residual amplification which otherwise lets the data-normalized
HSD gate stop before the recovered original-coordinate gate is satisfied.

The refinement is fail closed: it is adopted only when the unchanged strict
optimal verifier accepts at the caller's requested tolerance. Exp dual blocks
also admit the exact `u=v` boundary representative when `u` is structurally
free in both affine equations; this removes a second-order cone residual
without altering stationarity or the gap. No cone-membership check or
tolerance is relaxed.
"""
function _product_hsd_refined_optimal_result!(
    state::ProductConeHSDState{T},
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    tol::T,
    reason::ProductHSDSolveReason,
    last_step::HSDStepCode,
    terminal_alpha::T=zero(T),
) where {T}
    base = state.base
    base.tau > tol || return nothing
    hsd_residual!(base)
    recovered_residual = hsd_normalized_residual(base) / base.tau
    recovered_residual <= sqrt(tol) || return nothing

    canonical = base.canonical
    # Reuse the dense canonical matrix already owned by HSDState; the cold
    # refinement must not create a second sparse-to-dense copy.
    A = base.Ad
    x = base.x ./ base.tau
    s = base.s ./ base.tau
    y = base.y ./ base.tau
    primal_residual = A * x + s - canonical.b

    try
        if base.n == 0
            _product_hsd_refinement_maxabs(primal_residual) <= tol || return nothing
        else
            x .+= A \ (-primal_residual)
        end

        # Near the x=0 exposed face of K_exp, the exact boundary relation is
        # z=y. Scaling failure can occur a few ulps before that representative
        # is reached. Form it only as a cold certificate candidate, then
        # restore affine primal feasibility. The candidate is adopted below
        # only if every strict original-coordinate gate accepts it.
        exp_boundary_changed = false
        for block in canonical.cone_layout.blocks
            block.cone === :exp || continue
            u = block.offset
            v = u + 1
            w = u + 2
            scale = max(one(T), abs(s[u]), abs(s[v]), abs(s[w]))
            neighborhood = sqrt(tol) * scale
            if abs(s[u]) <= neighborhood &&
               abs(s[w] - s[v]) <= neighborhood && s[v] > zero(T)
                s[u] = zero(T)
                s[w] = s[v]
                exp_boundary_changed = true
            end
        end
        if exp_boundary_changed
            primal_residual = A * x + s - canonical.b
            if base.n == 0
                _product_hsd_refinement_maxabs(primal_residual) <= tol ||
                    return nothing
            else
                x .+= A \ (-primal_residual)
            end
        end

        dual_residual = transpose(A) * y + canonical.c
        gap = dot(canonical.c, x) + dot(canonical.b, y)
        affine_dual = Matrix{T}(undef, base.n + 1, base.m)
        @inbounds for column in 1:base.m
            for row in 1:base.n
                affine_dual[row, column] = A[column, row]
            end
            affine_dual[end, column] = canonical.b[column]
        end
        rhs = Vector{T}(undef, base.n + 1)
        @inbounds for row in 1:base.n
            rhs[row] = -dual_residual[row]
        end
        rhs[end] = -gap
        y .+= affine_dual \ rhs

        # For K_exp^*, L_E(u,v,w)=(u-v,-u,w). When the u coordinate is
        # structurally absent from A' and b', replacing u by v preserves both
        # affine equations and selects the stable x=0 boundary representative.
        if !in_canonical_cone(canonical, y; dual=true, tol=tol)
            for block in canonical.cone_layout.blocks
                block.cone === :exp || continue
                u = block.offset
                structurally_free = true
                @inbounds for row in axes(affine_dual, 1)
                    if !iszero(affine_dual[row, u])
                        structurally_free = false
                        break
                    end
                end
                structurally_free && (y[u] = y[u + 1])
            end
        end
    catch exception
        exception isa LinearAlgebra.SingularException ||
            exception isa LinearAlgebra.PosDefException ||
            exception isa LinearAlgebra.RankDeficientException || rethrow()
        return nothing
    end

    primal_residual = A * x + s - canonical.b
    dual_residual = transpose(A) * y + canonical.c
    primal_scale = max(
        _product_hsd_refinement_scale(x),
        _product_hsd_refinement_scale(s),
        _product_hsd_refinement_scale(canonical.b),
    )
    dual_scale = max(
        _product_hsd_refinement_scale(y),
        _product_hsd_refinement_scale(canonical.c),
    )
    _product_hsd_refinement_maxabs(primal_residual) <= tol * primal_scale ||
        return nothing
    _product_hsd_refinement_maxabs(dual_residual) <= tol * dual_scale ||
        return nothing
    abs(dot(canonical.c, x) + dot(canonical.b, y)) <=
        tol * (one(T) + abs(dot(canonical.c, x)) + abs(dot(canonical.b, y))) ||
        return nothing
    in_canonical_cone(canonical, s; dual=false, tol=tol) || return nothing
    in_canonical_cone(canonical, y; dual=true, tol=tol) || return nothing

    saved_x = copy(base.x)
    saved_s = copy(base.s)
    saved_y = copy(base.y)
    saved_kappa = base.kappa
    @inbounds for index in eachindex(base.x)
        base.x[index] = base.tau * x[index]
    end
    @inbounds for index in eachindex(base.s)
        base.s[index] = base.tau * s[index]
    end
    @inbounds for index in eachindex(base.y)
        base.y[index] = base.tau * y[index]
    end
    # Once primal feasibility, stationarity, and the recovered gap have been
    # refined, the scalar HSD equation has the exact solution kappa=0. Keep a
    # small positive representative so the homogeneous point remains valid.
    base.kappa = min(base.kappa, base.tau * tol / T(8))
    hsd_residual!(base)
    if verify_optimal!(
        canonical, base, x_original, s_original, y_original; tol=tol,
    )
        return _product_hsd_make_result(
            state, ProductHSDOptimal, reason, last_step, terminal_alpha,
            x_original, s_original, y_original,
        )
    end

    copyto!(base.x, saved_x)
    copyto!(base.s, saved_s)
    copyto!(base.y, saved_y)
    base.kappa = saved_kappa
    hsd_residual!(base)
    return nothing
end

@inline function _product_hsd_candidate_result!(
    state::ProductConeHSDState{T},
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    tol::T,
    reason::ProductHSDSolveReason,
    last_step::HSDStepCode,
    terminal_alpha::T=zero(T),
) where {T}
    refined = _product_hsd_refined_optimal_result!(
        state, x_original, s_original, y_original, tol, reason, last_step,
        terminal_alpha,
    )
    refined === nothing || return refined
    return _product_hsd_verified_result(
        state, x_original, s_original, y_original, tol, reason, last_step,
        terminal_alpha; check_optimal=true,
    )
end

@inline function _product_hsd_terminal_alpha(
    state::ProductConeHSDState{T}, tol::T,
) where {T}
    base = state.base
    # A failed nonsymmetric scaling update invalidates the runtime factor.
    # There is then no authoritative cone metric from which to compute a
    # terminal boundary step; fail closed instead of calling the throwing
    # max-step API on invalid state.
    state.runtime.valid || return T(NaN)
    ap = max_step_primal!(state.runtime, base.s, base.ds)
    ad = max_step_dual!(state.runtime, base.y, base.dy)
    (isfinite(ap) || ap == T(Inf)) || return T(NaN)
    (isfinite(ad) || ad == T(Inf)) || return T(NaN)
    (ap >= zero(T) && ad >= zero(T)) || return T(NaN)
    alpha = min(one(T), ap, ad)
    base.dtau < zero(T) && (alpha = min(alpha, -base.tau / base.dtau))
    base.dkappa < zero(T) &&
        (alpha = min(alpha, -base.kappa / base.dkappa))

    # A terminal certificate may be much closer to the boundary than a point
    # from which another NT factor must be computed.  It remains strictly
    # interior and uses one common alpha for every HSD coordinate.
    margin = max(sqrt(eps(T)), min(T(1) / T(1000), tol / T(10)))
    return (one(T) - margin) * alpha
end

@inline function _product_hsd_stage_terminal_trial!(
    state::ProductConeHSDState{T}, alpha::T,
) where {T}
    base = state.base
    @inbounds for j in 1:base.n
        base.xt[j] = base.x[j] + alpha * base.dx[j]
    end
    @inbounds for k in 1:base.m
        base.st[k] = base.s[k] + alpha * base.ds[k]
        base.yt[k] = base.y[k] + alpha * base.dy[k]
    end
    base.tau_t = base.tau + alpha * base.dtau
    base.kappa_t = base.kappa + alpha * base.dkappa
    isfinite(base.tau_t) && isfinite(base.kappa_t) &&
        base.tau_t > zero(T) && base.kappa_t > zero(T) || return false
    product_strictly_interior(
        state.runtime, base.st, base.yt,
    ) || return false

    _hsd_trial_residual!(base)
    p2 = _hsd_maxinf(base.rPt)
    d2 = _hsd_maxinf(base.rDt)
    gap2 = -dot(base.c, base.xt) - dot(base.b, base.yt) + base.kappa_t
    (isfinite(p2) && isfinite(d2) && isfinite(gap2)) || return false
    _hsd_residual_homotopy_ok(base, alpha, p2, d2, gap2) || return false
    scale = max(
        one(T), _hsd_maxinf(base.rP), _hsd_maxinf(base.rD), abs(base.rG),
    )
    guard = T(256) * sqrt(eps(T)) * scale
    return max(p2, d2, abs(gap2)) <= scale * T(1.0005) + guard
end

"""
Check the already-computed Newton direction after an ordinary line-search
breakdown.  This is not an alternative solver: it performs no factorization
and changes neither the direction nor the HSD equations.  A single global
strict-interior trial is promoted only if residual homotopy and an ordinary
original-coordinate certificate verifier both pass.
"""
function _product_hsd_terminal_verified_result!(
    state::ProductConeHSDState{T},
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    tol::T,
    last_step::HSDStepCode,
) where {T}
    base = state.base
    alpha = _product_hsd_terminal_alpha(state, tol)
    (isfinite(alpha) && alpha > zero(T)) || return nothing
    _product_hsd_stage_terminal_trial!(state, alpha) || return nothing

    saved_x = copy(base.x)
    saved_s = copy(base.s)
    saved_y = copy(base.y)
    saved_tau = base.tau
    saved_kappa = base.kappa
    copyto!(base.x, base.xt)
    copyto!(base.s, base.st)
    copyto!(base.y, base.yt)
    base.tau = base.tau_t
    base.kappa = base.kappa_t

    result = _product_hsd_candidate_result!(
        state, x_original, s_original, y_original, tol,
        ProductHSDVerifiedTerminalNewtonTrial, last_step, alpha,
    )

    # Keep the mutable state on its last accepted pair: unlike the terminal
    # certificate, that pair has an NT runtime which may safely be reused.
    copyto!(base.x, saved_x)
    copyto!(base.s, saved_s)
    copyto!(base.y, saved_y)
    base.tau = saved_tau
    base.kappa = saved_kappa
    hsd_residual!(base)
    restored = try_update_scaling!(state.runtime, base.s, base.y, base.mu)
    if !restored
        # Do not let a discarded verified trial leak through the output
        # buffers of the ensuing fail-closed breakdown result.
        fill!(x_original, zero(T))
        fill!(s_original, zero(T))
        fill!(y_original, zero(T))
        return nothing
    end
    return result
end

"""
    product_hsd_solve!(state; max_iterations=300, max_time=Inf, tol=nothing)

Drive the internal native LP/SOC/PSD HSD step and return a typed cold-path
result.  Status promotion is certificate-only.  This routine never calls the
legacy orthant solve, a projected-gradient Farkas search, or a lifted cone
formulation, and it is intentionally not connected to a public solver route.
"""
function product_hsd_solve!(
    state::ProductConeHSDState{T};
    max_iterations::Integer=300,
    max_time::Real=Inf,
    tol::Union{Nothing,T}=nothing,
    initialization::Symbol=:auto,
) where {T}
    initialization in (:auto, :identity, :kkt) || throw(ArgumentError(
        "initialization must be :auto, :identity, or :kkt",
    ))
    max_iterations >= 0 || throw(ArgumentError(
        "max_iterations must be nonnegative, got $max_iterations",
    ))
    time_limit = Float64(max_time)
    (isfinite(time_limit) || isinf(time_limit)) && time_limit >= 0.0 ||
        throw(ArgumentError("max_time must be nonnegative and finite, or Inf"))
    started_ns = time_ns()
    certificate_tol = tol === nothing ? T(default_certificate_tol(T)) : tol
    (isfinite(certificate_tol) && certificate_tol > zero(T)) ||
        throw(ArgumentError("tol must be finite and positive"))

    base = state.base
    x_original = zeros(T, base.n)
    s_original = zeros(T, base.m)
    y_original = zeros(T, base.m)

    if base.rank_ambiguous
        return _product_hsd_make_result(
            state, ProductHSDRankAmbiguous, ProductHSDRankAmbiguousSetup,
            HSDStepDirectionFailed, zero(T), x_original, s_original,
            y_original,
        )
    end
    if base.rank_incompatible
        copyto!(base.x, base.rank_ray)
        if verify_dual_infeasibility!(
            base.canonical, base, x_original, s_original; tol=certificate_tol,
        )
            return _product_hsd_make_result(
                state, ProductHSDDualInfeasible,
                ProductHSDVerifiedInitialPoint, HSDStepDirectionFailed,
                zero(T), x_original, s_original, y_original,
            )
        end
        return _product_hsd_make_result(
            state, ProductHSDBreakdown,
            ProductHSDRankRayVerificationFailed, HSDStepDirectionFailed,
            zero(T), x_original, s_original, y_original,
        )
    end

    selected_initialization = initialization === :auto ?
        (state.kkt_route === :expanded ? :kkt : :identity) : initialization
    if selected_initialization === :kkt
        start_report = kkt_derived_start!(state)
        start_report.ok || return _product_hsd_make_result(
            state, ProductHSDBreakdown, ProductHSDKKTInitializationFailed,
            HSDStepDirectionFailed, zero(T), x_original, s_original,
            y_original,
        )
    else
        product_hsd_cold_start!(state)
    end
    initial = _product_hsd_candidate_result!(
        state, x_original, s_original, y_original, certificate_tol,
        ProductHSDVerifiedInitialPoint, HSDStepOK,
    )
    initial === nothing || return initial

    for _ in 1:Int(max_iterations)
        elapsed_seconds = Float64(time_ns() - started_ns) * 1.0e-9
        if elapsed_seconds >= time_limit
            return _product_hsd_make_result(
                state, ProductHSDTimeLimit, ProductHSDTimeLimitReached,
                HSDStepOK, zero(T), x_original, s_original, y_original,
            )
        end
        code = product_hsd_step!(state)
        if code === HSDStepSingularKKT
            return _product_hsd_make_result(
                state, ProductHSDSingular, ProductHSDSingularKKTReason, code,
                zero(T), x_original, s_original, y_original,
            )
        elseif code === HSDStepBreakdown
            terminal = _product_hsd_terminal_verified_result!(
                state, x_original, s_original, y_original, certificate_tol,
                code,
            )
            terminal === nothing || return terminal
            return _product_hsd_make_result(
                state, ProductHSDBreakdown, ProductHSDLineSearchBreakdown,
                code, zero(T), x_original, s_original, y_original,
            )
        elseif code === HSDStepDirectionFailed
            # A failed *next* Newton direction does not invalidate the current
            # accepted iterate. First run the authoritative original-coordinate
            # gates, then the existing finite terminal-trial verifier; only an
            # unverified pair may be reported as direction breakdown.
            current = _product_hsd_candidate_result!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDVerifiedAcceptedStep, code,
            )
            current === nothing || return current
            terminal = _product_hsd_terminal_verified_result!(
                state, x_original, s_original, y_original, certificate_tol,
                code,
            )
            terminal === nothing || return terminal
            return _product_hsd_make_result(
                state, ProductHSDBreakdown, ProductHSDDirectionBreakdown,
                code, zero(T), x_original, s_original, y_original,
            )
        elseif code === HSDStepAlreadyOptimal
            verified = _product_hsd_candidate_result!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDVerifiedAcceptedStep, code,
            )
            verified === nothing || return verified
            return _product_hsd_make_result(
                state, ProductHSDBreakdown,
                ProductHSDUnverifiedZeroComplementarity, code, zero(T),
                x_original, s_original, y_original,
            )
        end

        verified = _product_hsd_candidate_result!(
            state, x_original, s_original, y_original, certificate_tol,
            ProductHSDVerifiedAcceptedStep, code,
        )
        verified === nothing || return verified
    end

    return _product_hsd_make_result(
        state, ProductHSDMaxIterations, ProductHSDIterationLimitReached,
        HSDStepOK, zero(T), x_original, s_original, y_original,
    )
end

"""Construct an internal product-HSD state and solve it."""
function product_hsd_solve(
    canonical::CanonicalConicProgram{T};
    kkt_route::Symbol=:bordered, kwargs...,
) where {T<:AbstractFloat}
    state = ProductConeHSDState(canonical; kkt_route=kkt_route)
    return product_hsd_solve!(state; kwargs...)
end
