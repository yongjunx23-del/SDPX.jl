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
    ProductHSDInsufficientPrecision
end

"""Typed, inspectable reason for a product-HSD terminal status."""
@enum ProductHSDSolveReason::UInt8 begin
    ProductHSDVerifiedInitialPoint
    ProductHSDVerifiedAcceptedStep
    ProductHSDVerifiedTerminalNewtonTrial
    ProductHSDVerifiedTerminationRay
    ProductHSDIterationLimitReached
    ProductHSDSingularKKTReason
    ProductHSDLineSearchBreakdown
    ProductHSDDirectionBreakdown
    ProductHSDUnverifiedZeroComplementarity
    ProductHSDRankAmbiguousSetup
    ProductHSDRankRayVerificationFailed
    ProductHSDKKTInitializationFailed
    ProductHSDTimeLimitReached
    ProductHSDTauCollapseRecoveryExhausted
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
    tau_collapse_recoveries::Int
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    hsd_x::Vector{T}
    hsd_s::Vector{T}
    hsd_y::Vector{T}
end

"""
Return the recovered homogeneous residual without forming an unbounded
`inv(tau)`.  For a resolvable tau this is bit-for-bit the historical
`hsd_normalized_residual/tau` expression.  At or below the arithmetic floor
there is no numerically authoritative recovered point, so `Inf` is returned
instead of amplifying embedding roundoff into an overflow.
"""
@inline function _product_hsd_recovered_residual(
    base::HSDState{T}, floor::T=sqrt(eps(T)),
) where {T}
    hsd_residual!(base)
    (isfinite(base.tau) && base.tau > floor) || return T(Inf)
    recovered = hsd_normalized_residual(base) / base.tau
    return isfinite(recovered) ? recovered : T(Inf)
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
    normalized_residual = status === ProductHSDOptimal ?
        _product_hsd_recovered_residual(base) : hsd_normalized_residual(base)
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
        state.tau_collapse_recoveries,
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
    base = state.base
    canonical = base.canonical
    # Do not send an arithmetically unresolved tau to the recovered-point
    # verifier. Ray gates below still run and remain the only infeasibility
    # authority. Healthy tau follows the unchanged verifier path.
    tau_floor = max(tol, sqrt(eps(T)))
    if check_optimal && base.tau > tau_floor && verify_optimal!(
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

"""
Return whether the current homogeneous state must be treated as a
τ-collapse ray candidate.  The setup-time floor
`max(tol, sqrt(eps(T)))` separates a resolvable positive τ from the numerical
zero regime.  Collapse additionally requires `κ/τ >= 1/floor` (with positive
κ and nonpositive τ treated as an infinite ratio).  These conditions only
trigger certificate verification; they never promote a status themselves.
"""
@inline function _product_hsd_tau_collapsed(base::HSDState{T}, tol::T) where {T}
    floor = max(tol, sqrt(eps(T)))
    isfinite(base.tau) && isfinite(base.kappa) || return false
    base.tau <= floor || return false
    base.kappa > zero(T) || return false
    return base.tau <= zero(T) || base.kappa >= base.tau / floor
end

"""
A collapse is recoverable only after the embedding equations and global
complementarity have converged to the arithmetic neighborhood.  This avoids
mistaking an ordinary early iterate with small tau for an infeasibility face.
The ratio predicate is shared with termination-ray handling; neither predicate
can publish a certificate status.
"""
@inline function _product_hsd_tau_collapse_ready(
    base::HSDState{T}, tol::T,
) where {T}
    _product_hsd_tau_collapsed(base, tol) || return false
    hsd_residual!(base)
    floor = max(tol, sqrt(eps(T)))
    residual_converged = hsd_normalized_residual(base) <= floor
    complementarity_converged = base.mu <=
        floor * max(one(T), abs(base.kappa))
    return residual_converged && complementarity_converged
end

"""
Restore one tau-collapsed trajectory to a centered KKT-derived interior point.
The sign audit found no family-specific border defect: the frozen gap row is
`-c'dx-b'dy+dκ=-rG` (gap coefficient `+1` on dκ) and the scalar row is
`κ*dτ+τ*dκ=scalar_rhs`; `_product_hsd_recover_dkappa!` evaluates candidates
against both equations. Mixed-sign orthants are already canonicalized to the
same nonnegative pairing, and PSD svec uses the Euclidean trace pairing.

The cone cross-centering pass in `kkt_derived_start!` changes the observed
global mu after it initially sets kappa=1.  One scalar cross-centering update,
`kappa <- mu/tau`, balances the scalar pair with that observed global mu and
changes the projective drive which selected the tau=0 face.  Recovery is
bounded by the solve loop and the attempt count is retained in the result.
"""
function _product_hsd_tau_collapse_recenter!(
    state::ProductConeHSDState{T},
) where {T}
    report = kkt_derived_start!(state)
    report.ok || begin
        state.diagnostic = :tau_collapse_recenter_initialization_failed
        return false
    end
    base = state.base
    hsd_residual!(base)
    scalar_center = base.mu / base.tau
    (isfinite(scalar_center) && scalar_center > zero(T)) || begin
        state.diagnostic = :tau_collapse_recenter_nonfinite
        return false
    end
    base.kappa = scalar_center
    hsd_residual!(base)
    (isfinite(base.mu) && base.mu > zero(T)) || begin
        state.diagnostic = :tau_collapse_recenter_nonfinite
        return false
    end
    state.tau_collapse_recoveries += 1
    state.diagnostic = :tau_collapse_recentered
    return true
end

@inline function _product_hsd_finite_ray_candidate(x::AbstractVector{T}) where {T}
    nonzero = false
    @inbounds for value in x
        isfinite(value) || return false
        nonzero |= !iszero(value)
    end
    return nonzero
end

"""
Run the authoritative original-coordinate dual-infeasibility verifier before
returning a terminal non-certificate status.  A collapsed τ/κ state and every
finite nonzero primal-ray candidate are both checked.  The requested failure,
time, or iteration status is preserved unless the unchanged verifier accepts
the ray; no HSD ratio or internal residual can promote DualInfeasible.
"""
function _product_hsd_termination_or_dual_ray!(
    state::ProductConeHSDState{T},
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    tol::T,
    status::ProductHSDSolveStatus,
    reason::ProductHSDSolveReason,
    last_step::HSDStepCode,
) where {T}
    base = state.base
    candidate = _product_hsd_tau_collapsed(base, tol) ||
                _product_hsd_finite_ray_candidate(base.x)
    if candidate && verify_dual_infeasibility!(
        base.canonical, base, x_original, s_original; tol=tol,
    )
        return _product_hsd_make_result(
            state, ProductHSDDualInfeasible,
            ProductHSDVerifiedTerminationRay, last_step, zero(T),
            x_original, s_original, y_original,
        )
    end
    return _product_hsd_make_result(
        state, status, reason, last_step, zero(T),
        x_original, s_original, y_original,
    )
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
    recovered_floor = max(tol, sqrt(eps(T)))
    recovered_residual = _product_hsd_recovered_residual(base, recovered_floor)
    isfinite(recovered_residual) || return nothing
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
    # The unchanged accepted HSD point is always the first certificate
    # candidate, including Exp/Power products. Refinement is a recovery step,
    # never a prerequisite for invoking the authoritative verifier.
    direct = _product_hsd_verified_result(
        state, x_original, s_original, y_original, tol, reason, last_step,
        terminal_alpha; check_optimal=true,
    )
    direct === nothing || return direct
    return _product_hsd_refined_optimal_result!(
        state, x_original, s_original, y_original, tol, reason, last_step,
        terminal_alpha,
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
@inline function _product_hsd_finish_terminal_restore!(state, result, restored::Bool)
    if !restored
        state.runtime.valid = false
        state.diagnostic = :post_result_state_restore_failed
    end
    return result
end

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
    # `_product_hsd_make_result` has copied every verified coordinate. A
    # failure to restore this reusable mutable runtime cannot revoke that
    # immutable result; retain it and invalidate only the abandoned state.
    return _product_hsd_finish_terminal_restore!(state, result, restored)
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
    max_tau_collapse_recoveries::Integer=1,
) where {T}
    initialization in (:auto, :identity, :kkt) || throw(ArgumentError(
        "initialization must be :auto, :identity, or :kkt",
    ))
    max_iterations >= 0 || throw(ArgumentError(
        "max_iterations must be nonnegative, got $max_iterations",
    ))
    max_tau_collapse_recoveries >= 0 || throw(ArgumentError(
        "max_tau_collapse_recoveries must be nonnegative",
    ))
    time_limit = Float64(max_time)
    (isfinite(time_limit) || isinf(time_limit)) && time_limit >= 0.0 ||
        throw(ArgumentError("max_time must be nonnegative and finite, or Inf"))
    started_ns = time_ns()
    certificate_tol = tol === nothing ? T(default_certificate_tol(T)) : tol
    (isfinite(certificate_tol) && certificate_tol > zero(T)) ||
        throw(ArgumentError("tol must be finite and positive"))

    base = state.base
    state.tau_collapse_recoveries = 0
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
        start_report.ok || return _product_hsd_termination_or_dual_ray!(
            state, x_original, s_original, y_original, certificate_tol,
            ProductHSDBreakdown, ProductHSDKKTInitializationFailed,
            HSDStepDirectionFailed,
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
            return _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDTimeLimit, ProductHSDTimeLimitReached, HSDStepOK,
            )
        end
        if _product_hsd_tau_collapse_ready(base, certificate_tol)
            # The preceding accepted-point candidate gate already checked all
            # three certificate classes. Re-run the ray-only gates explicitly
            # before numerical recovery so a genuine infeasibility face is
            # never redirected into an optimal-face restoration.
            ray = _product_hsd_verified_result(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDVerifiedTerminationRay, HSDStepOK;
                check_optimal=false,
            )
            ray === nothing || return ray
            if state.tau_collapse_recoveries < max_tau_collapse_recoveries &&
               _product_hsd_tau_collapse_recenter!(state)
                continue
            end
            return _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDInsufficientPrecision,
                ProductHSDTauCollapseRecoveryExhausted, HSDStepOK,
            )
        end
        code = product_hsd_step!(state)
        if code === HSDStepSingularKKT
            return _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDSingular, ProductHSDSingularKKTReason, code,
            )
        elseif code === HSDStepBreakdown
            terminal = _product_hsd_terminal_verified_result!(
                state, x_original, s_original, y_original, certificate_tol,
                code,
            )
            terminal === nothing || return terminal
            return _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDBreakdown, ProductHSDLineSearchBreakdown, code,
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
            return _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDBreakdown, ProductHSDDirectionBreakdown, code,
            )
        elseif code === HSDStepAlreadyOptimal
            verified = _product_hsd_candidate_result!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDVerifiedAcceptedStep, code,
            )
            verified === nothing || return verified
            return _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDBreakdown,
                ProductHSDUnverifiedZeroComplementarity, code,
            )
        end

        verified = _product_hsd_candidate_result!(
            state, x_original, s_original, y_original, certificate_tol,
            ProductHSDVerifiedAcceptedStep, code,
        )
        verified === nothing || return verified
    end

    if _product_hsd_tau_collapse_ready(base, certificate_tol)
        ray = _product_hsd_verified_result(
            state, x_original, s_original, y_original, certificate_tol,
            ProductHSDVerifiedTerminationRay, HSDStepOK;
            check_optimal=false,
        )
        ray === nothing || return ray
        return _product_hsd_termination_or_dual_ray!(
            state, x_original, s_original, y_original, certificate_tol,
            ProductHSDInsufficientPrecision,
            ProductHSDTauCollapseRecoveryExhausted, HSDStepOK,
        )
    end
    return _product_hsd_termination_or_dual_ray!(
        state, x_original, s_original, y_original, certificate_tol,
        ProductHSDMaxIterations, ProductHSDIterationLimitReached, HSDStepOK,
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
