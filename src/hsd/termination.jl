# Typed terminal statuses and certificate-authoritative result construction.
# Extracted from product_cone_solve.jl without changing verification semantics.

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
    state::ProductConeHSDState{T}, floor::T=sqrt(eps(T)),
) where {T}
    base = state.base
    _product_hsd_residual!(state)
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
    _product_hsd_residual!(state)
    normalized_residual = status === ProductHSDOptimal ?
        _product_hsd_recovered_residual(state) : hsd_normalized_residual(base)
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
    optimal_probe_ready = true
    if state.symmetric_core isa FixedTraceQ3CoreWorkspace
        # Exact necessary condition from `verify_optimal!`: avoid scanning
        # every Q3 block while normalized complementarity is still too large.
        # This can only skip a verifier call that would certainly return false.
        normalized_mu = base.tau > tau_floor ?
            base.mu / (base.tau * base.tau) : T(Inf)
        mu_limit = tol * (one(T) + T(base.nu))
        optimal_probe_ready = isfinite(normalized_mu) &&
                              isfinite(mu_limit) &&
                              normalized_mu <= mu_limit
    end
    if check_optimal && optimal_probe_ready && base.tau > tau_floor && verify_optimal!(
        canonical, state.base, x_original, s_original, y_original; tol=tol,
    )
        return _product_hsd_make_result(
            state, ProductHSDOptimal, reason, last_step, terminal_alpha,
            x_original, s_original, y_original,
        )
    end
    ray_probe_ready = !(state.symmetric_core isa FixedTraceQ3CoreWorkspace) ||
                      _product_hsd_tau_collapsed(base, tol)
    if ray_probe_ready && verify_primal_infeasibility!(
        canonical, state.base, y_original; tol=tol,
    )
        return _product_hsd_make_result(
            state, ProductHSDPrimalInfeasible, reason, last_step,
            terminal_alpha, x_original, s_original, y_original,
        )
    end
    if ray_probe_ready && verify_dual_infeasibility!(
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
