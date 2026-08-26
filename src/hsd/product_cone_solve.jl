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
iterate.  `terminal_alpha > 0` records that case.
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
    return ProductHSDSolveResult{T}(
        status,
        reason,
        last_step,
        base.record.iterations,
        kkt_factor_count(base.driver),
        terminal_alpha,
        base.tau,
        base.kappa,
        base.mu,
        hsd_normalized_residual(base),
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
    terminal_alpha::T=zero(T),
) where {T}
    canonical = state.base.canonical
    if verify_optimal!(
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

@inline function _product_hsd_terminal_alpha(
    state::ProductConeHSDState{T}, tol::T,
) where {T}
    base = state.base
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

    result = _product_hsd_verified_result(
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
) where {T}
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

    product_hsd_cold_start!(state)
    initial = _product_hsd_verified_result(
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
            return _product_hsd_make_result(
                state, ProductHSDBreakdown, ProductHSDDirectionBreakdown,
                code, zero(T), x_original, s_original, y_original,
            )
        elseif code === HSDStepAlreadyOptimal
            verified = _product_hsd_verified_result(
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

        verified = _product_hsd_verified_result(
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
    canonical::CanonicalConicProgram{T}; kwargs...,
) where {T<:AbstractFloat}
    state = ProductConeHSDState(canonical)
    return product_hsd_solve!(state; kwargs...)
end
