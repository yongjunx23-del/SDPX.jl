#=====================================================================
    Guarded adaptive interior-point parameters

    β controls centering/complementarity reduction. γ controls the
    line-search contraction or fraction-to-boundary safety. Decisions
    are bounded, smoothed, recorded, and disabled after repeated
    numerical regressions.

    `parameter_strategy` defaults to `:fixed`, and the benchmark that
    gates that default (plan Milestone H) is recorded here because the
    result is not the one the machinery was built expecting:

        problem            strategy   status    iters   rel. gap    secs
        SDP Task_Low08     fixed      Optimal      27   4.55e-07    59.2
        SDP Task_Low08     adaptive   Stalled      41   2.73e-05    93.1
        LP sparse          fixed      Optimal      13   3.75e-11   0.016
        LP sparse          adaptive   Optimal      13   3.13e-11   0.070

    On the SDP, adaptive loses on every axis at once — it converts an
    Optimal solve into a Stalled one, ends two orders of magnitude
    further from the requested tolerance, and takes longer to get there.
    On the LP it matches the fixed schedule iteration for iteration and
    so earns nothing. Milestone H's exit condition is that adaptive is
    default only where it passes the runtime, robustness, and accuracy
    gates; on the benchmarks in this repository it passes none of them,
    which is why the default stays `:fixed` and this stays opt-in.
=====================================================================#

mutable struct AdaptiveIPMController{T}
    strategy::Symbol
    default_beta::T
    default_gamma::T
    beta::T
    gamma::T
    previous_complementarity::T
    previous_feasibility::T
    instability_count::Int
    fallback::Bool
    history::Vector{NamedTuple}
end

function AdaptiveIPMController(opts::SolverOptions{T}) where {T}
    opts.parameter_strategy in (:fixed, :adaptive) ||
        throw(ArgumentError("parameter_strategy must be :fixed or :adaptive"))
    return AdaptiveIPMController{T}(
        opts.parameter_strategy,
        opts.β,
        opts.γ,
        opts.β,
        opts.γ,
        T(NaN),
        T(NaN),
        0,
        false,
        NamedTuple[],
    )
end

"""
    _safe_parameter_bounds(T, default_beta=T(0.02), default_gamma=T(0.85))

Guard rails for the adaptive β/γ search.

The floors have to admit the profile's own starting values. `recommended_parameters`
selects β=0.01 for large arrow-structured models — below the generic 0.02 floor —
and clamping that back up silently discards a deliberately chosen parameter, so
the floor is lowered to whatever the caller asked for. Same for γ at either end.
"""
function _safe_parameter_bounds(::Type{T}, default_beta::T=T(0.02),
                                default_gamma::T=T(0.85)) where {T}
    return (
        beta_min=min(T(0.02), default_beta),
        beta_max=max(T(0.50), default_beta),
        gamma_min=min(T(0.65), default_gamma),
        gamma_max=max(T(0.95), default_gamma),
    )
end

function _nonnegative_int_saturating(
    value,
    rounding::RoundingMode=RoundNearest,
)
    isnan(value) && return 0
    !isfinite(value) && return value > 0 ? typemax(Int) : 0
    value <= 0 && return 0
    upper = try
        oftype(value, typemax(Int))
    catch
        typemax(Int)
    end
    value >= upper && return typemax(Int)
    return try
        max(0, round(Int, value, rounding))
    catch
        typemax(Int)
    end
end

function estimate_backtracking_count(step, gamma, step_rule::Symbol)
    step_rule === :backtrack || return 0
    (!isfinite(step) || step <= 0 || step >= 1 || gamma <= 0 || gamma >= 1) &&
        return 0
    denominator = log(gamma)
    iszero(denominator) && return typemax(Int)
    ratio = log(step) / denominator
    return _nonnegative_int_saturating(ratio)
end

function record_and_update!(
    controller::AdaptiveIPMController{T};
    iteration::Int,
    predictor_quality::T,
    complementarity_before::T,
    complementarity_after::T,
    primal_residual::T,
    dual_residual::T,
    primal_step::T,
    dual_step::T,
    backtracking_count::Int,
) where {T}
    selected_beta = controller.beta
    selected_gamma = controller.gamma
    reduction = complementarity_before > zero(T) ?
                complementarity_after / complementarity_before : one(T)
    feasibility = max(primal_residual, dual_residual)
    previous_feasibility = controller.previous_feasibility
    feasibility_ratio =
        isfinite(previous_feasibility) && previous_feasibility > zero(T) ?
        feasibility / previous_feasibility : one(T)
    fallback_reason = :none

    if controller.strategy === :adaptive && !controller.fallback
        bounds = _safe_parameter_bounds(T, controller.default_beta, controller.default_gamma)
        quality = clamp(predictor_quality, zero(T), one(T))
        observed = clamp(reduction, zero(T), one(T))
        # Mehrotra's affine predictor contributes quality^3. Observed
        # complementarity reduction prevents an over-aggressive predictor from
        # repeatedly choosing a centering target the accepted steps cannot
        # realize.
        # Infeasible-start methods can increase complementarity sharply in the
        # first two steps while repairing feasibility. Do not interpret that
        # cold-start transient as failed centering.
        beta_candidate = iteration <= 2 ? controller.default_beta : clamp(
            max(quality^3, observed^2, bounds.beta_min),
            bounds.beta_min,
            bounds.beta_max,
        )
        controller.beta = clamp(
            T(0.55) * controller.beta + T(0.45) * beta_candidate,
            bounds.beta_min,
            bounds.beta_max,
        )

        minimum_step = min(primal_step, dual_step)
        gamma_candidate = controller.gamma
        if iteration <= 2
            gamma_candidate = controller.default_gamma
        elseif minimum_step < T(0.20) ||
               feasibility_ratio > T(1.50)
            gamma_candidate -= T(0.04)
        elseif minimum_step > T(0.95) &&
               feasibility_ratio < T(0.95)
            gamma_candidate += T(0.015)
        elseif backtracking_count >= 5 && minimum_step < T(0.50)
            gamma_candidate -= T(0.02)
        else
            gamma_candidate =
                T(0.90) * gamma_candidate + T(0.10) * controller.default_gamma
        end
        controller.gamma = clamp(
            gamma_candidate,
            bounds.gamma_min,
            bounds.gamma_max,
        )

        nonfinite_parameters =
            !isfinite(controller.beta) ||
            !isfinite(controller.gamma) ||
            !isfinite(reduction)
        complementarity_growth =
            iteration > 3 && reduction > T(2)
        feasibility_growth =
            iteration > 3 && feasibility_ratio > T(3)
        repeated_tiny_step =
            iteration > 3 &&
            primal_step < T(1e-6) &&
            dual_step < T(1e-6) &&
            feasibility_ratio >= one(T)
        # A one-sided cone boundary can stall while the other step remains
        # large. Requiring *both* steps to be tiny misses that failure mode:
        # beta drifts toward its upper bound, complementarity stops falling,
        # and feasibility changes by less than a percent per iteration. Two
        # consecutive observations are enough to restore the known-safe fixed
        # profile before the global stagnation detector terminates the solve.
        stalled_progress =
            iteration > 5 &&
            minimum_step < T(0.02) &&
            feasibility_ratio > T(0.98) &&
            reduction > T(0.98)
        unstable =
            nonfinite_parameters ||
            complementarity_growth ||
            feasibility_growth ||
            repeated_tiny_step ||
            stalled_progress
        fallback_reason =
            nonfinite_parameters ? :nonfinite_parameter :
            complementarity_growth ? :complementarity_growth :
            feasibility_growth ? :feasibility_growth :
            repeated_tiny_step ? :repeated_tiny_step :
            stalled_progress ? :stalled_progress :
            :none
        controller.instability_count =
            unstable ? controller.instability_count + 1 :
            max(controller.instability_count - 1, 0)
        if controller.instability_count >= 2
            controller.beta = controller.default_beta
            controller.gamma = controller.default_gamma
            controller.fallback = true
            fallback_reason =
                fallback_reason === :none ?
                :repeated_instability : fallback_reason
        end
    end
    controller.fallback && fallback_reason === :none &&
        (fallback_reason = :previous_instability)

    push!(
        controller.history,
        (
            iteration=iteration,
            beta=selected_beta,
            gamma=selected_gamma,
            beta_used=selected_beta,
            gamma_used=selected_gamma,
            predictor_quality=predictor_quality,
            complementarity_before=complementarity_before,
            complementarity_after=complementarity_after,
            complementarity_reduction=reduction,
            feasibility_ratio=feasibility_ratio,
            primal_step=primal_step,
            dual_step=dual_step,
            backtracking_count=backtracking_count,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            instability_count=controller.instability_count,
            fallback=controller.fallback,
            fallback_reason=fallback_reason,
        ),
    )
    controller.previous_complementarity = complementarity_after
    controller.previous_feasibility = feasibility
    return controller
end

function controller_options(
    options::SolverOptions{T},
    controller::AdaptiveIPMController{T},
) where {T}
    return _replace_solver_options(
        options;
        β=controller.beta,
        γ=controller.gamma,
    )
end
