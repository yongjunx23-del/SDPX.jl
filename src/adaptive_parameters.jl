#=====================================================================
    Guarded adaptive interior-point parameters

    β controls centering/complementarity reduction. γ controls the
    line-search contraction or fraction-to-boundary safety. Decisions
    are bounded, smoothed, recorded, and disabled after repeated
    numerical regressions.
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

function estimate_backtracking_count(step, gamma, step_rule::Symbol)
    step_rule === :backtrack || return 0
    (!isfinite(step) || step <= 0 || step >= 1 || gamma <= 0 || gamma >= 1) &&
        return 0
    return max(0, round(Int, log(Float64(step)) / log(Float64(gamma))))
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

        unstable = !isfinite(controller.beta) ||
                   !isfinite(controller.gamma) ||
                   !isfinite(reduction) ||
                   (iteration > 3 && reduction > T(2))
        controller.instability_count =
            unstable ? controller.instability_count + 1 :
            max(controller.instability_count - 1, 0)
        if controller.instability_count >= 2
            controller.beta = controller.default_beta
            controller.gamma = controller.default_gamma
            controller.fallback = true
        end
    end

    push!(
        controller.history,
        (
            iteration=iteration,
            beta=selected_beta,
            gamma=selected_gamma,
            predictor_quality=predictor_quality,
            complementarity_reduction=reduction,
            primal_step=primal_step,
            dual_step=dual_step,
            backtracking_count=backtracking_count,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            fallback=controller.fallback,
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
