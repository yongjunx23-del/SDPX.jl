#=====================================================================
    Interior-point parameter policies

    The policy layer is intentionally independent of the Newton and line-search
    implementations.  It consumes typed iteration diagnostics and returns one
    bounded set of parameters.  The fixed policy reproduces the historical
    trajectory.  The adaptive policy is a guarded Mehrotra-style controller and
    remains opt-in until it passes the solver-wide promotion benchmarks.
=====================================================================#

abstract type AbstractParameterPolicy end

"""
    FixedParameterPolicy{T}

The exact fixed-parameter behavior selected by `parameter_strategy=:fixed`.
`sigma` is the historical `β`; both fraction-to-boundary values and the
backtracking contraction are the historical `γ`.
"""
struct FixedParameterPolicy{T} <: AbstractParameterPolicy
    sigma::T
    primal_fraction_to_boundary::T
    dual_fraction_to_boundary::T
    backtracking_factor::T
    primal_regularization::T
    dual_regularization::T
    refinement_tolerance::T
    refinement_max_count::Int
end

"""
    AdaptiveParameterPolicy{T}

Safe bounds and controller settings for the opt-in adaptive policy.  All
numeric constants are converted to the solver arithmetic once, at policy
construction; selection never narrows a `MultiFloat` or `BigFloat` diagnostic
through `Float64`.
"""
struct AdaptiveParameterPolicy{T} <: AbstractParameterPolicy
    fallback::FixedParameterPolicy{T}
    sigma_min::T
    sigma_max::T
    fraction_min::T
    fraction_max::T
    backtracking_min::T
    backtracking_max::T
    centrality_floor::T
    severe_centrality_floor::T
    refinement_floor::T
    requested_tolerance::T
    instability_limit::Int
end

"""
    IterationDiagnostics{T}

Numerical state available after the affine predictor and before the corrector.
Residuals are normalized by the same data scales as termination.  `mu` and
`mu_aff` are global average complementarity values over the total cone
dimension.  A factorization quality of one is ideal; values near zero indicate
a small Cholesky-diagonal ratio.
"""
Base.@kwdef struct IterationDiagnostics{T}
    iteration::Int = 0
    primal_residual::T = T(Inf)
    dual_residual::T = T(Inf)
    relative_gap::T = T(Inf)
    mu::T = T(Inf)
    mu_aff::T = T(Inf)
    affine_primal_step::T = zero(T)
    affine_dual_step::T = zero(T)
    previous_primal_step::T = one(T)
    previous_dual_step::T = one(T)
    backtracking_count::Int = 0
    regularization::T = zero(T)
    refinement_count::Int = 0
    factorization_quality::T = one(T)
    predicted_residual_reduction::T = one(T)
    achieved_residual_reduction::T = one(T)
    primal_psd_margin::T = one(T)
    dual_psd_margin::T = one(T)
    precision_floor::Bool = false
end

"""
    IterationParameters{T}

Parameters selected for one corrector and accepted step.  SDP currently has a
single Schur regularization, so the primal/dual values are kept equal by the
built-in policies; separate fields make the policy interface usable by the LP
and future symmetric-indefinite KKT paths without another API change.
"""
Base.@kwdef struct IterationParameters{T}
    sigma::T
    primal_fraction_to_boundary::T
    dual_fraction_to_boundary::T
    backtracking_factor::T
    primal_regularization::T = zero(T)
    dual_regularization::T = zero(T)
    refinement_tolerance::T = zero(T)
    refinement_max_count::Int = 0
    centrality_target::T = sigma
    fallback::Bool = false
    fallback_reason::Symbol = :none
end

function FixedParameterPolicy(opts::SolverOptions{T}) where {T}
    tolerance = opts.refine_tol > zero(T) ?
                opts.refine_tol :
                T(REFINE_DEFAULT_TOL_ULPS) * eps(T)
    maximum_count = opts.refine_policy === :fixed ?
                    opts.refine_steps :
                    opts.refine_max_steps
    return FixedParameterPolicy{T}(
        opts.β,
        opts.γ,
        opts.γ,
        opts.γ,
        zero(T),
        zero(T),
        tolerance,
        maximum_count,
    )
end

function AdaptiveParameterPolicy(opts::SolverOptions{T}) where {T}
    fixed = FixedParameterPolicy(opts)
    requested = min(opts.ϵ_gap, opts.ϵ_primal, opts.ϵ_dual)
    refinement_floor = max(
        T(REFINE_DEFAULT_TOL_ULPS) * eps(T),
        requested > zero(T) ? requested * requested : zero(T),
    )
    return AdaptiveParameterPolicy{T}(
        fixed,
        min(T(1) / T(50), fixed.sigma),
        max(T(1) / T(2), fixed.sigma),
        min(T(4) / T(5), fixed.primal_fraction_to_boundary),
        max(T(99) / T(100), fixed.primal_fraction_to_boundary),
        min(T(11) / T(20), fixed.backtracking_factor),
        max(T(9) / T(10), fixed.backtracking_factor),
        T(2) / T(25),
        T(1) / T(5),
        refinement_floor,
        requested,
        2,
    )
end

function _fixed_iteration_parameters(
    policy::FixedParameterPolicy{T};
    fallback::Bool=false,
    fallback_reason::Symbol=:none,
) where {T}
    return IterationParameters{T}(
        sigma=policy.sigma,
        primal_fraction_to_boundary=policy.primal_fraction_to_boundary,
        dual_fraction_to_boundary=policy.dual_fraction_to_boundary,
        backtracking_factor=policy.backtracking_factor,
        primal_regularization=policy.primal_regularization,
        dual_regularization=policy.dual_regularization,
        refinement_tolerance=policy.refinement_tolerance,
        refinement_max_count=policy.refinement_max_count,
        centrality_target=policy.sigma,
        fallback=fallback,
        fallback_reason=fallback_reason,
    )
end

"""
    select_parameters(policy, diagnostics, history) -> IterationParameters

Pure parameter selection.  No solver state is mutated, making the controller
independently testable and deterministic for a supplied diagnostic history.
"""
function select_parameters(
    policy::FixedParameterPolicy,
    ::IterationDiagnostics,
    history,
)
    return _fixed_iteration_parameters(policy)
end

@inline function _history_value(history, name::Symbol, default)
    isempty(history) && return default
    row = last(history)
    return hasproperty(row, name) ? getproperty(row, name) : default
end

function _recent_instability_count(history)
    count = 0
    for row in Iterators.reverse(history)
        hasproperty(row, :unstable) && row.unstable || break
        count += 1
    end
    return count
end

function _finite_diagnostics(diagnostics::IterationDiagnostics)
    values = (
        diagnostics.primal_residual,
        diagnostics.dual_residual,
        diagnostics.mu,
        diagnostics.mu_aff,
        diagnostics.affine_primal_step,
        diagnostics.affine_dual_step,
        diagnostics.factorization_quality,
        diagnostics.primal_psd_margin,
        diagnostics.dual_psd_margin,
    )
    return all(isfinite, values) &&
           diagnostics.mu > zero(diagnostics.mu) &&
           diagnostics.mu_aff >= zero(diagnostics.mu_aff)
end

@inline function _adaptive_fraction(
    policy::AdaptiveParameterPolicy{T},
    affine_step::T,
    previous_step::T,
    backtracking_count::Int,
    affine_complementarity_ratio::T,
) where {T}
    candidate = if affine_step < T(1) / T(20)
        T(21) / T(25)
    elseif affine_step < T(1) / T(5)
        T(9) / T(10)
    elseif affine_step < T(7) / T(10)
        T(19) / T(20)
    else
        T(99) / T(100)
    end
    previous_step < T(1) / T(10) &&
        (candidate = min(candidate, T(9) / T(10)))
    backtracking_count >= 4 &&
        (candidate = min(candidate, T(22) / T(25)))
    affine_complementarity_ratio < T(1) / T(1_000_000) &&
        (candidate = min(candidate, T(9) / T(10)))
    return clamp(candidate, policy.fraction_min, policy.fraction_max)
end

function select_parameters(
    policy::AdaptiveParameterPolicy{T},
    diagnostics::IterationDiagnostics{T},
    history,
) where {T}
    if !_finite_diagnostics(diagnostics)
        return _fixed_iteration_parameters(
            policy.fallback;
            fallback=true,
            fallback_reason=:nonfinite_diagnostics,
        )
    end
    # A rank-revealing or otherwise degraded factorization invalidates the
    # affine quality estimate. Prefer the validated fixed controller rather
    # than adapting from a direction whose effective accuracy is unknown. The
    # Newton layer encodes an explicit rank-revealing fallback as zero quality;
    # ordinary small-but-positive Cholesky-diagonal ratios stay adaptive.
    if diagnostics.factorization_quality <= zero(T)
        return _fixed_iteration_parameters(
            policy.fallback;
            fallback=true,
            fallback_reason=:degraded_factorization,
        )
    end
    if _recent_instability_count(history) >= policy.instability_limit
        return _fixed_iteration_parameters(
            policy.fallback;
            fallback=true,
            fallback_reason=:repeated_instability,
        )
    end

    ratio = clamp(
        diagnostics.mu_aff / diagnostics.mu,
        zero(T),
        one(T),
    )
    affine_min = min(
        diagnostics.affine_primal_step,
        diagnostics.affine_dual_step,
    )
    # A good affine predictor earns the classical cubic rule.  When the affine
    # step is short, squaring is deliberately less aggressive and is combined
    # with a centrality floor.
    exponent = affine_min >= T(3) / T(4) ? 3 : 2
    sigma_candidate = ratio^exponent
    centrality_floor = if affine_min < T(1) / T(20) ||
                          diagnostics.achieved_residual_reduction >
                          T(5) / T(4) ||
                          diagnostics.factorization_quality < sqrt(eps(T))
        policy.severe_centrality_floor
    elseif affine_min < T(1) / T(4) ||
           diagnostics.backtracking_count >= 4 ||
           diagnostics.refinement_count >=
           max(policy.fallback.refinement_max_count - 1, 1)
        policy.centrality_floor
    else
        zero(T)
    end
    sigma_candidate = max(sigma_candidate, centrality_floor)
    # Infeasible cold starts can report an artificially optimistic affine
    # ratio.  Keep the first corrector at least as centered as the validated
    # fixed profile, but never above the general adaptive cap.
    diagnostics.iteration <= 1 &&
        (sigma_candidate = max(
            sigma_candidate,
            min(policy.fallback.sigma, policy.sigma_max),
        ))
    previous_sigma = _history_value(
        history,
        :sigma,
        policy.fallback.sigma,
    )
    sigma = clamp(
        T(4) / T(5) * sigma_candidate +
        T(1) / T(5) * previous_sigma,
        policy.sigma_min,
        policy.sigma_max,
    )

    primal_fraction = _adaptive_fraction(
        policy,
        diagnostics.affine_primal_step,
        diagnostics.previous_primal_step,
        diagnostics.backtracking_count,
        ratio,
    )
    dual_fraction = _adaptive_fraction(
        policy,
        diagnostics.affine_dual_step,
        diagnostics.previous_dual_step,
        diagnostics.backtracking_count,
        ratio,
    )

    backtracking = if diagnostics.backtracking_count >= 6 ||
                      diagnostics.achieved_residual_reduction >
                      T(5) / T(4)
        T(13) / T(20)
    elseif diagnostics.backtracking_count >= 3
        T(18) / T(25)
    elseif affine_min > T(4) / T(5)
        T(17) / T(20)
    else
        T(4) / T(5)
    end
    backtracking = clamp(
        backtracking,
        policy.backtracking_min,
        policy.backtracking_max,
    )

    # Keep the unregularized system as the first choice.  A nonzero value is a
    # carry-forward hint only after the factorization has already demonstrated
    # that regularization is necessary.
    regularization = diagnostics.regularization > zero(T) ?
                     max(diagnostics.regularization / T(10), sqrt(eps(T))) :
                     zero(T)
    refinement_tolerance = max(
        policy.refinement_floor,
        policy.fallback.refinement_tolerance,
    )
    refinement_max_count =
        diagnostics.factorization_quality < sqrt(eps(T)) ||
        diagnostics.refinement_count > 0 ?
        policy.fallback.refinement_max_count :
        min(policy.fallback.refinement_max_count, 2)

    return IterationParameters{T}(
        sigma=sigma,
        primal_fraction_to_boundary=primal_fraction,
        dual_fraction_to_boundary=dual_fraction,
        backtracking_factor=backtracking,
        primal_regularization=regularization,
        dual_regularization=regularization,
        refinement_tolerance=refinement_tolerance,
        refinement_max_count=refinement_max_count,
        centrality_target=sigma,
    )
end

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
    policy::Union{FixedParameterPolicy{T},AdaptiveParameterPolicy{T}}
    parameters::IterationParameters{T}
end

function AdaptiveIPMController(opts::SolverOptions{T}) where {T}
    opts.parameter_strategy in (:fixed, :adaptive) ||
        throw(ArgumentError("parameter_strategy must be :fixed or :adaptive"))
    policy = opts.parameter_strategy === :adaptive ?
             AdaptiveParameterPolicy(opts) :
             FixedParameterPolicy(opts)
    initial = _fixed_iteration_parameters(FixedParameterPolicy(opts))
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
        policy,
        initial,
    )
end

"""
    _safe_parameter_bounds(
        T,
        default_beta=T(1) / T(50),
        default_gamma=T(17) / T(20),
    )

Backward-compatible view of the built-in policy bounds.
"""
function _safe_parameter_bounds(
    ::Type{T},
    default_beta::T=T(1) / T(50),
    default_gamma::T=T(17) / T(20),
) where {T}
    return (
        beta_min=min(T(1) / T(50), default_beta),
        beta_max=max(T(1) / T(2), default_beta),
        gamma_min=min(T(13) / T(20), default_gamma),
        gamma_max=max(T(19) / T(20), default_gamma),
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
    catch exception
        _recoverable(exception) || rethrow()
        typemax(Int)
    end
    value >= upper && return typemax(Int)
    return try
        max(0, round(Int, value, rounding))
    catch exception
        _recoverable(exception) || rethrow()
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

function select_iteration_parameters!(
    controller::AdaptiveIPMController{T},
    diagnostics::IterationDiagnostics{T},
) where {T}
    parameters = controller.fallback ?
                 _fixed_iteration_parameters(
                     FixedParameterPolicy{T}(
                         controller.default_beta,
                         controller.default_gamma,
                         controller.default_gamma,
                         controller.default_gamma,
                         zero(T),
                         zero(T),
                         controller.parameters.refinement_tolerance,
                         controller.parameters.refinement_max_count,
                     );
                     fallback=true,
                     fallback_reason=:previous_instability,
                 ) :
                 select_parameters(
                     controller.policy,
                     diagnostics,
                     controller.history,
                 )
    controller.parameters = parameters
    controller.beta = parameters.sigma
    controller.gamma = min(
        parameters.primal_fraction_to_boundary,
        parameters.dual_fraction_to_boundary,
    )
    if parameters.fallback
        controller.fallback = true
        controller.beta = controller.default_beta
        controller.gamma = controller.default_gamma
    end
    return controller.parameters
end

function _outcome_is_unstable(
    iteration::Int,
    complementarity_reduction,
    feasibility_ratio,
    primal_step,
    dual_step,
)
    T = typeof(complementarity_reduction)
    nonfinite =
        !isfinite(complementarity_reduction) ||
        !isfinite(feasibility_ratio) ||
        !isfinite(primal_step) ||
        !isfinite(dual_step)
    complementarity_growth =
        iteration > 3 && complementarity_reduction > 2
    feasibility_growth =
        iteration > 3 && feasibility_ratio > 3
    minimum_step = min(primal_step, dual_step)
    repeated_tiny_step =
        iteration > 3 &&
        primal_step < T(1) / T(1_000_000) &&
        dual_step < T(1) / T(1_000_000) &&
        feasibility_ratio >= one(T)
    stalled_progress =
        iteration > 5 &&
        minimum_step < T(1) / T(50) &&
        feasibility_ratio > T(49) / T(50) &&
        complementarity_reduction > T(49) / T(50)
    reason =
        nonfinite ? :nonfinite_outcome :
        complementarity_growth ? :complementarity_growth :
        feasibility_growth ? :feasibility_growth :
        repeated_tiny_step ? :repeated_tiny_step :
        stalled_progress ? :stalled_progress :
        :none
    return reason !== :none, reason
end

"""
    record_and_update!(controller; ...)

Record the parameters actually used and the achieved iteration outcome.
Selection itself is performed by [`select_iteration_parameters!`](@ref) before
the corrector.  For source compatibility, callers that omit
`selected_parameters` get a synthetic diagnostic and select the next
iteration's values here.
"""
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
    primal_residual_after::T=
        primal_residual * abs(one(T) - primal_step),
    dual_residual_after::T=
        dual_residual * abs(one(T) - dual_step),
    backtracking_count::Int,
    affine_primal_step::T=primal_step,
    affine_dual_step::T=dual_step,
    mu_before::T=complementarity_before,
    mu_affine::T=predictor_quality * mu_before,
    relative_gap::T=T(Inf),
    regularization::T=zero(T),
    refinement_count::Int=0,
    refinement_residual::T=zero(T),
    factorization_quality::T=one(T),
    primal_psd_margin::T=one(T),
    dual_psd_margin::T=one(T),
    precision_floor::Bool=false,
    selected_parameters::Union{Nothing,IterationParameters{T}}=nothing,
) where {T}
    selected_beta = controller.beta
    selected_gamma = controller.gamma
    if selected_parameters === nothing
        previous_primal_step =
            _history_value(controller.history, :primal_step, one(T))
        previous_dual_step =
            _history_value(controller.history, :dual_step, one(T))
        diagnostics = IterationDiagnostics{T}(
            iteration=iteration,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            relative_gap=relative_gap,
            mu=mu_before,
            mu_aff=mu_affine,
            affine_primal_step=affine_primal_step,
            affine_dual_step=affine_dual_step,
            previous_primal_step=previous_primal_step,
            previous_dual_step=previous_dual_step,
            backtracking_count=backtracking_count,
            regularization=regularization,
            refinement_count=refinement_count,
            factorization_quality=factorization_quality,
            primal_psd_margin=primal_psd_margin,
            dual_psd_margin=dual_psd_margin,
            precision_floor=precision_floor,
        )
        selected_parameters =
            select_iteration_parameters!(controller, diagnostics)
    end

    reduction = complementarity_before > zero(T) ?
                complementarity_after / complementarity_before :
                one(T)
    feasibility = max(primal_residual, dual_residual)
    previous_feasibility = controller.previous_feasibility
    feasibility_ratio =
        isfinite(previous_feasibility) && previous_feasibility > zero(T) ?
        feasibility / previous_feasibility :
        one(T)
    unstable, fallback_reason = _outcome_is_unstable(
        iteration,
        reduction,
        feasibility_ratio,
        primal_step,
        dual_step,
    )

    if controller.strategy === :adaptive && !controller.fallback
        severe_instability =
            fallback_reason in (
                :nonfinite_outcome,
                :complementarity_growth,
                :feasibility_growth,
            )
        controller.instability_count = if severe_instability
            2
        elseif unstable
            controller.instability_count + 1
        else
            max(controller.instability_count - 1, 0)
        end
        if controller.instability_count >= 2
            controller.beta = controller.default_beta
            controller.gamma = controller.default_gamma
            controller.fallback = true
        end
    end
    selected_parameters.fallback &&
        (fallback_reason = selected_parameters.fallback_reason)
    controller.fallback && fallback_reason === :none &&
        (fallback_reason = :previous_instability)

    predicted_reduction = max(
        abs(one(T) - primal_step),
        abs(one(T) - dual_step),
    )
    residual_before = max(primal_residual, dual_residual, eps(T))
    achieved_reduction =
        max(primal_residual_after, dual_residual_after) /
        residual_before
    push!(
        controller.history,
        (
            iteration=iteration,
            beta=selected_beta,
            gamma=selected_gamma,
            beta_used=selected_beta,
            gamma_used=selected_gamma,
            sigma=selected_parameters.sigma,
            primal_fraction_to_boundary=
                selected_parameters.primal_fraction_to_boundary,
            dual_fraction_to_boundary=
                selected_parameters.dual_fraction_to_boundary,
            backtracking_factor=
                selected_parameters.backtracking_factor,
            centrality_target=selected_parameters.centrality_target,
            selected_primal_regularization=
                selected_parameters.primal_regularization,
            selected_dual_regularization=
                selected_parameters.dual_regularization,
            selected_refinement_tolerance=
                selected_parameters.refinement_tolerance,
            selected_refinement_max_count=
                selected_parameters.refinement_max_count,
            predictor_quality=predictor_quality,
            mu=mu_before,
            mu_aff=mu_affine,
            complementarity_before=complementarity_before,
            complementarity_after=complementarity_after,
            complementarity_reduction=reduction,
            feasibility_ratio=feasibility_ratio,
            predicted_residual_reduction=predicted_reduction,
            achieved_residual_reduction=achieved_reduction,
            affine_primal_step=affine_primal_step,
            affine_dual_step=affine_dual_step,
            primal_step=primal_step,
            dual_step=dual_step,
            backtracking_count=backtracking_count,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            primal_residual_after=primal_residual_after,
            dual_residual_after=dual_residual_after,
            relative_gap=relative_gap,
            regularization=regularization,
            refinement_count=refinement_count,
            refinement_residual=refinement_residual,
            factorization_quality=factorization_quality,
            primal_psd_margin=primal_psd_margin,
            dual_psd_margin=dual_psd_margin,
            precision_floor=precision_floor,
            unstable=unstable,
            instability_count=controller.instability_count,
            fallback=controller.fallback || selected_parameters.fallback,
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
        parameter_strategy=
            controller.fallback ? :fixed : controller.strategy,
    )
end
