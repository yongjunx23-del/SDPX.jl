"""
    FormulationCandidate

One mathematical KKT candidate plus execution-time feasibility facts supplied
by later planning layers. Candidate names never encode an LA provider.
"""
struct FormulationCandidate
    formulation::Symbol
    structurally_valid::Bool
    backend_feasible::Bool
    memory_feasible::Bool
    required_capabilities::Tuple{Vararg{Symbol}}
    system_dimension::Int
    estimated_memory_bytes::Int
    reason::Symbol
end

@inline formulation_candidate_feasible(candidate::FormulationCandidate) =
    candidate.structurally_valid &&
    candidate.backend_feasible &&
    candidate.memory_feasible

"""Backend/resource facts computed after the mathematical preference."""
struct FormulationFeasibility
    normal_backend::Bool
    augmented_backend::Bool
    normal_memory::Bool
    augmented_memory::Bool
    normal_memory_bytes::Int
    augmented_memory_bytes::Int
    augmented_reason::Symbol
end


FormulationFeasibility(
    normal_backend::Bool,
    augmented_backend::Bool,
    normal_memory::Bool,
    augmented_memory::Bool,
    normal_memory_bytes::Int,
    augmented_memory_bytes::Int,
) = FormulationFeasibility(
    normal_backend,
    augmented_backend,
    normal_memory,
    augmented_memory,
    normal_memory_bytes,
    augmented_memory_bytes,
    augmented_backend ? :available : :augmented_backend_capability_unavailable,
)

"""
    FormulationDecision{T}

Inspectable result of the static formulation planner. `preferred` is the
mathematical choice before backend/resource filtering; `selected` is the
feasible choice frozen into `ExecutionPlan` before numerical execution.
"""
struct FormulationDecision{T}
    requested::Symbol
    preferred::Symbol
    selected::Symbol
    reason::Symbol
    candidates::Tuple{FormulationCandidate,FormulationCandidate}
    supporting_features::DenseFormulationFeatures{T}
    equality_evidence::EqualityPlanningEvidence
    risk_indicators::NamedTuple
end

"""
    FormulationPlannerConfig

Internal numerical policy for the pure, static formulation planner.  This is
deliberately not a public solve option: providers and runtime measurements do
not participate in these thresholds, and the resulting decision is frozen
into `ExecutionPlan` before numerical execution.
"""
struct FormulationPlannerConfig
    scale_spread_threshold::Float64
    rrqr_quality_threshold::Float64
end

const DEFAULT_FORMULATION_PLANNER_CONFIG =
    FormulationPlannerConfig(1.0e8, 1.0e-8)

@inline function _formulation_risk_indicators(
    features::DenseFormulationFeatures,
    equality::EqualityPlanningEvidence,
    policy::FormulationPlannerConfig,
)
    scale_spread = try
        Float64(features.equality_scale_spread)
    catch exception
        exception isa InterruptException && rethrow()
        Inf
    end
    rrqr_quality = equality.available && equality.basis_verified ?
                   equality.relative_rrqr_quality : nothing
    large_equality_scale_spread =
        isfinite(scale_spread) ?
        scale_spread >= policy.scale_spread_threshold :
        scale_spread > 0
    poor_equality_quality =
        rrqr_quality !== nothing && isfinite(rrqr_quality) &&
        rrqr_quality <= policy.rrqr_quality_threshold
    strong_numerical_risk =
        equality.basis_verified &&
        (large_equality_scale_spread || poor_equality_quality)
    return (
        equality_scale_spread=scale_spread,
        equality_rrqr_quality=rrqr_quality,
        large_equality_scale_spread,
        poor_equality_quality,
        strong_numerical_risk,
        scale_spread_threshold=policy.scale_spread_threshold,
        rrqr_quality_threshold=policy.rrqr_quality_threshold,
    )
end

function _dense_formulation_candidates(
    features::DenseFormulationFeatures,
    feasibility::FormulationFeasibility,
)
    dense_valid = features.variables > 0
    augmented_valid = dense_valid
    normal = FormulationCandidate(
        :dense_normal_equations,
        dense_valid,
        feasibility.normal_backend,
        feasibility.normal_memory,
        (:cholesky, :factor_solve, :multi_rhs),
        features.normal_dimension,
        feasibility.normal_memory_bytes,
        !dense_valid ? :empty_newton_system :
        !feasibility.normal_backend ?
        :normal_equations_backend_capability_unavailable :
        !feasibility.normal_memory ? :normal_equations_memory_unavailable :
        :general_dense_route,
    )
    augmented = FormulationCandidate(
        :dense_augmented_kkt,
        augmented_valid,
        feasibility.augmented_backend,
        feasibility.augmented_memory,
        (:pivoted_symmetric_ldlt, :factor_solve, :multi_rhs),
        features.augmented_dimension,
        feasibility.augmented_memory_bytes,
        !dense_valid ? :empty_newton_system :
        !feasibility.augmented_backend ? feasibility.augmented_reason :
        !feasibility.augmented_memory ? :augmented_memory_unavailable :
        :general_dense_route,
    )
    return (normal, augmented)
end

"""
    plan_formulation(features, requested, equality_evidence, feasibility)

Pure first-generation dense formulation policy. It performs no factorization,
condition estimate, provider lookup, benchmark lookup, or runtime retry.
"""
function plan_formulation(
    dense::DenseFormulationFeatures{T},
    requested::Symbol,
    equality_evidence::EqualityPlanningEvidence,
    feasibility::FormulationFeasibility,
    ;
    policy::FormulationPlannerConfig=DEFAULT_FORMULATION_PLANNER_CONFIG,
) where {T}
    requested in (:auto, :primal, :normal_equations, :augmented) ||
        throw(ArgumentError(
            "dense formulation request must be :auto, :primal, " *
            ":normal_equations, or :augmented",
        ))
    candidates = _dense_formulation_candidates(dense, feasibility)
    normal, augmented = candidates
    risk = _formulation_risk_indicators(dense, equality_evidence, policy)

    if requested in (:primal, :normal_equations)
        formulation_candidate_feasible(normal) || throw(ArgumentError(
            "explicit normal-equations formulation is not feasible",
        ))
        return FormulationDecision(
            requested,
            :dense_normal_equations,
            :dense_normal_equations,
            :user_forced_normal,
            candidates,
            dense,
            equality_evidence,
            risk,
        )
    elseif requested === :augmented
        formulation_candidate_feasible(augmented) || throw(ArgumentError(
            "explicit augmented formulation is not feasible: $(augmented.reason)",
        ))
        return FormulationDecision(
            requested,
            :dense_augmented_kkt,
            :dense_augmented_kkt,
            :user_forced_augmented,
            candidates,
            dense,
            equality_evidence,
            risk,
        )
    end

    preferred = risk.strong_numerical_risk ?
                :dense_augmented_kkt : :dense_normal_equations
    if preferred === :dense_augmented_kkt &&
       formulation_candidate_feasible(augmented)
        reason = risk.large_equality_scale_spread ?
                 :large_equality_scale_spread : :poor_equality_quality
        return FormulationDecision(
            requested,
            preferred,
            preferred,
            reason,
            candidates,
            dense,
            equality_evidence,
            risk,
        )
    elseif formulation_candidate_feasible(normal)
        reason = preferred === :dense_augmented_kkt ?
                 (!augmented.backend_feasible ?
                  augmented.reason :
                  !augmented.memory_feasible ?
                  :augmented_memory_unavailable :
                  :augmented_not_feasible) :
                 !equality_evidence.basis_verified && dense.equalities > 0 ?
                 :equality_quality_unavailable :
                 :default_dense_normal_equations
        return FormulationDecision(
            requested,
            preferred,
            :dense_normal_equations,
            reason,
            candidates,
            dense,
            equality_evidence,
            risk,
        )
    elseif formulation_candidate_feasible(augmented)
        return FormulationDecision(
            requested,
            preferred,
            :dense_augmented_kkt,
            :normal_equations_not_feasible,
            candidates,
            dense,
            equality_evidence,
            risk,
        )
    end
    throw(ArgumentError("no feasible dense KKT formulation candidate"))
end

function formulation_decision_summary(decision::FormulationDecision)
    summarize(candidate) = (
        formulation=candidate.formulation,
        structurally_valid=candidate.structurally_valid,
        backend_feasible=candidate.backend_feasible,
        memory_feasible=candidate.memory_feasible,
        feasible=formulation_candidate_feasible(candidate),
        required_capabilities=candidate.required_capabilities,
        system_dimension=candidate.system_dimension,
        estimated_memory_bytes=candidate.estimated_memory_bytes,
        reason=candidate.reason,
    )
    equality = decision.equality_evidence
    equality_summary = (
        available=equality.available,
        basis_verified=equality.basis_verified,
        rank_before=equality.rank_before,
        rank_after=equality.rank_after,
        relative_rrqr_quality=equality.available ?
            equality.relative_rrqr_quality : nothing,
        reason=equality.reason,
    )
    return (
        requested=decision.requested,
        preferred=decision.preferred,
        selected=decision.selected,
        reason=decision.reason,
        candidates=map(summarize, decision.candidates),
        supporting_features=decision.supporting_features,
        equality_evidence=equality_summary,
        risk_indicators=decision.risk_indicators,
    )
end
