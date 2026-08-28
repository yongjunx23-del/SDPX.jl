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

# ---------------------------------------------------------------------------
# Calibrated route planner scaffolding (GPTPro Phase 8 planning layer).
#
# This is scaffolding, not policy. It adds the typed feature vector, the
# versioned calibration model/receipt, conservative memory upper-bound
# eligibility, residual/prediction-interval eligibility, and a route
# result/regret recording schema. It never changes the default route: the
# capability-and-risk decision from `plan_formulation` remains authoritative,
# and a calibration model that is absent, unversioned, provider-mismatched,
# or interval-unsupported can never claim a performance threshold.
# ---------------------------------------------------------------------------

"""Version of the `RouteCalibrationFeatures` field layout."""
const ROUTE_CALIBRATION_FEATURE_VERSION = 1
"""Version of the `RouteCalibrationModel` coefficient layout."""
const ROUTE_CALIBRATION_MODEL_VERSION = 1
"""Version of the `RouteCalibrationReceipt` field layout."""
const ROUTE_CALIBRATION_RECEIPT_VERSION = 1
"""Version of the `RouteResultRecord` recording schema."""
const ROUTE_RESULT_SCHEMA_VERSION = 1

"""
    RouteCalibrationFeatures{T}

Typed feature vector for the calibrated dense-formulation route comparison.
Fields follow the planner's vocabulary: `variables` (`n`) and `equalities`
(`m`) are the post-presolve dimensions, `rank` is the verified equality rank
when the equality-basis evidence is available, `normal_dimension` and
`expanded_dimension` are the two KKT system dimensions, and
`reduced_dimension` is the equality-eliminated variable count when the rank
is known. `input_nnz` counts nonzero coefficient entries (`A`), `kkt_nnz`
counts nonzero KKT/Schur entries (`K`), and `predicted_fill` is the
structural worst-case factor fill ratio (a dense lower-triangular factor over
the K nonzero count) — never a guessed central fill estimate.

`condition_spread`/`rrqr_quality` are the equality scale-spread and RRQR
basis-quality facts produced by the equality analysis; they are descriptive
facts, not decisions. `current_rss_bytes` and `memory_limit_bytes` feed the
conservative memory gate; `provider` and `thread_count` are the
LA-provider/thread facts that scope any calibration model to the machinery it
measured. A `nothing` field means the fact was not recorded, never a measured
zero.
"""
struct RouteCalibrationFeatures{T}
    variables::Int
    equalities::Int
    rank::Union{Nothing,Int}
    normal_dimension::Int
    reduced_dimension::Union{Nothing,Int}
    expanded_dimension::Int
    input_nnz::Union{Nothing,Int}
    kkt_nnz::Union{Nothing,Int}
    predicted_fill::Union{Nothing,Float64}
    precision_bits::Int
    condition_spread::Union{Nothing,T}
    rrqr_quality::Union{Nothing,Float64}
    current_rss_bytes::Union{Nothing,Int}
    memory_limit_bytes::Union{Nothing,Int}
    provider::Union{Nothing,Symbol}
    thread_count::Int
end

"""
    route_calibration_features(features, evidence; kwargs...) -> RouteCalibrationFeatures

Build the typed feature vector from the already-computed dense formulation
features and equality evidence. `input_nnz`/`kkt_nnz` are the coefficient
(`A`) and KKT/Schur (`K`) nonzero counts recorded by the structural analysis
when the caller has them; `predicted_fill` defaults to the conservative dense
worst-case factor fill ratio (`n(n+1)/2 / nnz(K)`), an upper bound that needs
no calibration to be true. `current_rss_bytes`, `memory_limit_bytes`,
`provider`, and `thread_count` are recorded as given; `nothing` stays
`nothing` (unknown, never zero).
"""
function route_calibration_features(
    features::DenseFormulationFeatures{T},
    evidence::EqualityPlanningEvidence;
    input_nnz::Union{Nothing,Integer}=nothing,
    kkt_nnz::Union{Nothing,Integer}=nothing,
    current_rss_bytes::Union{Nothing,Integer}=nothing,
    memory_limit_bytes::Union{Nothing,Integer}=nothing,
    provider::Union{Nothing,Symbol}=nothing,
    thread_count::Integer=1,
) where {T}
    rank = evidence.available ? evidence.rank_after : nothing
    reduced_dimension = rank === nothing ? nothing :
                        features.variables - (features.equalities - rank)
    fill_ratio = if kkt_nnz === nothing || kkt_nnz <= 0
        nothing
    else
        worst_case_factor = features.variables * (features.variables + 1) ÷ 2
        Float64(worst_case_factor) / Float64(kkt_nnz)
    end
    return RouteCalibrationFeatures{T}(
        features.variables,
        features.equalities,
        rank,
        features.normal_dimension,
        reduced_dimension,
        features.augmented_dimension,
        input_nnz === nothing ? nothing : Int(input_nnz),
        kkt_nnz === nothing ? nothing : Int(kkt_nnz),
        fill_ratio,
        T === BigFloat ? precision(BigFloat) : sig_bits(T),
        features.equality_scale_spread,
        evidence.available ? evidence.relative_rrqr_quality : nothing,
        current_rss_bytes === nothing ? nothing : Int(current_rss_bytes),
        memory_limit_bytes === nothing ? nothing : Int(memory_limit_bytes),
        provider,
        max(Int(thread_count), 1),
    )
end

"""
    route_calibration_features(problem, features, evidence; kwargs...)

Convenience construction that lifts the coefficient (`A`) and Schur (`K`)
nonzero counts from the problem's structural analysis.
"""
function route_calibration_features(
    problem::SDPProblem{T},
    features::DenseFormulationFeatures{T},
    evidence::EqualityPlanningEvidence;
    current_rss_bytes::Union{Nothing,Integer}=nothing,
    memory_limit_bytes::Union{Nothing,Integer}=nothing,
    provider::Union{Nothing,Symbol}=nothing,
    thread_count::Integer=1,
) where {T}
    return route_calibration_features(
        features,
        evidence;
        input_nnz=problem.structure.coefficient_nnz,
        kkt_nnz=problem.structure.schur_upper_nnz,
        current_rss_bytes=current_rss_bytes,
        memory_limit_bytes=memory_limit_bytes,
        provider=provider,
        thread_count=thread_count,
    )
end

"""
    RouteCalibrationModel

Versioned, provider-scoped linear model that predicts the executed-time ratio
`t_augmented / t_normal` from the typed feature vector. `coefficients` is a
NamedTuple whose names must be a subset of the computable feature map (see
`predict_route_performance`): `:intercept` plus model-chosen terms such as
`log_dimension_ratio`, `log_density`, `log_fill`, `log_kkt_nnz`,
`log_input_nnz`. A model whose feature version does not match the planner's,
whose provider does not match the solve's, or whose residual statistics are
not finite is invalid and is never applied. The model is calibration
*evidence*: nothing here is a policy threshold, and applying it can only ever
*report* an interval-supported preference, never silently change the default
route.
"""
struct RouteCalibrationModel
    model_version::Int
    feature_version::Int
    coefficients::NamedTuple
    residual_std::Float64
    fitted_samples::Int
    provider::Union{Nothing,Symbol}
    hardware_signature::String
    coverage::Float64
    source::Symbol
end

"""
    RouteCalibrationReceipt

Versioned provenance record of one calibration-model application (or of its
absence). `valid` records whether the model passed `valid_calibration_model`
for the specific feature vector, `source` is `:calibrated`, `:static`, or
`:unavailable`, and the remaining fields freeze the model identity, host,
provider scope, sample count, residual scale, and interval coverage. When no
model exists every field is `nothing`/zero and `valid=false`: an absent
receipt must never read as a measured threshold.
"""
struct RouteCalibrationReceipt
    schema_version::Int
    model_version::Union{Nothing,Int}
    feature_version::Union{Nothing,Int}
    hardware_signature::String
    provider::Union{Nothing,Symbol}
    fitted_samples::Int
    residual_std::Union{Nothing,Float64}
    coverage::Float64
    source::Symbol
    valid::Bool
end

"""
    RoutePrediction{T}

Output of applying one versioned calibration model to one feature vector: the
predicted `t_augmented / t_normal` ratio with a two-sided prediction interval
at the model's recorded coverage. `residual_eligible` is true only when the
entire interval supports one side of parity (`interval_high < 1` favors the
bordered/augmented route, `interval_low > 1` favors normal equations); an
interval straddling 1.0 means the residuals do not support a performance
claim. `preferred_route` is `nothing` unless the interval supports a claim.
`available=false` when no valid model exists; a missing prediction is never
replaced by a guessed number.
"""
struct RoutePrediction{T}
    available::Bool
    predicted_ratio::Union{Nothing,T}
    interval_low::Union{Nothing,T}
    interval_high::Union{Nothing,T}
    residual_eligible::Bool
    preferred_route::Union{Nothing,Symbol}
    reason::Symbol
end

"""
    MemoryUpperBoundEligibility

Result of the conservative memory gate for a calibrated route claim. The
upper bound is the workspace estimate (itself already an upper bound after
the `WORKSPACE_ESTIMATE_MARGIN_*` margin in `src/pipeline/workspace_estimate.jl`)
plus the current process peak RSS; the claim is eligible only when that sum
fits the recorded memory limit. `eligible=false` with a named reason whenever
any input is unknown, because an unknown cannot certify an upper bound. The
type lives with the calibrated planner; the bound arithmetic that produces it
lives in `src/pipeline/workspace_estimate.jl`.
"""
struct MemoryUpperBoundEligibility
    eligible::Bool
    reason::Symbol
    upper_bound_bytes::Int
    estimate_bytes::Int
    current_rss_bytes::Int
    limit_bytes::Int
end

"""
    CalibratedFormulationPlanning{T}

Result of the calibrated formulation planner scaffold. `capability_decision`
is the unchanged `plan_formulation` decision (the capability/risk policy),
`selected` mirrors its `selected`, and `calibrated_route_change_allowed`
reports whether the calibration evidence *would* justify switching the route
to `prediction.preferred_route` — it never performs that switch. `reason`
explains the evidence state. `features`, `prediction`, `receipt`, and
`memory` are the typed facts the performance trace and calibration data
collection read.
"""
struct CalibratedFormulationPlanning{T}
    capability_decision::FormulationDecision{T}
    features::RouteCalibrationFeatures{T}
    prediction::RoutePrediction{T}
    receipt::RouteCalibrationReceipt
    memory::MemoryUpperBoundEligibility
    reason::Symbol
    selected::Symbol
    calibrated_route_change_allowed::Bool
end

"""
    valid_calibration_model(model, features) -> Bool

Whether a calibration model may be applied to a feature vector. All checks
are fail-closed: feature-schema compatibility, a finite non-negative residual
scale, at least one fitted sample, a coverage in `(0, 1]`, an `intercept`
coefficient, a non-empty host signature, and a provider fact on *both* sides
that matches. A model without provider evidence (or a solve without provider
evidence) cannot claim a performance threshold.
"""
function valid_calibration_model(
    model::RouteCalibrationModel,
    features::RouteCalibrationFeatures,
)
    model.feature_version == ROUTE_CALIBRATION_FEATURE_VERSION || return false
    model.provider !== nothing || return false
    features.provider === model.provider || return false
    isfinite(model.residual_std) && model.residual_std >= 0 || return false
    model.fitted_samples >= 1 || return false
    isfinite(model.coverage) && 0 < model.coverage <= 1 || return false
    isempty(model.hardware_signature) && return false
    haskey(model.coefficients, :intercept) || return false
    isfinite(model.coefficients.intercept) || return false
    return true
end

"""
    calibration_receipt(model, features) -> RouteCalibrationReceipt

Versioned receipt for a calibration model applied to a feature vector, or for
the absence of a model. `valid` is `valid_calibration_model(model, features)`
when a model exists and `false` otherwise.
"""
function calibration_receipt(
    model::Union{Nothing,RouteCalibrationModel},
    features::RouteCalibrationFeatures,
)
    model === nothing && return RouteCalibrationReceipt(
        ROUTE_CALIBRATION_RECEIPT_VERSION,
        nothing,
        nothing,
        "",
        nothing,
        0,
        nothing,
        0.0,
        :unavailable,
        false,
    )
    return RouteCalibrationReceipt(
        ROUTE_CALIBRATION_RECEIPT_VERSION,
        model.model_version,
        model.feature_version,
        model.hardware_signature,
        model.provider,
        model.fitted_samples,
        model.residual_std,
        model.coverage,
        model.source,
        valid_calibration_model(model, features),
    )
end

"""Computable numeric feature map for one feature vector, as `Float64`."""
function _model_feature_map(features::RouteCalibrationFeatures{T}) where {T}
    feature_map = Dict{Symbol,Float64}()
    normal = features.normal_dimension
    expanded = features.expanded_dimension
    if normal > 0 && expanded > 0
        feature_map[:log_dimension_ratio] =
            log(Float64(expanded) / Float64(normal))
    end
    if features.input_nnz !== nothing &&
       features.equalities > 0 &&
       features.variables > 0
        density = Float64(features.input_nnz) /
                  (Float64(features.equalities) * Float64(features.variables))
        feature_map[:log_density] = log(max(density, eps(Float64)))
    end
    if features.predicted_fill !== nothing &&
       isfinite(features.predicted_fill) &&
       features.predicted_fill > 0
        feature_map[:log_fill] = log(features.predicted_fill)
    end
    if features.kkt_nnz !== nothing && features.kkt_nnz > 0
        feature_map[:log_kkt_nnz] = log(Float64(features.kkt_nnz))
    end
    if features.input_nnz !== nothing && features.input_nnz > 0
        feature_map[:log_input_nnz] = log(Float64(features.input_nnz))
    end
    return feature_map
end

"""
    _normal_inverse_cdf(p) -> Float64

Inverse standard-normal CDF at `0 < p < 1` via the Acklam rational
approximation. Numerical machinery, not a policy threshold; used only to turn
the calibration model's recorded interval coverage into a two-sided
prediction-interval scale.
"""
function _normal_inverse_cdf(p::Real)
    p = Float64(p)
    0 < p < 1 || throw(ArgumentError(
        "normal quantile requires 0 < p < 1, got $p",
    ))
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    if p < 0.02425
        q = sqrt(-2 * log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    elseif p > 0.97575
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    else
        q = p - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
               (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
    end
end

"""
    _normal_two_sided_quantile(coverage) -> Float64

Two-sided standard-normal quantile for a prediction interval at the model's
recorded coverage. The coverage itself is calibration evidence carried by the
model receipt; this function only inverts it.
"""
@inline function _normal_two_sided_quantile(coverage::Real)
    tail = (1 - Float64(coverage)) / 2
    return _normal_inverse_cdf(1 - tail)
end

"""
    predict_route_performance(model, features) -> RoutePrediction

Apply a versioned calibration model to the typed feature vector, returning
the predicted `t_augmented / t_normal` ratio with a two-sided prediction
interval at the model's recorded coverage. The interval uses the standard
new-observation scale `residual_std * sqrt(1 + 1/n)` with the normal
quantile of the model's coverage; a coefficient naming a feature that cannot
be computed from this feature vector makes the prediction unavailable
(fail-closed), never partially applied.
"""
function predict_route_performance(
    model::RouteCalibrationModel,
    features::RouteCalibrationFeatures{T},
) where {T}
    invalid = RoutePrediction{T}(
        false, nothing, nothing, nothing, false, nothing,
        :invalid_calibration_model,
    )
    valid_calibration_model(model, features) || return invalid
    feature_map = _model_feature_map(features)
    predicted = Float64(model.coefficients.intercept)
    for (name, coefficient) in pairs(model.coefficients)
        name === :intercept && continue
        haskey(feature_map, name) || return RoutePrediction{T}(
            false, nothing, nothing, nothing, false, nothing,
            :model_feature_unavailable,
        )
        value = feature_map[name]
        (coefficient isa Real && isfinite(coefficient)) || return invalid
        predicted += Float64(coefficient) * value
    end
    isfinite(predicted) || return invalid
    ratio = exp(predicted)
    scale = model.residual_std *
            sqrt(1 + 1 / max(model.fitted_samples, 1))
    quantile = _normal_two_sided_quantile(model.coverage)
    interval_low = exp(predicted - quantile * scale)
    interval_high = exp(predicted + quantile * scale)
    preferred = if interval_high < 1.0
        :dense_augmented_kkt
    elseif interval_low > 1.0
        :dense_normal_equations
    else
        nothing
    end
    eligible = preferred !== nothing
    reason = preferred === :dense_augmented_kkt ?
             :interval_supports_augmented :
             preferred === :dense_normal_equations ?
             :interval_supports_normal : :interval_straddles_parity
    return RoutePrediction{T}(
        true,
        T(ratio),
        T(interval_low),
        T(interval_high),
        eligible,
        preferred,
        reason,
    )
end

"""
    plan_calibrated_formulation(dense, requested, equality_evidence, feasibility, features; model=nothing) -> CalibratedFormulationPlanning

Calibrated route planner scaffold. The capability/risk decision from
`plan_formulation` is computed unchanged and stays authoritative (`selected`
mirrors it — there is **no default route change**). On top of it this
function records the typed feature vector, applies a versioned calibration
model when one is supplied and valid, applies the conservative memory
upper-bound gate, and reports whether the calibration evidence would justify
a route change. Missing calibration or provider evidence selects the existing
capability-only policy with an explicit `:calibration_evidence_missing_*`
reason; the scaffold never claims a performance threshold it cannot
interval-support.
"""
function plan_calibrated_formulation(
    dense::DenseFormulationFeatures{T},
    requested::Symbol,
    equality_evidence::EqualityPlanningEvidence,
    feasibility::FormulationFeasibility,
    features::RouteCalibrationFeatures{T};
    model::Union{Nothing,RouteCalibrationModel}=nothing,
) where {T}
    capability = plan_formulation(
        dense,
        requested,
        equality_evidence,
        feasibility,
    )
    receipt = calibration_receipt(model, features)
    prediction = if receipt.valid
        predict_route_performance(model, features)
    else
        RoutePrediction{T}(
            false, nothing, nothing, nothing, false, nothing,
            model === nothing ? :no_calibration_model : :invalid_calibration_model,
        )
    end
    estimate = capability.selected === :dense_augmented_kkt ?
               feasibility.augmented_memory_bytes :
               feasibility.normal_memory_bytes
    memory = conservative_memory_upper_bound_eligibility(features, estimate)
    route_change_allowed = (
        receipt.valid &&
        prediction.available &&
        prediction.residual_eligible &&
        prediction.preferred_route !== capability.selected &&
        memory.eligible
    )
    reason = if !receipt.valid
        model === nothing ? :calibration_evidence_missing_capability_only :
        :calibration_evidence_invalid_capability_only
    elseif !prediction.available
        :calibration_prediction_unavailable_capability_only
    elseif !prediction.residual_eligible
        :calibration_prediction_not_interval_supported
    elseif !memory.eligible
        :calibration_route_memory_upper_bound_violated
    elseif prediction.preferred_route === capability.selected
        :calibration_supports_selected_route
    else
        :calibration_route_change_supported_not_applied
    end
    return CalibratedFormulationPlanning{T}(
        capability,
        features,
        prediction,
        receipt,
        memory,
        reason,
        capability.selected,
        route_change_allowed,
    )
end

"""
    planner_reason_facts(planning) -> NamedTuple

The planner reason facts recorded for one calibrated planning run: the typed
feature-vector summary, prediction facts, calibration receipt, memory gate,
and the decision reason. This is the schema the performance trace projects
and future calibration data collection consumes.
"""
function planner_reason_facts(planning::CalibratedFormulationPlanning{T}) where {T}
    features = planning.features
    prediction = planning.prediction
    receipt = planning.receipt
    memory = planning.memory
    return (
        features=(
            variables=features.variables,
            equalities=features.equalities,
            rank=features.rank,
            normal_dimension=features.normal_dimension,
            reduced_dimension=features.reduced_dimension,
            expanded_dimension=features.expanded_dimension,
            input_nnz=features.input_nnz,
            kkt_nnz=features.kkt_nnz,
            predicted_fill=features.predicted_fill,
            precision_bits=features.precision_bits,
            condition_spread=features.condition_spread,
            condition_spread_available=features.condition_spread isa Number,
            rrqr_quality=features.rrqr_quality,
            current_rss_bytes=features.current_rss_bytes,
            memory_limit_bytes=features.memory_limit_bytes,
            provider=features.provider,
            threads=features.thread_count,
        ),
        prediction=(
            available=prediction.available,
            predicted_ratio=prediction.predicted_ratio,
            interval_low=prediction.interval_low,
            interval_high=prediction.interval_high,
            residual_eligible=prediction.residual_eligible,
            preferred_route=prediction.preferred_route,
            reason=prediction.reason,
        ),
        receipt=(
            schema_version=receipt.schema_version,
            model_version=receipt.model_version,
            feature_version=receipt.feature_version,
            hardware_signature=receipt.hardware_signature,
            provider=receipt.provider,
            fitted_samples=receipt.fitted_samples,
            residual_std=receipt.residual_std,
            coverage=receipt.coverage,
            source=receipt.source,
            valid=receipt.valid,
        ),
        memory=(
            eligible=memory.eligible,
            reason=memory.reason,
            upper_bound_bytes=memory.upper_bound_bytes,
            estimate_bytes=memory.estimate_bytes,
            current_rss_bytes=memory.current_rss_bytes,
            limit_bytes=memory.limit_bytes,
        ),
        reason=planning.reason,
        selected=planning.selected,
        route_change_allowed=planning.calibrated_route_change_allowed,
    )
end

"""
    formulation_decision_summary(decision, planning)

Capability decision summary merged with the calibrated planner reason facts,
so a caller that runs the calibrated scaffold can record both layers under
the single `formulation_decision` provenance key. The capability summary is
byte-for-byte the one-argument form; calibration facts are added, never
overwritten.
"""
function formulation_decision_summary(
    decision::FormulationDecision,
    planning::CalibratedFormulationPlanning,
)
    summary = formulation_decision_summary(decision)
    return merge(summary, (calibration=planner_reason_facts(planning),))
end

"""
    RouteResultRecord{T}

One immutable calibration sample: the typed feature vector observed before
planning, the planned and executed routes, the executed wall-clock seconds,
the terminal status, the memory gate facts, and — when a counterfactual
measurement exists — the executed time of the alternative route and the
resulting regret (`elapsed_seconds - counterfactual_seconds`).
`regret_seconds` is `nothing` until a counterfactual is recorded: a missing
regret is not a zero regret. Diagnostics-only; records never feed back into
solver math.
"""
struct RouteResultRecord{T}
    schema_version::Int
    features::RouteCalibrationFeatures{T}
    planned_route::Symbol
    executed_route::Symbol
    elapsed_seconds::Float64
    status::Symbol
    termination_reason::Symbol
    memory_upper_bound_bytes::Int
    rss_bytes::Int
    regret_seconds::Union{Nothing,Float64}
    counterfactual_route::Union{Nothing,Symbol}

    function RouteResultRecord{T}(
        features::RouteCalibrationFeatures{T},
        planned_route::Symbol,
        executed_route::Symbol,
        elapsed_seconds::Real,
        status::Symbol,
        termination_reason::Symbol,
        memory_upper_bound_bytes::Integer,
        rss_bytes::Integer;
        regret_seconds::Union{Nothing,Real}=nothing,
        counterfactual_route::Union{Nothing,Symbol}=nothing,
    ) where {T}
        return new{T}(
            ROUTE_RESULT_SCHEMA_VERSION,
            features,
            planned_route,
            executed_route,
            Float64(elapsed_seconds),
            status,
            termination_reason,
            Int(memory_upper_bound_bytes),
            Int(rss_bytes),
            regret_seconds === nothing ? nothing : Float64(regret_seconds),
            counterfactual_route,
        )
    end
end

"""
    route_regret(record) -> Float64 | unavailable

The recorded route regret, or the `unavailable` marker when no counterfactual
measurement was recorded. A missing regret is never reported as zero.
"""
route_regret(record::RouteResultRecord) =
    record.regret_seconds === nothing ? unavailable : record.regret_seconds

"""
    RouteResultLog{T}

Append-only, versioned collection of `RouteResultRecord`s for calibration
data collection. Versioning is explicit so a future schema change cannot be
silently mixed with older records.
"""
mutable struct RouteResultLog{T}
    schema_version::Int
    records::Vector{RouteResultRecord{T}}
end

RouteResultLog{T}() where {T} =
    RouteResultLog{T}(ROUTE_RESULT_SCHEMA_VERSION, RouteResultRecord{T}[])

"""
    record_route_result!(log, record)

Append one route result record to the calibration log. The record's schema
version must match the log's; mismatched versions are rejected rather than
silently coerced.
"""
function record_route_result!(
    log::RouteResultLog{T},
    record::RouteResultRecord{T},
) where {T}
    record.schema_version == log.schema_version || throw(ArgumentError(
        "route result record schema version $(record.schema_version) does not " *
        "match the log schema version $(log.schema_version)",
    ))
    push!(log.records, record)
    return log
end

route_result_count(log::RouteResultLog) = length(log.records)
