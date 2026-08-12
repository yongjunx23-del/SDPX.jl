"""
    AutoPlanner

Stateless boundary for structural planning.  The first midend stage records
only the choices a caller made explicitly; it deliberately does not resolve
any `:auto` policy or inspect the runtime environment.
"""
struct AutoPlanner end

const _PLANNING_INTENT_FIELDS = (
    :algorithm,
    :presolve,
    :scaling,
    :sparse,
    :formulation,
    :chordal_decomposition,
    :equality_solver,
    :working_precision_policy,
    :threads,
)

"""Normalized, caller-supplied structural choices."""
struct StructuralPlanningIntent
    algorithm::Union{Nothing,Symbol}
    presolve::Union{Nothing,Bool}
    scaling::Union{Nothing,Symbol}
    sparse::Union{Nothing,Bool}
    formulation::Union{Nothing,Symbol}
    chordal_decomposition::Union{Nothing,Symbol}
    equality_solver::Union{Nothing,Symbol}
    working_precision_policy::Union{Nothing,Symbol}
    threads::Union{Nothing,Int}
end

@inline function _planning_auto(value)
    return value === :auto ||
           (value isa AbstractString && lowercase(strip(value)) == "auto")
end

function _planning_policy(
    value,
    allowed::Tuple,
    label::AbstractString;
    boolean_result::Bool=false,
)
    _planning_auto(value) && return nothing
    candidate = if value isa AbstractString
        Symbol(lowercase(strip(value)))
    elseif value isa Symbol || value isa Bool
        value
    else
        throw(ArgumentError("$label must be one of :auto, $(allowed)"))
    end
    candidate in allowed || throw(ArgumentError(
        "$label must be one of :auto, $(allowed), got $(repr(value))",
    ))
    if boolean_result
        candidate === :on && return true
        candidate === :off && return false
    end
    return candidate
end

function _planning_threads(value)
    _planning_auto(value) && return nothing
    result = if value isa Integer
        Int(value)
    elseif value isa AbstractString
        try
            parse(Int, strip(value))
        catch exception
            exception isa InterruptException && rethrow()
            throw(ArgumentError("threads must be an integer or :auto"))
        end
    else
        throw(ArgumentError("threads must be an integer or :auto"))
    end
    result >= 1 || throw(ArgumentError("threads must be at least one"))
    return result
end

"""
    StructuralPlanningIntent(; kwargs...)

Normalize the structural fields of `SolveOptions` without resolving them.  An
`auto` symbol (or case-insensitive string) is represented uniformly by
`nothing`; precision and numerical tolerances are intentionally not read.
"""
function StructuralPlanningIntent(
    ;
    algorithm=:auto,
    presolve=:auto,
    scaling=:auto,
    sparse=:auto,
    formulation=:auto,
    chordal_decomposition=:auto,
    equality_solver=:auto,
    working_precision_policy=:auto,
    threads=:auto,
)
    return StructuralPlanningIntent(
        _planning_policy(algorithm, (:lp, :socp, :sdp), "algorithm"),
        _planning_policy(
            presolve,
            (:on, :off, true, false),
            "presolve";
            boolean_result=true,
        ),
        _planning_policy(scaling, (:none, :equilibrate), "scaling"),
        _planning_policy(
            sparse,
            (:on, :off, true, false),
            "sparse";
            boolean_result=true,
        ),
        _planning_policy(formulation, (:primal, :dual), "formulation"),
        _planning_policy(
            chordal_decomposition,
            (:on, :off),
            "chordal_decomposition",
        ),
        _planning_policy(
            equality_solver,
            (:normal_equations, :qr),
            "equality_solver",
        ),
        _planning_policy(
            working_precision_policy,
            (:fixed,),
            "working_precision_policy",
        ),
        _planning_threads(threads),
    )
end

StructuralPlanningIntent(options::SolveOptions) = StructuralPlanningIntent(
    ;
    algorithm=options.algorithm,
    presolve=options.presolve,
    scaling=options.scaling,
    sparse=options.sparse,
    formulation=options.formulation,
    chordal_decomposition=options.chordal_decomposition,
    equality_solver=options.equality_solver,
    working_precision_policy=options.working_precision_policy,
    threads=options.threads,
)

"""Borrowed feature facts plus normalized structural intent."""
struct AutoPlannerSnapshot{T}
    features::ProblemFeatures{T}
    intent::StructuralPlanningIntent
end

Base.eltype(::AutoPlannerSnapshot{T}) where {T} = T

"""
    PlanningDecision

An inspection-only decision produced by the exact structural planner.  A
resolved decision always carries a value; a deferred decision never does.
`provenance` and `reason` are deliberately symbolic so summaries remain
stable, hash-free, and independent of runtime state.
"""
struct PlanningDecision{V}
    value::V
    status::Symbol
    provenance::Symbol
    reason::Symbol

    function PlanningDecision(
        value::V,
        status::Symbol,
        provenance::Symbol,
        reason::Symbol,
    ) where {V}
        V <: Union{Nothing,Symbol} || throw(ArgumentError(
            "planning decision values must be symbols or nothing",
        ))
        status in (:resolved, :deferred) || throw(ArgumentError(
            "planning decision status must be :resolved or :deferred",
        ))
        if status === :deferred
            value === nothing || throw(ArgumentError(
                "deferred planning decisions must not carry a value",
            ))
        else
            value === nothing && throw(ArgumentError(
                "resolved planning decisions must carry a value",
            ))
        end
        return new{V}(value, status, provenance, reason)
    end
end

@inline _resolved_decision(value::Symbol, provenance::Symbol, reason::Symbol) =
    PlanningDecision(value, :resolved, provenance, reason)

@inline _deferred_decision(provenance::Symbol, reason::Symbol) =
    PlanningDecision(nothing, :deferred, provenance, reason)

"""Fixed policy label for the pure exact structural planner."""
const _EXACT_PLANNER_POLICY = :exact_structural_v1

"""
    ResolvedAutoPlannerSnapshot

The exact planner borrows `snapshot` and records only decisions that follow
from its structural facts and explicit intent.  It intentionally does not
select a backend, inspect values or runtime resources, or call the execution
plan builder.
"""
struct ResolvedAutoPlannerSnapshot{T,A,S}
    snapshot::AutoPlannerSnapshot{T}
    policy::Symbol
    algorithm::PlanningDecision{A}
    scaling::PlanningDecision{S}
end

Base.eltype(::ResolvedAutoPlannerSnapshot{T,A,S}) where {T,A,S} = T

"""
    planner_snapshot(::AutoPlanner, features, options=SolveOptions())

Create a pure, allocation-light planning snapshot.  The feature object is
borrowed exactly as supplied; this function never copies model data.
"""
function planner_snapshot(
    ::AutoPlanner,
    features::ProblemFeatures{T},
    options::SolveOptions=SolveOptions(),
) where {T}
    return AutoPlannerSnapshot{T}(features, StructuralPlanningIntent(options))
end

"""Return unresolved structural choices in a fixed field order."""
function unresolved_options(snapshot::AutoPlannerSnapshot)
    intent = snapshot.intent
    values = ntuple(index -> getfield(intent, index), length(_PLANNING_INTENT_FIELDS))
    return Tuple(
        _PLANNING_INTENT_FIELDS[index]
        for index in eachindex(values)
        if values[index] === nothing
    )
end

"""Return a deterministic, hash-free summary of the planning intent."""
function planner_summary(snapshot::AutoPlannerSnapshot)
    intent = snapshot.intent
    return NamedTuple{
        _PLANNING_INTENT_FIELDS,
    }(ntuple(index -> getfield(intent, index), length(_PLANNING_INTENT_FIELDS)))
end

@inline function _feature_block_counts(features::ProblemFeatures)
    scalar_psd = count(cone -> cone.dimension == 1, features.psd_cones)
    psd2 = count(cone -> cone.dimension == 2, features.psd_cones)
    lorentz_psd2 = count(cone -> cone.dimension in (2, 3), features.lorentz_cones)
    large_lorentz = count(cone -> cone.dimension > 3, features.lorentz_cones)
    return (
        scalar=length(features.linear_cones) + scalar_psd,
        lorentz=length(features.lorentz_cones),
        psd=length(features.psd_cones),
        scalar_psd=scalar_psd,
        psd2=psd2,
        lorentz_psd2=lorentz_psd2,
        large_lorentz=large_lorentz,
    )
end

@inline function _exact_algorithm_decision(snapshot::AutoPlannerSnapshot)
    features = snapshot.features
    intent = snapshot.intent
    counts = _feature_block_counts(features)
    if intent.algorithm === :sdp
        return _resolved_decision(:sdp_primal_dual, :explicit, :explicit_sdp)
    elseif intent.algorithm === :lp
        counts.lorentz == 0 && counts.psd == counts.scalar_psd || return _deferred_decision(
            :explicit,
            :explicit_lp_incompatible,
        )
        return _resolved_decision(:lp_primal_dual, :explicit, :explicit_lp)
    end

    pure_scalar = counts.lorentz == 0 && counts.psd == counts.scalar_psd

    # The canonical linear family is exact LP evidence.  An explicit SOCP
    # request is not silently reinterpreted as LP, even when the model has no
    # Lorentz blocks; it remains a deferred formulation choice.
    if intent.algorithm === nothing && counts.lorentz == 0 && pure_scalar
        return _resolved_decision(:lp_primal_dual, :features, :pure_scalar_linear)
    end

    if counts.lorentz > 0 && counts.large_lorentz > 0 &&
       all(cone -> cone.dimension <= 2, features.psd_cones)
        return _resolved_decision(:socp_psd_lift, :features, :lorentz_dimension_gt3)
    end

    if counts.scalar > 0 &&
       counts.psd == counts.scalar_psd + counts.psd2 &&
       counts.lorentz == counts.lorentz_psd2 &&
       (counts.psd > counts.scalar_psd || counts.lorentz > 0)
        return _resolved_decision(:socp_psd2, :features, :scalar_and_small_psd)
    end

    all_q3 = counts.scalar == 0 &&
              (counts.psd2 + counts.lorentz_psd2) > 0 &&
              counts.psd2 == counts.psd &&
              counts.lorentz_psd2 == counts.lorentz
    if intent.algorithm in (nothing, :socp) && intent.scaling === :equilibrate &&
       counts.scalar == 0 && all_q3
        return _resolved_decision(
            :socp_psd2,
            :explicit,
            :equilibrate_disables_native_q3,
        )
    end

    # Native Q3 requires a scalar-free, all-Q3 structure.  Keep this branch
    # deferred because the remaining choice depends on policies outside the
    # exact structural layer (and, in particular, the current native route).
    return _deferred_decision(:features, :requires_deferred_route)
end

@inline function _exact_scaling_decision(
    snapshot::AutoPlannerSnapshot,
    algorithm::PlanningDecision,
)
    explicit = snapshot.intent.scaling
    explicit === :none && return _resolved_decision(:none, :explicit, :explicit_none)
    explicit === :equilibrate && return _resolved_decision(
        algorithm.status === :resolved && algorithm.value === :lp_primal_dual ?
        :lp_geometric : :sdp_ruiz,
        :explicit,
        :explicit_equilibrate,
    )
    algorithm.status === :resolved || return _deferred_decision(
        :policy,
        :scaling_waits_for_algorithm,
    )
    return _resolved_decision(
        algorithm.value === :lp_primal_dual ? :lp_geometric : :sdp_ruiz,
        :policy,
        algorithm.value === :lp_primal_dual ? :auto_lp : :auto_non_q3,
    )
end

"""
    resolve_planner_snapshot(snapshot)

Resolve only exact algorithm/scaling consequences of canonical structural
facts.  The result is inspection-only; it never constructs or mutates an
`ExecutionPlan` and leaves unsupported routes explicitly deferred.
"""
function resolve_planner_snapshot(
    ::AutoPlanner,
    snapshot::AutoPlannerSnapshot{T},
) where {T}
    algorithm = _exact_algorithm_decision(snapshot)
    scaling = _exact_scaling_decision(snapshot, algorithm)
    return ResolvedAutoPlannerSnapshot(
        snapshot,
        _EXACT_PLANNER_POLICY,
        algorithm,
        scaling,
    )
end

resolve_planner_snapshot(snapshot::AutoPlannerSnapshot) =
    resolve_planner_snapshot(AutoPlanner(), snapshot)

function _validate_planning_decision(decision::PlanningDecision)
    if decision.status === :deferred
        decision.value === nothing || throw(ArgumentError(
            "deferred decision invariant violated",
        ))
    elseif decision.status === :resolved
        decision.value isa Symbol || throw(ArgumentError(
            "resolved decision invariant violated",
        ))
    else
        throw(ArgumentError("unknown planning decision status"))
    end
    return true
end

"""Return a deterministic summary of the exact planner result."""
function resolved_planner_summary(result::ResolvedAutoPlannerSnapshot)
    _validate_planning_decision(result.algorithm)
    _validate_planning_decision(result.scaling)
    return (
        policy=result.policy,
        algorithm=(
            status=result.algorithm.status,
            value=result.algorithm.value,
            provenance=result.algorithm.provenance,
            reason=result.algorithm.reason,
        ),
        scaling=(
            status=result.scaling.status,
            value=result.scaling.value,
            provenance=result.scaling.provenance,
            reason=result.scaling.reason,
        ),
    )
end
