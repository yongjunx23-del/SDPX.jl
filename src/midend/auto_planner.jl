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
