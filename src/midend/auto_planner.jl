"""
    AutoPlanner

Stateless boundary for the execution planner. The authoritative route is
resolved after equality presolve by `resolve_execution_route` and frozen into
an `ExecutionPlan` by `build_execution_plan`; this type carries no runtime or
policy state of its own.
"""
struct AutoPlanner end

const _EXECUTION_ROUTE_TOKEN = Ref{Nothing}(nothing)

"""
    ResolvedExecutionRoute

The post-presolve route resolved from the mature value-level planner. This is
intentionally narrower than `ExecutionPlan`: scaling, parameter profiles,
backends, and resource decisions remain late-bound by the execution-plan
builder (and by the equilibrated SDP core where required).  `problem` and
`options` are borrowed so a route cannot accidentally be reused for a stale
presolve result or a different solver configuration.
"""
struct ResolvedExecutionRoute{T}
    problem::SDPProblem{T}
    options::SolverOptions{T}
    classification::ProblemClassification
    equality_evidence::EqualityPlanningEvidence
    algorithm::Symbol
    provenance::Symbol

    function ResolvedExecutionRoute(
        problem::SDPProblem{T},
        options::SolverOptions{T},
        classification::ProblemClassification,
        equality_evidence::EqualityPlanningEvidence,
        algorithm::Symbol,
        provenance::Symbol,
        token,
    ) where {T}
        token === _EXECUTION_ROUTE_TOKEN || throw(ArgumentError(
            "resolved execution routes are created only by the planner",
        ))
        return new{T}(
            problem,
            options,
            classification,
            equality_evidence,
            algorithm,
            provenance,
        )
    end
end
