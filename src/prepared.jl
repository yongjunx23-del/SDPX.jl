"""
    PreparedSolver(problem, options)

Sequential reusable solve session. Constraint matrices, sparsity metadata, and
the ingested model are retained while the objective may change between solves.
The session is deliberately non-reentrant: create one session per concurrent
worker. A successful previous solution can be used as the next warm start.
"""
mutable struct PreparedSolver{T}
    problem::SDPProblem{T}
    options::SolverOptions{T}
    previous::Union{Nothing,SDPResult{T}}
    solve_count::Int
    busy::Bool
    preparation_time::Float64
    fixed_trace::FixedTraceAnalysis{T}
end

"""
    prepare(problem, options=SolverOptions{T}()) -> PreparedSolver{T}

Create a non-reentrant sequential solve session for `problem`. The session
retains the ingested constraint representation and fixed-trace analysis, and
can reuse a successful result as the warm start for a later objective.
"""
function prepare(
    problem::SDPProblem{T},
    options::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    started = time()
    trace_analysis = analyze_fixed_trace(problem)
    return PreparedSolver{T}(
        problem,
        options,
        nothing,
        0,
        false,
        time() - started,
        trace_analysis,
    )
end

function _prepared_problem(
    prepared::PreparedSolver{T},
    objective,
) where {T}
    objective === nothing && return prepared.problem
    length(objective) == prepared.problem.dims.m || throw(DimensionMismatch(
        "the replacement objective must have $(prepared.problem.dims.m) entries",
    ))
    converted = _ingest_owned_array(T, objective)
    _check_finite(converted, "objective")
    problem = prepared.problem
    return SDPProblem{T}(
        converted,
        problem.C,
        problem.B,
        problem.b,
        problem.cons,
        problem.dims,
        problem.structure,
    )
end

function _prepared_warm_start(prepared::PreparedSolver, warm_start)
    warm_start === nothing && return nothing
    warm_start isa NamedTuple && return warm_start
    warm_start === :previous || throw(ArgumentError(
        "warm_start must be :previous, nothing, or a NamedTuple",
    ))
    previous = prepared.previous
    previous === nothing && return nothing
    previous.status in (Optimal, AlmostOptimal) || return nothing
    return (
        x0=previous.x,
        X0=previous.X,
        y0=previous.y,
        Y0=previous.Y,
    )
end

"""
    solve!(prepared; objective=nothing, warm_start=:previous)

Solve using a reusable session. Objective replacement reuses the complete
ingested constraint representation. A session rejects concurrent access rather
than racing on its retained previous result.
"""
function solve!(
    prepared::PreparedSolver{T};
    objective=nothing,
    warm_start=:previous,
) where {T}
    prepared.busy && throw(ArgumentError(
        "PreparedSolver is sequential; create a separate session for each concurrent solve",
    ))
    prepared.busy = true
    try
        problem = _prepared_problem(prepared, objective)
        start = _prepared_warm_start(prepared, warm_start)
        result = start === nothing ?
                 solve!(problem, prepared.options) :
                 solve!(problem, prepared.options; start...)
        prepared.previous = result
        prepared.solve_count += 1
        return result
    finally
        prepared.busy = false
    end
end
