@inline function _frontend_requested_bigfloat_bits(options::SolveOptions)
    value = options.precision
    if value isa Integer
        return Int(value)
    elseif value isa AbstractString && !isempty(strip(value)) && all(isdigit, strip(value))
        return parse(Int, strip(value))
    end
    return nothing
end

function _solve_sdp_with_frontend(problem::SDPProblem{T}, options::SolveOptions) where {T}
    frontend_started = time_ns()
    resolved = resolve_solve_options(T, options)
    frontend_seconds = (time_ns() - frontend_started) / 1.0e9
    # Native product-HSD production path: the all-auto frontend builds a
    # typed Model/Settings/Outputs and calls the public `optimize!` seam
    # (engine=:native_hsd) through the entrypoint bridge; the returned
    # result is adapted back to the legacy SDPResult schema in original
    # coordinates. No interior_point solve! is reachable from here.
    result = _bridge_sdp_solve(problem, resolved.core)
    return _with_frontend_timing(
        result,
        frontend_seconds,
        resolved.core.timing,
    )
end

"""
    solve(problem, options::SolveOptions)

Solve an already-ingested LP/SDP model through the small all-auto frontend.
The returned numerical result is unchanged; the resolved structural choices
remain available through `result.diagnostics.plan`.

For `SDPProblem{BigFloat}`, an integer `options.precision` changes the working
precision scope for the solve.  It cannot recreate digits that were already
rounded when the problem was constructed; construct high-precision input in
the requested precision scope or use the CLI, which parses input there.
"""
function solve(problem::SDPProblem{T}, options::SolveOptions) where {T}
    if T === BigFloat
        bits = _frontend_requested_bigfloat_bits(options)
        if bits !== nothing && Base.precision(BigFloat) != bits
            _require_bigfloat_precision_bits(bits, "precision")
            return setprecision(BigFloat, bits) do
                _solve_sdp_with_frontend(problem, options)
            end
        end
    end
    return _solve_sdp_with_frontend(problem, options)
end

function _solve_conic_with_frontend(
    problem::ConicProblem{T}, options::SolveOptions,
) where {T}
    resolved = resolve_solve_options(T, options)
    return _bridge_conic_solve(problem, resolved.core)
end

Base.@noinline function _solve_socp_keyword_dispatch(
    problem::ConicProblem{T}, sparse, verbosity, specialization, kwargs,
) where {T}
    Base.@nospecialize kwargs
    specialization === :auto || throw(ArgumentError(
        "the standalone NativeSOC specialization selector is retired; " *
        "product HSD accepts specialization=:auto only",
    ))
    options = haskey(kwargs, :timing) ?
        SolverOptions(T; verbosity, sparse, kwargs...) :
        SolverOptions(T; verbosity, sparse, timing=true, kwargs...)
    if T === BigFloat && Base.precision(BigFloat) != options.precision_bits
        bits = options.precision_bits
        _require_bigfloat_precision_bits(bits, "precision_bits")
        return setprecision(BigFloat, bits) do
            _bridge_conic_solve(problem, options)
        end
    end
    return _bridge_conic_solve(problem, options)
end

"""Solve a compact LP/SOC model through the product-cone HSD bridge."""
Base.@noinline function solve_socp(
    problem::ConicProblem{T};
    sparse=:auto,
    verbosity::Int=1,
    specialization::Symbol=:auto,
    kwargs...,
) where {T}
    Base.@nospecialize kwargs
    dispatch = Base.inferencebarrier(_solve_socp_keyword_dispatch)
    return Base.invokelatest(
        dispatch, problem, sparse, verbosity, specialization, kwargs,
    )
end

function solve_socp(problem::ConicProblem{T}, options::SolveOptions) where {T}
    if T === BigFloat
        bits = _frontend_requested_bigfloat_bits(options)
        if bits !== nothing && Base.precision(BigFloat) != bits
            _require_bigfloat_precision_bits(bits, "precision")
            return setprecision(BigFloat, bits) do
                _solve_conic_with_frontend(problem, options)
            end
        end
    end
    return _solve_conic_with_frontend(problem, options)
end

solve(problem::ConicProblem, options::SolveOptions) = solve_socp(problem, options)

"""
    solve_lp(c, G, h, options::SolveOptions; Aeq=nothing, beq=nothing, ...)

Convenience overload for the native LP frontend.  If an explicit high
precision is required, pass `T=BigFloat` while constructing the LP (or use the
CLI, which parses the input directly at the requested precision).
"""
function solve_lp(
    c::AbstractVector,
    G::AbstractMatrix,
    h::AbstractVector,
    options::SolveOptions;
    Aeq=nothing,
    beq=nothing,
    T::Union{Nothing,Type}=nothing,
    sparse::Union{Bool,Symbol}=:auto,
    validate::Bool=true,
)
    problem = linear_program(
        c, G, h;
        Aeq=Aeq,
        beq=beq,
        T=T,
        sparse=sparse,
        validate=validate,
        verbosity=0,
    )
    return solve(problem, options)
end
