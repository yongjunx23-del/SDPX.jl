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
    resolved = resolve_solve_options(T, options)
    return solve!(problem, resolved.core)
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
            bits > 0 || throw(ArgumentError("precision must be positive"))
            return setprecision(BigFloat, bits) do
                _solve_sdp_with_frontend(problem, options)
            end
        end
    end
    return _solve_sdp_with_frontend(problem, options)
end

"""Solve a compact SOC model through the all-auto frontend policy."""
function solve_socp(problem::ConicProblem{T}, options::SolveOptions) where {T}
    run = function ()
        resolved = resolve_solve_options(T, options)
        lifted = _soc_psd_lift(problem; sparse=resolved.core.sparse,
                               verbosity=resolved.core.verbosity)
        result = solve!(lifted, resolved.core)
        return _conic_result(problem, result)
    end
    if T === BigFloat
        bits = _frontend_requested_bigfloat_bits(options)
        if bits !== nothing && Base.precision(BigFloat) != bits
            bits > 0 || throw(ArgumentError("precision must be positive"))
            return setprecision(run, BigFloat, bits)
        end
    end
    return run()
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
