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
    result = solve!(problem, resolved.core)
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
            bits > 0 || throw(ArgumentError("precision must be positive"))
            return setprecision(BigFloat, bits) do
                _solve_sdp_with_frontend(problem, options)
            end
        end
    end
    return _solve_sdp_with_frontend(problem, options)
end

function _native_soc_frontend_timing(
    result::ConicResult{T},
    frontend_seconds::Float64,
    certification_seconds::Float64,
    timing::Bool,
) where {T}
    !(timing && result.diagnostics isa NativeSOCDiagnostics) && return result
    diagnostics = result.diagnostics
    return ConicResult{T}(
        result.status,
        result.message,
        result.x,
        result.slack,
        result.dual,
        result.equality_dual,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        NativeSOCDiagnostics(
            diagnostics.plan,
            merge(
                diagnostics.timings,
                (
                    frontend=frontend_seconds,
                    certification=certification_seconds,
                ),
            ),
            diagnostics.memory,
            diagnostics.selected_algorithms,
            diagnostics.warnings,
            diagnostics.termination,
        ),
        result.lifted,
    )
end

Base.@noinline function _solve_socp_keyword_dispatch(
    problem,
    sparse,
    verbosity,
    soc_representation,
    specialization,
    kwargs,
)
    Base.@nospecialize problem kwargs
    soc_representation in (:auto, :native) ||
        throw(ArgumentError(
            "soc_representation must be :auto or :native; the historical " *
            "PSD lift is now a test-only reference",
        ))
    # Preserve the historical public `solve_socp` default: unlike the expert
    # `SolverOptions` constructor, the compact API records timings unless the
    # caller explicitly disables them.
    options = haskey(kwargs, :timing) ?
              SolverOptions(eltype(problem); verbosity, sparse, kwargs...) :
              SolverOptions(
                  eltype(problem); verbosity, sparse, timing=true, kwargs...,
              )
    return _run_native_soc_frontend(problem, options, specialization)
end

Base.@noinline function _run_native_soc_frontend(
    problem::ConicProblem{T},
    options::SolverOptions{T},
    specialization::Symbol,
) where {T}
    frontend_started = time_ns()
    # NativeSOC has no representation transform: frontend work is limited to
    # validating the specialization selector and crossing the public boundary.
    specialization in (:auto, :off, :fixed_trace) || throw(ArgumentError(
        "native SOC specialization must be :auto, :off, or :fixed_trace",
    ))
    frontend_seconds = (time_ns() - frontend_started) / 1.0e9
    result = _solve_native_soc_core(problem, options; specialization)
    certification_started = time_ns()
    result = certify_native_soc_result(problem, result, options)
    certification_seconds =
        (time_ns() - certification_started) / 1.0e9
    return _native_soc_frontend_timing(
        result,
        frontend_seconds,
        certification_seconds,
        options.timing,
    )
end

"""
    solve_socp(problem; kwargs...)

Solve a compact LP+SOC model directly in Lorentz coordinates. Production
`:auto` and `:native` never construct PSD matrices; the historical PSD lift
lives only in test/benchmark reference code.
"""
Base.@noinline function solve_socp(
    problem::ConicProblem{T};
    sparse=:auto,
    verbosity::Int=1,
    soc_representation::Symbol=:auto,
    specialization::Symbol=:auto,
    kwargs...,
) where {T}
    Base.@nospecialize kwargs
    dispatch = Base.inferencebarrier(_solve_socp_keyword_dispatch)
    return Base.invokelatest(
        dispatch,
        problem,
        sparse,
        verbosity,
        soc_representation,
        specialization,
        kwargs,
    )
end

"""Solve a compact SOC model through the all-auto frontend policy."""
function solve_socp(problem::ConicProblem{T}, options::SolveOptions) where {T}
    run = function ()
        resolved = resolve_solve_options(T, options)
        return _run_native_soc_frontend(problem, resolved.core, :auto)
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
