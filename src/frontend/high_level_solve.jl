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
            _require_bigfloat_precision_bits(bits, "precision")
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
    )
end

Base.@noinline function _solve_socp_keyword_dispatch(
    problem,
    sparse,
    verbosity,
    specialization,
    kwargs,
)
    Base.@nospecialize problem kwargs
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
    ;
    x0=nothing,
    z0=nothing,
    y0=nothing,
) where {T}
    # The compact keyword API accepts an expert `precision_bits` setting
    # directly, without passing through the all-auto `SolveOptions` wrapper.
    # Keep every NativeSOC arithmetic phase (including presolve, factorization,
    # reconstruction, and certification) inside that exact BigFloat scope.
    if T === BigFloat && Base.precision(BigFloat) != options.precision_bits
        bits = options.precision_bits
        _require_bigfloat_precision_bits(bits, "precision_bits")
        return setprecision(BigFloat, bits) do
            _run_native_soc_frontend(
                problem,
                options,
                specialization;
                x0,
                z0,
                y0,
            )
        end
    end
    _require_supported_arithmetic_type(T)
    frontend_started = time_ns()
    # NativeSOC has no representation transform: frontend work is limited to
    # validating the specialization selector and crossing the public boundary.
    specialization in (:auto, :off, :fixed_trace) || throw(ArgumentError(
        "native SOC specialization must be :auto, :off, or :fixed_trace",
    ))
    frontend_seconds = (time_ns() - frontend_started) / 1.0e9
    # Singleton substitution is a strictly guarded execution reduction.  The
    # original problem remains the certification authority; when any guard is
    # rejected this branch falls through to the historical one-plan route.
    presolve_started = time_ns()
    decision = _native_soc_presolve(
        problem,
        options;
        specialization,
        x0,
        z0,
        y0,
    )
    presolve_seconds = (time_ns() - presolve_started) / 1.0e9
    precomputed_certificate = nothing
    if decision.applied
        reduced_problem = decision.problem
        reduced_map = decision.map
        reduced_plan = build_execution_plan(
            AutoPlanner(), reduced_problem, options; specialization,
        )
        reduced_result = _solve_native_soc_core(
            reduced_problem,
            options,
            reduced_plan;
            x0=nothing,
            z0=nothing,
            y0=nothing,
            objective_offset=reduced_map.kappa,
        )
        reconstruction_started = time_ns()
        result = _native_soc_restore_result(
            problem,
            reduced_problem,
            reduced_result,
            reduced_map,
        )
        reconstruction_seconds =
            (time_ns() - reconstruction_started) / 1.0e9
        result = _native_soc_presolve_annotate(
            result,
            decision,
            options,
            presolve_seconds,
            reconstruction_seconds,
        )
        precomputed_certificate = result_certificate(problem, result, options)
        result = _native_soc_recompute_result_metrics(
            result,
            precomputed_certificate,
        )
    else
        # One top-level plan per ordinary solve: the AutoPlanner freezes the
        # NativeSOC payload, and the core validates it instead of planning a
        # second time.  Presolve-off remains byte-for-byte on this path.
        plan = build_execution_plan(
            AutoPlanner(), problem, options; specialization,
        )
        result = _solve_native_soc_core(
            problem,
            options,
            plan;
            x0=x0,
            z0=z0,
            y0=y0,
        )
        if decision.reason !== :disabled
            result = _native_soc_presolve_annotate(
                result,
                decision,
                options,
                presolve_seconds,
            )
        end
    end
    certification_started = time_ns()
    result = certify_native_soc_result(
        problem,
        result,
        options;
        precomputed_certificate,
    )
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
always uses NativeSOC; the historical PSD lift lives only in test/benchmark
reference code and is not exposed as a public solve option.
"""
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
        dispatch,
        problem,
        sparse,
        verbosity,
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
            _require_bigfloat_precision_bits(bits, "precision")
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
