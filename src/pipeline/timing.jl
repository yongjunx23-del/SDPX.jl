function _with_frontend_timing(
    result::SDPResult{T},
    elapsed::Float64,
    enabled::Bool,
) where {T}
    enabled || return result
    result_timings = result.timings === nothing ?
                     (frontend=elapsed,) :
                     merge(
                         result.timings,
                         (
                             frontend=
                                 get(result.timings, :frontend, 0.0) + elapsed,
                         ),
                     )
    diagnostics = result.diagnostics
    updated_diagnostics = if diagnostics === nothing
        nothing
    else
        diagnostic_timings = merge(
            diagnostics.timings,
            (
                frontend=
                    get(diagnostics.timings, :frontend, 0.0) + elapsed,
            ),
        )
        SolveDiagnostics(
            diagnostics.classification,
            diagnostics.plan,
            diagnostics.presolve,
            diagnostic_timings,
            diagnostics.memory,
            diagnostics.selected_algorithms,
            diagnostics.parameter_history,
            diagnostics.warnings,
            diagnostics.termination,
            diagnostics.attempts,
            diagnostics.precision_ladder,
        )
    end
    return SDPResult{T}(
        result.status,
        result.message,
        result.x,
        result.X,
        result.y,
        result.Y,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.restarts,
        result.regularizations,
        result_timings,
        result.parameter_history,
        updated_diagnostics,
        result.termination,
    )
end

function _inconsistent_presolve_result(
    prob::SDPProblem{T},
    report::PresolveReport,
    plan::ExecutionPlan,
    opts::SolverOptions{T},
    pipeline_timings::NamedTuple=NamedTuple(),
    ladder_context::Union{Nothing,PrecisionLadderContext}=nothing,
) where {T}
    # A negative fixed scalar block is exactly the dedicated LP zero-row
    # contradiction. Preserve that established, more specific termination
    # reason even though the generic fixed-trace stage now detects it first.
    lp_zero_row = plan.classification.cone === :lp &&
                  !isempty(analyze_fixed_trace(prob).infeasible_blocks)
    termination_reason = lp_zero_row ?
                         :lp_zero_row_infeasible :
                         :structural_presolve_infeasibility
    X = [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k]
    Y = [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k]
    result = SDPResult{T}(
        InfeasibleCert,
        "Presolve detected a structural constraint contradiction.",
        alloc_zeros(T, prob.dims.m),
        X,
        alloc_zeros(T, prob.dims.n),
        Y,
        zero(T),
        zero(T),
        T(Inf),
        T(Inf),
        T(Inf),
        0,
        0,
        0,
        (total=report.elapsed,),
        NamedTuple[],
        nothing,
        (
            reason=termination_reason,
            certificate_method=:presolve_contradiction,
            certificate_generator=:analytic_presolve,
            executed=(
                solver=plan.algorithm,
                parameter_profile=:not_resolved,
                parameter_source=:not_resolved,
                parameter_resolution_count=0,
                stage=:not_resolved,
                kkt=:not_executed,
                planned_backend=planned_backend_name(plan),
                executed_backend=:not_executed,
                fallback_reason=:none,
                backend_resolution=:not_resolved,
                lp_formulation=plan.algorithm === :lp_primal_dual ?
                               :not_resolved : :not_applicable,
                gram=:not_executed,
            ),
        ),
    )
    certification_started = time_ns()
    certificate = opts.certification ?
                  result_certificate(prob, result, opts) :
                  (
            available=false,
            reason=:certification_disabled,
            minimal_gate=(available=false, reason=:not_applicable),
        )
    recorded_pipeline_timings = opts.timing ?
        merge(
            pipeline_timings,
            (
                certification=
                    get(pipeline_timings, :certification, 0.0) +
                    (time_ns() - certification_started) / 1.0e9,
            ),
        ) : NamedTuple()
    return _attach_diagnostics(
        result,
        plan,
        report,
        report.elapsed,
        [
            "Presolve produced a structural infeasibility proof at the " *
            "configured tolerance.",
        ],
        0,
        opts.diagnostics,
        (reason=:none,),
        certificate,
        recorded_pipeline_timings,
        ladder_context,
    )
end

function _time_limit_pipeline_result(
    prob::SDPProblem{T},
    report::PresolveReport,
    plan::ExecutionPlan,
    elapsed::Float64,
    warnings::Vector{String},
    diagnostics_enabled::Bool,
    max_time::Float64,
    certification_enabled::Bool,
    pipeline_timings::NamedTuple=NamedTuple(),
    ladder_context::Union{Nothing,PrecisionLadderContext}=nothing,
) where {T}
    X = [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k]
    Y = [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k]
    result = SDPResult{T}(
        TimeLimit,
        "Time limit ($(max_time)s) exceeded during automatic pipeline setup.",
        alloc_zeros(T, prob.dims.m),
        X,
        alloc_zeros(T, prob.dims.n),
        Y,
        zero(T),
        zero(T),
        T(Inf),
        T(Inf),
        T(Inf),
        0,
        0,
        0,
        (total=elapsed,),
        NamedTuple[],
        nothing,
        (reason=:time_limit, stage=:pipeline_setup),
    )
    push!(
        warnings,
        "The wall-clock budget expired before numerical iterations began.",
    )
    return _attach_diagnostics(
        result,
        plan,
        report,
        elapsed,
        warnings,
        0,
        diagnostics_enabled,
        (reason=:none,),
        certification_enabled ?
        (available=false,) :
        (
            available=false,
            reason=:certification_disabled,
            minimal_gate=(available=false, reason=:not_applicable),
        ),
        pipeline_timings,
        ladder_context,
    )
end
