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
