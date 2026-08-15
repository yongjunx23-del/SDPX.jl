#=
    Internal performance trace

`PerformanceTrace` is a stable, additive projection of a finished solve.  It
never recomputes objectives, residuals, or certificates: every value is lifted
from `SDPResult` (`timings`, `termination`, `diagnostics`) as the solve already
recorded it.  A field whose source did not exist at solve time is marked with
the singleton `unavailable` instead of being defaulted to zero, `nothing`, or a
guessed symbol.  This keeps the benchmark tables honest: a missing phase timing
and a measured zero-second phase are observably different.

The four sections are:

  * `setup`   - presolve/pipeline/core seconds, setup-phase timings, and the
                routing/provider/fallback facts recorded by the solve.
  * `iteration` - per-phase interior-point iteration timings.
  * `final`   - terminal status, termination reason, objective/residual/gap,
                certificate availability, and memory facts.
  * `counters` - integer counters (iterations, restarts, regularizations,
                parameter-history length).

This is an internal inspection surface.  It is reachable as
`SDPX.PerformanceTrace` / `SDPX.performance_trace` and mirrored under
`SDPX.Experimental`; it is not part of the top-level stable export set.
=#

"""Distinct marker for a field the solve never recorded."""
struct Unavailable end

const unavailable = Unavailable()

Base.show(io::IO, ::Unavailable) = print(io, "unavailable")

isavailable(value) = value !== unavailable

"""
    PerformanceTrace

Stable projection of one solve.  See the module-level notes for the meaning of
each section and the `unavailable` marker.
"""
struct PerformanceTrace
    setup::NamedTuple
    iteration::NamedTuple
    final::NamedTuple
    counters::NamedTuple
end

@inline function _project_field(source, key)
    source === nothing && return unavailable
    source isa NamedTuple && return get(source, key, unavailable)
    hasproperty(source, key) && return getproperty(source, key)
    return unavailable
end

@inline _seconds(source, key) = _project_field(source, key)

@inline function _first_recorded_seconds(timings, primary, fallback)
    value = _seconds(timings, primary)
    return value === unavailable ? _seconds(timings, fallback) : value
end

function _direction_recovery_seconds(timings)
    direct = _seconds(timings, :direction_recovery)
    direct !== unavailable && return direct
    predictor = _seconds(timings, :predictor_direction_recovery)
    corrector = _seconds(timings, :corrector_direction_recovery)
    (predictor === unavailable || corrector === unavailable) && return unavailable
    return predictor + corrector
end

function _selected_algorithms(result::SDPResult)
    diagnostics = result.diagnostics
    diagnostics === nothing && return unavailable
    return getfield(diagnostics, :selected_algorithms)
end

function _selected_algorithms(result::ConicResult)
    diagnostics = result.diagnostics
    diagnostics === nothing && return unavailable
    return getfield(diagnostics, :selected_algorithms)
end

function _recorded_timings(result::SDPResult)
    diagnostics = result.diagnostics
    # `diagnostics.timings` is `result.timings` merged with presolve/core/
    # pipeline, so it is a superset whenever diagnostics were retained.
    diagnostics === nothing && return result.timings
    return getfield(diagnostics, :timings)
end


function _recorded_timings(result::ConicResult)
    diagnostics = result.diagnostics
    diagnostics === nothing && return nothing
    return getfield(diagnostics, :timings)
end

function _setup_facts(result::SDPResult)
    timings = _recorded_timings(result)
    selected = _selected_algorithms(result)
    return (
        frontend_seconds=_seconds(timings, :frontend),
        preprocess_seconds=_seconds(timings, :preprocess),
        presolve_seconds=_seconds(timings, :presolve),
        equality_presolve_seconds=_seconds(timings, :equality_presolve),
        structural_analysis_seconds=_seconds(timings, :structural_analysis),
        execution_planning_seconds=_seconds(timings, :execution_planning),
        pipeline_seconds=_seconds(timings, :pipeline),
        core_seconds=_seconds(timings, :core),
        setup_seconds=_seconds(timings, :setup),
        setup_validation_seconds=_seconds(timings, :setup_validation),
        precision_preparation_seconds=
            _seconds(timings, :precision_preparation),
        equilibration_seconds=_seconds(timings, :equilibration),
        parameter_selection_seconds=
            _seconds(timings, :parameter_selection),
        workspace_setup_seconds=_seconds(timings, :workspace_setup),
        initialization_seconds=_seconds(timings, :initialization),
        solver=_project_field(selected, :solver),
        scaling=_project_field(selected, :scaling),
        kkt=_project_field(selected, :kkt),
        requested_kkt_formulation=
            _project_field(selected, :requested_kkt_formulation),
        planned_kkt_formulation=
            _project_field(selected, :planned_kkt_formulation),
        executed_kkt_formulation=
            _project_field(selected, :executed_kkt_formulation),
        planned_factorization=
            _project_field(selected, :planned_factorization),
        executed_factorization=
            _project_field(selected, :executed_factorization),
        planned_regularization=
            _project_field(selected, :planned_regularization),
        executed_regularization=
            _project_field(selected, :executed_regularization),
        gram=_project_field(selected, :gram),
        equality=_project_field(selected, :equality),
        planned_backend=_project_field(selected, :planned_backend),
        executed_backend=_project_field(selected, :executed_backend),
        fallback_reason=_project_field(selected, :fallback_reason),
        backend_resolution=_project_field(selected, :backend_resolution),
        lp_formulation=_project_field(selected, :lp_formulation),
        executed_la_backend=_project_field(selected, :la_backend),
        executed_la_provider=_project_field(selected, :la_executed_provider),
        executed_la_ownership=_project_field(selected, :la_executed_ownership),
        executed_la_fallback_reason=
            _project_field(selected, :la_fallback_reason),
        planned_la_backend=_project_field(selected, :planned_la_backend),
        planned_la_provider=_project_field(selected, :planned_la_provider),
        planned_la_ownership=_project_field(selected, :planned_la_ownership),
        planned_la_fallback_reason=
            _project_field(selected, :planned_la_fallback_reason),
        parameter_source=_project_field(selected, :parameter_source),
    )
end

_setup_facts(result::ConicResult) = _setup_facts_for_record(result)

function _setup_facts_for_record(result)
    timings = _recorded_timings(result)
    selected = _selected_algorithms(result)
    return (
        frontend_seconds=_seconds(timings, :frontend),
        preprocess_seconds=_seconds(timings, :preprocess),
        presolve_seconds=_seconds(timings, :presolve),
        equality_presolve_seconds=_seconds(timings, :equality_presolve),
        structural_analysis_seconds=_seconds(timings, :structural_analysis),
        execution_planning_seconds=_seconds(timings, :execution_planning),
        pipeline_seconds=_seconds(timings, :pipeline),
        core_seconds=_seconds(timings, :core),
        setup_seconds=_seconds(timings, :setup),
        setup_validation_seconds=_seconds(timings, :setup_validation),
        precision_preparation_seconds=_seconds(timings, :precision_preparation),
        equilibration_seconds=_seconds(timings, :equilibration),
        parameter_selection_seconds=_seconds(timings, :parameter_selection),
        workspace_setup_seconds=_seconds(timings, :workspace_setup),
        initialization_seconds=_seconds(timings, :initialization),
        solver=_project_field(selected, :solver),
        scaling=_project_field(selected, :scaling),
        kkt=_project_field(selected, :kkt),
        requested_kkt_formulation=_project_field(selected, :requested_kkt_formulation),
        planned_kkt_formulation=_project_field(selected, :planned_kkt_formulation),
        executed_kkt_formulation=_project_field(selected, :executed_kkt_formulation),
        planned_factorization=_project_field(selected, :planned_factorization),
        executed_factorization=_project_field(selected, :executed_factorization),
        planned_regularization=_project_field(selected, :planned_regularization),
        executed_regularization=_project_field(selected, :executed_regularization),
        gram=_project_field(selected, :gram),
        equality=_project_field(selected, :equality),
        planned_backend=_project_field(selected, :planned_backend),
        executed_backend=_project_field(selected, :executed_backend),
        fallback_reason=_project_field(selected, :fallback_reason),
        backend_resolution=_project_field(selected, :backend_resolution),
        lp_formulation=_project_field(selected, :lp_formulation),
        executed_la_backend=_project_field(selected, :la_backend),
        executed_la_provider=_project_field(selected, :la_executed_provider),
        executed_la_ownership=_project_field(selected, :la_executed_ownership),
        executed_la_fallback_reason=_project_field(selected, :la_fallback_reason),
        planned_la_backend=_project_field(selected, :planned_la_backend),
        planned_la_provider=_project_field(selected, :planned_la_provider),
        planned_la_ownership=_project_field(selected, :planned_la_ownership),
        planned_la_fallback_reason=_project_field(selected, :planned_la_fallback_reason),
        parameter_source=_project_field(selected, :parameter_source),
    )
end

function _iteration_facts(result::SDPResult)
    timings = _recorded_timings(result)
    return (
        cone_scaling_metric_seconds=_seconds(timings, :cone_scaling_metric),
        initial_residual_seconds=_seconds(timings, :initial_residual),
        residual_and_block_factor_seconds=
            _seconds(timings, :residual_and_block_factor),
        schur_assembly_seconds=_seconds(timings, :schur_assembly),
        equality_assembly_seconds=_seconds(timings, :kkt_equality_gram),
        kkt_factorization_seconds=_seconds(timings, :kkt_factorization),
        kkt_numeric_factorization_seconds=
            _seconds(timings, :kkt_schur_factorization),
        predictor_seconds=_seconds(timings, :predictor),
        predictor_rhs_seconds=_seconds(timings, :predictor_rhs),
        predictor_solve_seconds=
            _seconds(timings, :predictor_linear_solve),
        corrector_seconds=_seconds(timings, :corrector),
        corrector_rhs_seconds=_seconds(timings, :corrector_rhs),
        corrector_solve_seconds=
            _seconds(timings, :corrector_linear_solve),
        line_search_seconds=_seconds(timings, :line_search),
        refinement_seconds=_seconds(timings, :refinement),
        accepted_update_seconds=_first_recorded_seconds(
            timings,
            :accepted_update,
            :update,
        ),
        direction_recovery_seconds=_direction_recovery_seconds(timings),
        complementarity_analysis_seconds=
            _seconds(timings, :complementarity_analysis),
        finalization_seconds=_seconds(timings, :finalization),
        fixed_local_scaling_metric_seconds=
            _seconds(timings, :fixed_local_scaling_metric),
        fixed_local_metric_seconds=_seconds(timings, :fixed_local_metric),
        fixed_local_factor_seconds=_seconds(timings, :fixed_local_factor),
        fixed_rhs_contraction_seconds=
            _seconds(timings, :fixed_rhs_contraction),
        equality_panel_transform_seconds=
            _seconds(timings, :equality_panel_transform),
        equality_gram_syrk_seconds=_seconds(timings, :equality_gram_syrk),
        equality_factor_seconds=_seconds(timings, :equality_factor),
        fixed_block_residual_seconds=
            _seconds(timings, :fixed_block_residual),
        fixed_block_recovery_seconds=
            _seconds(timings, :fixed_block_recovery),
    )
end


function _iteration_facts(result::ConicResult)
    timings = _recorded_timings(result)
    return (
        cone_scaling_metric_seconds=_seconds(timings, :cone_scaling_metric),
        initial_residual_seconds=_seconds(timings, :initial_residual),
        residual_and_block_factor_seconds=_seconds(timings, :residual_and_block_factor),
        schur_assembly_seconds=_seconds(timings, :schur_assembly),
        equality_assembly_seconds=_seconds(timings, :kkt_equality_gram),
        kkt_factorization_seconds=_seconds(timings, :kkt_factorization),
        kkt_numeric_factorization_seconds=_seconds(timings, :kkt_schur_factorization),
        predictor_seconds=_seconds(timings, :predictor),
        predictor_rhs_seconds=_seconds(timings, :predictor_rhs),
        predictor_solve_seconds=_seconds(timings, :predictor_linear_solve),
        corrector_seconds=_seconds(timings, :corrector),
        corrector_rhs_seconds=_seconds(timings, :corrector_rhs),
        corrector_solve_seconds=_seconds(timings, :corrector_linear_solve),
        line_search_seconds=_seconds(timings, :line_search),
        refinement_seconds=_seconds(timings, :refinement),
        accepted_update_seconds=_seconds(timings, :accepted_update),
        direction_recovery_seconds=_seconds(timings, :direction_recovery),
        complementarity_analysis_seconds=_seconds(timings, :complementarity_analysis),
        finalization_seconds=_seconds(timings, :finalization),
        fixed_local_scaling_metric_seconds=_seconds(timings, :fixed_local_scaling_metric),
        fixed_local_metric_seconds=_seconds(timings, :fixed_local_metric),
        fixed_local_factor_seconds=_seconds(timings, :fixed_local_factor),
        fixed_rhs_contraction_seconds=_seconds(timings, :fixed_rhs_contraction),
        equality_panel_transform_seconds=_seconds(timings, :equality_panel_transform),
        equality_gram_syrk_seconds=_seconds(timings, :equality_gram_syrk),
        equality_factor_seconds=_seconds(timings, :equality_factor),
        fixed_block_residual_seconds=_seconds(timings, :fixed_block_residual),
        fixed_block_recovery_seconds=_seconds(timings, :fixed_block_recovery),
    )
end

function _final_facts(result::SDPResult)
    timings = _recorded_timings(result)
    diagnostics = result.diagnostics
    selected = _selected_algorithms(result)
    certificate = _project_field(selected, :certificate)
    memory = diagnostics === nothing ? unavailable :
             getfield(diagnostics, :memory)
    return (
        total_seconds=_seconds(timings, :total),
        status=Symbol(result.status),
        termination_reason=_project_field(result.termination, :reason),
        termination_stage=_project_field(result.termination, :stage),
        primal_objective=result.pObj,
        dual_objective=result.dObj,
        relative_gap=result.gap_rel,
        primal_residual=result.p_res,
        dual_residual=result.d_res,
        certificate_available=_project_field(certificate, :available),
        certificate_reason=_project_field(certificate, :reason),
        certification_seconds=_seconds(timings, :certification),
        workspace_bytes=_project_field(memory, :workspace_bytes),
        process_peak_rss_bytes=_project_field(memory, :process_peak_rss_bytes),
        memory_budget_bytes=_project_field(memory, :memory_budget_bytes),
        warnings=diagnostics === nothing ? String[] :
                 getfield(diagnostics, :warnings),
    )
end

function _counter_facts(result::SDPResult)
    sparse = _project_field(result.termination, :sparse_schur_backend)
    return (
        iterations=result.iterations,
        restarts=result.restarts,
        regularizations=result.regularizations,
        parameter_history_length=length(result.parameter_history),
        numeric_factorizations=_project_field(sparse, :factorizations),
        rhs_solves=_project_field(result.termination, :rhs_solves),
        refinement_solves=
            _project_field(result.termination, :total_refinement_steps),
        local_metric_preparations=_project_field(
            result.termination, :local_metric_preparations,
        ),
        local_factorizations=_project_field(
            result.termination, :local_factorizations,
        ),
        equality_panel_transforms=_project_field(
            result.termination, :equality_panel_transforms,
        ),
        equality_gram_assemblies=_project_field(
            result.termination, :equality_gram_assemblies,
        ),
        equality_factorizations=_project_field(
            result.termination, :equality_factorizations,
        ),
        kkt_rhs_solves=_project_field(result.termination, :kkt_rhs_solves),
        predictor_rhs_solves=_project_field(
            result.termination, :predictor_rhs_solves,
        ),
        corrector_rhs_solves=_project_field(
            result.termination, :corrector_rhs_solves,
        ),
        symbolic_analyses=_project_field(sparse, :analyses),
        symbolic_analysis_reuse=
            _project_field(sparse, :symbolic_reuse_ratio),
        schur_nnz=_project_field(sparse, :schur_nnz),
        factor_nnz=_project_field(sparse, :factor_nonzeros),
        kkt_nnz=_project_field(result.termination, :kkt_nonzeros),
        factor_memory_estimate_bytes=
            _project_field(result.termination, :factor_memory_estimate_bytes),
    )
end


function _final_facts(result::ConicResult)
    timings = _recorded_timings(result)
    diagnostics = result.diagnostics
    selected = _selected_algorithms(result)
    certificate = _project_field(selected, :certificate)
    memory = diagnostics === nothing ? unavailable : getfield(diagnostics, :memory)
    termination = diagnostics === nothing ? nothing : getfield(diagnostics, :termination)
    return (
        total_seconds=_seconds(timings, :total),
        status=Symbol(result.status),
        termination_reason=_project_field(termination, :reason),
        termination_stage=_project_field(termination, :stage),
        primal_objective=result.pObj,
        dual_objective=result.dObj,
        relative_gap=result.gap_rel,
        primal_residual=result.p_res,
        dual_residual=result.d_res,
        certificate_available=_project_field(certificate, :available),
        certificate_reason=_project_field(certificate, :reason),
        certification_seconds=_seconds(timings, :certification),
        workspace_bytes=_project_field(memory, :workspace_bytes),
        process_peak_rss_bytes=_project_field(memory, :process_peak_rss_bytes),
        memory_budget_bytes=_project_field(memory, :memory_budget_bytes),
        warnings=diagnostics === nothing ? String[] : getfield(diagnostics, :warnings),
    )
end

function _counter_facts(result::ConicResult)
    diagnostics = result.diagnostics
    termination = diagnostics === nothing ? nothing : getfield(diagnostics, :termination)
    return (
        iterations=result.iterations,
        restarts=0,
        regularizations=_project_field(termination, :regularizations),
        parameter_history_length=0,
        numeric_factorizations=_project_field(termination, :numeric_factorizations),
        rhs_solves=_project_field(termination, :rhs_solves),
        refinement_solves=_project_field(termination, :refinement_solves),
        local_metric_preparations=_project_field(
            termination, :local_metric_preparations,
        ),
        local_factorizations=_project_field(termination, :local_factorizations),
        equality_panel_transforms=_project_field(
            termination, :equality_panel_transforms,
        ),
        equality_gram_assemblies=_project_field(
            termination, :equality_gram_assemblies,
        ),
        equality_factorizations=_project_field(
            termination, :equality_factorizations,
        ),
        kkt_rhs_solves=_project_field(termination, :kkt_rhs_solves),
        predictor_rhs_solves=_project_field(termination, :predictor_rhs_solves),
        corrector_rhs_solves=_project_field(termination, :corrector_rhs_solves),
        symbolic_analyses=unavailable,
        symbolic_analysis_reuse=unavailable,
        schur_nnz=unavailable,
        factor_nnz=unavailable,
        kkt_nnz=unavailable,
        factor_memory_estimate_bytes=unavailable,
    )
end

"""
    performance_trace(result::SDPResult) -> PerformanceTrace

Project a finished solve into a stable four-section `PerformanceTrace`.
Fields absent from the recorded result are returned as `unavailable`, never
zeroed or guessed.
"""
function performance_trace(result::SDPResult)
    return PerformanceTrace(
        _setup_facts(result),
        _iteration_facts(result),
        _final_facts(result),
        _counter_facts(result),
    )
end

"""
    performance_trace(result::ConicResult) -> PerformanceTrace

Native Lorentz-cone results project their own diagnostics. Test-only
reference results carry SDP-shaped diagnostics and project them directly.
"""
function performance_trace(result::ConicResult)
    return PerformanceTrace(
        _setup_facts(result),
        _iteration_facts(result),
        _final_facts(result),
        _counter_facts(result),
    )
end

function Base.show(io::IO, trace::PerformanceTrace)
    print(io, "PerformanceTrace(status=", trace.final.status,
        ", iterations=", trace.counters.iterations,
        ", termination=", trace.final.termination_reason, ")")
end
