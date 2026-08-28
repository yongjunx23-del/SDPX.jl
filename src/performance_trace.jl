#=
    Internal performance trace

`PerformanceTrace` is a stable, additive projection of a finished solve.  It
never recomputes objectives, residuals, or certificates: every value is lifted
from `SDPResult` (`timings`, `termination`, `diagnostics`) as the solve already
recorded it.  A field whose source did not exist at solve time is marked with
the singleton `unavailable` instead of being defaulted to zero, `nothing`, or a
guessed symbol.  This keeps the benchmark tables honest: a missing phase timing
and a measured zero-second phase are observably different.

The sections are:

  * `setup`   - presolve/pipeline/core seconds, setup-phase timings, and the
                routing/provider/fallback facts recorded by the solve.
  * `iteration` - per-phase interior-point iteration timings.
  * `final`   - terminal status, termination reason, objective/residual/gap,
                certificate availability, and memory facts.
  * `counters` - integer counters (iterations, restarts, regularizations,
                parameter-history length).
  * `phases`   - phase-separated timing receipt (one canonical slot per
                traced phase plus wall-clock accounting).
  * `attempts` - per-attempt fallback timing and counts.

This is the typed backing record for the public `performance_trace(result)`
accessor.  Callers reach it through the single v0.5 `Result` interface; there
is no parallel experimental namespace or alternate solve route.
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
    # P0 phase-separated timing receipt: one canonical slot per traced phase
    # (cold setup, equality/rank, symbolic, assembly, numeric factor,
    # predictor solve, corrector RHS/solve, refinement, line search, state
    # update, original-coordinate certificate) plus phase-accounting facts
    # against a wall-clock reference. A phase that was never recorded is
    # `unavailable`, never a zero.
    phases::NamedTuple
    # Per-attempt fallback timing/counts: one `ExecutionAttemptRecord` per
    # attempt in the chain, with wall-clock `elapsed_seconds` per attempt and
    # the number of fallback events per attempt.
    attempts::NamedTuple
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

@inline function _sum_seconds(timings, keys)
    values = Tuple(_seconds(timings, key) for key in keys)
    all(value -> value !== unavailable, values) || return unavailable
    return sum(values)
end

@inline _all_available(record) =
    all(value -> value !== unavailable,
        record isa NamedTuple ? values(record) : record)

"""
    _phase_facts(result) -> NamedTuple

Phase-separated timing receipt. Each canonical phase maps to existing
recorded timing keys (never recomputed); a phase whose source was not
recorded stays `unavailable`. `reference_seconds` is the whole-solve
`pipeline` wall clock when diagnostics retained it, falling back to the
core `total`; `accounted_seconds` is the sum of the canonical phases that
lie inside that reference (pre-core phases are excluded when only the core
total is available), `unaccounted_seconds` is the honest remainder, and
`consistent` is whether the accounted phases stay within the reference
wall time. All phases are disjoint wall-time sub-intervals, so the sum can
never exceed the reference beyond float noise.
"""
function _phase_facts(result)
    timings = _recorded_timings(result)
    pipeline = _seconds(timings, :pipeline)
    total = _seconds(timings, :total)
    cold_setup = _seconds(timings, :setup)
    if cold_setup === unavailable
        cold_setup = _sum_seconds(
            timings,
            (
                :setup_validation, :precision_preparation, :equilibration,
                :parameter_selection, :workspace_setup, :initialization,
                :initial_residual,
            ),
        )
    end
    phases = (
        cold_setup_seconds=cold_setup,
        equality_rank_seconds=_seconds(timings, :equality_presolve),
        symbolic_seconds=_seconds(timings, :structural_analysis),
        assembly_seconds=_seconds(timings, :schur_assembly),
        numeric_factor_seconds=_seconds(timings, :kkt_factorization),
        predictor_solve_seconds=_seconds(timings, :predictor_linear_solve),
        corrector_rhs_seconds=_seconds(timings, :corrector_rhs),
        corrector_solve_seconds=_seconds(timings, :corrector_linear_solve),
        refinement_seconds=_seconds(timings, :refinement),
        line_search_seconds=_seconds(timings, :line_search),
        state_update_seconds=
            _first_recorded_seconds(timings, :accepted_update, :update),
        certificate_seconds=_seconds(timings, :certification),
    )
    reference = pipeline !== unavailable ? pipeline : total
    # The accounted phases are always a plain tuple of scalar values so
    # `_all_available`/`sum` are backend-agnostic. When only the core
    # `total` is available the pre-core phases (equality/rank presolve,
    # symbolic analysis, certification) lie outside that reference and are
    # excluded from the accounting sum.
    phase_values = (
        phases.cold_setup_seconds,
        phases.equality_rank_seconds,
        phases.symbolic_seconds,
        phases.assembly_seconds,
        phases.numeric_factor_seconds,
        phases.predictor_solve_seconds,
        phases.corrector_rhs_seconds,
        phases.corrector_solve_seconds,
        phases.refinement_seconds,
        phases.line_search_seconds,
        phases.state_update_seconds,
        phases.certificate_seconds,
    )
    accounted_inputs = pipeline !== unavailable ?
                       phase_values :
                       (
            phases.cold_setup_seconds,
            phases.assembly_seconds,
            phases.numeric_factor_seconds,
            phases.predictor_solve_seconds,
            phases.corrector_rhs_seconds,
            phases.corrector_solve_seconds,
            phases.refinement_seconds,
            phases.line_search_seconds,
            phases.state_update_seconds,
        )
    accounted = _all_available(accounted_inputs) ?
                sum(accounted_inputs) : unavailable
    unaccounted = (reference !== unavailable && accounted !== unavailable) ?
                  reference - accounted : unavailable
    tolerance = reference === unavailable ?
                nothing : 1.0e-6 * max(1.0, reference)
    consistent = (reference !== unavailable && accounted !== unavailable) ?
                 unaccounted >= -tolerance : unavailable
    return merge(phases, (
        reference_seconds=reference,
        accounted_seconds=accounted,
        unaccounted_seconds=unaccounted,
        consistent=consistent,
    ))
end

"""
    _attempt_facts(result) -> NamedTuple

Per-attempt fallback timing/counts projected from the retained
`SolveDiagnostics.attempts` tuple. Every recorded attempt carries its own
`elapsed_seconds` (recorded at `_attach_diagnostics` from the pipeline wall
clock); totals are unavailable when no attempt record exists.
"""
function _attempt_facts(result::SDPResult)
    diagnostics = result.diagnostics
    if diagnostics === nothing || !hasproperty(diagnostics, :attempts)
        return _missing_attempt_facts()
    end
    attempts = getfield(diagnostics, :attempts)
    elapsed = Tuple(record.elapsed_seconds for record in attempts)
    fallback_counts =
        Tuple(length(record.fallback_events) for record in attempts)
    return (
        count=length(attempts),
        elapsed_seconds=elapsed,
        fallback_event_counts=fallback_counts,
        elapsed_seconds_total=isempty(elapsed) ? unavailable : sum(elapsed),
        fallback_events_total=
            isempty(fallback_counts) ? unavailable : sum(fallback_counts),
    )
end

function _attempt_facts(result::ConicResult)
    return _missing_attempt_facts()
end

_missing_attempt_facts() = (
    count=unavailable,
    elapsed_seconds=unavailable,
    fallback_event_counts=unavailable,
    elapsed_seconds_total=unavailable,
    fallback_events_total=unavailable,
)

_setup_facts(result::SDPResult) = _setup_facts_for_record(result)

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
        reconstruction_seconds=_seconds(timings, :reconstruction),
        solver=_project_field(selected, :solver),
        scaling=_project_field(selected, :scaling),
        planned_scaling=_project_field(selected, :planned_scaling),
        executed_scaling=_project_field(selected, :executed_scaling),
        native_kernel=_project_field(selected, :native_kernel),
        planned_native_kernel=_project_field(selected, :planned_native_kernel),
        executed_native_kernel=_project_field(selected, :executed_native_kernel),
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
        la_factorization=_project_field(selected, :la_factorization),
        effective_threads=_project_field(selected, :effective_threads),
        schur_threads=_project_field(selected, :schur_threads),
        factor_threads=_project_field(selected, :factor_threads),
        lp_pack_threads=_project_field(selected, :lp_pack_threads),
        arrow_linear_solve=_project_field(selected, :arrow_linear_solve),
        fine_grained_block_tasks=_project_field(selected, :fine_grained_block_tasks),
        fine_grained_block_partition=
            _project_field(selected, :fine_grained_block_partition),
        parameter_resolution_count=
            _project_field(selected, :parameter_resolution_count),
        parameter_resolution_stage=_project_field(selected, :stage),
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
        kkt_schur_copy_seconds=_seconds(timings, :kkt_schur_copy),
        kkt_constraint_triangular_solve_seconds=
            _seconds(timings, :kkt_constraint_triangular_solve),
        kkt_equality_factorization_seconds=
            _seconds(timings, :kkt_equality_factorization),
        kkt_other_seconds=_seconds(timings, :kkt_other),
        predictor_direction_recovery_seconds=
            _seconds(timings, :predictor_direction_recovery),
        corrector_direction_recovery_seconds=
            _seconds(timings, :corrector_direction_recovery),
        best_iterate_seconds=_seconds(timings, :best_iterate),
        objective_and_targets_seconds=
            _seconds(timings, :objective_and_targets),
        other_seconds=_seconds(timings, :other),
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
        kkt_schur_copy_seconds=_seconds(timings, :kkt_schur_copy),
        kkt_constraint_triangular_solve_seconds=
            _seconds(timings, :kkt_constraint_triangular_solve),
        kkt_equality_factorization_seconds=
            _seconds(timings, :kkt_equality_factorization),
        kkt_other_seconds=_seconds(timings, :kkt_other),
        predictor_direction_recovery_seconds=
            _seconds(timings, :predictor_direction_recovery),
        corrector_direction_recovery_seconds=
            _seconds(timings, :corrector_direction_recovery),
        best_iterate_seconds=_seconds(timings, :best_iterate),
        objective_and_targets_seconds=
            _seconds(timings, :objective_and_targets),
        other_seconds=_seconds(timings, :other),
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
        certificate_valid=_project_field(certificate, :valid),
        certificate_kind=_project_field(certificate, :kind),
        certificate_failures=_project_field(certificate, :failures),
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
        factorization_attempts=_project_field(sparse, :factorization_attempts),
        factorization_successes=_project_field(sparse, :factorization_successes),
        factorization_failures=_project_field(sparse, :factorization_failures),
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
        schur_assembly_count=_project_field(sparse, :assembly_count),
        sparse_factor_failures=_project_field(sparse, :failures),
        symbolic_analysis_reuses=_project_field(sparse, :reused),
        fixed_residual_blocks=_project_field(
            result.termination, :fixed_residual_blocks,
        ),
        fixed_rhs_contractions=_project_field(
            result.termination, :fixed_rhs_contractions,
        ),
        fixed_direction_recoveries=_project_field(
            result.termination, :fixed_direction_recoveries,
        ),
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
        certificate_valid=_project_field(certificate, :valid),
        certificate_kind=_project_field(certificate, :kind),
        certificate_failures=_project_field(certificate, :failures),
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
        factorization_attempts=unavailable,
        factorization_successes=unavailable,
        factorization_failures=unavailable,
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
        schur_assembly_count=unavailable,
        sparse_factor_failures=unavailable,
        symbolic_analysis_reuses=unavailable,
        fixed_residual_blocks=_project_field(termination, :fixed_residual_blocks),
        fixed_rhs_contractions=_project_field(termination, :fixed_rhs_contractions),
        fixed_direction_recoveries=
            _project_field(termination, :fixed_direction_recoveries),
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
        _phase_facts(result),
        _attempt_facts(result),
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
        _phase_facts(result),
        _attempt_facts(result),
    )
end

function Base.show(io::IO, trace::PerformanceTrace)
    print(io, "PerformanceTrace(status=", trace.final.status,
        ", iterations=", trace.counters.iterations,
        ", termination=", trace.final.termination_reason, ")")
end
