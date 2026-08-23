"""
    _lp_sparse_diagnostics_la_config(plan, payload)

Return the LA descriptor for a finalized sparse LP route.  The ordinary LA
planner runs before LP row presolve and cannot know that this route will use
CHOLMOD (or the arithmetic-generic sparse provider), so copying its BLAS
descriptor into the post-execution diagnostics plan would make planned and
executed provider facts disagree.  This descriptor is diagnostics-only: the
sparse factor/solve path never instantiates it or uses it for numerical
fallback.
"""
function _lp_sparse_diagnostics_la_config(
    plan::ExecutionPlan,
    payload::LPRoutePlan,
)
    payload.storage === :sparse || return plan.la_config
    payload.route === :sparse_normal || throw(ArgumentError(
        "sparse LP diagnostics payload must use :sparse_normal",
    ))
    provider = payload.provider
    provider in (:cholmod, :generic) || throw(ArgumentError(
        "unknown sparse LP diagnostics provider $(provider)",
    ))
    implementation = provider === :cholmod ?
                     :cholmod_sparse_cholesky : :generic_sparse_cholesky
    # The sparse layer has its own capability registry.  This compact LA
    # projection is only for the immutable diagnostics descriptor and keeps
    # the provider-owned factor/solve and sparse-factorization requirements
    # explicit without claiming dense BLAS/LAPACK ownership.
    capabilities = LAProviderCapabilities(
        cholesky=true,
        factor_solve=true,
        multi_rhs=true,
        sparse_factorization=true,
    )
    return LABackendConfiguration(
        plan.la_config.arithmetic,
        plan.la_config.requested,
        :sparse,
        provider,
        la_capability_symbols(capabilities),
        capabilities,
        (:cholesky, :factor_solve, :multi_rhs, :sparse_factorization),
        implementation,
        (),
        :none,
        :provider_owned,
    )
end
"""Build the canonical diagnostics plan for one finalized LP route.

Dense and reduced payloads deliberately preserve the pre-existing
`ExecutionPlan(plan, payload)` copy.  Sparse routes replace only the LA
descriptor so the payload provider is the planned provider as well as the
executed provider; mathematical formulation, backend deferment, and all
numerical settings remain untouched.
"""
function _lp_finalized_diagnostics_plan(
    plan::ExecutionPlan,
    payload::LPRoutePlan,
)
    payload.storage === :sparse || return ExecutionPlan(plan, payload)
    la_config = _lp_sparse_diagnostics_la_config(plan, payload)
    return ExecutionPlan(
        plan.classification,
        plan.algorithm,
        plan.scaling,
        plan.kkt_backend,
        plan.backend_config,
        plan.formulation_plan,
        la_config,
        plan.gram_kernel,
        plan.schedule,
        plan.threads,
        plan.parameter_profile,
        plan.memory_budget_bytes,
        plan.parameters,
        payload,
    )
end

function _attach_diagnostics(
    result::SDPResult{T},
    plan::ExecutionPlan,
    report::PresolveReport,
    pipeline_time::Float64,
    warnings::Vector{String},
    workspace_bytes::Int,
    diagnostics_enabled::Bool,
    termination::NamedTuple=(reason=:none,),
    certificate::NamedTuple=(available=false,),
    pipeline_timings::NamedTuple=NamedTuple(),
    ladder_context::Union{Nothing,PrecisionLadderContext}=nothing,
) where {T}
    diagnostics_enabled || return result
    # The finalized LP route payload (built after LP row presolve and
    # `_scale_lp!`) is the authoritative diagnostics plan: it carries the
    # resolved route that actually executed, while the outer pre-row plan
    # remains the deferred `:not_applicable` planner artifact.  The finalized
    # payload is never falsified into the outer plan's formulation.
    lp_route_payload = get(
        get(result.termination, :executed, NamedTuple()),
        :lp_route_payload,
        nothing,
    )
    diagnostics_plan = if lp_route_payload isa LPRoutePlan
        _lp_finalized_diagnostics_plan(plan, lp_route_payload)
    else
        plan
    end
    core_time = result.timings === nothing ? NaN :
                get(result.timings, :total, NaN)
    timings = result.timings === nothing ?
              (
                  presolve=report.elapsed,
                  core=core_time,
                  pipeline=pipeline_time,
              ) :
              merge(
                  result.timings,
                  (
                      presolve=report.elapsed,
                      core=core_time,
                      pipeline=pipeline_time,
                  ),
                  pipeline_timings,
              )
    memory = (
        workspace_bytes=workspace_bytes,
        process_peak_rss_bytes=_process_peak_rss_bytes(),
        memory_budget_bytes=plan.memory_budget_bytes,
    )
    # `kkt`/`gram` report what actually executed whenever the solve path
    # said so (`result.termination.executed`), falling back to the plan
    # otherwise. The plan stays visible under `planned`. Before this split
    # the record was the plan alone, and the LP path -- which selects its
    # sparse Newton system at runtime, after the plan is frozen -- reported
    # a dense LU and a BLAS Gram kernel for solves that executed neither.
    executed = get(result.termination, :executed, NamedTuple())
    executed_parameter_profile = get(
        executed,
        :parameter_profile,
        plan.parameter_profile,
    )
    executed_parameters = get(
        executed,
        :executed_parameters,
        plan.parameters,
    )
    actual_initial_parameters = merge(
        plan.parameters,
        (
            beta=get(executed_parameters, :beta, plan.parameters.beta),
            gamma=get(executed_parameters, :gamma, plan.parameters.gamma),
            omega_p=get(
                executed_parameters,
                :omega_p,
                plan.parameters.omega_p,
            ),
            omega_d=get(
                executed_parameters,
                :omega_d,
                plan.parameters.omega_d,
            ),
            predictor=get(
                executed_parameters,
                :predictor,
                plan.parameters.predictor,
            ),
            strategy=get(
                executed_parameters,
                :strategy,
                plan.parameters.strategy,
            ),
            adaptive_sigma_max=get(
                executed_parameters,
                :adaptive_sigma_max,
                plan.parameters.adaptive_sigma_max,
            ),
        ),
    )
    parameter_source = get(executed, :parameter_source, :plan)
    parameter_resolution_count = get(
        executed,
        :parameter_resolution_count,
        nothing,
    )
    stage = get(executed, :stage, :not_resolved)
    selected = (
        solver=get(executed, :solver, plan.algorithm),
        scaling=plan.scaling,
        kkt=get(executed, :kkt, plan.kkt_backend),
        planned_backend=get(
            executed,
            :planned_backend,
            planned_backend_name(plan),
        ),
        planned_kkt_formulation=plan.kkt_formulation,
        requested_kkt_formulation=get(
            plan.parameters,
            :formulation,
            :auto,
        ),
        formulation_decision=get(
            plan.parameters,
            :formulation_decision,
            (
                requested=:auto,
                preferred=plan.kkt_formulation,
                selected=plan.kkt_formulation,
                reason=plan.formulation_plan.reason,
                candidates=(),
            ),
        ),
        executed_kkt_formulation=get(
            executed,
            :kkt_formulation,
            :not_executed,
        ),
        planned_factorization=get(
            plan.parameters,
            :planned_factorization,
            :not_applicable,
        ),
        executed_factorization=get(
            executed,
            :la_factorization,
            :not_executed,
        ),
        planned_regularization=get(
            plan.parameters,
            :planned_regularization,
            :not_recorded,
        ),
        executed_regularization=get(
            executed,
            :la_regularization,
            nothing,
        ),
        executed_backend=get(
            executed,
            :executed_backend,
            :not_executed,
        ),
        fallback_reason=get(
            executed,
            :fallback_reason,
            :none,
        ),
        la_backend=get(executed, :la_backend, :not_executed),
        la_executed_provider=get(executed, :la_provider, :not_executed),
        la_executed_ownership=get(executed, :la_ownership, :not_executed),
        la_fallback_reason=get(executed, :la_fallback_reason, :none),
        la_factorization=get(executed, :la_factorization, :not_executed),
        factor_diagnostics=get(executed, :factor_diagnostics, nothing),
        planned_la_backend=diagnostics_plan.la_config.selected,
        planned_la_fallback_reason=diagnostics_plan.la_config.fallback_reason,
        la_provider=diagnostics_plan.la_config.provider,
        la_ownership=diagnostics_plan.la_config.ownership,
        planned_la_provider=diagnostics_plan.la_config.provider,
        planned_la_ownership=diagnostics_plan.la_config.ownership,
        backend_resolution=get(
            executed,
            :backend_resolution,
            :planned,
        ),
        lp_formulation=get(
            executed,
            :lp_formulation,
            :not_applicable,
        ),
        gram=get(executed, :gram, plan.gram_kernel),
        equality=get(executed, :equality, :not_executed),
        planned=(
            kkt=plan.kkt_backend,
            gram=plan.gram_kernel,
        ),
        scheduling=plan.schedule,
        threads=plan.threads,
        effective_threads=get(executed, :effective_threads, plan.threads),
        fine_grained_block_tasks=get(
            executed,
            :fine_grained_block_tasks,
            plan.threads,
        ),
        fine_grained_block_partition=get(
            executed,
            :fine_grained_block_partition,
            :lpt,
        ),
        schur_threads=get(executed, :schur_threads, plan.threads),
        lp_pack_threads=get(executed, :lp_pack_threads, nothing),
        factor_threads=get(executed, :factor_threads, nothing),
        arrow_linear_solve=get(
            executed,
            :arrow_linear_solve,
            nothing,
        ),
        # `parameter_profile`/`initial_parameters` describe the parameters
        # that actually reached the core when that provenance is available.
        # Keep the pre-equilibration planner choice separately named so a
        # post-Ruiz auto selection cannot be mistaken for the plan.
        parameter_profile=executed_parameter_profile,
        initial_parameters=actual_initial_parameters,
        parameter_source,
        parameter_resolution_count,
        stage,
        executed_parameters,
        initialization=get(executed, :initialization, nothing),
        planned_parameter_profile=plan.parameter_profile,
        planned_parameters=plan.parameters,
        certificate=certificate,
    )
    effective_termination =
        termination.reason === :none ? result.termination : termination
    explicit_bits = ladder_context === nothing ?
                    nothing : ladder_context.explicit_bits
    attempt_id = ladder_context === nothing ? 1 : ladder_context.attempt_id
    attempt_record = _build_execution_attempt_record(
        diagnostics_plan,
        result,
        executed,
        effective_termination,
        certificate,
        explicit_bits,
        attempt_id,
        attempt_id,
    )
    attempts = (attempt_record,)
    ladder = if ladder_context === nothing
        nothing
    else
        elapsed = max(
            0.0,
            (time_ns() - ladder_context.rung_started_ns) / 1.0e9,
        )
        remaining_after = max(
            0.0,
            ladder_context.remaining_budget_seconds - elapsed,
        )
        status = result.status
        retry_decision = _ladder_retry_decision(
            ladder_context.plan,
            ladder_context.rung,
            status,
            remaining_after,
        )
        spec = ladder_context.plan.rungs[ladder_context.rung]
        rung_report = PrecisionAttemptReport(
            spec,
            diagnostics_plan,
            attempt_record,
            PrecisionAttemptScalarFacts(
                status,
                get(effective_termination, :reason, :none),
                elapsed,
                _working_precision_success(status),
                retry_decision,
                remaining_after,
            ),
        )
        push!(ladder_context.reports, rung_report)
        PrecisionLadderReport(
            ladder_context.plan,
            Tuple(copy(ladder_context.reports)),
        )
    end
    diagnostics = SolveDiagnostics(
        diagnostics_plan.classification,
        diagnostics_plan,
        report,
        timings,
        memory,
        selected,
        result.parameter_history,
        warnings,
        effective_termination,
        attempts,
        ladder,
    )
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
        result.timings,
        result.parameter_history,
        diagnostics,
        result.termination,
    )
end
