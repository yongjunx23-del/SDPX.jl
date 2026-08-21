# ---------------------------------------------------------------------------
# A0 — execution-attempt record construction (diagnostics only).
#
# Every helper below runs exactly once per solve, inside diagnostics-enabled
# `_attach_diagnostics`. The records are built only from facts that already
# exist in the plan, the result, and the termination record; no solver path is
# touched and no history is fabricated. `attempt_id`/`plan_id` are
# attempt-local stable indices for now (1/1).
# ---------------------------------------------------------------------------

# Canonical solver family names. The plan carries exact algorithm labels
# (`:lp_primal_dual`, `:sdp_primal_dual`),
# while termination records report the executing family (`:lp`, `:sdp`,
# `:native_soc`). Both sides of the attempt record use these canonical names
# so an ordinary LP solve does not look like a planned/executed divergence.
function _attempt_solver_family(algorithm::Symbol)
    algorithm === :lp_primal_dual && return :lp
    algorithm === :sdp_primal_dual &&
        return :sdp
    return algorithm
end

"""Downgrade reasons: an originally successful status demoted by certification
evidence in the original coordinates. These are certificate facts, never
fallback events."""
const _ATTEMPT_CERTIFICATE_DOWNGRADE_REASONS = (
    :minimal_original_coordinate_gate_failed,
    :final_certificate_failed,
)

function _attempt_planned_precision_facts(
    plan::ExecutionPlan,
    ::Type{T},
    explicit_bits::Union{Nothing,Int}=nothing,
) where {T}
    bits = explicit_bits === nothing ?
           # Legacy fallback: the attempt builder runs inside the exact
           # `setprecision(..., bits)` scope used by the solve (plan
           # construction and `_attach_diagnostics` are both under it), so
           # the active width is factual planned evidence.
           (plan.classification.arithmetic === :bigfloat ?
            precision(BigFloat) : sig_bits(T)) :
           explicit_bits
    return AttemptPrecisionFacts(
        plan.classification.arithmetic,
        # The mode is the planner's frozen decision, not the user request
        # hint: `mixed_precision_kkt=:auto` may resolve to `:off`.
        plan.backend_config.mixed_precision_mode === :off ?
        :fixed : :mixed_precision,
        bits,
    )
end

function _attempt_executed_precision_facts(
    executed::NamedTuple,
    ::Type{T},
    explicit_bits::Union{Nothing,Int}=nothing,
) where {T}
    solver = get(executed, :solver, :not_executed)
    solver === :not_executed &&
        return AttemptPrecisionFacts(_arithmetic_class(T), :not_executed, nothing)
    executed_backend = get(executed, :kkt, :not_executed)
    # `_attach_diagnostics` runs inside the exact `setprecision(..., bits)`
    # scope used by the executed solve, so `precision(BigFloat)` here is
    # factual execution evidence, not a post-hoc ambient guess.
    bits = explicit_bits === nothing ?
           (T === BigFloat ? precision(BigFloat) : sig_bits(T)) :
           explicit_bits
    return AttemptPrecisionFacts(
        _arithmetic_class(T),
        executed_backend === :mixed_precision ? :mixed_precision : :fixed,
        bits,
    )
end

"""Storage implied by an executed SDP formulation when no explicit storage
fact was recorded (the LP path always records `executed_storage`)."""
function _attempt_storage_from_formulation(formulation::Symbol)
    formulation in (
        :dense_normal_equations,
        :dense_augmented_kkt,
        :block_arrow,
    ) && return :dense
    formulation === :sparse_normal_equations && return :sparse
    return :not_executed
end

function _attempt_planned_route_facts(
    plan::ExecutionPlan,
    ::Type{T},
    explicit_bits::Union{Nothing,Int}=nothing,
) where {T}
    return ExecutionRouteFacts(
        _attempt_solver_family(plan.algorithm),
        plan.kkt_formulation,
        plan.storage_plan.storage,
        plan.la_config.provider,
        _attempt_planned_precision_facts(plan, T, explicit_bits),
        plan.threads,
    )
end

function _attempt_executed_route_facts(
    plan::ExecutionPlan,
    executed::NamedTuple,
    ::Type{T},
    explicit_bits::Union{Nothing,Int}=nothing,
) where {T}
    solver = get(executed, :solver, :not_executed)
    formulation = if solver === :not_executed
        :not_executed
    elseif haskey(executed, :kkt_formulation)
        executed.kkt_formulation
    else
        get(executed, :lp_formulation, :not_executed)
    end
    # The finalized LP route payload is the authoritative provider for the
    # dedicated LP path: sparse routes run provider-owned frozen-CSC kernels
    # and the diagonal reduced route owns its own kernel, neither of which is
    # a BLAS/LAPACK `la_provider`.
    payload = get(executed, :lp_route_payload, nothing)
    provider = payload isa LPRoutePlan ?
               payload.provider :
               get(executed, :la_provider, :not_executed)
    storage = get(
        executed,
        :executed_storage,
        solver === :not_executed ?
        :not_executed : _attempt_storage_from_formulation(formulation),
    )
    threads = solver === :not_executed ?
              0 : get(executed, :effective_threads, plan.threads)
    return ExecutionRouteFacts(
        solver === :not_executed ? :not_executed : _attempt_solver_family(solver),
        formulation,
        storage,
        provider,
        _attempt_executed_precision_facts(executed, T, explicit_bits),
        threads,
    )
end

function _attempt_planned_la_fallback_event(plan::ExecutionPlan)
    reason = plan.la_config.fallback_reason
    reason === :none && return nothing
    return FallbackEvent(
        :la_route,
        reason,
        true,
        :plan,
        (
            selected=plan.la_config.selected,
            provider=plan.la_config.provider,
            fallback_chain=plan.la_config.fallback_chain,
        ),
    )
end

function _attempt_runtime_backend_fallback_event(
    plan::ExecutionPlan,
    executed::NamedTuple,
)
    reason = get(executed, :fallback_reason, :none)
    reason in (:none, :not_executed, :not_applicable) && return nothing
    executed_backend = get(executed, :executed_backend, :not_executed)
    planned_backend = planned_backend_name(plan)
    # A runtime reason is not by itself evidence of a route switch.  In
    # particular, a terminal factor failure may leave the planned backend in
    # place, while the dedicated LP path resolves its intentionally deferred
    # backend after row presolve.  Neither is a fallback event.  The event is
    # created only for an observed implementation change from a non-deferred
    # planned backend.
    plan.backend_config.deferred && return nothing
    executed_backend === planned_backend && return nothing
    chain = plan.backend_config.fallback_chain
    # Fail-closed authorization: the actual executed backend must be an exact
    # target of the plan's fallback chain (mixed plans list their structural
    # fallback `:dense_cholesky` alongside the mixed route). The planned route
    # itself is never an authorization: a forged executed backend/ reason
    # must not become "authorized" merely because it names the plan.
    authorized = executed_backend in chain
    return FallbackEvent(
        :backend_structural,
        reason,
        authorized,
        :runtime_executed,
        (
            executed_backend=executed_backend,
            planned_backend,
            fallback_chain=chain,
        ),
    )
end

function _attempt_runtime_la_fallback_event(
    plan::ExecutionPlan,
    executed::NamedTuple,
)
    reason = get(executed, :la_fallback_reason, :none)
    reason in (:none, :not_executed, :not_applicable) && return nothing
    # `la_fallback_reason` is overloaded: it records terminal provider
    # failures (`:la_factor_failed`, `:la_equality_factor_failed`) and planned
    # legacy provenance alike, and neither alone demonstrates a route
    # divergence. Only evidence *inside the executed record* that a
    # rank-revealing QR equality route actually ran may build an LA equality
    # fallback event; planned/runtime identical reasons must not duplicate.
    kind = :la_equality
    executed_equality = get(executed, :equality, :not_executed)
    demonstrated = executed_equality === :rank_revealing_qr
    demonstrated || return nothing
    chain = plan.la_config.fallback_chain
    authorized = :rank_revealing_qr in chain
    return FallbackEvent(
        kind,
        reason,
        authorized,
        :runtime_executed,
        (
            selected=plan.la_config.selected,
            fallback_chain=chain,
            planned_fallback_reason=plan.la_config.fallback_reason,
            executed_equality=executed_equality,
        ),
    )
end

"""Ordered fallback events. Only events whose causal order is derivable from
explicit facts are ordered: planned LA provenance first, then a runtime
structural backend fallback (the mixed backend switch), then a demonstrated
runtime LA equality fallback. Unknown or undemonstrated provenance is never
turned into a guessed event. Regularization and certification downgrades are
not fallbacks and never appear here."""
function _attempt_fallback_events(
    plan::ExecutionPlan,
    executed::NamedTuple,
)
    events = FallbackEvent[]
    planned = _attempt_planned_la_fallback_event(plan)
    planned === nothing || push!(events, planned)
    backend = _attempt_runtime_backend_fallback_event(plan, executed)
    backend === nothing || push!(events, backend)
    la = _attempt_runtime_la_fallback_event(plan, executed)
    la === nothing || push!(events, la)
    return Tuple(events)
end

function _attempt_regularization_facts(
    result::SDPResult{T},
    executed::NamedTuple,
) where {T}
    events = NamedTuple[]
    if result.regularizations > 0
        push!(
            events,
            (
                kind=:factor_regularization,
                count=result.regularizations,
                source=:solve_counter,
            ),
        )
    end
    initialization = get(executed, :initialization, nothing)
    if initialization isa NamedTuple
        for counter in (
            :regularization_attempts,
            :native_regularization_attempts,
        )
            count = get(initialization, counter, 0)
            if count isa Integer && count > 0
                push!(
                    events,
                    (
                        kind=:factor_regularization,
                        count=count,
                        source=:kkt_cold_start,
                        counter=counter,
                    ),
                )
            end
        end
    end
    return RegularizationFacts(
        result.regularizations,
        Tuple(events),
        get(executed, :la_regularization, nothing),
    )
end

function _attempt_certificate_facts(
    termination::NamedTuple,
    certificate::NamedTuple,
)
    available = get(certificate, :available, false)
    minimal = get(certificate, :minimal_gate, nothing)
    method = available ? :detailed :
             minimal isa NamedTuple ? :minimal_gate : :not_applicable
    minimal_valid =
        minimal isa NamedTuple ? get(minimal, :valid, nothing) : nothing
    valid = available ? get(certificate, :valid, nothing) : minimal_valid
    # A certification downgrade is a certificate fact, not a fallback event.
    downgrade = get(termination, :reason, :none) in
                _ATTEMPT_CERTIFICATE_DOWNGRADE_REASONS
    return CertificateFacts(
        available,
        method,
        get(certificate, :reason, :not_recorded),
        valid,
        downgrade,
        minimal_valid,
    )
end

function _attempt_prepared_reuse_facts()
    # A6 owns truthful attempt-level reuse reporting; until then these
    # explicit facts are always unavailable/not reused.
    return PreparedReuseFacts(false, false, :unavailable_until_a6)
end

function _build_execution_attempt_record(
    plan::ExecutionPlan,
    result::SDPResult{T},
    executed::NamedTuple,
    termination::NamedTuple,
    certificate::NamedTuple,
    explicit_bits::Union{Nothing,Int}=nothing,
    attempt_id::Int=1,
    plan_id::Int=1,
) where {T}
    return ExecutionAttemptRecord(
        attempt_id,
        plan_id,
        _attempt_planned_route_facts(plan, T, explicit_bits),
        _attempt_executed_route_facts(plan, executed, T, explicit_bits),
        _attempt_fallback_events(plan, executed),
        _attempt_regularization_facts(result, executed),
        _attempt_certificate_facts(termination, certificate),
        _attempt_prepared_reuse_facts(),
        result.status,
        get(termination, :reason, :none),
    )
end

