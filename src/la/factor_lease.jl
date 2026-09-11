# src/la/factor_lease.jl
#
# ADR-002 §4/§5: the logical Newton lease.
#
# Ownership split (ADR-001 §2, reaffirmed by ADR-002 §5):
#
#   provider owns  -> physical factor, pivots, block grammar, factor scratch,
#                     the provider_generation counter
#   SDPX owns      -> the current Newton operator, the authorized factor input,
#                     and the logical lease `matrix_epoch -> provider_generation`
#
# §4 is the load-bearing rule:
#
#   on ANY failed `refactor_numeric!`, revoke the logical lease BEFORE reading
#   any provider status, regardless of whether the provider retained the
#   physical factor.
#
# §8/§9 established that *both* providers do retain the physical factor and the
# previous success flag across a **preflight** rejection (BFLA via
# `status = FactorStatus(:unprepared, nothing)` after the checks; MFLA via
# `invalidate!(cache)` after the checks). So an adapter cannot tell
# "old factor still valid" from "old factor still valid FOR THE OLD REQUEST" by
# inspecting the provider. It must not try. Hence the ordering below: `_revoke!`
# runs first on every non-committing path.

# ---------------------------------------------------------------------------
# 1. lease state machine
# ---------------------------------------------------------------------------

"""
    LeaseState

The logical lease state. `LeaseRevoked` is absorbing: a revoked lease is never
re-validated. Freshness comes only from a new successful `refactor_numeric!`.
"""
@enum LeaseState begin
    LeaseVacant       # no numeric factor was ever bound
    LeaseBound        # bound to a provider generation that is currently authorized
    LeaseRevoked      # explicitly revoked; a solve must fail closed
    LeaseSymbolicOnly # symbolic structure exists, no numeric factor authorized
end

"""
    LeaseEvent

Why a lease transition happened. Recorded so the reason a solve failed is
auditable without re-deriving it.
"""
@enum LeaseEvent begin
    EvPrepared
    EvRefactorSucceeded
    EvRefactorPreflightRejected
    EvRefactorCommitFailed
    EvRefactorThrew
    EvSolveRefused
    EvInvalidated
    EvSymbolicRebuilt
end

"""
    FactorLease

The logical lease: `matrix_epoch -> provider_generation`. Nothing here is a
physical factor. `provider_generation` is a *number the provider gave us*, not a
handle into its storage: SDPX never dereferences it.

`physical_retention_expected` records what we believe the provider did with its
physical factor across the last failure. It is **evidence, never
authorization** — the whole point of ADR-002 §4 is that this belief cannot
re-authorize a new request, and `refactor_numeric!` sets it without ever having
asked.
"""
mutable struct FactorLease
    state::LeaseState
    matrix_epoch::UInt64
    provider_generation::UInt64
    bound_request_digest::UInt64
    bound_rhs_cols::Int
    bound_op::SolveOp
    revocation_event::LeaseEvent
    revocation_detail::String
    physical_retention_expected::Bool
    n_revocations::Int
end

FactorLease() = FactorLease(LeaseVacant, UInt64(0), UInt64(0), UInt64(0), 0, OpPrepareFactor,
                            EvPrepared, "", false, 0)

"""
    is_valid(lease) -> Bool

The only authorization predicate in the system. A solve consults this and
nothing else.
"""
is_valid(l::FactorLease) = l.state === LeaseBound

"""
    authorize(lease)

The single authorization gate. False means the solve fails closed. There is no
"stale but probably fine" branch anywhere in this file.
"""
authorize(l::FactorLease) = is_valid(l)

"""
    physical_factor_retained(lease) -> Bool

True iff the provider said it kept its physical factor across the last failure.
ADR-002 §4: this fact must never be used to authorize a solve. It is exposed so
that the tests can prove that a retained physical factor still does not
authorize anything.
"""
physical_factor_retained(l::FactorLease) = l.physical_retention_expected

function _bind!(l::FactorLease, matrix_epoch::UInt64, gen::UInt64, digest::UInt64,
                ncols::Int, op::SolveOp, ev::LeaseEvent)
    l.state = LeaseBound
    l.matrix_epoch = matrix_epoch
    l.provider_generation = gen
    l.bound_request_digest = digest
    l.bound_rhs_cols = ncols
    l.bound_op = op
    l.revocation_event = ev
    l.revocation_detail = ""
    l.physical_retention_expected = false
    l
end

"""
    _revoke!(lease, ev, detail; physical_retention_expected=false)

Revoke the logical lease. Called on every path that does not end in a committed
numeric factor. Idempotent-by-counting: each revocation is recorded, so a test
can assert *when* it happened relative to reading provider status.
"""
function _revoke!(l::FactorLease, ev::LeaseEvent, detail::AbstractString;
                  physical_retention_expected::Bool=false)
    l.state = LeaseRevoked
    l.provider_generation = UInt64(0)
    l.bound_request_digest = UInt64(0)
    l.bound_rhs_cols = 0
    l.revocation_event = ev
    l.revocation_detail = String(detail)
    l.physical_retention_expected = physical_retention_expected
    l.n_revocations += 1
    l
end

"""
    revoke!(handle, reason)

Public revocation (ADR-002 §2 `invalidate_numeric!` semantics).
"""
function revoke!(h, reason::AbstractString="invalidate_numeric!")
    _revoke!(h.lease, EvInvalidated, reason)
    h
end

# ---------------------------------------------------------------------------
# 2. the handle: logical lease + provider generation counter
# ---------------------------------------------------------------------------

"""
    FactorHandle

Binds a provider (which owns the physical factor) to the logical lease (which
SDPX owns) and to the cheap summary cache.

The handle deliberately does **not** expose the provider's factor storage: there
is no accessor returning a matrix, and `provider_generation` is only ever
compared, never dereferenced.
"""
mutable struct FactorHandle{P}
    provider::P
    request::FactorRequest
    lease::FactorLease
    provider_generation::UInt64
    summary_cache::FactorSummary
    summary_valid::Bool
    digest::UInt64
    diagnostics::DeepDiagnosticsGuard
    # Counters that make hot-path claims measurable.
    n_hot_solves::Int
    n_batched_calls::Int
    n_per_column_calls::Int
    n_factor_copies::Int
end

"""
    request_digest(request) -> UInt64

A stable digest of the request facts that must match for a bound factor to still
be authorized. This is SDPX-side bookkeeping; it is *not* a provider fact.
"""
function request_digest(req::FactorRequest)
    h = 0xcbf29ce484222325 % UInt64
    mix(x::UInt64) = (h = (h ⊻ x) * 0x100000001b3 % UInt64)
    mix(UInt64(req.op))
    mix(UInt64(req.scalar.family))
    mix(UInt64(req.scalar.min_bits % UInt32))
    mix(UInt64(req.scalar.exact_bits))
    mix(UInt64(req.shape.rows % UInt32))
    mix(UInt64(req.shape.cols % UInt32))
    mix(UInt64(req.shape.rectangular))
    mix(UInt64(req.triangle))
    mix(UInt64(req.indices.index_bits % UInt32))
    mix(UInt64(req.concurrency.allow_threads))
    mix(UInt64(req.concurrency.serial_required))
    mix(UInt64(req.rhs.needs))
    mix(UInt64(req.known_nnz % UInt32))
    h
end

function factor_handle(provider, req::FactorRequest)
    empty = FactorSummary(UInt64(0), StatusUnprepared, 0, 0, OpFactorSummary, 0, 0, 0,
                          :none, UInt64(0), UInt64(0), false)
    FactorHandle(provider, req, FactorLease(), UInt64(0), empty, false, request_digest(req),
                 DeepDiagnosticsGuard(), 0, 0, 0, 0)
end

# ---------------------------------------------------------------------------
# 3. the adapter-facing raw provider API
# ---------------------------------------------------------------------------

"""
    raw_provider_generation(provider) -> UInt64

The provider's own generation counter. Owned by the provider (ADR-002 §5).
"""
function raw_provider_generation end

"""
    raw_prepare!(provider, req) -> Nothing

Both providers allocate capacity here. MUST NOT solve and MUST NOT factor.
"""
function raw_prepare! end

"""
    raw_refactor!(provider, values) -> status symbol

The provider's numeric factorization. Two-phase in both providers (ADR-002
§8/§9): throwing validation checks first, then a commit marker, then the numeric
work. May return a non-`:success` status without throwing.
"""
function raw_refactor! end

"""
    raw_status(provider) -> provider status symbol

The provider's *physical* status. ADR-002 §4: reading this must never influence
whether a solve is authorized. The lease is already revoked by then.
"""
function raw_status end

"""
    raw_solve!(provider, dest, rhs, op) -> Int

Perform the solve. Returns the number of columns written.
"""
function raw_solve! end

"""
    raw_summary(provider) -> NamedTuple

Provider-reported scalar facts. MUST be O(1) and MUST NOT recompute inertia or
allocate a matrix.
"""
function raw_summary end

"""
    raw_pivots(provider) -> PivotMetadata

Provider pivot metadata under a named grammar.
"""
function raw_pivots end

"""
    raw_snapshot(provider) -> Matrix

The explicit, expensive operator/factor copy. Only `copy_operator_snapshot` may
call this.
"""
function raw_snapshot end

"""
    raw_deep_check(provider, audit_spec) -> NamedTuple

The explicit deep numeric audit. Only `inspect_factor` may call this. NEVER on a
hot solve path.
"""
function raw_deep_check end

"""
    raw_invalidated!(provider) -> Nothing

Tell the provider to drop its numeric factor. SDPX never drops provider storage
itself.
"""
function raw_invalidated! end

"""
    raw_retained_physical(provider) -> Bool

Did the provider keep its physical factor across the last failure? Evidence
only — see [`physical_factor_retained`](@ref).
"""
raw_retained_physical(provider) = false

# ---------------------------------------------------------------------------
# 4. the contract entry points
# ---------------------------------------------------------------------------

"""
    prepare_factor!(handle) -> Admission

ADR-002 §2 `prepare_factor!`: allocate capacity, establish symbolic/shape/
precision. No solve. Refuses explicitly if the request is not admissible.
"""
function prepare_factor!(h::FactorHandle)
    facts = capabilities(h.request, h.provider)
    adm = admit(h.request, facts)
    adm.allowed || return adm
    raw_prepare!(h.provider, h.request)
    # A new operator is a new matrix epoch, synchronously. If a numeric factor
    # was bound to the previous epoch it stops being authorized right here, even
    # though the provider may still hold that physical factor.
    was_bound = is_valid(h.lease)
    h.lease.matrix_epoch += UInt64(1)
    was_bound && _revoke!(h.lease, EvSymbolicRebuilt,
                          "prepare_factor! for a new operator (matrix_epoch " *
                          "$(h.lease.matrix_epoch)); previous numeric lease revoked")
    h.lease.state = LeaseSymbolicOnly
    h.lease.revocation_event = EvPrepared
    h.lease.revocation_detail = ""
    h.summary_valid = false
    adm
end

"""
    refactor_numeric!(handle, values) -> NamedTuple

ADR-002 §2 `refactor_numeric!` + §4 failure semantics.

The ordering here is the entire point of the task and is enforced structurally:

1. refusal check (no provider mutation);
2. digest check;
3. `raw_refactor!` inside a `try`;
4. **on ANY failure — thrown, returned non-success, or returned with a
   digest/shape mismatch — `_revoke!` runs BEFORE `raw_status` is read.**

Step 4 is why a preflight rejection that retains the physical factor cannot be
reused by a new Newton request. There is no code path in this function that
reads provider status before revoking.
"""
function refactor_numeric!(h::FactorHandle, values)
    facts = capabilities(h.request, h.provider)
    adm = admit(h.request, facts)
    if !adm.allowed
        # ADR-002 §4 requires revocation on ANY failure, and an admission refusal
        # is a failure. It is also the one failure that never reaches the provider,
        # which is why it used to fall through the revoke-before-status discipline
        # above: there is no provider status to read, so there was nothing to order
        # against. But `h.request` has already changed, so a lease left bound here
        # authorises a request no factor was ever built for -- measured as
        # `is_valid(lease) == true` with `bound_request_digest != request_digest`,
        # and `commit_failure_observation` throwing because it refuses to run while
        # a lease is valid (S07-F1). Revoking here is what makes the §4 guarantee
        # hold for every exit from this function rather than all but one.
        _revoke!(h.lease, EvRefactorPreflightRejected,
                 "request not admitted: $(adm.detail)")
        return (ok=false, status=StatusUnsupported, generation=h.provider_generation,
                revoked=true, provider_status=nothing, admission=adm,
                detail="request not admitted: $(adm.detail)")
    end

    digest = request_digest(h.request)
    # --- phase 1: the provider call, entirely inside a try ---
    outcome = try
        raw_refactor!(h.provider, values)
    catch err
        # A THROW. Both providers reach this with their previous physical factor
        # and previous success flag intact if the throw came from preflight.
        # We do not look: revoke first, ask questions later.
        _revoke!(h.lease, EvRefactorThrew,
                 "refactor_numeric! threw $(typeof(err)); lease revoked before any provider status read";
                 physical_retention_expected=true)
        h.summary_valid = false
        return (ok=false, status=StatusError, generation=h.provider_generation,
                revoked=true, provider_status=nothing, admission=adm,
                detail=string(err))
    end

    raw_status_sym = Symbol(lowercase(string(outcome)))
    std = standardize_status(_vocab(h.provider), raw_status_sym)
    ok = std === StatusOk

    if !ok
        # --- phase 2a: provider returned a non-success status ---
        # Revoke BEFORE reading raw_status(). The provider is entitled to keep
        # both its storage and its previous success flag here.
        _revoke!(h.lease, EvRefactorCommitFailed,
                 "refactor_numeric! returned $(raw_status_sym) for a NEW request; " *
                 "logical lease revoked regardless of retained physical factor";
                 physical_retention_expected=true)
        h.summary_valid = false
        return (ok=false, status=std, generation=h.provider_generation, revoked=true,
                provider_status=nothing, admission=adm, detail=string(raw_status_sym))
    end

    # --- phase 3 (success): commit the new lease ---
    h.provider_generation = raw_provider_generation(h.provider)
    h.digest = digest
    _bind!(h.lease, h.lease.matrix_epoch, h.provider_generation, digest,
           h.request.rhs.ncols, h.request.op, EvRefactorSucceeded)
    h.summary_valid = false
    (ok=true, status=StatusOk, generation=h.provider_generation, revoked=false,
     provider_status=nothing, admission=adm, detail="")
end

"""
    commit_failure_observation(handle) -> NamedTuple

Cleanup-phase observation, explicitly NAMED as such. This is the only function
that reads the provider's physical status after a failure, and it is not on the
authorization path: it returns evidence for the caller's log, and it asserts the
lease is already revoked. Calling it after authorization would be a bug; it
throws if the lease is still bound.
"""
function commit_failure_observation(h::FactorHandle)
    is_valid(h.lease) && throw(ArgumentError(
        "commit_failure_observation called with a valid lease: provider status must never " *
        "be consulted while a lease is authorizable (ADR-002 §4)"))
    (provider_status=standardize_status(_vocab(h.provider), Symbol(lowercase(string(raw_status(h.provider))))),
     physical_retention_expected=raw_retained_physical(h.provider),
     lease_state=h.lease.state,
     lease_revoked_at=h.lease.revocation_event,
     n_revocations=h.lease.n_revocations)
end

function _vocab(provider)
    n = Symbol(lowercase(string(declared_facts(provider).provider_name)))
    n
end

"""
    solve_into!(dest, handle, rhs, op) -> SolveOutcome

ADR-002 §2 `solve_into!(dest, handle, rhs, op)`: vector/matrix RHS, N/T
explicitly named. Must NOT allocate a fresh factor, must NOT silently transpose,
must NOT call deep diagnostics, and must NOT copy the factor.

Hot-path budget, all of it enforced below and measured by the tests:

- one `admit`-independent lease check (`authorize`) — no `capabilities` call;
- zero `raw_deep_check` calls (`inspect_factor` is elsewhere);
- zero `raw_snapshot` calls (`copy_operator_snapshot` is elsewhere);
- batched RHS goes to `raw_solve!` as one call; a per-column-only provider is
  driven by an *explicit, visible* loop, never by a hidden provider fallback.
"""
function solve_into!(dest::AbstractMatrix, h::FactorHandle, rhs::AbstractMatrix, op::SolveOp;
                     needs::Union{Nothing,MultiRHSKind}=nothing)
    h.n_hot_solves += 1
    if op !== OpSolveN && op !== OpSolveT && op !== OpSolveAdjoint
        return solve_refused(RefuseOperation,
            "solve_into! takes OpSolveN/OpSolveT/OpSolveAdjoint; got $(op)")
    end
    # FAIL CLOSED. This is the single authorization test, and it runs before any
    # provider interaction at all.
    authorize(h.lease) || return solve_refused(RefuseOperation,
        "logical lease is $(h.lease.state) (last event $(h.lease.revocation_event)); " *
        "refusing to reuse a physical factor that may predate this request — " *
        "$(h.lease.revocation_detail)")

    facts = declared_facts(h.provider)   # declared metadata only: no factorization, no benchmark
    # The op is named explicitly and the triangle stays exactly what the caller's
    # operator is; a transpose solve is NOT expressed by silently flipping the
    # triangle convention. If the provider cannot do this op/triangle pairing, it
    # is refused.
    #
    # `needs` defaults to the provider's DECLARED multi-RHS kind, not to a
    # batching demand: `solve_into!` is the routine solve path, and requiring
    # batching there would refuse every per-column provider for any 2-column RHS.
    # A caller that genuinely requires batching passes `needs=MultiRHSBatched`
    # and is then refused explicitly if the provider only loops columns.
    ncols = size(rhs, 2)
    need = needs === nothing ?
           (ncols == 1 ? MultiRHSUnsupported : facts.solve.multi_rhs) : needs
    req = FactorRequest(op, h.request.scalar, h.request.shape;
                        indices=h.request.indices, triangle=h.request.triangle,
                        concurrency=h.request.concurrency,
                        rhs=RHSKind(is_matrix=true, ncols=ncols, needs=need))
    adm = admit(req, facts)
    if !adm.allowed
        # A solve that cannot be performed as requested is REFUSED; it is not
        # silently transposed, narrowed to one column, or run serially.
        return solve_refused(adm.reason, adm.detail)
    end

    written = if facts.solve.multi_rhs in (MultiRHSBatched, MultiRHSBlocked)
        h.n_batched_calls += 1
        raw_solve!(h.provider, dest, rhs, op)
    else
        # Per-column provider: the loop is OURS and is counted, so that
        # "multi_rhs = per-column" is a visible fact at the call site and not a
        # throughput claim.
        w = 0
        for j in 1:ncols
            h.n_per_column_calls += 1
            w += raw_solve!(h.provider, view(dest, :, j), view(rhs, :, j), op)
        end
        w
    end
    written == ncols || return solve_refused(RefuseOperation,
        "provider wrote $(written) of $(ncols) columns")
    solve_ok(h.provider_generation)
end

"""
    factor_summary(handle) -> FactorSummary

ADR-002 §2: **O(1) small report**. No matrix allocation, no inertia
recomputation, no factor copy. The numeric payload comes from the provider's
cached scalar facts via `raw_summary`, which the provider contract requires to
be O(1) as well.
"""
function factor_summary(h::FactorHandle)
    if !h.summary_valid
        raw = raw_summary(h.provider)
        pv = raw_pivots(h.provider)
        h.summary_cache = FactorSummary(
            h.provider_generation,
            standardize_status(_vocab(h.provider), get(raw, :status, :unprepared)),
            Int(get(raw, :n, 0)),
            Int(get(raw, :nrhs_last, 0)),
            SolveOp(Int(get(raw, :op_last, Int(OpFactorSummary)))),
            Int(get(raw, :rank, 0)),
            Int(get(raw, :sign, 0)),
            Int(get(raw, :nnz_factor, 0)),
            pv.pivot_grammar,
            h.lease.matrix_epoch,
            h.provider_generation,
            is_valid(h.lease),
        )
        h.summary_valid = true
    end
    h.summary_cache
end

"""
    pivot_report(handle) -> PivotReport

Step 3: standardized pivot-metadata interpretation. The provider's raw encoding
is interpreted only through its named grammar.
"""
pivot_report(h::FactorHandle) = interpret_pivots(raw_pivots(h.provider))

"""
    copy_operator_snapshot(handle) -> Matrix

ADR-002 §2: explicit, expensive copy; MUST NOT be used on a hot path. This is
the *only* function in the contract that copies a factor, and it counts itself.
"""
function copy_operator_snapshot(h::FactorHandle)
    note_factor_copy!(h.diagnostics)
    h.n_factor_copies += 1
    raw_snapshot(h.provider)
end

"""
    inspect_factor(handle, audit_spec)

ADR-002 §2: explicit deep check; MUST NOT substitute for the routine numerical
residual. This is the only function that calls `raw_deep_check`.
"""
function inspect_factor(h::FactorHandle, audit_spec)
    note_deep_call!(h.diagnostics)
    raw_deep_check(h.provider, audit_spec)
end

"""
    invalidate_numeric!(handle)

ADR-002 §2: revoke the numeric factor only. MUST NOT revoke the symbolic lease.
"""
function invalidate_numeric!(h::FactorHandle)
    _revoke!(h.lease, EvInvalidated, "invalidate_numeric!")
    raw_invalidated!(h.provider)
    h.summary_valid = false
    h.lease.state == LeaseRevoked && h.lease.matrix_epoch > UInt64(0)
    h
end

"""
    hot_path_counters(handle) -> NamedTuple

Instrumentation for the acceptance claims: how many solve calls, how many went
out batched vs per-column, how many factor copies, and how many deep
diagnostics were invoked from anywhere through this handle.
"""
hot_path_counters(h::FactorHandle) = (
    hot_solves=h.n_hot_solves,
    batched_calls=h.n_batched_calls,
    per_column_calls=h.n_per_column_calls,
    factor_copies=h.n_factor_copies,
    deep_calls=h.diagnostics.deep_calls,
    guard=deep_snapshot(h.diagnostics),
)

# ---------------------------------------------------------------------------
# 5. one-way shim for the old entry points (integrator-owned)
# ---------------------------------------------------------------------------

"""
    LegacyFactorEntry

The old entry-point vocabulary. The shim is **one-way**: new contract -> old
call shape. There is deliberately no `legacy_to_new` inverse, because the old
entries cannot express the facts the new request needs (operation, multi-RHS
kind, thread scope), and guessing them is exactly what ADR-002 §3 forbids.

Owned by the integrator (I01/I02/I03); this file only defines the shape.
"""
struct LegacyFactorEntry
    name::Symbol
    factor_kind::Symbol
    n::Int
    m::Int
    with_pivoting::Bool
end

"""
    legacy_shim(handle) -> LegacyFactorEntry

Project the new handle onto the old vocabulary. One-way by construction: the
result cannot be turned back into a `FactorHandle`, and it carries no
authorization.
"""
function legacy_shim(h::FactorHandle)
    LegacyFactorEntry(:legacy_shim, _legacy_kind(h.request), h.request.shape.cols,
                      h.request.shape.rows, h.request.triangle !== TriangleUnused)
end

function _legacy_kind(req::FactorRequest)
    req.shape.rectangular && return :qr
    req.triangle === TriangleUnused && return :lu
    req.scalar.family === ArithMultiFloat && return :ldlt_mf
    return req.scalar.min_bits > 64 ? :ldlt_bf : :ldlt
end

"""
    legacy_shim_authorizes(entry) -> false

The shim never authorizes a solve. The old entry points must go through the new
lease; a legacy struct carries no generation and therefore no authority.
"""
legacy_shim_authorizes(::LegacyFactorEntry) = false

# The one-way property is enforced structurally, not merely asserted: a legacy
# entry carries no `provider_generation`, so building a handle from one would
# manufacture authority that no provider ever granted. Refusing here means the
# only way to a `FactorHandle` is through a real provider plus a real request.
function factor_handle(entry::LegacyFactorEntry, ::FactorRequest)
    throw(ArgumentError(
        "factor_handle cannot be built from $(typeof(entry)) ($(entry.name)): a legacy " *
        "entry point carries no provider generation and therefore no lease authority. " *
        "The shim is one-way (ADR-002 §5); re-enter through factor_handle(provider, request)."))
end
