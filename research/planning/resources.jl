# S06 — setup resource admission: memory ledger and the session thread budget.
#
# Card step 2: "将coarse锥任务、LA kernel和BLAS预算独立描述；admission纳入fill/
# owned标量与fallback内存."
#
# Two resources, two separate problems:
#
#   * **Memory** is admitted from an explicit ledger whose terms are named so no
#     term can hide: the declared pattern, the modelled factor *fill*, the
#     *owned scalars* (per-block cone operators and iterate vectors), provider
#     workspace, and a *fallback* allowance for the route we would have to build
#     if the chosen one fails. A ledger whose total exceeds the declared budget
#     is refused, and the refusal is thrown by the execution boundary *before*
#     the first allocation (`reserve_setup_memory`), not discovered afterwards.
#
#   * **Threads** are a session resource (ADR-001 §6), not a synonym for
#     `Threads.nthreads()`. The three consumers described in `costs.jl` are
#     reconciled by the inherited `ThreadBudget` rule — exactly one parallel
#     layer is ever active — so the limits are never multiplied and nested
#     doubling is impossible by construction. The dense-BLAS thread count is
#     process-global and therefore *not* session-local: a session may hold it
#     only inside an exclusive scope (`with_session_thread_scope`), which
#     restores the previous global value on exit and records every attempt,
#     grant and serialization so contention is visible rather than silent.
#
# Nothing here allocates on behalf of a plan. Admission is arithmetic; the
# allocation happens only in `execute_setup_plan` after admission has passed.

# ---------------------------------------------------------------------------
# Thread capacity and the packet's tiers
# ---------------------------------------------------------------------------

"""
    ThreadCapacity

The thread capacity of the *host*, measured once and carried explicitly in the
setup context, so a plan's reproducibility does not depend on ambient global
state.

Fields: `julia_threads` (`Threads.nthreads()`), `blas_threads` (the active
dense BLAS backend's own count, measured through `blas_threads()`), `cpu_threads`
(`Sys.CPU_THREADS`).
"""
struct ThreadCapacity
    julia_threads::Int
    blas_threads::Int
    cpu_threads::Int
end

"""Measure the host's thread capacity. Read-only; changes no global setting."""
function thread_capacity()
    julia_threads = Int(Threads.nthreads())
    blas_count = Int(blas_threads())
    cpu_threads = Int(Sys.CPU_THREADS)
    return ThreadCapacity(julia_threads, blas_count, cpu_threads)
end

"""
    SETUP_THREAD_TIERS

The thread tiers the packet names: 1, 2, 4, 16, 64. A tier is only `:supported`
when the host can actually supply it.
"""
const SETUP_THREAD_TIERS = (1, 2, 4, 16, 64)

"""
    thread_tier_status(requested, capacity) -> Symbol

`:supported` | `:off_tier` | `:unsupported` | `:invalid`.

`unsupported` is a first-class answer, not a failure to be papered over. The
comparison is against `min(capacity.julia_threads, capacity.cpu_threads)`: a
tier this *process* cannot supply is unsupported here even when the hardware
could, because a Julia process cannot grow its thread pool. So on a host with
`Sys.CPU_THREADS == 4` the 16- and 64-thread tiers are `:unsupported` in every
configuration, and the 2/4 tiers read `:supported` only in a process started
with enough threads — which is why the receipt records the capacity alongside
the status instead of a bare boolean.
"""
function thread_tier_status(requested::Integer, capacity::ThreadCapacity)
    req = Int(requested)
    req < 1 && return :invalid
    req in SETUP_THREAD_TIERS || return :off_tier
    req <= min(capacity.julia_threads, capacity.cpu_threads) && return :supported
    return :unsupported
end

# ---------------------------------------------------------------------------
# The session thread budget
# ---------------------------------------------------------------------------

"""
    SessionThreadBudget

The thread budget of **one session**, built from the three independent consumer
descriptions and the inherited single-active-layer rule.

Fields:
- `requested_threads`: what the caller asked for (recorded, never lost).
- `capacity`: the [`ThreadCapacity`](@ref) the plan was built against.
- `budget`: the inherited `ThreadBudget`; exactly one layer is active.
- `cone_available`, `la_available`, `blas_available`: the widths the three
  consumers *offer*, kept side by side and never multiplied.
- `active_consumer`: which single consumer owns the granted threads
  (`:none` in `:serial`).
- `tier_status`: [`thread_tier_status`](@ref) for the request.
- `reasons`: ordered `Symbol`s explaining the decision.

Invariant (asserted by `validate_session_thread_budget`): the granted counts
inside `budget` never exceed `requested_threads`, never exceed the capacity of
the layer they are granted on, and no granted count is the product of two
consumers' widths.
"""
struct SessionThreadBudget
    requested_threads::Int
    capacity::ThreadCapacity
    budget::ThreadBudget
    cone_available::Int
    la_available::Int
    blas_available::Int
    active_consumer::Symbol
    tier_status::Symbol
    reasons::Vector{Symbol}
end

"""
    session_thread_budget(; requested_threads, capacity=thread_capacity(),
                          cone_task_width, la_kernel_width, blas_width,
                          provider_available=false) -> SessionThreadBudget

Reconcile the three independent consumer widths with the requested thread count
under the inherited rule that exactly one parallel layer runs.

* A request of one thread pins every layer to one (`:serial`) regardless of how
  wide the problem is — this is what makes a 1-thread request respected inside a
  multi-threaded process, and the width fields record that parallelism was
  *available* and deliberately not used.
* Otherwise the widest consumer that can actually use more than one thread is
  granted `min(requested, capacity of that layer, its own available width)`.
  Cone tasks take the outer Julia layer; a wide BLAS panel takes the
  provider/BLAS layer. Both are never granted at once.
"""
function session_thread_budget(;
    requested_threads::Integer,
    capacity::ThreadCapacity=thread_capacity(),
    cone_task_width::Integer=1,
    la_kernel_width::Integer=1,
    blas_width::Integer=1,
    provider_available::Bool=false,
)
    requested = max(Int(requested_threads), 1)
    cone = max(Int(cone_task_width), 1)
    la = max(Int(la_kernel_width), 1)
    blas_w = max(Int(blas_width), 1)
    tier = thread_tier_status(requested, capacity)
    reasons = Symbol[Symbol("requested_threads_$(requested)")]
    tier === :unsupported && push!(reasons, :requested_tier_unsupported_on_host)
    tier === :off_tier && push!(reasons, :requested_threads_off_tier)

    julia_cap = max(capacity.julia_threads, 1)
    blas_cap = max(capacity.blas_threads, 1)
    outer_grant = min(requested, julia_cap, cone)
    blas_grant = min(requested, blas_cap, blas_w)

    consumer = :none
    budget = nothing
    if requested == 1
        push!(reasons, :single_thread_request_pins_every_layer)
        budget = ThreadBudget(:serial, 1, 1, 1, 1)
    elseif outer_grant > 1 && cone >= la && cone >= blas_w
        consumer = :coarse_cone_tasks
        push!(reasons, :coarse_cone_batch_owns_outer_layer)
        budget = ThreadBudget(:julia_outer, outer_grant, 1, 1, outer_grant)
    elseif blas_grant > 1
        consumer = :blas_layer
        push!(reasons, :blas_layer_owns_provider_threads)
        budget = ThreadBudget(
            :provider_blas, 1, blas_grant,
            provider_available ? blas_grant : 1, 1,
        )
    elseif outer_grant > 1
        consumer = :coarse_cone_tasks
        push!(reasons, :coarse_cone_batch_owns_outer_layer)
        budget = ThreadBudget(:julia_outer, outer_grant, 1, 1, outer_grant)
    elseif la > 1
        # A kernel offering independent panels but no reachable parallel layer
        # is a recorded limitation of this host, not a silent serialization.
        push!(reasons, :la_kernel_parallelism_not_grantable)
        budget = ThreadBudget(:serial, 1, 1, 1, 1)
    else
        push!(reasons, :no_consumer_offers_parallelism)
        budget = ThreadBudget(:serial, 1, 1, 1, 1)
    end
    result = SessionThreadBudget(
        requested, capacity, budget, cone, la, blas_w, consumer, tier, reasons,
    )
    validate_session_thread_budget(result)
    return result
end

"""
    validate_session_thread_budget(budget) -> Nothing

Fail-closed invariant check. Throws `ArgumentError` when a granted count exceeds
the request or the layer capacity, when the tier claim does not match the
capacity, or when a granted count is larger than every single consumer width
(the signature of an accidental product).
"""
function validate_session_thread_budget(b::SessionThreadBudget)
    max_available = max(b.cone_available, b.la_available, b.blas_available)
    for (layer, granted, capacity) in (
        (:julia, b.budget.julia_outer_threads, b.capacity.julia_threads),
        (:blas, b.budget.blas_threads, b.capacity.blas_threads),
        (:provider, b.budget.provider_threads, b.capacity.blas_threads),
    )
        granted >= 1 || throw(ArgumentError(
            "session thread budget $(layer) grant must be at least one",
        ))
        granted <= b.requested_threads || throw(ArgumentError(
            "session thread budget $(layer) grant $(granted) exceeds the " *
            "requested $(b.requested_threads) threads",
        ))
        granted <= max(capacity, 1) || throw(ArgumentError(
            "session thread budget $(layer) grant $(granted) exceeds the host " *
            "capacity $(capacity)",
        ))
        granted <= max(max_available, 1) || throw(ArgumentError(
            "session thread budget $(layer) grant $(granted) exceeds every " *
            "consumer width (max $(max_available)); a grant must never be a " *
            "product of consumer limits",
        ))
    end
    b.tier_status === thread_tier_status(b.requested_threads, b.capacity) ||
        throw(ArgumentError("session thread tier status is inconsistent"))
    return nothing
end

"""
    max_granted_threads(budget) -> Int

The largest single layer grant. This — not a product — is the number of threads
the session may have running at once, because exactly one layer is active.
"""
max_granted_threads(b::SessionThreadBudget) = max(
    b.budget.julia_outer_threads, b.budget.blas_threads, b.budget.provider_threads,
)

"""The granted width for one consumer (`0` when that consumer owns no layer)."""

# ---------------------------------------------------------------------------
# The process-global BLAS scope
# ---------------------------------------------------------------------------

"""
    ThreadContentionError

Raised when a session tries to change the *global* BLAS thread count while it
already holds it with a different value (nested re-configuration would multiply
the effective thread count), or when a lease is released by a non-holder.
"""
struct ThreadContentionError <: Exception
    reason::Symbol
    holder::Union{Nothing,Symbol}
    requester::Symbol
    requested_threads::Int
    held_threads::Int
end

function Base.showerror(io::IO, e::ThreadContentionError)
    print(
        io, "ThreadContentionError(", e.reason, "): session ", e.requester,
        " requested ", e.requested_threads, " BLAS threads",
    )
    e.holder === nothing || print(io, " while session ", e.holder, " holds ", e.held_threads)
end

"""
    BlasLeaseRegistry

Book-keeping for the process-global dense BLAS thread count.

The registry exists so the global knob cannot be mutated invisibly. It records
every attempt, grant, release, and *arrival while another session held the
scope*, plus the maximum number of simultaneous holders ever observed, so a test
can measure that concurrent sessions did not contend — mutual exclusion held and
every session observed its own value — instead of asserting it from a field.

`scope_lock` is the mutual-exclusion primitive and is held for the whole scope;
`lock` only protects the counters. Sessions cannot hold different BLAS counts at
the same time, because the underlying setting is one process-wide integer;
pretending otherwise — or multiplying counts — is the failure ADR-001 §6 names.
"""
mutable struct BlasLeaseRegistry
    lock::ReentrantLock
    scope_lock::ReentrantLock
    holder::Union{Nothing,Symbol}
    held_threads::Int
    depth::Int
    active_holders::Int
    max_concurrent_holders::Int
    attempts::Int
    grants::Int
    releases::Int
    arrivals_while_held::Int
    value_mismatches::Int
    scopes_entered::Int
end

BlasLeaseRegistry() = BlasLeaseRegistry(
    ReentrantLock(), ReentrantLock(), nothing, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

"""
    SETUP_THREAD_REGISTRY

The process-global registry. Exactly one exists per process; sessions
*coordinate* through it rather than each owning a private copy of a global knob.
"""
const SETUP_THREAD_REGISTRY = BlasLeaseRegistry()

"""Snapshot of the registry counters, for receipts and tests."""
function blas_lease_record(registry::BlasLeaseRegistry=SETUP_THREAD_REGISTRY)
    lock(registry.lock)
    try
        return (
            holder=registry.holder,
            held_threads=registry.held_threads,
            depth=registry.depth,
            active_holders=registry.active_holders,
            max_concurrent_holders=registry.max_concurrent_holders,
            attempts=registry.attempts,
            grants=registry.grants,
            releases=registry.releases,
            arrivals_while_held=registry.arrivals_while_held,
            value_mismatches=registry.value_mismatches,
            scopes_entered=registry.scopes_entered,
        )
    finally
        unlock(registry.lock)
    end
end

"""
    acquire_blas_lease!(registry, session, threads) -> Int

Take the global BLAS count for `session`. The caller must already hold
`registry.scope_lock`, which is what makes the global setting exclusive.

Re-entrant for the same session with the same value (a nested scope does **not**
re-apply or multiply anything). Throws [`ThreadContentionError`](@ref) if the
same session tries a different value while holding it, or if a different session
holds the scope.
"""
function acquire_blas_lease!(
    registry::BlasLeaseRegistry, session::Symbol, threads::Integer,
)
    count = max(Int(threads), 1)
    lock(registry.lock)
    try
        registry.attempts += 1
        if registry.holder === session
            registry.held_threads == count || begin
                registry.value_mismatches += 1
                throw(ThreadContentionError(
                    :nested_budget_change, session, session, count,
                    registry.held_threads,
                ))
            end
            registry.depth += 1
            registry.grants += 1
            return registry.held_threads
        end
        registry.holder === nothing || throw(ThreadContentionError(
            :scope_owned_by_another_session, registry.holder, session, count,
            registry.held_threads,
        ))
        registry.holder = session
        registry.held_threads = count
        registry.depth = 1
        registry.active_holders = 1
        registry.max_concurrent_holders = max(
            registry.max_concurrent_holders, 1,
        )
        registry.grants += 1
        return count
    finally
        unlock(registry.lock)
    end
end

"""
    release_blas_lease!(registry, session) -> Union{Nothing,Int}

Release one level of `session`'s hold. Throws
[`ThreadContentionError`](@ref) if `session` is not the holder.
"""
function release_blas_lease!(registry::BlasLeaseRegistry, session::Symbol)
    lock(registry.lock)
    try
        registry.holder === session || throw(ThreadContentionError(
            :release_by_non_holder, registry.holder, session, 0,
            registry.held_threads,
        ))
        registry.depth -= 1
        registry.releases += 1
        if registry.depth <= 0
            registry.holder = nothing
            registry.held_threads = 0
            registry.depth = 0
            registry.active_holders = 0
            return nothing
        end
        return registry.held_threads
    finally
        unlock(registry.lock)
    end
end

"""
    with_session_thread_scope(f, budget; registry=SETUP_THREAD_REGISTRY,
                              session=:anonymous) -> (result, observation)

Run `f()` with the process-global BLAS count pinned to this session's granted
BLAS threads, restoring the previous global value on every exit path (including
exceptions), and return `f()`'s result together with a measured observation.

Mutual exclusion is taken *before* the previous value is read, so the restore
target cannot be a value another session installed in between. The observation
is measured, not copied from the request: `observed_blas_threads` is read back
from the backend inside the scope (`blas_threads()`), and `restored_blas_threads`
is read back after restoration.

Nested scopes for the same session reuse the existing hold instead of applying
the setting a second time, so nesting can never double the thread count.
"""
function with_session_thread_scope(
    f::Function,
    budget::SessionThreadBudget;
    registry::BlasLeaseRegistry=SETUP_THREAD_REGISTRY,
    session::Symbol=:anonymous,
)
    granted = Int(budget.budget.blas_threads)
    lock(registry.lock)
    try
        registry.holder === nothing || (registry.arrivals_while_held += 1)
    finally
        unlock(registry.lock)
    end
    lock(registry.scope_lock)
    try
        previous = Int(blas_threads())
        held = acquire_blas_lease!(registry, session, granted)
        applied = try
            # A serial session must genuinely run serial: pin the backend, and
            # read back what it actually reports rather than trusting the setter.
            set_blas_threads!(held)
            Int(blas_threads())
        catch
            release_blas_lease!(registry, session)
            set_blas_threads!(previous)
            rethrow()
        end
        lock(registry.lock)
        try
            registry.scopes_entered += 1
        finally
            unlock(registry.lock)
        end
        result = try
            f()
        finally
            # Restore the *measured* previous global value on every path, so a
            # concurrent session is never left with this session's setting.
            set_blas_threads!(previous)
            release_blas_lease!(registry, session)
        end
        restored = Int(blas_threads())
        return result, (
            session=session,
            granted_blas_threads=granted,
            observed_blas_threads=applied,
            previous_blas_threads=previous,
            restored_blas_threads=restored,
            restored=restored == previous,
        )
    finally
        unlock(registry.scope_lock)
    end
end

# ---------------------------------------------------------------------------
# Memory ledger and admission
# ---------------------------------------------------------------------------

"""
    SetupMemoryLedger

Named terms of the setup memory estimate. Every term is either inherited from
the previous round's estimator or explicitly modelled here; none is implicit.

Fields:
- `declared_storage`: `:sparse_lower` or `:dense` — what the caller declared.
- `resolved_storage`: what the plan will actually build. For a `:sparse_lower`
  declaration this is `:sparse_lower`, always (see
  [`admit_setup_memory`](@ref) and `execute_setup_plan`), so a sparse-declared
  path cannot silently densify.
- `structural_bytes`: declared pattern (values + indices + column pointers, or
  the dense `d^2` when dense).
- `fill_bytes`: modelled factor fill *beyond* the stored pattern.
- `owned_scalar_bytes`: owned scalars — per-block cone operators (`k^2` each) and
  iterate/workspace vectors.
- `workspace_bytes`: provider scratch, snapshots and canonical copies.
- `fallback_bytes`: the allowance for the route we would have to build if the
  chosen route fails. Budgeted, never allocated unless the fallback is taken.
- `inherited_core_bytes`: the previous round's
  `symmetric_core_state_prepare_bytes` result, when it applies to this route —
  carried so the receipt shows the inherited estimate next to the attributed
  terms.
- `attributed_bytes`: the sum of the named terms above (excluding the fallback),
  recorded separately so a divergence from `inherited_core_bytes` is visible
  instead of hidden by a `max`.
- `total_bytes`: `max(inherited_core_bytes, attributed_bytes) + fallback_bytes`,
  each step saturating at `typemax(Int)`.
- `precision_bits`, `scalar_bytes`: the precision facts the estimate was made at.
"""
struct SetupMemoryLedger
    declared_storage::Symbol
    resolved_storage::Symbol
    structural_bytes::Int
    fill_bytes::Int
    owned_scalar_bytes::Int
    workspace_bytes::Int
    fallback_bytes::Int
    inherited_core_bytes::Int
    attributed_bytes::Int
    total_bytes::Int
    precision_bits::Int
    scalar_bytes::Int
end

"""
    setup_memory_ledger(; T, precision_bits, dimension, block_sizes,
                        structural_nnz, fill_factor, route, declared_storage,
                        rhs_count=3, basis_nnz=0, variable_dimension=0,
                        canonical_nnz=0, fallback_dimension=nothing) -> SetupMemoryLedger

Build the memory ledger from frozen structure.

`declared_storage === :sparse_lower` is honoured exactly: the structural term is
computed from `structural_nnz` and the resolved storage stays `:sparse_lower`.
Densifying is not a fallback that happens quietly — it requires a `:dense`
declaration, and the receipt records which one was requested.
"""
function setup_memory_ledger(;
    T::Type=Float64,
    precision_bits::Integer=64,
    dimension::Integer,
    block_sizes,
    structural_nnz::Integer,
    fill_factor::Real=3.0,
    route::Symbol=:full_core,
    declared_storage::Symbol=:sparse_lower,
    rhs_count::Integer=3,
    basis_nnz::Integer=0,
    variable_dimension::Integer=0,
    canonical_nnz::Integer=0,
    fallback_dimension::Union{Nothing,Integer}=nothing,
)
    declared_storage in (:sparse_lower, :dense) || throw(ArgumentError(
        "declared storage must be :sparse_lower or :dense, got $(declared_storage)",
    ))
    d = max(Int(dimension), 0)
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    stored_nnz = max(Int(structural_nnz), 0)
    resolved = declared_storage

    structural = if resolved === :dense
        saturating_bytes(scalar_bytes, max(d, 1), max(d, 1))
    else
        saturating_sum_bytes(
            saturating_bytes(scalar_bytes + sizeof(Int), stored_nnz),
            saturating_bytes(sizeof(Int), d + 1),
        )
    end

    fill = Float64(max(fill_factor, 0.0))
    fill_extra = fill <= 1.0 ? 0 : Int(min(
        round(Float64(stored_nnz) * (fill - 1.0)),
        Float64(typemax(Int) ÷ 2),
    ))
    fill_bytes = saturating_bytes(scalar_bytes + sizeof(Int), fill_extra)

    # Owned scalars: the per-block cone operators are owned outright (the
    # previous round's estimate counts them too), plus iterate vectors.
    block_operators = 0
    for size in block_sizes
        size <= 0 && continue
        k = Int(size)
        block_operators = saturating_sum_bytes(
            block_operators, saturating_bytes(scalar_bytes, k, k),
        )
    end
    owned_scalars = saturating_sum_bytes(
        block_operators,
        saturating_bytes(max(Int(rhs_count), 1) + 8, scalar_bytes, max(d, 1)),
    )

    workspace = saturating_sum_bytes(
        saturating_bytes(sizeof(Int), d + 1),
        saturating_bytes(2 * (scalar_bytes + sizeof(Int)), stored_nnz),
        saturating_bytes(
            scalar_bytes + sizeof(Int), max(Int(basis_nnz), 0),
        ),
        saturating_bytes(
            scalar_bytes + sizeof(Int), max(Int(canonical_nnz), 0),
        ),
        saturating_bytes(sizeof(Int), max(Int(variable_dimension), 0) + 1),
    )

    fallback_dim = fallback_dimension === nothing ? d : max(Int(fallback_dimension), 0)
    fallback_bytes = saturating_sum_bytes(
        saturating_bytes(scalar_bytes, div((fallback_dim + 1) * fallback_dim, 2)),
        saturating_bytes(24, scalar_bytes, max(fallback_dim, 1)),
    )

    attributed = saturating_sum_bytes(
        structural, fill_bytes, owned_scalars, workspace,
    )

    inherited = if route === :full_core && resolved === :sparse_lower && T === Float64
        # Inheritance, not re-derivation: the previous round's conservative
        # estimator, which already covers pattern + symbolic/numeric fill
        # allowance + snapshots + vectors + basis + canonical.
        symmetric_core_state_prepare_bytes(
            T, d, collect(Int, block_sizes);
            ar_nnz=stored_nnz,
            variable_dimension=max(Int(variable_dimension), 0),
            basis_nnz=max(Int(basis_nnz), 0),
            canonical_nnz=max(Int(canonical_nnz), 0),
        )
    elseif route === :compact_schur
        saturating_sum_bytes(
            saturating_bytes(8, scalar_bytes, max(d, 1), max(d, 1)),
            saturating_bytes(24, scalar_bytes, max(d, 1)),
        )
    else
        saturating_bytes(scalar_bytes, max(d, 1), max(d, 1))
    end

    total = saturating_sum_bytes(max(inherited, attributed), fallback_bytes)
    return SetupMemoryLedger(
        declared_storage, resolved, structural, fill_bytes, owned_scalars,
        workspace, fallback_bytes, inherited, attributed, total,
        max(Int(precision_bits), 1), scalar_bytes,
    )
end

"""
    MemoryAdmission

The admission verdict: the ledger, the budget it was compared against, and why.

`admitted == false` is a *refusal*, not a warning. `execute_setup_plan` throws
[`SetupMemoryRefusal`](@ref) for such a plan before it allocates anything.

Fields: `admitted`, `estimate_bytes`, `inherited_bytes`, `fallback_bytes`,
`budget_bytes`, `headroom_bytes` (all `nothing` when no budget was declared),
`rss_bytes`, `reason`.
"""
struct MemoryAdmission
    admitted::Bool
    estimate_bytes::Int
    inherited_bytes::Int
    fallback_bytes::Int
    budget_bytes::Union{Nothing,Int}
    headroom_bytes::Union{Nothing,Int}
    rss_bytes::Union{Nothing,Int}
    reason::Symbol
end

"""
    SetupMemoryRefusal

Raised before any allocation when a plan's ledger does not fit the declared
budget, or when no budget could be established. Fail-closed: an unknown budget
is a refusal, not a licence.
"""
struct SetupMemoryRefusal <: Exception
    reason::Symbol
    estimate_bytes::Int
    budget_bytes::Union{Nothing,Int}
end

function Base.showerror(io::IO, e::SetupMemoryRefusal)
    print(
        io, "SetupMemoryRefusal(", e.reason, "): setup estimate ",
        e.estimate_bytes, " bytes against budget ",
        e.budget_bytes === nothing ? "unknown" : string(e.budget_bytes),
    )
end

"""
    admit_setup_memory(ledger; budget_bytes, rss_bytes=nothing) -> MemoryAdmission

Decide whether the ledger fits. Pure arithmetic; allocates nothing.

Rules, in order:
1. a saturated (`typemax(Int)`) estimate cannot certify an upper bound → refuse
   (`:estimate_saturated`);
2. no declared budget → refuse (`:no_memory_budget_declared`) — fail closed;
3. `estimate > budget` → refuse (`:budget_exceeded`);
4. the estimate is compared against the *remaining* budget after the current
   resident set when `rss_bytes` is supplied
   (`:rss_plus_estimate_exceeds_budget`);
5. otherwise admit, reporting the headroom.
"""
function admit_setup_memory(
    ledger::SetupMemoryLedger;
    budget_bytes::Union{Nothing,Integer}=nothing,
    rss_bytes::Union{Nothing,Integer}=nothing,
)
    estimate = ledger.total_bytes
    if estimate >= typemax(Int)
        return MemoryAdmission(
            false, estimate, ledger.inherited_core_bytes, ledger.fallback_bytes,
            budget_bytes === nothing ? nothing : Int(budget_bytes), nothing,
            rss_bytes === nothing ? nothing : Int(rss_bytes),
            :estimate_saturated,
        )
    end
    if budget_bytes === nothing
        return MemoryAdmission(
            false, estimate, ledger.inherited_core_bytes, ledger.fallback_bytes,
            nothing, nothing,
            rss_bytes === nothing ? nothing : Int(rss_bytes),
            :no_memory_budget_declared,
        )
    end
    budget = max(Int(budget_bytes), 0)
    if estimate > budget
        return MemoryAdmission(
            false, estimate, ledger.inherited_core_bytes, ledger.fallback_bytes,
            budget, 0, rss_bytes === nothing ? nothing : Int(rss_bytes),
            :budget_exceeded,
        )
    end
    rss = rss_bytes === nothing ? nothing : max(Int(rss_bytes), 0)
    if rss !== nothing && rss + estimate > budget
        return MemoryAdmission(
            false, estimate, ledger.inherited_core_bytes, ledger.fallback_bytes,
            budget, 0, rss, :rss_plus_estimate_exceeds_budget,
        )
    end
    headroom = saturating_sum_bytes(budget - estimate, 0)
    return MemoryAdmission(
        true, estimate, ledger.inherited_core_bytes, ledger.fallback_bytes,
        budget, headroom, rss, :admitted,
    )
end

"""
    reserve_setup_memory(admission) -> Int

The execution boundary's memory gate: throws
[`SetupMemoryRefusal`](@ref) unless admission passed, and returns the admitted
byte count otherwise. It performs no allocation, so it necessarily runs *before*
the first array of the planned workload exists.
"""
function reserve_setup_memory(admission::MemoryAdmission)
    admission.admitted || throw(SetupMemoryRefusal(
        admission.reason, admission.estimate_bytes, admission.budget_bytes,
    ))
    return admission.estimate_bytes
end
