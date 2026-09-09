# R2-A internal session-owned CHOLMOD lease foundation. Not wired into Prepared
# execution yet. A cache retains old numeric buffers; revocation prevents their
# use, it does not erase them. No global mutable factor registry is introduced.

_frozen_symbolic_value(x::Union{Int,UInt64,Float64,Bool,Symbol,Nothing}) = true
_frozen_symbolic_value(x::Tuple) = all(_frozen_symbolic_value, x)
_frozen_symbolic_value(x::NamedTuple) = all(_frozen_symbolic_value, values(x))
_frozen_symbolic_value(x) = false

struct SessionSymbolicKey
    context::NamedTuple
    n::Int
    colptr::Tuple{Vararg{Int}}
    rowval::Tuple{Vararg{Int}}
    dsigns::Tuple{Vararg{Int}}
    symbolic_epoch::Int
    structure_generation::UInt64
    function SessionSymbolicKey(context::NamedTuple, req::SparseSymbolicRequirements,
        structure_generation::UInt64)
        _frozen_symbolic_value(context) || throw(ArgumentError("symbolic context must contain immutable scalar/tuple facts"))
        all(k -> haskey(context,k), (:prepared_fingerprint,:arithmetic,:precision_bits,
            :provider,:route,:core_owner,:threads,:reduction,:cone_layout,:ordering)) ||
            throw(ArgumentError("incomplete symbolic compatibility context"))
        context.arithmetic === :float64 && context.precision_bits === 53 &&
            context.provider === :cholmod && context.route === :bordered &&
            context.core_owner === :generic || throw(ArgumentError("unsupported symbolic reuse context"))
        new(context,req.n,Tuple(req.pattern.colptr),Tuple(req.pattern.rowval),
            Tuple(req.dsigns),req.symbolic_epoch,structure_generation)
    end
end
_same_symbolic_key(a::SessionSymbolicKey,b::SessionSymbolicKey) =
    all(k -> isequal(getfield(a,k),getfield(b,k)),fieldnames(SessionSymbolicKey))
function _cache_matches_key(c::SparseSymbolicNumericCache{Float64}, k::SessionSymbolicKey)
    c.n == k.n && c.symbolic_epoch == k.symbolic_epoch &&
        Tuple(c.colptr) == k.colptr && Tuple(c.rowval) == k.rowval &&
        Tuple(c.dsigns) == k.dsigns && size(c.factor_view) == (k.n,k.n) &&
        Tuple(c.factor_view.colptr) == k.colptr && Tuple(c.factor_view.rowval) == k.rowval
end
struct SessionSymbolicEntry
    key::SessionSymbolicKey
    cache::SparseSymbolicNumericCache{Float64}
end
mutable struct SessionSymbolicSlot
    lock::ReentrantLock
    entry::Union{Nothing,SessionSymbolicEntry}
    attempt::UInt64
    active::Bool
end
SessionSymbolicSlot() = SessionSymbolicSlot(ReentrantLock(),nothing,UInt64(0),false)
mutable struct SessionSymbolicLease
    slot::Union{Nothing,SessionSymbolicSlot}
    entry::Union{Nothing,SessionSymbolicEntry}
    attempt::UInt64
    task::Union{Nothing,Task}
    active::Bool
    attached::Bool
end

function _check_symbolic_lease(lease::SessionSymbolicLease)
    slot = lease.slot
    lease.active && slot !== nothing && slot.active &&
        lease.attempt == slot.attempt && lease.task === current_task() ||
        throw(ArgumentError("closed, stale or non-owner symbolic lease"))
    return slot
end

"""Exclusively move the idle entry into one solve attempt and revoke old numerics."""
function checkout_symbolic!(slot::SessionSymbolicSlot)
    trylock(slot.lock) || throw(ArgumentError("symbolic slot already leased"))
    try
        slot.active && throw(ArgumentError("symbolic slot already leased"))
        attempt = Base.Checked.checked_add(slot.attempt,UInt64(1))
        entry = slot.entry
        if entry !== nothing
            if entry.cache.status in (Prepared,Fresh) && entry.cache.factor !== nothing
                revoke_numeric!(entry.cache)
            else
                invalidate!(entry.cache)
                entry = nothing
            end
        end
        # Allocate before publishing ownership. If construction fails, the
        # idle slot still owns its (revoked) entry and remains retryable.
        lease = SessionSymbolicLease(slot,entry,attempt,current_task(),true,false)
        slot.entry = nothing
        slot.attempt = attempt
        slot.active = true
        return lease
    catch
        unlock(slot.lock)
        rethrow()
    end
end

"""After ordinary provider selection, attach one compatible cache to a fresh workspace.
The trusted factory must create a newly owned, already prepared Float64 sparse
cache, never one retained by another session or caller; it does not
select a provider or create a workspace. Call once per lease, not per epoch.
"""
function lease_symbolic_cache!(lease::SessionSymbolicLease, key::SessionSymbolicKey, factory)
    _check_symbolic_lease(lease)
    lease.attached && throw(ArgumentError("symbolic lease already attached to a workspace"))
    entry = lease.entry
    if entry !== nothing && !(_same_symbolic_key(entry.key,key) &&
            _cache_matches_key(entry.cache,key) && entry.cache.status === Prepared)
        invalidate!(entry.cache)
        lease.entry = entry = nothing
    end
    if entry === nothing
        cache = factory()
        cache isa SparseSymbolicNumericCache{Float64} ||
            throw(ArgumentError("symbolic factory returned an unsupported cache"))
        # Register ownership before any subsequent validation/construction fails.
        lease.entry = entry = SessionSymbolicEntry(key,cache)
        cache.status === Prepared && _cache_matches_key(cache,key) ||
            throw(ArgumentError("prepared cache does not match symbolic key"))
    end
    revoke_numeric!(entry.cache) # unconditional, including same-epoch collisions
    lease.attached = true
    return entry.cache
end

"""Close the exclusive attempt. Retain only an eligible certified final solve.
On failed/nonoptimal/fallback execution pass certified_optimal=false. The caller
must invoke this in a finally block, nested inside its existing session unlock.
"""
function finish_symbolic!(lease::SessionSymbolicLease;
    certified_optimal::Bool=false, eligible::Bool=false,
    structure_generation::UInt64=UInt64(0))
    slot = _check_symbolic_lease(lease)
    entry = lease.entry
    try
        keep = certified_optimal && eligible && lease.attached && entry !== nothing &&
            entry.key.structure_generation == structure_generation &&
            entry.cache.status === Fresh && entry.cache.factor !== nothing &&
            _cache_matches_key(entry.cache,entry.key)
        if keep
            revoke_numeric!(entry.cache)
            slot.entry = entry
        elseif entry !== nothing
            invalidate!(entry.cache)
        end
        return keep
    catch
        # Validation may allocate (exact CSC comparisons). Failed check-in
        # must detach even if a caller/workspace still references the cache.
        # invalidate! here only assigns concrete fields; preserve the error.
        slot.entry = nothing
        entry === nothing || invalidate!(entry.cache)
        rethrow()
    finally
        lease.entry = nothing
        lease.slot = nothing
        lease.task = nothing
        lease.active = false
        slot.active = false
        unlock(slot.lock)
    end
end

"""Discard an idle session entry (e.g. an external structural mismatch)."""
function discard_symbolic!(slot::SessionSymbolicSlot)
    trylock(slot.lock) || throw(ArgumentError("cannot discard an actively leased slot"))
    try
        slot.active && throw(ArgumentError("cannot discard an actively leased slot"))
        slot.entry === nothing || invalidate!(slot.entry.cache)
        slot.entry = nothing
    finally
        unlock(slot.lock)
    end
    return nothing
end
