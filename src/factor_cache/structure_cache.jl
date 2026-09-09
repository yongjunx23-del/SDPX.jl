#=====================================================================#
#    Structure-keyed cross-solve cache for the frozen symmetric-core
#    CSC structure (review slice 2).
#
#    Same-structure repeated solves (parameter sweeps, repeated solves of
#    the same model) rebuild the identical symmetric-core CSC structure
#    every time:  the frozen lower-triangle column pointers, row indices,
#    the Ar/Theta/x-diagonal slot maps, and the block layout.  This cache
#    stores ONLY that structural content and never the numeric `nzval`
#    buffer:
#
#      * cache key: (arithmetic type, full CSC structure signature)
#        — the signature already mixes nr, m, the Ar colptr/rowval
#          pattern, the block ranges, and the block shape codes, so a
#          change in dimension, cone partition, sparsity pattern, or
#          formulation (block shapes) produces a different key;
#      * on a hit, a NEW SymmetricCorePattern is assembled that shares
#        the frozen immutable structural arrays but owns a FRESH zero
#        numeric buffer — no numeric value can survive a reuse;
#      * the numeric refresh contract is unchanged: the factor path
#        (`_core_refill_from_system!`) rewrites every theta slot and the
#        workspace snapshot rebuilds `original_nzval` from the live
#        values on every synchronize, exactly as before;
#      * misses fall through to the ordinary construction and populate
#        the cache;
#      * `clear_structure_cache!` drops every entry (used by tests and
#        by callers that must guarantee no cross-solve state).
#
#    The cache is process-global and guarded by a ReentrantLock; lookups
#    hash the structural signature only (no values), so hot-path reuse
#    between two solves of one structure costs one hash + one Dict hit.
#=====================================================================#

mutable struct SymmetricCoreStructureCache
    enabled::Bool
    patterns::Dict{Tuple{Type,UInt64},Any}
    hits::Int
    misses::Int
    # Invalidation generation: captured under lock by an eligible lookup and
    # re-checked under lock at publication.  Advanced under the same lock by
    # every disable and every clear, so a build whose lookup predates an
    # invalidation can never publish afterwards — even across
    # disable/clear/re-enable, where the flag alone would read enabled again.
    # UInt64: reaching max via a normal advance is benign (cache stays
    # usable); attempting ANOTHER advance while already at max fails closed
    # (forced disabled, entries dropped, counter pinned, never wraps).
    generation::UInt64
   SymmetricCoreStructureCache() = new(true, Dict{Tuple{Type,UInt64},Any}(), 0, 0, UInt64(0))
end

const _SYMMETRIC_CORE_STRUCTURE_CACHE = SymmetricCoreStructureCache()
const _SYMMETRIC_CORE_STRUCTURE_LOCK = ReentrantLock()

"""Enable/disable the cross-solve symmetric-core structure cache.

Takes the global cache lock so the flag write and (when disabling) the
entry drop happen atomically: a concurrent lookup or publication either
fully precedes or fully follows the toggle.  Disabling advances the
invalidation generation, so an eligible in-flight build whose lookup
predates the disable cannot publish afterwards, even after a later
re-enable (publication re-checks the captured token under the same lock).
At typemax(UInt64), enabling is a no-op preserving the current flag.
Reaching max via a normal advance leaves an already-enabled cache usable;
a further invalidation attempt at max disables and empties it. Subsequent
enable calls cannot revive that disabled state. See
`_structure_cache_advance_generation!`."""
function set_structure_cache_enabled!(enabled::Bool)
    lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        cache = _SYMMETRIC_CORE_STRUCTURE_CACHE
        if enabled
            cache.generation == typemax(UInt64) && return cache
            cache.enabled = true
        else
            cache.enabled = false
            empty!(cache.patterns)
            _structure_cache_advance_generation!(cache)
        end
    end
    return _SYMMETRIC_CORE_STRUCTURE_CACHE
end

"""Drop every cached core structure (used on test teardown and cache
invalidation paths)."""
function clear_structure_cache!()
    lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        cache = _SYMMETRIC_CORE_STRUCTURE_CACHE
        empty!(cache.patterns)
        cache.hits = 0
        cache.misses = 0
        # A clear is an invalidation for in-flight builders even when the
        # flag is unchanged: advance the generation so a lookup that
        # predates this clear cannot publish afterwards.
        _structure_cache_advance_generation!(cache)
    end
    return _SYMMETRIC_CORE_STRUCTURE_CACHE
end

function _structure_cache_record_hit!()
    lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        _SYMMETRIC_CORE_STRUCTURE_CACHE.hits += 1
    end
    return nothing
end

function _structure_cache_record_miss!()
    lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        _SYMMETRIC_CORE_STRUCTURE_CACHE.misses += 1
    end
    return nothing
end

"""Atomic cache lookup with hit/miss accounting (call with the lock NOT held).

Returns `(cached, enabled, generation)`.  When the cache is disabled,
returns `(nothing, false, <live generation>)` without touching the hit/miss
counters, so a disabled build is invisible to stats and can never observe
a stale entry.  When enabled, a present entry counts exactly one hit and a
missing key counts exactly one miss, atomically with the lookup, so
concurrent builders never lose counter updates.  The returned `generation`
is the caller's publication token: `_structure_cache_try_publish!` stores
the entry only if no disable/clear advanced the generation in between."""
function _structure_cache_lookup_and_account!(key)
    return lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        cache = _SYMMETRIC_CORE_STRUCTURE_CACHE
        cache.enabled || return (nothing, false, cache.generation)
        cached = get(cache.patterns, key, nothing)
        if cached isa NamedTuple && haskey(cached, :colptr)
            cache.hits += 1
            return (cached, true, cache.generation)
        end
        cache.misses += 1
        return (nothing, true, cache.generation)
    end
end

"""Advance the invalidation generation (call with the lock HELD).

Takes the cache object so the boundary rule is unit-testable on a local
instance without touching global state.  Two distinct cases at the numeric
boundary:
- advancing ONTO typemax (from max-1) succeeds and the cache stays fully
  usable;
- attempting ANOTHER advance while already at typemax fails closed: the
  cache is forced disabled with every entry dropped and the counter pinned
  at max (it wraps NEVER), so subsequent lookups bypass and every
  in-flight publication mismatches.
At max, `set_structure_cache_enabled!(true)` preserves the current flag.
After an additional invalidation disables the cache, enabling cannot revive it.
The boundary is tested on a local instance rather than by repeated invalidation
of the global cache."""
function _structure_cache_advance_generation!(cache)
    if cache.generation == typemax(UInt64)
        cache.enabled = false
        empty!(cache.patterns)
        return cache.generation
    end
    cache.generation += UInt64(1)
    return cache.generation
end

"""Lookup/build/publish protocol (call with the lock NOT held).

Private production-used helper: `SymmetricCorePattern` construction calls
this (it is the only production caller), and regression tests call it
directly with instrumented closures.  Not exported; no test callback or
global hook is involved — the closures below are ordinary arguments.

- `build_structure()` runs the expensive CSC construction.  It is invoked
  synchronously by this helper OUTSIDE the global lock, and ONLY when the
  lookup missed or ran while disabled (never on a hit).  Tests block
  inside their closure on Channels to force an exact post-lookup /
  pre-publication interleaving with zero timing dependence.
- `build_payload(built)` assembles the cache entry from the built pieces.
  It is invoked ONLY when the lookup was eligible (`enabled` at lookup),
  so a disabled-lookup build constructs no payload even if the cache is
  re-enabled before publication.

Returns `(hit, enabled, built)`: a hit carries the cached entry (the
caller assembles a pattern with a fresh numeric buffer); `built` carries
the constructed pieces on a miss or disabled build (`nothing` on a hit).
Publication of a miss is gated first-wins on the captured generation
token (see `_structure_cache_try_publish!`)."""
function _structure_cache_lookup_build_publish!(
    key, build_structure::Function, build_payload::Function,
)
    cached, cache_enabled, cache_generation =
        _structure_cache_lookup_and_account!(key)
    if cache_enabled && cached isa NamedTuple && haskey(cached, :colptr)
        return (hit=cached, enabled=true, built=nothing)
    end
    built = build_structure()
    if cache_enabled
        _structure_cache_try_publish!(
            key, build_payload(built), cache_generation,
        )
    end
    return (hit=nothing, enabled=cache_enabled, built=built)
end

"""Gated first-wins publication (call with the lock NOT held).

The expensive CSC structure build and the publication payload copies always
happen BEFORE this call, outside the lock; the critical section is one
token comparison, one Dict probe, plus at most one insert (a Dict insert
may itself resize, but no CSC construction, factorization, or payload copy
happens under the lock).  The entry is stored only if the cache is still
enabled AND the caller's `generation` still matches (no disable/clear
landed since its lookup) AND no entry already won the race for `key`
(first-wins: concurrent publishers never clobber each other).  Returns
`true` iff this call published.  A disable+clear (or a bare clear) between
lookup and publication therefore suppresses the insert instead of being
resurrected by an in-flight build — including across a re-enable, where
the flag alone would read enabled again.  A losing builder keeps its own
owned arrays, which are structurally identical by the key contract, so
correctness never depends on winning."""
function _structure_cache_try_publish!(key, entry, generation::UInt64)
    return lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        cache = _SYMMETRIC_CORE_STRUCTURE_CACHE
        (cache.enabled && cache.generation == generation &&
            !haskey(cache.patterns, key)) || return false
        cache.patterns[key] = entry
        return true
    end
end

"""Current structure-cache stats: `(hits, misses, entries, enabled)`."""
function structure_cache_stats()
    cache = _SYMMETRIC_CORE_STRUCTURE_CACHE
    return lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        (hits=cache.hits, misses=cache.misses,
            entries=length(cache.patterns), enabled=cache.enabled)
    end
end