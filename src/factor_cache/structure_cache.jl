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
   SymmetricCoreStructureCache() = new(true, Dict{Tuple{Type,UInt64},Any}(), 0, 0)
end

const _SYMMETRIC_CORE_STRUCTURE_CACHE = SymmetricCoreStructureCache()
const _SYMMETRIC_CORE_STRUCTURE_LOCK = ReentrantLock()

"""Enable/disable the cross-solve symmetric-core structure cache."""
function set_structure_cache_enabled!(enabled::Bool)
    _SYMMETRIC_CORE_STRUCTURE_CACHE.enabled = enabled
    enabled || empty!(_SYMMETRIC_CORE_STRUCTURE_CACHE.patterns)
    return _SYMMETRIC_CORE_STRUCTURE_CACHE
end

"""Drop every cached core structure (used on test teardown and cache
invalidation paths)."""
function clear_structure_cache!()
    lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        empty!(_SYMMETRIC_CORE_STRUCTURE_CACHE.patterns)
        _SYMMETRIC_CORE_STRUCTURE_CACHE.hits = 0
        _SYMMETRIC_CORE_STRUCTURE_CACHE.misses = 0
    end
    return _SYMMETRIC_CORE_STRUCTURE_CACHE
end

@inline function _structure_cache_record_hit!()
    _SYMMETRIC_CORE_STRUCTURE_CACHE.hits += 1
    return nothing
end

@inline function _structure_cache_record_miss!()
    _SYMMETRIC_CORE_STRUCTURE_CACHE.misses += 1
    return nothing
end

"""Current structure-cache stats: `(hits, misses, entries, enabled)`."""
function structure_cache_stats()
    cache = _SYMMETRIC_CORE_STRUCTURE_CACHE
    return lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        (hits=cache.hits, misses=cache.misses,
            entries=length(cache.patterns), enabled=cache.enabled)
    end
end