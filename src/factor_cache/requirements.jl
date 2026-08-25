#=====================================================================#
#    FactorCache capacity requirements (Subagent D).
#
#    `FactorRequirements` is the single, exact source of capacity.  All storage
#    allocation happens in `prepare!(cache, requirements)`; after `prepare!`
#    returns, NO resize / growth (no `push!`, `resize!`, realloc) may occur
#    anywhere.  Providers read the sizes they need from the one requirements
#    object and never grow later.
#=====================================================================#

"""
    FactorRequirements

Exact capacity declaration for a factor cache.  Passed to `prepare!` and is the
sole allocation source.  `n` is the matrix dimension; `symbolic_epoch` fixes the
identity of the symbolic structure so it can be reused across iterations.

Providers with additional capacity needs (sparsity, arrow size, ...) may either
extend this struct with more fields or subtype `AbstractFactorRequirements`.
"""
struct FactorRequirements
    n::Int
    symbolic_epoch::Int
end

FactorRequirements(n::Integer) = FactorRequirements(Int(n), 0)

"""
    AbstractFactorRequirements

Marker supertype for provider-specific requirements objects, should a provider
need capacity fields beyond `n`.  The protocol accepts any object here; the
reference and route caches use `FactorRequirements`.
"""
abstract type AbstractFactorRequirements end
