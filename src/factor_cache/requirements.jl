#=====================================================================#
#    FactorCache capacity requirements.
#
#    `FactorRequirements` is the single, exact source of capacity.  All storage
#    allocation happens in `prepare!(cache, requirements)`; after `prepare!`
#    returns, NO resize / growth (no `push!`, `resize!`, realloc) may occur
#    anywhere.  Providers read the sizes they need from the one requirements
#    object and never grow later.
#=====================================================================#

"""
    AbstractFactorRequirements

Supertype for provider-specific requirements objects.  `FactorRequirements`
subtypes it so one protocol signature covers the standard and provider
shapes; providers may add capacity fields beyond `n`.
"""
abstract type AbstractFactorRequirements end

"""
    struct FactorRequirements <: AbstractFactorRequirements

Standard frozen-shape capacity declaration for a factor cache.  Passed to
`prepare!`; `n` is the matrix dimension and `symbolic_epoch` fixes the
identity of the symbolic structure so it can be reused across iterations.

Providers with additional capacity needs (sparsity, arrow size, ...) may
either extend this struct with more fields or subtype
`AbstractFactorRequirements`.
"""
struct FactorRequirements <: AbstractFactorRequirements
    n::Int
    symbolic_epoch::Int
end

FactorRequirements(n::Integer) = FactorRequirements(Int(n), 0)
