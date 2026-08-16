"""
    saturating_bytes(elements..., element_bytes) -> Int

Product of the arguments, clamped to `typemax(Int)` instead of wrapping.

A memory estimate that overflows is worse than no estimate at all: it comes
back negative, compares as less than any budget, and the guard it feeds then
approves precisely the allocation it exists to refuse. Measured before this
was added, `nullspace_memory_bytes(2_000_000_000, 1, Float64)` returned
-4893488163419103232.
"""
function saturating_bytes(factors::Integer...)
    total = big(1)
    for factor in factors
        factor <= 0 && return 0
        total *= big(factor)
        total > typemax(Int) && return typemax(Int)
    end
    return Int(total)
end

"""
    saturating_sum_bytes(terms...) -> Int

Sum of already-saturated byte counts, clamped to `typemax(Int)`. The companion
to [`saturating_bytes`](@ref): products saturate individually, but a sum of
several near-limit products can still wrap, and a wrapped total defeats the
same budget comparisons.
"""
function saturating_sum_bytes(terms::Integer...)
    total = big(0)
    for term in terms
        total += big(max(term, 0))
        total > typemax(Int) && return typemax(Int)
    end
    return Int(total)
end
