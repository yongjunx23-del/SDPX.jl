# Why this solver carries extended precision at all.
#
# The bootstrap programs SDPX targets are conditioned badly enough that
# `Float64` runs out of accuracy before the solve reaches the tolerance that
# was asked for. This example shows exactly where that happens, on the same
# small problem as `01_basic_sdp.jl` so the exact optimum `2√6` is available
# to measure against.
#
# The pattern to notice: the error tracks the *requested tolerance*, not the
# arithmetic, right up until the arithmetic gives out. Then `Float64` stalls
# and `Float64x4` keeps going.
#
# Run with:  julia --project=examples examples/02_extended_precision.jl

using LinearAlgebra
using MultiFloats                    # loading this enables the Float64xN backend
using Printf
using SDPX

coefficients = zeros(2, 2, 2)
coefficients[1, 1, 1] = 1.0
coefficients[2, 2, 2] = 1.0
constant = [0.0 1.0; 1.0 0.0]
objective = [2.0, 3.0]

# Reference value at far more precision than any solve below will reach.
exact = 2 * sqrt(BigFloat(6))

function attempt(::Type{T}, tolerance) where {T}
    result = solve(
        T.(objective),
        [T.(coefficients)],
        [T.(constant)],
        Matrix{T}(undef, 2, 0),
        T[];
        verbosity=0,
        tolerance=tolerance,
    )
    # Widen before differencing so the comparison itself is not the limit.
    error = abs(BigFloat(Float64x4(result.pObj)) - exact)
    return (status=result.status, iterations=result.iterations, error=error)
end

@printf("%12s %12s %10s %7s %14s\n",
    "arithmetic", "tolerance", "status", "iters", "|error|")
for tolerance in (1e-8, 1e-14, 1e-20, 1e-30)
    for (name, T) in (("Float64", Float64), ("Float64x4", Float64x4))
        # Float64 cannot represent a tolerance below its own epsilon; asking
        # for one is not a fair comparison, it is a category error.
        T === Float64 && tolerance < 1e-16 && continue
        outcome = attempt(T, tolerance)
        @printf("%12s %12.0e %10s %7d %14.3e\n",
            name, tolerance, outcome.status, outcome.iterations, outcome.error)
    end
end

println("""

Reading the table: at 1e-8 both arithmetics agree exactly -- the tolerance is
what binds, so the wider type buys nothing and costs time. At 1e-14 Float64
stalls short of the target while Float64x4 reaches it. Past that only
Float64x4 remains, and its error keeps tracking the tolerance down to 1e-30.

The practical rule: use Float64 until it stalls, then widen. Reaching for
Float64x4 first is a real slowdown for no accuracy.""")

# The claim above is the point of the example, so check it rather than trust it.
narrow = attempt(Float64, 1e-14)
wide = attempt(Float64x4, 1e-14)
narrow.status == SDPX.Optimal &&
    error("Float64 was expected to fall short at 1e-14, but returned Optimal")
wide.status == SDPX.Optimal ||
    error("Float64x4 was expected to reach 1e-14, got $(wide.status)")
wide.error < narrow.error ||
    error("Float64x4 error $(wide.error) should be below Float64's $(narrow.error)")
