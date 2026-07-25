#!/usr/bin/env julia
# Plan §19.5: all rank-one, triangular and block-update loops must follow
# Julia's column-major order. The audit found the hot kernels already do; this
# quantifies what a regression would cost, so the review rule has a number
# behind it rather than being folk wisdom.
using LinearAlgebra, Printf, MultiFloats
using MultiFloats: Float64x4

# NOTE: a plain summation benchmark was tried and removed. The compiler elides
# the reduction entirely (it times at 0.000000 for every size), so it measures
# nothing. The rank-one update below writes to memory and cannot be elided, and
# is the shape §19.5 names in any case.

"""Rank-one update in each order — the shape §19.5 calls out by name."""
function rank_one_column_major!(A, x, y)
    @inbounds for j in axes(A, 2)
        yj = y[j]
        for i in axes(A, 1)
            A[i, j] += x[i] * yj
        end
    end
    return A
end

function rank_one_row_major!(A, x, y)
    @inbounds for i in axes(A, 1)
        xi = x[i]
        for j in axes(A, 2)
            A[i, j] += xi * y[j]
        end
    end
    return A
end

# More repetitions for the cheap traversal case, whose single-shot time is
# below timer resolution.
best(f, args...; reps=20) = (f(args...); minimum(@elapsed(f(args...)) for _ in 1:reps))

@printf("%-12s %6s  %-22s %10s %10s %8s\n",
        "type", "n", "operation", "col-major", "row-major", "penalty")
for T in (Float64, Float64x4), n in (512, 1024)
    A = rand(T, n, n)
    x = rand(T, n); y = rand(T, n)
    tc = best(rank_one_column_major!, A, x, y)
    tr = best(rank_one_row_major!, A, x, y)
    @printf("%-12s %6d  %-22s %10.6f %10.6f %7.2fx\n",
            T, n, "rank-one update", tc, tr, tr / tc)
end
