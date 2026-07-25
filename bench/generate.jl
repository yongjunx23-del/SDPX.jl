#=====================================================================
    Bootstrap-shaped benchmark instance generator (§0.2): dense A_i,
    heterogeneous blocks, one normalization equality, deliberately bad
    scaling (entries spanning `scalespread` orders of magnitude) — the
    profile that stresses the Schur build and KKT solve the way real
    bootstrap data does, unlike the uniform-[0,1] toy random data in
    the correctness tests.
=====================================================================#

using LinearAlgebra
using StableRNGs

"""
    bootstrap_like(T; L, m, ks, n, scalespread, seed) -> (; c, A, C, B, b)

Bootstrap-shaped benchmark instance: dense `A_i`, heterogeneous block sizes, one
normalization equality, and entries deliberately spanning `scalespread` orders
of magnitude — the profile that stresses Schur assembly and the KKT solve the
way real bootstrap data does.

The instance is **well posed by construction**: a strictly feasible primal point
and a strictly feasible dual point are both generated first, and `C` and `c` are
then defined from them. Slater holds on both sides, so a finite optimum exists
and is attained, and a benchmark run can be validated rather than merely timed.

This matters. The earlier generator drew `A` and `C` independently at random,
which gives no guarantee of feasibility or boundedness — the `:small` tier
diverged to `dObj = -6.6e26` with `p_res = 1e5`, stalling after 11 iterations,
and the harness reported it as a successful benchmark because it never checked
the answer. Timing a divergent solve measures nothing.

Numbers from the previous generator are not comparable with these.
"""
function bootstrap_like(T; L=20, m=250, ks=10:5:60, n=1, scalespread=8, seed=1)
    rng = StableRNG(seed)
    k = rand(rng, ks, L)

    # Badly scaled symmetric coefficient matrices: the actual stressor.
    A = [Array{T,3}(undef, m, k[l], k[l]) for l in 1:L]
    for l in 1:L, i in 1:m
        Mi = randn(rng, k[l], k[l]) .* exp10.(scalespread .* rand(rng, k[l], k[l]))
        A[l][i, :, :] = T.(Mi + Mi')
    end

    B = ones(T, m, n)
    b = ones(T, n)

    # Strictly feasible primal point. B'x0 = b must hold exactly, and with
    # `B = ones(m, n)` that is just "the entries sum to one" per column.
    x0 = rand(rng, m) .+ T(0.5)
    x0 ./= sum(x0)                      # => B'x0 = b for the all-ones B
    x0 = T.(x0)

    # C is then defined so that X0 = Σ x0_i A_i − C is a chosen positive
    # definite matrix, making x0 strictly primal feasible by construction.
    C = Vector{Matrix{T}}(undef, L)
    for l in 1:L
        slack = randn(rng, k[l], k[l])
        slack = T.(slack * slack') + T(k[l]) * Matrix{T}(LinearAlgebra.I, k[l], k[l])
        combination = zeros(T, k[l], k[l])
        for i in 1:m
            @views combination .+= x0[i] .* A[l][i, :, :]
        end
        C[l] = combination .- slack
    end

    # Strictly feasible dual point (Y0 ≻ 0, y0 free); c is defined from the dual
    # equality Σ_l tr(A_i^(l) Y^(l)) + (B y)_i = c_i, so the dual is strictly
    # feasible too. Slater on both sides gives a finite, attained optimum.
    Y0 = [begin
        M = randn(rng, k[l], k[l])
        T.(M * M') + Matrix{T}(LinearAlgebra.I, k[l], k[l])
    end for l in 1:L]
    y0 = T.(randn(rng, n))
    c = zeros(T, m)
    for i in 1:m
        total = zero(T)
        for l in 1:L
            @views total += LinearAlgebra.dot(A[l][i, :, :], Y0[l])
        end
        for j in 1:n
            total += B[i, j] * y0[j]
        end
        c[i] = total
    end

    return (; c, A, C, B, b)
end

const TIERS = (
    small=(L=3, m=50, ks=5:5:30, n=1),
    medium=(L=10, m=150, ks=10:5:40, n=1),
    reference=(L=20, m=250, ks=10:5:60, n=1),
    large_m=(L=10, m=600, ks=10:5:30, n=1),
)

tier_instance(tier::Symbol, T; seed=1, scalespread=8) =
    bootstrap_like(T; TIERS[tier]..., scalespread=scalespread, seed=seed)
