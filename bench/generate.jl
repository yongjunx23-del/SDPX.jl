#=====================================================================
    Bootstrap-shaped benchmark instance generator (§0.2): dense A_i,
    heterogeneous blocks, one normalization equality, deliberately bad
    scaling (entries spanning `scalespread` orders of magnitude) — the
    profile that stresses the Schur build and KKT solve the way real
    bootstrap data does, unlike the uniform-[0,1] toy random data in
    the correctness tests.
=====================================================================#

using StableRNGs

function bootstrap_like(T; L=20, m=250, ks=10:5:60, n=1, scalespread=8, seed=1)
    rng = StableRNG(seed)
    k = rand(rng, ks, L)
    A = [Array{T,3}(undef, m, k[l], k[l]) for l in 1:L]
    for l in 1:L, i in 1:m
        Mi = randn(rng, k[l], k[l]) .* exp10.(scalespread .* rand(rng, k[l], k[l]))
        A[l][i, :, :] = T.(Mi + Mi')
    end
    C = [begin
        Mc = randn(rng, k[l], k[l])
        T.(Mc + Mc')
    end for l in 1:L]
    B = ones(T, m, n)
    b = ones(T, n)
    c = T.(randn(rng, m))
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
