# Focused evidence for the optional QDLDL sparse signed-LDL provider seam.
#
# QDLDL (upstream oxfordcontrol/QDLDL.jl 0.4.1) is wrapped by MFLA/BFLA as
# an *optional* extension; SDPX exposes a provider-neutral SparseQDLDLCache
# that delegates to it.  Preconditions pinned by upstream and the adapters:
#   * symmetric quasi-definite operator; only upper triangle in CSC;
#   * every structural column nonempty;
#   * +1/-1 D-sign vector;
#   * NO dynamic regularization (MFLA/BFLA disable it; caller owns any
#     explicit signed static shift).
# The SDPX symmetric augmented core K=[0 Ar'; Ar -Theta] stores structural
# zeros in reduced-x diagonals, so it is NOT quasi-definite as stored and
# must not be sent here; QDLDL is for operators that already satisfy the
# precondition (e.g. explicitly shifted systems).
using SDPX, Test, LinearAlgebra, SparseArrays
using MultiFloats: Float64x2, Float64x4
using MultiFloatLinearAlgebra
import QDLDL

@testset "SDPX QDLDL sparse provider seam" begin
    for T in (Float64x2, Float64x4)
        n = 8
        nr = n - 4
        Ar = T.(randn(4, nr))
        Theta = T.(Diagonal(1.0 .+ rand(4)))
        Kfull = zeros(T, n, n)
        for j in 1:nr
            Kfull[j, j] = T(1e-8)   # explicit quasi-definite x diagonal
        end
        Kfull[nr+1:n, 1:nr] .= Ar
        Kfull[1:nr, nr+1:n] .= transpose(Ar)
        Kfull[nr+1:n, nr+1:n] .= -Theta
        Kup = sparse(UpperTriangular(Matrix(Symmetric(Kfull, :U))))
        dsigns = vcat(fill(1, nr), fill(-1, 4))
        cache = SDPX.SparseQDLDLCache{T}(Kup, dsigns; nrhs=1)
        SDPX.factorize!(cache, Kup, 1)
        rhs = T.(randn(n))
        expected = Kfull \ rhs
        got = zeros(T, n)
        SDPX.solve!(cache, got, rhs)
        @test norm(got - expected) < 1e-10
        # multi-RHS
        rhs2 = T.(randn(n, 3))
        got2 = zeros(T, n, 3)
        SDPX.solve_multi!(cache, got2, rhs2)
        @test norm(got2 - (Kfull \ rhs2)) < 1e-10
        # epoch reuse skips a re-factor; solve still holds authority
        SDPX.factorize!(cache, Kup, 1)
        got3 = zeros(T, n)
        SDPX.solve!(cache, got3, rhs)
        @test got3 ≈ expected
        # fail closed: non-quasi-definite (zero x diagonal) rejects
        bad = zeros(T, n, n)
        bad[nr+1:n, 1:nr] .= Ar
        bad[1:nr, nr+1:n] .= transpose(Ar)
        bad[nr+1:n, nr+1:n] .= -Theta
        bad_up = sparse(UpperTriangular(Matrix(Symmetric(bad, :U))))
        @test_throws Exception SDPX.SparseQDLDLCache{T}(
            bad_up, dsigns; nrhs=1,
        )
    end
end

@testset "QDLDL refine lifecycle (C3)" begin
    using SDPX, MultiFloats, SparseArrays, LinearAlgebra, Random
    for T in (Float64x2, Float64x4)
        n = 12
        rng = MersenneTwister(777)
        A = T.(randn(rng, n, n))
        A = A' * A + T(1) * I
        if T === Float64x2
            A = T.(Matrix{Float64}(A))
        end
        # build a quasi-definite upper-triangular CSC like the seam test
        nr, m = n ÷ 2, n - n ÷ 2
        Ar = T.(randn(rng, m, nr))
        Theta = T.(Matrix{T}(I, m, m))
        Kfull = zeros(T, n, n)
        for j in 1:nr
            Kfull[j, j] = T(1e-8)
        end
        Kfull[nr+1:n, 1:nr] .= Ar
        Kfull[1:nr, nr+1:n] .= transpose(Ar)
        Kfull[nr+1:n, nr+1:n] .= -Theta
        Kup = sparse(UpperTriangular(Matrix(Symmetric(Kfull, :U))))
        dsigns = vcat(fill(1, nr), fill(-1, m))
        cache = SDPX.SparseQDLDLCache{T}(Kup, dsigns; nrhs=1)
        req = SDPX.FactorRequirements(n, 1)
        SDPX.prepare!(cache, req)
        SDPX.factorize!(cache, Kup, 1)
        b = T.(randn(rng, n))
        x = zeros(T, n)
        SDPX.solve!(cache, x, b)
        @test norm(Kfull * x .- b) < T(1e-8)
        # refine: residual from the original (unregularized) operator
        res = zeros(T, n)
        mul!(res, Kfull, x)
        res .= b .- res
        corr = zeros(T, n)
        SDPX.refine_once!(cache, res, corr)
        x .+= corr
        @test norm(Kfull * x .- b) < T(1e-9)
        # fail closed: nonfinite residual rejects
        bad = copy(res); bad[1] = T(NaN)
        @test_throws Exception SDPX.refine_once!(cache, bad, zeros(T, n))
        # fail closed: wrong dimension rejects
        @test_throws Exception SDPX.refine_once!(cache, res, zeros(T, n + 1))
        # counter semantics: refine_count incremented once per call
        @test cache.refine_count == 1
    end
end

@testset "QDLDL frozen-shape ownership (C4)" begin
    using SDPX, MultiFloats, SparseArrays, LinearAlgebra, Random
    T = Float64x2
    n = 10; nr = 5; m = 5
    rng = MersenneTwister(4242)
    Ar = T.(randn(rng, m, nr))
    Kfull = zeros(T, n, n)
    for j in 1:nr; Kfull[j, j] = T(1e-8); end
    Kfull[nr+1:n, 1:nr] .= Ar
    Kfull[1:nr, nr+1:n] .= transpose(Ar)
    Kfull[nr+1:n, nr+1:n] .= -T(1) * T.(Matrix{Float64}(I, m, m))
    Kup = sparse(UpperTriangular(Matrix(Symmetric(Kfull, :U))))
    dsigns = vcat(fill(1, nr), fill(-1, m))
    cache = SDPX.SparseQDLDLCache{T}(Kup, dsigns; nrhs=1)
    # standard requirements dispatch works (C4 hierarchy)
    req = SDPX.FactorRequirements(n, 1)
    @test req isa SDPX.AbstractFactorRequirements
    SDPX.prepare!(cache, req)
    @test cache.status === SDPX.Prepared
    # same-shape new epoch resets state deterministically
    SDPX.factorize!(cache, Kup, 1)
    @test cache.factor_epoch == 1
    req2 = SDPX.FactorRequirements(n, 2)
    SDPX.prepare!(cache, req2)
    @test cache.factor_epoch == 0
    @test cache.status === SDPX.Prepared
    # dimension change rejects rather than leaving a stale Fresh provider
    req3 = SDPX.FactorRequirements(n + 1, 2)
    @test_throws Exception SDPX.prepare!(cache, req3)
end
