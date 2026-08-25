# Wave A-3 reference: FactorCache state machine + 0-byte warm hot-path contract.

using SDPX
using Test
using LinearAlgebra

@testset "FactorCache 0-byte warm path" begin
    # On an already-prepared cache, a warm factorize! + solve! + refinement
    # must allocate ZERO Julia bytes: all storage was committed by prepare!.
    n = 32
    cache = SDPX.DenseFactorCache{Float64}(n)

    M = randn(n, n)
    A1 = M * M' + n * I
    A2 = A1 + 0.25 * I
    A3 = A2 + 0.5 * I

    # Warm up JIT and buffers.
    SDPX.factorize!(cache, A1, 1)
    SDPX.factorize!(cache, A2, 2)
    b = randn(n)
    x = zeros(n)
    corr = zeros(n)
    SDPX.solve!(cache, x, b)
    SDPX.refine_once!(cache, corr, b)

    # Measure warm calls. Each must be exactly 0 bytes.
    a_factorize = @allocated SDPX.factorize!(cache, A3, 3)
    a_solve     = @allocated SDPX.solve!(cache, x, b)
    a_refine    = @allocated SDPX.refine_once!(cache, corr, b)
    # Same-epoch skip must also be 0 bytes.
    a_skip      = @allocated SDPX.factorize!(cache, A3, 3)

    @test a_factorize == 0
    @test a_solve == 0
    @test a_refine == 0
    @test a_skip == 0
    @test SDPX.factor_status(cache) === SDPX.Fresh

    # Correctness still holds after the 0-byte refactor.
    @test isapprox(A3 * x, b; atol=1e-9)
    @test isapprox(A3 * corr, b; atol=1e-9)
end

@testset "FactorCache state machine: success → failure → solve rejected → recovery" begin
    n = 4
    cache = SDPX.DenseFactorCache{Float64}(n)
    M = randn(n, n)
    A_good = M * M' + n * I          # positive definite: factorizes cleanly
    A_bad = -Matrix{Float64}(I, n, n)  # negative definite: cholesky throws
    x_true = randn(n)
    b = A_good * x_true
    x = zeros(n)

    # (1) SUCCESS: prepare → factorize → Fresh → solve works.
    @test SDPX.prepare!(cache, SDPX.FactorRequirements(n)) === cache
    @test SDPX.factor_status(cache) === SDPX.Prepared
    SDPX.factorize!(cache, A_good, 1)
    @test SDPX.factor_status(cache) === SDPX.Fresh
    SDPX.solve!(cache, x, b)
    @test isapprox(x, x_true; atol=1e-9)

    # (2) FAILURE: a factorization that throws must leave the cache Failed
    #     (fail-closed) and surface the underlying exception.
    @test_throws Exception SDPX.factorize!(cache, A_bad, 2)
    @test SDPX.factor_status(cache) === SDPX.Failed

    # (3) SOLVE REJECTED: a Failed cache must reject solve! (and refinement).
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(cache, x, b)
    @test_throws SDPX.FactorCacheStateError SDPX.refine_once!(cache, x, b)
    @test_throws SDPX.FactorCacheStateError SDPX.solve_multi!(cache, reshape(x, n, 1), reshape(b, n, 1))

    # (4) RECOVERY: a fresh factorize! recovers to Fresh and solve works again.
    SDPX.factorize!(cache, A_good, 3)
    @test SDPX.factor_status(cache) === SDPX.Fresh
    @test SDPX.factor_matrix_epoch(cache) == 3
    SDPX.solve!(cache, x, b)
    @test isapprox(A_good * x, b; atol=1e-9)
    # factor_epoch counts actual factor productions (two successes after prepare).
    @test SDPX.factor_epoch(cache) == 2
end
