# Wave A-3: provider-neutral FactorCache protocol + dense reference tests.

using SDPX
using Test
using LinearAlgebra

@testset "FactorCache protocol" begin
    @testset "abstract type and protocol dispatch" begin
        # AbstractFactorCache is abstract and parameterized by element type.
        @test !isconcretetype(SDPX.AbstractFactorCache)
        @test SDPX.AbstractFactorCache{Float64} isa Type

        # The generic protocol methods throw MethodError unless overridden.
        # Use a minimal anonymous subtype that overrides nothing.
        struct _NoopCache{T} <: SDPX.AbstractFactorCache{T} end
        c = _NoopCache{Float64}()
        @test_throws MethodError SDPX.prepare!(c, nothing)
        @test_throws MethodError SDPX.reserve!(c, nothing)
        @test_throws MethodError SDPX.factorize!(c, nothing, 1)
        @test_throws MethodError SDPX.solve!(c, nothing, nothing)
        @test_throws MethodError SDPX.solve_multi!(c, nothing, nothing)
        @test_throws MethodError SDPX.refine_once!(c, nothing, nothing)
        @test_throws MethodError SDPX.invalidate!(c)
        @test_throws MethodError SDPX.factor_status(c)
        @test_throws MethodError SDPX.factor_diagnostics(c)
        @test_throws MethodError SDPX.factor_matrix_epoch(c)
    end

    @testset "concrete storage structs" begin
        # All fields are concrete (no Any, no abstract backend fields).
        @test isconcretetype(SDPX.SymbolicCache{Float64})
        @test isconcretetype(SDPX.NumericFactorCache{Float64})
        @test isconcretetype(SDPX.SolveScratch{Float64})
        @test isconcretetype(SDPX.DenseFactorCache{Float64})

        cache = SDPX.DenseFactorCache{Float64}(4)
        @test cache.symbolic.n == 4
        @test cache.numeric.factor isa Cholesky{Float64,Matrix{Float64}}
        @test cache.numeric.status === :invalid
        @test cache.numeric.matrix_epoch == -1
        @test length(cache.scratch.rhs) == 4
        @test length(cache.scratch.solution) == 4
    end

    @testset "lifecycle" begin
        n = 5
        cache = SDPX.DenseFactorCache{Float64}(n)

        # Initial state: invalid, no epoch.
        @test SDPX.factor_status(cache) === :invalid
        @test SDPX.factor_matrix_epoch(cache) == -1

        # prepare! / reserve! are no-ops for the dense reference.
        @test SDPX.prepare!(cache, n) === cache
        @test SDPX.reserve!(cache, nothing) === cache

        # Build a symmetric positive-definite matrix.
        M = randn(n, n)
        A1 = M * M' + n * I

        # factorize! at epoch 1 -> fresh.
        SDPX.factorize!(cache, A1, 1)
        @test SDPX.factor_status(cache) === :fresh
        @test SDPX.factor_matrix_epoch(cache) == 1

        # solve! recovers x from A*x = b.
        x_true = randn(n)
        b = A1 * x_true
        x_sol = zeros(n)
        SDPX.solve!(cache, x_sol, b)
        @test norm(x_sol - x_true) < 1e-10

        # factorize! at epoch 2 with a changed matrix -> re-factor, still fresh.
        A2 = A1 + 0.5 * I
        SDPX.factorize!(cache, A2, 2)
        @test SDPX.factor_status(cache) === :fresh
        @test SDPX.factor_matrix_epoch(cache) == 2
        b2 = A2 * x_true
        SDPX.solve!(cache, x_sol, b2)
        @test norm(x_sol - x_true) < 1e-10

        # invalidate! -> :invalid.
        SDPX.invalidate!(cache)
        @test SDPX.factor_status(cache) === :invalid
    end

    @testset "same-epoch reuse" begin
        n = 4
        cache = SDPX.DenseFactorCache{Float64}(n)
        M = randn(n, n)
        A = M * M' + n * I

        # First factorization at epoch 1.
        SDPX.factorize!(cache, A, 1)
        @test SDPX.factor_status(cache) === :fresh
        @test SDPX.factor_matrix_epoch(cache) == 1

        # Second factorization with the SAME epoch must not re-factor: the
        # stored epoch is unchanged and status stays :fresh.
        SDPX.factorize!(cache, A, 1)
        @test SDPX.factor_status(cache) === :fresh
        @test SDPX.factor_matrix_epoch(cache) == 1

        # A different epoch forces a re-factor.
        SDPX.factorize!(cache, A, 2)
        @test SDPX.factor_matrix_epoch(cache) == 2
    end

    @testset "type stability" begin
        n = 6
        cache = SDPX.DenseFactorCache{Float64}(n)
        M = randn(n, n)
        A = M * M' + n * I
        b = randn(n)
        dest = zeros(n)

        @inferred SDPX.factorize!(cache, A, 1)
        @inferred SDPX.solve!(cache, dest, b)
        @inferred SDPX.factor_status(cache)
        @inferred SDPX.factor_matrix_epoch(cache)
        @inferred SDPX.factor_diagnostics(cache)
        @inferred SDPX.invalidate!(cache)
    end

    @testset "correctness" begin
        n = 8
        cache = SDPX.DenseFactorCache{Float64}(n)
        M = randn(n, n)
        A = M * M' + n * I
        SDPX.factorize!(cache, A, 1)

        # Single rhs: solve! recovers x from A*x = b.
        x_true = randn(n)
        b = A * x_true
        x_sol = zeros(n)
        SDPX.solve!(cache, x_sol, b)
        @test norm(x_sol - x_true) < 1e-10

        # Multi rhs: solve_multi! recovers each column.
        k = 3
        X_true = randn(n, k)
        B = A * X_true
        X_sol = zeros(n, k)
        SDPX.solve_multi!(cache, X_sol, B)
        @test norm(X_sol - X_true) < 1e-10

        # refine_once!: correction = F \ residual.
        residual = randn(n)
        correction = zeros(n)
        SDPX.refine_once!(cache, correction, residual)
        @test norm(A * correction - residual) < 1e-10
    end

    @testset "diagnostics" begin
        n = 3
        cache = SDPX.DenseFactorCache{Float64}(n)
        M = randn(n, n)
        A = M * M' + n * I
        SDPX.factorize!(cache, A, 7)
        diag = SDPX.factor_diagnostics(cache)
        @test diag.n == n
        @test diag.matrix_epoch == 7
        @test diag.status === :fresh
        @test diag.info == 0
    end
end
