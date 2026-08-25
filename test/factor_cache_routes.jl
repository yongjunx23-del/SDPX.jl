# Route-specific FactorCache implementations (Subagent B, PR1/PR4).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra
using SparseArrays

@testset "LPLUCache factorize + solve" begin
    A = [4.0 3.0; 6.0 3.0]
    cache = SDPX.LPLUCache{Float64}(2)
    SDPX.factorize!(cache, A, 1)
    @test SDPX.factor_status(cache) === :factored
    @test SDPX.factor_matrix_epoch(cache) == 1
    b = [1.0, 2.0]
    x = zeros(2)
    SDPX.solve!(cache, x, b)
    @test A * x ≈ b
    # epoch reuse: same epoch does not re-factorize
    SDPX.factorize!(cache, A, 1)
    @test SDPX.factor_matrix_epoch(cache) == 1
    # new epoch re-factorizes
    SDPX.factorize!(cache, A, 2)
    @test SDPX.factor_matrix_epoch(cache) == 2
    SDPX.invalidate!(cache)
    @test SDPX.factor_status(cache) === :unprepared
end

@testset "DenseSchurCholeskyCache factorize + solve" begin
    A = [4.0 1.0; 1.0 3.0]
    cache = SDPX.DenseSchurCholeskyCache{Float64}(2)
    SDPX.factorize!(cache, A, 1)
    @test SDPX.factor_status(cache) === :factored
    b = [1.0, 2.0]
    x = zeros(2)
    SDPX.solve!(cache, x, b)
    @test A * x ≈ b
    # epoch reuse
    SDPX.factorize!(cache, A, 1)
    @test SDPX.factor_matrix_epoch(cache) == 1
end

@testset "ArrowFactorCache factorize + solve" begin
    # block-arrow matrix [D B; Bᵀ C]
    D = [2.0 0.0; 0.0 3.0]
    B = [1.0; 1.0]
    C = [5.0]
    A = [D B; B' C]
    cache = SDPX.ArrowFactorCache{Float64}(3, 2)
    SDPX.factorize!(cache, A, 1)
    @test SDPX.factor_status(cache) === :factored
    b = [1.0, 2.0, 3.0]
    x = zeros(3)
    SDPX.solve!(cache, x, b)
    @test A * x ≈ b
    # epoch reuse
    SDPX.factorize!(cache, A, 1)
    @test SDPX.factor_matrix_epoch(cache) == 1
end

@testset "route caches work in BigFloat" begin
    setprecision(BigFloat, 256) do
        A = BigFloat[4 3; 6 3]
        cache = SDPX.LPLUCache{BigFloat}(2)
        SDPX.factorize!(cache, A, 1)
        b = BigFloat[1, 2]
        x = zeros(BigFloat, 2)
        SDPX.solve!(cache, x, b)
        @test A * x ≈ b
    end
end
