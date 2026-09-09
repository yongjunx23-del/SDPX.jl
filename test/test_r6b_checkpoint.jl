# R6-B checkpoint round-trip qualification (standalone; not wired into runtests.jl).
#
# Exercises the Serialization-backed save_checkpoint/load_checkpoint pair
# (§5.5): Float64 and BigFloat payload round-trips, the atomic tmp+rename
# discipline, corrupt-input fail-closed loads, wrong-type rejection, and
# forged format_version rejection.

using Test
using Serialization
using SDPX

@testset "R6-B checkpoint round-trip and recovery" begin
    @testset "Float64 payload round-trips exactly" begin
        dir = mktempdir()
        path = joinpath(dir, "checkpoint.bin")
        x = [1.0, 2.0]
        X = [Matrix{Float64}([1.0 0.0; 0.0 1.0])]
        y = [0.5]
        Y = [Matrix{Float64}([2.0 0.0; 0.0 2.0])]
        μ = [1.0e-3]
        dims = (L = 1, m = 2, n = 2, k = [2])
        SDPX.save_checkpoint(path, Float64, x, X, y, Y, μ, 7, 1, dims)
        @test isfile(path)
        # Atomic tmp+rename discipline: no .tmp residue remains.
        @test !isfile(path * ".tmp")
        cp = SDPX.load_checkpoint(path, Float64)
        @test cp.format_version == SDPX.CHECKPOINT_FORMAT_VERSION
        @test cp.x == x
        @test cp.X == X
        @test cp.y == y
        @test cp.Y == Y
        @test cp.μ == μ
        @test cp.iter == 7
        @test cp.restarts == 1
        @test cp.dims == dims
    end

    @testset "BigFloat payload round-trips exactly" begin
        dir = mktempdir()
        path = joinpath(dir, "checkpoint_big.bin")
        x = BigFloat[1, 2]
        X = [BigFloat[1 0; 0 1]]
        y = BigFloat[big"0.5"]
        Y = [BigFloat[2 0; 0 2]]
        μ = BigFloat[big"1e-3"]
        dims = (L = 1, m = 2, n = 2, k = [2])
        SDPX.save_checkpoint(path, BigFloat, x, X, y, Y, μ, 3, 0, dims)
        @test isfile(path)
        @test !isfile(path * ".tmp")
        cp = SDPX.load_checkpoint(path, BigFloat)
        @test cp.format_version == SDPX.CHECKPOINT_FORMAT_VERSION
        @test cp.x == x
        @test cp.X == X
        @test cp.y == y
        @test cp.Y == Y
        @test cp.μ == μ
        @test cp.iter == 3
        @test cp.restarts == 0
        @test cp.dims == dims
    end

    @testset "corrupt input fails closed on load" begin
        dir = mktempdir()
        garbage_path = joinpath(dir, "garbage.bin")
        open(garbage_path, "w") do io
            write(io, rand(UInt8, 32))
        end
        # Garbage bytes must throw, never silently return a checkpoint.
        @test_throws Exception SDPX.load_checkpoint(garbage_path, Float64)

        path = joinpath(dir, "checkpoint.bin")
        SDPX.save_checkpoint(
            path,
            Float64,
            [1.0],
            [Matrix{Float64}([1.0;;])],
            [0.5],
            [Matrix{Float64}([2.0;;])],
            [1.0e-3],
            7,
            1,
            (L = 1, m = 1, n = 1, k = [1]),
        )
        truncated_path = joinpath(dir, "truncated.bin")
        open(truncated_path, "w") do io
            write(io, read(path)[1:10])
        end
        @test_throws Exception SDPX.load_checkpoint(truncated_path, Float64)
    end

    @testset "wrong-type load is rejected with ArgumentError" begin
        dir = mktempdir()
        path = joinpath(dir, "checkpoint.bin")
        SDPX.save_checkpoint(
            path,
            Float64,
            [1.0],
            [Matrix{Float64}([1.0;;])],
            [0.5],
            [Matrix{Float64}([2.0;;])],
            [1.0e-3],
            7,
            1,
            (L = 1, m = 1, n = 1, k = [1]),
        )
        @test_throws ArgumentError SDPX.load_checkpoint(path, BigFloat)
    end

    @testset "forged format_version is rejected with ArgumentError" begin
        dir = mktempdir()
        path = joinpath(dir, "forged.bin")
        forged = SDPX.Checkpoint{Float64}(
            SDPX.CHECKPOINT_FORMAT_VERSION + 999,
            [1.0],
            [Matrix{Float64}([1.0;;])],
            [0.5],
            [Matrix{Float64}([2.0;;])],
            [1.0e-3],
            7,
            1,
            (L = 1, m = 1, n = 1, k = [1]),
        )
        open(path, "w") do io
            Serialization.serialize(io, forged)
        end
        @test_throws ArgumentError SDPX.load_checkpoint(path, Float64)
    end
end
