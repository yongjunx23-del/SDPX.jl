using JLD2
using JuMP
using LinearAlgebra
import MathOptInterface as EXT_MOI
using Test

@testset "optional extension regressions" begin
    @testset "JLD2 checkpoint round trips and validates metadata" begin
        mktempdir() do directory
            path = joinpath(directory, "checkpoint.jld2")
            dims = (L=1, m=2, n=1, k=[2])
            x = [1.0, 2.0]
            X = [Matrix{Float64}(I, 2, 2)]
            y = [3.0]
            Y = [2.0 .* Matrix{Float64}(I, 2, 2)]
            mu = [0.25]

            SDPX.save_checkpoint_jld2(
                path,
                Float64,
                x,
                X,
                y,
                Y,
                mu,
                7,
                2,
                dims,
            )
            checkpoint =
                SDPX.load_checkpoint_jld2(path, Float64)
            @test checkpoint.x == x
            @test checkpoint.X == X
            @test checkpoint.y == y
            @test checkpoint.Y == Y
            @test checkpoint.μ == mu
            @test checkpoint.iter == 7
            @test checkpoint.restarts == 2
            @test checkpoint.dims == dims
            @test !isfile(path * ".tmp")
            @test_throws ArgumentError SDPX.load_checkpoint_jld2(
                path,
                BigFloat,
            )

            # A second save atomically replaces the old complete file.
            SDPX.save_checkpoint_jld2(
                path,
                Float64,
                2 .* x,
                X,
                y,
                Y,
                mu,
                8,
                3,
                dims,
            )
            replacement =
                SDPX.load_checkpoint_jld2(path, Float64)
            @test replacement.x == 2 .* x
            @test replacement.iter == 8
            @test replacement.restarts == 3
            @test !isfile(path * ".tmp")

            invalid = SDPX.Checkpoint{Float64}(
                SDPX.CHECKPOINT_FORMAT_VERSION + 1,
                x,
                X,
                y,
                Y,
                mu,
                0,
                0,
                dims,
            )
            JLD2.jldsave(path; checkpoint=invalid)
            @test_throws ArgumentError SDPX.load_checkpoint_jld2(
                path,
                Float64,
            )

            setprecision(BigFloat, 192) do
                big_path =
                    joinpath(directory, "checkpoint-bigfloat.jld2")
                big_x = BigFloat[1, 2]
                big_X = [
                    BigFloat[2 0; 0 3],
                ]
                big_y = BigFloat[4]
                big_Y = [
                    BigFloat[5 0; 0 6],
                ]
                big_mu = BigFloat[big"0.125"]
                SDPX.save_checkpoint_jld2(
                    big_path,
                    BigFloat,
                    big_x,
                    big_X,
                    big_y,
                    big_Y,
                    big_mu,
                    9,
                    1,
                    dims,
                )
                big_checkpoint =
                    SDPX.load_checkpoint_jld2(big_path, BigFloat)
                @test big_checkpoint.x == big_x
                @test big_checkpoint.X == big_X
                @test all(
                    value -> precision(value) == 192,
                    big_checkpoint.x,
                )
            end
        end
    end
end
