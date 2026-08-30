using Test
using SDPX
using LinearAlgebra

# Independent reference mathematics is validation-owned and is not loaded by
# production `using SDPX`.
@test !isdefined(SDPX, :nonsymmetric_full_newton_reference)
Base.include(SDPX, joinpath(@__DIR__, "oracles", "full_newton_reference.jl"))
@test isdefined(SDPX, :nonsymmetric_full_newton_reference)

@testset "validation-owned nonsymmetric full-Newton oracle" begin
    A = reshape([1.0, -0.25, 0.5], 3, 1)
    b = [0.0, 1.0, 2.0]
    c = [0.2]
    slack = [0.0, 1.0, 2.0] # strict Exp-cone primal interior
    blocks = (SDPX.NewtonExpBlock(),)
    result = SDPX.nonsymmetric_full_newton_reference(
        A, b, c, slack, 1.0, blocks,
        zeros(3), zeros(1), 0.0, zeros(3), 1.0, 1.0, 0.0;
        precision_bits=256,
    )
    @test result.status === SDPX.NS_NEWTON_SOLVED
    @test maximum(abs, result.residuals.primal) < big"1e-60"
    @test maximum(abs, result.residuals.dual) < big"1e-60"
    @test abs(result.residuals.gap) < big"1e-60"
    @test maximum(abs, result.residuals.complementarity) < big"1e-60"
    @test abs(result.residuals.tau) < big"1e-60"
end
