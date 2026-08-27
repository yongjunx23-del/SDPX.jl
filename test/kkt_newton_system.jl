using Test
using LinearAlgebra
using SparseArrays
using SDPX

@testset "semantic HSD Newton system" begin
    T = Float64
    first_block = SDPX.LocalConeLinearization(
        1:1, reshape(T[2], 1, 1), T[0.25],
    )
    second_block = SDPX.LocalConeLinearization(
        2:3, T[3 1; 1 4], T[-0.5, 0.75],
    )
    cone = SDPX.assemble_cone_linearization(
        T, 3, [first_block, second_block],
    )
    @test cone.block_ranges == [1:1, 2:3]
    @test cone.operator == T[2 0 0; 0 3 1; 0 1 4]
    @test cone.corrector_rhs == T[0.25, -0.5, 0.75]

    A = sparse(T[1 2; -1 0.5; 0 3])
    b = T[1, -2, 0.5]
    c = T[-1.5, 2]
    tau = T(1.25)
    kappa = T(0.8)
    dx = T[0.3, -0.2]
    dy = T[0.4, -0.1, 0.25]
    ds = T[-0.6, 0.7, 0.2]
    dtau = T(-0.15)
    dkappa = T(0.35)

    primal = A * dx + ds - b * dtau
    dual = transpose(A) * dy + c * dtau
    gap = -dot(c, dx) - dot(b, dy) + dkappa
    cone_rhs = ds + cone.operator * dy
    scalar = kappa * dtau + tau * dkappa
    rhs = SDPX.HSDNewtonRHS(primal, dual, gap, cone_rhs, scalar)
    system = SDPX.NewtonSystem(A, b, c, cone, tau, kappa, rhs)
    direction = SDPX.NewtonDirection(dx, dy, ds, dtau, dkappa)
    residual = SDPX.NewtonResidual(system)
    SDPX.newton_residual!(residual, system, direction)
    @test SDPX.max_newton_residual(residual) <= 16eps(T)
    @test residual.primal_affine ≈ zeros(T, 3) atol=16eps(T)
    @test residual.dual_affine ≈ zeros(T, 2) atol=16eps(T)
    @test residual.cone_complementarity ≈ zeros(T, 3) atol=16eps(T)

    from_residuals = SDPX.residual_newton_rhs(
        -primal, -dual, -gap, cone_rhs, scalar,
    )
    @test from_residuals.primal_affine == primal
    @test from_residuals.dual_affine == dual
    @test from_residuals.homogeneous_gap == gap
    @test from_residuals.cone_corrector == cone_rhs
    @test from_residuals.tau_kappa == scalar
end

@testset "cone linearization fail-closed invariants" begin
    T = Float64
    good = SDPX.LocalConeLinearization(1:1, ones(T, 1, 1), zeros(T, 1))
    gap = SDPX.LocalConeLinearization(3:3, ones(T, 1, 1), zeros(T, 1))
    overlap = SDPX.LocalConeLinearization(1:1, ones(T, 1, 1), zeros(T, 1))
    @test_throws ArgumentError SDPX.assemble_cone_linearization(T, 3, [good, gap])
    @test_throws ArgumentError SDPX.assemble_cone_linearization(T, 2, [good, overlap])
    @test_throws ArgumentError SDPX.LocalConeLinearization(
        1:2, T[1 2; 0 1], zeros(T, 2),
    )
    @test_throws ArgumentError SDPX.LocalConeLinearization(
        1:1, reshape(T[Inf], 1, 1), zeros(T, 1),
    )
end

@testset "generic precision semantic residual" begin
    setprecision(BigFloat, 192) do
        T = BigFloat
        cone = SDPX.assemble_cone_linearization(
            T, 1,
            [SDPX.LocalConeLinearization(1:1, reshape(T[3], 1, 1), T[2])],
        )
        A = reshape(T[2], 1, 1)
        b = T[1]
        c = T[-1]
        direction = SDPX.NewtonDirection(T[0.5], T[0.25], T[1.25], T(0.5), T(0.75))
        rhs = SDPX.HSDNewtonRHS(
            A * direction.dx + direction.ds - b * direction.dtau,
            transpose(A) * direction.dy + c * direction.dtau,
            -dot(c, direction.dx) - dot(b, direction.dy) + direction.dkappa,
            direction.ds + cone.operator * direction.dy,
            T(2) * direction.dtau + T(1.5) * direction.dkappa,
        )
        system = SDPX.NewtonSystem(A, b, c, cone, T(1.5), T(2), rhs)
        residual = SDPX.NewtonResidual(system)
        SDPX.newton_residual!(residual, system, direction)
        @test iszero(SDPX.max_newton_residual(residual))
    end
end
