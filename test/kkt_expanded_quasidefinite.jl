using Test
using LinearAlgebra
using SparseArrays
using SDPX

function expanded_fixture(::Type{T}) where {T<:AbstractFloat}
    A = sparse(T[1 2; -1 0.5; 0 3])
    b = T[1, -2, 0.5]
    c = T[-1.5, 2]
    H = T[2 0 0; 0 3 1; 0 1 4]
    cone = SDPX.assemble_cone_linearization(
        T, 3,
        [SDPX.LocalConeLinearization(1:3, H, zeros(T, 3))],
    )
    tau = T(1.25)
    kappa = T(0.8)
    expected = SDPX.NewtonDirection(
        T[0.3, -0.2], T[0.4, -0.1, 0.25],
        T[-0.6, 0.7, 0.2], T(-0.15), T(0.35),
    )
    primal = A * expected.dx + expected.ds - b * expected.dtau
    dual = transpose(A) * expected.dy + c * expected.dtau
    gap = -dot(c, expected.dx) - dot(b, expected.dy) + expected.dkappa
    cone_rhs = expected.ds + H * expected.dy
    scalar = kappa * expected.dtau + tau * expected.dkappa
    rhs = SDPX.HSDNewtonRHS(primal, dual, gap, cone_rhs, scalar)
    system = SDPX.NewtonSystem(A, b, c, cone, tau, kappa, rhs)
    return system, expected
end

@testset "expanded exact frozen-sign KKT route" begin
    system, expected = expanded_fixture(Float64)
    session = SDPX.ExpandedKKTSession(Float64, 2, 3; rhs_count=2)
    @test SDPX.factor_expanded_kkt!(session, system)
    @test session.status == SDPX.EXPANDED_KKT_FACTORED
    @test SDPX.expected_expanded_inertia(system) == SDPX.KKTInertia(2, 4, 0)
    @test session.expected_inertia == SDPX.KKTInertia(2, 4, 0)
    @test session.inertia_factor.inertia == session.expected_inertia
    @test session.regularization_attempts >= 0

    rhs = zeros(Float64, session.dimension)
    SDPX.expanded_rhs!(rhs, system)
    solution = similar(rhs)
    @test SDPX.solve_expanded!(solution, session, rhs)
    @test SDPX.refine_expanded!(solution, session, rhs)
    @test session.status == SDPX.EXPANDED_KKT_UNREGULARIZED_CERTIFIED
    direction = SDPX.recover_expanded_direction(system, solution)
    residual = SDPX.NewtonResidual(system)
    SDPX.newton_residual!(residual, system, direction)
    @test SDPX.max_newton_residual(residual) <= 2e-12
    @test direction.dx ≈ expected.dx atol=2e-12
    @test direction.dy ≈ expected.dy atol=2e-12
    @test direction.ds ≈ expected.ds atol=2e-12
    @test direction.dtau ≈ expected.dtau atol=2e-12
    @test direction.dkappa ≈ expected.dkappa atol=2e-12

    # Predictor/corrector RHS panels share exactly one factorization.
    rhs_panel = hcat(rhs, 2rhs)
    solution_panel = similar(rhs_panel)
    # The earlier solve was semantically certified; the same numeric factor
    # remains owned by the session and is explicitly reopened for the panel.
    session.status = SDPX.EXPANDED_KKT_FACTORED
    @test SDPX.solve_expanded!(solution_panel, session, rhs_panel)
    @test SDPX.refine_expanded!(solution_panel, session, rhs_panel)
    @test solution_panel[:, 2] ≈ 2solution_panel[:, 1] atol=2e-12
end

@testset "expanded route rejects wrong signed inertia" begin
    T = Float64
    A = zeros(T, 1, 1)
    b = zeros(T, 1)
    c = zeros(T, 1)
    cone = SDPX.assemble_cone_linearization(
        T, 1,
        [SDPX.LocalConeLinearization(1:1, reshape(T[-1], 1, 1), zeros(T, 1))],
    )
    rhs = SDPX.HSDNewtonRHS(zeros(T, 1), zeros(T, 1), zero(T), zeros(T, 1), zero(T))
    system = SDPX.NewtonSystem(A, b, c, cone, one(T), one(T), rhs)
    session = SDPX.ExpandedKKTSession(T, 1, 1)
    # The expected inertia is a structure-derived authority, not mutable input.
    session.expected_inertia = SDPX.KKTInertia(3, 0, 0)
    @test !SDPX.factor_expanded_kkt!(session, system; max_regularization_attempts=3)
    @test session.expected_inertia == SDPX.KKTInertia(1, 2, 0)
    @test session.status == SDPX.EXPANDED_KKT_WRONG_INERTIA
    @test session.inertia_factor.inertia != session.expected_inertia
    @test !session.factor.success
end

@testset "expanded generic precision fallback" begin
    setprecision(BigFloat, 192) do
        system, expected = expanded_fixture(BigFloat)
        session = SDPX.ExpandedKKTSession(BigFloat, 2, 3)
        @test SDPX.factor_expanded_kkt!(session, system)
        rhs = zeros(BigFloat, session.dimension)
        SDPX.expanded_rhs!(rhs, system)
        solution = similar(rhs)
        @test SDPX.solve_expanded!(solution, session, rhs)
        @test SDPX.refine_expanded!(solution, session, rhs)
        direction = SDPX.recover_expanded_direction(system, solution)
        residual = SDPX.NewtonResidual(system)
        SDPX.newton_residual!(residual, system, direction)
        @test SDPX.max_newton_residual(residual) < big"1e-45"
        @test direction.dx ≈ expected.dx rtol=big"1e-45" atol=big"1e-45"
    end
end
