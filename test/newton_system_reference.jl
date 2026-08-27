# Independent semantic Newton-system verification.
#
# This file deliberately constructs the five equations directly. It does not
# call production residual helpers while forming reference values.

using Test
using LinearAlgebra
using SparseArrays
using SDPX

function _ns_direct_lhs(A, b, c, H, tau, kappa, direction)
    return (
        primal=A * direction.dx + direction.ds - b * direction.dtau,
        dual=transpose(A) * direction.dy + c * direction.dtau,
        gap=-dot(c, direction.dx) - dot(b, direction.dy) + direction.dkappa,
        cone=direction.ds + H * direction.dy,
        scalar=kappa * direction.dtau + tau * direction.dkappa,
    )
end

function _ns_direct_residuals(A, b, c, H, tau, kappa, rhs, direction)
    lhs = _ns_direct_lhs(A, b, c, H, tau, kappa, direction)
    return (
        primal=lhs.primal - rhs.primal_affine,
        dual=lhs.dual - rhs.dual_affine,
        gap=lhs.gap - rhs.homogeneous_gap,
        cone=lhs.cone - rhs.cone_corrector,
        scalar=lhs.scalar - rhs.tau_kappa,
    )
end

function _ns_fixture(family::Symbol, ::Type{T}=Float64) where {T<:AbstractFloat}
    if family === :lp
        A = sparse(T[1 0.5; -0.25 1.5])
        b = T[1, -0.5]
        c = T[-0.75, 1.25]
        H = T[1.5 0; 0 0.75]
        direction = SDPX.NewtonDirection(
            T[0.2, -0.3], T[0.3, -0.1], T[-0.4, 0.25],
            T(-0.15), T(0.35),
        )
        tau, kappa = T(1.2), T(0.8)
    elseif family === :soc
        A = sparse(T[1 0.5; -0.25 1.5; 0.75 -1])
        b = T[1, -0.5, 0.25]
        c = T[-0.75, 1.25]
        # Frozen self-adjoint SOC scaling in canonical Lorentz coordinates.
        H = T[2 0.2 0; 0.2 1.4 0.1; 0 0.1 1.1]
        direction = SDPX.NewtonDirection(
            T[0.2, -0.3], T[0.1, -0.2, 0.15], T[-0.4, 0.3, 0.2],
            T(-0.1), T(0.25),
        )
        tau, kappa = T(1.1), T(0.9)
    elseif family === :psd
        A = sparse(T[1 -0.5; 0.25 1.25; -0.75 0.5])
        b = T[0.5, -1, 0.75]
        c = T[1.5, -0.25]
        # A 2x2 PSD block has three svec coordinates. H is self-adjoint in the
        # Euclidean svec metric used by the semantic system.
        H = T[2.2 0.1 0.2; 0.1 1.6 -0.1; 0.2 -0.1 2.4]
        direction = SDPX.NewtonDirection(
            T[-0.15, 0.35], T[0.2, -0.1, 0.3], T[-0.25, 0.4, -0.2],
            T(0.12), T(-0.3),
        )
        tau, kappa = T(1.3), T(0.7)
    else
        throw(ArgumentError("unknown fixture family $family"))
    end

    lhs = _ns_direct_lhs(A, b, c, H, tau, kappa, direction)
    rhs = SDPX.HSDNewtonRHS(lhs.primal, lhs.dual, lhs.gap, lhs.cone, lhs.scalar)
    cone = SDPX.assemble_cone_linearization(
        T, size(A, 1),
        [SDPX.LocalConeLinearization(1:size(A, 1), H, copy(lhs.cone))],
    )
    system = SDPX.NewtonSystem(A, b, c, cone, tau, kappa, rhs)
    return (; family, A, b, c, H, tau, kappa, rhs, system, direction)
end

function _ns_production_direction(system::SDPX.NewtonSystem{T}) where {T}
    m, n = size(system.A)
    session = SDPX.ExpandedKKTSession(T, n, m)
    @test SDPX.factor_expanded_kkt!(session, system)
    condensed_rhs = zeros(T, session.dimension)
    SDPX.expanded_rhs!(condensed_rhs, system)
    condensed = similar(condensed_rhs)
    @test SDPX.solve_expanded!(condensed, session, condensed_rhs)
    @test SDPX.refine_expanded!(condensed, session, condensed_rhs)
    return SDPX.recover_expanded_direction(system, condensed)
end

function _ns_production_residuals(system, direction)
    residual = SDPX.NewtonResidual(system)
    SDPX.newton_residual!(residual, system, direction)
    return residual
end

function _ns_maximum(residuals)
    return maximum((
        maximum(abs, residuals.primal; init=zero(eltype(residuals.primal))),
        maximum(abs, residuals.dual; init=zero(eltype(residuals.dual))),
        abs(residuals.gap),
        maximum(abs, residuals.cone; init=zero(eltype(residuals.cone))),
        abs(residuals.scalar),
    ))
end

@testset "Newton system direct LP/SOC/PSD fixtures" begin
    for family in (:lp, :soc, :psd)
        @testset "$family" begin
            fixture = _ns_fixture(family)
            direction = _ns_production_direction(fixture.system)
            direct = _ns_direct_residuals(
                fixture.A, fixture.b, fixture.c, fixture.H,
                fixture.tau, fixture.kappa, fixture.rhs, direction,
            )
            production = _ns_production_residuals(fixture.system, direction)

            @test direct.primal ≈ production.primal_affine atol=2e-13 rtol=0
            @test direct.dual ≈ production.dual_affine atol=2e-13 rtol=0
            @test direct.gap ≈ production.homogeneous_gap atol=2e-13 rtol=0
            @test direct.cone ≈ production.cone_complementarity atol=2e-13 rtol=0
            @test direct.scalar ≈ production.tau_kappa atol=2e-13 rtol=0
            @test _ns_maximum(direct) <= 2e-12
            @test direction.dx ≈ fixture.direction.dx atol=2e-12 rtol=0
            @test direction.dy ≈ fixture.direction.dy atol=2e-12 rtol=0
            @test direction.ds ≈ fixture.direction.ds atol=2e-12 rtol=0
            @test direction.dtau ≈ fixture.direction.dtau atol=2e-12 rtol=0
            @test direction.dkappa ≈ fixture.direction.dkappa atol=2e-12 rtol=0
        end
    end
end

@testset "affine residual homotopy under one common step" begin
    fixture = _ns_fixture(:soc)
    A, b, c, H = fixture.A, fixture.b, fixture.c, fixture.H
    x = [0.2, -0.1]
    s = [1.2, 0.15, -0.05]
    y = [0.8, -0.2, 0.1]
    tau, kappa = 1.15, 0.85
    rP = A * x + s - b * tau
    rD = transpose(A) * y + c * tau
    rG = -dot(c, x) - dot(b, y) + kappa
    cone_rhs = [-0.4, 0.25, -0.1]
    scalar_rhs = -tau * kappa
    rhs = SDPX.residual_newton_rhs(rP, rD, rG, cone_rhs, scalar_rhs)
    cone = SDPX.assemble_cone_linearization(
        Float64, 3, [SDPX.LocalConeLinearization(1:3, H, cone_rhs)],
    )
    system = SDPX.NewtonSystem(A, b, c, cone, tau, kappa, rhs)
    direction = _ns_production_direction(system)

    alpha = 0.37
    x_new = x + alpha * direction.dx
    s_new = s + alpha * direction.ds
    y_new = y + alpha * direction.dy
    tau_new = tau + alpha * direction.dtau
    kappa_new = kappa + alpha * direction.dkappa
    rP_new = A * x_new + s_new - b * tau_new
    rD_new = transpose(A) * y_new + c * tau_new
    rG_new = -dot(c, x_new) - dot(b, y_new) + kappa_new

    @test rP_new ≈ (1 - alpha) * rP atol=2e-13 rtol=0
    @test rD_new ≈ (1 - alpha) * rD atol=2e-13 rtol=0
    @test rG_new ≈ (1 - alpha) * rG atol=2e-13 rtol=0
end

@testset "predictor and corrector preserve affine RHS" begin
    fixture = _ns_fixture(:psd)
    rP = [0.3, -0.2, 0.1]
    rD = [-0.4, 0.25]
    rG = 0.15
    predictor = SDPX.residual_newton_rhs(
        rP, rD, rG, [-1.0, -0.75, -0.5], -fixture.tau * fixture.kappa,
    )
    corrector = SDPX.residual_newton_rhs(
        rP, rD, rG, [0.2, 0.1, 0.3], 0.45,
    )
    @test predictor.primal_affine == corrector.primal_affine == -rP
    @test predictor.dual_affine == corrector.dual_affine == -rD
    @test predictor.homogeneous_gap == corrector.homogeneous_gap == -rG
    @test predictor.cone_corrector != corrector.cone_corrector
    @test predictor.tau_kappa != corrector.tau_kappa
end

@testset "independent BigFloat five-equation residual oracle" begin
    setprecision(BigFloat, 256) do
        for family in (:lp, :soc, :psd)
            fixture = _ns_fixture(family)
            direction = _ns_production_direction(fixture.system)
            rhs_big = (
                primal_affine=BigFloat.(fixture.rhs.primal_affine),
                dual_affine=BigFloat.(fixture.rhs.dual_affine),
                homogeneous_gap=BigFloat(fixture.rhs.homogeneous_gap),
                cone_corrector=BigFloat.(fixture.rhs.cone_corrector),
                tau_kappa=BigFloat(fixture.rhs.tau_kappa),
            )
            direction_big = (
                dx=BigFloat.(direction.dx), dy=BigFloat.(direction.dy),
                ds=BigFloat.(direction.ds), dtau=BigFloat(direction.dtau),
                dkappa=BigFloat(direction.dkappa),
            )
            oracle = _ns_direct_residuals(
                BigFloat.(Matrix(fixture.A)), BigFloat.(fixture.b),
                BigFloat.(fixture.c), BigFloat.(fixture.H),
                BigFloat(fixture.tau), BigFloat(fixture.kappa),
                rhs_big, direction_big,
            )
            @test _ns_maximum(oracle) <= big"2e-12"

            production = _ns_production_residuals(fixture.system, direction)
            @test oracle.primal ≈ BigFloat.(production.primal_affine) atol=big"2e-13" rtol=0
            @test oracle.dual ≈ BigFloat.(production.dual_affine) atol=big"2e-13" rtol=0
            @test oracle.gap ≈ BigFloat(production.homogeneous_gap) atol=big"2e-13" rtol=0
            @test oracle.cone ≈ BigFloat.(production.cone_complementarity) atol=big"2e-13" rtol=0
            @test oracle.scalar ≈ BigFloat(production.tau_kappa) atol=big"2e-13" rtol=0
        end
    end
end
