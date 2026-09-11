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

# ---------------------------------------------------------------------
# C6a: semantic shadow parity — symmetric augmented core vs expanded exact.
# ---------------------------------------------------------------------

"""Rebuild the RHS of a fixture with changed cone/scalar shifts."""
function _c6a_corrected_rhs(fixture, cone_shift, scalar_shift)
    rhs = fixture.rhs
    return SDPX.HSDNewtonRHS(
        copy(rhs.primal_affine), copy(rhs.dual_affine), rhs.homogeneous_gap,
        rhs.cone_corrector .+ cone_shift, rhs.tau_kappa + scalar_shift,
    )
end

"""Run predictor + changed corrector through the symmetric core."""
function _c6a_symmetric_directions(system, V; delta=1e-6)
    workspace = SDPX.build_symmetric_core_workspace(
        system, V, 1, 256, typemax(Int), 0, delta;
        symbolic_epoch=0,
    )
    predictor, predictor_residual = SDPX.solve_core_direction!(workspace, system)
    m = length(system.rhs.cone_corrector)
    cone_shift = [0.13, -0.07, 0.11, 0.05, -0.09][1:m]
    corrected = SDPX.HSDNewtonRHS(
        copy(system.rhs.primal_affine), copy(system.rhs.dual_affine),
        system.rhs.homogeneous_gap,
        system.rhs.cone_corrector .+ cone_shift,
        system.rhs.tau_kappa + 0.19,
    )
    corrector_system = SDPX.NewtonSystem(
        system.A, system.b, system.c, system.cone,
        system.tau, system.kappa, corrected,
    )
    corrector, corrector_residual = SDPX.solve_core_direction!(
        workspace, corrector_system,
    )
    return (workspace, predictor, predictor_residual, corrector, corrector_residual)
end

function _ns_residual_maximum(residual::SDPX.NewtonResidual)
    return maximum((
        maximum(abs, residual.primal_affine; init=zero(eltype(residual.primal_affine))),
        maximum(abs, residual.dual_affine; init=zero(eltype(residual.dual_affine))),
        abs(residual.homogeneous_gap),
        maximum(abs, residual.cone_complementarity; init=zero(eltype(residual.cone_complementarity))),
        abs(residual.tau_kappa),
    ))
end

@testset "C6a symmetric-core semantic shadow parity" begin
    for family in (:lp, :soc, :psd)
        @testset "$family" begin
            fixture = _ns_fixture(family)
            V = Matrix{Float64}(I, size(fixture.A, 2), size(fixture.A, 2))
            (workspace, predictor, predictor_residual, corrector, corrector_residual) =
                _c6a_symmetric_directions(fixture.system, V)

            # Predictor: direction must match the expanded exact route and the
            # frozen five-equation residual must be small.
            expanded = _ns_production_direction(fixture.system)
            @test predictor.dx ≈ expanded.dx atol=2e-10 rtol=0
            @test predictor.dy ≈ expanded.dy atol=2e-10 rtol=0
            @test predictor.ds ≈ expanded.ds atol=2e-10 rtol=0
            @test predictor.dtau ≈ expanded.dtau atol=2e-10 rtol=0
            @test predictor.dkappa ≈ expanded.dkappa atol=2e-10 rtol=0
            @test _ns_residual_maximum(predictor_residual) <= 2e-10

            # Changed corrector: same factor and homogeneous solution reused.
            m = length(fixture.rhs.cone_corrector)
            cone_shift = [0.13, -0.07, 0.11, 0.05, -0.09][1:m]
            corrected_rhs = SDPX.HSDNewtonRHS(
                copy(fixture.rhs.primal_affine), copy(fixture.rhs.dual_affine),
                fixture.rhs.homogeneous_gap,
                fixture.rhs.cone_corrector .+ cone_shift,
                fixture.rhs.tau_kappa + 0.19,
            )
            corrector_system = SDPX.NewtonSystem(
                fixture.A, fixture.b, fixture.c, fixture.system.cone,
                fixture.tau, fixture.kappa, corrected_rhs,
            )
            expanded_corrector = _ns_production_direction(corrector_system)
            @test corrector.dx ≈ expanded_corrector.dx atol=2e-10 rtol=0
            @test corrector.dy ≈ expanded_corrector.dy atol=2e-10 rtol=0
            @test corrector.ds ≈ expanded_corrector.ds atol=2e-10 rtol=0
            @test corrector.dtau ≈ expanded_corrector.dtau atol=2e-10 rtol=0
            @test corrector.dkappa ≈ expanded_corrector.dkappa atol=2e-10 rtol=0
            @test _ns_residual_maximum(corrector_residual) <= 2e-10

            # One factor, one homogeneous solve, two variable solves, no
            # refactor between predictor and corrector.
            @test workspace.homogeneous_solves == 1
            @test workspace.variable_solves == 2
            @test workspace.directions == 2
            @test workspace.homogeneous_epoch == workspace.factor_epoch

            # Independent snapshots (not aliased workspace buffers).
            @test predictor.dx !== corrector.dx
            @test predictor.dy !== corrector.dy
            @test predictor.ds !== corrector.ds
        end
    end

    # Preflight validation must reject an invalid rank basis / operator before
    # any Theta/A*V/pattern materialization or factorization work.
    fixture = _ns_fixture(:soc)
    V = Matrix{Float64}(I, size(fixture.A, 2), size(fixture.A, 2))
    # Non-orthonormal V (columns not unit length) is rejected by isometry.
    bad_V = 2.0 .* V
    @test_throws ArgumentError SDPX.build_symmetric_core_workspace(
        fixture.system, bad_V, 1, 256, typemax(Int), 0, 1e-6;
        symbolic_epoch=0,
    )
    # Rank-reduced V: any A row or c with a component outside range(V) must
    # fail closed before pattern construction.
    V_reduced = reshape([1.0, 0.0], size(fixture.A, 2), 1)  # span e1
    @test_throws ArgumentError SDPX.symmetric_core_pattern_from_system(
        fixture.system, V_reduced,
    )
    # A system genuinely in range(V_reduced) passes preconditions.
    A_reduced = [1.0 0; 0.5 0; -0.25 0]
    c_reduced = [0.75, 0.0]
    reduced_system = SDPX.NewtonSystem(
        A_reduced, fixture.b, c_reduced, fixture.system.cone,
        fixture.tau, fixture.kappa, fixture.rhs,
    )
    pattern_reduced = SDPX.symmetric_core_pattern_from_system(
        reduced_system, V_reduced,
    )
    @test SDPX.symmetric_core_dimension(pattern_reduced) ==
          size(V_reduced, 2) + length(fixture.b)

    # Unsupported arithmetic has no dense LDL provider and fails closed.
    @test_throws ArgumentError SDPX.symmetric_core_provider_available(
        Float32, 24,
    )
    @test_throws ArgumentError SDPX.build_symmetric_core_ldlt_cache(
        Float32, SDPX.SymmetricCorePattern{Float32}(
            sparse(Float32.(Matrix(fixture.A))), [1:size(fixture.A, 1)],
            [:dense_lower],
        ), 24, typemax(Int), 0,
    )

    # BlockProduct Theta materialization is owned and block-diagonal.
    m = size(fixture.A, 1)
    block_lin = SDPX.BlockProductConeLinearization{Float64}(
        [fixture.H], fixture.rhs.cone_corrector, [1:m],
    )
    theta = SDPX.symmetric_core_theta(block_lin)
    @test theta == fixture.H
    @test theta !== fixture.H
end

# ---------------------------------------------------------------------
# C7.1a: epoch-refactorable symmetric core lifecycle.
# ---------------------------------------------------------------------

"""Build a fresh C7 fixture workspace from the LP family."""
function _c71_bundle()
    fixture = _ns_fixture(:lp)
    system = fixture.system
    V = Matrix{Float64}(I, size(system.A, 2), size(system.A, 2))
    workspace = SDPX.build_symmetric_core_workspace(
        system, V, 1, 53, typemax(Int), 0, 1e-6;
        symbolic_epoch=0,
    )
    return (; fixture, system, V, workspace)
end

@testset "C7.1a symmetric core epoch refactor" begin
    bundle = _c71_bundle()
    ws = bundle.workspace
    cache = ws.cache

    # One factor, one homogeneous solve for the first epoch.
    @test SDPX.factor_epoch(cache) == 1
    @test ws.homogeneous_solves == 1
    @test ws.homogeneous_epoch == ws.factor_epoch
    @test ws.factor_receipt !== nothing
    @test ws.factor_receipt.provider === :cholmod
    @test ws.factor_receipt.matrix_epoch == 1
    @test !ws.factor_receipt.proof_valid
    @test ws.factor_receipt.regularization > 0

    # Predictor + corrector share one factor, one homogeneous solve.
    dir1, res1 = SDPX.solve_core_direction!(ws, bundle.system)
    corrected = SDPX.HSDNewtonRHS(
        copy(bundle.system.rhs.primal_affine),
        copy(bundle.system.rhs.dual_affine),
        bundle.system.rhs.homogeneous_gap,
        bundle.system.rhs.cone_corrector .+ [0.13, -0.07],
        bundle.system.rhs.tau_kappa + 0.19,
    )
    corrector_system = SDPX.NewtonSystem(
        bundle.system.A, bundle.system.b, bundle.system.c,
        bundle.system.cone, bundle.system.tau, bundle.system.kappa,
        corrected,
    )
    dir2, res2 = SDPX.solve_core_direction!(ws, corrector_system)
    @test SDPX.factor_epoch(cache) == 1
    @test ws.homogeneous_solves == 1
    @test ws.variable_solves == 2

    # A changed Theta/tau/kappa is a new numeric epoch on the SAME static
    # pattern: symbolic stays 1, numeric becomes 2, homogeneous resets to 2.
    theta2 = bundle.system.cone.operator .* 1.1
    lin2 = SDPX.ProductConeLinearization{Float64}(
        theta2, copy(bundle.system.rhs.cone_corrector),
        bundle.system.cone.block_ranges,
    )
    system2 = SDPX.NewtonSystem(
        bundle.system.A, bundle.system.b, bundle.system.c, lin2,
        bundle.system.tau * 1.05, bundle.system.kappa * 1.05,
        bundle.system.rhs,
    )
    SDPX.factor_symmetric_core_epoch!(ws, system2, 2)
    @test SDPX.factor_epoch(cache) == 2
    @test SDPX.factor_diagnostics(cache).symbolic_count == 1
    @test SDPX.factor_diagnostics(cache).numeric_count == 2
    @test ws.homogeneous_solves == 2
    @test ws.homogeneous_epoch == ws.factor_epoch
    @test ws.factor_receipt.matrix_epoch == 2
    @test ws.original_scale > 1.0

    # Direction at the new epoch still passes the frozen five-equation gate.
    dir3, res3 = SDPX.solve_core_direction!(ws, system2)
    @test maximum(abs, res3.primal_affine) <= 4096 * eps(Float64)
    @test SDPX.factor_epoch(cache) == 2

    # A regularization delta change on the same pattern preserves symbolic and
    # only revokes/refreshes the numeric factor.
    symbolic_before = SDPX.factor_diagnostics(cache).symbolic_count
    delta0 = cache.regularization
    @test delta0 > 0
    SDPX.set_regularization!(cache, 2 * delta0)
    @test SDPX.factor_status(cache) === SDPX.Prepared
    SDPX.factorize!(cache, SDPX.symmetric_core_lower_sparse(ws.pattern), 2)
    @test SDPX.factor_status(cache) === SDPX.Fresh
    @test SDPX.factor_diagnostics(cache).symbolic_count == symbolic_before
    @test SDPX.factor_diagnostics(cache).regularization == 2 * delta0
    @test SDPX.factor_epoch(cache) == 3

    # The workspace guard is genuinely exercised after the direct refactor:
    # the cache advanced to epoch 3 without the workspace knowing, so a solve
    # before resynchronization must fail closed.
    @test ws.factor_epoch == 2
    @test_throws ArgumentError SDPX.solve_core_direction!(ws, system2)
    # Re-synchronize the workspace with the current system and solve the
    # homogeneous core at the new epoch so every later guard is exercised
    # against epoch 3, not a stale epoch 2 snapshot.
    SDPX.sync_core_factor_epoch!(ws; system=system2)
    @test ws.factor_epoch == 3
    @test ws.homogeneous_epoch == -1
    @test_throws ArgumentError SDPX.solve_core_direction!(ws, system2)
    SDPX.solve_core_homogeneous!(ws, system2)
    @test ws.homogeneous_epoch == 3
    @test ws.homogeneous_solves == 3
    @test ws.factor_receipt !== nothing
    @test ws.factor_receipt.matrix_epoch == 2
    @test ws.factor_receipt.regularization == 2 * delta0
    @test ws.factor_receipt.regularization_kind === :signed_diagonal
    dir_epoch3, res_epoch3 = SDPX.solve_core_direction!(ws, system2)
    @test maximum(abs, res_epoch3.primal_affine) <= 4096 * eps(Float64)
    @test SDPX.factor_epoch(cache) == 3

    # Theta changed within the same factor epoch is rejected.
    theta_bad = system2.cone.operator .* 1.3
    lin_bad = SDPX.ProductConeLinearization{Float64}(
        theta_bad, copy(system2.rhs.cone_corrector), system2.cone.block_ranges,
    )
    bad_system = SDPX.NewtonSystem(
        system2.A, system2.b, system2.c, lin_bad,
        system2.tau, system2.kappa, system2.rhs,
    )
    @test_throws ArgumentError SDPX.solve_core_direction!(ws, bad_system)

    # A changed block partition with the same dense operator values is a
    # static-identity change and must be rejected across epochs.  The LP
    # operator is diagonal, so splitting the single [1:2] block into
    # [1:1, 2:2] leaves the dense operator values unchanged but changes the
    # partition.
    partition_bad = SDPX.ProductConeLinearization{Float64}(
        system2.cone.operator, copy(system2.rhs.cone_corrector), [1:1, 2:2],
    )
    partition_system = SDPX.NewtonSystem(
        system2.A, system2.b, system2.c, partition_bad,
        system2.tau, system2.kappa, system2.rhs,
    )
    @test_throws ArgumentError SDPX.factor_symmetric_core_epoch!(
        ws, partition_system, 4,
    )

    # Static drift (A change) is rejected across epochs.
    A_bad = copy(system2.A)
    A_bad[1, 1] += 0.5
    bad_A_system = SDPX.NewtonSystem(
        A_bad, system2.b, system2.c, system2.cone,
        system2.tau, system2.kappa, system2.rhs,
    )
    @test_throws ArgumentError SDPX.factor_symmetric_core_epoch!(ws, bad_A_system, 4)

    # tau/kappa change without a new factor is rejected.
    bad_tk = SDPX.NewtonSystem(
        system2.A, system2.b, system2.c, system2.cone,
        system2.tau * 2.0, system2.kappa, system2.rhs,
    )
    @test_throws ArgumentError SDPX.solve_core_direction!(ws, bad_tk)

    # Factor receipt facts are truthful.
    @test ws.factor_receipt.scalar_type === Float64
    @test ws.factor_receipt.factor_status === :factored
    @test ws.factor_receipt.provider === :cholmod
    @test ws.factor_receipt.regularization_kind === :signed_diagonal
end
