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
    @test session.backward_error <= session.backward_target
    @test session.unregularized_residual_norm <=
          session.backward_target * max(
              SDPX._expanded_operator_scale(session.unregularized) *
              norm(solution, Inf) + norm(rhs, Inf), 1.0,
          )
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

@testset "expanded expected inertia is derived and gated by applicability" begin
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
    @test SDPX.factor_expanded_kkt!(session, system; max_regularization_attempts=3)
    @test session.expected_inertia == SDPX.KKTInertia(1, 2, 0)
    # The negative cone operator refutes the sign-definite-block premise, so
    # the companion-inertia expectation is not applicable: the observed
    # mismatch is recorded diagnostically and the exact factor proceeds (the
    # backward-error authority still gates any direction).
    @test session.status == SDPX.EXPANDED_KKT_FACTORED
    @test session.inertia_factor.inertia != session.expected_inertia
    @test session.inertia_applicability.status == SDPX.INERTIA_NOT_APPLICABLE
    @test session.inertia_applicability.reason ===
          :y_block_not_positive_definite
    @test any(
        attempt.reason == SDPX.EXPANDED_ATTEMPT_INERTIA_NOT_APPLICABLE
        for attempt in session.attempts
    )
    @test session.factor.success
end

@testset "expanded structured regularization ladder" begin
    T = Float64
    A = zeros(T, 1, 1)
    b = zeros(T, 1)
    c = zeros(T, 1)
    cone = SDPX.assemble_cone_linearization(
        T, 1,
        [SDPX.LocalConeLinearization(1:1, reshape(T[1], 1, 1), zeros(T, 1))],
    )
    rhs = SDPX.HSDNewtonRHS(
        zeros(T, 1), zeros(T, 1), zero(T), zeros(T, 1), zero(T),
    )
    system = SDPX.NewtonSystem(A, b, c, cone, one(T), one(T), rhs)
    session = SDPX.ExpandedKKTSession(T, 1, 1)
    @test SDPX.factor_expanded_kkt!(session, system)
    @test length(session.attempts) == 2
    @test session.attempts[1].stage == SDPX.EXPANDED_REGULARIZATION_NONE
    @test session.attempts[1].reason == SDPX.EXPANDED_ATTEMPT_TINY_PIVOT
    @test session.attempts[2].stage == SDPX.EXPANDED_REGULARIZATION_STATIC
    @test session.attempts[2].reason == SDPX.EXPANDED_ATTEMPT_ACCEPTED
    @test session.attempts[2].observed_inertia == session.expected_inertia
    @test session.attempts[2].minimum_pivot > session.attempts[2].pivot_threshold

    # A negative cone operator refutes the sign-definite-block premise of the
    # companion-inertia expectation: the mismatch is recorded diagnostically
    # and the exact factor proceeds instead of exhausting the ladder on a
    # wrong-inertia gate that is not applicable.
    wrong_cone = SDPX.assemble_cone_linearization(
        T, 1,
        [SDPX.LocalConeLinearization(1:1, reshape(T[-1], 1, 1), zeros(T, 1))],
    )
    wrong = SDPX.NewtonSystem(A, b, c, wrong_cone, one(T), one(T), rhs)
    accepted = SDPX.ExpandedKKTSession(T, 1, 1)
    @test SDPX.factor_expanded_kkt!(
        accepted, wrong; max_regularization_attempts=5,
    )
    @test accepted.status == SDPX.EXPANDED_KKT_FACTORED
    @test accepted.inertia_applicability.status == SDPX.INERTIA_NOT_APPLICABLE
    @test any(
        attempt.reason == SDPX.EXPANDED_ATTEMPT_INERTIA_NOT_APPLICABLE
        for attempt in accepted.attempts
    )
    @test any(
        attempt.reason == SDPX.EXPANDED_ATTEMPT_ACCEPTED
        for attempt in accepted.attempts
    )
    @test accepted.factor_receipt !== nothing
    @test_throws ArgumentError SDPX.factor_expanded_kkt!(
        session, system; max_regularization_attempts=-1,
    )
end

@testset "expanded unregularized refinement contract" begin
    system, _ = expanded_fixture(Float64)
    session = SDPX.ExpandedKKTSession(Float64, 2, 3)
    @test SDPX.factor_expanded_kkt!(session, system)
    rhs = zeros(Float64, session.dimension)
    SDPX.expanded_rhs!(rhs, system)

    # Deliberately replace the accepted factor by a strongly regularized one.
    # Its finite solve is still rejected when the original equations do not
    # satisfy the same-precision backward-error contract.
    SDPX._assemble_regularized!(session, 1.0)
    @test SDPX.factorize_pivoted_lu!(
        session.factor, session.regularized; threshold=eps(Float64),
    )
    session.status = SDPX.EXPANDED_KKT_FACTORED
    solution = similar(rhs)
    @test SDPX.solve_expanded!(solution, session, rhs)

    # With recovery disabled, the deliberately poor factor remains a typed,
    # fail-closed refinement failure above the unchanged strict target.
    @test !SDPX.refine_expanded!(
        solution, session, rhs;
        max_refinements=0, max_dynamic_attempts=0,
    )
    @test session.status == SDPX.EXPANDED_KKT_REFINEMENT_STAGNATED
    @test session.backward_error > session.backward_target
    @test session.unregularized_residual_norm > 0

    # Reopen the same deliberately poor starting factor.  The operator
    # rewrite again revokes the receipt; rebuild it through the owned factor
    # seam.  The default policy must then resume on a dynamic signed factor
    # and re-refine from the original RHS rather than failing immediately
    # after the static factor stagnates.
    SDPX._assemble_regularized!(session, 1.0)
    @test SDPX.factorize_pivoted_lu!(
        session.factor, session.regularized; threshold=eps(Float64),
    )
    @test !SDPX._expanded_factor_receipt_current(session)
    @test !SDPX.solve_expanded!(solution, session, rhs)
    @test SDPX._factor_expanded_exact!(session, eps(Float64))
    @test SDPX._build_expanded_factor_receipt!(session) !== nothing
    session.status = SDPX.EXPANDED_KKT_FACTORED
    @test SDPX.solve_expanded!(solution, session, rhs)
    @test SDPX.refine_expanded!(
        solution, session, rhs;
        max_refinements=2, max_dynamic_attempts=3,
    )
    @test session.refinement_recovery_attempts >= 1
    @test any(
        attempt.stage == SDPX.EXPANDED_REGULARIZATION_DYNAMIC &&
        attempt.reason == SDPX.EXPANDED_ATTEMPT_ACCEPTED
        for attempt in session.attempts
    )
    @test session.backward_error <= session.backward_target
    @test session.status == SDPX.EXPANDED_KKT_UNREGULARIZED_CERTIFIED
    @test any(
        isfinite(step.reduction_ratio) && step.reduction_ratio < 1
        for step in session.refinement_trajectory
    )
    @test length(unique(
        step.factor_attempt for step in session.refinement_trajectory
    )) >= 2

    @test_throws ArgumentError SDPX.refine_expanded!(
        solution, session, rhs; max_refinements=-1,
    )
    @test_throws ArgumentError SDPX.refine_expanded!(
        solution, session, rhs; max_dynamic_attempts=-1,
    )
end

@testset "expanded arithmetic floor is diagnostic, not authority" begin
    system, _ = expanded_fixture(Float64)
    session = SDPX.ExpandedKKTSession(Float64, 2, 3)
    @test SDPX.factor_expanded_kkt!(session, system)
    rhs = zeros(Float64, session.dimension)
    SDPX.expanded_rhs!(rhs, system)
    solution = zeros(Float64, session.dimension)

    # Force only the condition diagnostic to its conservative branch.  The
    # direction still fails because the original-equation backward error is
    # above 256eps; the floor cannot relax that contract.
    session.factor.minimum_pivot = eps(Float64) *
                                   SDPX._expanded_operator_scale(
                                       session.unregularized,
                                   )
    session.status = SDPX.EXPANDED_KKT_FACTORED
    @test !SDPX.refine_expanded!(
        solution, session, rhs;
        max_refinements=0, max_dynamic_attempts=0,
    )
    @test session.at_arithmetic_floor
    @test session.status == SDPX.EXPANDED_KKT_REFINEMENT_AT_FLOOR
    @test session.attainable_floor >= session.backward_error
    @test session.backward_error > session.backward_target
end

function expanded_hot_allocation_counts()
    system, _ = expanded_fixture(Float64)
    session = SDPX.ExpandedKKTSession(Float64, 2, 3)
    rhs = zeros(Float64, session.dimension)
    solution = similar(rhs)
    SDPX.expanded_rhs!(rhs, system)
    SDPX.factor_expanded_kkt!(session, system)
    SDPX.solve_expanded!(solution, session, rhs)
    SDPX.refine_expanded!(solution, session, rhs)
    GC.gc()
    factor_bytes = @allocated SDPX.factor_expanded_kkt!(session, system)
    SDPX.expanded_rhs!(rhs, system)
    session.status = SDPX.EXPANDED_KKT_FACTORED
    solve_bytes = @allocated SDPX.solve_expanded!(solution, session, rhs)
    refine_bytes = @allocated SDPX.refine_expanded!(solution, session, rhs)
    return factor_bytes, solve_bytes, refine_bytes
end

@testset "expanded Float64 hot route allocation contract" begin
    @test expanded_hot_allocation_counts() == (0, 0, 0)
end

@testset "expanded BigFloat provider dispatch" begin
    setprecision(BigFloat, 192) do
        system, expected = expanded_fixture(BigFloat)
        descriptor = SDPX.la_provider_descriptor(BigFloat)
        if descriptor.available &&
           descriptor.provider == :bigfloat_linear_algebra
            session = SDPX.ExpandedKKTSession(BigFloat, 2, 3)
            @test session.la_backend isa SDPX.BFLALABackend
            @test SDPX.factor_expanded_kkt!(session, system)
            @test session.provider_inertia_factor !== nothing
            @test session.provider_exact_factor !== nothing
            rhs = SDPX.alloc_zeros(BigFloat, session.dimension)
            SDPX.expanded_rhs!(rhs, system)
            solution = SDPX.alloc_zeros(BigFloat, session.dimension)
            @test SDPX.solve_expanded!(solution, session, rhs)
            provider_residual = SDPX.alloc_zeros(BigFloat, session.dimension)
            SDPX.la_residual!(
                session.la_backend, :N, session.unregularized,
                solution, rhs, provider_residual,
            )
            @test all(isfinite, provider_residual)
            @test isfinite(SDPX.la_normwise_backward_error(
                session.la_backend, :N, session.unregularized,
                solution, rhs, provider_residual,
            ))
            @test SDPX.refine_expanded!(solution, session, rhs)
            direction = SDPX.recover_expanded_direction(system, solution)
            residual = SDPX.NewtonResidual(system)
            SDPX.newton_residual!(residual, system, direction)
            @test SDPX.max_newton_residual(residual) < big"1e-45"
            @test direction.dx ≈ expected.dx rtol=big"1e-45" atol=big"1e-45"
        else
            # Binding provider policy: no private arbitrary-precision LDLT/LU.
            # An environment without the BFLA extension is a visible provider
            # gap and must fail before constructing a factorization session.
            exception = try
                SDPX.ExpandedKKTSession(BigFloat, 2, 3)
                nothing
            catch error
                error
            end
            @test exception isa ArgumentError
            @test occursin("pivoted_symmetric_ldlt", sprint(showerror, exception))
        end
    end
end
