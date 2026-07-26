using LinearAlgebra
using Random
using SDPX
using Test

function _legacy_arrow_rank_add!(destination, coupling, solved_coupling)
    local_count, global_count = size(coupling)
    @inbounds for p in 1:local_count
        for row in 1:global_count
            factor = coupling[p, row]
            iszero(factor) && continue
            for column in 1:global_count
                destination[row, column] +=
                    factor * solved_coupling[p, column]
            end
        end
    end
    return destination
end

function _legacy_arrow_rank_sub!(destination, coupling, solved_coupling)
    local_count, global_count = size(coupling)
    @inbounds for p in 1:local_count
        for row in 1:global_count
            factor = coupling[p, row]
            iszero(factor) && continue
            for column in 1:global_count
                destination[row, column] -=
                    factor * solved_coupling[p, column]
            end
        end
    end
    return destination
end

function _dense_workspace_problem(B::Matrix{Float64})
    variables = size(B, 1)
    coefficients = [zeros(Float64, variables, 1, 1)]
    return SDPX.ingest(
        zeros(variables),
        coefficients,
        [zeros(1, 1)],
        B,
        zeros(size(B, 2));
        sparse=false,
        verbosity=0,
    )
end

@testset "KKT regressions" begin
    @testset "column-major arrow rank updates" begin
        rng = MersenneTwister(90210)
        local_count = 3
        global_count = 320
        coupling = randn(rng, local_count, global_count)
        solved = randn(rng, local_count, global_count)
        coupling[2, 7:13] .= 0.0

        legacy = zeros(global_count, global_count)
        optimized = zeros(global_count, global_count)
        _legacy_arrow_rank_add!(legacy, coupling, solved)
        SDPX._arrow_rank_add!(optimized, coupling, solved)
        @test optimized == legacy

        initial = randn(rng, global_count, global_count)
        expected = copy(initial)
        _legacy_arrow_rank_sub!(expected, coupling, solved)
        actual = copy(initial)
        SDPX._arrow_rank_sub!(actual, coupling, solved)
        @test actual == expected
    end

    @testset "adaptive refinement restores last accepted direction" begin
        problem = _dense_workspace_problem(zeros(2, 0))
        workspace = SDPX.Workspace(problem; thread_count=1)
        workspace.S .= Matrix{Float64}(I, 2, 2)
        fill!(workspace.Sbuf, 0.0)
        workspace.Sbuf[1, 1] = 0.5
        workspace.Sbuf[2, 2] = 0.5
        initial_direction = [0.25, -0.5]
        copyto!(workspace.dx, initial_direction)
        rhs = [1.0, -2.0]
        initial_residual =
            norm(rhs - workspace.S * initial_direction, Inf)
        options = SDPX.SolverOptions{Float64}(
            verbosity=0,
            refine_policy=:adaptive,
            refine_max_steps=3,
            refine_tol=0.0,
        )

        steps, residual =
            SDPX.refine_direction!(workspace, problem, options, rhs)
        @test steps == 0
        @test residual == initial_residual
        @test workspace.dx == initial_direction
    end

    @testset "allocation-free equality KKT right-hand side" begin
        B = reshape([1.0, 2.0, -1.0], 3, 1)
        problem = _dense_workspace_problem(B)
        workspace = SDPX.Workspace(problem; thread_count=1)
        schur = [4.0 0.5 -0.2; 0.5 3.0 0.1; -0.2 0.1 2.0]
        copyto!(workspace.S, schur)
        options = SDPX.SolverOptions{Float64}(verbosity=0)
        factor = SDPX.factor_kkt!(workspace, problem, options)
        @test factor.ok
        @test !factor.q_pivoted

        primal_rhs = [0.4, -0.7, 1.2]
        equality_rhs = [0.3]
        dx = zeros(3)
        dy = zeros(1)
        SDPX.solve_kkt!(
            workspace,
            1,
            primal_rhs,
            equality_rhs,
            dx,
            dy,
        )
        reference = [
            schur -B
            transpose(B) zeros(1, 1)
        ] \ [primal_rhs; equality_rhs]
        @test dx ≈ reference[1:3] rtol=1e-13 atol=1e-13
        @test dy ≈ reference[4:4] rtol=1e-13 atol=1e-13

        allocated = @allocated SDPX.solve_kkt!(
            workspace,
            1,
            primal_rhs,
            equality_rhs,
            dx,
            dy,
        )
        # Julia 1.10 charges 32 bytes here that 1.12 does not -- a fixed
        # cost from the call itself, not from the solve. The regression this
        # guards against is an allocation that scales with the problem, so a
        # small constant allowance keeps the guard meaningful on every
        # supported version instead of passing on one and failing on another.
        @test allocated <= 64
    end

    @testset "pivoted equality solve reuses workspace scratch" begin
        column = [1.0, -0.5, 2.0]
        B = hcat(column, column)
        problem = _dense_workspace_problem(B)
        workspace = SDPX.Workspace(problem; thread_count=1)
        schur = [3.0 0.2 0.1; 0.2 2.5 -0.3; 0.1 -0.3 4.0]
        copyto!(workspace.S, schur)
        options = SDPX.SolverOptions{Float64}(verbosity=0)
        factor = SDPX.factor_kkt!(workspace, problem, options)
        @test factor.ok
        @test factor.q_pivoted

        primal_rhs = [0.2, 0.1, -0.4]
        equality_rhs = [0.3, 0.3]
        dx = zeros(3)
        dy = zeros(2)
        SDPX.solve_kkt!(
            workspace,
            2,
            primal_rhs,
            equality_rhs,
            dx,
            dy,
        )
        @test schur * dx - B * dy ≈ primal_rhs rtol=1e-12 atol=1e-12
        @test transpose(B) * dx ≈ equality_rhs rtol=1e-12 atol=1e-12
        @test all(isfinite, dx)
        @test all(isfinite, dy)

        allocated = @allocated SDPX.solve_kkt!(
            workspace,
            2,
            primal_rhs,
            equality_rhs,
            dx,
            dy,
        )
        # Julia 1.10 charges 32 bytes here that 1.12 does not -- a fixed
        # cost from the call itself, not from the solve. The regression this
        # guards against is an allocation that scales with the problem, so a
        # small constant allowance keeps the guard meaningful on every
        # supported version instead of passing on one and failing on another.
        @test allocated <= 64
    end

    @testset "Schur regularization escalates only on unfactorable S" begin
        # The escalation exists for the iterates near convergence where the
        # Schur complement loses positivity to rounding. Its trigger is a
        # failed `cholesky!`, so this pins down both halves of that contract:
        # a merely ill-conditioned S must factor untouched, and a singular or
        # indefinite one must be regularized back to a usable factorization
        # rather than propagating a failure.
        m, side, blocks = 12, 4, 2
        rng = MersenneTwister(3)
        coefficients = [zeros(m, side, side) for _ in 1:blocks]
        for l in 1:blocks, i in 1:m
            entry = randn(rng, side, side)
            coefficients[l][i, :, :] = entry + entry'
        end
        problem = SDPX.ingest(
            ones(m),
            coefficients,
            [Matrix{Float64}(1.0I, side, side) for _ in 1:blocks],
            zeros(m, 0),
            Float64[];
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(verbosity=0)
        workspace = SDPX.Workspace(problem)
        basis =
            qr(randn(MersenneTwister(9), m, m)).Q * Matrix{Float64}(1.0I, m, m)
        function factor_with_smallest_eigenvalue(smallest)
            eigenvalues = collect(range(1.0, 2.0, length=m))
            eigenvalues[1] = smallest
            schur = basis * Diagonal(eigenvalues) * basis'
            SDPX.copy_owned!(workspace.S, schur)
            return SDPX._factor_dense_kkt_native!(workspace, problem, options)
        end

        # Positive definite, however badly scaled: no regularization at all.
        for smallest in (1.0, 1e-8, 1e-14)
            outcome = factor_with_smallest_eigenvalue(smallest)
            @test outcome.ok
            @test outcome.reg_attempts == 0
        end

        # Singular and borderline-indefinite: whatever `cholesky!` decides, the
        # contract is that a usable factorization comes back. Whether the
        # escalation had to run for these is a property of the LAPACK build,
        # not of this solver -- an earlier version of this test asserted at
        # least one attempt here and passed on macOS/aarch64 while failing on
        # ubuntu/x86_64, where the borderline matrices factor successfully.
        for smallest in (0.0, -1e-14, -1e-8)
            outcome = factor_with_smallest_eigenvalue(smallest)
            @test outcome.ok
            @test 0 <= outcome.reg_attempts <= 6
        end

        # Clearly indefinite, but only just: a -1e-6 eigenvalue among
        # eigenvalues of order one is far above rounding, so every LAPACK
        # rejects it, and the shift needed to repair it is within the
        # escalation's reach. That combination is what makes this the case
        # proving the path is reachable, without depending on how a particular
        # build treats a borderline matrix.
        decisive = factor_with_smallest_eigenvalue(-1e-6)
        @test decisive.ok
        @test 1 <= decisive.reg_attempts <= 6

        # The escalation is scoped to rounding-level loss of positivity, not to
        # genuine indefiniteness, and it is worth recording where that ends. It
        # starts at sqrt(eps) and multiplies by ten at most six times, so the
        # largest shift it can apply is around 1.5e-2 relative. A matrix that
        # needs more than that is reported as a failure rather than silently
        # shifted into something else.
        hopeless = factor_with_smallest_eigenvalue(-1.0)
        @test !hopeless.ok
        @test hopeless.reg_attempts == 6
    end

end
