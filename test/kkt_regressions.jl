using LinearAlgebra
using MultiFloats: Float64x4
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

function _dense_duplicate_equality_fixture()
    variables = 4
    coefficients = [
        zeros(Float64, variables, 2, 2),
        zeros(Float64, variables, 2, 2),
    ]
    coefficients[1][1, :, :] .= [0.0 1.0; 1.0 0.0]
    coefficients[1][2, :, :] .= [1.0 0.0; 0.0 -1.0]
    coefficients[2][3, :, :] .= [0.0 1.0; 1.0 0.0]
    coefficients[2][4, :, :] .= [1.0 0.0; 0.0 -1.0]
    B = [1.0 0.2; -0.5 0.7; 0.3 -0.4; 0.8 1.1]
    problem = SDPX.ingest(
        zeros(variables),
        coefficients,
        [zeros(2, 2), zeros(2, 2)],
        hcat(B[:, 1], B[:, 1]),
        zeros(2);
        sparse=false,
        verbosity=0,
    )
    X = [
        [2.0 0.1; 0.1 1.5],
        [1.8 -0.2; -0.2 2.2],
    ]
    Y = [
        [1.4 0.2; 0.2 1.9],
        [2.1 -0.1; -0.1 1.6],
    ]
    return problem, X, Y
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

        lower_add = zeros(global_count, global_count)
        SDPX._arrow_rank_add_lower!(lower_add, coupling, solved)
        @test LowerTriangular(lower_add) == LowerTriangular(legacy)

        lower_sub = copy(initial)
        SDPX._arrow_rank_sub_lower!(lower_sub, coupling, solved)
        @test LowerTriangular(lower_sub) == LowerTriangular(expected)

        setprecision(BigFloat, 256) do
            big_coupling = BigFloat.(coupling[:, 1:16])
            big_solved = BigFloat.(solved[:, 1:16])
            big_full = SDPX.alloc_zeros(BigFloat, 16, 16)
            big_lower = SDPX.alloc_zeros(BigFloat, 16, 16)
            SDPX._arrow_rank_add!(
                big_full,
                big_coupling,
                big_solved,
            )
            SDPX._arrow_rank_add_lower!(
                big_lower,
                big_coupling,
                big_solved,
            )
            @test LowerTriangular(big_lower) ==
                  LowerTriangular(big_full)
            @test !(big_lower[1, 1] === big_lower[2, 1])
        end
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

    @testset "fixed refinement restores last accepted direction" begin
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
            refine_policy=:fixed,
            refine_steps=1,
            refine_tol=0.0,
        )

        steps, residual =
            SDPX.refine_direction!(workspace, problem, options, rhs)
        @test steps == 0
        @test residual == initial_residual
        @test workspace.dx == initial_direction
    end

    @testset "accepted SDP trial residuals include direction error" begin
        problem = _dense_workspace_problem(reshape([1.0], 1, 1))
        workspace = SDPX.Workspace(problem; thread_count=1)

        # Minimal counterexample from the numerical audit: the ideal carry
        # would report zero at a full step, while rho_p keeps the true equality
        # residual at one.
        workspace.p[1] = 1.0
        workspace.ρp[1] = 1.0
        workspace.d[1] = 0.0
        workspace.ρr[1] = 0.0
        primal_after, dual_after =
            SDPX._accepted_sdp_trial_residuals!(
                workspace, 0.0, 1.0, 1.0,
            )
        @test primal_after == 1.0
        @test workspace.p == [1.0]
        @test dual_after == 0.0

        # The dual affine residual has the opposite structured-residual sign:
        # d+ = (1-tY)d - tY*rho_r.
        workspace.p[1] = 0.0
        workspace.ρp[1] = 0.0
        workspace.d[1] = 1.0
        workspace.ρr[1] = 1.0
        primal_after, dual_after =
            SDPX._accepted_sdp_trial_residuals!(
                workspace, 0.0, 1.0, 1.0,
            )
        @test primal_after == 0.0
        @test workspace.d == [-1.0]
        @test dual_after == 1.0

        # Exact directions retain the historical affine carry formula.
        workspace.p[1] = 2.0
        workspace.ρp[1] = 0.0
        workspace.d[1] = -4.0
        workspace.ρr[1] = 0.0
        primal_after, dual_after =
            SDPX._accepted_sdp_trial_residuals!(
                workspace, 3.0, 0.25, 0.5,
            )
        @test primal_after == 2.25
        @test dual_after == 2.0
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

    @testset "block-diagonal sparse Schur with equality QR" begin
        variables = 4
        coefficients = [
            zeros(Float64, variables, 2, 2),
            zeros(Float64, variables, 2, 2),
        ]
        coefficients[1][1, :, :] .=
            [0.0 1.0; 1.0 0.0]
        coefficients[1][2, :, :] .=
            [1.0 0.0; 0.0 -1.0]
        coefficients[2][3, :, :] .=
            [0.0 1.0; 1.0 0.0]
        coefficients[2][4, :, :] .=
            [1.0 0.0; 0.0 -1.0]
        B = [
            1.0 0.2
            -0.5 0.7
            0.3 -0.4
            0.8 1.1
        ]
        C = [zeros(2, 2), zeros(2, 2)]
        c = zeros(variables)
        b = zeros(2)
        sparse_problem = SDPX.ingest(
            c,
            coefficients,
            C,
            B,
            b;
            sparse=true,
            verbosity=0,
        )
        dense_problem = SDPX.ingest(
            c,
            coefficients,
            C,
            B,
            b;
            sparse=false,
            verbosity=0,
        )
        sparse_workspace = SDPX.Workspace(
            sparse_problem;
            equality_solver=:qr,
            thread_count=1,
        )
        dense_workspace = SDPX.Workspace(
            dense_problem;
            equality_solver=:qr,
            thread_count=1,
        )
        @test sparse_workspace.arrow !== nothing
        @test isempty(sparse_workspace.arrow.global_ids)
        @test isempty(sparse_workspace.S)

        X = [
            [2.0 0.1; 0.1 1.5],
            [1.8 -0.2; -0.2 2.2],
        ]
        Y = [
            [1.4 0.2; 0.2 1.9],
            [2.1 -0.1; -0.1 1.6],
        ]
        @test SDPX.factor_blocks!(
            sparse_workspace,
            X,
            Y,
        )
        @test SDPX.factor_blocks!(
            dense_workspace,
            X,
            Y,
        )
        SDPX.schur_build!(
            sparse_workspace,
            sparse_problem,
            sparse_problem.cons,
            X,
            Y,
        )
        SDPX.schur_build!(
            dense_workspace,
            dense_problem,
            dense_problem.cons,
            X,
            Y,
        )
        options = SDPX.SolverOptions{Float64}(
            verbosity=0,
            equality_solver=:qr,
        )
        sparse_factor = SDPX.factor_kkt!(
            sparse_workspace,
            sparse_problem,
            options,
        )
        dense_factor = SDPX.factor_kkt!(
            dense_workspace,
            dense_problem,
            options,
        )
        @test sparse_factor.ok
        @test dense_factor.ok
        @test sparse_factor.equality_solver ==
              :rank_revealing_qr

        primal_rhs = [0.4, -0.7, 1.2, -0.3]
        equality_rhs = [0.3, -0.2]
        sparse_dx, sparse_dy = zeros(4), zeros(2)
        dense_dx, dense_dy = zeros(4), zeros(2)
        SDPX.solve_kkt!(
            sparse_workspace,
            2,
            primal_rhs,
            equality_rhs,
            sparse_dx,
            sparse_dy,
        )
        SDPX.solve_kkt!(
            dense_workspace,
            2,
            primal_rhs,
            equality_rhs,
            dense_dx,
            dense_dy,
        )
        @test sparse_dx ≈ dense_dx rtol=1e-12 atol=1e-12
        @test sparse_dy ≈ dense_dy rtol=1e-12 atol=1e-12
        @test transpose(B) * sparse_dx ≈
              equality_rhs rtol=1e-12 atol=1e-12

        duplicate_problem = SDPX.ingest(
            c,
            coefficients,
            C,
            hcat(B[:, 1], B[:, 1]),
            b;
            sparse=true,
            verbosity=0,
        )
        duplicate_workspace = SDPX.Workspace(
            duplicate_problem;
            equality_solver=:auto,
            thread_count=1,
        )
        @test SDPX.factor_blocks!(
            duplicate_workspace,
            X,
            Y,
        )
        SDPX.schur_build!(
            duplicate_workspace,
            duplicate_problem,
            duplicate_problem.cons,
            X,
            Y,
        )
        duplicate_options = SDPX.SolverOptions{Float64}(
            verbosity=0,
            equality_solver=:auto,
        )
        duplicate_factor = SDPX.factor_kkt!(
            duplicate_workspace,
            duplicate_problem,
            duplicate_options,
        )
        @test duplicate_factor.ok
        @test duplicate_factor.q_rank_deficient
        @test duplicate_factor.equality_solver ==
              :rank_revealing_qr
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
        @test factor.equality_solver == :rank_revealing_qr
        @test workspace.Qchol isa SDPX.EqualityQRFactor{Float64}
        equality_diagnostics =
            SDPX._equality_factor_diagnostics(workspace, 2)
        @test equality_diagnostics.method == :rank_revealing_qr
        @test equality_diagnostics.rank == 1
        @test equality_diagnostics.rank_deficient
        @test equality_diagnostics.gram_kernel == :blas_syrk

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

    @testset "Standard dense equality QR fallback records provenance" begin
        problem, X, Y = _dense_duplicate_equality_fixture()
        plan = SDPX.build_execution_plan(
            SDPX.AutoPlanner(),
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
                equality_solver=:auto,
                threads=1,
                verbosity=0,
            ),
        )
        @test plan.la_config.selected === :standard
        @test plan.la_config.fallback_chain === (:rank_revealing_qr,)

        workspace = SDPX.Workspace(
            problem;
            thread_count=1,
            execution_plan=plan,
        )
        @test workspace.la_backend isa
              SDPX.Experimental.StandardLABackend
        @test workspace.la_fallback_chain === (:rank_revealing_qr,)

        @test SDPX.factor_blocks!(workspace, X, Y)
        SDPX.schur_build!(workspace, problem, problem.cons, X, Y)

        options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            verbosity=0,
            equality_solver=:auto,
        )
        factor = SDPX.factor_kkt!(workspace, problem, options)
        @test factor.ok
        @test factor.equality_solver === :rank_revealing_qr
        @test workspace.Qchol isa SDPX.EqualityQRFactor{Float64}
        @test workspace.la_fallback_reason ===
              :la_equality_factor_failed
    end

    @testset "Standard dense normal equations never report QR fallback" begin
        problem, X, Y = _dense_duplicate_equality_fixture()
        plan = SDPX.build_execution_plan(
            SDPX.AutoPlanner(),
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
                equality_solver=:normal_equations,
                threads=1,
                verbosity=0,
            ),
        )
        @test plan.la_config.selected === :standard
        @test plan.la_config.fallback_chain === ()

        workspace = SDPX.Workspace(
            problem;
            thread_count=1,
            execution_plan=plan,
        )
        @test SDPX.factor_blocks!(workspace, X, Y)
        SDPX.schur_build!(workspace, problem, problem.cons, X, Y)

        options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            verbosity=0,
            equality_solver=:normal_equations,
        )
        factor = SDPX.factor_kkt!(workspace, problem, options)
        @test factor.ok
        @test factor.equality_solver === :normal_equations
        @test !(workspace.Qchol isa SDPX.EqualityQRFactor{Float64})
        @test workspace.la_fallback_reason === :none
    end

    @testset "Legacy dense equality QR fallback stays on legacy provider" begin
        problem, X, Y = _dense_duplicate_equality_fixture()
        plan = SDPX.build_execution_plan(
            SDPX.AutoPlanner(),
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
                equality_solver=:auto,
                linear_algebra_backend=:legacy,
                threads=1,
                verbosity=0,
            ),
        )
        @test plan.la_config.selected === :legacy
        @test plan.la_config.fallback_chain === (:rank_revealing_qr,)

        workspace = SDPX.Workspace(
            problem;
            thread_count=1,
            execution_plan=plan,
        )
        @test workspace.la_backend isa
              SDPX.Experimental.LegacyLABackend
        @test workspace.la_fallback_chain === (:rank_revealing_qr,)
        @test SDPX.factor_blocks!(workspace, X, Y)
        SDPX.schur_build!(workspace, problem, problem.cons, X, Y)

        options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            verbosity=0,
            equality_solver=:auto,
        )
        factor = SDPX.factor_kkt!(workspace, problem, options)
        @test factor.ok
        @test factor.equality_solver === :rank_revealing_qr
        @test workspace.Qchol isa SDPX.EqualityQRFactor{Float64}
        @test SDPX.la_backend_name(workspace.la_backend) === :legacy
        @test SDPX.la_backend_provider(workspace.la_backend) ===
              :sdpx_legacy_la
        @test workspace.la_fallback_reason ===
              :la_equality_factor_failed
    end

    @testset "rank-revealing QR avoids squared equality conditioning" begin
        rng = MersenneTwister(771)
        rows = 24
        base = randn(rng, rows, 3)
        nearly_dependent = base[:, 1] + 1e-12 * randn(rng, rows)
        B = hcat(base, nearly_dependent)
        problem = _dense_workspace_problem(B)
        workspace = SDPX.Workspace(
            problem;
            equality_solver=:qr,
            thread_count=1,
        )
        schur = Matrix(Diagonal(range(1.0, 2.0; length=rows)))
        copyto!(workspace.S, schur)
        options = SDPX.SolverOptions{Float64}(
            verbosity=0,
            equality_solver=:qr,
        )
        factor = SDPX.factor_kkt!(workspace, problem, options)
        @test factor.ok
        @test factor.equality_solver == :rank_revealing_qr
        @test workspace.Qchol isa SDPX.EqualityQRFactor{Float64}
        @test workspace.equality_gram_kernel ==
              :not_formed_qr

        primal_rhs = randn(rng, rows)
        seed_direction = randn(rng, rows)
        equality_rhs = transpose(B) * seed_direction
        dx = zeros(rows)
        dy = zeros(size(B, 2))
        SDPX.solve_kkt!(
            workspace,
            size(B, 2),
            primal_rhs,
            equality_rhs,
            dx,
            dy,
        )
        @test schur * dx - B * dy ≈ primal_rhs rtol=1e-8 atol=1e-8
        @test transpose(B) * dx ≈ equality_rhs rtol=1e-8 atol=1e-8
        @test all(isfinite, dx)
        @test all(isfinite, dy)
    end

    @testset "rank-revealing equality QR supports extended arithmetic" begin
        rng = MersenneTwister(772)
        for T in (Float64x4, BigFloat)
            setprecision(BigFloat, 256) do
                B = T.(randn(rng, 12, 5))
                options = SDPX.SolverOptions{T}(
                    verbosity=0,
                    equality_solver=:qr,
                )
                factor = SDPX._factor_equality_qr(
                    SDPX.Experimental.StandardLABackend(
                        SDPX._la_arithmetic_symbol(T),
                    ),
                    B,
                    options,
                )
                rhs = T.(randn(rng, 5))
                direction = SDPX.alloc_zeros(T, 5)
                scratch = SDPX.alloc_zeros(T, 5)
                SDPX._solve_Q!(direction, factor, rhs, scratch)
                relative_residual =
                    norm(transpose(B) * (B * direction) - rhs) /
                    norm(rhs)
                @test factor.rank == 5
                @test relative_residual <= T(1_000) * eps(T)
                if T === BigFloat
                    @test length(unique(objectid.(direction))) ==
                          length(direction)
                    @test length(unique(objectid.(scratch))) ==
                          length(scratch)
                end
            end
        end
    end

    @testset "equality Gram crossover rejects tiny panels" begin
        panel = zeros(Float64x4, 48, 18)
        options = SDPX.SolverOptions{Float64x4}(
            verbosity=0,
            extended_precision_blas=:auto,
            threads=4,
        )
        decision =
            SDPX._equality_gram_crossover(panel, options, 4)
        @test !decision.enabled
        @test decision.reason in (
            :equality_gram_too_small,
            :problem_too_small,
            :memory_budget,
        )
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

    @testset "equality Cholesky rank uses explicit precision" begin
        rng = MersenneTwister(991)
        rows = 24
        base = randn(rng, rows, 3)
        tau = SDPX._equality_cholesky_rank_tau(
            base,
            base,
            SDPX.SolverOptions{Float64}(verbosity=0),
        )
        @test tau ≈ max(24.0 * eps(Float64), 1e-10)

        below = Matrix(Diagonal([1.0, 1e-12]))
        above = Matrix(Diagonal([1.0, 1e-6]))
        options = SDPX.SolverOptions{Float64}(verbosity=0)
        B_below = below
        B_above = above
        Q_below = transpose(B_below) * B_below
        Q_above = transpose(B_above) * B_above
        L_below = cholesky(Symmetric(Q_below, :L); check=false)
        L_above = cholesky(Symmetric(Q_above, :L); check=false)

        # The threshold is relative: both the near-dependent and the healthy
        # Gram factors may succeed, but only the healthy one clears the
        # explicit-precision rank policy. Scale invariance is covered below
        # by factoring the same dependency at 1e3 and 1e-3 scales.
        @test !SDPX._cholesky_has_numerical_rank(
            L_below,
            B_below,
            options,
        )
        @test SDPX._cholesky_has_numerical_rank(
            L_above,
            B_above,
            options,
        )
        @test !SDPX._cholesky_has_numerical_rank(
            L_below,
            B_below,
            SDPX.SolverOptions{Float64}(
                verbosity=0,
                ϵ_gap=1e-4,
                ϵ_primal=1e-4,
                ϵ_dual=1e-4,
            ),
        )

        for scale in (1e-3, 1e3)
            scaled_below = scale .* B_below
            scaled_q = transpose(scaled_below) * scaled_below
            scaled_l = cholesky(
                Symmetric(scaled_q, :L);
                check=false,
            )
            @test !SDPX._cholesky_has_numerical_rank(
                scaled_l,
                scaled_below,
                options,
            )
        end

        setprecision(BigFloat, 256) do
            big_below = BigFloat.(B_below)
            big_q = transpose(big_below) * big_below
            big_l = cholesky(Symmetric(big_q, :L); check=false)
            big_options = SDPX.SolverOptions{BigFloat}(
                verbosity=0,
                ϵ_gap=big"1e-12",
                ϵ_primal=big"1e-12",
                ϵ_dual=big"1e-12",
            )
            # Near-dependent at Float64 rounding noise is far above BigFloat
            # eps, so a genuinely higher-precision factor owns full rank.
            @test SDPX._cholesky_has_numerical_rank(
                big_l,
                big_below,
                big_options,
            )
            # A BigFloat factor of the exact duplicate has an exactly zero
            # diagonal and is therefore rejected even at tight tolerances.
            duplicated = hcat(big_below, big_below[:, 1])
            duplicated_q = transpose(duplicated) * duplicated
            duplicated_l = cholesky(
                Symmetric(duplicated_q, :L);
                check=false,
            )
            @test !SDPX._cholesky_has_numerical_rank(
                duplicated_l,
                duplicated,
                big_options,
            )
        end
    end

    @testset "equality lower-triangle copies never read upper storage" begin
        source = Matrix{Float64}([
            4.0 1.0
            1.0 3.0
        ])
        destination = fill(NaN, 2, 2)
        SDPX._copy_lower_triangle!(destination, source)
        @test destination[1, 1] == 4.0
        @test destination[2, 1] == 1.0
        @test isnan(destination[1, 2])
        @test destination[2, 2] == 3.0

        poisoned = copy(source)
        poisoned[1, 2] = NaN
        destination = fill(NaN, 2, 2)
        SDPX._copy_lower_triangle!(destination, poisoned)
        @test destination[2, 1] == 1.0
        @test isnan(destination[1, 2])
        @test destination[2, 2] == 3.0

        # BigFloat lower copies also deep-copy into independent storage.
        setprecision(BigFloat, 256) do
            big_source = BigFloat[
                4.0 1.0
                1.0 3.0
            ]
            big_source[1, 2] = BigFloat(NaN)
            big_destination = SDPX.alloc_zeros(BigFloat, 2, 2)
            SDPX._copy_lower_triangle!(big_destination, big_source)
            @test big_destination[2, 1] == big_source[2, 1]
            @test iszero(big_destination[1, 2])
            @test big_destination[2, 2] == big_source[2, 2]
            @test big_destination[1, 1] !== big_source[1, 1]
        end

        # Standard Float64 BLAS SYRK owns only the lower triangle. With
        # beta=0 the upper poison is preserved rather than mirrored from the
        # computed lower triangle, and the dense equality consumer still
        # factors successfully from lower-authoritative storage.
        backend = SDPX.Experimental.StandardLABackend(:float64)
        panel = [1.0 2.0; 3.0 4.0]
        gram = fill(NaN, 2, 2)
        SDPX.la_syrk!(backend, gram, panel, 1.0, 0.0)
        @test gram[1, 1] ≈ 10.0
        @test gram[2, 1] ≈ 14.0
        @test gram[2, 2] ≈ 20.0
        @test isnan(gram[1, 2])
    end

    @testset "explicit normal equations keep pivoted Cholesky" begin
        column = [1.0, -0.5, 2.0]
        B = hcat(column, column)
        problem = _dense_workspace_problem(B)
        plan = SDPX.build_execution_plan(
            SDPX.AutoPlanner(),
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
                equality_solver=:normal_equations,
                threads=1,
                verbosity=0,
            ),
        )
        workspace = SDPX.Workspace(
            problem;
            thread_count=1,
            execution_plan=plan,
        )
        copyto!(
            workspace.S,
            [3.0 0.2 0.1; 0.2 2.5 -0.3; 0.1 -0.3 4.0],
        )
        workspace.Q[1, 2] = workspace.Q[2, 1] = NaN
        workspace.Q[2, 2] = NaN
        options = SDPX.SolverOptions{Float64}(
            verbosity=0,
            equality_solver=:normal_equations,
        )
        factor = SDPX.factor_kkt!(workspace, problem, options)
        @test factor.ok
        @test factor.equality_solver === :normal_equations
        @test factor.q_pivoted
        @test workspace.Qchol isa
              LinearAlgebra.CholeskyPivoted{Float64}
        @test all(isfinite, workspace.Qbuf)
    end

end
