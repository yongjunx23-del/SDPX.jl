using LinearAlgebra
using SDPX
using SparseArrays
using Test

function _augmented_test_problem(::Type{T}; equalities::Int=2) where {T}
    variables = 3
    coefficients = zeros(T, variables, 3, 3)
    @inbounds for index in 1:variables
        coefficients[index, index, index] = one(T)
    end
    B = equalities == 0 ? Matrix{T}(undef, variables, 0) :
        T[1 0; 0 1; 1 -1][:, 1:equalities]
    return SDPX.ingest(
        ones(T, variables),
        [coefficients],
        [zeros(T, 3, 3)],
        B,
        equalities == 0 ? T[] : ones(T, equalities);
        sparse=false,
        verbosity=0,
    )
end

@testset "dense augmented KKT algebra and planning" begin
    @testset "exact signs, direction equivalence, and reduced residual" begin
        S = [4.0 0.5 -0.25; 0.5 3.0 0.2; -0.25 0.2 2.5]
        B = [1.0 0.0; 0.25 1.0; -0.5 0.75]
        r = [0.4, -0.7, 1.2]
        p = [0.3, -0.2]
        augmented = SDPX.DenseAugmentedKKTWorkspace(Float64, 3, 2)
        SDPX.assemble_dense_augmented_kkt!(augmented, S, B)

        K = [S -B; -transpose(B) zeros(2, 2)]
        @test Symmetric(augmented.matrix, :L) == Symmetric(K)
        augmented_solution = K \ [r; -p]

        # Independent mature normal-equation elimination.
        normal_factor = cholesky(Symmetric(S))
        SinvB = normal_factor \ B
        dx0 = normal_factor \ r
        dy = (transpose(B) * SinvB) \ (p - transpose(B) * dx0)
        dx = normal_factor \ (r + B * dy)
        @test augmented_solution[1:3] ≈ dx rtol=1e-13 atol=1e-13
        @test augmented_solution[4:5] ≈ dy rtol=1e-13 atol=1e-13

        residual = SDPX.reduced_augmented_kkt_residual!(
            augmented,
            r,
            p,
            view(augmented_solution, 1:3),
            view(augmented_solution, 4:5),
        )
        @test residual <= 100eps(Float64)
        @test norm(augmented.residual, Inf) <= 100eps(Float64)

        # A provider may report a successful singular factor instead of
        # returning `nothing`. Preserve that rejected inertia after the
        # numerical handle is cleared so diagnostics remain auditable.
        problem = _augmented_test_problem(Float64; equalities=2)
        workspace = SDPX.Workspace(problem)
        workspace.augmented = augmented
        augmented.factor = nothing
        augmented.inertia = (positive=3, negative=1, zero=1)
        augmented.rank_deficient = true
        augmented.factor_diagnostics = (success=true, zero_pivot=true)
        rejected = SDPX._equality_factor_diagnostics(workspace, 2)
        @test rejected.available
        @test !rejected.factor_available
        @test rejected.rank == 1
        @test rejected.rank_deficient
        @test rejected.inertia.zero == 1
        @test rejected.factor_diagnostics.zero_pivot
    end

    @testset "recovered cone Newton equations" begin
        problem = _augmented_test_problem(Float64; equalities=2)
        P = [0.1 0.02 0.0; 0.02 -0.05 0.01; 0.0 0.01 0.03]
        R = [0.3 0.01 0.0; 0.01 0.25 -0.02; 0.0 -0.02 0.2]
        X = [2.0 0.1 0.0; 0.1 1.5 0.05; 0.0 0.05 1.8]
        Y = [1.4 0.05 0.0; 0.05 1.7 -0.03; 0.0 -0.03 1.2]
        dx = [0.2, -0.1, 0.05]
        dy = [-0.3, 0.15]
        dX = zeros(3, 3)
        SDPX.buildP!(dX, problem.cons, 1, dx)
        dX .+= P
        raw_dY = X \ (R - dX * Y)
        expected_dY = (raw_dY + transpose(raw_dY)) / 2
        dY = copy(expected_dY)

        primal_block = zeros(3, 3)
        SDPX.buildP!(primal_block, problem.cons, 1, dx)
        @test norm(P + primal_block - dX, Inf) <= 20eps(Float64)
        @test norm(dY - expected_dY, Inf) <= 20eps(Float64)

        dual_target = zeros(3)
        SDPX.accumulate_v!(dual_target, problem.cons, 1, dY, one(Float64))
        dual_target .+= problem.B * dy
        equality_target = transpose(problem.B) * dx

        dual_residual = copy(dual_target)
        SDPX.accumulate_v!(dual_residual, problem.cons, 1, dY, -one(Float64))
        dual_residual .-= problem.B * dy
        equality_residual = equality_target - transpose(problem.B) * dx
        @test norm(dual_residual, Inf) <= 20eps(Float64)
        @test norm(equality_residual, Inf) <= 20eps(Float64)

        # Independently derive the same reduced RHS from the four current
        # Newton blocks, then solve the symmetric augmented system and recover
        # every implementation-level direction equation.
        trial_dx = [0.13, -0.08, 0.04]
        trial_dy = [-0.11, 0.07]
        trial_dX = zeros(3, 3)
        SDPX.buildP!(trial_dX, problem.cons, 1, trial_dx)
        trial_dX .+= P
        trial_raw_dY = X \ (R - trial_dX * Y)
        trial_dY = (trial_raw_dY + transpose(trial_raw_dY)) / 2
        trial_d = zeros(3)
        SDPX.accumulate_v!(
            trial_d,
            problem.cons,
            1,
            trial_dY,
            one(Float64),
        )
        trial_d .+= problem.B * trial_dy
        trial_p = transpose(problem.B) * trial_dx

        Z = X \ (P * Y - R)
        v = zeros(3)
        SDPX.accumulate_v!(v, problem.cons, 1, Z, one(Float64))
        trial_r = -(trial_d + v)
        schur = zeros(3, 3)
        @inbounds for column in 1:3
            Acolumn = reshape(view(problem.cons.Av[1], :, column), 3, 3)
            transformed = X \ (Acolumn * Y)
            for row in 1:3
                Arow = reshape(view(problem.cons.Av[1], :, row), 3, 3)
                schur[row, column] = dot(Arow, transformed)
            end
        end

        augmented = SDPX.DenseAugmentedKKTWorkspace(Float64, 3, 2)
        SDPX.assemble_dense_augmented_kkt!(augmented, schur, problem.B)
        solution = Symmetric(augmented.matrix, :L) \ [trial_r; -trial_p]
        recovered_dx = solution[1:3]
        recovered_dy = solution[4:5]
        @test recovered_dx ≈ trial_dx rtol=1e-12 atol=1e-12
        @test recovered_dy ≈ trial_dy rtol=1e-12 atol=1e-12
        @test SDPX.reduced_augmented_kkt_residual!(
            augmented,
            trial_r,
            trial_p,
            recovered_dx,
            recovered_dy,
        ) <= 500eps(Float64)

        recovered_dX = zeros(3, 3)
        SDPX.buildP!(recovered_dX, problem.cons, 1, recovered_dx)
        recovered_dX .+= P
        recovered_raw_dY = X \ (R - recovered_dX * Y)
        recovered_dY =
            (recovered_raw_dY + transpose(recovered_raw_dY)) / 2
        recovery_residual = recovered_dY -
            (X \ (R - recovered_dX * Y) +
             transpose(X \ (R - recovered_dX * Y))) / 2
        recovered_dual = copy(trial_d)
        SDPX.accumulate_v!(
            recovered_dual,
            problem.cons,
            1,
            recovered_dY,
            -one(Float64),
        )
        recovered_dual .-= problem.B * recovered_dy
        recovered_equality = trial_p - transpose(problem.B) * recovered_dx
        @test norm(P + reshape(
            problem.cons.Av[1] * recovered_dx,
            3,
            3,
        ) - recovered_dX, Inf) <= 500eps(Float64)
        @test norm(recovery_residual, Inf) <= 500eps(Float64)
        @test norm(recovered_dual, Inf) <= 500eps(Float64)
        @test norm(recovered_equality, Inf) <= 500eps(Float64)
    end

    @testset "formulation precedes provider and conservative auto stays normal" begin
        problem = _augmented_test_problem(Float64; equalities=2)
        auto = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
                formulation=:auto,
                threads=1,
                verbosity=0,
            ),
        )
        @test auto.formulation_plan.formulation isa SDPX.DenseNormalEquations
        @test auto.kkt_backend === :dense_cholesky

        explicit = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            formulation=:augmented,
            threads=1,
            verbosity=0,
        )
        @test_throws ArgumentError SDPX.build_execution_plan(problem, explicit)

        zero_variable_problem = SDPX.ingest(
            Float64[],
            [zeros(Float64, 0, 3, 3)],
            [Matrix{Float64}(I, 3, 3)],
            zeros(Float64, 0, 0),
            Float64[];
            sparse=false,
            verbosity=0,
        )
        @test_throws ArgumentError SDPX.build_execution_plan(
            zero_variable_problem,
            explicit,
        )

        @test_throws ArgumentError SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
                formulation=:augmented,
                equality_solver=:qr,
                threads=1,
                verbosity=0,
            ),
        )
        @test_throws ArgumentError SDPX.plan_la_backend(
            Float64;
            requested=:legacy,
            route=:dense_augmented_ldlt,
        )

        # The public frontend accepts the explicit formulation name, but a
        # provider without the required LDLT capability still fails during
        # planning instead of switching formulation or implementation.
        resolved = SDPX.resolve_solve_options(
            Float64,
            SDPX.SolveOptions(
                formulation=:augmented,
                linear_algebra_backend=:standard,
            ),
        )
        @test resolved.core.formulation === :augmented
        @test_throws ArgumentError SDPX.build_execution_plan(
            problem,
            resolved.core,
        )

        @test_throws ArgumentError SDPX.build_execution_plan(
            SDPX.linear_program(
                [1.0],
                reshape([1.0], 1, 1),
                [0.0];
                sparse=false,
                verbosity=0,
            ),
            SDPX.SolverOptions{Float64}(
                algorithm=:lp,
                formulation=:augmented,
                verbosity=0,
            ),
        )

        sparse_problem = SDPX.ingest(
            [1.0],
            [[sparse([1], [1], [1.0], 3, 3)]],
            [Matrix{Float64}(I, 3, 3)],
            spzeros(Float64, 1, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        @test_throws ArgumentError SDPX.build_execution_plan(
            sparse_problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                formulation=:augmented,
                sparse=true,
                verbosity=0,
            ),
        )
    end
end
