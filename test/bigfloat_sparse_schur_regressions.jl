using LinearAlgebra
using SparseArrays
using Test

function _bigfloat_arrow_fixture()
    block_count = 4
    variable_count = block_count + 2
    coefficients = [
        zeros(BigFloat, variable_count, 2, 2)
        for _ in 1:block_count
    ]
    active = [[1, 2, block + 2] for block in 1:block_count]
    for block in 1:block_count
        for (position, variable) in pairs(active[block])
            # The three active coefficients form an exact basis of Sym(2).
            # This keeps the arrow Schur matrix strictly positive definite,
            # so the KKT regression checks the unregularized solve rather than
            # a deliberately rank-deficient fixture.
            if position == 1
                coefficients[block][variable, 1, 1] = one(BigFloat)
            elseif position == 2
                coefficients[block][variable, 1, 2] = one(BigFloat)
                coefficients[block][variable, 2, 1] = one(BigFloat)
            else
                coefficients[block][variable, 2, 2] = one(BigFloat)
            end
        end
    end
    problem = SDPX.ingest(
        ones(BigFloat, variable_count),
        coefficients,
        [zeros(BigFloat, 2, 2) for _ in 1:block_count],
        zeros(BigFloat, variable_count, 0),
        BigFloat[];
        sparse=true,
        verbosity=0,
    )
    X = [
        BigFloat[
            2 + block / 20 1 / 13
            1 / 13 3 / 2 + block / 30
        ]
        for block in 1:block_count
    ]
    Y = [
        BigFloat[
            7 / 5 + block / 40 1 / 17
            1 / 17 19 / 10 + block / 50
        ]
        for block in 1:block_count
    ]
    return problem, X, Y
end

function _reference_fused_arrow_build!(workspace, problem, X, Y)
    arrow = workspace.arrow
    SDPX._zero_arrow_schur!(arrow)
    for block in 1:problem.dims.L
        SDPX._fused_arrow_schur_block_generic!(
            arrow,
            workspace.blk[block],
            problem.cons,
            block,
            X[block],
            Y[block],
            arrow.Sgg,
        )
    end
    return arrow.Sgg
end

@testset "BigFloat sparse Schur allocation regressions" begin
    setprecision(BigFloat, 256) do
        problem, X, Y = _bigfloat_arrow_fixture()
        workspace = SDPX.Workspace(
            problem;
            thread_count=Threads.nthreads(),
            extended_precision_blas=:auto,
        )
        @test workspace.arrow !== nothing
        @test workspace.fused_arrow
        @test workspace.thread_count == 1
        @test workspace.extended_precision.packing_bytes == 0
        @test !workspace.extended_precision.lower_only
        @test all(
            plan.decision.reason === :fused_arrow_specialized &&
            !plan.decision.enabled
            for plan in workspace.extended_precision.block_plans
        )
        plan = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{BigFloat}(
                extended_precision_blas=:auto,
                verbosity=0,
            ),
        )
        @test plan.gram_kernel == :fused_arrow_2x2
        @test SDPX.factor_blocks!(workspace, X, Y)

        SDPX.schur_build!(
            workspace,
            problem,
            problem.cons,
            X,
            Y,
        )
        optimized = SDPX.alloc_zeros(
            BigFloat,
            problem.dims.m,
            problem.dims.m,
        )
        SDPX.materialize_schur!(optimized, workspace)
        optimized = deepcopy(optimized)

        _reference_fused_arrow_build!(workspace, problem, X, Y)
        reference = SDPX.alloc_zeros(
            BigFloat,
            problem.dims.m,
            problem.dims.m,
        )
        SDPX.materialize_schur!(reference, workspace)
        scale = max(maximum(abs, reference), one(BigFloat))
        @test maximum(abs, optimized - reference) / scale <= big"1e-70"
        @test issymmetric(optimized)

        # Warm both paths before measuring allocations. The optimized kernel
        # retains only fixed per-block Cholesky-solve scratch; the reference
        # allocates for every scalar operation in every variable pair.
        SDPX.schur_build!(
            workspace,
            problem,
            problem.cons,
            X,
            Y,
        )
        _reference_fused_arrow_build!(workspace, problem, X, Y)
        optimized_allocations = @allocated SDPX.schur_build!(
            workspace,
            problem,
            problem.cons,
            X,
            Y,
        )
        reference_allocations = @allocated _reference_fused_arrow_build!(
            workspace,
            problem,
            X,
            Y,
        )
        @test optimized_allocations * 4 < reference_allocations

        arrow = workspace.arrow
        SDPX._zero_arrow_schur!(arrow)
        for storage in (
            (arrow.Sgg,),
            arrow.Dsrc,
            arrow.coupling,
        )
            for matrix in storage
                length(matrix) < 2 && continue
                @test matrix[1] !== matrix[2]
            end
        end
    end
end

@testset "BigFloat packed sparse contractions" begin
    setprecision(BigFloat, 256) do
        problem, _, Y = _bigfloat_arrow_fixture()
        coefficients = problem.cons
        x = BigFloat.(1:problem.dims.m)

        optimized = zeros(BigFloat, 2, 2)
        @test optimized[1] === optimized[2]
        reference = SDPX.alloc_zeros(BigFloat, 2, 2)
        SDPX.buildP!(optimized, coefficients, 1, x)
        SDPX._buildP_sparse_generic!(reference, coefficients, 1, x)
        @test optimized == reference
        @test optimized[1] !== optimized[2]
        @test optimized[1] !== optimized[3]

        source_before = deepcopy(problem.c)
        safe = Vector{BigFloat}(problem.c)
        @test safe[1] === problem.c[1]
        SDPX.accumulate_v!(
            safe,
            coefficients,
            1,
            Y[1],
            -one(BigFloat),
        )
        @test problem.c == source_before

        owned = SDPX.alloc_zeros(BigFloat, problem.dims.m)
        SDPX.copy_owned!(owned, problem.c)
        SDPX.accumulate_v_owned!(
            owned,
            coefficients,
            1,
            Y[1],
            -one(BigFloat),
        )
        scale = max(maximum(abs, safe), one(BigFloat))
        @test maximum(abs, owned - safe) / scale <= big"1e-70"
        @test problem.c == source_before

        SDPX.copy_owned!(owned, problem.c)
        safe_allocations = @allocated begin
            copyto!(safe, problem.c)
            SDPX.accumulate_v!(
                safe,
                coefficients,
                1,
                Y[1],
                -one(BigFloat),
            )
        end
        owned_allocations = @allocated begin
            SDPX.copy_owned!(owned, problem.c)
            SDPX.accumulate_v_owned!(
                owned,
                coefficients,
                1,
                Y[1],
                -one(BigFloat),
            )
        end
        @test owned_allocations < safe_allocations
    end
end

@testset "BigFloat general sparse COO Schur contractions" begin
    setprecision(BigFloat, 256) do
        variable_count = 12
        dimension = 5
        coefficients =
            [zeros(BigFloat, variable_count, dimension, dimension)]
        for variable in 1:variable_count
            row = mod1(variable, dimension)
            column = mod1(3variable + 1, dimension)
            diagonal =
                one(BigFloat) + BigFloat(variable) / BigFloat(20)
            off_diagonal = BigFloat(variable) / BigFloat(137)
            coefficients[1][variable, row, row] = diagonal
            coefficients[1][variable, row, column] = off_diagonal
            coefficients[1][variable, column, row] = off_diagonal
            coefficients[1][variable, column, column] =
                diagonal / BigFloat(11)
        end
        objective = ones(BigFloat, variable_count)
        constants = [zeros(BigFloat, dimension, dimension)]
        equalities = zeros(BigFloat, variable_count, 1)
        equality_rhs = zeros(BigFloat, 1)
        sparse_problem = SDPX.ingest(
            objective,
            coefficients,
            constants,
            equalities,
            equality_rhs;
            sparse=true,
            verbosity=0,
        )
        dense_problem = SDPX.ingest(
            objective,
            coefficients,
            constants,
            equalities,
            equality_rhs;
            sparse=false,
            verbosity=0,
        )
        sparse_workspace = SDPX.Workspace(
            sparse_problem;
            extended_precision_blas=:off,
            thread_count=1,
        )
        dense_workspace = SDPX.Workspace(
            dense_problem;
            extended_precision_blas=:off,
            thread_count=1,
        )
        @test sparse_workspace.arrow === nothing

        X = [Matrix{BigFloat}(BigFloat(2) * I, dimension, dimension)]
        Y = [Matrix{BigFloat}(BigFloat(3) * I, dimension, dimension)]
        @test SDPX.factor_blocks!(sparse_workspace, X, Y)
        @test SDPX.factor_blocks!(dense_workspace, X, Y)

        SDPX.sparse_schur_block!(
            sparse_workspace.blk[1],
            sparse_problem.cons,
            1,
            X[1],
            Y[1],
        )
        allocation_bytes = @allocated SDPX.sparse_schur_block!(
            sparse_workspace.blk[1],
            sparse_problem.cons,
            1,
            X[1],
            Y[1],
        )
        @test allocation_bytes <= 64_000

        expected = SDPX.schur_build!(
            dense_workspace,
            dense_problem,
            dense_problem.cons,
            X,
            Y,
        )
        scattered =
            SDPX.alloc_zeros(BigFloat, variable_count, variable_count)
        SDPX.sparse_schur_block_scatter!(
            scattered,
            sparse_workspace.blk[1],
            sparse_problem.cons,
            1,
            X[1],
            Y[1],
        )
        @test maximum(abs, scattered - expected) /
              max(maximum(abs, expected), one(BigFloat)) <= big"1e-60"
        scatter_allocation_bytes =
            @allocated SDPX.sparse_schur_block_scatter!(
                scattered,
                sparse_workspace.blk[1],
                sparse_problem.cons,
                1,
                X[1],
                Y[1],
            )
        @test scatter_allocation_bytes <= 64_000

        actual = SDPX.schur_build!(
            sparse_workspace,
            sparse_problem,
            sparse_problem.cons,
            X,
            Y,
        )
        scale = max(maximum(abs, expected), one(BigFloat))
        @test maximum(abs, actual - expected) / scale <= big"1e-60"
    end
end

@testset "BigFloat dense contraction alias safety" begin
    setprecision(BigFloat, 256) do
        variable_count = 8
        coefficients = [zeros(BigFloat, variable_count, 2, 2)]
        for variable in 1:variable_count
            coefficients[1][variable, 1, 1] = BigFloat(variable)
            coefficients[1][variable, 1, 2] =
                BigFloat(variable) / BigFloat(19)
            coefficients[1][variable, 2, 1] =
                BigFloat(variable) / BigFloat(19)
            coefficients[1][variable, 2, 2] =
                BigFloat(variable + 2)
        end
        problem = SDPX.ingest(
            BigFloat.(1:variable_count),
            coefficients,
            [zeros(BigFloat, 2, 2)],
            zeros(BigFloat, variable_count, 0),
            BigFloat[];
            sparse=false,
            verbosity=0,
        )
        x = BigFloat.(1:variable_count)
        public_block = zeros(BigFloat, 2, 2)
        owned_block = SDPX.alloc_zeros(BigFloat, 2, 2)
        SDPX.buildP!(public_block, problem.cons, 1, x)
        SDPX.buildP_owned!(owned_block, problem.cons, 1, x)
        @test public_block == owned_block
        @test public_block[1] !== public_block[2]

        matrix = BigFloat[7 / 5 1 / 13; 2 / 17 11 / 6]
        source_before = deepcopy(problem.c)
        public_vector = Vector{BigFloat}(problem.c)
        @test public_vector[1] === problem.c[1]
        SDPX.accumulate_v!(
            public_vector,
            problem.cons,
            1,
            matrix,
            -one(BigFloat),
        )
        @test problem.c == source_before

        owned_vector = SDPX.alloc_zeros(BigFloat, variable_count)
        SDPX.copy_owned!(owned_vector, problem.c)
        SDPX.accumulate_v_owned!(
            owned_vector,
            problem.cons,
            1,
            matrix,
            -one(BigFloat),
        )
        @test public_vector == owned_vector
        @test problem.c == source_before

        public_build_allocations =
            @allocated SDPX.buildP!(public_block, problem.cons, 1, x)
        owned_build_allocations =
            @allocated SDPX.buildP_owned!(
                owned_block,
                problem.cons,
                1,
                x,
            )
        @test owned_build_allocations < public_build_allocations

        public_accumulate_allocations = @allocated begin
            copyto!(public_vector, problem.c)
            SDPX.accumulate_v!(
                public_vector,
                problem.cons,
                1,
                matrix,
                -one(BigFloat),
            )
        end
        owned_accumulate_allocations = @allocated begin
            SDPX.copy_owned!(owned_vector, problem.c)
            SDPX.accumulate_v_owned!(
                owned_vector,
                problem.cons,
                1,
                matrix,
                -one(BigFloat),
            )
        end
        @test owned_accumulate_allocations <
              public_accumulate_allocations
    end
end

@testset "BigFloat exact-arrow KKT owns its scratch" begin
    setprecision(BigFloat, 256) do
        problem, X, Y = _bigfloat_arrow_fixture()
        workspace = SDPX.Workspace(problem; thread_count=1)
        @test SDPX.factor_blocks!(workspace, X, Y)
        SDPX.schur_build!(
            workspace,
            problem,
            problem.cons,
            X,
            Y,
        )
        arrow = workspace.arrow
        Schur = SDPX.alloc_zeros(
            BigFloat,
            problem.dims.m,
            problem.dims.m,
        )
        SDPX.materialize_schur!(Schur, workspace)
        Schur = deepcopy(Schur)
        source_global = deepcopy(arrow.Sgg)
        source_local = deepcopy(arrow.Dsrc)
        source_coupling = deepcopy(arrow.coupling)

        options = SDPX.SolverOptions{BigFloat}(verbosity=0)
        factorization = SDPX.factor_kkt!(workspace, problem, options)
        @test factorization.ok
        @test arrow.Sgg == source_global
        @test arrow.Dsrc == source_local
        @test arrow.coupling == source_coupling

        rhs = BigFloat.(range(
            BigFloat("0.2"),
            BigFloat("1.6");
            length=problem.dims.m,
        ))
        rhs_before = deepcopy(rhs)
        solution = zeros(BigFloat, problem.dims.m)
        SDPX.solve_kkt!(
            workspace,
            0,
            rhs,
            BigFloat[],
            solution,
            BigFloat[],
        )
        @test rhs == rhs_before
        @test arrow.Sgg == source_global
        @test arrow.Dsrc == source_local
        @test arrow.coupling == source_coupling
        residual = Schur * solution - rhs
        @test maximum(abs, residual) <= big"1e-65"
        @test solution[1] !== solution[2]
        for (position, variable) in pairs(arrow.global_ids)
            @test arrow.rg[position] !== rhs[variable]
            @test solution[variable] !== arrow.rg[position]
        end
        for block in eachindex(arrow.local_ids)
            for (position, variable) in pairs(arrow.local_ids[block])
                @test arrow.tmp[block][position] !== rhs[variable]
                @test solution[variable] !== arrow.tmp[block][position]
            end
        end
    end
end
