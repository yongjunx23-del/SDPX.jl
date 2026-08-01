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

function _bigfloat_block_diagonal_equality_fixture(
    ;
    block_count::Int=24,
    equality_count::Int=48,
    rank_deficient::Bool=false,
)
    variable_count = 3 * block_count
    equality_count <= variable_count ||
        throw(ArgumentError("equalities cannot exceed variables"))
    coefficients = [
        zeros(BigFloat, variable_count, 2, 2)
        for _ in 1:block_count
    ]
    for block in 1:block_count
        first = 3 * block - 2
        coefficients[block][first, 1, 1] = one(BigFloat)
        coefficients[block][first + 1, 1, 2] = one(BigFloat)
        coefficients[block][first + 1, 2, 1] = one(BigFloat)
        coefficients[block][first + 2, 2, 2] = one(BigFloat)
    end
    equality = SDPX.alloc_zeros(
        BigFloat,
        variable_count,
        equality_count,
    )
    for column in 1:equality_count
        equality[column, column] = one(BigFloat)
        for row in (equality_count + 1):variable_count
            numerator = mod(17 * row + 11 * column, 29) - 14
            equality[row, column] =
                BigFloat(numerator) / BigFloat(257)
        end
    end
    if rank_deficient && equality_count >= 2
        for row in 1:variable_count
            equality[row, equality_count] = BigFloat(equality[row, 1])
        end
    end
    problem = SDPX.ingest(
        ones(BigFloat, variable_count),
        coefficients,
        [zeros(BigFloat, 2, 2) for _ in 1:block_count],
        equality,
        zeros(BigFloat, equality_count);
        sparse=true,
        verbosity=0,
    )
    X = [
        BigFloat[
            2 + block / 100 1 / 31
            1 / 31 3 / 2 + block / 200
        ]
        for block in 1:block_count
    ]
    Y = [
        BigFloat[
            7 / 5 + block / 150 1 / 37
            1 / 37 19 / 10 + block / 250
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

@testset "BigFloat native reduced-arrow Schur" begin
    setprecision(BigFloat, 256) do
        problem, X, Y = _bigfloat_arrow_fixture()
        reference = SDPX.Workspace(
            problem;
            thread_count=1,
            extended_precision_blas=:off,
        )
        requested_threads = min(Threads.nthreads(), 4)
        reduced = SDPX.Workspace(
            problem;
            thread_count=requested_threads,
            extended_precision_blas=:on,
        )
        @test reduced.arrow.reduced_panel_enabled
        @test reduced.thread_count == requested_threads
        @test SDPX.factor_blocks!(reference, X, Y)
        @test SDPX.factor_blocks!(reduced, X, Y)

        SDPX.schur_build!(reference, problem, problem.cons, X, Y)
        SDPX.schur_build!(reduced, problem, problem.cons, X, Y)
        @test reduced.arrow.reduced_panel_ready
        reference_schur = SDPX.alloc_zeros(
            BigFloat,
            problem.dims.m,
            problem.dims.m,
        )
        reduced_schur = SDPX.alloc_zeros(
            BigFloat,
            problem.dims.m,
            problem.dims.m,
        )
        SDPX.materialize_schur!(reference_schur, reference)
        SDPX.materialize_schur!(reduced_schur, reduced)
        scale = max(maximum(abs, reference_schur), one(BigFloat))
        @test maximum(abs, reduced_schur - reference_schur) / scale <=
              big"1e-70"

        reference_factor = SDPX.factor_arrow_kkt!(
            reference,
            SDPX.SolverOptions{BigFloat}(verbosity=0),
        )
        reduced_factor = SDPX.factor_arrow_kkt!(
            reduced,
            SDPX.SolverOptions{BigFloat}(verbosity=0),
        )
        @test reference_factor.ok
        @test reduced_factor.ok
        right_hand_side = BigFloat.(1:problem.dims.m) ./ BigFloat(11)
        reference_solution = SDPX.alloc_zeros(
            BigFloat,
            problem.dims.m,
        )
        reduced_solution = SDPX.alloc_zeros(
            BigFloat,
            problem.dims.m,
        )
        SDPX.solve_arrow_kkt!(
            reference,
            right_hand_side,
            reference_solution,
        )
        SDPX.solve_arrow_kkt!(
            reduced,
            right_hand_side,
            reduced_solution,
        )
        solution_scale =
            max(maximum(abs, reference_solution), one(BigFloat))
        @test maximum(abs, reduced_solution - reference_solution) /
              solution_scale <= big"1e-68"
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

@testset "BigFloat singleton-arrow Float64x4 preconditioner" begin
    setprecision(BigFloat, 256) do
        problem, X, Y = _bigfloat_arrow_fixture()
        reference = SDPX.Workspace(problem; thread_count=1)
        mixed = SDPX.Workspace(
            problem;
            thread_count=min(Threads.nthreads(), 4),
            extended_precision_blas=:on,
            mixed_precision_kkt=:on,
        )
        @test mixed.mixed_precision === nothing
        @test mixed.arrow.mixed_reduced_enabled
        @test eltype(mixed.arrow.mixed_reduced_panel) ==
              SDPX.mixed_arrow_arithmetic(BigFloat)
        @test SDPX.factor_blocks!(reference, X, Y)
        @test SDPX.factor_blocks!(mixed, X, Y)
        SDPX.schur_build!(reference, problem, problem.cons, X, Y)
        SDPX.schur_build!(mixed, problem, problem.cons, X, Y)
        @test mixed.arrow.mixed_reduced_ready
        diagnostics = SDPX._mixed_precision_kkt_diagnostics(mixed)
        @test diagnostics.available
        @test diagnostics.backend === :float64x4_reduced_arrow
        @test diagnostics.active
        @test diagnostics.attempted
        @test !diagnostics.fell_back
        @test diagnostics.threads ==
              min(Threads.nthreads(), 4)
        reference_schur =
            SDPX.alloc_zeros(BigFloat, problem.dims.m, problem.dims.m)
        mixed_schur = zeros(BigFloat, problem.dims.m, problem.dims.m)
        SDPX.materialize_schur!(reference_schur, reference)
        SDPX.materialize_schur!(mixed_schur, mixed)
        @test maximum(abs, mixed_schur - reference_schur) /
              max(maximum(abs, reference_schur), one(BigFloat)) <
              big"1e-65"
        @test all(
            left == right ||
            mixed_schur[left] !== mixed_schur[right]
            for left in eachindex(mixed_schur),
                right in eachindex(mixed_schur)
        )

        options = SDPX.SolverOptions{BigFloat}(verbosity=0)
        @test SDPX.factor_arrow_kkt!(reference, options).ok
        @test SDPX.factor_arrow_kkt!(mixed, options).ok
        vector = BigFloat.(range(
            BigFloat("0.17"),
            BigFloat("1.31");
            length=problem.dims.m,
        ))
        reference_product = SDPX.alloc_zeros(BigFloat, problem.dims.m)
        mixed_product = SDPX.alloc_zeros(BigFloat, problem.dims.m)
        SDPX.schur_mul!(
            reference_product,
            reference,
            vector,
            one(BigFloat),
            zero(BigFloat),
        )
        SDPX.schur_mul!(
            mixed_product,
            mixed,
            vector,
            one(BigFloat),
            zero(BigFloat),
        )
        @test maximum(abs, mixed_product - reference_product) /
              max(maximum(abs, reference_product), one(BigFloat)) <
              big"1e-65"

        reference_solution = SDPX.alloc_zeros(BigFloat, problem.dims.m)
        mixed_solution = SDPX.alloc_zeros(BigFloat, problem.dims.m)
        SDPX.solve_arrow_kkt!(reference, vector, reference_solution)
        SDPX.solve_arrow_kkt!(mixed, vector, mixed_solution)
        @test maximum(abs, mixed_solution - reference_solution) /
              max(maximum(abs, reference_solution), one(BigFloat)) <
              big"1e-55"

        SDPX.materialize_mixed_arrow_native_fallback!(
            mixed,
            :regression_fallback,
        )
        @test !mixed.arrow.mixed_reduced_ready
        @test !mixed.arrow.mixed_reduced_enabled
        @test mixed.arrow.reduced_panel_ready
        @test mixed.arrow.reduced_panel_enabled
        fallback_product = SDPX.alloc_zeros(
            BigFloat,
            problem.dims.m,
        )
        SDPX.schur_mul!(
            fallback_product,
            mixed,
            vector,
            one(BigFloat),
            zero(BigFloat),
        )
        @test maximum(abs, fallback_product - reference_product) /
              max(maximum(abs, reference_product), one(BigFloat)) <
              big"1e-65"
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
        panel = workspace.arrow.reduced_panel
        if size(panel, 1) >= 4
            @test panel[2, 1] !== panel[4, 1]
        end
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

@testset "BigFloat block-diagonal equality arrow KKT" begin
    setprecision(BigFloat, 256) do
        problem, X, Y = _bigfloat_block_diagonal_equality_fixture()
        requested_threads = min(Threads.nthreads(), 4)
        options = SDPX.SolverOptions{BigFloat}(
            verbosity=0,
            threads=requested_threads,
            extended_precision_blas=:on,
            equality_solver=:normal_equations,
        )
        plan = SDPX.build_execution_plan(problem, options)
        @test plan.kkt_backend === :block_arrow
        @test plan.threads == requested_threads
        @test plan.schedule === (
            requested_threads > 1 ?
            :owned_bigfloat_equality_tiles : :serial
        )

        workspace = SDPX.Workspace(
            problem;
            thread_count=requested_threads,
            extended_precision_blas=:on,
            equality_solver=:normal_equations,
        )
        @test workspace.arrow !== nothing
        @test workspace.fused_arrow
        @test isempty(workspace.arrow.global_ids)
        @test workspace.thread_count == requested_threads
        @test length(workspace.vpartial) == 1
        @test SDPX.factor_blocks!(workspace, X, Y)
        SDPX.schur_build!(
            workspace,
            problem,
            problem.cons,
            X,
            Y,
        )
        schur = SDPX.alloc_zeros(
            BigFloat,
            problem.dims.m,
            problem.dims.m,
        )
        SDPX.materialize_schur!(schur, workspace)
        factorization = SDPX.factor_kkt!(workspace, problem, options)
        @test factorization.ok
        @test !factorization.q_rank_deficient
        expected_gram = transpose(workspace.Btil) * workspace.Btil
        gram_scale = max(maximum(abs, expected_gram), one(BigFloat))
        @test maximum(abs, LowerTriangular(workspace.Q) -
                           LowerTriangular(expected_gram)) /
              gram_scale <= big"1e-70"
        selected_gram_workers =
            SDPX.ExtendedPrecisionBLAS._syrk_bigfloat_selected_workers(
                workspace.Btil,
                SDPX._equality_gram_crossover(
                    workspace.Btil,
                    options,
                    requested_threads,
                ).config,
                requested_threads,
            )
        @test workspace.equality_gram_kernel === (
            selected_gram_workers > 1 ?
            :threaded_blocked_triangular_syrk :
            :blocked_triangular_syrk
        )

        primal_rhs = BigFloat.(range(
            BigFloat("-0.7"),
            BigFloat("1.1");
            length=problem.dims.m,
        ))
        equality_rhs = BigFloat.(range(
            BigFloat("-0.2"),
            BigFloat("0.3");
            length=problem.dims.n,
        ))
        dx = zeros(BigFloat, problem.dims.m)
        dy = zeros(BigFloat, problem.dims.n)
        SDPX.solve_kkt!(
            workspace,
            problem.dims.n,
            primal_rhs,
            equality_rhs,
            dx,
            dy,
        )
        first_residual = schur * dx - problem.B * dy - primal_rhs
        second_residual = transpose(problem.B) * dx - equality_rhs
        scale = max(
            maximum(abs, primal_rhs),
            maximum(abs, equality_rhs),
            one(BigFloat),
        )
        @test max(
            maximum(abs, first_residual),
            maximum(abs, second_residual),
        ) / scale <= big"1e-65"
        @test length(unique(objectid.(dx))) == length(dx)
        @test length(unique(objectid.(dy))) == length(dy)
    end
end

@testset "BigFloat rank-deficient block-diagonal equalities" begin
    setprecision(BigFloat, 256) do
        problem, X, Y = _bigfloat_block_diagonal_equality_fixture(
            block_count=6,
            equality_count=4,
            rank_deficient=true,
        )
        workspace = SDPX.Workspace(
            problem;
            thread_count=min(Threads.nthreads(), 4),
            extended_precision_blas=:on,
            equality_solver=:auto,
        )
        @test workspace.arrow !== nothing
        @test SDPX.factor_blocks!(workspace, X, Y)
        SDPX.schur_build!(
            workspace,
            problem,
            problem.cons,
            X,
            Y,
        )
        options = SDPX.SolverOptions{BigFloat}(
            verbosity=0,
            extended_precision_blas=:on,
            equality_solver=:auto,
        )
        factorization = SDPX.factor_kkt!(workspace, problem, options)
        @test factorization.ok
        @test factorization.q_rank_deficient
        @test factorization.equality_solver === :rank_revealing_qr
        diagnostics =
            SDPX._equality_factor_diagnostics(workspace, problem.dims.n)
        @test diagnostics.rank == problem.dims.n - 1
        @test diagnostics.rank_deficient
    end
end
