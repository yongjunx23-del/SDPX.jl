using LinearAlgebra
using MultiFloats
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

@testset "BigFloat mixed reduced arrow without native panel" begin
    # Each scenario uses a fresh supplier so the prewarm count is proven
    # independently of any earlier cache warming.
    function probe(value=4096)
        count = Ref(0)
        memory = SDPX._lazy_memory_supplier() do
            count[] += 1
            value
        end
        return memory, count
    end

    # A: both stored -> no probe and both compute thunks short-circuit.
    memory_a, count_a = probe()
    @test !SDPX._arrow_crossover_needs_memory((
        reduced_arrow_decision=:r,
        mixed_reduced_arrow_decision=:m,
    ))
    @test SDPX._planned_or_computed_decision(
        (reduced_arrow_decision=:r,),
        :reduced_arrow_decision,
        () -> memory_a(),
    ) === :r
    @test SDPX._planned_or_computed_mixed_reduced_decision(
        (mixed_reduced_arrow_decision=:m,),
        () -> memory_a(),
    ) === :m
    @test count_a[] == 0

    # B: stored reduced + missing mixed -> prewarm once, mixed helper shares it.
    memory_b, count_b = probe()
    @test SDPX._arrow_crossover_needs_memory((
        reduced_arrow_decision=:r,
    ))
    memory_b()
    @test count_b[] == 1
    @test SDPX._planned_or_computed_mixed_reduced_decision(
        (reduced_arrow_decision=:r,),
        () -> memory_b(),
    ) == 4096
    @test count_b[] == 1

    # C: both missing -> cold prewarm once, both helpers share it.
    memory_c, count_c = probe()
    @test SDPX._arrow_crossover_needs_memory(NamedTuple())
    memory_c()
    @test count_c[] == 1
    @test SDPX._planned_or_computed_decision(
        NamedTuple(),
        :reduced_arrow_decision,
        () -> memory_c(),
    ) == 4096
    @test SDPX._planned_or_computed_mixed_reduced_decision(
        NamedTuple(),
        () -> memory_c(),
    ) == 4096
    @test count_c[] == 1

    # D: zero-valued probe still caches and counts once.
    memory_d, count_d = probe(0)
    @test memory_d() == 0
    @test memory_d() == 0
    @test count_d[] == 1

    setprecision(BigFloat, 256) do
        problem, _, _ = _bigfloat_arrow_fixture()
        requested_threads = max(Threads.nthreads(), 16)
        plan = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{BigFloat}(
                algorithm=:sdp,
                scaling=:none,
                presolve=false,
                extended_precision_blas=:off,
                mixed_precision_kkt=:on,
                threads=requested_threads,
            ),
        )
        @test plan.backend_config.route === :block_arrow
        @test !plan.backend_config.reduced_arrow
        @test plan.backend_config.mixed_reduced_arrow
        @test plan.threads <= requested_threads
        @test plan.threads <= Threads.nthreads()
        workspace = SDPX.Workspace(problem; execution_plan=plan)
        @test workspace.arrow.mixed_reduced_enabled
        @test workspace.thread_count == plan.threads
        @test workspace.arrow.mixed_reduced_threads == plan.threads
        @test workspace.arrow.mixed_reduced_threads <= Threads.nthreads()
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
        mixed_kkt_norm = SDPX._kkt_direction_operator_infinity_norm(mixed, problem)
        @test all(iszero, mixed.arrow.Sgg)
        SDPX.materialize_schur!(mixed_schur, mixed)
        @test mixed_kkt_norm ≈ maximum(vec(sum(abs.(reference_schur), dims=2)))
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
        factorization = SDPX.factorize!(SDPX.select_backend(workspace), workspace, problem, options)
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
        factorization = SDPX.factorize!(SDPX.select_backend(workspace), workspace, problem, options)
        @test factorization.ok
        @test !factorization.q_rank_deficient
        automatic_refinement = SDPX.SolverOptions{BigFloat}(
            verbosity=0,
            threads=requested_threads,
            extended_precision_blas=:on,
            equality_solver=:normal_equations,
            refine_policy=:auto,
            ϵ_gap=big"1e-10",
            ϵ_primal=big"1e-10",
            ϵ_dual=big"1e-10",
        )
        @test SDPX._has_owned_bigfloat_equality_arrow(
            workspace,
            workspace.arrow,
        )
        @test SDPX._skip_automatic_refinement(
            workspace,
            automatic_refinement,
            factorization,
        )
        @test !SDPX._skip_automatic_refinement(
            workspace,
            SDPX._replace_solver_options(
                automatic_refinement;
                refine_policy=:adaptive,
            ),
            factorization,
        )
        @test !SDPX._skip_automatic_refinement(
            workspace,
            automatic_refinement,
            merge(factorization, (reg_attempts=1,)),
        )
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

@testset "Owned BigFloat equality GEMV" begin
    setprecision(BigFloat, 256) do
        rows = 512
        columns = 192
        requested_threads = min(Threads.nthreads(), 4)
        panel = Matrix{BigFloat}(undef, rows, columns)
        @inbounds for column in 1:columns, row in 1:rows
            panel[row, column] = BigFloat(
                mod(17row + 29column, 257) - 128,
            ) / BigFloat(257)
        end
        column_vector = BigFloat.(range(
            BigFloat("-0.4"),
            BigFloat("0.7");
            length=columns,
        ))
        row_vector = BigFloat.(range(
            BigFloat("-0.8"),
            BigFloat("0.2");
            length=rows,
        ))

        row_reference = SDPX.alloc_zeros(BigFloat, rows)
        row_candidate = SDPX.alloc_zeros(BigFloat, rows)
        SDPX.kmul_owned!(row_reference, panel, column_vector)
        SDPX._arrow_equality_gemv!(
            row_candidate,
            panel,
            column_vector,
            requested_threads,
        )
        @test row_candidate == row_reference
        @test length(unique(objectid.(row_candidate))) == rows

        column_reference = SDPX.alloc_zeros(BigFloat, columns)
        column_candidate = SDPX.alloc_zeros(BigFloat, columns)
        SDPX.kmul_owned!(column_reference, transpose(panel), row_vector)
        SDPX._arrow_equality_gemv!(
            column_candidate,
            transpose(panel),
            row_vector,
            requested_threads,
        )
        @test column_candidate == column_reference
        @test length(unique(objectid.(column_candidate))) == columns

        sparse_panel = sparse(panel)
        sparse_row_candidate = SDPX.alloc_zeros(BigFloat, rows)
        SDPX._sparse_bigfloat_gemv_owned!(
            sparse_row_candidate,
            sparse_panel,
            column_vector,
        )
        @test sparse_row_candidate == row_reference
        @test length(unique(objectid.(sparse_row_candidate))) == rows

        sparse_column_candidate = SDPX.alloc_zeros(BigFloat, columns)
        SDPX._sparse_bigfloat_transpose_gemv_owned!(
            sparse_column_candidate,
            sparse_panel,
            row_vector,
            requested_threads,
        )
        @test sparse_column_candidate == column_reference
        @test length(unique(objectid.(sparse_column_candidate))) == columns

        expected_workers = requested_threads > 1 ? requested_threads : 1
        @test SDPX._bigfloat_gemv_worker_count(
            rows,
            columns,
            requested_threads,
        ) == expected_workers
    end
end

@testset "Owned BigFloat all-local block scheduling" begin
    @test SDPX._owned_bigfloat_block_task_count(0) == 1
    @test SDPX._owned_bigfloat_block_task_count(1) == 1
    @test SDPX._owned_bigfloat_block_task_count(64) == 64
    @test SDPX._owned_bigfloat_block_task_count(128) == 64
    setprecision(BigFloat, 256) do
        problem, X, Y = _bigfloat_block_diagonal_equality_fixture(
            block_count=256,
            equality_count=16,
        )
        requested_threads = min(Threads.nthreads(), 4)
        serial = SDPX.Workspace(
            problem;
            thread_count=1,
            extended_precision_blas=:on,
            equality_solver=:normal_equations,
        )
        parallel = SDPX.Workspace(
            problem;
            thread_count=requested_threads,
            extended_precision_blas=:on,
            equality_solver=:normal_equations,
        )
        options = SDPX.SolverOptions{BigFloat}(
            verbosity=0,
            threads=requested_threads,
            parameter_strategy=:adaptive,
            extended_precision_blas=:on,
            equality_solver=:normal_equations,
        )
        @test SDPX.use_owned_bigfloat_block_loops(parallel, problem) ==
              (requested_threads > 1)
        @test SDPX.use_owned_bigfloat_residual_path(serial, problem)

        x = BigFloat.(range(
            BigFloat("-0.1"),
            BigFloat("0.2");
            length=problem.dims.m,
        ))
        y = BigFloat.(range(
            BigFloat("-0.05"),
            BigFloat("0.08");
            length=problem.dims.n,
        ))
        mu = [BigFloat("0.3") for _ in 1:problem.dims.L]
        serial_residuals = SDPX.compute_residuals!(
            serial,
            problem,
            x,
            X,
            y,
            Y,
            mu,
            options,
        )
        serial_ok = SDPX.factor_blocks!(serial, X, Y)
        parallel_residuals = SDPX.threaded_compute_residuals!(
            parallel,
            problem,
            x,
            X,
            y,
            Y,
            mu,
            options;
            factor=true,
        )
        @test serial_ok
        @test parallel_residuals == (serial_residuals..., true)
        @test parallel.d == serial.d
        @test parallel.p == serial.p
        for block in 1:problem.dims.L
            @test parallel.blk[block].P == serial.blk[block].P
            @test parallel.blk[block].R == serial.blk[block].R
            @test parallel.blk[block].LX == serial.blk[block].LX
            @test parallel.blk[block].MY == serial.blk[block].MY
        end

        serial_rhs = SDPX._predictor_corrector_rhs!(serial, problem, Y)
        parallel_rhs = SDPX.threaded_predictor_corrector_rhs!(
            parallel,
            problem,
            Y,
        )
        @test parallel_rhs == serial_rhs
        SDPX.copy_owned!(serial.dx, x)
        SDPX.copy_owned!(parallel.dx, x)
        SDPX.threaded_direction_blocks!(serial, problem, Y)
        SDPX.threaded_direction_blocks!(parallel, problem, Y)
        for block in 1:problem.dims.L
            @test parallel.blk[block].dX == serial.blk[block].dX
            @test parallel.blk[block].dY == serial.blk[block].dY
            @test length(unique(objectid.(parallel.blk[block].dY))) == 4
        end

        # Allocation-free direction recovery reuses dY on every iteration.
        # A previous symmetrization implementation aliased its two mutable
        # off-diagonal BigFloat entries, which corrupted the second reuse but
        # was invisible in one-shot tests.
        SDPX.threaded_direction_blocks!(serial, problem, Y)
        SDPX.threaded_direction_blocks!(parallel, problem, Y)
        for block in 1:problem.dims.L
            @test parallel.blk[block].dX == serial.blk[block].dX
            @test parallel.blk[block].dY == serial.blk[block].dY
            @test length(unique(objectid.(parallel.blk[block].dY))) == 4
        end

        serial_affine = SDPX._affine_predictor_diagnostics!(
            serial,
            problem,
            X,
            Y,
        )
        parallel_affine = SDPX._affine_predictor_diagnostics!(
            parallel,
            problem,
            X,
            Y,
        )
        @test parallel_affine == serial_affine
        serial_legacy = SDPX._legacy_predictor_diagnostics!(
            serial,
            problem,
            X,
            Y,
        )
        parallel_legacy = SDPX._legacy_predictor_diagnostics!(
            parallel,
            problem,
            X,
            Y,
        )
        @test parallel_legacy == serial_legacy

        primal_fraction = BigFloat("0.97")
        dual_fraction = BigFloat("0.96")
        serial_steps = SDPX.threaded_line_search!(
            serial,
            X,
            Y,
            primal_fraction,
            dual_fraction,
            BigFloat("0.85"),
            zero(BigFloat),
            :fraction_to_boundary,
        )
        parallel_steps = SDPX.threaded_line_search!(
            parallel,
            X,
            Y,
            primal_fraction,
            dual_fraction,
            BigFloat("0.85"),
            zero(BigFloat),
            :fraction_to_boundary,
        )
        @test parallel_steps == serial_steps

        serial_objective = SDPX.threaded_dual_objective(
            serial,
            problem,
            y,
            Y,
        )
        parallel_objective = SDPX.threaded_dual_objective(
            parallel,
            problem,
            y,
            Y,
        )
        @test parallel_objective == serial_objective

        serial_X = deepcopy(X)
        serial_Y = deepcopy(Y)
        parallel_X = deepcopy(X)
        parallel_Y = deepcopy(Y)
        serial_update = SDPX.threaded_update_blocks!(
            serial,
            serial_X,
            serial_Y,
            serial_steps...,
        )
        parallel_update = SDPX.threaded_update_blocks!(
            parallel,
            parallel_X,
            parallel_Y,
            parallel_steps...,
        )
        @test parallel_update == serial_update
        @test parallel_X == serial_X
        @test parallel_Y == serial_Y

        serial_mu = [BigFloat("0.3") for _ in 1:problem.dims.L]
        parallel_mu = [BigFloat("0.3") for _ in 1:problem.dims.L]
        SDPX.threaded_update_mu!(
            serial,
            serial_mu,
            BigFloat("0.1"),
            problem.dims.k,
            serial_update[1],
            true,
        )
        SDPX.threaded_update_mu!(
            parallel,
            parallel_mu,
            BigFloat("0.1"),
            problem.dims.k,
            parallel_update[1],
            true,
        )
        @test parallel_mu == serial_mu

        sigma = BigFloat("0.2")
        average_mu = BigFloat("0.15")
        serial_corrector = SDPX.threaded_mehrotra_corrector_rhs!(
            serial,
            problem,
            X,
            Y,
            sigma,
            average_mu,
        )
        parallel_corrector = SDPX.threaded_mehrotra_corrector_rhs!(
            parallel,
            problem,
            X,
            Y,
            sigma,
            average_mu,
        )
        @test parallel_corrector == serial_corrector
        for block in 1:problem.dims.L
            @test parallel.blk[block].R == serial.blk[block].R
            @test parallel.blk[block].Z == serial.blk[block].Z
        end
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
        factorization = SDPX.factorize!(SDPX.select_backend(workspace), workspace, problem, options)
        @test factorization.ok
        @test factorization.q_rank_deficient
        @test factorization.equality_solver === :rank_revealing_qr
        @test workspace.Qchol isa SDPX.EqualityQRFactor{BigFloat}
        diagnostics =
            SDPX._equality_factor_diagnostics(workspace, problem.dims.n)
        @test diagnostics.rank == problem.dims.n - 1
        @test diagnostics.rank_deficient
    end
end
