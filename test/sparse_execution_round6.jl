using LinearAlgebra
using Random
using SparseArrays
using Test

@testset "Round 6 sparse execution core" begin
    pattern = sparse([4.0 1 0; 1 3 1; 0 1 2])
    symbolic = SDPX.analyze_sparse_pattern(pattern)
    @test symbolic.n == 3
    @test length(symbolic.input_colptr) == 4
    @test length(symbolic.input_rowval) == symbolic.input_nnz
    @test length(symbolic.factor_colptr) == 4
    @test length(symbolic.factor_rowval) == symbolic.factor_nnz
    @test sort(symbolic.permutation) == collect(1:3)
    @test symbolic.factor_nnz >= symbolic.input_nnz

    @testset "CHOLMOD refactor and solve" begin
        factor = SDPX.sparse_factor(
            pattern;
            provider=SDPX.CHOLMODSparseProvider(),
        )
        rhs = [1.0, 2.0, 3.0]
        solution = zeros(3)
        SDPX.sparse_factor_solve!(solution, factor, rhs)
        @test solution ≈ Matrix(pattern) \ rhs
        changed = copy(pattern)
        changed.nzval .*= 2
        SDPX.numeric_factorize!(factor, changed)
        changed_diag = SDPX.sparse_factor_diagnostics(factor)
        @test changed_diag.numeric_refactorizations == 2
        @test changed_diag.pattern_reused == 1
        @test changed_diag.factorization_attempts == 2
        @test changed_diag.factorization_successes == 2
        @test changed_diag.factorization_failures == 0
        @test changed_diag.factorization_attempts ==
              changed_diag.factorization_successes + changed_diag.factorization_failures
        bad = copy(changed)
        bad.nzval[bad.colptr[1]] = -1.0
        SDPX.numeric_factorize!(factor, bad)
        @test !issuccess(factor)
        bad_diag = SDPX.sparse_factor_diagnostics(factor)
        @test bad_diag.factorization_attempts == 3
        @test bad_diag.factorization_successes == 2
        @test bad_diag.factorization_failures == 1
        SDPX.numeric_factorize!(factor, changed)
        @test issuccess(factor)
        recovered_diag = SDPX.sparse_factor_diagnostics(factor)
        @test recovered_diag.numeric_refactorizations == 3
        @test recovered_diag.factorization_attempts == 4
        @test recovered_diag.factorization_successes == 3
        @test recovered_diag.factorization_failures == 1
    end

    @testset "generic sparse factor remains sparse" begin
        factor = SDPX.sparse_factor(
            pattern;
            provider=SDPX.GenericSparseProvider(Float64),
        )
        @test length(factor.nzval) == factor.symbolic.factor_nnz
        @test length(factor.nzval) < length(pattern)^2
        rhs = [1.0, 2.0, 3.0]
        solution = zeros(3)
        SDPX.sparse_factor_solve!(solution, factor, rhs)
        @test pattern * solution ≈ rhs
    end

    @testset "generic refactor maps and failure recovery" begin
        # The star graph forces a deterministic nonidentity ordering.  Both
        # triangles are present with different values so this also exercises
        # the old authoritative-lower rule after permutation.
        n = 4
        rows = Int[]
        cols = Int[]
        values = Float64[]
        for index in 1:n
            push!(rows, index)
            push!(cols, index)
            push!(values, 8.0 + index)
        end
        for (row, column) in ((1, 2), (1, 3), (1, 4))
            push!(rows, row); push!(cols, column); push!(values, 0.11)
            push!(rows, column); push!(cols, row); push!(values, 0.22)
        end
        full = sparse(rows, cols, values, n, n)
        symbolic = SDPX.analyze_sparse_pattern(full)
        @test symbolic.permutation != collect(1:n)
        factor = SDPX.sparse_factor(
            full;
            provider=SDPX.GenericSparseProvider(Float64),
            symbolic=symbolic,
        )
        @test factor.input_colptr == full.colptr
        @test factor.input_rowval == full.rowval
        @test all(>(0), factor.diagonal_positions)
        @test length(factor.column_link_positions) == n
        source_map = copy(factor.source_pointers)
        work_id = objectid(factor.numeric_work)

        # Reconstruct exactly the symmetric matrix consumed by the old
        # permuted-coordinate Dict path, independently of the precomputed map.
        expected = zeros(Float64, n, n)
        selected = falses(n, n)
        inverse = symbolic.inverse_permutation
        for column in 1:n
            for pointer in full.colptr[column]:(full.colptr[column + 1] - 1)
                row = full.rowval[pointer]
                prow, pcol = inverse[row], inverse[column]
                lower = (max(prow, pcol), min(prow, pcol))
                if !selected[lower...] || prow >= pcol
                    selected[lower...] = true
                    expected[row, column] = full.nzval[pointer]
                    expected[column, row] = full.nzval[pointer]
                end
            end
        end

        rhs = [1.0, -2.0, 3.0, 0.5]
        solution = zeros(n)
        SDPX.sparse_factor_solve!(solution, factor, rhs)
        @test expected * solution ≈ rhs atol=1e-12

        changed = copy(full)
        changed.nzval .*= 1.25
        SDPX.numeric_factorize!(factor, changed)
        @test factor.numeric_refactorizations == 2
        @test factor.factorization_attempts == 2
        @test factor.factorization_successes == 2
        @test factor.factorization_failures == 0
        @test factor.source_pointers == source_map
        @test objectid(factor.numeric_work) == work_id
        refreshed = zeros(n)
        SDPX.sparse_factor_solve!(refreshed, factor, rhs)
        @test norm(expected .* 1.25 * refreshed - rhs, Inf) <= 1e-12

        changed_pattern = sparse([9.0 0 1 0; 0 10 0 0; 1 0 11 0; 0 0 0 12])
        @test_throws ArgumentError SDPX.numeric_factorize!(factor, changed_pattern)
        @test factor.numeric_refactorizations == 2
        @test factor.factorization_attempts == 2
        @test factor.factorization_successes == 2
        @test factor.factorization_failures == 0

        bad = copy(changed)
        bad.nzval[bad.colptr[1]] = -1.0
        SDPX.numeric_factorize!(factor, bad)
        @test !issuccess(factor)
        @test factor.factorization_attempts == 3
        @test factor.factorization_successes == 2
        @test factor.factorization_failures == 1
        SDPX.numeric_factorize!(factor, changed)
        @test issuccess(factor)
        @test factor.numeric_refactorizations == 3
        @test factor.factorization_attempts == 4
        @test factor.factorization_successes == 3
        @test factor.factorization_failures == 1
        @test factor.factorization_attempts ==
              factor.factorization_successes + factor.factorization_failures

        # This cycle has one genuine symbolic fill slot.  Repeated numeric
        # refactors must clear that slot per factor column; a single
        # factor-wide work reset would leak a previous column's update here.
        cycle = sparse([6.0 1 0 1; 1 6 1 0; 0 1 6 1; 1 0 1 6])
        cycle_factor = SDPX.sparse_factor(
            cycle;
            provider=SDPX.GenericSparseProvider(Float64),
        )
        @test count(==(0), cycle_factor.source_pointers) > 0
        cycle_rhs = [1.0, -2.0, 0.5, 3.0]
        cycle_solution = zeros(4)
        SDPX.sparse_factor_solve!(cycle_solution, cycle_factor, cycle_rhs)
        @test cycle * cycle_solution ≈ cycle_rhs atol=1e-12
        cycle_changed = copy(cycle)
        cycle_changed.nzval .*= 1.1
        SDPX.numeric_factorize!(cycle_factor, cycle_changed)
        SDPX.sparse_factor_solve!(cycle_solution, cycle_factor, cycle_rhs)
        @test cycle_changed * cycle_solution ≈ cycle_rhs atol=1e-12

        # Setup remains fail-closed for a symbolic analysis from a strict
        # superset pattern, even when the numeric matrix is a valid subset.
        super_pattern = sparse([6.0 1 0 1; 1 6 1 0; 0 1 6 1; 1 0 1 6])
        sub_pattern = sparse([6.0 1 0 0; 1 6 1 0; 0 1 6 1; 0 0 1 6])
        provider = SDPX.GenericSparseProvider(Float64)
        super_symbolic = SDPX.analyze_sparse_pattern(super_pattern, provider)
        @test_throws ArgumentError SDPX.instantiate_sparse_factor(
            provider, super_symbolic, sub_pattern,
        )
    end

    @testset "generic solve remains safe for concurrent RHS" begin
        A = sparse([7.0 1 0 1; 1 6 1 0; 0 1 5 1; 1 0 1 8])
        factor = SDPX.sparse_factor(
            A;
            provider=SDPX.GenericSparseProvider(Float64),
        )
        rhs_values = [
            [1.0, 2.0, 3.0, 4.0],
            [-2.0, 0.5, 1.0, 3.0],
            [4.0, -1.0, 2.0, 0.25],
            [0.0, 2.5, -3.0, 1.0],
        ]
        serial = map(rhs_values) do rhs
            destination = zeros(4)
            SDPX.sparse_factor_solve!(destination, factor, rhs)
            destination
        end
        tasks = map(rhs_values) do rhs
            Base.Threads.@spawn begin
                destination = zeros(4)
                SDPX.sparse_factor_solve!(destination, factor, rhs)
                destination
            end
        end
        parallel = fetch.(tasks)
        for index in eachindex(serial)
            @test parallel[index] ≈ serial[index] atol=1e-12
        end
    end

    @testset "frozen CSC assembly map" begin
        G = sparse([1.0 2 0; 0 3 4; 5 0 6])
        probe = transpose(G) * G
        storage = SDPX.freeze_sparse_csc(
            SDPX.sparse_lower_csc(probe);
            provider=SDPX.GenericSparseProvider(Float64),
        )
        map = SDPX.sparse_gram_assembly_map(G, storage)
        old_colptr, old_rowval = copy(storage.matrix.colptr), copy(storage.matrix.rowval)
        SDPX.assemble_sparse_gram!(storage, map, ones(3); regularization=0.1)
        @test storage.matrix.colptr == old_colptr
        @test storage.matrix.rowval == old_rowval
        @test Matrix(Symmetric(storage.matrix, :L)) ≈ Matrix(probe + 0.1I)
    end

    @testset "BigFloat ownership and precision" begin
        setprecision(256) do
            A = sparse(BigFloat[4 1 0; 1 3 1; 0 1 2])
            factor = SDPX.sparse_factor(
                A;
                provider=SDPX.GenericSparseProvider(BigFloat),
            )
            @test all(precision(value) == 256 for value in factor.nzval)
            @test length(unique(objectid.(factor.nzval))) == length(factor.nzval)
            @test all(precision(value) == 256 for value in factor.numeric_work)
            @test length(unique(objectid.(factor.numeric_work))) == length(factor.numeric_work)
            minimum_id = objectid(factor.minimum_diagonal)
            rhs = SDPX.alloc_zeros(BigFloat, 3)
            rhs .= BigFloat[1, 2, 3]
            solution = SDPX.alloc_zeros(BigFloat, 3)
            SDPX.sparse_factor_solve!(solution, factor, rhs)
            @test maximum(abs, A * solution - rhs) == 0

            # Failed numeric attempts retain the factor's owned MPFR scalar
            # at the same precision rather than assigning an ambient-precision
            # `zero(BigFloat)` into the diagnostic state.
            failed = copy(A)
            failed.nzval[failed.colptr[1]] = BigFloat(-1)
            SDPX.numeric_factorize!(factor, failed)
            @test !issuccess(factor)
            @test factor.status === :failed
            @test factor.factorization_attempts == 2
            @test factor.factorization_successes == 1
            @test factor.factorization_failures == 1
            @test precision(factor.minimum_diagonal) == 256
            @test objectid(factor.minimum_diagonal) == minimum_id

            # A factor owns a fixed MPFR precision.  A same-pattern numeric
            # matrix at another precision must fail closed rather than
            # mutating that ownership contract.
            A128 = setprecision(128) do
                sparse(BigFloat[4 1 0; 1 3 1; 0 1 2])
            end
            @test precision(first(A128.nzval)) == 128
            @test_throws ArgumentError SDPX.numeric_factorize!(factor, A128)
            @test factor.precision_bits == 256
            @test factor.factorization_attempts == 2
            @test factor.factorization_successes == 1
            @test factor.factorization_failures == 1
            SDPX.numeric_factorize!(factor, A)
            @test issuccess(factor)
            @test factor.factorization_attempts == 3
            @test factor.factorization_successes == 2
            @test factor.factorization_failures == 1
            @test factor.factorization_attempts ==
                  factor.factorization_successes + factor.factorization_failures
        end
    end

    @testset "fill ladder and explicit storage policy" begin
        band = sparse(1.0I, 12, 12) + sparse(2:12, 1:11, ones(11), 12, 12)
        arrow = sparse(ones(12, 1) * ones(1, 12)) + sparse(1.0I, 12, 12)
        band_diag = SDPX.sparse_factor_diagnostics(
            SDPX.sparse_factor(sparse(Symmetric(band, :L)); provider=SDPX.GenericSparseProvider(Float64)),
        )
        arrow_diag = SDPX.sparse_factor_diagnostics(
            SDPX.sparse_factor(sparse(Symmetric(arrow, :L)); provider=SDPX.GenericSparseProvider(Float64)),
        )
        @test band_diag.fill_ratio > 0
        @test arrow_diag.fill_ratio > 0
        @test SDPX.select_lp_formulation(
            dimension=12,
            nonzeros=30,
            equalities=0,
            arithmetic=BigFloat,
            storage=:sparse,
        ) === :sparse_normal
        @test SDPX.select_lp_formulation(
            dimension=800,
            nonzeros=4_800,
            equalities=0,
            arithmetic=BigFloat,
            storage=:auto,
        ) === :dense_lu
        @test_throws ArgumentError SDPX.select_lp_formulation(
            dimension=800,
            nonzeros=4_800,
            equalities=2,
            arithmetic=Float64,
            storage=:sparse,
        )
        # Explicit Float64 sparse normal equations use the same first-class
        # frozen storage as the generic providers.  Values are refactorized in
        # place while CSC identity and symbolic counts stay fixed.
        G = sparse([1.0 2 0; 0 3 4; 5 0 6])
        B = spzeros(Float64, size(G, 2), 0)
        system = SDPX.lp_sparse_candidate(G, B, Float64; storage=:sparse)
        @test system !== nothing
        @test system.storage isa SDPX.SparseKKTStorage{Float64}
        colptr = system.storage.matrix.colptr
        rowval = system.storage.matrix.rowval
        @test SDPX.lp_sparse_factor!(system, ones(3), 0.1)
        @test SDPX.lp_sparse_factor!(system, fill(2.0, 3), 0.1)
        @test system.storage.matrix.colptr === colptr
        @test system.storage.matrix.rowval === rowval
        backend_stats = SDPX.statistics(system.backend)
        @test backend_stats.analyses == 1
        @test backend_stats.factorizations == 2
        @test backend_stats.reused == 1

        # Planner storage is sourced from the explicit SolverOptions policy,
        # not merely from the ingest-time classification storage.
        c = [1.0, 2.0, 3.0]
        h = [-1.0, -1.0, -1.0]
        problem = SDPX.linear_program(c, G, h; sparse=true, verbosity=0)
        dense_plan = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{Float64}(sparse=:dense, verbosity=0),
        )
        sparse_plan = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{Float64}(sparse=:sparse, verbosity=0),
        )
        @test dense_plan.storage_plan.requested === :dense
        @test dense_plan.storage_plan.storage === :dense
        @test sparse_plan.storage_plan.requested === :sparse
        @test sparse_plan.storage_plan.storage === :sparse
    end

    # The dedicated LP loop is intentionally opt-in here: first-time Julia
    # compilation dominates a tiny test process.  The production A/B check
    # uses identical data/options and changes only explicit storage policy.
    if get(ENV, "SDPX_RUN_LP_SPARSE_INTEGRATION", "0") == "1"
        rng = MersenneTwister(9)
        variables, rows = 6, 18
        G = sprandn(rng, rows, variables, 0.25)
        G += sparse(1.0I, rows, variables)
        h = fill(-1.0, rows)
        c = collect(1.0:variables)
        dense_problem = SDPX.linear_program(c, G, h; sparse=false, verbosity=0)
        sparse_problem = SDPX.linear_program(c, G, h; sparse=true, verbosity=0)
        dense_options = SDPX.SolverOptions{Float64}(
            sparse=:dense, iter_max=8, verbosity=0,
        )
        sparse_options = SDPX.SolverOptions{Float64}(
            sparse=:sparse, iter_max=8, verbosity=0,
        )
        dense_result = SDPX.solve!(dense_problem, dense_options)
        sparse_result = SDPX.solve!(sparse_problem, sparse_options)
        @test dense_result.status == sparse_result.status
        @test sparse_result.termination.executed.lp_formulation === :sparse_normal
        @test sparse_result.termination.executed.planned_storage === :sparse
        @test sparse_result.termination.executed.executed_storage === :sparse
        sparse_backend = sparse_result.termination.sparse_schur_backend
        @test sparse_backend.backend === :cholmod_sparse_cholesky
        @test sparse_backend.provider === :cholmod
        @test sparse_backend.arithmetic === Float64
        @test sparse_backend.analyses == 1
        @test sparse_backend.factorizations > 1
        @test sparse_backend.reused > 0
        @test sparse_backend.reuse_ratio > 0
        @test sparse_backend.dimension == variables
        @test sparse_backend.input_nnz > 0
        @test sparse_backend.factor_nnz > 0
        @test sparse_backend.fill_ratio > 0
        @test sparse_backend.ordering === :cholmod_amd
        @test sparse_result.pObj ≈ dense_result.pObj atol=1e-12
        @test sparse_result.dObj ≈ dense_result.dObj atol=1e-12
    end

    # Optional arithmetic extension: when explicitly requested, exercise a
    # true sparse Float64x2 factor.  An enabled request is a test contract:
    # import/algorithm errors must fail loudly rather than being converted to
    # an informational skip.
    if get(ENV, "SDPX_RUN_MULTIFLOAT_SPARSE", "0") == "1"
        @eval import MultiFloats
        # One arithmetic-independent symbolic object is shared by x2/x3/x4;
        # each provider owns only its numeric CSC values and solve workspace.
        symbolic = SDPX.analyze_sparse_pattern(
            sparse([4.0 1 0; 1 3 1; 0 1 2]),
        )
        for T in (
            MultiFloats.Float64x2,
            MultiFloats.Float64x3,
            MultiFloats.Float64x4,
        )
            A = sparse(T[4 1 0; 1 3 1; 0 1 2])
            factor = SDPX.sparse_factor(
                A;
                provider=SDPX.GenericSparseProvider(T),
                symbolic=symbolic,
            )
            rhs = T[1, 2, 3]
            solution = zeros(T, 3)
            SDPX.sparse_factor_solve!(solution, factor, rhs)
            @test A * solution ≈ rhs
            @test factor.symbolic.pattern_signature == symbolic.pattern_signature
        end
    end
end
