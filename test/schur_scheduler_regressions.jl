using LinearAlgebra
using MultiFloats: Float64x4
using SDPX
using SparseArrays
using Test

@testset "Schur scheduler regressions" begin
    @testset "dense task-launch crossover reflects arithmetic cost" begin
        @test !SDPX._dense_schur_threading_profitable(
            Float64,
            96,
            fill(6, 8),
        )
        @test SDPX._dense_schur_threading_profitable(
            Float64,
            512,
            fill(10, 8),
        )
        @test SDPX._dense_schur_threading_profitable(
            Float64x4,
            48,
            fill(4, 8),
        )
    end

    @testset "block-loop crossover accounts for block dimensions" begin
        @test !SDPX._block_loop_threading_profitable(
            Float64,
            fill(8, 32),
            8,
        )
        @test SDPX._block_loop_threading_profitable(
            Float64,
            [
                52, 23, 34, 40, 70, 33, 37, 26,
                44, 74, 73, 67, 58, 64, 41, 37,
                34, 38, 43, 34, 34, 43, 51, 23,
                34, 40, 70, 33, 37, 26, 44, 74,
            ],
            8,
        )
        @test SDPX._block_loop_threading_profitable(
            Float64,
            fill(2, 256),
            8,
        )
        @test SDPX._block_loop_threading_profitable(
            Float64x4,
            fill(24, 8),
            4,
        )
        @test !SDPX._block_loop_threading_profitable(
            Float64,
            fill(64, 8),
            1,
        )
    end

    @testset "partial reduction preserves requested storage triangle" begin
        for T in (Float64, Float64x4)
            dimension = 11
            partials = [
                reshape(
                    T.(
                        block .+
                        collect(1:(dimension * dimension)),
                    ),
                    dimension,
                    dimension,
                )
                for block in 1:3
            ]
            expected = similar(first(partials))
            @inbounds for index in eachindex(expected)
                expected[index] =
                    partials[1][index] +
                    partials[2][index] +
                    partials[3][index]
            end

            sentinel = T(-17)
            triangular = fill(sentinel, dimension, dimension)
            SDPX._reduce_schur_column_range!(
                triangular,
                partials,
                1,
                dimension,
                true,
            )
            @inbounds for column in 1:dimension, row in 1:dimension
                if row >= column
                    @test triangular[row, column] == expected[row, column]
                else
                    @test triangular[row, column] == sentinel
                end
            end

            full = fill(sentinel, dimension, dimension)
            SDPX._reduce_schur_column_range!(
                full,
                partials,
                1,
                dimension,
                false,
            )
            @test full == expected
        end
    end

    @testset "reduction task count avoids synchronization-only work" begin
        @test SDPX._schur_reduction_task_count(
            Float64,
            8,
            4_656,
            8,
        ) == 1
        @test 1 < SDPX._schur_reduction_task_count(
            Float64,
            8,
            131_328,
            8,
        ) <= 8
        @test 1 < SDPX._schur_reduction_task_count(
            Float64x4,
            8,
            16_384,
            8,
        ) <= 8
        @test !SDPX.thread_safe_arithmetic(BigFloat)
    end

    @testset "small Float64 assembly takes the stable serial crossover" begin
        T = Float64
        blocks = 4
        variables = 24
        dimension = 3
        coefficients = [
            reshape(
                T.(
                    block .+
                    collect(1:(variables * dimension * dimension)),
                ),
                variables,
                dimension,
                dimension,
            )
            for block in 1:blocks
        ]
        for panel in coefficients
            @inbounds for variable in 1:variables
                for column in 1:dimension, row in 1:(column - 1)
                    panel[variable, column, row] =
                        panel[variable, row, column]
                end
            end
        end
        problem = SDPX.ingest(
            ones(T, variables),
            coefficients,
            [zeros(T, dimension, dimension) for _ in 1:blocks],
            zeros(T, variables, 0),
            T[];
            sparse=false,
            validate=false,
            symmetrize=false,
            verbosity=0,
        )
        identity_blocks = [
            Matrix{T}(I, dimension, dimension)
            for _ in 1:blocks
        ]
        serial = SDPX.Workspace(problem; thread_count=1)
        scheduled = SDPX.Workspace(
            problem;
            thread_count=min(4, Threads.nthreads()),
        )
        @test SDPX.factor_blocks!(serial, identity_blocks, identity_blocks)
        @test SDPX.factor_blocks!(
            scheduled,
            identity_blocks,
            identity_blocks,
        )
        SDPX.schur_build!(
            serial,
            problem,
            problem.cons,
            identity_blocks,
            identity_blocks,
        )
        SDPX.threaded_schur_build!(
            scheduled,
            problem,
            problem.cons,
            identity_blocks,
            identity_blocks,
        )
        @test scheduled.S == serial.S
    end

    @testset "threaded Float64x4 triangular storage matches serial Schur" begin
        T = Float64x4
        blocks = 8
        variables = 48
        dimension = 4
        coefficients = Vector{Array{T,3}}(undef, blocks)
        for block in 1:blocks
            panel = zeros(T, variables, dimension, dimension)
            @inbounds for variable in 1:variables
                for column in 1:dimension, row in 1:column
                    value =
                        T(
                            mod(
                                17variable + 11block + 5row + 3column,
                                29,
                            ) - 14,
                        ) / T(19)
                    panel[variable, row, column] = value
                    panel[variable, column, row] = value
                end
            end
            coefficients[block] = panel
        end
        problem = SDPX.ingest(
            ones(T, variables),
            coefficients,
            [zeros(T, dimension, dimension) for _ in 1:blocks],
            zeros(T, variables, 0),
            T[];
            sparse=false,
            validate=false,
            symmetrize=false,
            verbosity=0,
        )
        identity_blocks = [
            Matrix{T}(I, dimension, dimension)
            for _ in 1:blocks
        ]
        serial = SDPX.Workspace(problem; thread_count=1)
        triangular = SDPX.Workspace(
            problem;
            thread_count=min(4, Threads.nthreads()),
            extended_precision_blas=:on,
            extended_precision_memory_fraction=0.05,
        )
        @test triangular.extended_precision.lower_only
        @test SDPX.factor_blocks!(serial, identity_blocks, identity_blocks)
        @test SDPX.factor_blocks!(
            triangular,
            identity_blocks,
            identity_blocks,
        )
        SDPX.schur_build!(
            serial,
            problem,
            problem.cons,
            identity_blocks,
            identity_blocks,
        )
        SDPX.threaded_schur_build!(
            triangular,
            problem,
            problem.cons,
            identity_blocks,
            identity_blocks,
        )
        materialized = similar(serial.S)
        SDPX.materialize_schur!(materialized, triangular)
        relative_error =
            maximum(abs, materialized - serial.S) /
            max(maximum(abs, serial.S), one(T))
        @test relative_error < T(1e-55)
    end

    @testset "BLAS width is restored when Schur threading is declined" begin
        # `_schur_parallel_bins` caps task-local `m x m` accumulators against
        # free memory, so a large `m` yields one bin no matter how many threads
        # were requested and the assembly falls back to a single serial
        # `syrk!`. Serializing BLAS on top of that would leave the phase with
        # no parallelism from either source; `schur_threading_engaged` is what
        # `newton_step!` consults to avoid it. Measured on this path at
        # `m = 2000` and `m = 3500`, restoring the BLAS width is 2.5x-2.7x
        # faster and bit-identical.
        blocks, side, m = 4, 6, 40
        coefficients = [zeros(m, side, side) for _ in 1:blocks]
        for l in 1:blocks, i in 1:m
            entry = float(i + l)
            coefficients[l][i, 1, 1] = entry
            coefficients[l][i, side, side] = entry
        end
        problem = SDPX.ingest(
            ones(m),
            coefficients,
            [Matrix{Float64}(1.0I, side, side) for _ in 1:blocks],
            zeros(m, 0),
            Float64[];
            verbosity=0,
        )
        identity_blocks = [Matrix{Float64}(1.0I, k, k) for k in problem.dims.k]

        # One bin is exactly the condition under which the threaded path
        # declines, whatever produced it.
        single = SDPX.Workspace(problem; thread_count=1)
        @test length(single.schur_bins) <= 1
        @test !SDPX.schur_threading_engaged(single, problem, problem.cons)

        # The predicate must agree with what `threaded_schur_build!` actually
        # does: when it reports "engaged", the threaded path really runs, and
        # either way the result matches the serial reference exactly.
        parallel = SDPX.Workspace(problem)
        @test SDPX.factor_blocks!(single, identity_blocks, identity_blocks)
        @test SDPX.factor_blocks!(parallel, identity_blocks, identity_blocks)
        SDPX.schur_build!(
            single,
            problem,
            problem.cons,
            identity_blocks,
            identity_blocks,
        )
        SDPX.threaded_schur_build!(
            parallel,
            problem,
            problem.cons,
            identity_blocks,
            identity_blocks,
        )
        reference = similar(single.S)
        SDPX.materialize_schur!(reference, single)
        materialized = similar(parallel.S)
        SDPX.materialize_schur!(materialized, parallel)
        @test materialized == reference
    end

    @testset "BLAS width is widened only for the declined dense path" begin
        # Widening is worth it only when the declined fallback is one large
        # `syrk!`. With sparse constraints the fallback is many small per-block
        # operations instead, and BLAS threads cost more than they return:
        # on Task_Low08 (m = 6119, L = 32) widening took Schur assembly from
        # 8.26 s to 15.23 s. The gate must therefore look at the constraint
        # storage, not only at whether threading was declined.
        blocks, side, m = 3, 5, 30
        dense_coefficients = [zeros(m, side, side) for _ in 1:blocks]
        sparse_coefficients =
            [Vector{SparseMatrixCSC{Float64,Int}}(undef, m) for _ in 1:blocks]
        for l in 1:blocks, i in 1:m
            entry = float(i + l)
            dense_coefficients[l][i, 1, 1] = entry
            dense_coefficients[l][i, side, side] = entry
            sparse_coefficients[l][i] =
                sparse([1, side], [1, side], [entry, entry], side, side)
        end
        constants = [Matrix{Float64}(1.0I, side, side) for _ in 1:blocks]

        dense = SDPX.ingest(ones(m), dense_coefficients, constants,
            zeros(m, 0), Float64[]; sparse=false, verbosity=0)
        sparse_problem = SDPX.ingest(ones(m), sparse_coefficients, constants,
            zeros(m, 0), Float64[]; sparse=true, verbosity=0)
        @test dense.cons isa SDPX.DenseCons{Float64}
        @test sparse_problem.cons isa SDPX.SparseCons{Float64}

        # `thread_count = 1` forces the decline deterministically, without
        # depending on how much memory happens to be free.
        dense_ws = SDPX.Workspace(dense; thread_count=1)
        sparse_ws = SDPX.Workspace(sparse_problem; thread_count=1)
        @test !SDPX.schur_threading_engaged(dense_ws, dense, dense.cons)
        @test !SDPX.schur_threading_engaged(
            sparse_ws,
            sparse_problem,
            sparse_problem.cons,
        )

        # Declined + dense: take the full width. Declined + sparse: stay
        # serialized. Engaged: stay serialized regardless of storage.
        @test SDPX.schur_blas_threads(dense_ws, dense, dense.cons, 1, 8) == 8
        @test SDPX.schur_blas_threads(
            sparse_ws,
            sparse_problem,
            sparse_problem.cons,
            1,
            8,
        ) == 1
    end

    @testset "dense column-owner Schur mode" begin
        @test SDPX._dense_lower_owner_boundaries(2, 2) == [1, 2, 3]
        @test SDPX._dense_lower_owner_boundaries(8, 8) == collect(1:9)
        @test all(diff(SDPX._dense_lower_owner_boundaries(17, 6)) .> 0)

        function dense_owner_problem(::Type{T}, m, blocks, side) where {T}
            coefficients = [zeros(T, m, side, side) for _ in 1:blocks]
            for l in 1:blocks, i in 1:m
                coefficients[l][i, 1, 1] = T(i + l) / T(17)
                coefficients[l][i, side, side] = T(i * l) / T(23)
            end
            constants = [
                Matrix{T}(one(T) * I, side, side)
                for _ in 1:blocks
            ]
            return SDPX.ingest(
                ones(T, m),
                coefficients,
                constants,
                zeros(T, m, 0),
                T[];
                sparse=false,
                verbosity=0,
            )
        end

        threads = min(4, Threads.nthreads())
        if threads > 1
            for T in (Float64,)
                problem = dense_owner_problem(T, 300, 8, 6)
                identity_blocks = [
                    Matrix{T}(one(T) * I, 6, 6)
                    for _ in 1:problem.dims.L
                ]
                owner = SDPX.Workspace(problem; thread_count=threads)

                # Eligible mode: no per-bin m×m partials, but more than one
                # deterministic owner range.
                @test owner.dense_schur_owner
                @test isempty(owner.Spartial)
                @test length(owner.schur_bins) > 1
                @test length(owner.schur_column_boundaries) == threads + 1
                @test owner.schur_column_boundaries[1] == 1
                @test owner.schur_column_boundaries[end] == problem.dims.m + 1
                @test all(diff(owner.schur_column_boundaries) .> 0)
                @test SDPX.schur_threading_engaged(owner, problem, problem.cons)

                serial = SDPX.Workspace(problem; thread_count=1)
                @test !serial.dense_schur_owner
                @test SDPX.factor_blocks!(serial, identity_blocks, identity_blocks)
                @test SDPX.factor_blocks!(owner, identity_blocks, identity_blocks)
                SDPX.schur_build!(
                    serial,
                    problem,
                    problem.cons,
                    identity_blocks,
                    identity_blocks,
                )
                SDPX.threaded_schur_build!(
                    owner,
                    problem,
                    problem.cons,
                    identity_blocks,
                    identity_blocks,
                )
                reference = similar(serial.S)
                SDPX.materialize_schur!(reference, serial)
                materialized = similar(owner.S)
                SDPX.materialize_schur!(materialized, owner)
                tolerance = T(1e-12)
                @test maximum(abs, materialized - reference) /
                      max(maximum(abs, reference), one(T)) < tolerance

                # Fixed thread count is deterministic: a second build on the
                # same workspace reproduces the lower triangle bit-for-bit.
                first = copy(owner.S)
                SDPX.threaded_schur_build!(
                    owner,
                    problem,
                    problem.cons,
                    identity_blocks,
                    identity_blocks,
                )
                @test owner.S == first

                # Upper triangle stays untouched until materialization.
                sentinel = T(-17)
                fill!(owner.S, sentinel)
                SDPX.threaded_schur_build!(
                    owner,
                    problem,
                    problem.cons,
                    identity_blocks,
                    identity_blocks,
                )
                @test all(
                    owner.S[row, column] == sentinel
                    for column in 1:problem.dims.m
                    for row in 1:(column - 1)
                )

                # The accumulator memory cap no longer collapses this mode:
                # with an absurdly small budget the generic cap returns one
                # bin, while the workspace still owns every worker a range.
                @test SDPX._schur_parallel_bins(
                    T,
                    problem.dims.m,
                    problem.dims.L,
                    threads;
                    free_memory_bytes=1,
                ) == 1
                @test length(owner.schur_bins) ==
                      min(threads, problem.dims.L)
            end
        end

        # Small/unprofitable dense problems stay serial, but must not allocate
        # useless eligible partials.
        if threads > 1
            small = dense_owner_problem(Float64, 24, 4, 4)
            small_ws = SDPX.Workspace(small; thread_count=threads)
            @test small_ws.dense_schur_owner
            @test isempty(small_ws.Spartial)
            @test !SDPX.schur_threading_engaged(small_ws, small, small.cons)
            @test SDPX._dense_schur_threading_profitable(
                Float64,
                small.dims.m,
                small.dims.k,
            ) == false

            # Existing fixed-extended partial path is unchanged.
            wide = dense_owner_problem(Float64x4, 60, 8, 4)
            wide_ws = SDPX.Workspace(wide; thread_count=threads)
            @test !wide_ws.dense_schur_owner
            @test !isempty(wide_ws.Spartial)
            @test length(wide_ws.schur_column_boundaries) == 0

            # Estimates reflect the removed per-bin m² partials: Dense
            # Float64 stores no partial at any thread count, so the
            # multi-thread partial term equals the single-thread one.
            eligible = dense_owner_problem(Float64, 120, 8, 5)
            @test SDPX.estimate_dense_workspace_bytes(eligible, threads) ==
                  SDPX.estimate_dense_workspace_bytes(eligible, 1)
            @test SDPX.dense_workspace_floor_bytes(
                Float64,
                eligible.dims.m,
                eligible.dims.n,
                eligible.dims.L,
                threads,
            ) == SDPX.dense_workspace_floor_bytes(
                Float64,
                eligible.dims.m,
                eligible.dims.n,
                eligible.dims.L,
                1,
            )
            report = SDPX.schur_bin_report(
                Float64,
                eligible.dims.m,
                eligible.dims.L,
                threads,
                dense_owner=true,
            )
            @test report.assembly_mode === :column_owned
            @test report.owner_tasks == min(threads, eligible.dims.m)
            @test report.selected_bins == report.requested_bins
            @test !report.capped
            @test report.total_bytes == 0
            @test report.would_have_been_bytes > 0
        end
        @test SDPX.dense_workspace_floor_bytes(
            Float64,
            4_000_000_000,
            0,
            32,
            max(threads, 1),
        ) == typemax(Int)
    end
end
