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
end
