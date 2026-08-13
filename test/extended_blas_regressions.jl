using LinearAlgebra
using MultiFloats: Float64x2, Float64x4
using Random
using SDPX
using Test

const EXTENDED_BLAS_REGRESSION = SDPX.ExtendedPrecisionBLAS

function _regression_panel(
    ::Type{T},
    rows::Int,
    columns::Int,
) where {T}
    panel = Matrix{T}(undef, rows, columns)
    @inbounds for column in 1:columns, row in 1:rows
        panel[row, column] =
            T(sin(0.031 * row + 0.017 * column))
    end
    return panel
end

@testset "extended BLAS regressions" begin
    @testset "Float64x4 fine-grained phase cap" begin
        @test SDPX.fine_grained_block_bins(
            Float64x4,
            1,
            true,
            1_700,
        ) == 1
        @test SDPX.fine_grained_block_bins(
            Float64x4,
            32,
            true,
            1_700,
        ) == 32
        @test SDPX.fine_grained_block_bins(
            Float64x4,
            64,
            true,
            1_700,
        ) == 64
        @test SDPX.fine_grained_block_bins(
            Float64x4,
            128,
            true,
            1_700,
        ) == 32
        @test SDPX.fine_grained_block_bins(
            Float64x4,
            64,
            false,
            1_700,
        ) == 64
        @test SDPX.fine_grained_block_bins(
            Float64x4,
            64,
            true,
            128,
        ) == 64
        @test SDPX.fine_grained_block_bins(
            Float64,
            128,
            true,
            1_700,
        ) == 128
        @test SDPX.fine_grained_block_bins(
            BigFloat,
            128,
            true,
            1_700,
        ) == 128
        @test SDPX.reduced_arrow_worker_count(
            Float64x4,
            64,
            1_700,
            144,
        ) == 64
        @test SDPX.reduced_arrow_worker_count(
            Float64x4,
            128,
            1_700,
            144,
        ) == 64
        @test SDPX.reduced_arrow_worker_count(
            Float64x4,
            128,
            128,
            144,
        ) == 128
        @test SDPX.reduced_arrow_worker_count(
            Float64x4,
            128,
            1_700,
            300,
        ) == 128
        @test SDPX.reduced_arrow_worker_count(
            Float64,
            128,
            1_700,
            144,
        ) == 128
        @test SDPX.reduced_arrow_factor_worker_count(
            Float64x4,
            1,
            144,
        ) == 1
        @test SDPX.reduced_arrow_factor_worker_count(
            Float64x4,
            2,
            144,
        ) == min(2, Threads.nthreads())
        @test SDPX.reduced_arrow_factor_worker_count(
            Float64x4,
            32,
            144,
        ) == min(8, Threads.nthreads())
        @test SDPX.reduced_arrow_factor_worker_count(
            Float64x4,
            32,
            64,
        ) == 1
        @test SDPX.reduced_arrow_factor_worker_count(
            Float64,
            32,
            144,
        ) == 1
        @test SDPX.reduced_arrow_solver_worker_count(
            Float64x4,
            64,
            1_700,
            144,
        ) == 64
        @test SDPX.reduced_arrow_solver_worker_count(
            Float64x4,
            128,
            1_700,
            144,
        ) == 64
        @test SDPX.reduced_arrow_solver_worker_count(
            Float64x4,
            128,
            1_700,
            300,
        ) == 128
        @test SDPX.reduced_arrow_solver_worker_count(
            Float64,
            128,
            1_700,
            144,
        ) == 128
        # A whole-solver 128-to-64 cap must retain the phase-aware 32-bin
        # schedule selected from the original request. Re-selecting from the
        # effective width would expand short phases back to 64 tasks.
        @test min(
            SDPX.fine_grained_block_bins(
                Float64x4,
                128,
                true,
                1_700,
            ),
            SDPX.reduced_arrow_solver_worker_count(
                Float64x4,
                128,
                1_700,
                144,
            ),
        ) == 32
        @test SDPX.fine_grained_block_partition(
            Float64x4,
            true,
            fill(2, 1_700),
            32,
        ) == :contiguous
        @test SDPX.fine_grained_block_partition(
            Float64x4,
            true,
            fill(2, 1_700),
            48,
        ) == :lpt
        @test SDPX.fine_grained_block_partition(
            Float64x4,
            true,
            [2, 3, 2],
            3,
        ) == :lpt
        @test SDPX.fine_grained_block_partition(
            Float64x4,
            false,
            fill(2, 1_700),
            32,
        ) == :lpt
        @test SDPX.fine_grained_block_partition(
            Float64,
            true,
            fill(2, 1_700),
            32,
        ) == :lpt
        @test SDPX.contiguous_partition(7, 3) == [
            [1, 2],
            [3, 4],
            [5, 6, 7],
        ]
    end

    @testset "Float64x4 high-thread tile pressure" begin
        @test EXTENDED_BLAS_REGRESSION._reduced_arrow_kernel_config(
            Float64x4,
            32,
            144,
        ).column_tile == 12
        @test EXTENDED_BLAS_REGRESSION._reduced_arrow_kernel_config(
            Float64x4,
            64,
            144,
        ).column_tile == 8
        @test EXTENDED_BLAS_REGRESSION._reduced_arrow_kernel_config(
            Float64x4,
            64,
            300,
        ).column_tile == 12
        @test EXTENDED_BLAS_REGRESSION._reduced_arrow_kernel_config(
            Float64x4,
            128,
            144,
        ).column_tile == 12
        @test EXTENDED_BLAS_REGRESSION._reduced_arrow_kernel_config(
            Float64,
            64,
            144,
        ).column_tile == 16
        @test EXTENDED_BLAS_REGRESSION._kernel_config(
            Float64x4,
            64,
            144,
        ).column_tile == 12
    end

    @testset "Float64x4 reduced-arrow factor crossover" begin
        factor_rng = MersenneTwister(0x5d0f4)
        source = Float64x4.(randn(factor_rng, 144, 144))
        positive_definite =
            source * transpose(source) +
            Float64x4(2) * Matrix{Float64x4}(I, 144, 144)
        factor = copy(positive_definite)
        @test SDPX.reduced_arrow_cholesky!(
            factor,
            min(8, Threads.nthreads()),
        )
        allocation_factor = copy(positive_definite)
        SDPX.reduced_arrow_cholesky!(allocation_factor, 1)
        copyto!(allocation_factor, positive_definite)
        @test @allocated(
            SDPX.reduced_arrow_cholesky!(allocation_factor, 1),
        ) == 0
        lower = Matrix(LowerTriangular(factor))
        @test maximum(
            abs,
            lower * transpose(lower) - positive_definite,
        ) / maximum(abs, positive_definite) <= Float64x4(1e-60)

        outside_source = Float64x4.(randn(factor_rng, 64, 64))
        outside_positive_definite =
            outside_source * transpose(outside_source) +
            Float64x4(2) * Matrix{Float64x4}(I, 64, 64)
        outside_factor = copy(outside_positive_definite)
        @test SDPX.reduced_arrow_cholesky!(outside_factor, 8)
        outside_lower = Matrix(LowerTriangular(outside_factor))
        @test maximum(
            abs,
            outside_lower * transpose(outside_lower) -
            outside_positive_definite,
        ) / maximum(abs, outside_positive_definite) <= Float64x4(1e-60)

        indefinite = Matrix{Float64x4}(I, 144, 144)
        indefinite[144, 144] = -one(Float64x4)
        @test !SDPX.reduced_arrow_cholesky!(indefinite, 8)
    end

    @testset "requested worker limit and work threshold" begin
        config = EXTENDED_BLAS_REGRESSION.KernelConfig(
            row_tile=32,
            column_tile=8,
            micro_tile=2,
        )
        rows = 48
        columns = 32
        block_count = cld(columns, config.column_tile)
        jobs = block_count * (block_count + 1) ÷ 2
        for requested in (1, 2, 4)
            selected = EXTENDED_BLAS_REGRESSION._syrk_worker_count(
                Float64x4,
                rows,
                columns,
                jobs,
                requested,
            )
            @test selected == min(requested, Threads.nthreads(), jobs)
        end
        @test EXTENDED_BLAS_REGRESSION._syrk_worker_count(
            Float64x4,
            1,
            2,
            1,
            4,
        ) == 1
        @test EXTENDED_BLAS_REGRESSION._syrk_worker_count(
            Float64,
            rows,
            columns,
            jobs,
            4,
        ) == 1
        @test EXTENDED_BLAS_REGRESSION._syrk_worker_count(
            BigFloat,
            rows,
            columns,
            jobs,
            4,
        ) == 1
        @test EXTENDED_BLAS_REGRESSION._syrk_weighted_work(
            Float64x4,
            rows,
            columns,
        ) == 4 *
                 EXTENDED_BLAS_REGRESSION._syrk_weighted_work(
            Float64x2,
            rows,
            columns,
        )
    end

    @testset "Float64x4 worker-count correctness and allocations" begin
        T = Float64x4
        rows = 48
        columns = 32
        panel = _regression_panel(T, rows, columns)
        reference = transpose(panel) * panel
        config = EXTENDED_BLAS_REGRESSION.KernelConfig(
            row_tile=32,
            column_tile=8,
            micro_tile=2,
        )
        baseline = zeros(T, columns, columns)
        for requested in (1, 2, 4)
            output = zeros(T, columns, columns)
            EXTENDED_BLAS_REGRESSION.syrk!(
                output,
                panel,
                one(T),
                zero(T),
                config,
                requested,
            )
            @test maximum(
                abs,
                LowerTriangular(output - reference),
            ) < T(1e-55)
            @test all(iszero, triu(output, 1))
            if requested == 1
                copyto!(baseline, output)
            else
                @test output == baseline
            end

            EXTENDED_BLAS_REGRESSION.zero_triangle!(output)
            EXTENDED_BLAS_REGRESSION.syrk!(
                output,
                panel,
                one(T),
                zero(T),
                config,
                requested,
            )
            allocated = @allocated EXTENDED_BLAS_REGRESSION.syrk!(
                output,
                panel,
                one(T),
                zero(T),
                config,
                requested,
            )
            selected = EXTENDED_BLAS_REGRESSION._syrk_worker_count(
                T,
                rows,
                columns,
                10,
                requested,
            )
            if selected == 1
                @test allocated == 0
            else
                # Compute loops allocate nothing; only the exact number of
                # scheduler tasks is allocated at the call boundary.
                @test allocated < 64 * 1024
            end
        end
    end

    @testset "Float64 unchanged and small BigFloat serial" begin
        float_panel = _regression_panel(Float64, 12, 10)
        float_output = zeros(Float64, 10, 10)
        EXTENDED_BLAS_REGRESSION.syrk!(
            float_output,
            float_panel,
            1.0,
            0.0,
            EXTENDED_BLAS_REGRESSION.KernelConfig(
                row_tile=8,
                column_tile=4,
                micro_tile=2,
            ),
            4,
        )
        @test maximum(
            abs,
            float_output - LowerTriangular(
                transpose(float_panel) * float_panel,
            ),
        ) <= 8eps(Float64)

        setprecision(BigFloat, 192) do
            panel = _regression_panel(BigFloat, 12, 10)
            output = zeros(BigFloat, 10, 10)
            EXTENDED_BLAS_REGRESSION.prepare_storage!(output)
            EXTENDED_BLAS_REGRESSION.syrk!(
                output,
                panel,
                one(BigFloat),
                zero(BigFloat),
                EXTENDED_BLAS_REGRESSION.KernelConfig(
                    row_tile=8,
                    column_tile=4,
                    micro_tile=1,
                ),
                4,
            )
            reference = transpose(panel) * panel
            @test maximum(
                abs,
                LowerTriangular(output - reference),
            ) < big"1e-50"
            @test output[1, 1] !== output[2, 1]
            @test all(iszero, triu(output, 1))
        end
    end

    @testset "BigFloat exclusive-tile threading" begin
        setprecision(BigFloat, 192) do
            uninitialized = Matrix{BigFloat}(undef, 3, 3)
            EXTENDED_BLAS_REGRESSION.prepare_storage!(uninitialized)
            @test all(iszero, uninitialized)
            @test uninitialized[1] !== uninitialized[2]

            rows = 96
            columns = 48
            panel = _regression_panel(BigFloat, rows, columns)
            config = EXTENDED_BLAS_REGRESSION.KernelConfig(
                row_tile=32,
                column_tile=8,
                micro_tile=1,
            )
            serial = SDPX.alloc_zeros(BigFloat, columns, columns)
            threaded = SDPX.alloc_zeros(BigFloat, columns, columns)
            EXTENDED_BLAS_REGRESSION.syrk!(
                serial,
                panel,
                one(BigFloat),
                zero(BigFloat),
                config,
                1,
            )
            requested = min(Threads.nthreads(), 4)
            EXTENDED_BLAS_REGRESSION.syrk!(
                threaded,
                panel,
                one(BigFloat),
                zero(BigFloat),
                config,
                requested,
            )
            @test LowerTriangular(threaded) == LowerTriangular(serial)
            @test all(iszero, triu(threaded, 1))
            selected =
                EXTENDED_BLAS_REGRESSION._syrk_bigfloat_selected_workers(
                    panel,
                    config,
                    requested,
                )
            @test selected <= requested
            @test selected <= Threads.nthreads()
            Threads.nthreads() > 1 && @test selected > 1
            @test threaded[1, 1] !== threaded[2, 1]
            @test threaded[2, 1] !== threaded[2, 2]
        end
    end

    @testset "conservative memory budget" begin
        @test EXTENDED_BLAS_REGRESSION._parse_memory_bytes("1024") == 1_024
        @test EXTENDED_BLAS_REGRESSION._parse_memory_bytes("1 KiB") == 1_024
        @test EXTENDED_BLAS_REGRESSION._parse_memory_bytes("1.5 GiB") ==
              1_610_612_736
        @test EXTENDED_BLAS_REGRESSION._parse_memory_bytes("unlimited") ===
              nothing
        @test EXTENDED_BLAS_REGRESSION._effective_memory_budget(
            1_000,
            600,
        ) == 300
        @test EXTENDED_BLAS_REGRESSION._effective_memory_budget(
            100,
            600,
        ) == 100
        @test EXTENDED_BLAS_REGRESSION._effective_memory_budget(
            100,
            0,
        ) == 0
        @test EXTENDED_BLAS_REGRESSION._memory_budget_from_fraction(
            1_000,
            0.1,
        ) == 100
        @test EXTENDED_BLAS_REGRESSION._memory_budget_from_fraction(
            1_000,
            1.0,
        ) == 500

        features = EXTENDED_BLAS_REGRESSION.CrossoverFeatures(
            rows=8,
            columns=2,
            matrix_dimension=2,
            average_nnz=8.0,
            active_density=1.0,
            expected_schur_density=1.0,
            thread_count=1,
            memory_budget_bytes=1_024,
            sparse_input=false,
        )
        constrained = EXTENDED_BLAS_REGRESSION.choose_crossover(
            Float64x4,
            features;
            mode=:on,
            available_memory_bytes=512,
        )
        @test !constrained.enabled
        @test constrained.reason == :memory_budget
        @test constrained.packing_bytes == 512

        permitted = EXTENDED_BLAS_REGRESSION.choose_crossover(
            Float64x4,
            features;
            mode=:on,
            available_memory_bytes=4_096,
        )
        @test permitted.enabled
        @test permitted.reason == :forced
    end
end
