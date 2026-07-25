using LinearAlgebra
using MultiFloats: Float64x2, Float64x4
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

    @testset "Float64 unchanged and BigFloat serial" begin
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

    @testset "GEMM clamps zero tile sizes" begin
        T = Float64x4
        left = _regression_panel(T, 7, 5)
        right = transpose(_regression_panel(T, 6, 5))
        output = zeros(T, 7, 6)
        EXTENDED_BLAS_REGRESSION.gemm!(
            output,
            left,
            right,
            one(T),
            zero(T),
            EXTENDED_BLAS_REGRESSION.KernelConfig(
                row_tile=0,
                column_tile=0,
                micro_tile=1,
            ),
        )
        reference = left * right
        @test maximum(abs, output - reference) < T(1e-55)
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
