using LinearAlgebra
using Random
using SDPX
using Test

@testset "BigFloat kernel regressions" begin
    setprecision(BigFloat, 256) do
        rng = MersenneTwister(0x5d_50_58)

        @testset "matrix products preserve inputs and independent storage" begin
            A = BigFloat.(randn(rng, 7, 5))
            B = BigFloat.(randn(rng, 5, 6))
            A_original = deepcopy(A)
            B_original = deepcopy(B)
            C = zeros(BigFloat, 7, 6)
            SDPX.kmul!(C, A, B, big"1.25", big"-0.2")
            @test C ≈ big"1.25" .* (A_original * B_original) rtol=big"1e-60"
            @test A == A_original
            @test B == B_original
            @test !(C[1] === C[2])

            owned_C = SDPX.alloc_zeros(BigFloat, 7, 6)
            @inbounds for index in eachindex(owned_C)
                owned_C[index] = BigFloat(index) / BigFloat(100)
            end
            owned_initial = deepcopy(owned_C)
            first_storage = owned_C[1]
            second_storage = owned_C[2]
            SDPX.kmul_owned!(
                owned_C,
                A,
                B,
                big"1.25",
                big"-0.2",
            )
            owned_expected =
                big"1.25" .* (A_original * B_original) .-
                big"0.2" .* owned_initial
            @test owned_C ≈ owned_expected rtol=big"1e-60"
            @test owned_C[1] === first_storage
            @test owned_C[2] === second_storage
            @test @allocated(
                SDPX.kmul_owned!(
                    owned_C,
                    A,
                    B,
                    big"1.25",
                    big"-0.2",
                )
            ) <= 4096

            panel = BigFloat.(randn(rng, 11, 6))
            initial_generator = BigFloat.(randn(rng, 6, 6))
            initial =
                (initial_generator + transpose(initial_generator)) /
                BigFloat(2)
            gram = copy(initial)
            SDPX.ksyrk!(gram, panel, big"0.75", big"-0.125")
            expected =
                big"0.75" .* (transpose(panel) * panel) .-
                big"0.125" .* initial
            @test gram ≈ expected rtol=big"1e-60"
            @test !(gram[1, 2] === gram[2, 1])
        end

        @testset "factor and solve agree with generic linear algebra" begin
            dimension = 9
            generator = BigFloat.(randn(rng, dimension, dimension))
            matrix =
                generator * transpose(generator) +
                BigFloat(dimension) * I
            factor = deepcopy(matrix)
            @test SDPX.kchol!(factor)
            lower = Matrix(LowerTriangular(factor))
            @test lower * transpose(lower) ≈ matrix rtol=big"1e-60"

            matrix_rhs = BigFloat.(randn(rng, dimension, 4))
            expected_matrix = matrix \ matrix_rhs
            actual_matrix = deepcopy(matrix_rhs)
            SDPX.kcholsolve!(factor, actual_matrix)
            @test actual_matrix ≈ expected_matrix rtol=big"1e-60"

            vector_rhs = BigFloat.(randn(rng, dimension))
            expected_vector = matrix \ vector_rhs
            actual_vector = deepcopy(vector_rhs)
            SDPX.kcholsolve!(factor, actual_vector)
            @test actual_vector ≈ expected_vector rtol=big"1e-60"

            lower_rhs = BigFloat.(randn(rng, dimension))
            expected_lower = LowerTriangular(factor) \ lower_rhs
            actual_lower = copy(lower_rhs)
            SDPX.ktrsv_lower!(factor, actual_lower)
            @test actual_lower ≈ expected_lower rtol=big"1e-60"
            @test lower_rhs != actual_lower

            transpose_rhs = BigFloat.(randn(rng, dimension))
            expected_transpose =
                UpperTriangular(transpose(factor)) \ transpose_rhs
            actual_transpose = copy(transpose_rhs)
            SDPX.ktrsv_transpose!(factor, actual_transpose)
            @test actual_transpose ≈ expected_transpose rtol=big"1e-60"
            @test transpose_rhs != actual_transpose

            marker = SDPX.BigFloatCholeskyFactor(factor)
            @test marker.L === factor
        end

        @testset "alias-safe vector updates and trial construction" begin
            x = BigFloat.(randn(rng, 128))
            y = zeros(BigFloat, 128)
            y_original = deepcopy(y)
            expected = big"1.5" .* x .- big"0.25" .* y_original
            SDPX.kaxpby!(big"1.5", x, big"-0.25", y)
            @test y ≈ expected rtol=big"1e-60"
            @test !(y[1] === y[2])

            owned_y = SDPX.alloc_zeros(BigFloat, length(x))
            @inbounds for index in eachindex(owned_y)
                owned_y[index] = BigFloat(index) / BigFloat(100)
            end
            owned_y_initial = deepcopy(owned_y)
            first_storage = owned_y[1]
            second_storage = owned_y[2]
            SDPX.kaxpby_owned!(
                big"1.5",
                x,
                big"-0.25",
                owned_y,
            )
            @test owned_y ≈
                  big"1.5" .* x .- big"0.25" .* owned_y_initial rtol=big"1e-60"
            @test owned_y[1] === first_storage
            @test owned_y[2] === second_storage
            @test @allocated(
                SDPX.kaxpby_owned!(
                    big"1.5",
                    x,
                    big"-0.25",
                    owned_y,
                )
            ) <= 2048

            matrix = BigFloat.(randn(rng, 8, 8))
            direction = BigFloat.(randn(rng, 8, 8))
            matrix_original = deepcopy(matrix)
            direction_original = deepcopy(direction)
            trial = zeros(BigFloat, 8, 8)
            SDPX.trial_combine!(trial, matrix, big"0.125", direction)
            @test trial ≈ matrix .+ big"0.125" .* direction rtol=big"1e-60"
            @test matrix == matrix_original
            @test direction == direction_original
            @test !(trial[1] === trial[2])
        end

        @testset "allocation-light infinity norm" begin
            values = BigFloat[1, -7, 3, -2]
            @test SDPX.knrmInf(values) == 7
            @test SDPX.knrmInf(BigFloat[]) == 0
            @test isnan(SDPX.knrmInf(BigFloat[1, NaN, 2]))
            blocks = [BigFloat[-2, 4], BigFloat[3, -9]]
            @test SDPX.knrmInf(blocks) == 9

            large_values = BigFloat.(randn(rng, 1024))
            SDPX.knrmInf(large_values)
            @test @allocated(SDPX.knrmInf(large_values)) <= 4096
        end

        @testset "owned workspace reset reuses independent MPFR storage" begin
            workspace = SDPX.alloc_zeros(BigFloat, 64)
            @inbounds for index in eachindex(workspace)
                workspace[index] = BigFloat(index)
            end
            first_entry = workspace[1]
            second_entry = workspace[2]
            SDPX.zero_owned!(workspace)
            @test all(iszero, workspace)
            @test workspace[1] === first_entry
            @test workspace[2] === second_entry
            @test !(workspace[1] === workspace[2])
            @test @allocated(SDPX.zero_owned!(workspace)) <= 64

            arbitrary = zeros(BigFloat, 4)
            @test arbitrary[1] === arbitrary[2]
            SDPX.zero_distinct!(arbitrary)
            @test !(arbitrary[1] === arbitrary[2])

            source = BigFloat.(randn(rng, 64))
            destination = SDPX.alloc_zeros(BigFloat, 64)
            first_entry = destination[1]
            second_entry = destination[2]
            SDPX.copy_owned!(destination, source)
            @test destination == source
            @test destination[1] === first_entry
            @test destination[2] === second_entry
            @test @allocated(SDPX.copy_owned!(destination, source)) <= 64
        end

        @testset "hot operations avoid intermediate MPFR objects" begin
            x = BigFloat.(randn(rng, 512))
            y = SDPX.alloc_zeros(BigFloat, 512)
            SDPX.kaxpby!(big"1.25", x, big"-0.5", y)
            axpby_allocations =
                @allocated SDPX.kaxpby!(big"1.25", x, big"-0.5", y)
            @test axpby_allocations <= 128_000

            dimension = 16
            generator = BigFloat.(randn(rng, dimension, dimension))
            matrix =
                generator * transpose(generator) +
                BigFloat(dimension) * I
            factor = deepcopy(matrix)
            @test SDPX.kchol!(factor)
            rhs = BigFloat.(randn(rng, dimension, 4))
            SDPX.kcholsolve!(factor, rhs)
            solve_allocations = @allocated SDPX.kcholsolve!(factor, rhs)
            @test solve_allocations <= 128_000

            vector_rhs = BigFloat.(randn(rng, dimension))
            SDPX.ktrsv_lower!(factor, vector_rhs)
            lower_solve_allocations =
                @allocated SDPX.ktrsv_lower!(factor, vector_rhs)
            @test lower_solve_allocations <= 32_000
            SDPX.ktrsv_transpose!(factor, vector_rhs)
            transpose_solve_allocations =
                @allocated SDPX.ktrsv_transpose!(factor, vector_rhs)
            @test transpose_solve_allocations <= 32_000
        end
    end
end
