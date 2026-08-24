using Test
using LinearAlgebra
using SparseArrays

Base.@noinline function _repeat_sparse_coo_contraction(
    matrix::Matrix{Float64},
    coo::SDPX.SparseBlockCOO{Float64},
    repetitions::Int,
)
    accumulator = 0.0
    @inbounds for _ in 1:repetitions
        accumulator += SDPX._dot_dense_coo(matrix, coo, 1)
    end
    return accumulator
end

@testset "symmetric sparse-Schur COO compression" begin
    symmetric_coefficients = SparseMatrixCSC{Float64,Int}[
        sparse([
            0.0 2.0 0.0 0.0
            2.0 0.0 -3.0 0.0
            0.0 -3.0 0.0 0.0
            0.0 0.0 0.0 0.0
        ]),
        sparse([
            1.0 0.0 4.0 0.0
            0.0 -2.0 0.0 0.0
            4.0 0.0 0.0 5.0
            0.0 0.0 5.0 3.0
        ]),
    ]
    coo = SDPX.build_block_coo(symmetric_coefficients, [1, 2], 4)

    # The first coefficient has two independent off-diagonal entries. The
    # second has three diagonal and two independent off-diagonal entries.
    @test length(coo.val) == 7
    @test coo.ptr == Int32[1, 3, 8]
    @test count(<(0), coo.lin) == 4
    @test all(
        coo.lin[index] < 0 ? coo.row[index] < coo.col[index] : true
        for index in eachindex(coo.lin)
    )

    matrix = reshape(Float64.(1:16), 4, 4)
    left = reshape(Float64.(1:16) ./ 7, 4, 4)
    right = reshape(Float64.(17:32) ./ 11, 4, 4)
    destination = zeros(4, 4)
    for (position, coefficient) in pairs(symmetric_coefficients)
        @test SDPX._dot_dense_coo(matrix, coo, position) ≈
              sum(matrix .* Matrix(coefficient))
        SDPX._two_sided_coo_product!(
            destination,
            left,
            coo,
            position,
            right,
        )
        @test destination ≈ left * Matrix(coefficient) * right
    end

    _repeat_sparse_coo_contraction(matrix, coo, 10_000)
    @test @allocated(
        _repeat_sparse_coo_contraction(matrix, coo, 10_000)
    ) <= 256
    SDPX._two_sided_coo_product!(destination, left, coo, 1, right)
    @test @allocated(
        SDPX._two_sided_coo_product!(destination, left, coo, 1, right)
    ) <= 256

    empty_coo = SDPX.build_block_coo([spzeros(4, 4)], [1], 4)
    @test empty_coo.ptr == Int32[1, 1]
    fill!(destination, 1.0)
    SDPX._two_sided_coo_product!(destination, left, empty_coo, 1, right)
    @test iszero(SDPX._dot_dense_coo(matrix, empty_coo, 1))
    @test iszero(norm(destination, Inf))
end

@testset "nonsymmetric COO keeps the full fallback" begin
    coefficient = sparse([
        0.0 2.0 0.0
        0.0 0.0 -4.0
        5.0 0.0 0.0
    ])
    coo = SDPX.build_block_coo([coefficient], [1], 3)
    @test length(coo.val) == nnz(coefficient)
    @test all(>(0), coo.lin)

    matrix = reshape(Float64.(1:9), 3, 3)
    left = reshape(Float64.(1:9) ./ 5, 3, 3)
    right = reshape(Float64.(10:18) ./ 13, 3, 3)
    destination = zeros(3, 3)
    @test SDPX._dot_dense_coo(matrix, coo, 1) ==
          sum(matrix .* Matrix(coefficient))
    SDPX._two_sided_coo_product!(destination, left, coo, 1, right)
    @test destination ≈ left * Matrix(coefficient) * right
end

@testset "BigFloat symmetric COO preserves owned destinations" begin
    setprecision(BigFloat, 128) do
        coefficient = sparse(BigFloat[
            0 2 0
            2 3 -4
            0 -4 0
        ])
        coo = SDPX.build_block_coo([coefficient], [1], 3)
        @test length(coo.val) == 3
        @test count(<(0), coo.lin) == 2

        left = BigFloat[BigFloat(10i + j) / 17 for i in 1:3, j in 1:3]
        right = BigFloat[BigFloat(7i - j) / 19 for i in 1:3, j in 1:3]
        matrix = BigFloat[BigFloat(3i + 2j) / 23 for i in 1:3, j in 1:3]
        destination = SDPX.alloc_zeros(BigFloat, 3, 3)
        scratch = SDPX._coo_contraction_scratch(BigFloat)
        SDPX._two_sided_coo_product_owned!(
            destination,
            left,
            coo,
            1,
            right,
            scratch,
        )
        @test destination ≈ left * Matrix(coefficient) * right
        SDPX._two_sided_coo_product_owned!(
            destination,
            left,
            coo,
            1,
            right,
            scratch,
        )
        @test destination ≈ left * Matrix(coefficient) * right
        @test all(
            destination[first] !== destination[second]
            for first in eachindex(destination), second in eachindex(destination)
            if first != second
        )

        dot_destination = SDPX.alloc_zeros(BigFloat, 1)
        SDPX._dot_dense_coo_store!(
            dot_destination,
            1,
            matrix,
            coo,
            1,
            scratch,
        )
        expected_dot = sum(matrix .* Matrix(coefficient))
        @test dot_destination[1] ≈ expected_dot
        @test SDPX._dot_dense_coo_value!(
            matrix,
            coo,
            1,
            scratch,
        ) ≈ expected_dot
    end
end

@testset "dense sparse-Schur COO contraction" begin
    matrix = Matrix(reshape(Float64.(1:64), 8, 8))
    linear_indices = Int32[2, 13, 24, 37, 50, 63]
    rows = Int32[((Int(index) - 1) % 8) + 1 for index in linear_indices]
    columns = Int32[((Int(index) - 1) ÷ 8) + 1 for index in linear_indices]
    values = Float64[1.25, -2.0, 0.5, 3.0, -1.5, 0.75]
    coo = SDPX.SparseBlockCOO{Float64}(
        Int32[1, length(linear_indices) + 1],
        linear_indices,
        rows,
        columns,
        values,
    )

    expected = 0.0
    @inbounds for index in eachindex(values)
        expected += matrix[Int(rows[index]), Int(columns[index])] * values[index]
    end
    repetitions = 10_000

    @test SDPX._dot_dense_coo(matrix, coo, 1) == expected
    @test _repeat_sparse_coo_contraction(matrix, coo, 1) == expected

    # Compile the exact production-shaped loop before measuring. The bound is
    # intentionally loose for the call/result frame but rules out the former
    # 80-byte `vec(matrix)` wrapper on every contraction under Julia 1.10.
    _repeat_sparse_coo_contraction(matrix, coo, repetitions)
    allocated_bytes = @allocated _repeat_sparse_coo_contraction(
        matrix,
        coo,
        repetitions,
    )

    @test allocated_bytes <= 256
    @test _repeat_sparse_coo_contraction(matrix, coo, repetitions) ==
          repetitions * expected
end
