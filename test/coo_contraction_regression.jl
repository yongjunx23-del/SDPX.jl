using Test

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
    expected = sum(
        matrix[Int(rows[index]), Int(columns[index])] * values[index]
        for index in eachindex(values)
    )
    repetitions = 10_000

    @test SDPX._dot_dense_coo(matrix, coo, 1) == expected
    @test _repeat_sparse_coo_contraction(matrix, coo, 1) == expected

    # Compile the exact hot loop before measuring. The allocation bound guards
    # against reintroducing a `vec(matrix)` wrapper for every variable pair.
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
