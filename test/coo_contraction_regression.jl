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

Base.@noinline function _repeat_sparse_coo_contraction_projected(
    matrix::Matrix{Float64},
    coo::SDPX.SparseBlockCOO{Float64},
    symmetric_projection::Bool,
    repetitions::Int,
)
    accumulator = 0.0
    @inbounds for _ in 1:repetitions
        accumulator +=
            SDPX._dot_dense_coo(matrix, coo, 1, symmetric_projection)
    end
    return accumulator
end

function _projection_fixture(::Type{T}=Float64) where {T}
    variables = 3
    coefficients = [spzeros(T, 3, 3) for _ in 1:variables]
    coefficients[1][1, 2] = coefficients[1][2, 1] = one(T)
    coefficients[2][2, 3] = coefficients[2][3, 2] = T(2)
    coefficients[3][1, 1] = one(T)
    return SDPX.ingest(
        ones(T, variables),
        [coefficients],
        [zeros(T, 3, 3)],
        zeros(T, variables, 0),
        T[];
        sparse=true,
        verbosity=0,
    )
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
    SDPX._two_sided_coo_product!(destination, left, coo, 1, right, true)
    SDPX._dot_dense_coo(destination, coo, 1, true)
    @test @allocated(
        SDPX._two_sided_coo_product!(destination, left, coo, 1, right, true)
    ) <= 256
    @test @allocated(
        SDPX._dot_dense_coo(destination, coo, 1, true)
    ) <= 256

    empty_coo = SDPX.build_block_coo([spzeros(4, 4)], [1], 4)
    @test empty_coo.ptr == Int32[1, 1]
    @test SDPX._coo_supports_symmetric_projection(empty_coo)
    fill!(destination, 1.0)
    SDPX._two_sided_coo_product!(destination, left, empty_coo, 1, right)
    @test iszero(SDPX._dot_dense_coo(matrix, empty_coo, 1))
    @test iszero(norm(destination, Inf))

    diagonal_coo = SDPX.build_block_coo(
        [sparse(Diagonal([1.0, -2.0, 3.0, 0.0]))],
        [1],
        4,
    )
    @test SDPX._coo_supports_symmetric_projection(diagonal_coo)
    old_diagonal = copy(destination)
    projected_diagonal = copy(destination)
    SDPX._two_sided_coo_product!(
        old_diagonal,
        left,
        diagonal_coo,
        1,
        right,
    )
    SDPX._two_sided_coo_product!(
        projected_diagonal,
        left,
        diagonal_coo,
        1,
        right,
        true,
    )
    @test diag(projected_diagonal) == diag(old_diagonal)
    @test SDPX._dot_dense_coo(old_diagonal, diagonal_coo, 1) ==
          SDPX._dot_dense_coo(
              projected_diagonal,
              diagonal_coo,
              1,
              true,
          )
end

@testset "symmetric COO projection preserves Schur contractions" begin
    problem = _projection_fixture()
    coo = problem.cons.coo[1]
    @test SDPX._coo_supports_symmetric_projection(coo)
    workspace = SDPX.Workspace(problem; thread_count=1)
    @test workspace.blk[1].coo_symmetric_projection

    left = [
        1.2 0.2 0.1
        0.3 1.1 0.2
        0.4 0.1 0.9
    ]
    right = [
        0.8 0.4 0.2
        0.1 1.2 0.3
        0.3 0.2 1.1
    ]
    legacy = zeros(3, 3)
    projected = zeros(3, 3)
    SDPX._two_sided_coo_product!(legacy, left, coo, 1, right)
    SDPX._two_sided_coo_product!(
        projected,
        left,
        coo,
        1,
        right,
        true,
    )
    for entry in coo.ptr[1]:(coo.ptr[2] - Int32(1))
        linear_index = coo.lin[entry]
        linear_index < 0 || continue
        row = Int(coo.row[entry])
        column = Int(coo.col[entry])
        @test projected[-linear_index] ==
              legacy[-linear_index] + legacy[(row - 1) * 3 + column]
    end
    @test SDPX._dot_dense_coo(legacy, coo, 1) ==
          SDPX._dot_dense_coo(projected, coo, 1, true)

    X = [Matrix{Float64}(I, 3, 3)]
    Y = [Matrix{Float64}(I, 3, 3)]
    @test SDPX.factor_blocks!(workspace, X, Y)
    block_workspace = workspace.blk[1]
    block_workspace.coo_symmetric_projection = true
    projected_values = copy(
        SDPX.sparse_schur_block!(
            block_workspace,
            problem.cons,
            1,
            X[1],
            Y[1],
        ),
    )
    block_workspace.coo_symmetric_projection = false
    legacy_values = copy(
        SDPX.sparse_schur_block!(
            block_workspace,
            problem.cons,
            1,
            X[1],
            Y[1],
        ),
    )
    @test projected_values == legacy_values

    projected_scatter = zeros(3, 3)
    legacy_scatter = zeros(3, 3)
    block_workspace.coo_symmetric_projection = true
    SDPX.sparse_schur_block_scatter!(
        projected_scatter,
        block_workspace,
        problem.cons,
        1,
        X[1],
        Y[1],
    )
    block_workspace.coo_symmetric_projection = false
    SDPX.sparse_schur_block_scatter!(
        legacy_scatter,
        block_workspace,
        problem.cons,
        1,
        X[1],
        Y[1],
    )
    @test projected_scatter == legacy_scatter
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
    @test !SDPX._coo_supports_symmetric_projection(coo)

    reference_problem = _projection_fixture()
    raw_coefficients = [spzeros(Float64, 3, 3) for _ in 1:3]
    raw_coefficients[1][1, 2] = 1.0
    raw_coefficients[2][2, 3] = 2.0
    raw_coefficients[3][1, 1] = 1.0
    raw_cons = SDPX.SparseCons{Float64}(
        [raw_coefficients],
        [[1, 2, 3]],
        [[1, 2, 3]],
        [zeros(Float64, 0, 0)],
    )
    problem = SDPX.SDPProblem{Float64}(
        reference_problem.c,
        reference_problem.C,
        reference_problem.B,
        reference_problem.b,
        raw_cons,
        reference_problem.dims,
        reference_problem.structure,
    )
    @test !SDPX.Workspace(problem; thread_count=1).blk[1].coo_symmetric_projection

    matrix = reshape(Float64.(1:9), 3, 3)
    left = reshape(Float64.(1:9) ./ 5, 3, 3)
    right = reshape(Float64.(10:18) ./ 13, 3, 3)
    destination = zeros(3, 3)
    @test SDPX._dot_dense_coo(matrix, coo, 1) ==
          sum(matrix .* Matrix(coefficient))
    @test SDPX._dot_dense_coo(matrix, coo, 1, false) ==
          SDPX._dot_dense_coo(matrix, coo, 1)
    SDPX._two_sided_coo_product!(destination, left, coo, 1, right)
    @test destination ≈ left * Matrix(coefficient) * right
    projected_fallback = zeros(3, 3)
    SDPX._two_sided_coo_product!(
        projected_fallback,
        left,
        coo,
        1,
        right,
        false,
    )
    @test projected_fallback == destination
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
        projected_destination = SDPX.alloc_zeros(BigFloat, 3, 3)
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

        projected_scalar = projected_destination[1, 2]
        SDPX._two_sided_coo_product_owned!(
            projected_destination,
            left,
            coo,
            1,
            right,
            scratch,
            true,
        )
        expected_projected = deepcopy(destination)
        for column in 2:3, row in 1:(column - 1)
            expected_projected[row, column] += destination[column, row]
        end
        @test projected_destination ≈ expected_projected
        @test projected_destination[1, 2] === projected_scalar

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

        dot_projected_destination = SDPX.alloc_zeros(BigFloat, 1)
        dot_projected_scalar = dot_projected_destination[1]
        SDPX._dot_dense_coo_store!(
            dot_projected_destination,
            1,
            projected_destination,
            coo,
            1,
            scratch,
            true,
        )
        @test dot_projected_destination[1] ≈
              SDPX._dot_dense_coo_value!(
                  projected_destination,
                  coo,
                  1,
                  scratch,
                  true,
              )
        @test dot_projected_destination[1] === dot_projected_scalar
    end
end

if Base.find_package("MultiFloats") !== nothing
    import MultiFloats
    @testset "MultiFloat symmetric COO projection" begin
        T = MultiFloats.Float64x2
        coefficient = sparse(T[
            0 2 0
            2 3 -4
            0 -4 0
        ])
        coo = SDPX.build_block_coo([coefficient], [1], 3)
        left = T[
            1.2 0.2 0.1
            0.3 1.1 0.2
            0.4 0.1 0.9
        ]
        right = T[
            0.8 0.4 0.2
            0.1 1.2 0.3
            0.3 0.2 1.1
        ]
        legacy = zeros(T, 3, 3)
        projected = zeros(T, 3, 3)
        SDPX._two_sided_coo_product!(legacy, left, coo, 1, right)
        SDPX._two_sided_coo_product!(
            projected,
            left,
            coo,
            1,
            right,
            true,
        )
        @test SDPX._dot_dense_coo(legacy, coo, 1) ==
              SDPX._dot_dense_coo(projected, coo, 1, true)
        SDPX._two_sided_coo_product!(projected, left, coo, 1, right, true)
        SDPX._dot_dense_coo(projected, coo, 1, true)
        @test @allocated(
            SDPX._two_sided_coo_product!(
                projected,
                left,
                coo,
                1,
                right,
                true,
            )
        ) <= 256
        @test @allocated(
            SDPX._dot_dense_coo(projected, coo, 1, true)
        ) <= 256
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
