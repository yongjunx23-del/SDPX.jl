# Sparse-Schur COO contraction kernels.
#
# The generic compatibility implementation in `schur.jl` accepts any
# `AbstractMatrix` and obtains flat storage through `vec`. Production Schur
# workspaces pass owned dense `Matrix` values, and Julia 1.10 does not reliably
# eliminate that wrapper after this method is inlined into the quadratic
# active-variable loop. Specialize only the owned production case so every
# contraction indexes the same existing column-major storage directly.

@inline function _dot_dense_coo(
    matrix::Matrix{T},
    coo::SparseBlockCOO{T},
    position::Int,
) where {T}
    linear_indices = coo.lin
    rows = coo.row
    columns = coo.col
    values = coo.val
    dimension = size(matrix, 1)
    accumulator = zero(T)
    @inbounds for entry in
        coo.ptr[position]:(coo.ptr[position + 1] - Int32(1))
        linear_index = linear_indices[entry]
        matrix_value = if linear_index > 0
            matrix[linear_index]
        else
            row = Int(rows[entry])
            column = Int(columns[entry])
            matrix[-linear_index] + matrix[(row - 1) * dimension + column]
        end
        accumulator += matrix_value * values[entry]
    end
    return accumulator
end
