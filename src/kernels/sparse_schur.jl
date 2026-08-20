# Julia 1.10 does not reliably eliminate the `vec(matrix)` wrapper when the
# generic COO contraction is inlined into the large sparse-Schur pair loop.
# Production block workspaces are owned dense `Matrix` values, so specialize
# that hot path and index its existing column-major linear storage directly.
# The generic `AbstractMatrix` method in `schur.jl` remains the compatibility
# path for views and custom matrix types.
@inline function _dot_dense_coo(
    matrix::Matrix{T},
    coo::SparseBlockCOO{T},
    position::Int,
) where {T}
    linear_indices = coo.lin
    values = coo.val
    accumulator = zero(T)
    @inbounds for entry in
        coo.ptr[position]:(coo.ptr[position + 1] - Int32(1))
        accumulator += matrix[linear_indices[entry]] * values[entry]
    end
    return accumulator
end
