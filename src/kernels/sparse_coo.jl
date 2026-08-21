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
    values = coo.val
    accumulator = zero(T)
    @inbounds for entry in
        coo.ptr[position]:(coo.ptr[position + 1] - Int32(1))
        accumulator += matrix[linear_indices[entry]] * values[entry]
    end
    return accumulator
end
