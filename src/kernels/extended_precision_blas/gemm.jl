"""
    gemm!(C, A, B, alpha, beta, config)

Blocked in-place matrix multiplication for extended-precision scalar types.
Only caller-owned storage is used. BigFloat remains serial and delegates each
scalar reduction to SDPX's allocation-free MPFR column-dot primitive after
packing the required row of `A` into the supplied scratch vector.
"""
function gemm!(
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    alpha::T,
    beta::T,
    config::KernelConfig=KernelConfig(),
) where {T}
    m, reduction = size(A)
    size(B, 1) == reduction || throw(DimensionMismatch("gemm! inner dimensions differ"))
    n = size(B, 2)
    size(C) == (m, n) || throw(DimensionMismatch("gemm! output dimensions differ"))
    column_tile = max(config.column_tile, 1)
    row_tile = column_tile
    reduction_tile = max(config.row_tile, 1)
    @inbounds for column_block in 1:column_tile:n
        column_stop = min(column_block + column_tile - 1, n)
        for row_block in 1:row_tile:m
            row_stop = min(row_block + row_tile - 1, m)
            for column in column_block:column_stop, row in row_block:row_stop
                accumulator = zero(T)
                for reduction_block in 1:reduction_tile:reduction
                    reduction_stop =
                        min(reduction_block + reduction_tile - 1, reduction)
                    for index in reduction_block:reduction_stop
                        accumulator += A[row, index] * B[index, column]
                    end
                end
                C[row, column] =
                    alpha * accumulator + beta * C[row, column]
            end
        end
    end
    return C
end

function gemm!(
    C::AbstractMatrix{BigFloat},
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    alpha::BigFloat,
    beta::BigFloat,
    config::KernelConfig=KernelConfig(),
)
    # The solver's BigFloat matrix products already use the allocation-free
    # MPFR scalar kernel in `kernels/bigfloat.jl`. Keep that established,
    # alias-safe implementation as the GEMM backend.
    parent = parentmodule(@__MODULE__)
    return parent.kmul!(C, A, B, alpha, beta)
end
