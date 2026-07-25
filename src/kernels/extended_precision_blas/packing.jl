"""
    pack_columns!(destination, source, rows, columns)

Copy a logical `rows × columns` panel into contiguous column-major storage.
The caller owns both buffers, so the loop performs no allocation. Sparse
SDPX packing supplies columns in shared-pattern groups before calling the
matrix kernels; dense panels are already in this layout.
"""
function pack_columns!(
    destination::AbstractMatrix{T},
    source::AbstractMatrix{T},
    rows::Int=size(source, 1),
    columns::Int=size(source, 2),
) where {T}
    size(destination, 1) >= rows ||
        throw(DimensionMismatch("packed destination has too few rows"))
    size(destination, 2) >= columns ||
        throw(DimensionMismatch("packed destination has too few columns"))
    @inbounds for column in 1:columns, row in 1:rows
        destination[row, column] = source[row, column]
    end
    return destination
end

prepare_storage!(array::AbstractArray) = array

function prepare_storage!(array::AbstractArray{BigFloat})
    # `zeros(BigFloat, ...)` and `fill!` alias one mutable MPFR object across
    # all entries. Give every destination its own object before mutating it
    # with MutableArithmetics in the allocation-free hot loops.
    @inbounds for index in eachindex(array)
        array[index] = array[index] + zero(BigFloat)
    end
    return array
end

prepare_triangle_storage!(matrix::AbstractMatrix) = matrix

function prepare_triangle_storage!(matrix::AbstractMatrix{BigFloat})
    @inbounds for column in axes(matrix, 2)
        for row in column:size(matrix, 1)
            matrix[row, column] =
                matrix[row, column] + zero(BigFloat)
        end
    end
    return matrix
end

function zero_triangle!(matrix::AbstractMatrix{T}) where {T}
    @inbounds for column in axes(matrix, 2)
        for row in column:size(matrix, 1)
            matrix[row, column] = zero(T)
        end
    end
    return matrix
end

function zero_triangle!(matrix::AbstractMatrix{BigFloat})
    @inbounds for column in axes(matrix, 2)
        for row in column:size(matrix, 1)
            MA.operate!(zero, matrix[row, column])
        end
    end
    return matrix
end
