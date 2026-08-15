prepare_storage!(array::AbstractArray) = array

function prepare_storage!(array::AbstractArray{BigFloat})
    # `zeros(BigFloat, ...)` and `fill!` alias one mutable MPFR object across
    # all entries, while `similar` leaves reference slots uninitialized. Give
    # every destination its own object before mutating it with
    # MutableArithmetics in the allocation-free hot loops.
    @inbounds for index in eachindex(array)
        array[index] =
            isassigned(array, index) ?
            MA.mutable_copy(array[index]) : BigFloat(0)
    end
    return array
end

prepare_triangle_storage!(matrix::AbstractMatrix) = matrix

function prepare_triangle_storage!(matrix::AbstractMatrix{BigFloat})
    @inbounds for column in axes(matrix, 2)
        for row in column:size(matrix, 1)
            matrix[row, column] =
                isassigned(matrix, row, column) ?
                MA.mutable_copy(matrix[row, column]) : BigFloat(0)
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
