function _owned_start_vector(
    model::Model{T},
    values,
    expected::Int,
    label::AbstractString,
) where {T<:AbstractFloat}
    source = values isa Number ? (values,) : collect(values)
    length(source) == expected || throw(DimensionMismatch(
        "$label length $(length(source)) != expected $expected",
    ))
    result = Vector{T}(undef, expected)
    for index in 1:expected
        result[index] = _owned_model_scalar(model, source[index])
    end
    return result
end

function _owned_variable_start(
    block::VariableBlockRef{T},
    values,
    label,
) where {T<:AbstractFloat}
    record = _variable_record(block)
    if record.domain isa PSDCone
        values isa AbstractMatrix || throw(ArgumentError(
            "$label for a PSD variable must be a symmetric matrix",
        ))
        size(values) == (record.shape, record.shape) || throw(DimensionMismatch(
            "$label size $(size(values)) != $((record.shape, record.shape))",
        ))
        packed = Vector{T}(undef, record.length)
        for column in 1:record.shape, row in column:record.shape
            values[row, column] == values[column, row] || throw(ArgumentError(
                "$label for a PSD variable must be exactly symmetric",
            ))
            packed[psd_packed_index(row, column, record.shape)] =
                _owned_model_scalar(block.model, values[row, column])
        end
        return packed
    end
    return _owned_start_vector(block.model, values, record.length, label)
end

"""Set an owned primal warm start for one native variable block."""
function set_start!(block::VariableBlockRef{T}, values) where {T<:AbstractFloat}
    record = _variable_record(block)
    start = _owned_variable_start(block, values, "primal start")
    block.model.variable_blocks[block.block] = VariableBlockRecord{T}(
        record.name,
        record.domain,
        record.shape,
        record.offset,
        record.length,
        start,
        record.dual_slack_start,
    )
    return block
end

"""Set an owned variable-dual-slack warm start for one native variable block."""
function set_dual_slack_start!(
    block::VariableBlockRef{T},
    values,
) where {T<:AbstractFloat}
    record = _variable_record(block)
    start = _owned_variable_start(block, values, "dual-slack start")
    block.model.variable_blocks[block.block] = VariableBlockRecord{T}(
        record.name,
        record.domain,
        record.shape,
        record.offset,
        record.length,
        record.primal_start,
        start,
    )
    return block
end

"""Set an owned dual warm start for one affine constraint block."""
function set_dual_start!(
    block::ConstraintBlockRef{T},
    values,
) where {T<:AbstractFloat}
    record = _constraint_record(block)
    start = if record.domain isa PSDCone
        values isa AbstractMatrix || throw(ArgumentError(
            "dual start for a PSD constraint must be a symmetric matrix",
        ))
        size(values) == (record.shape, record.shape) || throw(DimensionMismatch(
            "PSD dual-start size $(size(values)) != $((record.shape, record.shape))",
        ))
        packed = Vector{T}(undef, length(record.refs))
        for column in 1:record.shape, row in column:record.shape
            values[row, column] == values[column, row] || throw(ArgumentError(
                "dual start for a PSD constraint must be exactly symmetric",
            ))
            packed[psd_packed_index(row, column, record.shape)] =
                _owned_model_scalar(block.model, values[row, column])
        end
        packed
    else
        _owned_start_vector(block.model, values, length(record.refs), "constraint dual start")
    end
    block.model.constraint_blocks[block.block] = AffineConstraintRecord{T}(
        record.name,
        record.domain,
        record.shape,
        record.expressions,
        record.refs,
        start,
    )
    return block
end
