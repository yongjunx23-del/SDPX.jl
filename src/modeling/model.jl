"""A typed handle to one native variable block owned by a `Model`."""
struct VariableBlockRef{T<:AbstractFloat}
    model::Model{T}
    block::Int
end

"""A scalar entry of a native variable block."""
struct VariableEntry{T<:AbstractFloat}
    model::Model{T}
    ref::VariableRef
end

@inline function _variable_record(block::VariableBlockRef)
    1 <= block.block <= length(block.model.variable_blocks) ||
        throw(ArgumentError("variable block $(block.block) is not registered in this model"))
    return block.model.variable_blocks[block.block]
end

@inline function _variable_global_index(entry::VariableEntry)
    record = entry.model.variable_blocks[entry.ref.block]
    return record.offset + entry.ref.index - 1
end

function _validate_variable_shape(domain::ProductConeDomain, shape::Int)
    shape >= 1 || throw(ArgumentError("variable dimension must be positive, got $shape"))
    domain isa RotatedLorentzCone && shape < 3 &&
        throw(ArgumentError("RotatedLorentzCone variable dimension must be at least 3"))
    domain isa ExponentialCone && shape != EXPONENTIAL_CONE_DIMENSION &&
        throw(ArgumentError(
            "ExponentialCone variable dimension must be exactly $EXPONENTIAL_CONE_DIMENSION, got $shape",
        ))
    domain isa PowerCone && shape != POWER_CONE_DIMENSION &&
        throw(ArgumentError(
            "PowerCone variable dimension must be exactly $POWER_CONE_DIMENSION, got $shape",
        ))
    return shape
end

function _check_new_variable_name(model::Model, name::Symbol)
    name === Symbol("") && throw(ArgumentError("variable name must not be empty"))
    haskey(model.block_names, name) &&
        throw(ArgumentError("variable block name $(repr(name)) is already registered"))
    return nothing
end

"""
    variable!(model, name, n; domain=Reals())
    variable!(model, name, n, n; domain=PSDCone())

Add one native product-cone block. Vector orthants and Lorentz cones remain one
vector block. An `n×n` PSD variable remains one lower-packed PSD block.
"""
function variable!(
    model::Model{T},
    name::Symbol,
    n::Integer;
    domain::ProductConeDomain=Reals(),
) where {T<:AbstractFloat}
    domain isa PSDCone && throw(ArgumentError(
        "PSDCone variables use variable!(model, name, n, n; domain=PSDCone())",
    ))
    _check_new_variable_name(model, name)
    shape = _validate_variable_shape(domain, Int(n))
    block_number = length(model.variable_blocks) + 1
    offset = length(model.variables) + 1
    block_length = variable_length(domain, shape)
    refs = Vector{VariableRef}(undef, block_length)
    identity = model_identity(model)
    @inbounds for index in 1:block_length
        refs[index] = VariableRef(identity, block_number, index)
    end
    record = VariableBlockRecord{T}(
        name,
        domain,
        shape,
        offset,
        block_length,
        nothing,
        nothing,
    )
    push!(model.variable_blocks, record)
    append!(model.variables, refs)
    model.block_names[name] = block_number
    model.next_variable_id = length(model.variables) + 1
    model.next_block_id = length(model.variable_blocks) + 1
    return VariableBlockRef{T}(model, block_number)
end

function variable!(
    model::Model{T},
    name::Symbol,
    rows::Integer,
    columns::Integer;
    domain::ProductConeDomain=PSDCone(),
) where {T<:AbstractFloat}
    domain isa PSDCone || throw(ArgumentError(
        "matrix variables are supported only for PSDCone(), got $domain",
    ))
    rows == columns || throw(DimensionMismatch(
        "PSDCone variable must be square, got $(rows)×$(columns)",
    ))
    _check_new_variable_name(model, name)
    shape = _validate_variable_shape(domain, Int(rows))
    block_number = length(model.variable_blocks) + 1
    offset = length(model.variables) + 1
    block_length = variable_length(domain, shape)
    refs = Vector{VariableRef}(undef, block_length)
    identity = model_identity(model)
    @inbounds for index in 1:block_length
        refs[index] = VariableRef(identity, block_number, index)
    end
    push!(model.variable_blocks, VariableBlockRecord{T}(
        name,
        domain,
        shape,
        offset,
        block_length,
        nothing,
        nothing,
    ))
    append!(model.variables, refs)
    model.block_names[name] = block_number
    model.next_variable_id = length(model.variables) + 1
    model.next_block_id = length(model.variable_blocks) + 1
    return VariableBlockRef{T}(model, block_number)
end

Base.eltype(::Type{VariableBlockRef{T}}) where {T} = VariableEntry{T}
Base.eltype(::VariableBlockRef{T}) where {T} = VariableEntry{T}
Base.length(block::VariableBlockRef) = _variable_record(block).length

function Base.size(block::VariableBlockRef)
    record = _variable_record(block)
    return record.domain isa PSDCone ? (record.shape, record.shape) : (record.shape,)
end

function Base.getindex(block::VariableBlockRef{T}, index::Integer) where {T}
    record = _variable_record(block)
    record.domain isa PSDCone && throw(ArgumentError(
        "PSD variables use two-dimensional indexing X[i,j]",
    ))
    1 <= index <= record.length || throw(BoundsError(block, index))
    return VariableEntry{T}(
        block.model,
        VariableRef(model_identity(block.model), block.block, Int(index)),
    )
end

function Base.getindex(
    block::VariableBlockRef{T},
    row::Integer,
    column::Integer,
) where {T}
    record = _variable_record(block)
    record.domain isa PSDCone || throw(ArgumentError(
        "two-dimensional indexing is available only for PSD variables",
    ))
    index = psd_packed_index(row, column, record.shape)
    return VariableEntry{T}(
        block.model,
        VariableRef(model_identity(block.model), block.block, index),
    )
end

function Base.iterate(block::VariableBlockRef, state::Int=1)
    record = _variable_record(block)
    record.domain isa PSDCone && throw(ArgumentError(
        "iterate PSD variables with matrix indices, not as unrelated scalar variables",
    ))
    state > record.length && return nothing
    return (block[state], state + 1)
end

variable_ref(entry::VariableEntry) = entry.ref

function Base.show(io::IO, block::VariableBlockRef{T}) where {T}
    record = _variable_record(block)
    print(io, "VariableBlockRef{", T, "}(", record.name,
          ", domain=", record.domain, ", shape=", size(block), ")")
end

Base.show(io::IO, entry::VariableEntry) = print(io, entry.ref)
