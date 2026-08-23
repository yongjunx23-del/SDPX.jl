"""Typed handle to one affine-in-cone constraint block."""
struct ConstraintBlockRef{T<:AbstractFloat}
    model::Model{T}
    block::Int
end

"""Scalar/packed entry of an affine constraint block."""
struct ConstraintEntry{T<:AbstractFloat}
    model::Model{T}
    ref::ConstraintRef
end

@inline function _constraint_record(block::ConstraintBlockRef)
    1 <= block.block <= length(block.model.constraint_blocks) ||
        throw(ArgumentError("constraint block $(block.block) is not registered"))
    return block.model.constraint_blocks[block.block]
end

Base.length(block::ConstraintBlockRef) = length(_constraint_record(block).refs)
function Base.size(block::ConstraintBlockRef)
    record = _constraint_record(block)
    return record.domain isa PSDCone ? (record.shape, record.shape) : (record.shape,)
end

function Base.getindex(block::ConstraintBlockRef{T}, index::Integer) where {T}
    record = _constraint_record(block)
    record.domain isa PSDCone && throw(ArgumentError(
        "PSD constraint blocks use two-dimensional indexing",
    ))
    1 <= index <= length(record.refs) || throw(BoundsError(block, index))
    return ConstraintEntry{T}(block.model, record.refs[index])
end

function Base.getindex(
    block::ConstraintBlockRef{T},
    row::Integer,
    column::Integer,
) where {T}
    record = _constraint_record(block)
    record.domain isa PSDCone || throw(ArgumentError(
        "two-dimensional indexing is available only for PSD constraint blocks",
    ))
    index = psd_packed_index(row, column, record.shape)
    return ConstraintEntry{T}(block.model, record.refs[index])
end

constraint_ref(entry::ConstraintEntry) = entry.ref
constraint_refs(block::ConstraintBlockRef) = copy(_constraint_record(block).refs)

function _check_new_constraint_name(model::Model, name::Symbol)
    name === Symbol("") && throw(ArgumentError("constraint name must not be empty"))
    haskey(model.constraint_names, name) && throw(ArgumentError(
        "constraint block name $(repr(name)) is already registered",
    ))
    return nothing
end

function _normalize_constraint_expression(
    model::Model{T},
    expression,
    domain::ProductConeDomain,
) where {T<:AbstractFloat}
    if domain isa PSDCone
        expression isa AbstractMatrix || throw(ArgumentError(
            "PSDCone constraint requires a square affine matrix",
        ))
        rows, columns = size(expression)
        rows == columns || throw(DimensionMismatch(
            "PSDCone constraint matrix must be square, got $(rows)×$(columns)",
        ))
        rows >= 1 || throw(ArgumentError("PSDCone constraint must be nonempty"))
        matrix = Matrix{ScalarAffine{T}}(undef, rows, columns)
        for column in 1:columns, row in 1:rows
            matrix[row, column] = _as_affine(model, expression[row, column])
        end
        for column in 1:columns, row in column:rows
            _affine_equal(matrix[row, column], matrix[column, row]) ||
                throw(ArgumentError("PSDCone affine matrix must be exactly symmetric"))
        end
        packed = Vector{ScalarAffine{T}}(undef, variable_length(domain, rows))
        for column in 1:columns, row in column:rows
            packed[psd_packed_index(row, column, rows)] = matrix[row, column]
        end
        return rows, packed
    end

    values = expression isa AbstractVector || expression isa Tuple ?
        expression : (expression,)
    shape = length(values)
    shape >= 1 || throw(ArgumentError("constraint block must be nonempty"))
    domain isa RotatedLorentzCone && shape < 3 && throw(ArgumentError(
        "RotatedLorentzCone constraint dimension must be at least 3",
    ))
    expressions = Vector{ScalarAffine{T}}(undef, shape)
    for index in 1:shape
        expressions[index] = _as_affine(model, values[index])
    end
    return shape, expressions
end

"""
    constraint!(model, name, expression, domain)

Add one affine block with canonical semantics `expression in domain`. The
compiler later stores each row as `A*x-rhs in domain`, with `rhs=-constant`.
"""
function constraint!(
    model::Model{T},
    name::Symbol,
    expression,
    domain::ProductConeDomain,
) where {T<:AbstractFloat}
    _check_new_constraint_name(model, name)
    shape, expressions = _normalize_constraint_expression(model, expression, domain)
    block_number = length(model.constraint_blocks) + 1
    identity = model_identity(model)
    refs = Vector{ConstraintRef}(undef, length(expressions))
    for index in eachindex(refs)
        refs[index] = ConstraintRef(identity, block_number, index)
    end
    record = AffineConstraintRecord{T}(
        name,
        domain,
        shape,
        expressions,
        refs,
        nothing,
    )
    push!(model.constraint_blocks, record)
    append!(model.constraints, refs)
    model.constraint_names[name] = block_number
    model.next_constraint_id = length(model.constraints) + 1
    return ConstraintBlockRef{T}(model, block_number)
end

constraint!(model::Model, name::Symbol, expression; domain::ProductConeDomain) =
    constraint!(model, name, expression, domain)

"""Set the model's single scalar affine objective."""
function objective!(
    model::Model{T},
    sense::Union{Minimize,Maximize},
    expression,
) where {T<:AbstractFloat}
    model.objective === nothing || throw(ArgumentError("objective! may be called only once"))
    affine = _as_affine(model, expression)
    # Keep the expression returned to the caller separate from the one held
    # by the model.  ScalarAffine's outer struct is immutable, but its index
    # and coefficient vectors (and BigFloat coefficient objects) are not;
    # retaining `affine` directly would let a caller mutate a registered
    # objective through either the original expression or this return value.
    stored = _owned_affine_copy(model, affine)
    model.objective = ObjectiveRecord{T}(sense, stored)
    return affine
end

function Base.show(io::IO, block::ConstraintBlockRef{T}) where {T}
    record = _constraint_record(block)
    print(io, "ConstraintBlockRef{", T, "}(", record.name,
          ", domain=", record.domain, ", shape=", size(block), ")")
end

Base.show(io::IO, entry::ConstraintEntry) = print(io, entry.ref)
