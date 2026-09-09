@inline function _owned_model_scalar(model::Model{T}, value) where {T<:AbstractFloat}
    converted = owned_arithmetic_copy(
        T,
        value;
        precision_bits=precision_bits(model),
    )
    isfinite(converted) || throw(ArgumentError("affine data contains NaN or Inf"))
    return converted
end

@inline function _owned_affine_scalar(::Type{T}, bits::Int, value) where {T<:AbstractFloat}
    converted = owned_arithmetic_copy(T, value; precision_bits=bits)
    isfinite(converted) || throw(ArgumentError("affine arithmetic produced NaN or Inf"))
    return converted
end

@inline function _owned_affine_eval(
    ::Type{T},
    bits::Int,
    operation::F,
) where {T<:AbstractFloat,F<:Function}
    return _owned_affine_scalar(
        T,
        bits,
        _owned_arithmetic_eval(T, operation; precision_bits=bits),
    )
end

function _constant_affine(model::Model{T}, value) where {T<:AbstractFloat}
    return ScalarAffine{T}(
        model_identity(model),
        precision_bits(model),
        Int[],
        T[],
        _owned_model_scalar(model, value),
    )
end

function _entry_affine(entry::VariableEntry{T}) where {T<:AbstractFloat}
    model = entry.model
    return ScalarAffine{T}(
        model_identity(model),
        precision_bits(model),
        Int[_variable_global_index(entry)],
        T[_owned_model_scalar(model, 1)],
        _owned_model_scalar(model, 0),
    )
end

"""Return a fully model-owned copy of one scalar affine expression.

`ScalarAffine` is immutable only at the outer struct level: its index and
coefficient vectors (and, for `BigFloat`, each coefficient object) remain
mutable.  Registration therefore cannot retain an expression supplied by a
caller, even when it already has the right model identity.  Copy every
mutable component through the model's owned-arithmetic boundary so BigFloat
values are rounded at the model precision rather than the ambient scope.
"""
function _owned_affine_copy(
    model::Model{T},
    expression::ScalarAffine{T},
) where {T<:AbstractFloat}
    expression.model == model_identity(model) || throw(ArgumentError(
        "affine expression belongs to a different model",
    ))
    expression.precision_bits == precision_bits(model) || throw(ArgumentError(
        "affine expression precision does not match model precision",
    ))
    length(expression.indices) == length(expression.coefficients) ||
        throw(DimensionMismatch("affine indices and coefficients must have equal length"))

    bits = precision_bits(model)
    indices = copy(expression.indices)
    coefficients = owned_vector_copy(
        T,
        expression.coefficients;
        precision_bits=bits,
    )
    constant = owned_arithmetic_copy(T, expression.constant; precision_bits=bits)
    return ScalarAffine{T}(
        model_identity(model),
        bits,
        indices,
        coefficients,
        constant,
    )
end

function _require_same_affine_model(left::ScalarAffine, right::ScalarAffine)
    left.model == right.model || throw(ArgumentError(
        "cannot combine affine expressions from different models",
    ))
    left.precision_bits == right.precision_bits || throw(ArgumentError(
        "cannot combine affine expressions with different precision ownership",
    ))
    return nothing
end

# ---------------------------------------------------------------------------
# Internal bulk affine builder.
#
# The public `Base.:+` below rebuilds and sorts the whole growing
# index/coefficient vector on every `expression = expression + term` step
# (O(k^2 log k) per row).  The builder accumulates terms once and materializes
# with a single stable sort + duplicate merge.  It is bit-identical to the
# left-fold `+` sequence because each index group's duplicates are accumulated
# in term-encounter order through the same owned-arithmetic operations with the
# same drop-zero rule.  `Base.:+` semantics are unchanged; this is an internal
# construction path only.
# ---------------------------------------------------------------------------
mutable struct _AffineBuilder{T<:AbstractFloat}
    model::UInt64
    precision_bits::Int
    indices::Vector{Int}
    coefficients::Vector{T}
    constant::T
    sealed::Bool
end

function _affine_builder(model::Model{T}, capacity::Integer=0) where {T<:AbstractFloat}
    capacity >= 0 || throw(ArgumentError("affine builder capacity must be nonnegative"))
    return _AffineBuilder{T}(
        model_identity(model),
        precision_bits(model),
        sizehint!(Int[], Int(capacity)),
        sizehint!(T[], Int(capacity)),
        _owned_model_scalar(model, 0),
        false,
    )
end

@inline function _check_builder_open(builder::_AffineBuilder)
    builder.sealed && throw(ArgumentError("affine builder is sealed"))
    return nothing
end

"""Append `scalar * <variable at global_index>` using the same owned-arithmetic
boundary as `scalar * ::VariableEntry`."""
function push_term!(
    builder::_AffineBuilder{T}, scalar, global_index::Integer,
) where {T<:AbstractFloat}
    _check_builder_open(builder)
    converted = _owned_affine_scalar(T, builder.precision_bits, scalar)
    coefficient = _owned_affine_eval(
        T, builder.precision_bits, () -> converted * one(T),
    )
    push!(builder.indices, Int(global_index))
    push!(builder.coefficients, coefficient)
    return builder
end

"""Append a constant contribution using the same owned-arithmetic boundary as
`Base.:+`."""
function add_constant!(
    builder::_AffineBuilder{T}, value,
) where {T<:AbstractFloat}
    _check_builder_open(builder)
    converted = _owned_affine_scalar(T, builder.precision_bits, value)
    builder.constant = _owned_affine_eval(
        T, builder.precision_bits, () -> builder.constant + converted,
    )
    return builder
end

"""Append one whole affine term (left-fold order preserved)."""
function push_affine!(builder::_AffineBuilder{T}, expression::ScalarAffine{T}) where {T}
    _check_builder_open(builder)
    expression.model == builder.model || throw(ArgumentError(
        "affine expression belongs to a different model",
    ))
    expression.precision_bits == builder.precision_bits || throw(ArgumentError(
        "affine expression precision does not match builder precision",
    ))
    for position in eachindex(expression.indices)
        push!(builder.indices, expression.indices[position])
        push!(builder.coefficients, expression.coefficients[position])
    end
    add_constant!(builder, expression.constant)
    return builder
end

"""Materialize the accumulated expression, moving the buffers into the result
and sealing the builder (further mutation throws)."""
function materialize(builder::_AffineBuilder{T}) where {T<:AbstractFloat}
    _check_builder_open(builder)
    indices = builder.indices
    coefficients = builder.coefficients
    # Stable sort: duplicates keep term-encounter order, matching the fold.
    permutation = sortperm(indices; alg=Base.Sort.DEFAULT_STABLE)
    out_indices = Int[]
    out_coefficients = T[]
    for position in permutation
        index = indices[position]
        coefficient = coefficients[position]
        if !isempty(out_indices) && out_indices[end] == index
            out_coefficients[end] = _owned_affine_eval(
                T, builder.precision_bits,
                () -> out_coefficients[end] + coefficient,
            )
            if iszero(out_coefficients[end])
                pop!(out_indices)
                pop!(out_coefficients)
            end
        elseif !iszero(coefficient)
            push!(out_indices, index)
            push!(out_coefficients, coefficient)
        end
    end
    result = ScalarAffine{T}(
        builder.model,
        builder.precision_bits,
        out_indices,
        out_coefficients,
        builder.constant,
    )
    builder.sealed = true
    empty!(builder.indices)
    empty!(builder.coefficients)
    return result
end

"""Bulk left-fold sum of affine terms into one expression (single sort/merge).
Matches `foldl(+, terms)` bit-for-bit: terms are accumulated in order and each
index group's duplicates keep term-encounter order."""
function _affine_sum(model::Model{T}, terms) where {T<:AbstractFloat}
    builder = _affine_builder(model, length(terms))
    for term in terms
        if term isa ScalarAffine{T}
            push_affine!(builder, term)
        elseif term isa VariableEntry{T}
            push_affine!(builder, _entry_affine(term))
        elseif term isa Number
            add_constant!(builder, term)
        else
            throw(ArgumentError(
                "expected a scalar affine value, got $(typeof(term))",
            ))
        end
    end
    return materialize(builder)
end

Base.convert(::Type{ScalarAffine{T}}, entry::VariableEntry{T}) where {T} =
    _entry_affine(entry)

Base.:+(entry::VariableEntry) = _entry_affine(entry)
Base.:-(entry::VariableEntry) = -_entry_affine(entry)

function Base.:+(left::ScalarAffine{T}, right::ScalarAffine{T}) where {T}
    _require_same_affine_model(left, right)
    indices = vcat(left.indices, right.indices)
    coefficients = vcat(copy(left.coefficients), copy(right.coefficients))
    # Reconstruct using either expression's ownership without a Model pointer.
    permutation = sortperm(indices)
    out_indices = Int[]
    out_coefficients = T[]
    for position in permutation
        index = indices[position]
        coefficient = coefficients[position]
        if !isempty(out_indices) && out_indices[end] == index
            out_coefficients[end] = _owned_affine_eval(
                T,
                left.precision_bits,
                () -> out_coefficients[end] + coefficient,
            )
            if iszero(out_coefficients[end])
                pop!(out_indices)
                pop!(out_coefficients)
            end
        elseif !iszero(coefficient)
            push!(out_indices, index)
            push!(out_coefficients, coefficient)
        end
    end
    return ScalarAffine{T}(
        left.model,
        left.precision_bits,
        out_indices,
        out_coefficients,
        _owned_affine_eval(
            T,
            left.precision_bits,
            () -> left.constant + right.constant,
        ),
    )
end

Base.:+(left::VariableEntry{T}, right::VariableEntry{T}) where {T} =
    _entry_affine(left) + _entry_affine(right)
Base.:+(left::ScalarAffine{T}, right::VariableEntry{T}) where {T} =
    left + _entry_affine(right)
Base.:+(left::VariableEntry{T}, right::ScalarAffine{T}) where {T} =
    _entry_affine(left) + right
Base.:-(left::ScalarAffine{T}, right::ScalarAffine{T}) where {T} = left + (-right)
Base.:-(left::VariableEntry{T}, right::VariableEntry{T}) where {T} =
    _entry_affine(left) - _entry_affine(right)
Base.:-(left::ScalarAffine{T}, right::VariableEntry{T}) where {T} =
    left - _entry_affine(right)
Base.:-(left::VariableEntry{T}, right::ScalarAffine{T}) where {T} =
    _entry_affine(left) - right

function Base.:-(expression::ScalarAffine{T}) where {T}
    return ScalarAffine{T}(
        expression.model,
        expression.precision_bits,
        copy(expression.indices),
        [
            _owned_affine_eval(
                T,
                expression.precision_bits,
                () -> -coefficient,
            ) for coefficient in expression.coefficients
        ],
        _owned_affine_eval(
            T,
            expression.precision_bits,
            () -> -expression.constant,
        ),
    )
end

function Base.:*(scalar::Number, expression::ScalarAffine{T}) where {T}
    converted = _owned_affine_scalar(T, expression.precision_bits, scalar)
    return ScalarAffine{T}(
        expression.model,
        expression.precision_bits,
        copy(expression.indices),
        [
            _owned_affine_eval(
                T,
                expression.precision_bits,
                () -> converted * coefficient,
            ) for coefficient in expression.coefficients
        ],
        _owned_affine_eval(
            T,
            expression.precision_bits,
            () -> converted * expression.constant,
        ),
    )
end

Base.:*(expression::ScalarAffine, scalar::Number) = scalar * expression
Base.:*(scalar::Number, entry::VariableEntry) = scalar * _entry_affine(entry)
Base.:*(entry::VariableEntry, scalar::Number) = scalar * entry

function Base.:+(expression::ScalarAffine{T}, constant::Number) where {T}
    converted = _owned_affine_scalar(T, expression.precision_bits, constant)
    return ScalarAffine{T}(
        expression.model,
        expression.precision_bits,
        copy(expression.indices),
        copy(expression.coefficients),
        _owned_affine_eval(
            T,
            expression.precision_bits,
            () -> expression.constant + converted,
        ),
    )
end

Base.:+(constant::Number, expression::ScalarAffine) = expression + constant
Base.:-(expression::ScalarAffine, constant::Number) = expression + (-constant)
Base.:-(constant::Number, expression::ScalarAffine) = constant + (-expression)
Base.:+(entry::VariableEntry, constant::Number) = _entry_affine(entry) + constant
Base.:+(constant::Number, entry::VariableEntry) = entry + constant
Base.:-(entry::VariableEntry, constant::Number) = _entry_affine(entry) - constant
Base.:-(constant::Number, entry::VariableEntry) = constant - _entry_affine(entry)

function Base.:*(matrix::AbstractMatrix, block::VariableBlockRef{T}) where {T}
    record = _variable_record(block)
    record.domain isa PSDCone && throw(ArgumentError("A*X is not a scalar-affine PSD operation"))
    size(matrix, 2) == record.length || throw(DimensionMismatch(
        "matrix has $(size(matrix,2)) columns, variable block has length $(record.length)",
    ))
    result = Vector{ScalarAffine{T}}(undef, size(matrix, 1))
    for row in axes(matrix, 1)
        builder = _affine_builder(block.model, size(matrix, 2))
        for column in axes(matrix, 2)
            coefficient = matrix[row, column]
            iszero(coefficient) && continue
            push_term!(builder, coefficient, _variable_global_index(block[column]))
        end
        result[row] = materialize(builder)
    end
    return result
end

function LinearAlgebra.dot(coefficients::AbstractVector, block::VariableBlockRef{T}) where {T}
    record = _variable_record(block)
    record.domain isa PSDCone && throw(ArgumentError("use dot(C, X) for a PSD variable"))
    length(coefficients) == record.length || throw(DimensionMismatch(
        "coefficient length $(length(coefficients)) != variable length $(record.length)",
    ))
    builder = _affine_builder(block.model, record.length)
    for index in 1:record.length
        coefficient = coefficients[index]
        iszero(coefficient) && continue
        push_term!(builder, coefficient, _variable_global_index(block[index]))
    end
    return materialize(builder)
end

function LinearAlgebra.dot(coefficients::AbstractMatrix, block::VariableBlockRef{T}) where {T}
    record = _variable_record(block)
    record.domain isa PSDCone || throw(ArgumentError("matrix dot is reserved for PSD variables"))
    size(coefficients) == (record.shape, record.shape) || throw(DimensionMismatch(
        "PSD coefficient size $(size(coefficients)) != $((record.shape, record.shape))",
    ))
    for column in 1:record.shape, row in column:record.shape
        left = coefficients[row, column]
        right = coefficients[column, row]
        left == right || throw(ArgumentError("PSD coefficient matrix must be exactly symmetric"))
        isfinite(left) || throw(ArgumentError("PSD coefficient matrix contains NaN or Inf"))
    end
    builder = _affine_builder(block.model, record.shape * (record.shape + 1) ÷ 2)
    for column in 1:record.shape, row in column:record.shape
        coefficient = coefficients[row, column]
        iszero(coefficient) && continue
        multiplier = row == column ? coefficient : _owned_affine_eval(
            T,
            precision_bits(block.model),
            () -> coefficient + coefficient,
        )
        push_term!(
            builder, multiplier, _variable_global_index(block[row, column]),
        )
    end
    return materialize(builder)
end

@inline function _as_affine(model::Model{T}, value) where {T<:AbstractFloat}
    if value isa ScalarAffine{T}
        return _owned_affine_copy(model, value)
    elseif value isa VariableEntry{T}
        value.model === model || throw(ArgumentError("variable belongs to a different model"))
        return _entry_affine(value)
    elseif value isa Number
        return _constant_affine(model, value)
    end
    throw(ArgumentError("expected a scalar affine value, got $(typeof(value))"))
end

function _affine_equal(left::ScalarAffine, right::ScalarAffine)
    return left.model == right.model &&
           left.precision_bits == right.precision_bits &&
           left.indices == right.indices &&
           left.coefficients == right.coefficients &&
           left.constant == right.constant
end
