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

function _affine_model(expression::ScalarAffine)
    return expression.model
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

function _canonical_affine(
    model::Model{T},
    indices::Vector{Int},
    coefficients::Vector{T},
    constant::T,
) where {T<:AbstractFloat}
    length(indices) == length(coefficients) || throw(DimensionMismatch(
        "affine indices and coefficients must have equal length",
    ))
    permutation = sortperm(indices)
    merged_indices = Int[]
    merged_coefficients = T[]
    for position in permutation
        index = indices[position]
        1 <= index <= num_variables(model) || throw(ArgumentError(
            "affine variable index $index is outside this model",
        ))
        coefficient = coefficients[position]
        isfinite(coefficient) || throw(ArgumentError("affine coefficient is NaN or Inf"))
        if !isempty(merged_indices) && merged_indices[end] == index
            merged_coefficients[end] = _owned_affine_eval(
                T,
                precision_bits(model),
                () -> merged_coefficients[end] + coefficient,
            )
            iszero(merged_coefficients[end]) && begin
                pop!(merged_indices)
                pop!(merged_coefficients)
            end
        elseif !iszero(coefficient)
            push!(merged_indices, index)
            push!(merged_coefficients, coefficient)
        end
    end
    isfinite(constant) || throw(ArgumentError("affine constant is NaN or Inf"))
    return ScalarAffine{T}(
        model_identity(model),
        precision_bits(model),
        merged_indices,
        merged_coefficients,
        constant,
    )
end

function _model_for_affine(expression::ScalarAffine{T}, candidates...) where {T}
    for candidate in candidates
        if candidate isa VariableEntry{T} &&
           model_identity(candidate.model) == expression.model
            return candidate.model
        end
    end
    throw(ArgumentError("the affine expression is not attached to the supplied model"))
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

function _sum_affines(expressions::AbstractVector{<:ScalarAffine{T}}) where {T}
    isempty(expressions) && throw(ArgumentError("cannot infer a model from an empty affine sum"))
    result = expressions[1]
    @inbounds for index in 2:length(expressions)
        result = result + expressions[index]
    end
    return result
end

function Base.:*(matrix::AbstractMatrix, block::VariableBlockRef{T}) where {T}
    record = _variable_record(block)
    record.domain isa PSDCone && throw(ArgumentError("A*X is not a scalar-affine PSD operation"))
    size(matrix, 2) == record.length || throw(DimensionMismatch(
        "matrix has $(size(matrix,2)) columns, variable block has length $(record.length)",
    ))
    result = Vector{ScalarAffine{T}}(undef, size(matrix, 1))
    for row in axes(matrix, 1)
        expression = _constant_affine(block.model, 0)
        for column in axes(matrix, 2)
            coefficient = matrix[row, column]
            iszero(coefficient) && continue
            expression = expression + coefficient * block[column]
        end
        result[row] = expression
    end
    return result
end

function LinearAlgebra.dot(coefficients::AbstractVector, block::VariableBlockRef{T}) where {T}
    record = _variable_record(block)
    record.domain isa PSDCone && throw(ArgumentError("use dot(C, X) for a PSD variable"))
    length(coefficients) == record.length || throw(DimensionMismatch(
        "coefficient length $(length(coefficients)) != variable length $(record.length)",
    ))
    expression = _constant_affine(block.model, 0)
    for index in 1:record.length
        coefficient = coefficients[index]
        iszero(coefficient) && continue
        expression = expression + coefficient * block[index]
    end
    return expression
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
    expression = _constant_affine(block.model, 0)
    for column in 1:record.shape, row in column:record.shape
        coefficient = coefficients[row, column]
        iszero(coefficient) && continue
        multiplier = row == column ? coefficient : _owned_affine_eval(
            T,
            precision_bits(block.model),
            () -> coefficient + coefficient,
        )
        expression = expression + multiplier * block[row, column]
    end
    return expression
end

@inline function _as_affine(model::Model{T}, value) where {T<:AbstractFloat}
    if value isa ScalarAffine{T}
        value.model == model_identity(model) || throw(ArgumentError(
            "affine expression belongs to a different model",
        ))
        return value
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
