# Unregularized backward-error authority for expanded KKT solves.
# Included from expanded_quasidefinite.jl after norm helpers are defined.

"""Scale-separated result of an unregularized expanded KKT residual check."""
struct ExpandedBackwardError{T<:AbstractFloat}
    residual_norm::T
    denominator::T
    normalized::T
    target::T
end

"""
    expanded_unregularized_backward_error!(residual, operator, solution, rhs)

Form `rhs - operator*solution` in working precision and evaluate a strict
same-precision normwise backward error.  The `256*eps(T)` authority is an
internal factor/solve gate in the same spirit as `_psd_nt_close`; it is not a
user certificate tolerance and cannot promote a public status.
"""
function expanded_unregularized_backward_error!(
    residual::AbstractMatrix{T}, operator::AbstractMatrix{T},
    solution::AbstractMatrix{T}, rhs::AbstractMatrix{T},
) where {T<:AbstractFloat}
    size(operator, 1) == size(operator, 2) || throw(DimensionMismatch(
        "expanded backward-error operator must be square",
    ))
    size(solution) == size(rhs) == size(residual) || throw(DimensionMismatch(
        "expanded backward-error panel dimensions disagree",
    ))
    size(operator, 2) == size(solution, 1) || throw(DimensionMismatch(
        "expanded backward-error operator/panel dimensions disagree",
    ))
    mul!(residual, operator, solution)
    @inbounds for index in eachindex(residual, rhs)
        residual[index] = rhs[index] - residual[index]
    end
    residual_norm = _matrix_infinity_norm(residual)
    operator_scale = _expanded_operator_scale(operator)
    rhs_scale = _matrix_infinity_norm(rhs)
    solution_scale = _matrix_infinity_norm(solution)
    denominator = max(
        operator_scale * solution_scale + rhs_scale, one(T),
    )
    target = T(256) * eps(T)
    normalized = residual_norm / denominator
    return ExpandedBackwardError(
        residual_norm, denominator, normalized, target,
    )
end

function expanded_unregularized_backward_error!(
    residual::AbstractVector{T}, operator::AbstractMatrix{T},
    solution::AbstractVector{T}, rhs::AbstractVector{T},
) where {T<:AbstractFloat}
    size(operator, 1) == size(operator, 2) || throw(DimensionMismatch(
        "expanded backward-error operator must be square",
    ))
    length(solution) == length(rhs) == length(residual) ||
        throw(DimensionMismatch(
            "expanded backward-error vector dimensions disagree",
        ))
    size(operator, 2) == length(solution) || throw(DimensionMismatch(
        "expanded backward-error operator/vector dimensions disagree",
    ))
    mul!(residual, operator, solution)
    @inbounds for index in eachindex(residual, rhs)
        residual[index] = rhs[index] - residual[index]
    end
    residual_norm = _matrix_infinity_norm(residual)
    operator_scale = _expanded_operator_scale(operator)
    rhs_scale = _matrix_infinity_norm(rhs)
    solution_scale = _matrix_infinity_norm(solution)
    denominator = max(
        operator_scale * solution_scale + rhs_scale, one(T),
    )
    target = T(256) * eps(T)
    normalized = residual_norm / denominator
    return ExpandedBackwardError(
        residual_norm, denominator, normalized, target,
    )
end

function expanded_unregularized_backward_error!(
    residual::AbstractVecOrMat{T}, operator::AbstractMatrix{T},
    solution::AbstractVecOrMat{T}, rhs::AbstractVecOrMat{T},
    backend::AbstractLABackend,
) where {T<:AbstractFloat}
    size(solution) == size(rhs) == size(residual) || throw(DimensionMismatch(
        "expanded provider backward-error dimensions disagree",
    ))
    la_residual!(backend, :N, operator, solution, rhs, residual)
    normalized = T(la_normwise_backward_error(
        backend, :N, operator, solution, rhs, residual,
    ))
    residual_norm = _matrix_infinity_norm(residual)
    denominator = if iszero(normalized)
        max(
            _expanded_operator_scale(operator) *
            _matrix_infinity_norm(solution) + _matrix_infinity_norm(rhs),
            one(T),
        )
    else
        residual_norm / normalized
    end
    return ExpandedBackwardError(
        residual_norm, denominator, normalized, T(256) * eps(T),
    )
end
