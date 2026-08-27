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
    rhs_scale = max(_matrix_infinity_norm(rhs), one(T))
    solution_scale = max(_matrix_infinity_norm(solution), one(T))
    denominator = operator_scale * solution_scale + rhs_scale
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
    rhs_scale = max(_matrix_infinity_norm(rhs), one(T))
    solution_scale = max(_matrix_infinity_norm(solution), one(T))
    denominator = operator_scale * solution_scale + rhs_scale
    target = T(256) * eps(T)
    normalized = residual_norm / denominator
    return ExpandedBackwardError(
        residual_norm, denominator, normalized, target,
    )
end
