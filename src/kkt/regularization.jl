# Structured signed-regularization policy for expanded KKT sessions.
# Included from expanded_quasidefinite.jl after KKTInertia is defined.

@enum ExpandedRegularizationStage::UInt8 begin
    EXPANDED_REGULARIZATION_NONE
    EXPANDED_REGULARIZATION_STATIC
    EXPANDED_REGULARIZATION_DYNAMIC
end

@enum ExpandedKKTAttemptReason::UInt8 begin
    EXPANDED_ATTEMPT_ACCEPTED
    EXPANDED_ATTEMPT_INERTIA_FACTOR_FAILED
    EXPANDED_ATTEMPT_WRONG_INERTIA
    EXPANDED_ATTEMPT_TINY_PIVOT
    EXPANDED_ATTEMPT_EXACT_FACTOR_FAILED
end

"""One auditable attempt in the expanded KKT regularization ladder."""
struct ExpandedKKTAttempt{T<:AbstractFloat}
    index::Int
    stage::ExpandedRegularizationStage
    regularization::T
    pivot_threshold::T
    minimum_pivot::T
    observed_inertia::KKTInertia
    reason::ExpandedKKTAttemptReason
end

@inline function _expanded_row_absolute_sum(matrix, row::Int)
    value = zero(eltype(matrix))
    @inbounds for column in axes(matrix, 2)
        value += abs(matrix[row, column])
    end
    return value
end

"""
Apply coordinate-wise dynamic signed regularization.  Every `x` shift is
positive and every `(y,tau)` shift is negative, so the correction itself has
the target quasidefinite inertia.  Row scaling keeps the perturbation tied to
the local operator norm; the failed pivot receives an additional floor.
"""
function _assemble_dynamic_signed_regularization!(
    destination::AbstractMatrix{T}, unregularized::AbstractMatrix{T},
    positive_dimension::Int, magnitude::T, operator_scale::T,
    pivot_floor::T, failed_pivot::Int,
) where {T<:AbstractFloat}
    copyto!(destination, unregularized)
    maximum_shift = zero(T)
    @inbounds for index in axes(destination, 1)
        local_scale = max(
            _expanded_row_absolute_sum(unregularized, index) / operator_scale,
            one(T),
        )
        shift = magnitude * local_scale
        if index == failed_pivot
            shift = max(shift, T(10) * pivot_floor)
        end
        maximum_shift = max(maximum_shift, shift)
        if index <= positive_dimension
            destination[index, index] += shift
        else
            destination[index, index] -= shift
        end
    end
    return maximum_shift
end
