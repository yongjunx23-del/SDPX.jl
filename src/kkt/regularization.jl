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
    EXPANDED_ATTEMPT_INERTIA_NOT_APPLICABLE
    EXPANDED_ATTEMPT_TINY_PIVOT
    EXPANDED_ATTEMPT_EXACT_FACTOR_FAILED
end

"""One refinement observation, keyed to its factorization attempt."""
struct ExpandedRefinementStep{T<:AbstractFloat}
    factor_attempt::Int
    iteration::Int
    residual_norm::T
    backward_error::T
    reduction_ratio::T
end

"""
    InertiaApplicabilityStatus

Whether the companion-inertia expectation is a *proven* structural fact for
the current `NewtonSystem` and regularization generation:

- `INERTIA_APPLICABLE`: every premise of the block-congruence argument holds,
  so an observed inertia mismatch is a genuine anomaly and retains the
  wrong-inertia protection (ladder advance, typed status, border retry);
- `INERTIA_NOT_APPLICABLE`: a premise is refuted for this system/generation
  (indefinite `(y,tau)` block, rank-deficient `x` reduction, violated bordered
  contract, or inconsistent regularization signs);
- `INERTIA_UNAVAILABLE`: the proof could not be completed (non-finite data,
  proof arithmetic, or an absent `NewtonSystem` in the recovery path).

In the last two cases the observed companion inertia is diagnostic only: an
exact direction is never rejected on it, and the unregularized nonsymmetric
backward error plus refinement remain the acceptance authority.
"""
@enum InertiaApplicabilityStatus::UInt8 begin
    INERTIA_APPLICABLE
    INERTIA_NOT_APPLICABLE
    INERTIA_UNAVAILABLE
end

"""One auditable attempt in the expanded KKT regularization ladder."""
struct ExpandedKKTAttempt{T<:AbstractFloat}
    index::Int
    stage::ExpandedRegularizationStage
    regularization::T
    pivot_threshold::T
    minimum_pivot::T
    observed_inertia::KKTInertia
    inertia_status::InertiaApplicabilityStatus
    reason::ExpandedKKTAttemptReason
end



"""
    InertiaApplicability

Typed receipt recording whether the companion-inertia expectation is
applicable for the current `NewtonSystem` and the regularization generation
that was actually applied.  The signed-regularized symmetric companion

    [ Dx   A'   c ]
    [ A   -G    -b ]
    [ c'  -b'  -s ]

with `G = H + Dy` (`H` = cone operator, `Dy` the applied `(y,tau)` shifts)
and `s = kappa/tau - shift_tau` has inertia `(n, m+1, 0)` by block
congruence exactly when:

- the `x` shifts `Dx` are nonnegative and `G` is positive definite
  (`sign_definite_blocks`);
- the reduced `x` block `W = Dx + A'G^-1 A` is positive definite (at zero
  `x` shifts this is the full-column-rank condition on `A`);
- the trailing bordered scalar is negative, i.e. the congruence contract
  `s > b'G^-1 b - c~'W^-1 c~` with `c~ = c - A'G^-1 b` holds
  (`congruence_contract`).

The expected inertia is recomputed from the actual dimensions and verified
sign convention of the current system.  Only when every premise is proven
(`INERTIA_APPLICABLE`) is an observed mismatch enforced as a protection; the
receipt is never inferred from the observed factor.
"""
struct InertiaApplicability{T<:AbstractFloat}
    status::InertiaApplicabilityStatus
    expected_inertia::KKTInertia
    sign_definite_blocks::Bool
    congruence_contract::Bool
    scaling_proven::Bool
    regularization_generation::Bool
    scale::T
    regularization::T
    reason::Symbol
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
    copy_owned!(destination, unregularized)
    maximum_shift = zero(T)
    @inbounds for index in axes(destination, 1)
        # `magnitude` is already scaled by the global operator norm.  Retain
        # the row fraction here (rather than clamping it to one, which made
        # every dynamic attempt identical to a static diagonal shift).  The
        # square-root-epsilon floor prevents a structurally zero row from
        # receiving a rounded zero shift.
        local_scale = max(
            _expanded_row_absolute_sum(unregularized, index) / operator_scale,
            sqrt(eps(T)),
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
