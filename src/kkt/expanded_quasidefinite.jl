# Robust expanded HSD route.
#
# The frozen HSD signs make the exactly condensed operator nonsymmetric:
#
#   [ 0    A'    c ]
#   [ A    -H   -b ]
#   [-c'   -b'  -κ/τ].
#
# In particular the (x,tau) blocks are skew adjoints. The symmetric matrix in
# the architecture review was schematic; forcing its row-three sign would
# change the frozen homogeneous-gap equation. We therefore solve the exact
# operator with generic pivoted LU, while certifying the expected inertia of
# its signed-regularized symmetric quasidefinite companion. Every accepted
# direction is still checked against `NewtonSystem`, never this condensation.

@enum ExpandedKKTStatus::UInt8 begin
    EXPANDED_KKT_READY
    EXPANDED_KKT_FACTORED
    EXPANDED_KKT_FACTOR_FAILED
    EXPANDED_KKT_WRONG_INERTIA
    EXPANDED_KKT_SOLVE_FAILED
    EXPANDED_KKT_REFINEMENT_STAGNATED
    EXPANDED_KKT_REFINEMENT_AT_FLOOR
    EXPANDED_KKT_UNREGULARIZED_CERTIFIED
end

struct KKTInertia
    positive::Int
    negative::Int
    zero::Int
end

Base.:(==)(left::KKTInertia, right::KKTInertia) =
    left.positive == right.positive && left.negative == right.negative &&
    left.zero == right.zero

"""
    expected_expanded_inertia(system)

Return the inertia forced by the signed quasidefinite block structure.  The
`x` block receives the positive-semidefinite `Rx` contribution and the
combined `(y, tau)` block receives the negative-semidefinite `Ry`
contribution.  Signed regularization makes those contributions definite, so
a nonsingular companion has exactly `n` positive and `m + 1` negative
directions.  This authority is recomputed from each `NewtonSystem`; it is not
a caller-provided hint and is never inferred from the observed factor.
"""
function expected_expanded_inertia(system::NewtonSystem)
    m, n = size(system.A)
    return KKTInertia(n, m + 1, 0)
end

include("regularization.jl")

"""Generic diagonal-pivoted `L*D*L'` used for inertia certification."""
mutable struct GenericPivotedLDL{T<:AbstractFloat}
    schur::Matrix{T}
    L::Matrix{T}
    diagonal::Vector{T}
    permutation::Vector{Int}
    inertia::KKTInertia
    threshold::T
    minimum_pivot::T
    failed_pivot::Int
    success::Bool
end

function GenericPivotedLDL(::Type{T}, dimension::Int) where {T<:AbstractFloat}
    return GenericPivotedLDL{T}(
        zeros(T, dimension, dimension), Matrix{T}(I, dimension, dimension),
        zeros(T, dimension), collect(1:dimension), KKTInertia(0, 0, dimension),
        zero(T), T(Inf), 0, false,
    )
end

function _swap_symmetric_active!(matrix, first_index::Int, second_index::Int)
    first_index == second_index && return matrix
    @inbounds for column in axes(matrix, 2)
        matrix[first_index, column], matrix[second_index, column] =
            matrix[second_index, column], matrix[first_index, column]
    end
    @inbounds for row in axes(matrix, 1)
        matrix[row, first_index], matrix[row, second_index] =
            matrix[row, second_index], matrix[row, first_index]
    end
    return matrix
end

function factorize_pivoted_ldl!(
    factor::GenericPivotedLDL{T}, matrix::AbstractMatrix{T};
    threshold::T,
) where {T<:AbstractFloat}
    dimension = size(matrix, 1)
    size(matrix, 2) == dimension || throw(DimensionMismatch(
        "LDL factorization requires a square matrix",
    ))
    size(factor.schur) == size(matrix) || throw(DimensionMismatch(
        "LDL workspace dimension mismatch",
    ))
    issymmetric(matrix) || throw(ArgumentError(
        "LDL inertia authority requires an exactly symmetric matrix",
    ))
    copyto!(factor.schur, matrix)
    fill!(factor.L, zero(T))
    fill!(factor.diagonal, zero(T))
    @inbounds for index in 1:dimension
        factor.L[index, index] = one(T)
        factor.permutation[index] = index
    end
    factor.threshold = threshold
    factor.minimum_pivot = T(Inf)
    factor.failed_pivot = 0
    factor.success = false

    positive = 0
    negative = 0
    zeros_count = 0
    @inbounds for pivot_index in 1:dimension
        selected = pivot_index
        selected_magnitude = abs(factor.schur[pivot_index, pivot_index])
        for candidate in (pivot_index + 1):dimension
            magnitude = abs(factor.schur[candidate, candidate])
            if magnitude > selected_magnitude
                selected = candidate
                selected_magnitude = magnitude
            end
        end
        factor.minimum_pivot = min(factor.minimum_pivot, selected_magnitude)
        if !(isfinite(selected_magnitude) && selected_magnitude > threshold)
            factor.failed_pivot = pivot_index
            zeros_count += dimension - pivot_index + 1
            factor.inertia = KKTInertia(positive, negative, zeros_count)
            return false
        end
        if selected != pivot_index
            _swap_symmetric_active!(factor.schur, pivot_index, selected)
            factor.permutation[pivot_index], factor.permutation[selected] =
                factor.permutation[selected], factor.permutation[pivot_index]
            for column in 1:(pivot_index - 1)
                factor.L[pivot_index, column], factor.L[selected, column] =
                    factor.L[selected, column], factor.L[pivot_index, column]
            end
        end

        pivot = factor.schur[pivot_index, pivot_index]
        factor.minimum_pivot = min(factor.minimum_pivot, abs(pivot))
        isfinite(pivot) && abs(pivot) > threshold || begin
            factor.failed_pivot = pivot_index
            zeros_count += dimension - pivot_index + 1
            factor.inertia = KKTInertia(positive, negative, zeros_count)
            return false
        end
        factor.diagonal[pivot_index] = pivot
        if pivot > zero(T)
            positive += 1
        else
            negative += 1
        end
        for row in (pivot_index + 1):dimension
            factor.L[row, pivot_index] = factor.schur[row, pivot_index] / pivot
            isfinite(factor.L[row, pivot_index]) || begin
                factor.failed_pivot = pivot_index
                return false
            end
        end
        for column in (pivot_index + 1):dimension
            coefficient = factor.L[column, pivot_index] * pivot
            for row in column:dimension
                value = factor.schur[row, column] -
                        factor.L[row, pivot_index] * coefficient
                isfinite(value) || begin
                    factor.failed_pivot = pivot_index
                    return false
                end
                factor.schur[row, column] = value
                factor.schur[column, row] = value
            end
        end
    end
    factor.inertia = KKTInertia(positive, negative, zeros_count)
    factor.success = true
    return true
end

"""Generic row-pivoted LU factorization for the exact condensed operator."""
mutable struct GenericPivotedLU{T<:AbstractFloat}
    factors::Matrix{T}
    row_permutation::Vector{Int}
    threshold::T
    minimum_pivot::T
    failed_pivot::Int
    success::Bool
end

function GenericPivotedLU(::Type{T}, dimension::Int) where {T<:AbstractFloat}
    return GenericPivotedLU{T}(
        zeros(T, dimension, dimension), collect(1:dimension), zero(T),
        T(Inf), 0, false,
    )
end

function factorize_pivoted_lu!(
    factor::GenericPivotedLU{T}, matrix::AbstractMatrix{T}; threshold::T,
) where {T<:AbstractFloat}
    dimension = size(matrix, 1)
    size(matrix, 2) == dimension || throw(DimensionMismatch(
        "LU factorization requires a square matrix",
    ))
    size(factor.factors) == size(matrix) || throw(DimensionMismatch(
        "LU workspace dimension mismatch",
    ))
    copyto!(factor.factors, matrix)
    @inbounds for index in 1:dimension
        factor.row_permutation[index] = index
    end
    factor.threshold = threshold
    factor.minimum_pivot = T(Inf)
    factor.failed_pivot = 0
    factor.success = false

    @inbounds for column in 1:dimension
        pivot_row = column
        pivot_magnitude = abs(factor.factors[column, column])
        for row in (column + 1):dimension
            magnitude = abs(factor.factors[row, column])
            if magnitude > pivot_magnitude
                pivot_row = row
                pivot_magnitude = magnitude
            end
        end
        factor.minimum_pivot = min(factor.minimum_pivot, pivot_magnitude)
        isfinite(pivot_magnitude) && pivot_magnitude > threshold || begin
            factor.failed_pivot = column
            return false
        end
        if pivot_row != column
            for j in 1:dimension
                factor.factors[column, j], factor.factors[pivot_row, j] =
                    factor.factors[pivot_row, j], factor.factors[column, j]
            end
            factor.row_permutation[column], factor.row_permutation[pivot_row] =
                factor.row_permutation[pivot_row], factor.row_permutation[column]
        end
        pivot = factor.factors[column, column]
        for row in (column + 1):dimension
            factor.factors[row, column] /= pivot
            multiplier = factor.factors[row, column]
            isfinite(multiplier) || begin
                factor.failed_pivot = column
                return false
            end
            for j in (column + 1):dimension
                factor.factors[row, j] -= multiplier * factor.factors[column, j]
                isfinite(factor.factors[row, j]) || begin
                    factor.failed_pivot = column
                    return false
                end
            end
        end
    end
    factor.success = true
    return true
end

function solve_pivoted_lu!(
    destination::AbstractVecOrMat{T}, factor::GenericPivotedLU{T},
    rhs::AbstractVecOrMat{T},
) where {T<:AbstractFloat}
    factor.success || return false
    dimension = size(factor.factors, 1)
    size(rhs, 1) == dimension || throw(DimensionMismatch("LU RHS dimension mismatch"))
    size(destination) == size(rhs) || throw(DimensionMismatch(
        "LU destination/RHS dimensions disagree",
    ))
    rhs_matrix = rhs isa AbstractVector ? reshape(rhs, :, 1) : rhs
    destination_matrix = destination isa AbstractVector ?
        reshape(destination, :, 1) : destination
    columns = size(rhs_matrix, 2)
    # Apply the accumulated row permutation.
    @inbounds for column in 1:columns, row in 1:dimension
        destination_matrix[row, column] =
            rhs_matrix[factor.row_permutation[row], column]
    end
    # Unit-lower solve.
    @inbounds for column in 1:columns
        for row in 1:dimension
            value = destination_matrix[row, column]
            for j in 1:(row - 1)
                value -= factor.factors[row, j] * destination_matrix[j, column]
            end
            destination_matrix[row, column] = value
        end
        # Upper solve.
        for row in dimension:-1:1
            value = destination_matrix[row, column]
            for j in (row + 1):dimension
                value -= factor.factors[row, j] * destination_matrix[j, column]
            end
            value /= factor.factors[row, row]
            isfinite(value) || return false
            destination_matrix[row, column] = value
        end
    end
    return true
end

mutable struct ExpandedKKTSession{T<:AbstractFloat,B}
    n::Int
    m::Int
    dimension::Int
    unregularized::Matrix{T}
    regularized::Matrix{T}
    factor::GenericPivotedLU{T}
    inertia_factor::GenericPivotedLDL{T}
    la_backend::B
    provider_inertia_factor::Union{Nothing,AbstractLAFactorization{T}}
    provider_exact_factor::Union{Nothing,AbstractLAFactorization{T}}
    provider_inertia_matrix::Matrix{T}
    provider_exact_matrix::Matrix{T}
    expected_inertia::KKTInertia
    regularization::T
    regularization_attempts::Int
    attempts::Vector{ExpandedKKTAttempt{T}}
    matrix_epoch::Int
    factor_epoch::Int
    pattern_signature::UInt64
    numeric_factor_count::Int
    factor_receipt::Union{Nothing,FactorReceipt{T}}
    receipt_build_count::Int
    factor_attempt_count::Int
    operator_generation::Int
    factor_generation::Int
    refinement_trajectory::Vector{ExpandedRefinementStep{T}}
    refinements::Int
    refinement_recovery_attempts::Int
    unregularized_residual_norm::T
    backward_error::T
    backward_target::T
    attainable_floor::T
    at_arithmetic_floor::Bool
    residual_vector::Vector{T}
    correction_vector::Vector{T}
    residual::Matrix{T}
    correction::Matrix{T}
    # Compact predictor/corrector/refinement workspace. Every RHS, solution,
    # correction, recovery and cone-linearization buffer is sized once here;
    # the provider destructive scratch above remains provider-owned.
    predictor_rhs::Vector{T}
    corrector_rhs::Vector{T}
    solution::Vector{T}
    negated_primal::Vector{T}
    negated_dual::Vector{T}
    recovery_dx::Vector{T}
    recovery_dy::Vector{T}
    recovery_ds::Vector{T}
    recovery_cone_action::Vector{T}
    cone_operator::Matrix{T}
    cone_corrector_rhs::Vector{T}
    cone_block_ranges::Vector{UnitRange{Int}}
    newton_residual::NewtonResidual{T}
    # Layout metadata and solve counters. Diagnostic only; no numeric gate.
    rhs_count::Int
    predictor_solve_count::Int
    corrector_solve_count::Int
    recovery_count::Int
    status::ExpandedKKTStatus
end

function _expanded_la_backend(::Type{T}) where {T<:AbstractFloat}
    T === Float64 && return nothing
    (T === BigFloat || is_multifloat_arithmetic(T)) || throw(ArgumentError(
        "expanded KKT has no authorized factor provider for arithmetic $(T)",
    ))
    config = plan_la_backend(
        T; requested=:auto, route=:dense_augmented_ldlt,
    )
    backend = instantiate_la_backend(config, T)
    capabilities = la_backend_capabilities(backend)
    for capability in (
        :pivoted_symmetric_ldlt, :ldlt_inertia, :lu, :factor_solve,
        :multi_rhs,
    )
        la_provider_supports(capabilities, capability) || throw(ArgumentError(
            "expanded KKT provider lacks required capability $(capability)",
        ))
    end
    return backend
end

function ExpandedKKTSession(::Type{T}, n::Int, m::Int; rhs_count::Int=2) where {T<:AbstractFloat}
    dimension = n + m + 1
    rhs_count >= 1 || throw(ArgumentError("expanded KKT requires at least one RHS"))
    attempts = ExpandedKKTAttempt{T}[]
    sizehint!(attempts, 16)
    refinement_trajectory = ExpandedRefinementStep{T}[]
    sizehint!(refinement_trajectory, 128)
    backend = _expanded_la_backend(T)
    cone_ranges = UnitRange{Int}[1:m]
    return ExpandedKKTSession{T,typeof(backend)}(
        n, m, dimension, alloc_zeros(T, dimension, dimension),
        alloc_zeros(T, dimension, dimension),
        GenericPivotedLU(T, dimension), GenericPivotedLDL(T, dimension),
        backend, nothing, nothing,
        alloc_zeros(T, dimension, dimension),
        alloc_zeros(T, dimension, dimension),
        KKTInertia(0, 0, dimension), zero(T), 0, attempts,
        0, 0, dense_factor_pattern_signature(dimension, dimension, :expanded),
        0, nothing, 0, 0, 0, 0,
        refinement_trajectory, 0, 0,
        T(Inf), T(Inf), T(256) * eps(T), T(256) * eps(T), false,
        alloc_zeros(T, dimension), alloc_zeros(T, dimension),
        alloc_zeros(T, dimension, rhs_count),
        alloc_zeros(T, dimension, rhs_count),
        alloc_zeros(T, dimension), alloc_zeros(T, dimension),
        alloc_zeros(T, dimension),
        alloc_zeros(T, m), alloc_zeros(T, n),
        alloc_zeros(T, n), alloc_zeros(T, m),
        alloc_zeros(T, m), alloc_zeros(T, m),
        alloc_zeros(T, m, m), alloc_zeros(T, m), cone_ranges,
        NewtonResidual{T}(
            alloc_zeros(T, m), alloc_zeros(T, n), zero(T),
            alloc_zeros(T, m), zero(T), alloc_zeros(T, m),
        ),
        rhs_count, 0, 0, 0,
        EXPANDED_KKT_READY,
    )
end

@inline function _expanded_operator_scale(matrix)
    scale = zero(eltype(matrix))
    @inbounds for row in axes(matrix, 1)
        row_sum = zero(eltype(matrix))
        for column in axes(matrix, 2)
            row_sum += abs(matrix[row, column])
        end
        scale = max(scale, row_sum)
    end
    return max(scale, one(eltype(matrix)))
end

"""Assemble the exact unregularized condensation of `NewtonSystem`."""
function assemble_expanded_kkt!(
    session::ExpandedKKTSession{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    n, m = session.n, session.m
    size(system.A) == (m, n) || throw(DimensionMismatch(
        "expanded session/system dimensions disagree",
    ))
    K = session.unregularized
    # Every assembly is an owned operator rewrite.  Bump the mutation token
    # and revoke the receipt before any write, so a partial or non-finite
    # assembly can never leave the previous receipt current.
    session.matrix_epoch += 1
    session.operator_generation += 1
    session.factor_receipt = nothing
    zero_owned!(K)
    xrows = 1:n
    yrows = (n + 1):(n + m)
    tau_index = n + m + 1
    @inbounds for j in 1:n
        for i in 1:m
            value = system.A[i, j]
            K[j, n + i] = value
            K[n + i, j] = value
        end
        K[j, tau_index] = system.c[j]
        K[tau_index, j] = -system.c[j]
    end
    @inbounds for j in 1:m
        for i in 1:m
            K[n + i, n + j] = -system.cone.operator[i, j]
        end
        K[n + j, tau_index] = -system.b[j]
        K[tau_index, n + j] = -system.b[j]
    end
    K[tau_index, tau_index] = -system.kappa / system.tau
    all(isfinite, K) || throw(ArgumentError(
        "expanded KKT assembly produced non-finite data",
    ))
    return K
end

"""Build the exact condensed RHS without rederiving signs in a route."""
function expanded_rhs!(
    destination::AbstractVector{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    length(destination) == n + m + 1 || throw(DimensionMismatch(
        "expanded RHS dimension mismatch",
    ))
    @inbounds for j in 1:n
        destination[j] = system.rhs.dual_affine[j]
    end
    @inbounds for i in 1:m
        destination[n + i] = system.rhs.primal_affine[i] -
                             system.rhs.cone_corrector[i]
    end
    destination[end] = system.rhs.homogeneous_gap -
                       system.rhs.tau_kappa / system.tau
    return destination
end

function _freeze_symmetric_companion!(session::ExpandedKKTSession)
    # The symmetric companion mirrors the upper x/tau coupling. It is an
    # inertia diagnostic for signed regularization, not the solved frozen-sign
    # operator (whose lower x/tau block has the opposite sign). It is frozen
    # into the inertia-factor LDL workspace (standard route) or the
    # provider-owned inertia scratch (provider route); no long-lived duplicate
    # dimension-by-dimension matrix is retained.
    target = session.la_backend === nothing ?
        session.inertia_factor.schur : session.provider_inertia_matrix
    copy_owned!(target, session.regularized)
    tau_index = session.dimension
    @inbounds for index in 1:session.n
        target[tau_index, index] = target[index, tau_index]
    end
    return target
end

function _assemble_regularized!(
    session::ExpandedKKTSession{T}, regularization::T,
) where {T<:AbstractFloat}
    session.operator_generation += 1
    session.factor_receipt = nothing
    copy_owned!(session.regularized, session.unregularized)
    n, m = session.n, session.m
    @inbounds for index in 1:n
        session.regularized[index, index] += regularization
    end
    @inbounds for index in (n + 1):(n + m + 1)
        session.regularized[index, index] -= regularization
    end
    _freeze_symmetric_companion!(session)
    return session.regularized
end

function _assemble_dynamic_regularized!(
    session::ExpandedKKTSession{T}, magnitude::T, operator_scale::T,
    pivot_floor::T, failed_pivot::Int,
) where {T<:AbstractFloat}
    session.operator_generation += 1
    session.factor_receipt = nothing
    applied = _assemble_dynamic_signed_regularization!(
        session.regularized, session.unregularized, session.n, magnitude,
        operator_scale, pivot_floor, failed_pivot,
    )
    _freeze_symmetric_companion!(session)
    return applied
end

function _expanded_provider_inertia(inertia)
    counts = if inertia isa NamedTuple
        (inertia.positive, inertia.negative, inertia.zero)
    else
        length(inertia) == 3 || throw(ArgumentError(
            "expanded KKT provider returned invalid inertia",
        ))
        (inertia[1], inertia[2], inertia[3])
    end
    converted = KKTInertia(Int(counts[1]), Int(counts[2]), Int(counts[3]))
    min(converted.positive, converted.negative, converted.zero) >= 0 ||
        throw(ArgumentError("expanded KKT provider returned negative inertia"))
    return converted
end

function _factor_expanded_inertia!(
    session::ExpandedKKTSession{T}, pivot_floor::T,
) where {T<:AbstractFloat}
    if session.la_backend === nothing
        # The symmetric companion was frozen into the LDL workspace by
        # `_freeze_symmetric_companion!` during assembly; the factor routine
        # re-copies it in place (a no-op) before elimination.
        return factorize_pivoted_ldl!(
            session.inertia_factor, session.inertia_factor.schur;
            threshold=pivot_floor,
        )
    end
    # Provider route: the companion already lives in the provider-owned
    # inertia scratch; `la_ldlt_factor!` consumes it destructively there.
    provider_factor = la_ldlt_factor!(
        session.la_backend, session.provider_inertia_matrix,
    )
    session.provider_inertia_factor = provider_factor
    if provider_factor === nothing
        session.inertia_factor.inertia = KKTInertia(0, 0, session.dimension)
        session.inertia_factor.minimum_pivot = zero(T)
        session.inertia_factor.failed_pivot = 1
        session.inertia_factor.success = false
        return false
    end
    session.inertia_factor.inertia = _expanded_provider_inertia(
        la_ldlt_inertia(provider_factor),
    )
    session.inertia_factor.minimum_pivot = T(NaN)
    session.inertia_factor.failed_pivot = 0
    session.inertia_factor.success = true
    return true
end

function _factor_expanded_exact!(
    session::ExpandedKKTSession{T}, pivot_floor::T,
) where {T<:AbstractFloat}
    # Every exact-factor call replaces (or fails to replace) the numeric
    # factor.  Bump the factor generation and count the attempt before any
    # numeric work, so a failed attempt can never masquerade as the previous
    # certified factor.
    session.factor_generation += 1
    session.factor_attempt_count += 1
    session.factor_receipt = nothing
    success = if session.la_backend === nothing
        factorize_pivoted_lu!(
            session.factor, session.regularized; threshold=pivot_floor,
        )
    else
        copy_owned!(session.provider_exact_matrix, session.regularized)
        provider_factor = la_lu_factor!(
            session.la_backend, session.provider_exact_matrix,
        )
        session.provider_exact_factor = provider_factor
        session.factor.minimum_pivot = T(NaN)
        session.factor.failed_pivot = provider_factor === nothing ? 1 : 0
        session.factor.success = provider_factor !== nothing
        provider_factor !== nothing
    end
    if success
        session.factor_epoch += 1
        session.numeric_factor_count += 1
    end
    return success
end

@inline function _expanded_factor_receipt_current(
    session::ExpandedKKTSession{T},
) where {T<:AbstractFloat}
    provider = session.la_backend === nothing ?
        :standard_pivoted_lu : la_backend_provider(session.la_backend)
    return factor_receipt_owned(
        session.factor_receipt;
        matrix_epoch=session.matrix_epoch,
        factor_epoch=session.factor_epoch,
        pattern_signature=session.pattern_signature,
        route=:expanded,
        provider=provider,
        regularization=session.regularization,
        operator_generation=session.operator_generation,
        factor_generation=session.factor_generation,
    )
end

@inline function _build_expanded_factor_receipt!(
    session::ExpandedKKTSession{T},
) where {T<:AbstractFloat}
    provider = session.la_backend === nothing ?
        :standard_pivoted_lu : la_backend_provider(session.la_backend)
    session.factor_receipt = FactorReceipt(
        session.matrix_epoch,
        session.factor_epoch,
        session.pattern_signature,
        :expanded,
        provider,
        T,
        factor_receipt_precision(T),
        session.regularization,
        iszero(session.regularization) ? :none : :signed_diagonal,
        :factored,
        T(Inf),
        false,
        session.operator_generation,
        session.factor_generation,
    )
    session.receipt_build_count += 1
    return session.factor_receipt
end

function _solve_expanded_factor!(
    destination::AbstractVecOrMat{T}, session::ExpandedKKTSession{T},
    rhs::AbstractVecOrMat{T},
) where {T<:AbstractFloat}
    _expanded_factor_receipt_current(session) || return false
    if session.la_backend === nothing
        return solve_pivoted_lu!(destination, session.factor, rhs)
    end
    factor = session.provider_exact_factor
    factor === nothing && return false
    copy_owned!(destination, rhs)
    la_factor_solve!(factor, destination)
    return all(isfinite, destination)
end

@inline function _record_expanded_attempt!(
    session::ExpandedKKTSession{T}, index::Int,
    stage::ExpandedRegularizationStage, regularization::T,
    pivot_floor::T, minimum_pivot::T, reason::ExpandedKKTAttemptReason,
) where {T<:AbstractFloat}
    push!(session.attempts, ExpandedKKTAttempt(
        index, stage, regularization, pivot_floor, minimum_pivot,
        session.inertia_factor.inertia, reason,
    ))
    return reason
end

"""
    factor_expanded_kkt!(session, system)

Regularization ladder: planned unregularized factorization, norm-scaled signed
static retries, then coordinate-wise dynamic signed retries.  Every attempt is
recorded.  A retry is accepted only if (1) the symmetric companion has the
exact structure-derived inertia and (2) pivoted LU of the exact frozen-sign
operator succeeds.  Wrong inertia and tiny pivots always advance the ladder.
"""
function factor_expanded_kkt!(
    session::ExpandedKKTSession{T}, system::NewtonSystem{T};
    max_regularization_attempts::Int=6,
) where {T<:AbstractFloat}
    max_regularization_attempts >= 0 || throw(ArgumentError(
        "max_regularization_attempts must be nonnegative",
    ))
    assemble_expanded_kkt!(session, system)
    # Recompute the target from the semantic system on every factorization.
    # A stale or caller-mutated diagnostic can therefore never authorize a
    # wrong-inertia factor.
    session.expected_inertia = expected_expanded_inertia(system)
    scale = _expanded_operator_scale(session.unregularized)
    base_regularization = sqrt(eps(T)) * scale
    pivot_floor = T(32) * eps(T) * scale
    static_attempts = min(3, max_regularization_attempts)
    session.status = EXPANDED_KKT_FACTOR_FAILED
    session.regularization = zero(T)
    session.regularization_attempts = 0
    session.factor.success = false
    session.inertia_factor.success = false
    session.provider_inertia_factor = nothing
    session.provider_exact_factor = nothing
    empty!(session.attempts)
    empty!(session.refinement_trajectory)
    failed_pivot = 0

    for attempt in 0:max_regularization_attempts
        stage = if attempt == 0
            EXPANDED_REGULARIZATION_NONE
        elseif attempt <= static_attempts
            EXPANDED_REGULARIZATION_STATIC
        else
            EXPANDED_REGULARIZATION_DYNAMIC
        end
        stage_index = stage === EXPANDED_REGULARIZATION_DYNAMIC ?
            attempt - static_attempts - 1 : attempt - 1
        magnitude = attempt == 0 ? zero(T) :
            base_regularization * T(10)^stage_index
        regularization = if stage === EXPANDED_REGULARIZATION_DYNAMIC
            _assemble_dynamic_regularized!(
                session, magnitude, scale, pivot_floor, failed_pivot,
            )
        else
            _assemble_regularized!(session, magnitude)
            magnitude
        end
        session.regularization_attempts = attempt

        inertia_factored = _factor_expanded_inertia!(session, pivot_floor)
        if !inertia_factored
            failed_pivot = session.inertia_factor.failed_pivot
            reason = failed_pivot == 0 ?
                EXPANDED_ATTEMPT_INERTIA_FACTOR_FAILED :
                EXPANDED_ATTEMPT_TINY_PIVOT
            _record_expanded_attempt!(
                session, attempt, stage, regularization, pivot_floor,
                session.inertia_factor.minimum_pivot, reason,
            )
            session.status = EXPANDED_KKT_FACTOR_FAILED
            continue
        end
        if session.inertia_factor.inertia != session.expected_inertia
            failed_pivot = 0
            _record_expanded_attempt!(
                session, attempt, stage, regularization, pivot_floor,
                session.inertia_factor.minimum_pivot,
                EXPANDED_ATTEMPT_WRONG_INERTIA,
            )
            session.status = EXPANDED_KKT_WRONG_INERTIA
            continue
        end

        if !_factor_expanded_exact!(session, pivot_floor)
            failed_pivot = session.factor.failed_pivot
            reason = failed_pivot == 0 ?
                EXPANDED_ATTEMPT_EXACT_FACTOR_FAILED :
                EXPANDED_ATTEMPT_TINY_PIVOT
            _record_expanded_attempt!(
                session, attempt, stage, regularization, pivot_floor,
                min(session.inertia_factor.minimum_pivot,
                    session.factor.minimum_pivot), reason,
            )
            session.status = EXPANDED_KKT_FACTOR_FAILED
            continue
        end

        session.regularization = regularization
        session.status = EXPANDED_KKT_FACTORED
        _record_expanded_attempt!(
            session, attempt, stage, regularization, pivot_floor,
            min(session.inertia_factor.minimum_pivot,
                session.factor.minimum_pivot),
            EXPANDED_ATTEMPT_ACCEPTED,
        )
        _build_expanded_factor_receipt!(session)
        return true
    end
    # Preserve the strongest typed structural diagnosis across the exhausted
    # ladder; a later tiny pivot must not erase an observed wrong inertia.
    any(attempt -> attempt.reason === EXPANDED_ATTEMPT_WRONG_INERTIA,
        session.attempts) && (session.status = EXPANDED_KKT_WRONG_INERTIA)
    return false
end

"""Solve one or many RHS columns through the current single factorization."""
function solve_expanded!(
    destination::AbstractVecOrMat{T}, session::ExpandedKKTSession{T},
    rhs::AbstractVecOrMat{T},
) where {T<:AbstractFloat}
    session.status in (
        EXPANDED_KKT_FACTORED, EXPANDED_KKT_UNREGULARIZED_CERTIFIED,
    ) || return false
    solved = _solve_expanded_factor!(destination, session, rhs)
    solved || (session.status = EXPANDED_KKT_SOLVE_FAILED)
    return solved
end

@inline function _matrix_infinity_norm(matrix)
    value = zero(eltype(matrix))
    @inbounds for entry in matrix
        value = max(value, abs(entry))
    end
    return value
end

include("refinement.jl")

"""
Estimate the same-precision refinement floor without changing the strict
backward-error gate.  The pivot ratio is an operational condition proxy for
the accepted regularized solve.  Squaring that ratio and multiplying by a
Higham-style `gamma_k = k*eps/(1-k*eps)` gives a deliberately conservative
`gamma*kappa` budget.  It is diagnostic only: a direction above `256*eps(T)`
is never accepted, even when it is at this estimated arithmetic floor.
"""
function expanded_refinement_attainable_floor(
    session::ExpandedKKTSession{T},
) where {T<:AbstractFloat}
    scale = _expanded_operator_scale(session.unregularized)
    minimum_pivot = session.factor.minimum_pivot
    if !(isfinite(minimum_pivot) && minimum_pivot > zero(T))
        minimum_pivot = sqrt(eps(T)) * scale
    end
    pivot_ratio = max(one(T), scale / minimum_pivot)
    condition_proxy = min(inv(eps(T)), pivot_ratio * pivot_ratio)
    operations = T(4 * session.dimension + 8)
    denominator = one(T) - operations * eps(T)
    denominator > zero(T) || return one(T)
    gamma = operations * eps(T) / denominator
    return min(one(T), T(8) * gamma * condition_proxy)
end

@inline function _expanded_refinement_failure_status!(
    session::ExpandedKKTSession{T},
) where {T<:AbstractFloat}
    session.attainable_floor = expanded_refinement_attainable_floor(session)
    session.at_arithmetic_floor =
        isfinite(session.backward_error) &&
        session.backward_error <= session.attainable_floor
    session.status = session.at_arithmetic_floor ?
        EXPANDED_KKT_REFINEMENT_AT_FLOOR :
        EXPANDED_KKT_REFINEMENT_STAGNATED
    return false
end

function _try_expanded_dynamic_factor!(
    session::ExpandedKKTSession{T}, dynamic_index::Int,
) where {T<:AbstractFloat}
    scale = _expanded_operator_scale(session.unregularized)
    pivot_floor = T(32) * eps(T) * scale
    magnitude = sqrt(eps(T)) * scale * T(10)^dynamic_index
    failed_pivot = 0
    if !isempty(session.attempts)
        previous = session.attempts[end]
        previous.reason in (
            EXPANDED_ATTEMPT_TINY_PIVOT,
            EXPANDED_ATTEMPT_EXACT_FACTOR_FAILED,
        ) && (failed_pivot = session.factor.failed_pivot)
    end
    regularization = _assemble_dynamic_regularized!(
        session, magnitude, scale, pivot_floor, failed_pivot,
    )
    attempt = isempty(session.attempts) ? 0 : session.attempts[end].index + 1
    session.regularization_attempts = attempt

    if !_factor_expanded_inertia!(session, pivot_floor)
        reason = session.inertia_factor.failed_pivot == 0 ?
            EXPANDED_ATTEMPT_INERTIA_FACTOR_FAILED :
            EXPANDED_ATTEMPT_TINY_PIVOT
        _record_expanded_attempt!(
            session, attempt, EXPANDED_REGULARIZATION_DYNAMIC,
            regularization, pivot_floor, session.inertia_factor.minimum_pivot,
            reason,
        )
        session.status = EXPANDED_KKT_FACTOR_FAILED
        return false
    end
    if session.inertia_factor.inertia != session.expected_inertia
        _record_expanded_attempt!(
            session, attempt, EXPANDED_REGULARIZATION_DYNAMIC,
            regularization, pivot_floor, session.inertia_factor.minimum_pivot,
            EXPANDED_ATTEMPT_WRONG_INERTIA,
        )
        session.status = EXPANDED_KKT_WRONG_INERTIA
        return false
    end
    if !_factor_expanded_exact!(session, pivot_floor)
        reason = session.factor.failed_pivot == 0 ?
            EXPANDED_ATTEMPT_EXACT_FACTOR_FAILED :
            EXPANDED_ATTEMPT_TINY_PIVOT
        _record_expanded_attempt!(
            session, attempt, EXPANDED_REGULARIZATION_DYNAMIC,
            regularization, pivot_floor,
            min(session.inertia_factor.minimum_pivot,
                session.factor.minimum_pivot),
            reason,
        )
        session.status = EXPANDED_KKT_FACTOR_FAILED
        return false
    end
    session.regularization = regularization
    session.status = EXPANDED_KKT_FACTORED
    _record_expanded_attempt!(
        session, attempt, EXPANDED_REGULARIZATION_DYNAMIC,
        regularization, pivot_floor,
        min(session.inertia_factor.minimum_pivot, session.factor.minimum_pivot),
        EXPANDED_ATTEMPT_ACCEPTED,
    )
    _build_expanded_factor_receipt!(session)
    return true
end

function _refine_current_expanded!(
    solution::AbstractVecOrMat{T}, residual::AbstractVecOrMat{T},
    correction::AbstractVecOrMat{T}, session::ExpandedKKTSession{T},
    rhs::AbstractVecOrMat{T}, max_refinements::Int,
) where {T<:AbstractFloat}
    previous = T(Inf)
    session.refinements = 0
    session.unregularized_residual_norm = T(Inf)
    session.backward_error = T(Inf)
    session.backward_target = T(256) * eps(T)
    session.attainable_floor = expanded_refinement_attainable_floor(session)
    session.at_arithmetic_floor = false
    factor_attempt = isempty(session.attempts) ? -1 : session.attempts[end].index
    for iteration in 0:max_refinements
        report = if session.la_backend === nothing
            expanded_unregularized_backward_error!(
                residual, session.unregularized, solution, rhs,
            )
        else
            expanded_unregularized_backward_error!(
                residual, session.unregularized, solution, rhs,
                session.la_backend,
            )
        end
        ratio = isfinite(previous) ? report.residual_norm / previous : T(NaN)
        push!(session.refinement_trajectory, ExpandedRefinementStep(
            factor_attempt, iteration, report.residual_norm,
            report.normalized, ratio,
        ))
        session.unregularized_residual_norm = report.residual_norm
        session.backward_error = report.normalized
        session.backward_target = report.target
        isfinite(report.residual_norm) && isfinite(report.normalized) ||
            return _expanded_refinement_failure_status!(session)
        if report.normalized <= report.target
            session.status = EXPANDED_KKT_UNREGULARIZED_CERTIFIED
            session.refinements = iteration
            # The arithmetic floor is a failure diagnosis, never an alternate
            # success authority.  Certification here came from the unchanged
            # strict backward-error contract.
            session.at_arithmetic_floor = false
            return true
        end
        iteration == max_refinements && break
        report.residual_norm < previous ||
            return _expanded_refinement_failure_status!(session)
        previous = report.residual_norm
        _solve_expanded_factor!(correction, session, residual) || begin
            session.status = EXPANDED_KKT_SOLVE_FAILED
            return false
        end
        session.refinements = iteration + 1
        @inbounds for index in eachindex(solution, correction)
            solution[index] += correction[index]
        end
    end
    return _expanded_refinement_failure_status!(session)
end

function _refine_expanded_with_recovery!(
    solution::AbstractVecOrMat{T}, residual::AbstractVecOrMat{T},
    correction::AbstractVecOrMat{T}, session::ExpandedKKTSession{T},
    rhs::AbstractVecOrMat{T}, max_refinements::Int,
    max_dynamic_attempts::Int,
) where {T<:AbstractFloat}
    session.refinement_recovery_attempts = 0
    _refine_current_expanded!(
        solution, residual, correction, session, rhs, max_refinements,
    ) && return true
    max_dynamic_attempts == 0 && return false

    # Resume after an accepted static (or earlier dynamic) factor instead of
    # discarding a recoverable direction.  Every candidate is rebuilt from the
    # immutable unregularized operator, solved from the original RHS, and
    # independently re-refined against the original equations.
    dynamic_used = count(
        attempt -> attempt.stage === EXPANDED_REGULARIZATION_DYNAMIC,
        session.attempts,
    )
    for recovery in 0:(max_dynamic_attempts - 1)
        dynamic_index = dynamic_used + recovery
        _try_expanded_dynamic_factor!(session, dynamic_index) || continue
        session.refinement_recovery_attempts += 1
        _solve_expanded_factor!(solution, session, rhs) || begin
            session.status = EXPANDED_KKT_SOLVE_FAILED
            continue
        end
        _refine_current_expanded!(
            solution, residual, correction, session, rhs, max_refinements,
        ) && return true
    end
    return false
end

"""
    refine_expanded!(solution, session, rhs)

Reuse an accepted regularized factor for corrections while forming every
residual with the unregularized operator.  If strict refinement stagnates,
resume on progressively stronger dynamic signed regularization candidates.
The public direction remains fail-closed unless the original operator meets
the unchanged `256*eps(T)` backward-error contract.
"""
function refine_expanded!(
    solution::AbstractVector{T}, session::ExpandedKKTSession{T},
    rhs::AbstractVector{T}; max_refinements::Int=4,
    max_dynamic_attempts::Int=3,
) where {T<:AbstractFloat}
    max_refinements >= 0 || throw(ArgumentError(
        "max_refinements must be nonnegative",
    ))
    max_dynamic_attempts >= 0 || throw(ArgumentError(
        "max_dynamic_attempts must be nonnegative",
    ))
    length(solution) == length(rhs) == session.dimension ||
        throw(DimensionMismatch(
            "expanded refinement vector dimensions disagree",
        ))
    return _refine_expanded_with_recovery!(
        solution, session.residual_vector, session.correction_vector,
        session, rhs, max_refinements, max_dynamic_attempts,
    )
end

function refine_expanded!(
    solution::AbstractMatrix{T}, session::ExpandedKKTSession{T},
    rhs::AbstractMatrix{T}; max_refinements::Int=4,
    max_dynamic_attempts::Int=3,
) where {T<:AbstractFloat}
    max_refinements >= 0 || throw(ArgumentError(
        "max_refinements must be nonnegative",
    ))
    max_dynamic_attempts >= 0 || throw(ArgumentError(
        "max_dynamic_attempts must be nonnegative",
    ))
    size(solution) == size(rhs) || throw(DimensionMismatch(
        "expanded refinement solution/RHS dimensions disagree",
    ))
    size(solution, 1) == session.dimension || throw(DimensionMismatch(
        "expanded refinement panel dimension mismatch",
    ))
    columns = size(rhs, 2)
    size(session.residual, 2) >= columns || throw(DimensionMismatch(
        "expanded refinement workspace has too few RHS columns",
    ))
    residual = @view session.residual[:, 1:columns]
    correction = @view session.correction[:, 1:columns]
    return _refine_expanded_with_recovery!(
        solution, residual, correction, session, rhs, max_refinements,
        max_dynamic_attempts,
    )
end

"""Recover all five semantic direction variables from the condensed solve."""
function recover_expanded_direction(
    system::NewtonSystem{T}, condensed::AbstractVector{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    length(condensed) == n + m + 1 || throw(DimensionMismatch(
        "expanded solution dimension mismatch",
    ))
    dx = copy(@view condensed[1:n])
    dy = copy(@view condensed[(n + 1):(n + m)])
    dtau = condensed[end]
    cone_action = zeros(T, m)
    apply_cone_linearization!(cone_action, system.cone, dy)
    ds = similar(cone_action)
    @inbounds for i in 1:m
        ds[i] = system.rhs.cone_corrector[i] - cone_action[i]
    end
    dkappa = (system.rhs.tau_kappa - system.kappa * dtau) / system.tau
    return NewtonDirection(dx, dy, ds, dtau, dkappa)
end

"""
    recover_expanded_direction!(session, system, condensed)

Zero-allocation recovery of all five semantic direction variables from the
condensed solve. Numeric semantics are identical to `recover_expanded_direction`;
the results are written into the session-owned recovery buffers, which are
overwritten on every call, and wrapped in the immutable `NewtonDirection`.
"""
function recover_expanded_direction!(
    session::ExpandedKKTSession{T}, system::NewtonSystem{T},
    condensed::AbstractVector{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    length(condensed) == n + m + 1 || throw(DimensionMismatch(
        "expanded solution dimension mismatch",
    ))
    length(session.recovery_dx) == n || throw(DimensionMismatch(
        "expanded recovery dx dimension mismatch",
    ))
    length(session.recovery_dy) == m || throw(DimensionMismatch(
        "expanded recovery dy dimension mismatch",
    ))
    dx = session.recovery_dx
    dy = session.recovery_dy
    copyto!(dx, @view condensed[1:n])
    copyto!(dy, @view condensed[(n + 1):(n + m)])
    dtau = condensed[end]
    cone_action = session.recovery_cone_action
    apply_cone_linearization!(cone_action, system.cone, dy)
    ds = session.recovery_ds
    @inbounds for i in 1:m
        ds[i] = system.rhs.cone_corrector[i] - cone_action[i]
    end
    dkappa = (system.rhs.tau_kappa - system.kappa * dtau) / system.tau
    session.recovery_count += 1
    return NewtonDirection(dx, dy, ds, dtau, dkappa)
end

"""
    expanded_workspace_layout(session)

Layout metadata for the compact expanded workspace: sizes of every
preallocated predictor/corrector/refinement RHS, solution, correction,
recovery and cone-linearization buffer, plus the solve counters. Diagnostic
only; no numeric behavior is gated on this data.
"""
function expanded_workspace_layout(session::ExpandedKKTSession)
    return (;
        n=session.n, m=session.m, dimension=session.dimension,
        rhs_count=session.rhs_count,
        predictor_rhs=length(session.predictor_rhs),
        corrector_rhs=length(session.corrector_rhs),
        solution=length(session.solution),
        negated_primal=length(session.negated_primal),
        negated_dual=length(session.negated_dual),
        recovery_dx=length(session.recovery_dx),
        recovery_dy=length(session.recovery_dy),
        recovery_ds=length(session.recovery_ds),
        recovery_cone_action=length(session.recovery_cone_action),
        cone_operator=size(session.cone_operator),
        cone_corrector_rhs=length(session.cone_corrector_rhs),
        cone_block_ranges=length(session.cone_block_ranges),
        newton_residual_primal=length(session.newton_residual.primal_affine),
        newton_residual_dual=length(session.newton_residual.dual_affine),
        residual_panel=size(session.residual),
        correction_panel=size(session.correction),
        predictor_solve_count=session.predictor_solve_count,
        corrector_solve_count=session.corrector_solve_count,
        recovery_count=session.recovery_count,
    )
end
