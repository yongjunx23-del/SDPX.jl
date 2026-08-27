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
    symmetric_companion::Matrix{T}
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
    refinements::Int
    unregularized_residual_norm::T
    backward_error::T
    backward_target::T
    residual_vector::Vector{T}
    correction_vector::Vector{T}
    residual::Matrix{T}
    correction::Matrix{T}
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
    backend = _expanded_la_backend(T)
    return ExpandedKKTSession{T,typeof(backend)}(
        n, m, dimension, alloc_zeros(T, dimension, dimension),
        alloc_zeros(T, dimension, dimension),
        alloc_zeros(T, dimension, dimension),
        GenericPivotedLU(T, dimension), GenericPivotedLDL(T, dimension),
        backend, nothing, nothing,
        alloc_zeros(T, dimension, dimension),
        alloc_zeros(T, dimension, dimension),
        KKTInertia(0, 0, dimension), zero(T), 0, attempts, 0,
        T(Inf), T(Inf), T(256) * eps(T),
        alloc_zeros(T, dimension), alloc_zeros(T, dimension),
        alloc_zeros(T, dimension, rhs_count),
        alloc_zeros(T, dimension, rhs_count),
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
    # operator (whose lower x/tau block has the opposite sign).
    copy_owned!(session.symmetric_companion, session.regularized)
    tau_index = session.dimension
    @inbounds for index in 1:session.n
        session.symmetric_companion[tau_index, index] =
            session.symmetric_companion[index, tau_index]
    end
    return session.symmetric_companion
end

function _assemble_regularized!(
    session::ExpandedKKTSession{T}, regularization::T,
) where {T<:AbstractFloat}
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
        return factorize_pivoted_ldl!(
            session.inertia_factor, session.symmetric_companion;
            threshold=pivot_floor,
        )
    end
    copy_owned!(session.provider_inertia_matrix, session.symmetric_companion)
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
    if session.la_backend === nothing
        return factorize_pivoted_lu!(
            session.factor, session.regularized; threshold=pivot_floor,
        )
    end
    copy_owned!(session.provider_exact_matrix, session.regularized)
    provider_factor = la_lu_factor!(
        session.la_backend, session.provider_exact_matrix,
    )
    session.provider_exact_factor = provider_factor
    session.factor.minimum_pivot = T(NaN)
    session.factor.failed_pivot = provider_factor === nothing ? 1 : 0
    session.factor.success = provider_factor !== nothing
    return provider_factor !== nothing
end

function _solve_expanded_factor!(
    destination::AbstractVecOrMat{T}, session::ExpandedKKTSession{T},
    rhs::AbstractVecOrMat{T},
) where {T<:AbstractFloat}
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
        return true
    end
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
    refine_expanded!(solution, session, rhs)

Reuse the same regularized factor for residual corrections, but form every
residual with the unregularized operator. Stagnation is fail-closed; a
regularized direction is accepted only after the unregularized backward-error
contract passes.
"""
function _refine_expanded!(
    solution::AbstractVecOrMat{T}, residual::AbstractVecOrMat{T},
    correction::AbstractVecOrMat{T}, session::ExpandedKKTSession{T},
    rhs::AbstractVecOrMat{T}, max_refinements::Int,
) where {T<:AbstractFloat}
    previous = T(Inf)
    session.refinements = 0
    session.unregularized_residual_norm = T(Inf)
    session.backward_error = T(Inf)
    session.backward_target = T(256) * eps(T)
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
        session.unregularized_residual_norm = report.residual_norm
        session.backward_error = report.normalized
        session.backward_target = report.target
        isfinite(report.residual_norm) && isfinite(report.normalized) || begin
            session.status = EXPANDED_KKT_REFINEMENT_STAGNATED
            return false
        end
        if report.normalized <= report.target
            session.status = EXPANDED_KKT_UNREGULARIZED_CERTIFIED
            session.refinements = iteration
            return true
        end
        iteration == max_refinements && break
        report.residual_norm < previous || begin
            session.status = EXPANDED_KKT_REFINEMENT_STAGNATED
            return false
        end
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
    session.status = EXPANDED_KKT_REFINEMENT_STAGNATED
    return false
end

function refine_expanded!(
    solution::AbstractVector{T}, session::ExpandedKKTSession{T},
    rhs::AbstractVector{T}; max_refinements::Int=4,
) where {T<:AbstractFloat}
    max_refinements >= 0 || throw(ArgumentError(
        "max_refinements must be nonnegative",
    ))
    length(solution) == length(rhs) == session.dimension ||
        throw(DimensionMismatch(
            "expanded refinement vector dimensions disagree",
        ))
    return _refine_expanded!(
        solution, session.residual_vector, session.correction_vector,
        session, rhs, max_refinements,
    )
end

function refine_expanded!(
    solution::AbstractMatrix{T}, session::ExpandedKKTSession{T},
    rhs::AbstractMatrix{T}; max_refinements::Int=4,
) where {T<:AbstractFloat}
    max_refinements >= 0 || throw(ArgumentError(
        "max_refinements must be nonnegative",
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
    return _refine_expanded!(
        solution, residual, correction, session, rhs, max_refinements,
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
