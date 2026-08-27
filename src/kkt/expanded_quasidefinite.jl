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

"""Generic diagonal-pivoted `L*D*L'` used for inertia certification."""
mutable struct GenericPivotedLDL{T<:AbstractFloat}
    schur::Matrix{T}
    L::Matrix{T}
    diagonal::Vector{T}
    permutation::Vector{Int}
    inertia::KKTInertia
    threshold::T
    success::Bool
end

function GenericPivotedLDL(::Type{T}, dimension::Int) where {T<:AbstractFloat}
    return GenericPivotedLDL{T}(
        zeros(T, dimension, dimension), Matrix{T}(I, dimension, dimension),
        zeros(T, dimension), collect(1:dimension), KKTInertia(0, 0, dimension),
        zero(T), false,
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
        if !(isfinite(selected_magnitude) && selected_magnitude > threshold)
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
        isfinite(pivot) && abs(pivot) > threshold || begin
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
            isfinite(factor.L[row, pivot_index]) || return false
        end
        for column in (pivot_index + 1):dimension
            coefficient = factor.L[column, pivot_index] * pivot
            for row in column:dimension
                value = factor.schur[row, column] -
                        factor.L[row, pivot_index] * coefficient
                isfinite(value) || return false
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
    success::Bool
end

function GenericPivotedLU(::Type{T}, dimension::Int) where {T<:AbstractFloat}
    return GenericPivotedLU{T}(
        zeros(T, dimension, dimension), collect(1:dimension), zero(T), false,
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
        isfinite(pivot_magnitude) && pivot_magnitude > threshold || return false
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
            isfinite(multiplier) || return false
            for j in (column + 1):dimension
                factor.factors[row, j] -= multiplier * factor.factors[column, j]
                isfinite(factor.factors[row, j]) || return false
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

mutable struct ExpandedKKTSession{T<:AbstractFloat}
    n::Int
    m::Int
    dimension::Int
    unregularized::Matrix{T}
    regularized::Matrix{T}
    symmetric_companion::Matrix{T}
    factor::GenericPivotedLU{T}
    inertia_factor::GenericPivotedLDL{T}
    expected_inertia::KKTInertia
    regularization::T
    regularization_attempts::Int
    refinements::Int
    residual::Matrix{T}
    correction::Matrix{T}
    status::ExpandedKKTStatus
end

function ExpandedKKTSession(::Type{T}, n::Int, m::Int; rhs_count::Int=2) where {T<:AbstractFloat}
    dimension = n + m + 1
    rhs_count >= 1 || throw(ArgumentError("expanded KKT requires at least one RHS"))
    return ExpandedKKTSession{T}(
        n, m, dimension, zeros(T, dimension, dimension),
        zeros(T, dimension, dimension), zeros(T, dimension, dimension),
        GenericPivotedLU(T, dimension), GenericPivotedLDL(T, dimension),
        KKTInertia(n, m + 1, 0), zero(T), 0, 0,
        zeros(T, dimension, rhs_count), zeros(T, dimension, rhs_count),
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
    fill!(K, zero(T))
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

function _assemble_regularized!(
    session::ExpandedKKTSession{T}, regularization::T,
) where {T<:AbstractFloat}
    copyto!(session.regularized, session.unregularized)
    n, m = session.n, session.m
    @inbounds for index in 1:n
        session.regularized[index, index] += regularization
    end
    @inbounds for index in (n + 1):(n + m + 1)
        session.regularized[index, index] -= regularization
    end
    # The symmetric companion mirrors the upper x/tau coupling. It is an
    # inertia diagnostic for signed regularization, not the solved frozen-sign
    # operator (whose lower x/tau block has the opposite sign).
    copyto!(session.symmetric_companion, session.regularized)
    tau_index = session.dimension
    @inbounds for index in 1:n
        session.symmetric_companion[tau_index, index] =
            session.symmetric_companion[index, tau_index]
    end
    return session.regularized
end

"""
    factor_expanded_kkt!(session, system)

Regularization ladder: unregularized attempt, then norm-scaled signed static
regularization increased by one decade per retry. A retry is accepted only if
(1) the symmetric companion has the exact expected inertia and (2) pivoted LU
of the exact frozen-sign operator succeeds. Wrong inertia is never accepted.
"""
function factor_expanded_kkt!(
    session::ExpandedKKTSession{T}, system::NewtonSystem{T};
    max_regularization_attempts::Int=6,
) where {T<:AbstractFloat}
    assemble_expanded_kkt!(session, system)
    scale = _expanded_operator_scale(session.unregularized)
    base_regularization = sqrt(eps(T)) * scale
    pivot_floor = T(32) * eps(T) * scale
    session.status = EXPANDED_KKT_FACTOR_FAILED
    session.regularization_attempts = 0
    for attempt in 0:max_regularization_attempts
        regularization = attempt == 0 ? zero(T) :
            base_regularization * T(10)^(attempt - 1)
        _assemble_regularized!(session, regularization)
        inertia_ok = factorize_pivoted_ldl!(
            session.inertia_factor, session.symmetric_companion;
            threshold=pivot_floor,
        ) && session.inertia_factor.inertia == session.expected_inertia
        if !inertia_ok
            session.status = EXPANDED_KKT_WRONG_INERTIA
            continue
        end
        if factorize_pivoted_lu!(
            session.factor, session.regularized; threshold=pivot_floor,
        )
            session.regularization = regularization
            session.regularization_attempts = attempt
            session.status = EXPANDED_KKT_FACTORED
            return true
        end
        session.status = EXPANDED_KKT_FACTOR_FAILED
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
    solved = solve_pivoted_lu!(destination, session.factor, rhs)
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

"""
    refine_expanded!(solution, session, rhs)

Reuse the same regularized factor for residual corrections, but form every
residual with the unregularized operator. Stagnation is fail-closed; a
regularized direction is accepted only after the unregularized backward-error
contract passes.
"""
function refine_expanded!(
    solution::AbstractVecOrMat{T}, session::ExpandedKKTSession{T},
    rhs::AbstractVecOrMat{T}; max_refinements::Int=4,
) where {T<:AbstractFloat}
    solution_matrix = solution isa AbstractVector ? reshape(solution, :, 1) : solution
    rhs_matrix = rhs isa AbstractVector ? reshape(rhs, :, 1) : rhs
    columns = size(rhs_matrix, 2)
    size(session.residual, 2) >= columns || throw(DimensionMismatch(
        "expanded refinement workspace has too few RHS columns",
    ))
    residual = @view session.residual[:, 1:columns]
    correction = @view session.correction[:, 1:columns]
    operator_scale = _expanded_operator_scale(session.unregularized)
    rhs_scale = max(_matrix_infinity_norm(rhs_matrix), one(T))
    solution_scale = max(_matrix_infinity_norm(solution_matrix), one(T))
    tolerance = T(256) * eps(T) *
                (operator_scale * solution_scale + rhs_scale)
    previous = T(Inf)
    session.refinements = 0
    for iteration in 0:max_refinements
        mul!(residual, session.unregularized, solution_matrix)
        @inbounds for index in eachindex(residual, rhs_matrix)
            residual[index] = rhs_matrix[index] - residual[index]
        end
        residual_norm = _matrix_infinity_norm(residual)
        isfinite(residual_norm) || begin
            session.status = EXPANDED_KKT_REFINEMENT_STAGNATED
            return false
        end
        if residual_norm <= tolerance
            session.status = EXPANDED_KKT_UNREGULARIZED_CERTIFIED
            session.refinements = iteration
            return true
        end
        iteration == max_refinements && break
        residual_norm < previous || begin
            session.status = EXPANDED_KKT_REFINEMENT_STAGNATED
            return false
        end
        previous = residual_norm
        solve_pivoted_lu!(correction, session.factor, residual) || begin
            session.status = EXPANDED_KKT_SOLVE_FAILED
            return false
        end
        @inbounds for index in eachindex(solution_matrix, correction)
            solution_matrix[index] += correction[index]
        end
    end
    session.status = EXPANDED_KKT_REFINEMENT_STAGNATED
    return false
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
