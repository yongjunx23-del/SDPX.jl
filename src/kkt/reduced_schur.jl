# Sparse reduced-Schur HSD route.
#
# The frozen HSD Newton equations (see src/kkt/system.jl) are
#
#   primal : A*dx + ds - b*dτ = rP
#   dual   : A'*dy + c*dτ     = rD
#   gap    : -c'*dx - b'*dy + dκ = rG
#   cone   : ds + H*dy        = rC
#   tau    : κ*dτ + τ*dκ      = rT
#
# Eliminating (dy, ds) through block-diagonal H yields the sparse reduced
# system in (dx, dτ).  The numerical route below owns only a structural CSC
# pattern, per-block inverse work, and a SparseArrays factor.  It never forms
# the global dense inverse of H.  Every recovered direction remains subject to
# the authoritative five-equation Newton residual in the HSD caller.

using SparseArrays
using LinearAlgebra

"""Whether stdlib sparse LU preserves the working scalar type exactly."""
@inline sparse_schur_factorization_supported(::Type{T}) where {T} = T === Float64

# SparseArrays has no public condition-estimate accessor for UmfpackLU. Its
# bundled wrapper does define the SuiteSparse information index; isolate that
# non-public capability seam here instead of scattering a magic literal.
const _SPARSE_UMFPACK_RCOND_INFO_INDEX =
    SparseArrays.LibSuiteSparse.UMFPACK_RCOND + 1

"""Status of a sparse reduced-Schur factor/solve epoch."""
@enum SparseSchurStatus::UInt8 begin
    SPARSE_SCHUR_READY
    SPARSE_SCHUR_ASSEMBLED
    SPARSE_SCHUR_FACTORED
    SPARSE_SCHUR_FACTOR_FAILED
    SPARSE_SCHUR_SOLVE_FAILED
    SPARSE_SCHUR_REFINEMENT_STAGNATED
    SPARSE_SCHUR_REFINEMENT_AT_FLOOR
    SPARSE_SCHUR_UNREGULARIZED_CERTIFIED
end

"""
    SparseSchurSession{T}

Per-solve sparse reduced-Schur storage.  Julia's public `SparseArrays.lu` API
combines symbolic and numeric work; therefore `symbolic_reuse_supported` is
honestly false.  SDPX still freezes one deterministic CSC pattern per session
and reuses its `colptr`, `rowval`, `nzval`, RHS, residual and block workspaces.
`structural_assembly_count` must remain one while `numeric_factor_count`
increments once for each sparse predictor/corrector epoch.
"""
mutable struct SparseSchurSession{T<:AbstractFloat}
    n::Int
    m::Int
    dimension::Int
    schur::SparseMatrixCSC{T,Int}
    pattern_lookup::Dict{Tuple{Int,Int},Int}
    pattern_signature::UInt64
    symbolic_reuse_supported::Bool
    symbolic::Union{Nothing,Any}
    factor::Union{Nothing,Any}
    factor_numeric_epoch::Int
    factor_pattern_signature::UInt64
    factor_receipt::Union{Nothing,FactorReceipt{T}}
    receipt_build_count::Int
    rhs::Vector{T}
    solution_vector::Vector{T}
    residual_vector::Vector{T}
    correction_vector::Vector{T}
    block_inverses::Vector{Matrix{T}}
    block_augmented::Matrix{T}
    block_rhs_delta::Vector{T}
    block_hinv_b::Vector{T}
    block_hinv_delta::Vector{T}
    block_column_work::Vector{T}
    maximum_block_dimension::Int
    atb::Vector{T}
    at_delta::Vector{T}
    active_columns::Vector{Int}
    pattern_i::Vector{Int}
    pattern_j::Vector{Int}
    structural_assembly_count::Int
    numeric_assembly_count::Int
    rhs_assembly_count::Int
    numeric_factor_count::Int
    pattern_reuse_count::Int
    regularization::T
    reciprocal_condition::T
    condition_floor::T
    backward_error::T
    backward_target::T
    attainable_floor::T
    at_arithmetic_floor::Bool
    status::SparseSchurStatus
    last_reason::Symbol
end

function SparseSchurSession(::Type{T}, n::Int, m::Int) where {T<:AbstractFloat}
    n >= 0 || throw(ArgumentError("sparse Schur column dimension must be nonnegative"))
    m >= 0 || throw(ArgumentError("sparse Schur row dimension must be nonnegative"))
    dimension = n + 1
    return SparseSchurSession{T}(
        n, m, dimension,
        spzeros(T, dimension, dimension),
        Dict{Tuple{Int,Int},Int}(),
        zero(UInt64), false, nothing, nothing, 0, zero(UInt64), nothing, 0,
        zeros(T, dimension), zeros(T, dimension), zeros(T, dimension),
        zeros(T, dimension),
        Matrix{T}[], zeros(T, 0, 0),
        T[], T[], T[], T[], 0,
        zeros(T, n), zeros(T, n), Int[], Int[], Int[],
        0, 0, 0, 0, 0,
        zero(T), zero(T), sqrt(eps(T)),
        T(Inf), T(256) * eps(T), T(256) * eps(T), false,
        SPARSE_SCHUR_READY, :none,
    )
end

@inline function _invalidate_sparse_schur_factor!(
    session::SparseSchurSession,
    status::SparseSchurStatus,
    reason::Symbol,
)
    session.factor = nothing
    session.factor_numeric_epoch = 0
    session.factor_pattern_signature = zero(UInt64)
    session.factor_receipt = nothing
    session.status = status
    session.last_reason = reason
    return false
end

@inline function _schur_operator_scale(matrix::AbstractMatrix{T}) where {T}
    scale = zero(T)
    @inbounds for row in axes(matrix, 1)
        row_sum = zero(T)
        for column in axes(matrix, 2)
            row_sum += abs(matrix[row, column])
        end
        scale = max(scale, row_sum)
    end
    return max(scale, one(T))
end

@inline function _sparse_pattern_mix(signature::UInt64, value::Integer)
    signature ⊻= UInt64(value)
    return signature * UInt64(0x100000001b3)
end

"""Deterministic signature of A's structural nonzeros and cone block ranges."""
function _sparse_schur_source_signature(system::NewtonSystem)
    signature = UInt64(0xcbf29ce484222325)
    m, n = size(system.A)
    signature = _sparse_pattern_mix(signature, m)
    signature = _sparse_pattern_mix(signature, n)
    @inbounds for column in 1:n, row in 1:m
        iszero(system.A[row, column]) && continue
        signature = _sparse_pattern_mix(signature, row)
        signature = _sparse_pattern_mix(signature, column)
    end
    for block in system.cone.block_ranges
        signature = _sparse_pattern_mix(signature, first(block))
        signature = _sparse_pattern_mix(signature, last(block))
    end
    return signature
end

"""Invert one diagonal cone block into caller-owned storage."""
function _invert_cone_block(
    operator_block::AbstractMatrix{T}, inverse::AbstractMatrix{T},
    augmented_storage::AbstractMatrix{T},
) where {T<:AbstractFloat}
    dimension = size(operator_block, 1)
    size(inverse) == (dimension, dimension) || throw(DimensionMismatch(
        "cone block inverse dimension mismatch",
    ))
    size(augmented_storage, 1) >= dimension &&
    size(augmented_storage, 2) >= 2 * dimension || throw(DimensionMismatch(
        "cone block augmented workspace is too small",
    ))
    # This setup-sized augmented panel is local to one cone block, never m×m G.
    augmented = @view augmented_storage[1:dimension, 1:(2 * dimension)]
    fill!(augmented, zero(T))
    @inbounds for j in 1:dimension, i in 1:dimension
        augmented[i, j] = operator_block[i, j]
    end
    @inbounds for i in 1:dimension
        augmented[i, dimension + i] = one(T)
    end
    for pivot in 1:dimension
        best = pivot
        best_value = abs(augmented[pivot, pivot])
        for row in (pivot + 1):dimension
            candidate = abs(augmented[row, pivot])
            if candidate > best_value
                best_value = candidate
                best = row
            end
        end
        best_value > zero(T) || return false
        if best != pivot
            for column in 1:(2 * dimension)
                augmented[pivot, column], augmented[best, column] =
                    augmented[best, column], augmented[pivot, column]
            end
        end
        pivot_value = augmented[pivot, pivot]
        for column in 1:(2 * dimension)
            augmented[pivot, column] /= pivot_value
        end
        for row in 1:dimension
            row == pivot && continue
            multiplier = augmented[row, pivot]
            iszero(multiplier) && continue
            for column in 1:(2 * dimension)
                augmented[row, column] -= multiplier * augmented[pivot, column]
            end
        end
    end
    @inbounds for j in 1:dimension, i in 1:dimension
        inverse[i, j] = augmented[i, dimension + j]
    end
    return true
end

function _invert_cone_block(
    operator_block::AbstractMatrix{T}, inverse::AbstractMatrix{T},
) where {T<:AbstractFloat}
    dimension = size(operator_block, 1)
    return _invert_cone_block(
        operator_block, inverse, zeros(T, dimension, 2 * dimension),
    )
end

function _sparse_schur_active_columns!(
    destination::Vector{Int}, A::AbstractMatrix, rows::UnitRange{Int},
)
    empty!(destination)
    @inbounds for column in axes(A, 2)
        active = false
        for row in rows
            if !iszero(A[row, column])
                active = true
                break
            end
        end
        active && push!(destination, column)
    end
    return destination
end

"""Freeze one structural CSC pattern for the life of `session`."""
function _setup_sparse_schur_pattern!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    session.structural_assembly_count == 0 || return true
    validate_cone_linearization(system.cone)
    ranges = product_cone_block_ranges(system.cone)
    n = session.n
    empty!(session.pattern_i)
    empty!(session.pattern_j)
    for block in ranges
        active = _sparse_schur_active_columns!(
            session.active_columns, system.A, block,
        )
        for column in active, row in active
            push!(session.pattern_i, row)
            push!(session.pattern_j, column)
        end
    end
    # Border structure is frozen independently of current numerical zeros.
    for column in 1:n
        push!(session.pattern_i, column)
        push!(session.pattern_j, n + 1)
        push!(session.pattern_i, n + 1)
        push!(session.pattern_j, column)
    end
    push!(session.pattern_i, n + 1)
    push!(session.pattern_j, n + 1)
    marker = ones(T, length(session.pattern_i))
    session.schur = sparse(
        session.pattern_i, session.pattern_j, marker,
        session.dimension, session.dimension,
    )
    fill!(session.schur.nzval, zero(T))
    empty!(session.pattern_lookup)
    for column in 1:session.dimension
        for slot in nzrange(session.schur, column)
            session.pattern_lookup[(session.schur.rowval[slot], column)] = slot
        end
    end
    maximum_block_dimension = isempty(ranges) ? 0 : maximum(length, ranges)
    session.maximum_block_dimension = maximum_block_dimension
    session.block_inverses = [
        zeros(T, length(rows), length(rows)) for rows in ranges
    ]
    session.block_augmented = zeros(
        T, maximum_block_dimension, 2 * maximum_block_dimension,
    )
    session.block_rhs_delta = zeros(T, maximum_block_dimension)
    session.block_hinv_b = zeros(T, maximum_block_dimension)
    session.block_hinv_delta = zeros(T, maximum_block_dimension)
    session.block_column_work = zeros(T, maximum_block_dimension)
    session.pattern_signature = _sparse_schur_source_signature(system)
    session.structural_assembly_count = 1
    return true
end

@inline function _add_sparse_schur_value!(
    session::SparseSchurSession{T}, row::Int, column::Int, value::T,
) where {T}
    slot = get(session.pattern_lookup, (row, column), 0)
    slot == 0 && return false
    session.schur.nzval[slot] += value
    return true
end

@inline function _set_sparse_schur_value!(
    session::SparseSchurSession{T}, row::Int, column::Int, value::T,
) where {T}
    slot = get(session.pattern_lookup, (row, column), 0)
    slot == 0 && return false
    session.schur.nzval[slot] = value
    return true
end

"""Assemble numeric reduced operator values into the frozen CSC pattern."""
function assemble_sparse_schur_operator!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    n, m = session.n, session.m
    size(system.A) == (m, n) || throw(DimensionMismatch(
        "sparse session/system dimensions disagree",
    ))
    cone_dimension(system.cone) == m || throw(DimensionMismatch(
        "cone operator dimension does not match rows of A",
    ))
    validate_cone_linearization(system.cone)
    _setup_sparse_schur_pattern!(session, system)
    signature = _sparse_schur_source_signature(system)
    signature == session.pattern_signature ||
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_drift,
        )

    # A new numeric operator invalidates every prior factor before any write.
    _invalidate_sparse_schur_factor!(
        session, SPARSE_SCHUR_READY, :numeric_operator_changed,
    )
    session.numeric_assembly_count > 0 && (session.pattern_reuse_count += 1)
    fill!(session.schur.nzval, zero(T))
    fill!(session.atb, zero(T))
    atb = session.atb
    btHb = zero(T)
    ranges = product_cone_block_ranges(system.cone)

    for (block_index, block) in enumerate(ranges)
        dimension = length(block)
        block_H = product_cone_block_operator(system.cone, block_index)
        block_inverse = session.block_inverses[block_index]
        _invert_cone_block(
            block_H, block_inverse, session.block_augmented,
        ) || return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :cone_block_singular,
        )
        @inbounds for local_row in 1:dimension
            value_b = zero(T)
            for local_column in 1:dimension
                value_b += block_inverse[local_row, local_column] *
                           system.b[block[local_column]]
            end
            session.block_hinv_b[local_row] = value_b
        end
        active = _sparse_schur_active_columns!(
            session.active_columns, system.A, block,
        )
        for column_j in active
            @inbounds for local_row in 1:dimension
                value = zero(T)
                for local_column in 1:dimension
                    value += block_inverse[local_row, local_column] *
                             system.A[block[local_column], column_j]
                end
                session.block_column_work[local_row] = value
            end
            for column_i in active
                value = zero(T)
                @inbounds for local_row in 1:dimension
                    value += system.A[block[local_row], column_i] *
                             session.block_column_work[local_row]
                end
                _add_sparse_schur_value!(
                    session, column_i, column_j, value,
                ) || return _invalidate_sparse_schur_factor!(
                    session, SPARSE_SCHUR_FACTOR_FAILED,
                    :sparse_pattern_drift,
                )
            end
            value_b = zero(T)
            @inbounds for local_row in 1:dimension
                value_b += system.A[block[local_row], column_j] *
                           session.block_hinv_b[local_row]
            end
            atb[column_j] += value_b
        end
        @inbounds for local_row in 1:dimension
            btHb += system.b[block[local_row]] *
                    session.block_hinv_b[local_row]
        end
    end
    @inbounds for column in 1:n
        _set_sparse_schur_value!(
            session, column, n + 1, system.c[column] - atb[column],
        ) || return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_drift,
        )
        _set_sparse_schur_value!(
            session, n + 1, column, system.c[column] + atb[column],
        ) || return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_drift,
        )
    end
    _set_sparse_schur_value!(
        session, n + 1, n + 1,
        system.kappa / system.tau - btHb,
    ) || return _invalidate_sparse_schur_factor!(
        session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_drift,
    )
    all(isfinite, session.schur.nzval) ||
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED,
            :sparse_operator_nonfinite,
        )
    session.numeric_assembly_count += 1
    session.status = SPARSE_SCHUR_ASSEMBLED
    session.last_reason = :none
    return true
end

"""Assemble only the reduced RHS, reusing per-block inverse storage."""
function assemble_sparse_schur_rhs!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    session.numeric_assembly_count > 0 || throw(ArgumentError(
        "sparse Schur operator must be assembled before its RHS",
    ))
    validate_cone_linearization(system.cone)
    signature = _sparse_schur_source_signature(system)
    signature == session.pattern_signature ||
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_drift,
        )
    ranges = product_cone_block_ranges(system.cone)
    length(ranges) == length(session.block_inverses) ||
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_drift,
        )

    n = session.n
    fill!(session.at_delta, zero(T))
    at_delta = session.at_delta
    bt_delta = zero(T)
    for (block_index, block) in enumerate(ranges)
        dimension = length(block)
        block_inverse = session.block_inverses[block_index]
        @inbounds for local_row in 1:dimension
            delta = system.rhs.cone_corrector[block[local_row]] -
                    system.rhs.primal_affine[block[local_row]]
            session.block_rhs_delta[local_row] = delta
        end
        @inbounds for local_row in 1:dimension
            value = zero(T)
            for local_column in 1:dimension
                value += block_inverse[local_row, local_column] *
                         session.block_rhs_delta[local_column]
            end
            session.block_hinv_delta[local_row] = value
        end
        active = _sparse_schur_active_columns!(
            session.active_columns, system.A, block,
        )
        for column in active
            value = zero(T)
            @inbounds for local_row in 1:dimension
                value += system.A[block[local_row], column] *
                         session.block_hinv_delta[local_row]
            end
            at_delta[column] += value
        end
        @inbounds for local_row in 1:dimension
            bt_delta += system.b[block[local_row]] *
                        session.block_hinv_delta[local_row]
        end
    end
    @inbounds for column in 1:n
        session.rhs[column] = system.rhs.dual_affine[column] - at_delta[column]
    end
    session.rhs[n + 1] = -system.rhs.homogeneous_gap - bt_delta +
                         system.rhs.tau_kappa / system.tau
    all(isfinite, session.rhs) || begin
        _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_SOLVE_FAILED, :sparse_rhs_nonfinite,
        )
        return false
    end
    session.rhs_assembly_count += 1
    return true
end

"""Compatibility assembly: numeric operator plus RHS."""
function assemble_sparse_schur!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    assemble_sparse_schur_operator!(session, system) || return false
    return assemble_sparse_schur_rhs!(session, system)
end

"""
Factor the sparse operator.  Julia's stdlib UMFPACK route is accepted only for
`Float64`; `Float32` silently converts to Float64 and arbitrary precision is
unsupported, so both must fail closed into the caller's explicit route ladder.
"""
function factor_sparse_schur!(session::SparseSchurSession{T}) where {T<:AbstractFloat}
    # Clear stale ownership before capability checks or numeric work.
    session.factor = nothing
    session.factor_numeric_epoch = 0
    session.factor_pattern_signature = zero(UInt64)
    session.factor_receipt = nothing
    if !sparse_schur_factorization_supported(T)
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED,
            :sparse_factor_type_unsupported,
        )
    end
    session.status === SPARSE_SCHUR_ASSEMBLED ||
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED,
            :sparse_operator_not_assembled,
        )
    session.numeric_factor_count += 1
    try
        factor = lu(session.schur; check=false)
        LinearAlgebra.issuccess(factor) ||
            return _invalidate_sparse_schur_factor!(
                session, SPARSE_SCHUR_FACTOR_FAILED,
                :sparse_factor_singular,
            )
        # SuiteSparse UMFPACK publishes reciprocal condition in its numeric
        # info vector. SparseArrays exposes neither a public accessor nor a
        # standalone symbolic API, so this is a deliberately isolated stdlib
        # capability seam.
        info = getproperty(factor, :info)
        rcond_index = _SPARSE_UMFPACK_RCOND_INFO_INDEX
        if length(info) < rcond_index || !isfinite(info[rcond_index]) ||
           info[rcond_index] <= session.condition_floor
            session.reciprocal_condition = length(info) >= rcond_index ?
                T(info[rcond_index]) : zero(T)
            return _invalidate_sparse_schur_factor!(
                session, SPARSE_SCHUR_FACTOR_FAILED,
                :sparse_condition_rejected,
            )
        end
        session.reciprocal_condition = T(info[rcond_index])
        session.factor = factor
        session.symbolic = nothing
        session.factor_numeric_epoch = session.numeric_assembly_count
        session.factor_pattern_signature = session.pattern_signature
        session.status = SPARSE_SCHUR_FACTORED
        session.last_reason = :none
        session.factor_receipt = FactorReceipt(
            session.factor_numeric_epoch,
            session.numeric_factor_count,
            session.pattern_signature,
            :sparse_schur,
            :sparsearrays_umfpack,
            T,
            factor_receipt_precision(T),
            session.regularization,
            iszero(session.regularization) ? :none : :diagonal,
            :factored,
            T(Inf),
            false,
        )
        session.receipt_build_count += 1
        return true
    catch exception
        exception isa InterruptException && rethrow()
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_factor_failed,
        )
    end
end

function _sparse_schur_backward_error!(
    session::SparseSchurSession{T}, solution::AbstractVector{T},
) where {T<:AbstractFloat}
    mul!(session.residual_vector, session.schur, solution)
    @inbounds for index in eachindex(session.residual_vector)
        session.residual_vector[index] = session.rhs[index] -
                                         session.residual_vector[index]
    end
    residual_norm = norm(session.residual_vector, Inf)
    denominator = max(
        _schur_operator_scale(session.schur) * norm(solution, Inf) +
        norm(session.rhs, Inf), one(T),
    )
    session.backward_error = residual_norm / denominator
    session.at_arithmetic_floor = session.backward_error <= session.attainable_floor
    return session.backward_error
end

"""Solve and refine against the current unregularized sparse factor."""
function solve_sparse_schur!(
    session::SparseSchurSession{T}, solution::AbstractVector{T},
) where {T<:AbstractFloat}
    length(solution) == session.dimension || throw(DimensionMismatch(
        "sparse reduced solution dimension mismatch",
    ))
    current_factor = session.status === SPARSE_SCHUR_FACTORED &&
        session.factor !== nothing &&
        session.factor_numeric_epoch == session.numeric_assembly_count &&
        session.factor_pattern_signature == session.pattern_signature &&
        factor_receipt_owned(
            session.factor_receipt;
            matrix_epoch=session.factor_numeric_epoch,
            factor_epoch=session.numeric_factor_count,
            pattern_signature=session.pattern_signature,
            route=:sparse_schur,
            provider=:sparsearrays_umfpack,
            regularization=session.regularization,
        )
    current_factor || return _invalidate_sparse_schur_factor!(
        session, SPARSE_SCHUR_SOLVE_FAILED, :sparse_factor_stale,
    )
    try
        copyto!(solution, session.factor \ session.rhs)
    catch exception
        exception isa InterruptException && rethrow()
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_SOLVE_FAILED, :sparse_solve_failed,
        )
    end
    all(isfinite, solution) || return _invalidate_sparse_schur_factor!(
        session, SPARSE_SCHUR_SOLVE_FAILED, :sparse_solution_nonfinite,
    )
    previous = _sparse_schur_backward_error!(session, solution)
    previous <= session.backward_target && return true
    for _ in 1:3
        try
            copyto!(
                session.correction_vector,
                session.factor \ session.residual_vector,
            )
        catch exception
            exception isa InterruptException && rethrow()
            return _invalidate_sparse_schur_factor!(
                session, SPARSE_SCHUR_SOLVE_FAILED,
                :sparse_refinement_solve_failed,
            )
        end
        all(isfinite, session.correction_vector) ||
            return _invalidate_sparse_schur_factor!(
                session, SPARSE_SCHUR_SOLVE_FAILED,
                :sparse_refinement_nonfinite,
            )
        @inbounds for index in eachindex(solution)
            solution[index] += session.correction_vector[index]
        end
        current = _sparse_schur_backward_error!(session, solution)
        current <= session.backward_target && return true
        if current >= previous
            status = session.at_arithmetic_floor ?
                SPARSE_SCHUR_REFINEMENT_AT_FLOOR :
                SPARSE_SCHUR_REFINEMENT_STAGNATED
            reason = session.at_arithmetic_floor ?
                :sparse_refinement_at_floor : :sparse_refinement_stagnated
            return _invalidate_sparse_schur_factor!(session, status, reason)
        end
        previous = current
    end
    status = session.at_arithmetic_floor ?
        SPARSE_SCHUR_REFINEMENT_AT_FLOOR :
        SPARSE_SCHUR_REFINEMENT_STAGNATED
    reason = session.at_arithmetic_floor ?
        :sparse_refinement_at_floor : :sparse_refinement_budget_exhausted
    return _invalidate_sparse_schur_factor!(session, status, reason)
end

"""Recover a full semantic direction using caller-owned block inverses."""
function _recover_reduced_direction(
    system::NewtonSystem{T}, condensed::AbstractVector{T},
    block_inverses::AbstractVector{<:AbstractMatrix{T}},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    length(condensed) == n + 1 || throw(DimensionMismatch(
        "reduced solution dimension mismatch",
    ))
    ranges = product_cone_block_ranges(system.cone)
    length(block_inverses) == length(ranges) || throw(DimensionMismatch(
        "reduced recovery block inverse count mismatch",
    ))
    dx = copy(@view condensed[1:n])
    dtau = condensed[n + 1]
    rhs_for_dy = zeros(T, m)
    @inbounds for i in 1:m
        rhs_for_dy[i] = system.rhs.cone_corrector[i] -
                        system.rhs.primal_affine[i]
        for j in 1:n
            rhs_for_dy[i] += system.A[i, j] * dx[j]
        end
        rhs_for_dy[i] -= system.b[i] * dtau
    end
    dy = zeros(T, m)
    for (block_index, block) in enumerate(ranges)
        block_inverse = block_inverses[block_index]
        dimension = length(block)
        size(block_inverse) == (dimension, dimension) ||
            throw(DimensionMismatch(
                "reduced recovery block inverse dimensions disagree",
            ))
        @inbounds for local_row in 1:dimension
            value = zero(T)
            for local_column in 1:dimension
                value += block_inverse[local_row, local_column] *
                         rhs_for_dy[block[local_column]]
            end
            dy[block[local_row]] = value
        end
    end
    ds = zeros(T, m)
    apply_cone_linearization!(ds, system.cone, dy)
    @inbounds for i in 1:m
        ds[i] = system.rhs.cone_corrector[i] - ds[i]
    end
    dkappa = (system.rhs.tau_kappa - system.kappa * dtau) / system.tau
    return NewtonDirection(dx, dy, ds, dtau, dkappa)
end

function recover_reduced_direction(
    system::NewtonSystem{T}, condensed::AbstractVector{T},
    session::SparseSchurSession{T},
) where {T<:AbstractFloat}
    _sparse_schur_source_signature(system) == session.pattern_signature ||
        throw(ArgumentError("reduced recovery source pattern drift"))
    return _recover_reduced_direction(
        system, condensed, session.block_inverses,
    )
end

function recover_reduced_direction(
    system::NewtonSystem{T}, condensed::AbstractVector{T},
) where {T<:AbstractFloat}
    validate_cone_linearization(system.cone)
    ranges = product_cone_block_ranges(system.cone)
    maximum_block_dimension = isempty(ranges) ? 0 : maximum(length, ranges)
    augmented = zeros(
        T, maximum_block_dimension, 2 * maximum_block_dimension,
    )
    block_inverses = Matrix{T}[]
    for (block_index, rows) in enumerate(ranges)
        dimension = length(rows)
        inverse = zeros(T, dimension, dimension)
        operator = product_cone_block_operator(system.cone, block_index)
        _invert_cone_block(operator, inverse, augmented) ||
            throw(ArgumentError("reduced recovery cone block singular"))
        push!(block_inverses, inverse)
    end
    return _recover_reduced_direction(system, condensed, block_inverses)
end

"""
Compatibility one-shot route used by manufactured tests.  Production HSD
uses the split operator/RHS/factor methods to share one factor across its
predictor and corrector.
"""
function solve_sparse_schur!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
    solution::AbstractVector{T}, _direction::NewtonDirection{T},
) where {T<:AbstractFloat}
    assemble_sparse_schur!(session, system) || return nothing
    factor_sparse_schur!(session) || return nothing
    solve_sparse_schur!(session, solution) || return nothing
    recovered = try
        recover_reduced_direction(system, solution)
    catch exception
        exception isa InterruptException && rethrow()
        session.status = SPARSE_SCHUR_SOLVE_FAILED
        session.last_reason = :sparse_recovery_failed
        return nothing
    end
    # Compatibility callers retain the historical contract: recovery is
    # returned and the semantic five-equation gate is owned by the caller.
    # Production HSD performs that gate in `_product_hsd_sparse_solve_shift!`.
    session.status = SPARSE_SCHUR_UNREGULARIZED_CERTIFIED
    return recovered
end
