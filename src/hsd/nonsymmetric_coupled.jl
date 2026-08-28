# Hybrid coupled Newton storage for product HSD systems containing one or
# more three-dimensional nonsymmetric blocks.  The matrix retains the
# nonsymmetric dual directions and eliminates only symmetric-cone rows.  A
# single pivoted LU is shared by predictor, corrector, and at most two
# refinement solves in one HSD epoch.

@enum ProductCoupledReason::UInt8 begin
    COUPLED_READY = 0x00
    COUPLED_ASSEMBLY_NONFINITE = 0x01
    COUPLED_FACTOR_FAILED = 0x02
    COUPLED_FACTOR_CERT_FAILED = 0x03
    COUPLED_EPOCH_MISMATCH = 0x04
    COUPLED_SOLVE_CERT_FAILED = 0x05
    COUPLED_REFINEMENT_STAGNATED = 0x06
    COUPLED_FIVE_EQUATION_FAILED = 0x07
    # Factor-coordinate failures are appended so the public values above
    # remain stable for callers which persisted the earlier diagnostics.
    COUPLED_TRANSFORM_FAILED = 0x08
    COUPLED_TRANSFORM_EPOCH_MISMATCH = 0x09
    COUPLED_TRANSFORM_ORDER_FAILED = 0x0a
    COUPLED_RECOVERY_FAILED = 0x0b
end

const _PRODUCT_COUPLED_TRANSFORM_ORDER = UInt8(0x01)

mutable struct NonsymmetricCoupledWorkspace{T}
    nr::Int
    nonsymmetric_dimension::Int
    dimension::Int
    offsets::Vector{Int}
    row_to_local::Vector{Int}
    matrix::Matrix{T}
    rhs::Vector{T}
    solution::Vector{T}
    correction::Vector{T}
    correction_rhs::Vector{T}
    residual::Vector{T}
    bound::Vector{T}
    permutation::Vector{Int}
    factor_error::Matrix{T}
    permuted_rhs::Vector{T}
    staged_y::Vector{T}
    forward_residual::Vector{T}
    backward_residual::Vector{T}
    identity_rhs::Vector{T}
    upper_work::Vector{T}
    lower_work::Vector{T}
    cache::LPLUCache{T}
    factor_certified::Bool
    last_reason::ProductCoupledReason
    solves::Int
    refinements::Int
    # Original K/q are retained as the authoritative physical system.  The
    # factor-coordinate buffers hold K̂/q̂ for the active LU, with
    # v=L'⋅dy and dy=L⁻ᵀ⋅v on each nonsymmetric three-row block.
    factor_coordinate_matrix::Matrix{T}
    factor_coordinate_rhs::Vector{T}
    factor_coordinate_factor::Matrix{T}
    physical_solution::Vector{T}
    transform_residual::Vector{T}
    transform_bound::Vector{T}
    recovery_residual::Vector{T}
    recovery_bound::Vector{T}
    transform_scratch::Vector{T}
    transform_block_scratch::Matrix{T}
    transform_valid::Bool
    factor_coordinate_rhs_valid::Bool
    transform_epoch::Int
    transform_order::UInt8
    # Route-local factor ownership: one immutable receipt per successfully
    # certified numeric factor, validated on every coupled solve.
    factor_receipt::Union{Nothing,FactorReceipt{T}}
    receipt_build_count::Int
    factor_attempt_count::Int
end

function NonsymmetricCoupledWorkspace(
    A::SparseMatrixCSC{T,Ti},
    nr::Integer,
    offsets_input::AbstractVector{<:Integer},
) where {T<:AbstractFloat,Ti<:Integer}
    rows = size(A, 1)
    nr_int = Int(nr)
    nr_int >= 0 || throw(ArgumentError("negative reduced dimension"))
    block_count = length(offsets_input)
    nsdim = 3 * block_count
    dimension = nr_int + nsdim + 2
    offsets = Vector{Int}(undef, block_count)
    row_to_local = zeros(Int, rows)
    @inbounds for block_index in 1:block_count
        offset = Int(offsets_input[block_index])
        1 <= offset && offset + 2 <= rows || throw(ArgumentError(
            "nonsymmetric coupled block has invalid offset $offset",
        ))
        offsets[block_index] = offset
        for coordinate in 1:3
            row = offset + coordinate - 1
            iszero(row_to_local[row]) || throw(ArgumentError(
                "overlapping nonsymmetric coupled rows at $row",
            ))
            row_to_local[row] = 3 * (block_index - 1) + coordinate
        end
    end
    return NonsymmetricCoupledWorkspace{T}(
        nr_int,
        nsdim,
        dimension,
        offsets,
        row_to_local,
        zeros(T, dimension, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        Vector{Int}(undef, dimension),
        zeros(T, dimension, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        LPLUCache{T}(dimension),
        false,
        COUPLED_READY,
        0,
        0,
        zeros(T, dimension, dimension),
        zeros(T, dimension),
        zeros(T, nsdim, nsdim),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, 3),
        zeros(T, 3, 3),
        false,
        false,
        0,
        _PRODUCT_COUPLED_TRANSFORM_ORDER,
        nothing,
        0,
        0,
    )
end

@inline function _product_coupled_gamma(::Type{T}, operations::Int) where {T}
    scaled = T(operations) * eps(one(T))
    isfinite(scaled) && zero(T) <= scaled < one(T) || return T(Inf)
    return scaled / (one(T) - scaled)
end

@inline function _product_coupled_zero_safe_close(
    residual::T, allowance::T,
) where {T}
    isfinite(residual) && isfinite(allowance) && allowance >= zero(T) ||
        return false
    iszero(allowance) && return iszero(residual)
    return abs(residual) <= allowance
end

@inline function _product_coupled_active_matrix(
    workspace::NonsymmetricCoupledWorkspace,
)
    return workspace.transform_valid ?
           workspace.factor_coordinate_matrix : workspace.matrix
end

@inline function _product_coupled_active_rhs(
    workspace::NonsymmetricCoupledWorkspace,
    rhs::AbstractVector,
)
    if workspace.transform_valid && rhs === workspace.rhs
        return workspace.factor_coordinate_rhs
    end
    return rhs
end

@inline function _product_coupled_transform_order_ok(
    workspace::NonsymmetricCoupledWorkspace,
)
    workspace.transform_order === _PRODUCT_COUPLED_TRANSFORM_ORDER ||
        return false
    nsdim = workspace.nonsymmetric_dimension
    length(workspace.offsets) * 3 == nsdim || return false
    workspace.dimension == workspace.nr + nsdim + 2 || return false
    rows = length(workspace.row_to_local)
    @inbounds for block_index in eachindex(workspace.offsets)
        offset = workspace.offsets[block_index]
        offset >= 1 && offset + 2 <= rows || return false
        local0 = 3 * (block_index - 1)
        for coordinate in 1:3
            workspace.row_to_local[offset + coordinate - 1] ==
                local0 + coordinate || return false
        end
    end
    @inbounds for local_row in 1:nsdim
        found = false
        for row in 1:rows
            if workspace.row_to_local[row] == local_row
                found = true
                break
            end
        end
        found || return false
    end
    return true
end

@inline function _product_coupled_factor_coordinate_finite(
    workspace::NonsymmetricCoupledWorkspace{T},
) where {T}
    L = workspace.factor_coordinate_factor
    nsdim = workspace.nonsymmetric_dimension
    size(L) == (nsdim, nsdim) || return false
    @inbounds for i in 1:nsdim, j in 1:nsdim
        value = L[i, j]
        isfinite(value) || return false
        if i < j
            iszero(value) || return false
        end
    end
    @inbounds for block_index in eachindex(workspace.offsets)
        local0 = 3 * (block_index - 1)
        for coordinate in 1:3
            value = L[local0 + coordinate, local0 + coordinate]
            isfinite(value) && value > zero(T) || return false
        end
    end
    return true
end

"""Copy and independently certify one accepted `Theta=L*L'` block."""
@inline function _product_coupled_copy_factor_block!(
    workspace::NonsymmetricCoupledWorkspace{T},
    local0::Int,
    source,
    theta,
) where {T}
    size(source) == (3, 3) && size(theta) == (3, 3) || return false
    L = workspace.factor_coordinate_factor
    @inbounds for j in 1:3, i in 1:3
        L[local0 + i, local0 + j] = source[i, j]
    end
    @inbounds for j in 1:3, i in 1:3
        value = L[local0 + i, local0 + j]
        isfinite(value) || return false
        i < j && !iszero(value) && return false
    end
    @inbounds for i in 1:3
        value = L[local0 + i, local0 + i]
        isfinite(value) && value > zero(T) || return false
    end
    gamma = _product_coupled_gamma(T, 32)
    isfinite(gamma) || return false
    @inbounds for j in 1:3
        for i in 1:3
            reconstruction = zero(T)
            reconstruction_work = zero(T)
            stop = min(i, j)
            for k in 1:stop
                term = L[local0 + i, local0 + k] *
                       L[local0 + j, local0 + k]
                reconstruction += term
                reconstruction_work += abs(term)
            end
            residual = theta[i, j] - reconstruction
            work = abs(theta[i, j]) + reconstruction_work
            allowance = gamma * work
            isfinite(residual) && isfinite(work) &&
                isfinite(allowance) || return false
            _product_coupled_zero_safe_close(residual, allowance) ||
                return false
            workspace.transform_residual[local0 + i] = residual
            workspace.transform_bound[local0 + i] = allowance
        end
    end
    return true
end

@inline function _product_coupled_lower_solve3!(
    workspace::NonsymmetricCoupledWorkspace{T}, local0::Int,
    a1::T, a2::T, a3::T,
) where {T}
    L = workspace.factor_coordinate_factor
    l11 = L[local0 + 1, local0 + 1]
    l21 = L[local0 + 2, local0 + 1]
    l22 = L[local0 + 2, local0 + 2]
    l31 = L[local0 + 3, local0 + 1]
    l32 = L[local0 + 3, local0 + 2]
    l33 = L[local0 + 3, local0 + 3]
    isfinite(l11) && isfinite(l21) && isfinite(l22) &&
        isfinite(l31) && isfinite(l32) && isfinite(l33) &&
        l11 > zero(T) && l22 > zero(T) && l33 > zero(T) || return false
    scratch = workspace.transform_scratch
    scratch[1] = a1 / l11
    scratch[2] = (a2 - l21 * scratch[1]) / l22
    scratch[3] = (a3 - l31 * scratch[1] - l32 * scratch[2]) / l33
    return isfinite(scratch[1]) && isfinite(scratch[2]) &&
           isfinite(scratch[3])
end

@inline function _product_coupled_transpose_solve3!(
    workspace::NonsymmetricCoupledWorkspace{T}, local0::Int,
    a1::T, a2::T, a3::T,
) where {T}
    L = workspace.factor_coordinate_factor
    l11 = L[local0 + 1, local0 + 1]
    l21 = L[local0 + 2, local0 + 1]
    l22 = L[local0 + 2, local0 + 2]
    l31 = L[local0 + 3, local0 + 1]
    l32 = L[local0 + 3, local0 + 2]
    l33 = L[local0 + 3, local0 + 3]
    isfinite(l11) && isfinite(l21) && isfinite(l22) &&
        isfinite(l31) && isfinite(l32) && isfinite(l33) &&
        l11 > zero(T) && l22 > zero(T) && l33 > zero(T) || return false
    scratch = workspace.transform_scratch
    scratch[3] = a3 / l33
    scratch[2] = (a2 - l32 * scratch[3]) / l22
    scratch[1] = (a1 - l21 * scratch[2] - l31 * scratch[3]) / l11
    return isfinite(scratch[1]) && isfinite(scratch[2]) &&
           isfinite(scratch[3])
end

@inline function _product_coupled_triangular_certificate!(
    workspace::NonsymmetricCoupledWorkspace{T}, local0::Int,
    a1::T, a2::T, a3::T, transpose_action::Bool,
) where {T}
    L = workspace.factor_coordinate_factor
    x = workspace.transform_scratch
    gamma = _product_coupled_gamma(T, 32)
    isfinite(gamma) || return false
    @inbounds for i in 1:3
        lhs = zero(T)
        work = zero(T)
        if transpose_action
            lhs = i == 1 ?
                  L[local0 + 1, local0 + 1] * x[1] +
                  L[local0 + 2, local0 + 1] * x[2] +
                  L[local0 + 3, local0 + 1] * x[3] :
                  i == 2 ?
                  L[local0 + 2, local0 + 2] * x[2] +
                  L[local0 + 3, local0 + 2] * x[3] :
                  L[local0 + 3, local0 + 3] * x[3]
            work = i == 1 ?
                   abs(a1) + abs(L[local0 + 1, local0 + 1] * x[1]) +
                   abs(L[local0 + 2, local0 + 1] * x[2]) +
                   abs(L[local0 + 3, local0 + 1] * x[3]) :
                   i == 2 ?
                   abs(a2) + abs(L[local0 + 2, local0 + 2] * x[2]) +
                   abs(L[local0 + 3, local0 + 2] * x[3]) :
                   abs(a3) + abs(L[local0 + 3, local0 + 3] * x[3])
        else
            lhs = i == 1 ?
                  L[local0 + 1, local0 + 1] * x[1] :
                  i == 2 ?
                  L[local0 + 2, local0 + 1] * x[1] +
                  L[local0 + 2, local0 + 2] * x[2] :
                  L[local0 + 3, local0 + 1] * x[1] +
                  L[local0 + 3, local0 + 2] * x[2] +
                  L[local0 + 3, local0 + 3] * x[3]
            work = i == 1 ?
                   abs(a1) + abs(L[local0 + 1, local0 + 1] * x[1]) :
                   i == 2 ?
                   abs(a2) + abs(L[local0 + 2, local0 + 1] * x[1]) +
                   abs(L[local0 + 2, local0 + 2] * x[2]) :
                   abs(a3) + abs(L[local0 + 3, local0 + 1] * x[1]) +
                   abs(L[local0 + 3, local0 + 2] * x[2]) +
                   abs(L[local0 + 3, local0 + 3] * x[3])
        end
        target = i == 1 ? a1 : i == 2 ? a2 : a3
        residual = lhs - target
        allowance = gamma * work
        isfinite(residual) && isfinite(work) && isfinite(allowance) ||
            return false
        _product_coupled_zero_safe_close(residual, allowance) ||
            return false
        index = local0 + i
        workspace.transform_residual[index] = residual
        workspace.transform_bound[index] = allowance
    end
    return true
end

@inline function _product_coupled_prepare_factor_coordinates!(
    workspace::NonsymmetricCoupledWorkspace{T}, epoch::Int,
) where {T}
    workspace.transform_valid = false
    workspace.factor_coordinate_rhs_valid = false
    workspace.transform_epoch = epoch
    _product_coupled_transform_order_ok(workspace) || begin
        workspace.last_reason = COUPLED_TRANSFORM_ORDER_FAILED
        return false
    end
    _product_coupled_factor_coordinate_finite(workspace) || begin
        workspace.last_reason = COUPLED_TRANSFORM_FAILED
        return false
    end
    copyto!(workspace.factor_coordinate_matrix, workspace.matrix)
    # Left transform: every C_N row block is premultiplied by L^-1.
    @inbounds for block_index in eachindex(workspace.offsets)
        local0 = 3 * (block_index - 1)
        rows = local0 + 1:local0 + 3
        for column in 1:workspace.dimension
            old1 = workspace.factor_coordinate_matrix[rows[1], column]
            old2 = workspace.factor_coordinate_matrix[rows[2], column]
            old3 = workspace.factor_coordinate_matrix[rows[3], column]
            _product_coupled_lower_solve3!(
                workspace, local0, old1, old2, old3,
            ) || begin
                workspace.last_reason = COUPLED_TRANSFORM_FAILED
                return false
            end
            workspace.factor_coordinate_matrix[rows[1], column] =
                workspace.transform_scratch[1]
            workspace.factor_coordinate_matrix[rows[2], column] =
                workspace.transform_scratch[2]
            workspace.factor_coordinate_matrix[rows[3], column] =
                workspace.transform_scratch[3]
            _product_coupled_triangular_certificate!(
                workspace, local0, old1, old2, old3, false,
            ) || begin
                workspace.last_reason = COUPLED_TRANSFORM_FAILED
                return false
            end
        end
    end
    # Right transform: dy columns are multiplied by L^-T.  Equivalently,
    # solve L*w = a for each transformed row vector a.
    @inbounds for block_index in eachindex(workspace.offsets)
        local0 = 3 * (block_index - 1)
        cols = workspace.nr + local0 + 1:workspace.nr + local0 + 3
        for row in 1:workspace.dimension
            old1 = workspace.factor_coordinate_matrix[row, cols[1]]
            old2 = workspace.factor_coordinate_matrix[row, cols[2]]
            old3 = workspace.factor_coordinate_matrix[row, cols[3]]
            _product_coupled_lower_solve3!(
                workspace, local0, old1, old2, old3,
            ) || begin
                workspace.last_reason = COUPLED_TRANSFORM_FAILED
                return false
            end
            workspace.factor_coordinate_matrix[row, cols[1]] =
                workspace.transform_scratch[1]
            workspace.factor_coordinate_matrix[row, cols[2]] =
                workspace.transform_scratch[2]
            workspace.factor_coordinate_matrix[row, cols[3]] =
                workspace.transform_scratch[3]
            _product_coupled_triangular_certificate!(
                workspace, local0, old1, old2, old3, false,
            ) || begin
                workspace.last_reason = COUPLED_TRANSFORM_FAILED
                return false
            end
        end
    end
    @inbounds for value in workspace.factor_coordinate_matrix
        isfinite(value) || begin
            workspace.last_reason = COUPLED_TRANSFORM_FAILED
            return false
        end
    end
    workspace.transform_valid = true
    workspace.last_reason = COUPLED_READY
    return true
end

@inline function _product_coupled_transform_rhs!(
    workspace::NonsymmetricCoupledWorkspace{T},
) where {T}
    workspace.transform_valid || return false
    _product_coupled_transform_order_ok(workspace) || return false
    copyto!(workspace.factor_coordinate_rhs, workspace.rhs)
    @inbounds for block_index in eachindex(workspace.offsets)
        local0 = 3 * (block_index - 1)
        row0 = local0 + 1
        old1 = workspace.rhs[row0]
        old2 = workspace.rhs[row0 + 1]
        old3 = workspace.rhs[row0 + 2]
        _product_coupled_lower_solve3!(
            workspace, local0, old1, old2, old3,
        ) || return false
        workspace.factor_coordinate_rhs[row0] = workspace.transform_scratch[1]
        workspace.factor_coordinate_rhs[row0 + 1] = workspace.transform_scratch[2]
        workspace.factor_coordinate_rhs[row0 + 2] = workspace.transform_scratch[3]
        _product_coupled_triangular_certificate!(
            workspace, local0, old1, old2, old3, false,
        ) || return false
    end
    @inbounds for value in workspace.factor_coordinate_rhs
        isfinite(value) || return false
    end
    workspace.factor_coordinate_rhs_valid = true
    return true
end

@inline function _product_coupled_transform_entry_close(
    actual::T, expected::T, gamma::T,
) where {T}
    work = abs(actual) + abs(expected)
    residual = actual - expected
    isfinite(work) && isfinite(residual) || return false
    return _product_coupled_zero_safe_close(residual, gamma * work)
end

"""Recompute `R*K*C` from the retained physical matrix and certify K̂."""
@inline function _product_coupled_factor_coordinate_matrix_certificate!(
    workspace::NonsymmetricCoupledWorkspace{T},
) where {T}
    workspace.transform_valid || return false
    _product_coupled_transform_order_ok(workspace) || return false
    _product_coupled_factor_coordinate_finite(workspace) || return false
    K = workspace.matrix
    Khat = workspace.factor_coordinate_matrix
    n = workspace.dimension
    nsdim = workspace.nonsymmetric_dimension
    nr = workspace.nr
    gamma = _product_coupled_gamma(T, 64)
    isfinite(gamma) || return false
    dtau = nr + nsdim + 1
    dkappa = dtau + 1

    # Non-dy columns of nonsymmetric rows only receive the left transform.
    @inbounds for block_index in eachindex(workspace.offsets)
        row0 = 3 * (block_index - 1)
        for column in 1:nr
            a1 = K[row0 + 1, column]
            a2 = K[row0 + 2, column]
            a3 = K[row0 + 3, column]
            _product_coupled_lower_solve3!(
                workspace, row0, a1, a2, a3,
            ) || return false
            _product_coupled_transform_entry_close(
                Khat[row0 + 1, column], workspace.transform_scratch[1], gamma,
            ) || return false
            _product_coupled_transform_entry_close(
                Khat[row0 + 2, column], workspace.transform_scratch[2], gamma,
            ) || return false
            _product_coupled_transform_entry_close(
                Khat[row0 + 3, column], workspace.transform_scratch[3], gamma,
            ) || return false
        end
        for column in (dtau, dkappa)
            a1 = K[row0 + 1, column]
            a2 = K[row0 + 2, column]
            a3 = K[row0 + 3, column]
            _product_coupled_lower_solve3!(
                workspace, row0, a1, a2, a3,
            ) || return false
            _product_coupled_transform_entry_close(
                Khat[row0 + 1, column], workspace.transform_scratch[1], gamma,
            ) || return false
            _product_coupled_transform_entry_close(
                Khat[row0 + 2, column], workspace.transform_scratch[2], gamma,
            ) || return false
            _product_coupled_transform_entry_close(
                Khat[row0 + 3, column], workspace.transform_scratch[3], gamma,
            ) || return false
        end

        # Each dy block receives its own right L^-T transform after the row
        # block's left solve.  The 3x3 scratch is the only temporary matrix.
        for target_block in eachindex(workspace.offsets)
            target0 = 3 * (target_block - 1)
            source_col0 = nr + target0
            for j in 1:3
                a1 = K[row0 + 1, source_col0 + j]
                a2 = K[row0 + 2, source_col0 + j]
                a3 = K[row0 + 3, source_col0 + j]
                _product_coupled_lower_solve3!(
                    workspace, row0, a1, a2, a3,
                ) || return false
                workspace.transform_block_scratch[1, j] =
                    workspace.transform_scratch[1]
                workspace.transform_block_scratch[2, j] =
                    workspace.transform_scratch[2]
                workspace.transform_block_scratch[3, j] =
                    workspace.transform_scratch[3]
            end
            for i in 1:3
                a1 = workspace.transform_block_scratch[i, 1]
                a2 = workspace.transform_block_scratch[i, 2]
                a3 = workspace.transform_block_scratch[i, 3]
                _product_coupled_lower_solve3!(
                    workspace, target0, a1, a2, a3,
                ) || return false
                _product_coupled_transform_entry_close(
                    Khat[row0 + i, source_col0 + 1],
                    workspace.transform_scratch[1], gamma,
                ) || return false
                _product_coupled_transform_entry_close(
                    Khat[row0 + i, source_col0 + 2],
                    workspace.transform_scratch[2], gamma,
                ) || return false
                _product_coupled_transform_entry_close(
                    Khat[row0 + i, source_col0 + 3],
                    workspace.transform_scratch[3], gamma,
                ) || return false
            end
        end
    end

    # Symmetric rows receive only the right transforms.  Every other column
    # is copied exactly from K and remains an exact zero-work certificate.
    @inbounds for row in (nsdim + 1):n
        for column in 1:nr
            _product_coupled_transform_entry_close(
                Khat[row, column], K[row, column], gamma,
            ) || return false
        end
        for column in (dtau, dkappa)
            _product_coupled_transform_entry_close(
                Khat[row, column], K[row, column], gamma,
            ) || return false
        end
        for target_block in eachindex(workspace.offsets)
            target0 = 3 * (target_block - 1)
            source_col0 = nr + target0
            a1 = K[row, source_col0 + 1]
            a2 = K[row, source_col0 + 2]
            a3 = K[row, source_col0 + 3]
            _product_coupled_lower_solve3!(
                workspace, target0, a1, a2, a3,
            ) || return false
            _product_coupled_transform_entry_close(
                Khat[row, source_col0 + 1], workspace.transform_scratch[1], gamma,
            ) || return false
            _product_coupled_transform_entry_close(
                Khat[row, source_col0 + 2], workspace.transform_scratch[2], gamma,
            ) || return false
            _product_coupled_transform_entry_close(
                Khat[row, source_col0 + 3], workspace.transform_scratch[3], gamma,
            ) || return false
        end
    end
    return true
end

@inline function _product_coupled_factor_coordinate_rhs_certificate!(
    workspace::NonsymmetricCoupledWorkspace{T},
) where {T}
    workspace.transform_valid && workspace.factor_coordinate_rhs_valid ||
        return false
    _product_coupled_transform_order_ok(workspace) || return false
    gamma = _product_coupled_gamma(T, 32)
    isfinite(gamma) || return false
    @inbounds for row in (workspace.nonsymmetric_dimension + 1):workspace.dimension
        _product_coupled_transform_entry_close(
            workspace.factor_coordinate_rhs[row], workspace.rhs[row], gamma,
        ) || return false
    end
    @inbounds for block_index in eachindex(workspace.offsets)
        local0 = 3 * (block_index - 1)
        row0 = local0 + 1
        a1 = workspace.rhs[row0]
        a2 = workspace.rhs[row0 + 1]
        a3 = workspace.rhs[row0 + 2]
        _product_coupled_lower_solve3!(
            workspace, local0, a1, a2, a3,
        ) || return false
        _product_coupled_transform_entry_close(
            workspace.factor_coordinate_rhs[row0],
            workspace.transform_scratch[1], gamma,
        ) || return false
        _product_coupled_transform_entry_close(
            workspace.factor_coordinate_rhs[row0 + 1],
            workspace.transform_scratch[2], gamma,
        ) || return false
        _product_coupled_transform_entry_close(
            workspace.factor_coordinate_rhs[row0 + 2],
            workspace.transform_scratch[3], gamma,
        ) || return false
    end
    return true
end

"""Recover physical `z=[dx_r;dy_N;dτ;dκ]` from a factor-coordinate solve."""
@inline function _product_coupled_recover_physical!(
    workspace::NonsymmetricCoupledWorkspace{T},
    solution::AbstractVector{T},
) where {T}
    n = workspace.dimension
    length(solution) == n || return false
    workspace.factor_certified || return false
    if workspace.transform_valid
        workspace.transform_epoch == factor_matrix_epoch(workspace.cache) ||
            return false
        _product_coupled_transform_order_ok(workspace) || return false
        _product_coupled_factor_coordinate_finite(workspace) || return false
        @inbounds for i in 1:workspace.nr
            workspace.physical_solution[i] = solution[i]
        end
        @inbounds for block_index in eachindex(workspace.offsets)
            local0 = 3 * (block_index - 1)
            v0 = workspace.nr + local0
            v1 = solution[v0 + 1]
            v2 = solution[v0 + 2]
            v3 = solution[v0 + 3]
            _product_coupled_transpose_solve3!(
                workspace, local0, v1, v2, v3,
            ) || return false
            y1 = workspace.transform_scratch[1]
            y2 = workspace.transform_scratch[2]
            y3 = workspace.transform_scratch[3]
            workspace.physical_solution[v0 + 1] = y1
            workspace.physical_solution[v0 + 2] = y2
            workspace.physical_solution[v0 + 3] = y3

            L = workspace.factor_coordinate_factor
            l11 = L[local0 + 1, local0 + 1]
            l21 = L[local0 + 2, local0 + 1]
            l22 = L[local0 + 2, local0 + 2]
            l31 = L[local0 + 3, local0 + 1]
            l32 = L[local0 + 3, local0 + 2]
            l33 = L[local0 + 3, local0 + 3]
            gamma = _product_coupled_gamma(T, 32)
            lhs1 = l11 * y1 + l21 * y2 + l31 * y3
            lhs2 = l22 * y2 + l32 * y3
            lhs3 = l33 * y3
            work1 = abs(v1) + abs(l11 * y1) + abs(l21 * y2) + abs(l31 * y3)
            work2 = abs(v2) + abs(l22 * y2) + abs(l32 * y3)
            work3 = abs(v3) + abs(l33 * y3)
            r1 = lhs1 - v1
            r2 = lhs2 - v2
            r3 = lhs3 - v3
            a1 = gamma * work1
            a2 = gamma * work2
            a3 = gamma * work3
            isfinite(gamma) && isfinite(r1) && isfinite(r2) &&
                isfinite(r3) && isfinite(a1) && isfinite(a2) &&
                isfinite(a3) || return false
            _product_coupled_zero_safe_close(r1, a1) &&
                _product_coupled_zero_safe_close(r2, a2) &&
                _product_coupled_zero_safe_close(r3, a3) || return false
            workspace.recovery_residual[local0 + 1] = r1
            workspace.recovery_residual[local0 + 2] = r2
            workspace.recovery_residual[local0 + 3] = r3
            workspace.recovery_bound[local0 + 1] = a1
            workspace.recovery_bound[local0 + 2] = a2
            workspace.recovery_bound[local0 + 3] = a3
        end
        workspace.physical_solution[workspace.nr + workspace.nonsymmetric_dimension + 1] =
            solution[workspace.nr + workspace.nonsymmetric_dimension + 1]
        workspace.physical_solution[workspace.nr + workspace.nonsymmetric_dimension + 2] =
            solution[workspace.nr + workspace.nonsymmetric_dimension + 2]
    else
        copyto!(workspace.physical_solution, solution)
    end
    @inbounds for value in workspace.physical_solution
        isfinite(value) || return false
    end
    return true
end

"""Independent componentwise certificate for the retained physical K*z=q."""
@inline function _product_coupled_original_solution_certificate!(
    workspace::NonsymmetricCoupledWorkspace{T},
    physical_solution::AbstractVector{T},
) where {T}
    length(physical_solution) == workspace.dimension || return false
    K = workspace.matrix
    q = workspace.rhs
    n = workspace.dimension
    gamma = _product_coupled_gamma(T, 8n)
    isfinite(gamma) || return false
    @inbounds for row in 1:n
        acc = -q[row]
        work = abs(q[row])
        for column in 1:n
            term = K[row, column] * physical_solution[column]
            acc += term
            work += abs(term)
        end
        allowance = gamma * work
        isfinite(acc) && isfinite(work) && isfinite(allowance) || return false
        _product_coupled_zero_safe_close(acc, allowance) || return false
        workspace.recovery_residual[row] = acc
        workspace.recovery_bound[row] = allowance
    end
    return true
end

@inline function _product_coupled_permutation!(
    workspace::NonsymmetricCoupledWorkspace,
)
    n = workspace.dimension
    permutation = workspace.permutation
    pivots = workspace.cache.ipiv
    @inbounds for i in 1:n
        permutation[i] = i
    end
    @inbounds for i in 1:n
        pivot = pivots[i]
        1 <= pivot <= n || return false
        if pivot != i
            permutation[i], permutation[pivot] =
                permutation[pivot], permutation[i]
        end
    end
    return true
end

"""Certify the stored pivoted factor by `E = P*K - L*U`."""
@inline function _product_coupled_factor_certificate!(
    workspace::NonsymmetricCoupledWorkspace{T},
) where {T}
    n = workspace.dimension
    T(8n) * eps(one(T)) < one(T) || return false
    _product_coupled_permutation!(workspace) || return false
    K = _product_coupled_active_matrix(workspace)
    F = workspace.cache.factors
    permutation = workspace.permutation
    E = workspace.factor_error
    # Standard GEPP supplies a componentwise O(gamma(3n)) backward error;
    # recomputing each n-term L*U dot product adds one further n-operation
    # budget.  gamma(4n) therefore certifies the stored Ehat without a norm
    # or output-relative shortcut.  The stronger 8n*eps<1 precondition is
    # shared with the subsequent staged-solve certificate.
    gamma_factor = _product_coupled_gamma(T, 4n)
    isfinite(gamma_factor) || return false
    @inbounds for j in 1:n
        for i in 1:n
            product = zero(T)
            product_work = zero(T)
            stop = min(i, j)
            for k in 1:stop
                lik = i == k ? one(T) : F[i, k]
                term = lik * F[k, j]
                product += term
                product_work += abs(term)
            end
            pk = K[permutation[i], j]
            error = pk - product
            work = abs(pk) + product_work
            E[i, j] = error
            allowance = gamma_factor * work
            _product_coupled_zero_safe_close(error, allowance) || return false
        end
    end
    return true
end

@inline function _product_coupled_factorize!(
    workspace::NonsymmetricCoupledWorkspace{T}, epoch::Int,
) where {T}
    workspace.factor_certified = false
    workspace.last_reason = COUPLED_READY
    # A factor attempt (successful or not) replaces the numeric factor.  Revoke
    # the previous receipt and count the attempt before any early exit so a
    # failed or uncertified attempt can never masquerade as reusable.
    workspace.factor_receipt = nothing
    workspace.factor_attempt_count += 1
    if workspace.transform_valid
        workspace.transform_epoch == epoch || begin
            workspace.last_reason = COUPLED_TRANSFORM_EPOCH_MISMATCH
            return false
        end
        _product_coupled_transform_order_ok(workspace) || begin
            workspace.last_reason = COUPLED_TRANSFORM_ORDER_FAILED
            return false
        end
        _product_coupled_factor_coordinate_finite(workspace) || begin
            workspace.last_reason = COUPLED_TRANSFORM_FAILED
            return false
        end
    end
    @inbounds for value in workspace.matrix
        if !isfinite(value)
            workspace.last_reason = COUPLED_ASSEMBLY_NONFINITE
            return false
        end
    end
    active_matrix = _product_coupled_active_matrix(workspace)
    @inbounds for value in active_matrix
        if !isfinite(value)
            workspace.last_reason = COUPLED_ASSEMBLY_NONFINITE
            return false
        end
    end
    try
        factorize!(workspace.cache, active_matrix, epoch)
    catch
        workspace.last_reason = COUPLED_FACTOR_FAILED
        return false
    end
    if factor_matrix_epoch(workspace.cache) != epoch
        workspace.last_reason = COUPLED_EPOCH_MISMATCH
        return false
    end
    if !_product_coupled_factor_certificate!(workspace)
        workspace.last_reason = COUPLED_FACTOR_CERT_FAILED
        return false
    end
    workspace.factor_certified = true
    active_matrix = _product_coupled_active_matrix(workspace)
    proof_scale = max(maximum(abs, active_matrix), one(T))
    proof_bound = maximum(abs, workspace.factor_error) / proof_scale
    workspace.factor_receipt = FactorReceipt(
        factor_matrix_epoch(workspace.cache),
        factor_epoch(workspace.cache),
        dense_factor_pattern_signature(
            workspace.dimension, workspace.dimension, :coupled,
        ),
        :coupled,
        :standard_pivoted_lu,
        T,
        factor_receipt_precision(T),
        zero(T),
        :none,
        :factored,
        proof_bound,
        true,
        0, 0,
    )
    workspace.receipt_build_count += 1
    return true
end

"""Allocation-free pivoted forward/back solve exposing the staged `y`.

The cache stores `P*K=L*U`.  This routine forms `P*q`, solves `L*y=P*q`,
then `U*z=y`, and retains both triangular residuals for the independent
factor-aware end certificate below.
"""
@inline function _product_coupled_staged_solve!(
    workspace::NonsymmetricCoupledWorkspace{T},
    destination::AbstractVector{T},
    rhs::AbstractVector{T},
) where {T}
    n = workspace.dimension
    length(destination) == n && length(rhs) == n || return false
    F = workspace.cache.factors
    permutation = workspace.permutation
    permuted_rhs = workspace.permuted_rhs
    staged_y = workspace.staged_y
    forward_residual = workspace.forward_residual
    backward_residual = workspace.backward_residual
    gamma_solve = _product_coupled_gamma(T, 8n)
    isfinite(gamma_solve) || return false

    @inbounds for i in 1:n
        value = rhs[permutation[i]]
        isfinite(value) || return false
        permuted_rhs[i] = value
    end
    @inbounds for i in 1:n
        value = permuted_rhs[i]
        for j in 1:(i - 1)
            value -= F[i, j] * staged_y[j]
        end
        isfinite(value) || return false
        staged_y[i] = value
    end
    @inbounds for i in n:-1:1
        value = staged_y[i]
        for j in (i + 1):n
            value -= F[i, j] * destination[j]
        end
        pivot = F[i, i]
        isfinite(pivot) && !iszero(pivot) || return false
        value /= pivot
        isfinite(value) || return false
        destination[i] = value
    end

    # Recompute both triangular equations.  These are actual f and u in
    # P(Kz-q)=E*z+L*u+f, not inferred norm bounds for a black-box solve.
    @inbounds for i in 1:n
        ly = staged_y[i]
        ly_work = abs(staged_y[i]) + abs(permuted_rhs[i])
        for j in 1:(i - 1)
            term = F[i, j] * staged_y[j]
            ly += term
            ly_work += abs(term)
        end
        f = ly - permuted_rhs[i]
        forward_residual[i] = f
        _product_coupled_zero_safe_close(f, gamma_solve * ly_work) ||
            return false

        uz = zero(T)
        uz_work = abs(staged_y[i])
        for j in i:n
            term = F[i, j] * destination[j]
            uz += term
            uz_work += abs(term)
        end
        u = uz - staged_y[i]
        backward_residual[i] = u
        _product_coupled_zero_safe_close(u, gamma_solve * uz_work) ||
            return false
    end
    return true
end

"""Rebuild staged `y`, `f`, and `u` for an arbitrary candidate and RHS.

Refinement changes `z` after a correction solve, so its staged residuals may
never be reused from that correction RHS.  This routine restores the exact
candidate/RHS association without modifying `z`.
"""
@inline function _product_coupled_recompute_staged_residuals!(
    workspace::NonsymmetricCoupledWorkspace{T},
    solution::AbstractVector{T},
    rhs::AbstractVector{T},
) where {T}
    n = workspace.dimension
    length(solution) == n && length(rhs) == n || return false
    F = workspace.cache.factors
    permutation = workspace.permutation
    permuted_rhs = workspace.permuted_rhs
    staged_y = workspace.staged_y
    f = workspace.forward_residual
    u = workspace.backward_residual
    @inbounds for i in 1:n
        value = rhs[permutation[i]]
        isfinite(value) || return false
        permuted_rhs[i] = value
    end
    @inbounds for i in 1:n
        value = permuted_rhs[i]
        for j in 1:(i - 1)
            value -= F[i, j] * staged_y[j]
        end
        isfinite(value) || return false
        staged_y[i] = value
    end
    @inbounds for i in 1:n
        ly = staged_y[i]
        ly_work = abs(staged_y[i]) + abs(permuted_rhs[i])
        for j in 1:(i - 1)
            term = F[i, j] * staged_y[j]
            ly += term
            ly_work += abs(term)
        end
        fi = ly - permuted_rhs[i]
        f[i] = fi
        isfinite(fi) && isfinite(ly_work) || return false

        uz = zero(T)
        uz_work = abs(staged_y[i])
        for j in i:n
            term = F[i, j] * solution[j]
            uz += term
            uz_work += abs(term)
        end
        ui = uz - staged_y[i]
        u[i] = ui
        isfinite(ui) && isfinite(uz_work) || return false
    end
    return true
end

"""Factor-aware end certificate for one solution of the original `K*z=q`.

With `E=P*K-L*U`, the computed residual is bounded using
`P(Kz-q)=Ez+Lu+f`.  The recomputed `gamma(8n)` term covers the triangular
solves and the two matrix-vector reassociations; it is deliberately not an
output-relative residual test on a strongly conditioned coupled matrix.
"""
@inline function _product_coupled_solution_certificate!(
    workspace::NonsymmetricCoupledWorkspace{T},
    solution::AbstractVector{T},
    rhs::AbstractVector{T},
) where {T}
    n = workspace.dimension
    workspace.factor_certified || return false, T(Inf)
    length(solution) == n && length(rhs) == n || return false, T(Inf)
    factor_matrix_epoch(workspace.cache) > 0 || return false, T(Inf)
    if workspace.transform_valid
        workspace.transform_epoch == factor_matrix_epoch(workspace.cache) ||
            return false, T(Inf)
        workspace.factor_coordinate_rhs_valid || return false, T(Inf)
        _product_coupled_factor_coordinate_matrix_certificate!(workspace) ||
            return false, T(Inf)
        _product_coupled_factor_coordinate_rhs_certificate!(workspace) ||
            return false, T(Inf)
    end
    K = _product_coupled_active_matrix(workspace)
    active_rhs = _product_coupled_active_rhs(workspace, rhs)
    length(active_rhs) == n || return false, T(Inf)
    F = workspace.cache.factors
    E = workspace.factor_error
    permutation = workspace.permutation
    f = workspace.forward_residual
    u = workspace.backward_residual
    identity_rhs = workspace.identity_rhs
    upper_work = workspace.upper_work
    lower_work = workspace.lower_work
    residual = workspace.residual
    bound = workspace.bound
    gamma_solve = _product_coupled_gamma(T, 8n)
    isfinite(gamma_solve) || return false, T(Inf)
    _product_coupled_recompute_staged_residuals!(
        workspace, solution, active_rhs,
    ) || return false, T(Inf)

    # Arithmetic work of |L|(|U||z|), needed even when a recomputed u rounds
    # to exact zero after cancellation.
    @inbounds for i in 1:n
        value = zero(T)
        for j in i:n
            value += abs(F[i, j]) * abs(solution[j])
        end
        upper_work[i] = value
    end
    @inbounds for i in 1:n
        value = upper_work[i]
        for j in 1:(i - 1)
            value += abs(F[i, j]) * upper_work[j]
        end
        lower_work[i] = value
    end

    # Explicitly form E*z and L*u+f, together with their arithmetic work.
    worst = zero(T)
    @inbounds for i in 1:n
        ez = zero(T)
        ez_work = zero(T)
        for j in 1:n
            term = E[i, j] * solution[j]
            ez += term
            ez_work += abs(term)
        end
        lu = f[i] + u[i]
        lu_work = abs(f[i]) + abs(u[i])
        for j in 1:(i - 1)
            term = F[i, j] * u[j]
            lu += term
            lu_work += abs(term)
        end
        identity = ez + lu
        identity_work = ez_work + lu_work
        isfinite(identity) && isfinite(identity_work) ||
            return false, T(Inf)
        identity_rhs[i] = identity
        original_row = permutation[i]
        acc = -active_rhs[original_row]
        pk_work = abs(active_rhs[original_row])
        for j in 1:n
            term = K[original_row, j] * solution[j]
            acc += term
            pk_work += abs(term)
        end
        identity_error = acc - identity
        reassociation_allowance = gamma_solve * (
            pk_work + identity_work + lower_work[i]
        )
        _product_coupled_zero_safe_close(
            identity_error, reassociation_allowance,
        ) || return false, T(Inf)
        allowance = identity_work +
                    gamma_solve * (pk_work + lower_work[i])
        residual[original_row] = acc
        bound[original_row] = allowance
        _product_coupled_zero_safe_close(acc, allowance) ||
            return false, T(Inf)
        if !iszero(allowance)
            ratio = abs(acc) / allowance
            isfinite(ratio) || return false, T(Inf)
            ratio > worst && (worst = ratio)
        end
    end
    return true, worst
end

@inline function _product_coupled_factor_receipt_current(
    workspace::NonsymmetricCoupledWorkspace{T},
) where {T}
    workspace.factor_certified || return false
    return factor_receipt_owned(
        workspace.factor_receipt;
        matrix_epoch=factor_matrix_epoch(workspace.cache),
        factor_epoch=factor_epoch(workspace.cache),
        pattern_signature=dense_factor_pattern_signature(
            workspace.dimension, workspace.dimension, :coupled,
        ),
        route=:coupled,
        provider=:standard_pivoted_lu,
        regularization=zero(T),
        require_proof=true,
    )
end

@inline function _product_coupled_solve!(
    workspace::NonsymmetricCoupledWorkspace{T},
    destination::Vector{T},
    rhs::Vector{T},
) where {T}
    workspace.factor_certified || begin
        workspace.last_reason = COUPLED_EPOCH_MISMATCH
        return false, T(Inf)
    end
    # Every coupled solve validates route-local factor ownership first: a
    # stale or revoked receipt rejects the solve even when a factor object is
    # still physically present.
    _product_coupled_factor_receipt_current(workspace) || begin
        workspace.last_reason = COUPLED_EPOCH_MISMATCH
        return false, T(Inf)
    end
    active_rhs = _product_coupled_active_rhs(workspace, rhs)
    if workspace.transform_valid && rhs === workspace.rhs &&
       !workspace.factor_coordinate_rhs_valid
        workspace.last_reason = COUPLED_TRANSFORM_FAILED
        return false, T(Inf)
    end
    if !_product_coupled_staged_solve!(workspace, destination, active_rhs)
        # The solve itself must consume q̂ whenever the caller supplied the
        # retained physical q.  Retry is intentionally not permitted: a
        # mismatched coordinate RHS is a typed fail-closed condition.
        if workspace.transform_valid && rhs === workspace.rhs
            workspace.last_reason = COUPLED_TRANSFORM_FAILED
            return false, T(Inf)
        end
        workspace.last_reason = COUPLED_SOLVE_CERT_FAILED
        return false, T(Inf)
    end
    workspace.solves += 1
    ok, merit = _product_coupled_solution_certificate!(
        workspace, destination, active_rhs,
    )
    if !ok
        workspace.last_reason = COUPLED_SOLVE_CERT_FAILED
        return false, T(Inf)
    end
    workspace.last_reason = COUPLED_READY
    return true, merit
end
