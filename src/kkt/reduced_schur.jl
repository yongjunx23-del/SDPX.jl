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
honestly false.  SDPX still freezes one deterministic `BlockIncidencePlan` per
session at first assembly and reuses its frozen Schur CSC `colptr`, `rowval`,
`nzval` slots, RHS, residual and block workspaces in every numeric epoch.
`structural_assembly_count` must remain one while `numeric_factor_count`
increments once for each *successfully certified* sparse predictor/corrector
epoch.  `factor_attempt_count` counts every numeric attempt (including failed
or condition-rejected ones); a failed attempt never increments
`numeric_factor_count` and never builds a receipt.

PSD congruence panels are owned by the session: `psd_panels`/`psd_panel_blocks`
hold one typed panel per verified PSD block (prepacked once at the structural
epoch, keyed by the frozen plan signature), `psd_panel_epoch` is the current
scaling epoch, and `psd_panel_slot` maps each plan block index to its panel
slot (zero = generic assembly).  Every numeric assembly that changes the block
operators advances `psd_panel_epoch`; panels rebuild their congruence only
when their stored epoch differs.
"""
mutable struct SparseSchurSession{T<:AbstractFloat}
    n::Int
    m::Int
    dimension::Int
    schur::SparseMatrixCSC{T,Int}
    plan::Union{Nothing,BlockIncidencePlan}
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
    structural_assembly_count::Int
    numeric_assembly_count::Int
    rhs_assembly_count::Int
    numeric_factor_count::Int
    factor_attempt_count::Int
    pattern_reuse_count::Int
    psd_panels::Vector{PSDCongruencePanel{T}}
    psd_panel_scratch::Vector{PSDPanelEpochScratch{T}}
    psd_panel_blocks::Vector{Int}
    psd_panel_slot::Vector{Int}
    psd_panel_backend::Union{Nothing,AbstractLABackend}
    psd_panel_epoch::Int
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
        nothing, zero(UInt64), false, nothing, nothing, 0, zero(UInt64), nothing, 0,
        zeros(T, dimension), zeros(T, dimension), zeros(T, dimension),
        zeros(T, dimension),
        Matrix{T}[], zeros(T, 0, 0),
        T[], T[], T[], T[], 0,
        zeros(T, n), zeros(T, n),
        0, 0, 0, 0, 0, 0,
        PSDCongruencePanel{T}[], PSDPanelEpochScratch{T}[], Int[], Int[],
        StandardLABackend(_la_arithmetic_symbol(T)), 0,
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

"""Freeze the immutable block incidence plan and its CSC slots once per session."""
function _setup_block_incidence_plan!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    session.structural_assembly_count == 0 || return true
    validate_cone_linearization(system.cone)
    plan = build_block_incidence_plan(system)
    session.plan = plan
    session.schur = SparseMatrixCSC{T,Int}(
        plan.dimension, plan.dimension, plan.schur_colptr,
        plan.schur_rowval, zeros(T, length(plan.schur_rowval)),
    )
    maximum_block_dimension = isempty(plan.block_ranges) ?
        0 : maximum(length, plan.block_ranges)
    session.maximum_block_dimension = maximum_block_dimension
    session.block_inverses = [
        zeros(T, length(rows), length(rows)) for rows in plan.block_ranges
    ]
    session.block_augmented = zeros(
        T, maximum_block_dimension, 2 * maximum_block_dimension,
    )
    session.block_rhs_delta = zeros(T, maximum_block_dimension)
    session.block_hinv_b = zeros(T, maximum_block_dimension)
    session.block_hinv_delta = zeros(T, maximum_block_dimension)
    session.block_column_work = zeros(T, maximum_block_dimension)
    session.pattern_signature = plan.signature
    _setup_psd_panels!(session, system, plan)
    session.structural_assembly_count = 1
    return true
end

"""
    _setup_psd_panels!(session, system, plan)

Structural-epoch PSD panel prepack (once per PSD block, keyed by the frozen
plan): for every plan block whose descriptor is structurally eligible
(`psd_k >= 3`), verify numerically that the block operator is a PSD
congruence operator and prepack the panel from the frozen A-block CSC
positions.  Blocks that fail verification keep the generic assembly
(unchanged behavior); a verified PSD block is removed from the generic
pair loop and assembled by the congruence panels in every later numeric
epoch.  This is the only prepack site, so `psd_panel_epoch` still starts at
zero and the first numeric assembly performs the first congruence rebuild.
"""
function _setup_psd_panels!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
    plan::BlockIncidencePlan,
) where {T<:AbstractFloat}
    panels = PSDCongruencePanel{T}[]
    scratch = PSDPanelEpochScratch{T}[]
    panel_blocks = Int[]
    slot = zeros(Int, length(plan.descriptors))
    for block_index in plan.psd_blocks
        descriptor = plan.descriptors[block_index]
        k = descriptor.psd_k
        k >= 3 || continue
        block_scratch = psd_panel_epoch_scratch(T, k)
        block_H = product_cone_block_operator(system.cone, block_index)
        psd_operator_congruence_verified!(block_scratch, block_H, k) ||
            continue
        panel = PSDCongruencePanel{T}(
            block_index, k, copy(descriptor.active_columns),
        )
        prepack_psd_panel!(
            panel, system.A, descriptor.rows,
            descriptor.colptr, descriptor.local_rows, k,
        )
        slot[block_index] = length(panels) + 1
        push!(panels, panel)
        push!(scratch, block_scratch)
        push!(panel_blocks, block_index)
    end
    session.psd_panels = panels
    session.psd_panel_scratch = scratch
    session.psd_panel_blocks = panel_blocks
    session.psd_panel_slot = slot
    session.psd_panel_epoch = 0
    return true
end

"""Cumulative PSD panel accounting of one sparse session (prepacks, rebuilds,
prepack/rebuild bytes, and Schur tile writes), for trace projection."""
function sparse_schur_session_psd_panel_diagnostics(
    session::SparseSchurSession,
)
    stats = psd_panel_stats(session.psd_panels)
    return (
        psd_panel_blocks=length(session.psd_panels),
        psd_panel_prepacks=stats.prepacks,
        psd_panel_rebuilds=stats.rebuilds,
        psd_panel_prepack_bytes=stats.prepack_bytes,
        psd_panel_rebuild_bytes=stats.rebuild_bytes,
        psd_panel_tile_writes=stats.tiles,
    )
end

"""Assemble numeric reduced operator values into the frozen plan slots."""
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
    _setup_block_incidence_plan!(session, system)
    plan = session.plan
    plan === nothing && return _invalidate_sparse_schur_factor!(
        session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_unavailable,
    )
    _block_incidence_source_signature(system) == plan.signature ||
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
    schur_nzval = session.schur.nzval

    for (block_index, descriptor) in enumerate(plan.descriptors)
        # PSD congruence panels own their block's tile, border, and recovery
        # work; the generic dense pair loop below never touches those blocks.
        session.psd_panel_slot[block_index] > 0 && continue
        block = descriptor.rows
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
        active = descriptor.active_columns
        colptr = descriptor.colptr
        local_rows = descriptor.local_rows
        tile_slots = descriptor.tile_slots
        nj = length(active)
        for j_pos in 1:nj
            column_j = active[j_pos]
            # H_b^{-1} A[R_b, j] from the frozen A-block CSC positions.
            @inbounds for local_row in 1:dimension
                value = zero(T)
                for pointer in colptr[j_pos]:(colptr[j_pos + 1] - 1)
                    local_column = local_rows[pointer]
                    value += block_inverse[local_row, local_column] *
                             system.A[block[local_column], column_j]
                end
                session.block_column_work[local_row] = value
            end
            for i_pos in 1:nj
                column_i = active[i_pos]
                value = zero(T)
                @inbounds for pointer in colptr[i_pos]:(colptr[i_pos + 1] - 1)
                    local_row = local_rows[pointer]
                    value += system.A[block[local_row], column_i] *
                             session.block_column_work[local_row]
                end
                schur_nzval[tile_slots[(i_pos - 1) * nj + j_pos]] += value
            end
            value_b = zero(T)
            @inbounds for pointer in colptr[j_pos]:(colptr[j_pos + 1] - 1)
                local_row = local_rows[pointer]
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

    # PSD congruence panels: authoritative Schur tiles written directly into
    # the frozen CSC slots plus the border contributions, without any block
    # inverse.  Each numeric assembly is one scaling epoch; a panel rebuilds
    # its congruence only when its stored epoch differs from the session's.
    if !isempty(session.psd_panels)
        backend = session.psd_panel_backend
        session.psd_panel_epoch += 1
        epoch = session.psd_panel_epoch
        @inbounds for (slot, block_index) in enumerate(session.psd_panel_blocks)
            descriptor = plan.descriptors[block_index]
            panel = session.psd_panels[slot]
            scratch = session.psd_panel_scratch[slot]
            block = descriptor.rows
            k = panel.k
            block_H = product_cone_block_operator(system.cone, block_index)
            if psd_panel_scaling_epoch(panel) != epoch
                try
                    P, Pinv = psd_operator_congruence_factors!(
                        scratch, block_H, k,
                    )
                    update_psd_panel_congruence!(backend, panel, P, Pinv)
                catch exception
                    exception isa InterruptException && rethrow()
                    exception isa DomainError || rethrow()
                    return _invalidate_sparse_schur_factor!(
                        session, SPARSE_SCHUR_FACTOR_FAILED,
                        :cone_block_singular,
                    )
                end
            end
            psd_panel_schur_tile_slots!(
                backend, schur_nzval, panel, descriptor.tile_slots,
            )
            # Border: svec(P⁻¹·b·P⁻¹) through the congruence inverse action,
            # then the frozen-CSC svec dots (the same values the generic path
            # obtains from block_hinv_b).
            b_block = @view system.b[block]
            psd_panel_apply_inverse!(
                scratch, backend, scratch.svec_a, b_block, k,
            )
            active = descriptor.active_columns
            colptr = descriptor.colptr
            local_rows = descriptor.local_rows
            for j_pos in eachindex(active)
                column_j = active[j_pos]
                value_b = zero(T)
                @inbounds for pointer in
                    colptr[j_pos]:(colptr[j_pos + 1] - 1)
                    local_row = local_rows[pointer]
                    value_b += system.A[block[local_row], column_j] *
                               scratch.svec_a[local_row]
                end
                atb[column_j] += value_b
            end
            @inbounds for local_row in 1:length(block)
                btHb += system.b[block[local_row]] * scratch.svec_a[local_row]
            end
        end
    end
    @inbounds for column in 1:n
        schur_nzval[plan.border_column_slots[column]] =
            system.c[column] - atb[column]
        schur_nzval[plan.border_row_slots[column]] =
            system.c[column] + atb[column]
    end
    schur_nzval[plan.border_diagonal_slot] =
        system.kappa / system.tau - btHb
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

"""
Explicit reference numeric assembly for later E2E parity.

Recomputes every block contribution from the original dense scans of `A`
(reading every block row, including structural zeros) and writes into the same
frozen plan slots.  This is intentionally NOT a runtime fallback: the
production `assemble_sparse_schur_operator!` always uses the frozen A-block
CSC positions, and callers may invoke this reference explicitly to compare
operators elementwise in parity tests.
"""
function assemble_sparse_schur_operator_reference!(
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
    _setup_block_incidence_plan!(session, system)
    plan = session.plan
    plan === nothing && return _invalidate_sparse_schur_factor!(
        session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_unavailable,
    )
    _block_incidence_source_signature(system) == plan.signature ||
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_drift,
        )

    _invalidate_sparse_schur_factor!(
        session, SPARSE_SCHUR_READY, :numeric_operator_changed,
    )
    session.numeric_assembly_count > 0 && (session.pattern_reuse_count += 1)
    fill!(session.schur.nzval, zero(T))
    fill!(session.atb, zero(T))
    atb = session.atb
    btHb = zero(T)
    schur_nzval = session.schur.nzval

    for (block_index, block) in enumerate(plan.block_ranges)
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
        descriptor = plan.descriptors[block_index]
        active = descriptor.active_columns
        tile_slots = descriptor.tile_slots
        nj = length(active)
        for (j_pos, column_j) in enumerate(active)
            @inbounds for local_row in 1:dimension
                value = zero(T)
                for local_column in 1:dimension
                    value += block_inverse[local_row, local_column] *
                             system.A[block[local_column], column_j]
                end
                session.block_column_work[local_row] = value
            end
            for (i_pos, column_i) in enumerate(active)
                value = zero(T)
                @inbounds for local_row in 1:dimension
                    value += system.A[block[local_row], column_i] *
                             session.block_column_work[local_row]
                end
                schur_nzval[tile_slots[(i_pos - 1) * nj + j_pos]] += value
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
        schur_nzval[plan.border_column_slots[column]] =
            system.c[column] - atb[column]
        schur_nzval[plan.border_row_slots[column]] =
            system.c[column] + atb[column]
    end
    schur_nzval[plan.border_diagonal_slot] =
        system.kappa / system.tau - btHb
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

"""Assemble only the reduced RHS, reusing frozen plan positions and inverses."""
function assemble_sparse_schur_rhs!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    session.numeric_assembly_count > 0 || throw(ArgumentError(
        "sparse Schur operator must be assembled before its RHS",
    ))
    validate_cone_linearization(system.cone)
    plan = session.plan
    plan === nothing && return _invalidate_sparse_schur_factor!(
        session, SPARSE_SCHUR_SOLVE_FAILED, :sparse_pattern_unavailable,
    )
    _block_incidence_source_signature(system) == plan.signature ||
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_drift,
        )
    length(plan.block_ranges) == length(session.block_inverses) ||
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_FACTOR_FAILED, :sparse_pattern_drift,
        )

    n = session.n
    fill!(session.at_delta, zero(T))
    at_delta = session.at_delta
    bt_delta = zero(T)
    for (block_index, descriptor) in enumerate(plan.descriptors)
        block = descriptor.rows
        dimension = length(block)
        @inbounds for local_row in 1:dimension
            delta = system.rhs.cone_corrector[block[local_row]] -
                    system.rhs.primal_affine[block[local_row]]
            session.block_rhs_delta[local_row] = delta
        end
        panel_slot = session.psd_panel_slot[block_index]
        if panel_slot > 0
            # PSD congruence panel: H_b⁻¹·delta = svec(P⁻¹·svec⁻¹(δ)·P⁻¹)
            # through the panel's congruence factors (no block inverse).
            scratch = session.psd_panel_scratch[panel_slot]
            panel = session.psd_panels[panel_slot]
            psd_panel_apply_inverse!(
                scratch, session.psd_panel_backend,
                session.block_hinv_delta, session.block_rhs_delta, panel.k,
            )
        else
            block_inverse = session.block_inverses[block_index]
            @inbounds for local_row in 1:dimension
                value = zero(T)
                for local_column in 1:dimension
                    value += block_inverse[local_row, local_column] *
                             session.block_rhs_delta[local_column]
                end
                session.block_hinv_delta[local_row] = value
            end
        end
        active = descriptor.active_columns
        colptr = descriptor.colptr
        local_rows = descriptor.local_rows
        for (j_pos, column) in enumerate(active)
            value = zero(T)
            @inbounds for pointer in colptr[j_pos]:(colptr[j_pos + 1] - 1)
                local_row = local_rows[pointer]
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
    session.factor_attempt_count += 1
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
        # Only a factor that passes every capability/condition gate counts as
        # a numeric factor; failed attempts are visible via
        # `factor_attempt_count` only.
        session.numeric_factor_count += 1
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
            0, 0,
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

"""Shared `q = A·dx + h + rP - b·dτ` recovery source vector."""
function _reduced_direction_rhs!(
    rhs_for_dy::AbstractVector{T}, system::NewtonSystem{T},
    dx::AbstractVector{T}, dtau::T,
) where {T<:AbstractFloat}
    m, n = size(system.A)
    @inbounds for i in 1:m
        rhs_for_dy[i] = system.rhs.cone_corrector[i] -
                        system.rhs.primal_affine[i]
        for j in 1:n
            rhs_for_dy[i] += system.A[i, j] * dx[j]
        end
        rhs_for_dy[i] -= system.b[i] * dtau
    end
    return rhs_for_dy
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
    _reduced_direction_rhs!(rhs_for_dy, system, dx, dtau)
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
    plan = session.plan
    plan === nothing && throw(ArgumentError(
        "reduced recovery requires a frozen block incidence plan",
    ))
    _block_incidence_source_signature(system) == plan.signature ||
        throw(ArgumentError("reduced recovery source pattern drift"))
    m, n = size(system.A)
    length(condensed) == n + 1 || throw(DimensionMismatch(
        "reduced solution dimension mismatch",
    ))
    ranges = product_cone_block_ranges(system.cone)
    length(session.block_inverses) == length(ranges) || throw(DimensionMismatch(
        "reduced recovery block inverse count mismatch",
    ))
    dx = copy(@view condensed[1:n])
    dtau = condensed[n + 1]
    rhs_for_dy = zeros(T, m)
    _reduced_direction_rhs!(rhs_for_dy, system, dx, dtau)
    dy = zeros(T, m)
    for (block_index, block) in enumerate(ranges)
        panel_slot = session.psd_panel_slot[block_index]
        if panel_slot > 0
            # PSD congruence panel: dy_block = H_b⁻¹·q_block through the
            # panel's congruence factors (no block inverse).
            scratch = session.psd_panel_scratch[panel_slot]
            panel = session.psd_panels[panel_slot]
            psd_panel_apply_inverse!(
                scratch, session.psd_panel_backend,
                @view(dy[block]), @view(rhs_for_dy[block]), panel.k,
            )
        else
            block_inverse = session.block_inverses[block_index]
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
