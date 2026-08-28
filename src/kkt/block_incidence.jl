# Immutable block incidence plan (GPTPro P3).
#
# Setup freezes, for every product-cone block, the structural layout that all
# later numeric epochs must reuse:
#
#   * cone block row ranges R_b,
#   * active variable columns J_b = { j : A[R_b, j] structurally nonzero },
#   * A block CSC positions (per active column, the local block rows that hold
#     structural nonzeros),
#   * sparse Schur output slots and tile ownership (the J_b × J_b output tile
#     mapped into the frozen Schur CSC, plus the border slots),
#   * a deterministic pattern signature over all of the above.
#
# The plan is immutable: once frozen, numeric epochs only read these slots.
# A structural change (pattern drift) is detected by signature mismatch and
# must fail closed; a numerically-zero value never changes the frozen
# structure, so it can never trigger a symbolic rebuild.
#
# The signs of the reduced operator and the block-local cone operators remain
# owned by `src/kkt/system.jl` and the route assembly; this file stores no
# scalar signs.

"""One frozen cone-block structural descriptor."""
struct BlockIncidenceDescriptor
    rows::UnitRange{Int}
    active_columns::Vector{Int}
    # A-block CSC over `active_columns`: `colptr` has length
    # length(active_columns)+1 and `local_rows[colptr[p]:colptr[p+1]-1]` are
    # the ascending local block rows holding structural nonzeros in column
    # `active_columns[p]`.
    colptr::Vector{Int}
    local_rows::Vector{Int}
    # Sparse Schur output tile slots: linear index `(i_pos - 1) * nj + j_pos`
    # maps the tile entry (active_columns[i_pos], active_columns[j_pos]) to
    # its frozen slot in the Schur CSC `nzval`.
    tile_slots::Vector{Int}
end

"""
    BlockIncidencePlan

Immutable structural plan for one sparse reduced-Schur setup.  Fields:

  * `m`, `n`, `dimension` — A dimensions and reduced dimension `n + 1`;
  * `block_ranges` — ordered, contiguous, non-overlapping cone row ranges;
  * `descriptors` — one `BlockIncidenceDescriptor` per cone block;
  * `schur_colptr`, `schur_rowval` — the frozen Schur CSC structure;
  * `border_column_slots` — frozen slots of the border entries `(j, n+1)`;
  * `border_row_slots` — frozen slots of the border entries `(n+1, j)`;
  * `border_diagonal_slot` — frozen slot of `(n+1, n+1)`;
  * `signature` — deterministic structural pattern signature.
"""
struct BlockIncidencePlan
    m::Int
    n::Int
    dimension::Int
    block_ranges::Vector{UnitRange{Int}}
    descriptors::Vector{BlockIncidenceDescriptor}
    schur_colptr::Vector{Int}
    schur_rowval::Vector{Int}
    border_column_slots::Vector{Int}
    border_row_slots::Vector{Int}
    border_diagonal_slot::Int
    signature::UInt64
end

@inline function _sparse_pattern_mix(signature::UInt64, value::Integer)
    signature ⊻= UInt64(value)
    return signature * UInt64(0x100000001b3)
end

"""Structural A-block CSC for one cone row range (sparse storage = entries)."""
function _block_a_structural_entries(
    A::SparseMatrixCSC, rows::UnitRange{Int},
)
    active = Int[]
    colptr = Int[1]
    local_rows = Int[]
    first_row = first(rows)
    @inbounds for column in axes(A, 2)
        first_entry = length(local_rows) + 1
        for pointer in nzrange(A, column)
            row = A.rowval[pointer]
            row in rows || continue
            push!(local_rows, row - first_row + 1)
        end
        if length(local_rows) >= first_entry
            push!(active, column)
            push!(colptr, length(local_rows) + 1)
        end
    end
    return active, colptr, local_rows
end

"""Dense A has no sparsity structure: every entry is a potential nonzero."""
function _block_a_structural_entries(
    A::AbstractMatrix, rows::UnitRange{Int},
)
    dimension = length(rows)
    ncols = size(A, 2)
    active = collect(1:ncols)
    colptr = collect(1:dimension:(dimension * ncols + 1))
    local_rows = repeat(collect(1:dimension), ncols)
    return active, colptr, local_rows
end

@inline function _mix_block_structure!(
    signature::UInt64, rows::UnitRange{Int},
    active::AbstractVector{Int}, colptr::AbstractVector{Int},
    local_rows::AbstractVector{Int},
)
    signature = _sparse_pattern_mix(signature, first(rows))
    signature = _sparse_pattern_mix(signature, last(rows))
    for (position, column) in enumerate(active)
        signature = _sparse_pattern_mix(signature, column)
        for pointer in colptr[position]:(colptr[position + 1] - 1)
            signature = _sparse_pattern_mix(signature, local_rows[pointer])
        end
    end
    return signature
end

"""
Deterministic structural signature of A's nonzeros and cone block ranges.
Stored structure only: numeric values never enter the signature, so a numeric
zero at a frozen position can never read as pattern drift.
"""
function _block_incidence_source_signature(system::NewtonSystem)
    signature = UInt64(0xcbf29ce484222325)
    m, n = size(system.A)
    signature = _sparse_pattern_mix(signature, m)
    signature = _sparse_pattern_mix(signature, n)
    for block in product_cone_block_ranges(system.cone)
        active, colptr, local_rows = _block_a_structural_entries(
            system.A, block,
        )
        signature = _mix_block_structure!(
            signature, block, active, colptr, local_rows,
        )
    end
    return signature
end

"""Column index of a CSC slot (colptr[c] is the first slot of column c)."""
function _schur_slot_column(colptr::AbstractVector{Int}, slot::Int)
    return searchsortedlast(colptr, slot)
end

"""
    build_block_incidence_plan(system::NewtonSystem) -> BlockIncidencePlan

Freeze the structural block incidence of `system`: cone block row ranges,
per-block active variable columns and A-block CSC positions, the sparse Schur
output slots each block tile owns (plus border slots), and the pattern
signature.  Gap/overlap of cone block ranges and tile-ownership consistency
are validated before the plan is returned.
"""
function build_block_incidence_plan(system::NewtonSystem)
    m, n = size(system.A)
    cone_dimension(system.cone) == m || throw(DimensionMismatch(
        "cone linearization dimension does not match rows of A",
    ))
    validate_cone_linearization(system.cone)
    ranges = product_cone_block_ranges(system.cone)
    dimension = n + 1
    signature = UInt64(0xcbf29ce484222325)
    signature = _sparse_pattern_mix(signature, m)
    signature = _sparse_pattern_mix(signature, n)
    pattern_i = Int[]
    pattern_j = Int[]
    descriptors = Vector{BlockIncidenceDescriptor}(undef, length(ranges))
    @inbounds for (block_index, block) in enumerate(ranges)
        active, colptr, local_rows = _block_a_structural_entries(
            system.A, block,
        )
        signature = _mix_block_structure!(
            signature, block, active, colptr, local_rows,
        )
        # The J_b × J_b output tile, stored full (both triangles and the
        # diagonal), exactly as the reference assembly does.
        for column in active, row in active
            push!(pattern_i, row)
            push!(pattern_j, column)
        end
        descriptors[block_index] = BlockIncidenceDescriptor(
            block, active, colptr, local_rows, Int[],
        )
    end
    # Border structure is frozen independently of current numerical zeros.
    @inbounds for column in 1:n
        push!(pattern_i, column)
        push!(pattern_j, n + 1)
        push!(pattern_i, n + 1)
        push!(pattern_j, column)
    end
    push!(pattern_i, n + 1)
    push!(pattern_j, n + 1)
    frozen = sparse(
        pattern_i, pattern_j, ones(Int, length(pattern_i)),
        dimension, dimension,
    )
    schur_colptr = copy(frozen.colptr)
    schur_rowval = copy(frozen.rowval)
    lookup = Dict{Tuple{Int,Int},Int}()
    @inbounds for column in 1:dimension
        for slot in nzrange(frozen, column)
            lookup[(frozen.rowval[slot], column)] = slot
        end
    end
    @inbounds for (block_index, descriptor) in enumerate(descriptors)
        active = descriptor.active_columns
        nj = length(active)
        tile_slots = Vector{Int}(undef, nj * nj)
        for j_pos in 1:nj
            column = active[j_pos]
            for i_pos in 1:nj
                tile_slots[(i_pos - 1) * nj + j_pos] =
                    lookup[(active[i_pos], column)]
            end
        end
        descriptors[block_index] = BlockIncidenceDescriptor(
            descriptor.rows, descriptor.active_columns,
            descriptor.colptr, descriptor.local_rows, tile_slots,
        )
    end
    border_column_slots = [lookup[(column, n + 1)] for column in 1:n]
    border_row_slots = [lookup[(n + 1, column)] for column in 1:n]
    border_diagonal_slot = lookup[(n + 1, n + 1)]
    plan = BlockIncidencePlan(
        m, n, dimension, ranges, descriptors,
        schur_colptr, schur_rowval,
        border_column_slots, border_row_slots, border_diagonal_slot,
        signature,
    )
    validate_block_incidence_plan(plan, system)
    return plan
end

"""
    validate_block_incidence_plan(plan, system) -> true

Reject plan/system disagreement: dimension mismatch, cone row-range gaps or
overlaps, A-block CSC inconsistencies, tile slots that do not map to their
declared (row, column), Schur slots with no block-tile or border owner, and
pattern-signature drift against the current system structure.
"""
function validate_block_incidence_plan(
    plan::BlockIncidencePlan, system::NewtonSystem,
)
    m, n = size(system.A)
    plan.m == m && plan.n == n || throw(DimensionMismatch(
        "block incidence plan dimensions disagree with system",
    ))
    cone_dimension(system.cone) == m || throw(DimensionMismatch(
        "block incidence plan cone dimension disagrees with rows of A",
    ))
    validate_product_cone_block_ranges(m, plan.block_ranges)
    length(plan.descriptors) == length(plan.block_ranges) ||
        throw(ArgumentError("block incidence plan descriptor count disagrees"))
    plan.dimension == n + 1 || throw(ArgumentError(
        "block incidence plan reduced dimension disagrees",
    ))
    length(plan.schur_colptr) == plan.dimension + 1 ||
        throw(ArgumentError("frozen Schur colptr length disagrees"))
    plan.schur_colptr[1] == 1 &&
        plan.schur_colptr[end] - 1 == length(plan.schur_rowval) ||
        throw(ArgumentError("frozen Schur CSC structure is inconsistent"))
    covered = falses(length(plan.schur_rowval))
    @inbounds for (block_index, descriptor) in enumerate(plan.descriptors)
        descriptor.rows == plan.block_ranges[block_index] ||
            throw(ArgumentError(
                "block descriptor rows drift from the frozen cone ranges",
            ))
        active = descriptor.active_columns
        colptr = descriptor.colptr
        local_rows = descriptor.local_rows
        block_dimension = length(descriptor.rows)
        length(active) + 1 == length(colptr) || throw(ArgumentError(
            "block A-CSC colptr length disagrees",
        ))
        colptr[1] == 1 && colptr[end] - 1 == length(local_rows) ||
            throw(ArgumentError("block A-CSC structure is inconsistent"))
        issorted(active) || throw(ArgumentError(
            "block active variable columns must be ascending",
        ))
        isempty(active) ||
            (first(active) >= 1 && last(active) <= n) ||
            throw(ArgumentError("block active variable column out of range"))
        length(descriptor.tile_slots) == length(active)^2 ||
            throw(ArgumentError("block tile slot count disagrees"))
        nj = length(active)
        for j_pos in 1:nj
            for pointer in colptr[j_pos]:(colptr[j_pos + 1] - 1)
                local_row = local_rows[pointer]
                1 <= local_row <= block_dimension || throw(ArgumentError(
                    "block A-CSC local row out of range",
                ))
            end
            for i_pos in 1:nj
                slot = descriptor.tile_slots[(i_pos - 1) * nj + j_pos]
                1 <= slot <= length(plan.schur_rowval) ||
                    throw(ArgumentError("block tile slot out of range"))
                row = plan.schur_rowval[slot]
                column = _schur_slot_column(plan.schur_colptr, slot)
                row == active[i_pos] && column == active[j_pos] ||
                    throw(ArgumentError(
                        "block tile slot does not match its (row, column) tile",
                    ))
                covered[slot] = true
            end
        end
    end
    @inbounds for column in 1:n
        slot = plan.border_column_slots[column]
        plan.schur_rowval[slot] == column &&
            _schur_slot_column(plan.schur_colptr, slot) == n + 1 ||
            throw(ArgumentError(
                "border column slot does not match (column, n+1)",
            ))
        covered[slot] = true
        slot = plan.border_row_slots[column]
        plan.schur_rowval[slot] == n + 1 &&
            _schur_slot_column(plan.schur_colptr, slot) == column ||
            throw(ArgumentError(
                "border row slot does not match (n+1, column)",
            ))
        covered[slot] = true
    end
    slot = plan.border_diagonal_slot
    plan.schur_rowval[slot] == n + 1 &&
        _schur_slot_column(plan.schur_colptr, slot) == n + 1 ||
        throw(ArgumentError(
            "border diagonal slot does not match (n+1, n+1)",
        ))
    covered[slot] = true
    all(covered) || throw(ArgumentError(
        "frozen Schur pattern has slots without a block tile or border owner",
    ))
    _block_incidence_source_signature(system) == plan.signature ||
        throw(ArgumentError(
            "block incidence plan signature does not match the current system structure",
        ))
    return true
end
