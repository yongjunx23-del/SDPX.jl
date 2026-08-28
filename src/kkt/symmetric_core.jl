#=====================================================================#
#    SymmetricCorePattern — frozen CSC pattern and in-place numeric
#    refill for the Clarabel-style symmetric augmented HSD core.
#
#        K = [ 0    Ar' ]
#            [ Ar  -Theta ]
#
#    This file owns no sign convention: the block signs `Ar` / `-Theta`
#    and the rank-reduced coordinates `x = V*xr` come from the frozen
#    five-equation system (docs/design/NEWTON_SYSTEM.md §"Symmetric
#    augmented-core oracle").  Only the lower triangle of `K` is stored
#    in one setup-owned CSC buffer.
#
#    Structure contract:
#      * the `x` block diagonal (rows/columns 1..nr) is structurally
#        present as numerical zeros so a later LDL driver never needs a
#        pattern change;
#      * the `Ar` block occupies rows nr+1..nr+m in x columns 1..nr
#        (lower triangle) with the value `+Ar[i,j]`, one slot per
#        structural Ar entry in Ar CSC order;
#      * the `-Theta` block occupies the lower triangle of the y block
#        (columns nr+1..nr+m) with a dense lower triangle per cone row
#        range.
#
#    The pattern signature depends only on Ar colptr/rowval, the
#    dimensions, the ordered cone block ranges, and the declared per-block
#    structural shape — never on numeric values.  Numeric refills mutate
#    only `nzval`; colptr/rowval/signature are frozen.
#=====================================================================#

"""
    SymmetricCorePattern{T}

Setup-owned frozen CSC pattern plus owned numeric buffer for the symmetric
augmented core `K = [0 Ar'; Ar -Theta]` in rank-reduced coordinates.
Only the lower triangle is stored.  `nzval` is a flat CSC value buffer
whose slots are addressed by `ar_slots`, `theta_slots`, and
`x_diag_slots`; `colptr`/`rowval` are immutable after construction.
"""
struct SymmetricCorePattern{T<:AbstractFloat}
    nr::Int                        # rank-reduced x dimension
    m::Int                         # y / cone dimension
    dimension::Int                 # nr + m
    ar_colptr::Vector{Int}         # frozen Ar colptr (m × nr)
    ar_rowval::Vector{Int}         # frozen Ar rowval
    block_ranges::Vector{UnitRange{Int}}
    block_shapes::Vector{Symbol}
    colptr::Vector{Int}            # frozen lower-triangle CSC colptr
    rowval::Vector{Int}            # frozen lower-triangle CSC rowval
    ar_slots::Vector{Int}          # nzval slots for Ar entries, Ar CSC order
    theta_slots::Vector{Int}       # nzval slots for -Theta lower triangles
    x_diag_slots::Vector{Int}      # nzval slots for the structural zero x diagonal
    nzval::Vector{T}               # owned numeric buffer (lower triangle)
    signature::UInt64
end

"""Validate that the ordered cone block ranges exactly cover `1:m`."""
function _validate_core_blocks(
    m::Int,
    block_ranges::AbstractVector{<:UnitRange{Int}},
    block_shapes::AbstractVector{Symbol},
)
    length(block_ranges) == length(block_shapes) || throw(ArgumentError(
        "symmetric core cone block range/shape counts disagree",
    ))
    m >= 0 || throw(ArgumentError("symmetric core cone dimension must be nonnegative"))
    expected = 1
    for (index, rows) in enumerate(block_ranges)
        isempty(rows) && throw(ArgumentError(
            "symmetric core cone block ranges must be nonempty",
        ))
        first(rows) == expected || throw(ArgumentError(
            "symmetric core cone block gap or overlap: expected row " *
            "$expected, got $rows",
        ))
        last(rows) <= m || throw(DimensionMismatch(
            "symmetric core cone block $rows exceeds cone dimension $m",
        ))
        _block_shape_code(block_shapes[index])  # validates the shape symbol
        expected = last(rows) + 1
    end
    expected == m + 1 || throw(ArgumentError(
        "symmetric core cone block ranges must cover 1:$m exactly",
    ))
    return true
end

"""Structural shape code for the per-block Theta triangle storage."""
@inline function _block_shape_code(shape::Symbol)
    shape === :dense_lower && return UInt8(0x01)
    throw(ArgumentError(
        "unsupported symmetric core block shape $(shape); only :dense_lower is frozen",
    ))
end

"""FNV-1a mix helper for the structural pattern signature."""
@inline function _core_pattern_mix(signature::UInt64, value::Integer)
    signature ⊻= UInt64(value)
    return signature * UInt64(0x100000001b3)
end

"""Deterministic structural signature over pattern inputs only (no values)."""
function _symmetric_core_structure_signature(
    nr::Int, m::Int,
    ar_colptr::AbstractVector{Int}, ar_rowval::AbstractVector{Int},
    block_ranges::AbstractVector{<:UnitRange{Int}},
    block_shapes::AbstractVector{Symbol},
)
    signature = UInt64(0xcbf29ce484222325)
    signature = _core_pattern_mix(signature, nr)
    signature = _core_pattern_mix(signature, m)
    for value in ar_colptr
        signature = _core_pattern_mix(signature, value)
    end
    for value in ar_rowval
        signature = _core_pattern_mix(signature, value)
    end
    for (index, rows) in enumerate(block_ranges)
        signature = _core_pattern_mix(signature, first(rows))
        signature = _core_pattern_mix(signature, last(rows))
        signature = _core_pattern_mix(
            signature, Int(_block_shape_code(block_shapes[index])),
        )
    end
    return signature
end

"""Build the frozen symmetric-core pattern from structural `Ar`."""
function SymmetricCorePattern{T}(
    Ar::SparseMatrixCSC{T},
    block_ranges::AbstractVector{<:UnitRange{Int}},
    block_shapes::AbstractVector{Symbol},
) where {T<:AbstractFloat}
    m, nr = size(Ar)
    _validate_core_blocks(m, block_ranges, block_shapes)
    dimension = nr + m

    # ---- Frozen lower-triangle CSC structure ---------------------
    colptr = Vector{Int}(undef, dimension + 1)
    rowval = Int[]
    colptr[1] = 1
    @inbounds for j in 1:nr
        push!(rowval, j)  # structural zero x diagonal
        for pointer in nzrange(Ar, j)
            push!(rowval, nr + Ar.rowval[pointer])
        end
        colptr[j + 1] = length(rowval) + 1
    end
    for (index, rows) in enumerate(block_ranges)
        last_row = last(rows)
        for column in first(rows):last_row
            for row in column:last_row
                push!(rowval, nr + row)
            end
            colptr[nr + column + 1] = length(rowval) + 1
        end
    end
    length(colptr) == dimension + 1 || throw(ArgumentError(
        "symmetric core CSC colptr length disagrees with its dimension",
    ))
    colptr[end] - 1 == length(rowval) || throw(ArgumentError(
        "symmetric core CSC structure is inconsistent",
    ))

    # ---- Slot maps -------------------------------------------------
    ar_slots = Int[]
    x_diag_slots = Vector{Int}(undef, nr)
    theta_slots = Int[]
    slot = 0
    @inbounds for j in 1:nr
        slot += 1
        x_diag_slots[j] = slot
        for _ in nzrange(Ar, j)
            slot += 1
            push!(ar_slots, slot)
        end
    end
    for (index, rows) in enumerate(block_ranges)
        last_row = last(rows)
        for column in first(rows):last_row
            for row in column:last_row
                slot += 1
                push!(theta_slots, slot)
            end
        end
    end
    slot == length(rowval) || throw(ArgumentError(
        "symmetric core slot maps do not cover the frozen CSC buffer",
    ))

    signature = _symmetric_core_structure_signature(
        nr, m, Ar.colptr, Ar.rowval, block_ranges, block_shapes,
    )
    nzval = alloc_zeros(T, slot)
    return SymmetricCorePattern{T}(
        nr, m, dimension, Vector{Int}(Ar.colptr), Vector{Int}(Ar.rowval),
        UnitRange{Int}[rows for rows in block_ranges],
        Symbol[shape for shape in block_shapes],
        colptr, rowval, ar_slots, theta_slots, x_diag_slots, nzval, signature,
    )
end

SymmetricCorePattern(
    Ar::SparseMatrixCSC{T},
    block_ranges::AbstractVector{<:UnitRange{Int}},
    block_shapes::AbstractVector{Symbol},
) where {T<:AbstractFloat} =
    SymmetricCorePattern{T}(Ar, block_ranges, block_shapes)

"""Dimension of the symmetric core operator `K` (`nr + m`)."""
symmetric_core_dimension(pattern::SymmetricCorePattern) = pattern.dimension

"""Frozen structural pattern signature of the core."""
symmetric_core_signature(pattern::SymmetricCorePattern) = pattern.signature

"""Lower-triangle CSC value buffer (owned, numeric)."""
symmetric_core_nzval(pattern::SymmetricCorePattern) = pattern.nzval

"""Frozen lower-triangle CSC colptr."""
symmetric_core_colptr(pattern::SymmetricCorePattern) = pattern.colptr

"""Frozen lower-triangle CSC rowval."""
symmetric_core_rowval(pattern::SymmetricCorePattern) = pattern.rowval

"""Validate the numeric `Ar`/`Theta` against the frozen pattern."""
function validate_symmetric_core(
    pattern::SymmetricCorePattern{T},
    Ar::SparseMatrixCSC{T},
    Theta::AbstractMatrix{T},
) where {T<:AbstractFloat}
    m, nr = size(Ar)
    m == pattern.m && nr == pattern.nr || throw(DimensionMismatch(
        "symmetric core numeric Ar dimensions $(size(Ar)) disagree with " *
        "frozen $(pattern.m)×$(pattern.nr)",
    ))
    size(Theta) == (pattern.m, pattern.m) || throw(DimensionMismatch(
        "symmetric core Theta dimension $(size(Theta)) disagrees with frozen " *
        "cone dimension $(pattern.m)",
    ))
    Ar.colptr == pattern.ar_colptr || throw(ArgumentError(
        "symmetric core numeric Ar colptr drifted from the frozen pattern",
    ))
    Ar.rowval == pattern.ar_rowval || throw(ArgumentError(
        "symmetric core numeric Ar rowval drifted from the frozen pattern",
    ))
    all(isfinite, Ar.nzval) || throw(ArgumentError(
        "symmetric core numeric Ar contains non-finite data",
    ))
    all(isfinite, Theta) || throw(ArgumentError(
        "symmetric core Theta contains non-finite data",
    ))
    @inbounds for column in 1:pattern.m
        for row in column:pattern.m
            Theta[row, column] == Theta[column, row] || throw(ArgumentError(
                "symmetric core Theta is not symmetric at ($row, $column)",
            ))
        end
    end
    return true
end

"""
    refill!(pattern, Ar, Theta) -> pattern

Validate `Ar`/`Theta` against the frozen pattern and write `Ar` and
`-Theta` (lower triangles) plus the structural zero `x` diagonal into the
owned `nzval` buffer.  Never changes `colptr`, `rowval`, or `signature`
and never allocates a new global matrix.  Each BigFloat store is a fresh
object, so the buffer stays ownership-safe.
"""
function refill!(
    pattern::SymmetricCorePattern{T},
    Ar::SparseMatrixCSC{T},
    Theta::AbstractMatrix{T},
) where {T<:AbstractFloat}
    validate_symmetric_core(pattern, Ar, Theta)
    nzval = pattern.nzval
    @inbounds for j in 1:pattern.nr
        nzval[pattern.x_diag_slots[j]] = zero(T)
    end
    slot_index = 0
    @inbounds for j in 1:pattern.nr
        for pointer in nzrange(Ar, j)
            slot_index += 1
            nzval[pattern.ar_slots[slot_index]] = Ar.nzval[pointer]
        end
    end
    slot_index = 0
    for (index, rows) in enumerate(pattern.block_ranges)
        last_row = last(rows)
        for column in first(rows):last_row
            for row in column:last_row
                slot_index += 1
                nzval[pattern.theta_slots[slot_index]] = -Theta[row, column]
            end
        end
    end
    return pattern
end

"""
    materialize_dense(pattern) -> Matrix{T}

Expand the stored lower triangle into a full dense `K`.  Upper entries are
copied from the lower triangle, so the result is bitwise symmetric for the
stored values.
"""
function materialize_dense(pattern::SymmetricCorePattern{T}) where {T<:AbstractFloat}
    d = pattern.dimension
    K = alloc_zeros(T, d, d)
    colptr = pattern.colptr
    rowval = pattern.rowval
    nzval = pattern.nzval
    @inbounds for j in 1:d
        for pointer in colptr[j]:(colptr[j + 1] - 1)
            row = rowval[pointer]
            K[row, j] = nzval[pointer]
        end
    end
    @inbounds for j in 1:d
        for row in (j + 1):d
            K[j, row] = K[row, j]
        end
    end
    return K
end
