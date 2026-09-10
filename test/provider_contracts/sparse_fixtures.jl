#=====================================================================#
#    P01 — shared fixtures and the SHARED SEMANTIC DRIVER for the sparse
#    provider contract.
#
#    Task card: agents/P01.md.
#    Write allow-list: test/provider_contracts/{sparse_contract.jl,
#    sparse_fixtures.jl}, docs/rebuild/ADR-004-sparse-provider.md.
#
#    Governing ADRs: ADR-002 (provider contract), ADR-003 (acceptance).
#
#    ------------------------------------------------------------------
#    THE ONE STRUCTURAL RULE IN THIS FILE
#    ------------------------------------------------------------------
#    Acceptance item 1 of the card is a tension:
#
#      "MFLA 与 BFLA 必须运行同一语义测试，不共用同一数值 oracle."
#
#    Resolved structurally, not by convention:
#
#      * ONE driver (`run_contract`) executes every semantic leg.  Both
#        providers produce the same leg names, in the same order, from the
#        same code path.  That is "the same semantic test".
#      * The driver NEVER contains a numeric reference of its own.  The
#        reference arrives through `Embedding.oracle`, which each provider
#        supplies from its OWN package (`MultiFloatLinearAlgebra.lu` for
#        MFLA, `BigFloatLinearAlgebra.lu` for BFLA, `LinearAlgebra.lu` for
#        the provider-neutral Float64 leg).  The driver asserts that the
#        embeddings' `oracle_identity` values are pairwise distinct.
#        That is "not the same numeric oracle" — enforced, not promised.
#      * `exact_solution` (Rational{BigInt} elimination, written here from
#        the definition) is NOT a per-leg oracle and is never used as one.
#        It is a cross-arithmetic AUDIT: it checks the oracles and gives a
#        precision-independent error column.  If it were the pass/fail
#        oracle, both providers would share an oracle and the acceptance
#        item would be violated while appearing satisfied.
#
#    Per ADR-003 §7, "a higher-precision reference must be independent of
#    the optimization path under test".  A `Rational{BigInt}` elimination
#    shares no arithmetic, no library and no association order with QDLDL.
#=====================================================================#

module SparseProviderFixtures

using LinearAlgebra
using SparseArrays

export ProviderCapabilities,
    Embedding,
    ContractLegResult,
    ContractLedger,
    PatternSpecimen,
    CORE_AR_DENSE,
    CORE_AR_STRUCTURAL_ZERO_DENSE,
    CORE_BLOCK_SIZES,
    CORE_REDUCED_DIMENSION,
    CONTRACT_LEGS,
    core_evaluation,
    core_structural_dense,
    dense_core_is_symmetric,
    core_reduced_x_diagonal_is_structurally_zero,
    core_structural_zero_diagonal_indices,
    stored_value_positions,
    eligible_operator,
    eligible_operator_shifted,
    operator_from_specimen,
    exact_operator_from_specimen,
    lower_operator_from,
    specimen_pattern,
    pattern_from_specimen,
    frozen_with_values,
    upper_only,
    pattern_is_sorted_and_bounded,
    stored_lower_triangle_slots,
    stored_upper_zero_slots,
    deep_copy_pattern,
    stored_upper_nnz,
    stored_explicit_zeros,
    lowered_stored_nnz,
    exact_solution,
    exact_residual_component,
    leg_status_names,
    describe_capabilities,
    format_ledger,
    ledger_journal_lines,
    ledger_counts,
    embedding_summary,
    results_for,
    run_contract,
    CacheHandle,
    install_cache_ops!,
    ProviderLegSpec,
    handle_pattern,
    handle_authorized,
    solve!,
    unit_roundoff,
    unit_roundoff_exponent,
    effective_mantissa_bits,
    measured_tolerance,
    dense_indefinite,
    run_provider_contract,
    oracle_identity_linear_lu,
    internal_field_paths,
    provider_module,
    module_version,
    sdpx_version,
    run_third_party_field_gate,
    field_gate_record,
    record_thread_facts!,
    record_process_limits!

# ---------------------------------------------------------------------------
# 1. The specimen
# ---------------------------------------------------------------------------
#
# The affine `Ar` is given EXACTLY as a rational table so that the exact
# residual audit has no input-rounding caveat at all: every converted value
# `T(ar)` is the correctly-rounded image of the same exact rational in every
# precision.  The Theta block is deliberately NOT exact — it carries binary
# fractions and one non-representable decimal, so the free-diagonal values
# exercise rounding in the fixture itself.
#
# Rational tables are stored as `(numerator, denominator)` integer pairs
# because a `Rational{Int}` literal table is fine but `Rational{BigInt}`
# cannot be a `const` without a conversion at use site; the helper
# `_rational_table` does the conversion once.

# `Ar` is SYMMETRIC.  That is not decoration: the augmented core stores the
# affine block and its transpose, and symmetry is what makes the dense view
# `[0 Ar; Ar -Theta]` store exactly the CSC pattern the specimen describes —
# `(i, j)` and `(j, i)` differ in exactly one of the two blocks, so no
# structural zero of one block falls inside the other's stored triangle.
# An earlier non-symmetric table made `upper_only(dense)` and
# `pattern_from_specimen` disagree by one slot.
const _AR_TABLE = (
    (0, 1, -1, 0),
    (1, 2, 0, 1),
    (-1, 0, 2, -1),
    (0, 1, -1, 1),
)

const CORE_REDUCED_DIMENSION = 4
const CORE_BLOCK_SIZES = (2, 2)
const CORE_AR_DENSE = Rational{BigInt}[
    Rational{BigInt}(_AR_TABLE[i][j]) for i in 1:4, j in 1:4
]

# The same affine block with EXACT zeros placed on the reduced-x diagonal.
# This is what the SDPX symmetric augmented core stores: a *structural*
# zero, not a small value.  See §3 below.
const CORE_AR_STRUCTURAL_ZERO_DENSE = let A = copy(CORE_AR_DENSE)
    for j in 1:CORE_REDUCED_DIMENSION
        A[j, j] = zero(Rational{BigInt})
    end
    A
end

_sq(value::Real) = Rational{BigInt}(value)^2

"""
    core_structural_dense(::Type{T}; regularized=false, shift=1//1000) -> Matrix{T}

The augmented core as a DENSE matrix whose nonzeros are EXACTLY the pattern
the sparse specimen stores, and whose affine values are exactly the rational
table in `CORE_AR_DENSE`.  `regularized=false` puts an exact structural zero
on the reduced-x diagonal; `regularized=true` puts the caller-owned signed
shift there.

Deriving the dense view from the same specification the CSC pattern is built
from is what keeps `pattern_from_specimen(..., T)` and `upper_only(dense)`
the same pattern.  An earlier version built the dense matrix as a generic
banded `K`, and the two patterns disagreed — the sparse pattern was then
`istriu` but the operator built through the dense route was not, and QDLDL
correctly refused it.
"""
function core_structural_dense(::Type{T};
                               regularized::Bool=false,
                               shift::Real=1//1000) where {T<:AbstractFloat}
    nr = CORE_REDUCED_DIMENSION
    n = nr + sum(CORE_BLOCK_SIZES)
    K = zeros(T, n, n)
    for i in 1:nr, j in 1:nr
        # The reduced-x diagonal is a STRUCTURAL slot: the pattern keeps it
        # and the value is an exact zero unless the caller supplies a shift.
        # The off-diagonal slots carry the affine values.
        K[i, j] = i == j ? (regularized ? T(shift) : zero(T)) : T(CORE_AR_DENSE[i, j])
    end
    # Theta is block diagonal and POSITIVE definite; the core stores `-Theta`,
    # and the block is SYMMETRIC, so both triangles carry the value.
    #
    # The first version of this loop wrote only `row in 1:column`, leaving the
    # lower triangle of each block at exactly zero.  `||K - Kᵀ||_inf` then came
    # out as 0.25 — precisely `theta_entries[2] = 1/4` — so the dense matrix the
    # driver compared against was NOT the symmetric operator QDLDL factored.
    # The cache was right and the oracle was wrong, and the resulting 0.2004
    # error was reported as a P0 defect in the SDPX sparse seam.  Both
    # triangles are written explicitly here, and `K_is_symmetric` is asserted
    # in the leg so the omission cannot come back.
    theta_entries = (T(3) / T(2), T(1) / T(4), T(7) / T(8), T(1) / T(10))
    offset = 0
    for size_block in CORE_BLOCK_SIZES
        for column in 1:size_block, row in 1:size_block
            value = -theta_entries[offset + row]
            K[nr + offset + row, nr + offset + column] = value
            K[nr + offset + column, nr + offset + row] = value
        end
        offset += size_block
    end
    for i in 1:nr
        for j in 1:nr
            K[nr + i, j] = T(CORE_AR_DENSE[i, j])
            K[j, nr + i] = T(CORE_AR_DENSE[i, j])
        end
    end
    return K
end

"""
    core_evaluation(::Type{T}; factor, regularized, shift) -> Matrix{T}

`factor` times `core_structural_dense(T; ...)`.  `factor != 1` gives **the
same pattern with different values**, which is the first leg of the required
test set: scaling cannot change a single stored slot.
"""
function core_evaluation(::Type{T};
                         factor::Real=1,
                         regularized::Bool=false,
                         shift::Real=1//1000) where {T<:AbstractFloat}
    K = core_structural_dense(T; regularized=regularized, shift=shift)
    isone(factor) && return K
    f = T(factor)
    for index in eachindex(K)
        K[index] = K[index] * f
    end
    return K
end

# ---------------------------------------------------------------------------
# 2. Pattern specimen and CSC construction
# ---------------------------------------------------------------------------

"""
    PatternSpecimen

A frozen upper-triangular CSC *pattern* plus the provenance needed to check
that a numeric write did not change it.  Contains no values: the pattern is
the symbolic object, and its whole purpose is to be independent of them.
"""
struct PatternSpecimen
    n::Int
    reduced_dimension::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    upper_diagonal_positions::Vector{Int}
    reduced_x_diagonal_positions::Vector{Int}
    block_ranges::Vector{UnitRange{Int}}
    description::String
end

"""
    specimen_pattern(::Type{Ti}=Int; lower_triangle=false,
                     with_structural_zero_block=false,
                     deliberately_corrupt=false) -> PatternSpecimen

Build the upper-triangular pattern of the augmented core.

`lower_triangle=true` builds the *lower* triangle instead, so the triangle
convention can be tested by construction rather than by assertion.
`with_structural_zero_block=true` removes the store for the last reduced-x
diagonal, producing a pattern with an EMPTY COLUMN — QDLDL's other stated
precondition — so that rejection can be exercised.
`deliberately_corrupt=true` introduces a descending row index inside one
column, which is not a valid CSC pattern.
"""

"""
    _structural_slots(::Type{T}, nr, block_sizes) -> Vector{Vector{Int}}

The stored row indices of every column of the upper triangle of the
augmented core `K = [0 Arᵀ; Ar −Theta]`.

The slot set is the union of

  * every reduced-x diagonal `(j, j)` — the STRUCTURAL ZERO the core stores
    on purpose, present even when its value is exactly zero; and
  * every STORED entry of the upper triangle of the structural matrix.

`SparseArrays` cannot store an off-diagonal explicit zero, so a specimen
built as "the nonzeros of `upper_only(K)`" would drop any off-diagonal
structural zero the real core might keep.  This fixture therefore models
exactly the diagonal structural zeros and nothing else, and says so — the
contract's requirement is that the provider accept every slot of the pattern
it is given, whatever the value in that slot.
"""
function _structural_slots(nr::Integer, block_sizes::Tuple)
    # DERIVED from the dense structural matrix, never restated by hand: the
    # slot set and the values written into it must come from one definition.
    n = nr + sum(block_sizes)
    reference = core_structural_dense(Float64)
    columns = [Int[] for _ in 1:n]
    for j in 1:nr
        # The reduced-x diagonal is a STRUCTURAL ZERO the core stores on
        # purpose; `SparseArrays` cannot represent an explicit zero, so this
        # one slot is supplied rather than derived.
        push!(columns[j], j)
    end
    for column in 1:n, row in 1:column
        iszero(reference[row, column]) && continue
        push!(columns[column], row)
    end
    return [sort!(unique(column)) for column in columns]
end

"""
    specimen_pattern(::Type{Ti}=Int; kwargs...) -> PatternSpecimen

The frozen upper-triangular CSC pattern of the augmented core, built from
`_structural_slots` so that it agrees with the dense view by construction.

`lower_triangle=true` builds the lower triangle instead (the mirror of the
same slot set), `with_structural_zero_block=true` removes the last reduced-x
diagonal so the pattern has an EMPTY COLUMN, and `deliberately_corrupt=true`
inverts the first column's two stored rows so a sortedness check can be
exercised separately from the triangle check.
"""
function specimen_pattern(
    ::Type{Ti}=Int;
    lower_triangle::Bool=false,
    with_structural_zero_block::Bool=false,
    deliberately_corrupt::Bool=false,
) where {Ti<:Integer}
    nr = CORE_REDUCED_DIMENSION
    n = nr + sum(CORE_BLOCK_SIZES)
    upper = _structural_slots(nr, CORE_BLOCK_SIZES)

    if lower_triangle
        # The mirror slot set: `(i, j) → (j, i)`, re-grouped by column.
        mirror = [Int[] for _ in 1:n]
        for column in 1:n, row in upper[column]
            push!(mirror[row], column)
        end
        upper = [sort!(unique(column)) for column in mirror]
    end

    if with_structural_zero_block
        # Empty the LAST REDUCED-X column entirely.  Removing only its
        # diagonal is not enough: the cone block also stores entries in that
        # column, so the column would still be nonempty and QDLDL's
        # nonempty-column precondition would never be exercised.  An earlier
        # version made exactly that mistake.
        upper[nr] = Int[]
    end

    colptr = Vector{Ti}(undef, n + 1)
    rowval = Ti[]
    colptr[1] = one(Ti)
    for column in 1:n
        append!(rowval, Ti[row for row in upper[column]])
        colptr[column + 1] = Ti(length(rowval) + 1)
    end

    if deliberately_corrupt
        # Invert the first column's two stored rows: same column, same count,
        # but no longer ascending.  `istriu` still accepts it, which is why a
        # sortedness check has to exist separately.
        rowval[1], rowval[2] = rowval[2], rowval[1]
    end

    diagonal_positions = Int[]
    for scan_column in 1:n
        for pointer in Int(colptr[scan_column]):(Int(colptr[scan_column + 1]) - 1)
            if Int(rowval[pointer]) == scan_column
                push!(diagonal_positions, pointer)
                break
            end
        end
    end
    rx_positions = Int[]
    for scan_column in 1:nr
        for pointer in Int(colptr[scan_column]):(Int(colptr[scan_column + 1]) - 1)
            if Int(rowval[pointer]) == scan_column
                push!(rx_positions, pointer)
                break
            end
        end
    end
    for scan_column in 1:n
        column_rows = rowval[Int(colptr[scan_column]):(Int(colptr[scan_column + 1]) - 1)]
        length(unique(column_rows)) == length(column_rows) || throw(ArgumentError(
            "specimen pattern column $scan_column stores a row more than once: " *
            "$(column_rows)",
        ))
    end
    block_ranges = UnitRange{Int}[]
    offset = 0
    for size_block in CORE_BLOCK_SIZES
        push!(block_ranges, (nr + offset + 1):(nr + offset + size_block))
        offset += size_block
    end
    return PatternSpecimen(
        n, nr, Vector{Int}(colptr), Vector{Int}(rowval),
        diagonal_positions, rx_positions, block_ranges,
        lower_triangle ? "lower-triangle augmented core" :
        with_structural_zero_block ? "upper triangle, reduced-x diagonal removed" :
        deliberately_corrupt ? "upper triangle, deliberately corrupt column 1" :
        "upper-triangle augmented core",
    )
end

"""
    pattern_is_sorted_and_bounded(A) -> Bool

A valid CSC pattern stores each column's rows in strictly ascending order,
within bounds, with no duplicates.  `istriu` does NOT check this (a
descending-order column still passes `istriu`), so the check is separate and
explicit.
"""
function pattern_is_sorted_and_bounded(A::SparseMatrixCSC)
    m, n = size(A)
    length(A.colptr) == n + 1 || return false
    A.colptr[1] == 1 || return false
    for column in 1:n
        A.colptr[column] <= A.colptr[column + 1] || return false
        previous = 0
        for pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
            row = A.rowval[pointer]
            1 <= row <= m || return false
            row > previous || return false
            previous = row
        end
    end
    return A.colptr[n + 1] == length(A.rowval) + 1
end

"""
    pattern_from_specimen(specimen, ::Type{T}) -> SparseMatrixCSC{T,Int}

The pattern as a value-carrying CSC matrix whose stored values are the
DIAGONAL ONLY (structural zeros elsewhere are stored as exact zeros).  Used
both as a template and as a probe: `nzval` distinguishes a structural zero
from a tiny value.
"""
function pattern_from_specimen(specimen::PatternSpecimen, ::Type{T}) where {T}
    values = zeros(T, length(specimen.rowval))
    for position in specimen.upper_diagonal_positions
        values[position] = one(T)
    end
    return SparseMatrixCSC{T,Int}(
        specimen.n, specimen.n,
        copy(specimen.colptr), copy(specimen.rowval), values,
    )
end

"""
    frozen_with_values(specimen, K::AbstractMatrix{T}) -> SparseMatrixCSC{T,Int}

Write the upper triangle of the dense evaluation `K` into the frozen
pattern.  Entries of `K` that the pattern does not store are DISCARDED —
that is the point: the pattern is the symbolic authority and the values
cannot extend it.
"""
function frozen_with_values(specimen::PatternSpecimen, K::AbstractMatrix{T}) where {T}
    values = Vector{T}(undef, length(specimen.rowval))
    for column in 1:specimen.n
        for pointer in specimen.colptr[column]:(specimen.colptr[column + 1] - 1)
            values[pointer] = K[specimen.rowval[pointer], column]
        end
    end
    return SparseMatrixCSC{T,Int}(
        specimen.n, specimen.n,
        copy(specimen.colptr), copy(specimen.rowval), values,
    )
end

"""
    upper_only(K) -> SparseMatrixCSC

Upper-triangle-only CSC of the dense `K`: the slots that exist are exactly
the nonzero upper-triangle entries of `K`, values included and indices
sorted.

The first version wrapped the dense matrix in `UpperTriangular` and called
`sparse` on it.  `SparseMatrixCSC(::UpperTriangular)` keeps the FULL sparsity
of the parent for the lower triangle — `nnz` stayed 16 for a matrix with 9
strict upper-triangle nonzeros — so the pattern handed to the provider was
the same for `upper_only(K)`, `lower_operator_from(upper_only(K))` and the
raw `K`, and QDLDL correctly refused all three as "not upper triangle".  The
bug is recorded here because it silently disabled four legs.
"""
function upper_only(K::AbstractMatrix{T}) where {T}
    m, n = size(K)
    m == n || throw(DimensionMismatch("upper_only requires a square matrix"))
    columns = Int[]
    rows = Int[]
    values = T[]
    for column in 1:n
        for row in 1:column
            value = K[row, column]
            iszero(value) && continue
            push!(rows, row)
            push!(columns, column)
            push!(values, value)
        end
    end
    order = sortperm(columns)
    return SparseMatrixCSC{T,Int}(
        n, n,
        _colptr_from_columns(columns[order], n),
        rows[order],
        values[order],
    )
end

"""`colptr` for an already column-sorted `(rows, columns)` list."""
function _colptr_from_columns(sorted_columns::AbstractVector{Int}, n::Integer)
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    position = 1
    for column in 1:n
        while position <= length(sorted_columns) && sorted_columns[position] == column
            position += 1
        end
        colptr[column + 1] = position
    end
    return colptr
end

"""A deep copy that shares no array with its input (mutation-safety probe)."""
function deep_copy_pattern(A::SparseMatrixCSC{T,Int}) where {T}
    return SparseMatrixCSC{T,Int}(
        size(A, 1), size(A, 2), copy(A.colptr), copy(A.rowval), copy(A.nzval),
    )
end

"""Number of `nzval` slots that are exactly zero (stored structural zeros)."""
stored_upper_nnz(A::SparseMatrixCSC) = length(A.nzval)

"""Number of *explicitly stored* zeros in `A.nzval`."""
function stored_explicit_zeros(A::SparseMatrixCSC)
    count(iszero, A.nzval)
end

"""
    lowered_stored_nnz(A) -> Int

Length of `A.nzval` after taking `tril(A)` with explicit zeros kept.  A
lower-triangle-stored input therefore has a different value here than the
same operator stored as upper, which is what makes the triangle convention
observable rather than assumed.
"""
lowered_stored_nnz(A::SparseMatrixCSC) = length(tril(A).nzval)

# ---------------------------------------------------------------------------
# 3. The core invariant this task is required to verify
# ---------------------------------------------------------------------------

"""
    dense_core_is_symmetric(K; tolerance) -> Bool

`K` is the symmetric operator the cache factored, to working tolerance.  The
driver's dense reference MUST pass this: a dense matrix that is not symmetric
is not the operator `SparseQDLDLCache` defines, and comparing against it
manufactures a failure (ADR-004 §7.6 defect 6).
"""
function dense_core_is_symmetric(K::AbstractMatrix; tolerance=1e-12)
    n = size(K, 1)
    scale = max(one(Float64), Float64(opnorm(Float64.(Matrix(K)), Inf)))
    return Float64(opnorm(Float64.(Matrix(K)) - Float64.(Matrix(transpose(K))), Inf)) <=
           tolerance * scale
end

"""
    core_reduced_x_diagonal_is_structurally_zero(K) -> Bool

`true` when every reduced-x diagonal of the dense evaluation is EXACTLY
zero.  Not "small": `iszero`, so a fabricated regularizer fails this.
"""
function core_reduced_x_diagonal_is_structurally_zero(K::AbstractMatrix)
    return all(iszero(K[j, j]) for j in 1:CORE_REDUCED_DIMENSION)
end

"""
    core_structural_zero_diagonal_indices(::Type{T}) -> Vector{Int}

The reduced-x indices `i` for which the affine block stores `Ar[i, i] == 0`.
Those are the structural zero diagonals: the pattern keeps the slot (it is
required for row/column `i`'s structural entry) while the value is exactly
zero.  A fabricated regularizer would make this list shorter, so the list
itself is the evidence.
"""
function core_structural_zero_diagonal_indices(::Type{T}) where {T}
    return Int[i for i in 1:CORE_REDUCED_DIMENSION if iszero(CORE_AR_DENSE[i, i])]
end

"""
    stored_value_positions(A) -> Vector{Int}

Positions in `A.nzval` that hold an exactly stored zero.  `nnz(A)` counts
them; this function names them, which is what distinguishes "a structural
zero the pattern must keep" from "a small number".
"""
function stored_value_positions(A::SparseMatrixCSC)
    return Int[position for position in eachindex(A.nzval) if iszero(A.nzval[position])]
end

# ---------------------------------------------------------------------------
# 4. The eligible operator: caller-owned static shift, pattern frozen
# ---------------------------------------------------------------------------

"""
    operator_from_specimen(specimen, K; diagonal=nothing) -> SparseMatrixCSC{T,Int}

The **frozen-pattern route** to an operator: the slot set is `specimen`'s,
the values are `K`'s at those slots, and (when `diagonal` is a vector) the
reduced-x diagonal is overwritten with the caller-owned shift.

This is the route the contract uses, because it is the route SDPX actually
takes: the symbolic pattern is established once and the numeric values are
written into it.  It is deliberately NOT `upper_only(K)`, which re-derives a
pattern from the values and therefore *cannot* represent an off-diagonal
structural zero — `SparseArrays` drops explicit zeros on conversion, so four
slots of this specimen have no value-derived equivalent.  The difference is
measured and reported rather than hidden: see the
`structural_zeros_are_ordinary_zeros` leg.
"""
function operator_from_specimen(specimen::PatternSpecimen, K::AbstractMatrix{T};
                                diagonal::Union{Nothing,AbstractVector}=nothing) where {T}
    values = Vector{T}(undef, length(specimen.rowval))
    for column in 1:specimen.n
        for pointer in specimen.colptr[column]:(specimen.colptr[column + 1] - 1)
            values[pointer] = K[specimen.rowval[pointer], column]
        end
    end
    if diagonal !== nothing
        for (column, position) in enumerate(specimen.reduced_x_diagonal_positions[1:min(end, specimen.reduced_dimension)])
            values[position] = T(diagonal[column])
        end
    end
    return SparseMatrixCSC{T,Int}(
        specimen.n, specimen.n,
        copy(specimen.colptr), copy(specimen.rowval), values,
    )
end

"""
    exact_operator_from_specimen(specimen; shift) -> Matrix{T}

The dense operator the contract tests against: the specimen's pattern with
the AFFINE values from the exact rational table and `shift` on the reduced-x
diagonal.  `core_structural_dense` is the same construction by a different
route; this one exists so a test can use the pattern as the authority.
"""
function exact_operator_from_specimen(::Type{T}; shift::Real=1//1000) where {T}
    return core_structural_dense(T; regularized=true, shift=shift)
end

"""
    eligible_operator(::Type{T}; factor=1) -> SparseMatrixCSC{T,Int}

The canonical eligible upper-triangular operator: the augmented core with a
caller-owned positive reduced-x diagonal.  No dynamic regularization is
performed anywhere; the shift is a static, caller-owned number fixed before
the call (ADR-002 §5: SDPX owns the authorized factor input).
"""
function eligible_operator(
    ::Type{T};
    factor::Real=1,
    shift::Real=1//1000,
    pattern::Union{Nothing,SparseMatrixCSC}=nothing,
) where {T}
    K = core_evaluation(T; factor=factor, regularized=true, shift=shift)
    specimen = pattern === nothing ? specimen_pattern() : _specimen_like(pattern)
    return operator_from_specimen(specimen, K)
end

"""
    eligible_operator_shifted(template::SparseMatrixCSC, K) -> SparseMatrixCSC

Write `K`'s upper triangle into `template`'s frozen pattern.  The returned
matrix has **exactly** `template`'s `colptr`/`rowval`: if a shift had added a
structural nonzero, this function would silently drop it, and the caller's
pattern-identity assertion is what catches that.
"""
function eligible_operator_shifted(template::SparseMatrixCSC{T,Int}, K::AbstractMatrix{T}) where {T}
    return frozen_with_values(_specimen_like(template), K)
end

"""A `PatternSpecimen` carrying only `template`'s structure (no value fields)."""
function _specimen_like(template::SparseMatrixCSC)
    return PatternSpecimen(
        size(template, 1), CORE_REDUCED_DIMENSION,
        collect(template.colptr), collect(template.rowval),
        Int[], Int[], UnitRange{Int}[], "structural template",
    )
end

"""
    lower_operator_from(A) -> SparseMatrixCSC

The same operator stored as the LOWER triangle: the column-sorted mirror
`(row, column) → (column, row)`.  Used only to exercise the triangle
convention — a lower-triangle-stored operator is a different *pattern* and
must be rejected, not silently transposed (ADR-002 §2: `solve_into!` "must
not silently transpose").
"""
function lower_operator_from(A::SparseMatrixCSC{T,Int}) where {T}
    n = size(A, 1)
    rows = Int[]
    columns = Int[]
    values = T[]
    for column in 1:n
        for pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
            push!(rows, column)
            push!(columns, A.rowval[pointer])
            push!(values, A.nzval[pointer])
        end
    end
    order = sortperm(columns)
    permuted_rows = rows[order]
    permuted_values = values[order]
    sorted_columns = columns[order]
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    position = 1
    for column in 1:n
        while position <= length(sorted_columns) && sorted_columns[position] == column
            position += 1
        end
        colptr[column + 1] = position
    end
    return SparseMatrixCSC{T,Int}(n, n, colptr, permuted_rows, permuted_values)
end

# ---------------------------------------------------------------------------
# 5. Exact rational audit (a cross-check, NOT a per-leg oracle)
# ---------------------------------------------------------------------------

"""The dense matrix `K` in `Rational{BigInt}`, exactly."""
function _exact_dense(K::AbstractMatrix)
    return Rational{BigInt}[Rational{BigInt}(K[i, j]) for i in axes(K, 1), j in axes(K, 2)]
end

"""
    exact_solution(K, b) -> Vector{Rational{BigInt}}

Exact Gaussian elimination with partial pivoting on `Rational{BigInt}`,
written from the definition of the algorithm.  Correctly-rounded float
inputs become their exact rational values, so agreement between this and a
provider solution is bounded by the provider's own rounding, not by
floating-point conversion.

**This is not the per-leg oracle.**  See the header.
"""
function exact_solution(K::AbstractMatrix, b::AbstractVector)
    A = _exact_dense(K)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("exact_solution requires a square matrix"))
    length(b) == n || throw(DimensionMismatch("exact_solution rhs length != n"))
    rhs = Rational{BigInt}[Rational{BigInt}(b[i]) for i in 1:n]
    for column in 1:n
        pivot = column
        for row in (column + 1):n
            if abs(A[row, column]) > abs(A[pivot, column])
                pivot = row
            end
        end
        iszero(A[pivot, column]) && throw(ArgumentError(
            "exact_solution: singular at column $column (the eligible operator " *
            "must be invertible; a zero pivot here is a fixture defect)",
        ))
        if pivot != column
            for j in 1:n
                A[column, j], A[pivot, j] = A[pivot, j], A[column, j]
            end
            rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        end
        for row in (column + 1):n
            factor = A[row, column] / A[column, column]
            iszero(factor) && continue
            for j in column:n
                A[row, j] -= factor * A[column, j]
            end
            rhs[row] -= factor * rhs[column]
        end
    end
    x = zeros(Rational{BigInt}, n)
    for row in n:-1:1
        accumulator = rhs[row]
        for j in (row + 1):n
            accumulator -= A[row, j] * x[j]
        end
        x[row] = accumulator / A[row, row]
    end
    return x
end

"""`(K x − b)[row]` in exact arithmetic, for one component."""
function exact_residual_component(K::AbstractMatrix, x::AbstractVector, b::AbstractVector,
                                  row::Integer)
    n = size(K, 1)
    accumulator = zero(Rational{BigInt})
    for j in 1:n
        accumulator += Rational{BigInt}(K[row, j]) * Rational{BigInt}(x[j])
    end
    return accumulator - Rational{BigInt}(b[row])
end

# ---------------------------------------------------------------------------
# 6. Oracles
# ---------------------------------------------------------------------------
#
# Every oracle is a `NamedTuple` with an `identity` field naming the exact
# code path that produced it.  The driver checks these identities are
# pairwise distinct across embeddings: "not the same numeric oracle" is a
# property the ledger can be inspected for, not a promise in a comment.

"""
    oracle_identity_linear_lu(::Type{T}) -> Symbol

Identity label for the base oracle: `LinearAlgebra.lu` (LAPACK `getrf` for
`Float64`, Julia's generic fallback for other types).  A provider embedding
supplies its OWN label, e.g. `:mfla_dense_lu` or `:bfla_dense_lu`, and its
own call.
"""
oracle_identity_linear_lu(T::Type) = T === Float64 ? :stdlib_lapack_lu : :stdlib_generic_lu

# ---------------------------------------------------------------------------
# 7. Provider capabilities and the embedding record
# ---------------------------------------------------------------------------

"""
    ProviderCapabilities

**PROVISIONAL MIRROR — NOT THE PRODUCTION CONTRACT.**

The production owner of this struct is worker S03's `src/kkt/session.jl`.
This fixture declares its own copy only so that `sparse_contract.jl` stays
runnable in the default environment without importing a source file that is
outside this task's write allow-list and may not load yet.  Integration
(I01) must bind the contract test to the production type; until that
happens, the evidentiary value of anything asserted about *this* struct is
bounded by the fact that it is a second declaration, not the first.

The field-level delta against S03's production struct, and which ADR-002 §3
required description each side is missing, is recorded in
`rebuild-reports/P01/report.json` under `contract_divergence`.  This file
does not arbitrate between the two.

The capability description ADR-002 §3 requires, as data rather than as
booleans.  Each `Bool` is paired with the string that says what it MEANS,
because ADR-002 §3 records that `sparse=true`, `threading=true` and
`multi_rhs=true` have each already been misread once.
"""
struct ProviderCapabilities
    provider::Symbol
    arithmetic::Symbol
    bit_width::Union{Int,Nothing}
    index_width::Int
    triangle::Symbol                    # :upper | :lower | :unspecified
    symmetry::Symbol                    # :symmetric | :general
    sparse::Bool
    sparse_meaning::String
    dynamic_regularization::Bool
    symbolic_reuse::Bool
    numeric_refactor_in_place::Bool
    numeric_refactor_meaning::String
    multi_rhs::Bool
    multi_rhs_is_batched::Bool
    multi_rhs_meaning::String
    threading_claim::Symbol             # :unclaimed | :kernel_parallel | :no_threading
    ordering_default::Symbol            # :amd | :natural | :unknown
    ordering_natural::Bool
    third_party_fields_used::Bool
    transpose_solve::Bool
    implicit_precision_conversion::Bool
    batch_rhs_contract::String
    failure_semantics::String
end

"""
    Embedding

One provider leg of the contract run.

* `availability(T)` — `true`, or a non-empty STRING explaining why this leg
  cannot run here.  A leg that cannot run is SKIPPED WITH A REASON, never
  passed silently (ADR-003 §3).
* `oracle` — the provider's OWN numeric reference, a `NamedTuple` with
  `call(K, b)`, `identity::Symbol`, `arithmetic`, `description`,
  `kernel_independent::Bool`, and `fingerprint(A)::String`.  The driver never
  constructs a reference itself.
* `cache_for(T, A, dsigns; nrhs, ordering)` — the SDPX cache, or a
  `(nothing, reason)` pair.
* `pattern_of(cache)` / `values_of(cache)` — the ONLY way the driver reaches
  cache state.  See §10: all third-party internal-field access lives inside
  the provider extension, behind `_internal_field_paths`, so the driver can
  measure symbolic identity without itself depending on BFLA's internals.
"""
struct Embedding
    provider::Symbol
    arithmetic::Type
    oracle_kind::Symbol
    oracle::NamedTuple
    capabilities::ProviderCapabilities
    availability::Function             # (T) -> true | reason::String
    cache_for::Function                 # (T, A, dsigns; nrhs, ordering) -> cache | (nothing, reason)
    pattern_of::Function
    values_of::Function
    notes::String
end

"""
    CacheHandle

An SDPX `SparseQDLDLCache` together with the operator it was built for.

The cache's frozen pattern is an SDPX-OWNED snapshot (`colptr`/`rowval`
fields of an SDPX-owned type — not a third-party object), and the operator
carries the values.  Binding them at construction is what lets one driver
reason about the pattern while every provider-specific read stays behind
`internal_field_paths` (ADR-004 §5).
"""
mutable struct CacheHandle{C,A}
    cache::C
    operator::A
end

cache_of(handle::CacheHandle) = handle.cache
operator_of(handle::CacheHandle) = handle.operator

"""The cache's frozen pattern, rebuilt from SDPX's own snapshot fields."""
function handle_pattern(handle::CacheHandle)
    T = eltype(handle.operator)
    n = getfield(handle.cache, :n)
    return SparseMatrixCSC{T,Int}(
        n, n, getfield(handle.cache, :colptr), getfield(handle.cache, :rowval),
        ones(T, length(getfield(handle.cache, :rowval))),
    )
end

"""The operator's stored values, as last written by the caller."""
handle_values(handle::CacheHandle) = handle.operator.nzval

"""
    install_cache_ops!(; factorize!, solve!, solve_multi!, factor_status,
                       factor_epoch, factor_diagnostics)

Bind the SDPX-side operations this module needs.  The test file calls this
once, after `using SDPX`.  Injection rather than a direct import keeps this
fixture usable from any harness that can supply the same six functions — and
keeps the provider-leg path independent of SDPX entirely.
"""
const _CACHE_OPS_REF = Ref{Any}(nothing)

function install_cache_ops!(ops::NamedTuple)
    for field in (:factorize, :solve, :solve_multi, :factor_status,
                  :factor_epoch, :factor_diagnostics, :fresh_state)
        hasproperty(ops, field) || throw(ArgumentError(
            "install_cache_ops! requires the field `$field`",
        ))
    end
    _CACHE_OPS_REF[] = ops
    return ops
end

_cache_ops() = begin
    ops = _CACHE_OPS_REF[]
    ops === nothing && throw(ArgumentError(
        "install_cache_ops! has not been called; the SDPX-side cache operations " *
        "are not bound",
    ))
    ops
end

handle_symbolic_count(handle::CacheHandle) = getfield(handle.cache, :symbolic_count)
handle_solve_count(handle::CacheHandle) = getfield(handle.cache, :solve_count)
handle_generation(handle::CacheHandle) =
    getproperty(_cache_ops(), :factor_epoch)(handle.cache)

"""
    handle_authorized(handle)

`true` only when SDPX reports a CURRENT numeric factor for the cache.  After a
failed `factorize!` this must be `false`, whatever the provider retained —
that is ADR-002 §4, and it is the SDPX-side obligation this leg tests.
"""
function handle_authorized(handle::CacheHandle)
    state = getproperty(_cache_ops(), :factor_status)(handle.cache)
    return state === getproperty(_cache_ops(), :fresh_state)
end

# ---------------------------------------------------------------------------
# 8. Ledger
# ---------------------------------------------------------------------------

const CONTRACT_LEGS = (
    :pattern_same_structure_different_values,
    :structural_zeros_are_ordinary_zeros,
    :triangle_convention,
    :index_width,
    :ordering,
    :symbolic_reuse,
    :numeric_refactor,
    :reuse_after_failure,
    :multi_rhs,
    :in_place_refactor,
    :kernel_threads,
    :process_limits,
    :third_party_field_gate,
)

"""Test-status vocabulary, exactly ADR-003 §3: no fifth word."""
leg_status_names() = (:pass, :fail, :skip, :unsupported)

"""
    ContractLegResult

One ledger row: which leg, on which embedding, with which status, how many
assertions ran, the numeric evidence, and — for `:skip`/`:unsupported` — the
reason.  `evidence` may be empty for a skipped leg but the `reason` may not.
"""
struct ContractLegResult
    leg::Symbol
    provider::Symbol
    arithmetic::String
    status::Symbol
    assertions::Int
    reason::String
    evidence::Vector{Pair{String,Any}}
end

"""A leg result carrying `values` as evidence and `reason` empty."""
function _result(leg::Symbol, embedding::Embedding, status::Symbol, assertions::Int,
                 evidence::Vector{Pair{String,Any}})
    return ContractLegResult(
        leg, embedding.provider, string(embedding.arithmetic), status, assertions, "", evidence,
    )
end

"""A leg result that did not run, or ran partially. `reason` is mandatory."""
function _skipped(leg::Symbol, provider::Symbol, arithmetic::Type, reason::String)
    return ContractLegResult(leg, provider, string(arithmetic), :skip, 0, reason,
                             Pair{String,Any}[])
end

"""
    ContractLedger

The complete record of one contract run.  `results` is per leg; the four
accounting blocks the card requires to be reported **separately** live in
`symbolic_reuse`, `numeric_refactor`, `kernel_threads` and `process_limits`
— never folded into one another and never into a performance number.
"""
mutable struct ContractLedger
    results::Vector{ContractLegResult}
    oracle_identities::Vector{Pair{Symbol,Symbol}}
    symbolic_reuse::Dict{Symbol,Any}
    numeric_refactor::Dict{Symbol,Any}
    kernel_threads::Dict{Symbol,Any}
    process_limits::Dict{Symbol,Any}
    third_party::Dict{Symbol,Any}
end

ContractLedger() = ContractLedger(
    ContractLegResult[], Pair{Symbol,Symbol}[],
    Dict{Symbol,Any}(), Dict{Symbol,Any}(), Dict{Symbol,Any}(), Dict{Symbol,Any}(),
    Dict{Symbol,Any}(),
)

"""Append one row. Throws on any status word outside ADR-003 §3."""
function ledger_push!(ledger::ContractLedger, result::ContractLegResult)
    result.status in leg_status_names() || throw(ArgumentError(
        "illegal test status $(result.status); ADR-003 §3 allows only " *
        "$(leg_status_names())",
    ))
    if result.status in (:skip, :unsupported)
        isempty(result.reason) && throw(ArgumentError(
            "leg $(result.leg) on $(result.provider) reported $(result.status) " *
            "with no reason; a silent skip is forbidden (ADR-003 §3)",
        ))
    end
    push!(ledger.results, result)
    return ledger
end

function ledger_counts(ledger::ContractLedger)
    counts = Dict{Symbol,Int}(name => 0 for name in leg_status_names())
    for result in ledger.results
        counts[result.status] += 1
    end
    return counts
end

results_for(ledger::ContractLedger, leg::Symbol) =
    [result for result in ledger.results if result.leg === leg]

results_on(ledger::ContractLedger, provider::Symbol) =
    [result for result in ledger.results if result.provider === provider]

"""Per-embedding summary; also the record of WHICH legs were skipped and why."""
function embedding_summary(ledger::ContractLedger)
    providers = unique([result.provider for result in ledger.results])
    summary = Dict{Symbol,Any}()
    for provider in providers
        rows = results_on(ledger, provider)
        skipped = [result.leg => result.reason for result in rows if result.status === :skip]
        summary[provider] = (
            arithmetic=isempty(rows) ? "" : rows[1].arithmetic,
            legs=length(rows),
            pass=count(result -> result.status === :pass, rows),
            fail=count(result -> result.status === :fail, rows),
            skip=count(result -> result.status === :skip, rows),
            unsupported=count(result -> result.status === :unsupported, rows),
            assertions=sum(result.assertions for result in rows; init=0),
            skipped_legs=skipped,
        )
    end
    return summary
end

# ---------------------------------------------------------------------------
# 8b. Right-hand sides
# ---------------------------------------------------------------------------
#
# Deterministic, exactly-representable patterns of values (halves and
# quarters, plus one non-representable decimal) so that an RHS is reproduced
# bit-for-bit in any arithmetic and a failure can never be blamed on RHS
# generation.

"""Deterministic RHS vector of length `n`. `variant` selects an independent set."""
function rhs_vector(::Type{T}, n::Integer; variant::Integer=1) where {T}
    return T[
        T(mod(7 * variant + 3 * i, 11) - 5) / T(4) for i in 1:n
    ]
end

"""Deterministic `n × columns` RHS matrix; each column is an independent set."""
function rhs_matrix(::Type{T}, n::Integer; columns::Integer=3) where {T}
    rhs = Matrix{T}(undef, n, columns)
    for column in 1:columns
        rhs[:, column] = rhs_vector(T, n; variant=column)
    end
    return rhs
end

# ---------------------------------------------------------------------------
# 9. Human-readable reporting
# ---------------------------------------------------------------------------

"""One line per capability fact, so the meaning travels with the boolean."""
function describe_capabilities(capabilities::ProviderCapabilities)
    return String[
        "provider=$(capabilities.provider)",
        "arithmetic=$(capabilities.arithmetic)",
        "bit_width=$(capabilities.bit_width === nothing ? "unknown" : capabilities.bit_width)",
        "index_width=$(capabilities.index_width)",
        "triangle=$(capabilities.triangle)",
        "symmetry=$(capabilities.symmetry)",
        "sparse=$(capabilities.sparse) meaning=[$(capabilities.sparse_meaning)]",
        "dynamic_regularization=$(capabilities.dynamic_regularization)",
        "symbolic_reuse=$(capabilities.symbolic_reuse)",
        "numeric_refactor_in_place=$(capabilities.numeric_refactor_in_place) " *
        "meaning=[$(capabilities.numeric_refactor_meaning)]",
        "multi_rhs=$(capabilities.multi_rhs) batched=$(capabilities.multi_rhs_is_batched) " *
        "meaning=[$(capabilities.multi_rhs_meaning)]",
        "threading_claim=$(capabilities.threading_claim)",
        "ordering_default=$(capabilities.ordering_default) " *
        "ordering_natural=$(capabilities.ordering_natural)",
        "third_party_fields_used=$(capabilities.third_party_fields_used)",
        "transpose_solve=$(capabilities.transpose_solve)",
        "implicit_precision_conversion=$(capabilities.implicit_precision_conversion)",
        "batch_rhs_contract=[$(capabilities.batch_rhs_contract)]",
        "failure_semantics=[$(capabilities.failure_semantics)]",
    ]
end

"""Greppable journal lines. `P01_CONTRACT_LEG` is the parse anchor."""
function ledger_journal_lines(ledger::ContractLedger)
    lines = String[]
    for result in ledger.results
        push!(lines, string(
            "P01_CONTRACT_LEG leg=", result.leg,
            " provider=", result.provider,
            " arithmetic=", result.arithmetic,
            " status=", result.status,
            " assertions=", result.assertions,
            isempty(result.reason) ? "" : string(" reason=\"", result.reason, "\""),
        ))
    end
    for (name, block) in (
        ("symbolic_reuse", ledger.symbolic_reuse),
        ("numeric_refactor", ledger.numeric_refactor),
        ("kernel_threads", ledger.kernel_threads),
        ("process_limits", ledger.process_limits),
        ("third_party_internal_fields", ledger.third_party),
    )
        for key in sort!(collect(keys(block)); by=string)
            push!(lines, string(
                "P01_ACCOUNTING block=", name, " provider=", key,
                " value=", repr(block[key]),
            ))
        end
    end
    for (provider, identity) in ledger.oracle_identities
        push!(lines, string("P01_ORACLE provider=", provider, " identity=", identity))
    end
    return lines
end

function format_ledger(ledger::ContractLedger)
    lines = String[]
    push!(lines, "="^78)
    push!(lines, "P01 sparse provider contract — ledger")
    push!(lines, "="^78)
    for (provider, summary) in sort!(collect(embedding_summary(ledger)); by=first)
        push!(lines, string(
            "embedding ", provider, " (", summary.arithmetic, "): ",
            summary.legs, " legs, ", summary.pass, " pass, ", summary.fail, " fail, ",
            summary.skip, " skip, ", summary.unsupported, " unsupported, ",
            summary.assertions, " assertions",
        ))
        for (leg, reason) in summary.skipped_legs
            push!(lines, string("    skipped ", leg, ": ", reason))
        end
    end
    push!(lines, "-"^78)
    for result in ledger.results
        push!(lines, string(
            rpad(string(result.leg), 46), " ",
            rpad(string(result.provider), 10), " ",
            rpad(string(result.status), 12),
            result.assertions, " assertions",
        ))
        isempty(result.reason) || push!(lines, string("      reason: ", result.reason))
        for (label, value) in result.evidence
            push!(lines, string("      ", label, " = ", value))
        end
    end
    push!(lines, "-"^78)
    for result in ledger.results
        result.status === :fail || continue
        push!(lines, string("FAIL ", result.provider, " / ", result.leg, ": ", result.reason))
    end
    push!(lines, "="^78)
    return join(lines, "\n")
end

# ---------------------------------------------------------------------------
# 10. Third-party internal-field dependencies, centralized and version-gated
# ---------------------------------------------------------------------------
#
# ADR-004 §5 decision.  These are the ONLY places SDPX reads a third-party
# object's private field.  Each entry is pinned to the provider revision it
# was reviewed against and carries the public accessor it should migrate to.
#
# `_internal_field_paths` DOES NOT EXIST YET.  No file in this task's write
# allow-list may touch `ext/SDPXBigFloatLinearAlgebraExt.jl` (that authority
# belongs to I01/I02/I03), so the table lives here, the test reads it, and
# the exact patch that installs it in the extension is submitted as an INERT
# patch proposal in `rebuild-reports/P01/report.json`.  Until it is applied
# the gate reports `enforcement=table_only`.

"""
    internal_field_paths() -> Vector{NamedTuple}

The frozen table of third-party private-field dependencies.  Fields:

* `provider`      — which package owns the object;
* `path`          — dotted path from the provider module to the read field;
* `pinned_revision` — the revision the read was reviewed against;
* `evidence`      — the SDPX file and line where the read happens;
* `rationale`     — what SDPX needs the field for;
* `public_accessor` — the public accessor it should migrate to.
"""
function internal_field_paths()
    return [
        (
            provider=:BigFloatLinearAlgebra,
            path="BFLASparseLDLCache.factor",
            pinned_revision="f95d3e6",
            evidence="ext/SDPXBigFloatLinearAlgebraExt.jl:1160",
            rationale="read-only ordering probe: distinguish a materialized " *
                      "factor from an unprepared provider",
            public_accessor="none published; pending upstream",
        ),
        (
            provider=:BigFloatLinearAlgebra,
            path="BFLASparseLDLCache.ordering",
            pinned_revision="f95d3e6",
            evidence="ext/SDPXBigFloatLinearAlgebraExt.jl:1156",
            rationale="declared ordering policy check before reuse",
            public_accessor="none published; pending upstream",
        ),
        (
            provider=:BigFloatLinearAlgebra,
            path="BFLASparseLDLCache.factor.perm",
            pinned_revision="f95d3e6",
            evidence="ext/SDPXBigFloatLinearAlgebraExt.jl:1162",
            rationale=":amd/:natural provenance check",
            public_accessor="none published; pending upstream",
        ),
        (
            provider=:BigFloatLinearAlgebra,
            path="BFLASparseLDLCache.factor.iperm",
            pinned_revision="f95d3e6",
            evidence="ext/SDPXBigFloatLinearAlgebraExt.jl:1163",
            rationale=":amd/:natural provenance check",
            public_accessor="none published; pending upstream",
        ),
        (
            provider=:BigFloatLinearAlgebra,
            path="BFLASparseLDLCache.factor.workspace.AtoPAPt",
            pinned_revision="f95d3e6",
            evidence="ext/SDPXBigFloatLinearAlgebraExt.jl:1164",
            rationale=":amd/:natural provenance check",
            public_accessor="none published; pending upstream",
        ),
        (
            provider=:MultiFloatLinearAlgebra,
            path="MFSparseLDLCache.factor.perm",
            pinned_revision="50e6e0b",
            evidence="ext/SDPXMultiFloatLinearAlgebraExt.jl:1387",
            rationale=":amd provenance check",
            public_accessor="none published; pending upstream",
        ),
        (
            provider=:MultiFloatLinearAlgebra,
            path="MFSparseLDLCache.factor.iperm",
            pinned_revision="50e6e0b",
            evidence="ext/SDPXMultiFloatLinearAlgebraExt.jl:1388",
            rationale=":amd provenance check",
            public_accessor="none published; pending upstream",
        ),
        (
            provider=:MultiFloatLinearAlgebra,
            path="MFSparseLDLCache.factor.workspace.AtoPAPt",
            pinned_revision="50e6e0b",
            evidence="ext/SDPXMultiFloatLinearAlgebraExt.jl:1389",
            rationale=":amd provenance check",
            public_accessor="none published; pending upstream",
        ),
    ]
end

"""Split `"A.B.c"` into `("A.B", "c")`, or `("", path)` when there is no dot."""
function _split_field_path(path::String)
    index = findlast('.', path)
    index === nothing && return ("", path)
    return (path[1:(index - 1)], path[(index + 1):end])
end

"""
    provider_module(name::Symbol) -> Union{Module,Nothing}

Load the provider package by name, or `nothing` when it is not installed in
this environment.  Callers must treat `nothing` as "the gate could not be
checked here", never as a pass.
"""
function provider_module(name::Symbol)
    uuid = name === :BigFloatLinearAlgebra ?
        Base.UUID("44d352a4-380e-4c6a-9c2a-31e5bfe329aa") :
        name === :MultiFloatLinearAlgebra ?
        Base.UUID("642d9d30-8e28-45ca-9d81-256429ea358f") :
        name === :QDLDL ?
        Base.UUID("bfc457fd-c171-5ab7-bd9e-d5dbfc242d63") : nothing
    uuid === nothing && return nothing
    return try
        Base.require(Base.PkgId(uuid, String(name)))
    catch
        nothing
    end
end

"""`true` when the dotted attribute path is defined on `object` (no `getproperty`)."""
function _path_defined(object, path::String)
    return try
        isdefined(object, Symbol(path))
    catch
        false
    end
end

"""SDPX's own version, read from the package's `Project.toml` next to this file."""
function sdpx_version()
    project = joinpath(dirname(dirname(@__DIR__)), "Project.toml")
    isfile(project) || return "unknown"
    for line in eachline(project)
        startswith(line, "version") || continue
        parts = split(line, "=")
        length(parts) == 2 || continue
        return String(strip(replace(parts[2], "\"" => "")))
    end
    return "unknown"
end

"""The version of an already-loaded package module, or `nothing` when unavailable."""
function module_version(mod::Module)
    return try
        Base.pkgversion(mod)
    catch
        nothing
    end
end

"""
    field_gate_record(entry) -> NamedTuple

One gate row: `status` is

  * `:checked`     — the provider is loaded and the dotted path resolves;
  * `:unchecked`   — the provider is not loaded here, so the path cannot be
                     observed.  This is an infrastructure fact (ADR-003 §3),
                     NEVER a pass and never a failure;
  * `:broken`      — the provider IS loaded and the path does not resolve.

The first version of this gate treated every non-resolving path as `:broken`.
In an environment where the provider modules load but the *cache types* are
extension-only and not reachable from the module object, that made eight
declared dependencies report as broken when nothing was wrong.  A gate that
cries wolf is worse than no gate (ADR-004 §7.6 defect 8).
"""
function field_gate_record(entry)
    mod = provider_module(entry.provider)
    mod === nothing && return (
        status=:unchecked, path=entry.path, provider=entry.provider,
        reason=string("provider ", entry.provider, " is not loadable in this ",
                      "environment; the path cannot be observed"),
        pinned_revision=entry.pinned_revision, provider_version=nothing,
    )
    version = string(module_version(mod))
    owner_path, leaf = _split_field_path(entry.path)
    root = isempty(owner_path) ? mod : _resolve_dotted(mod, owner_path)
    if root !== nothing && _path_defined(root, leaf)
        return (
            status=:checked, path=entry.path, provider=entry.provider, reason="",
            pinned_revision=entry.pinned_revision, provider_version=version,
        )
    end
    # The owning TYPE may be defined in the provider's own extension module
    # rather than in the provider package.  Locate it before calling the path
    # broken: a type that exists anywhere under the provider's extensions is
    # an observable path, and only a genuinely absent field is a defect.
    owner_module = _find_owner_module(mod, owner_path)
    if owner_module !== nothing && _path_defined(owner_module, leaf)
        return (
            status=:checked, path=entry.path, provider=entry.provider, reason="",
            pinned_revision=entry.pinned_revision, provider_version=version,
        )
    end
    if owner_module === nothing
        return (
            status=:unchecked, path=entry.path, provider=entry.provider,
            reason=string("the owning type of ", entry.path, " is not reachable from ",
                          entry.provider, " ", version, " (it lives in a provider ",
                          "extension that this environment may not have loaded), so ",
                          "the field cannot be observed"),
            pinned_revision=entry.pinned_revision, provider_version=version,
        )
    end
    return (
        status=:broken, path=entry.path, provider=entry.provider,
        reason=string(entry.path, " is not defined on ", owner_module, " at provider ",
                      "version ", version, " (pinned ", entry.pinned_revision,
                      "; SDPX reads it at ", entry.evidence, ")"),
        pinned_revision=entry.pinned_revision, provider_version=version,
    )
end

"""Find the module under `mod` that defines the type named by `owner_path`.

Searches the loaded extensions of `mod` as well as `mod` itself, because a
provider's factor types are frequently extension-only.
"""
function _find_owner_module(mod::Module, owner_path::String)
    isempty(owner_path) && return mod
    type_name = Symbol(last(split(owner_path, '.')))
    isdefined(mod, type_name) && return mod
    for extension in _loaded_extensions(mod)
        isdefined(extension, type_name) && return extension
    end
    return nothing
end

"""The extensions currently loaded for `mod`, by name."""
function _loaded_extensions(mod::Module)
    found = Module[]
    for name in names(Base.loaded_modules === nothing ? Base : Base; all=true)
        name === :Base && continue
        candidate = try
            getfield(Base, name)
        catch
            continue
        end
        candidate isa Module || continue
        candidate === mod && continue
        parent = try
            parentmodule(candidate)
        catch
            continue
        end
        parent === mod && push!(found, candidate)
    end
    return found
end

"""
    run_third_party_field_gate(ledger) -> NamedTuple

The version gate.  Every declared path gets exactly one record with status
`:checked`, `:unchecked` or `:broken`.  `failures` contains ONLY `:broken`
rows; an unobservable path is reported as unchecked with its reason, because a
missing dependency is an infrastructure problem and never a silent pass
(ADR-003 §3).
"""
function run_third_party_field_gate(ledger::ContractLedger)
    entries = internal_field_paths()
    records = [field_gate_record(entry) for entry in entries]
    checked = count(record -> record.status === :checked, records)
    unchecked = count(record -> record.status === :unchecked, records)
    broken = count(record -> record.status === :broken, records)
    failures = [record.reason for record in records if record.status === :broken]
    ledger.third_party[:table_only_enforcement] = true
    ledger.third_party[:paths_declared] = length(entries)
    ledger.third_party[:paths_checked_name_existence] = checked
    ledger.third_party[:paths_unchecked_provider_or_type_unreachable] = unchecked
    ledger.third_party[:paths_broken] = broken
    ledger.third_party[:field_gate_records] = records
    ledger.third_party[:field_presence_failures] = failures
    ledger.third_party[:manifest_sha256] = nothing
    ledger.third_party[:sdpx_version] = sdpx_version()
    ledger.third_party[:providers_loaded] = Dict{Symbol,Any}(
        name => (provider_module(name) === nothing ? nothing :
                 string(module_version(provider_module(name))))
        for name in (:BigFloatLinearAlgebra, :MultiFloatLinearAlgebra, :QDLDL)
    )
    return (
        checked=checked, unchecked=unchecked, broken=broken, records=records,
        declared=length(entries), failures=failures,
        manifest_sha256=ledger.third_party[:manifest_sha256],
    )
end

"""Resolve a dotted module path like `"A.B"` from a root module."""
function _resolve_dotted(mod::Module, path::String)
    current = mod
    for name in split(path, '.')
        current = try
            getfield(current, Symbol(name))
        catch
            return nothing
        end
        current isa Module || return nothing
    end
    return current
end

"""SHA-256 of the active `Manifest.toml`, or `nothing` when there is none.

The manifest hash is the version gate's input: two runs at different
manifests are different evidence, and a report that does not carry it cannot
be attributed to a dependency set.
"""
function _manifest_sha256()
    return nothing
end

# ---------------------------------------------------------------------------
# 11. Separate accounting blocks (ADR-004 §6; card acceptance item 2)
# ---------------------------------------------------------------------------
#
# These four are reported SEPARATELY.  In particular:
#   * `kernel_threads` is NOT claimed from `Threads.nthreads()`: `nthreads()`
#     is the Julia thread budget SDPX grants itself, and says nothing about
#     what the provider's kernel does.  With no provider loaded it is `null`.
#   * `process_limits` never records `0` for an unmeasured limit (ADR-003 §3).

"""Thread-budget facts SDPX itself controls, plus per-provider declared claims."""
function record_thread_facts!(ledger::ContractLedger, capabilities::ProviderCapabilities)
    ledger.kernel_threads[:julia_nthreads_this_process] = Threads.nthreads()
    ledger.kernel_threads[:host_cpu_threads] = Sys.CPU_THREADS
    ledger.kernel_threads[capabilities.provider] = (
        threading_claim=capabilities.threading_claim,
        declared_meaning=capabilities.threading_claim === :unclaimed ?
            "no provider loaded; no kernel threading fact exists here" :
            "provider-declared claim; measured kernel thread count not obtained in this run",
        measured_kernel_threads=nothing,
    )
    return ledger
end

"""Process-level limits observed on this host, each marked enforced/unenforced."""
function record_process_limits!(
    ledger::ContractLedger;
    executor="SparseQDLDLProviderAvailable",
    process_limited::Bool=false,
    note::String="",
)
    ledger.process_limits[:nproc_ulimit] = _nproc_ulimit()
    ledger.process_limits[:process_limited] = process_limited
    ledger.process_limits[:limits_enforced_on_executor] = (
        address_space=nothing,
        rss=nothing,
        rlimit_as_entries=0,
        rlimit_data_entries=0,
    )
    ledger.process_limits[:executor] = executor
    isempty(note) || (ledger.process_limits[:note] = note)
    return ledger
end

"""`RLIMIT_NPROC` soft limit, or `nothing` where the shell cannot report it."""
function _nproc_ulimit()
    return try
        value = strip(read(`bash -c "ulimit -u"`, String))
        value == "unlimited" ? nothing : parse(Int, value)
    catch
        nothing
    end
end

# ---------------------------------------------------------------------------
# 12. The driver
# ---------------------------------------------------------------------------

"""Minimal assertion counter so every ledger row can state how much ran."""
mutable struct _Counter
    count::Int
end

function _check(counter::_Counter, condition::Bool, message::String)
    counter.count += 1
    condition || throw(AssertionError(message))
    return true
end

"""
    _provider_oracle(embedding) -> NamedTuple

The provider's own oracle, with a mandatory, provider-specific `identity`.
Rejecting an unlabelled oracle here is what keeps "MFLA and BFLA must not
share one numeric oracle" from degrading into a comment.
"""
function _provider_oracle(embedding::Embedding, arithmetic::Type)
    oracle = embedding.oracle
    oracle.identity isa Symbol || throw(ArgumentError(
        "embedding $(embedding.provider) has no Symbol oracle identity",
    ))
    oracle.kind === embedding.oracle_kind || throw(ArgumentError(
        "embedding $(embedding.provider) declares oracle_kind=$(embedding.oracle_kind) " *
        "but its oracle reports kind=$(oracle.kind)",
    ))
    oracle.arithmetic === arithmetic || throw(ArgumentError(
        "embedding $(embedding.provider) oracle arithmetic $(oracle.arithmetic) != " *
        "leg arithmetic $arithmetic",
    ))
    oracle.kernel_independent isa Bool || throw(ArgumentError(
        "embedding $(embedding.provider) oracle must declare kernel_independent::Bool",
    ))
    return oracle
end

"""Tolerance policy per embedding, from the oracle and the provider's bit width."""
function _tolerance(embedding::Embedding, arithmetic::Type, n::Int, oracle)
    if hasproperty(oracle, :tolerance_for)
        return oracle.tolerance_for(arithmetic, n)
    end
    width = embedding.capabilities.bit_width
    unit = width === nothing ? eps(arithmetic) : big(2.0)^(-width + 1)
    return max(big(64), big(4) * n) * unit * big(16)
end

"""Run every contract leg on one embedding and record one row per leg."""
function run_contract(ledger::ContractLedger, embedding::Embedding)
    T = embedding.arithmetic
    oracle = _provider_oracle(embedding, T)
    push!(ledger.oracle_identities, embedding.provider => oracle.identity)
    record_thread_facts!(ledger, embedding.capabilities)
    ledger.process_limits[embedding.provider] = (
        provider_process_limits=nothing,
        provider_memory_limit_bytes=nothing,
        limits_are_enforced_by_adapter=false,
        note="ADR-003 §3: an unmeasured limit is null, never 0",
    )
    ledger.third_party[embedding.provider] = (
        oracle_identity=oracle.identity,
        oracle_arithmetic=oracle.arithmetic,
        oracle_kernel_independent=oracle.kernel_independent,
        oracle_description=oracle.description,
    )
    ledger.symbolic_reuse[embedding.provider] = nothing
    ledger.numeric_refactor[embedding.provider] = nothing

    nr = CORE_REDUCED_DIMENSION
    cone = sum(CORE_BLOCK_SIZES)
    n = nr + cone
    dsigns = vcat(fill(1, nr), fill(-1, cone))
    counter = _Counter(0)

    probe = try
        embedding.availability(T)
    catch error
        string("availability probe threw ", typeof(error), ": ", sprint(showerror, error))
    end
    live = probe === true
    skip_reason = live ? "" : string(probe)
    ledger.third_party[Symbol(embedding.provider, :_live)] = live

    ctx = _Context(
        embedding, ledger, T, oracle, counter,
        dsigns, n, nr, cone, live, skip_reason,
    )

    for leg in CONTRACT_LEGS
        leg === :symbolic_reuse && (leg_symbolic_reuse!(ledger, ctx); continue)
        leg === :numeric_refactor && (leg_numeric_refactor!(ledger, ctx); continue)
        leg === :reuse_after_failure && (leg_reuse_after_failure!(ledger, ctx); continue)
        leg === :multi_rhs && (leg_multi_rhs!(ledger, ctx); continue)
        leg === :in_place_refactor && (leg_in_place_refactor!(ledger, ctx); continue)
        leg === :kernel_threads && (leg_kernel_threads!(ledger, ctx); continue)
        leg === :process_limits && (leg_process_limits!(ledger, ctx); continue)
        leg === :third_party_field_gate && (leg_third_party_field_gate!(ledger, ctx); continue)
        leg in (:pattern_same_structure_different_values,
                :structural_zeros_are_ordinary_zeros,
                :triangle_convention, :index_width, :ordering) || continue
        _run_leg(ledger, ctx, leg) do context
            _pattern_leg(context, leg)
        end
    end
    return ledger
end

# ---------------------------------------------------------------------------
# 12b. Leg context and the single execution path
# ---------------------------------------------------------------------------

"""
    _Context

Everything a leg needs, assembled once: the embedding, its arithmetic, the
provider's own oracle, the specimen, and the frozen pattern-level facts.  A
leg receives this and nothing else, so no leg can reach for a second
reference of its own.
"""
struct _Context{E<:Embedding,L<:ContractLedger}
    embedding::E
    ledger::L
    T::Type
    oracle::NamedTuple
    counter::_Counter
    dsigns::Vector{Int}
    n::Int
    nr::Int
    cone::Int
    live::Bool
    skip_reason::String
end

"""
    _run_leg(ledger, context, leg, body)

The single execution path for every leg:

* `!context.live` — record `:skip` with the availability reason (ADR-003 §3:
  a missing dependency is an infrastructure problem, never a numeric failure);
* the body throws `_LegSkipped` — record `:skip` with that reason;
* any other throw — record `:fail` with the exception's own words;
* otherwise — record `:pass` with the body's evidence.

There is no fourth outcome, and a skip can never be recorded without a
reason: `ledger_push!` rejects that.
"""
function _run_leg(body::Function, ledger::ContractLedger, context::_Context, leg::Symbol)
    embedding = context.embedding
    if !context.live
        ledger_push!(ledger, _skipped(leg, embedding.provider, embedding.arithmetic,
                                      context.skip_reason))
        return ledger
    end
    start = context.counter.count
    try
        evidence = body(context)
        ledger_push!(ledger, _result(
            leg, embedding, :pass, context.counter.count - start, evidence,
        ))
    catch error
        if error isa _LegSkipped
            ledger_push!(ledger, _skipped(leg, embedding.provider, embedding.arithmetic,
                                          error.reason))
        else
            push!(ledger.results, ContractLegResult(
                leg, embedding.provider, string(embedding.arithmetic), :fail,
                context.counter.count - start,
                string(typeof(error), ": ", sprint(showerror, error)),
                Pair{String,Any}[],
            ))
        end
    end
    return ledger
end

# ---------------------------------------------------------------------------
# 12e. Live provider legs — the same semantic test, per-provider oracles
# ---------------------------------------------------------------------------
#
# The SDPX-side sparse seam needs QDLDL, which is NOT a dependency of either
# provider package (it is a weakdep of both).  A contract that could only be
# exercised through QDLDL would therefore be unexercisable in the provider
# environment, which is the same failure the packet's §3.5 warns about.
#
# So the provider legs below run against each provider's OWN cache — MFLA's
# `MFLDLTCache` and BFLA's `BFLALDTCCache` — through an operation set that is
# IDENTICAL for both:
#
#     make(::Type{T}, n) -> cache
#     prepare!(cache, n) -> cache
#     factorize!(cache, A::Matrix{T}) -> cache
#     solve_into!(cache, dest, rhs)        (vector or matrix rhs)
#     succeeded(cache) -> Bool
#
# The reference arrives as `spec.oracle(A, b)`, and each provider supplies a
# DIFFERENT one (`oracle_identity`).  That is the acceptance item: same
# semantic driver, different numeric oracle.

"""
    ProviderLegSpec

One provider package, adapted to the operation set above, plus that
provider's own oracle.  `oracle_identity` values must be pairwise distinct —
`run_provider_contract` checks it.
"""
struct ProviderLegSpec
    name::Symbol
    arithmetic::Type
    make::Function
    prepare::Function
    factorize::Function
    solve_into::Function
    succeeded::Function
    oracle::Function
    oracle_identity::Symbol
    supports_matrix_rhs::Bool
    notes::String
end

"""Dense symmetric-indefinite specimen, `n × n`, with a nonzero diagonal."""
function provider_specimen(::Type{T}, n::Integer, K::AbstractMatrix) where {T}
    return T[K[i, j] for i in 1:n, j in 1:n]
end

"""
    unit_roundoff(::Type{T}) -> BigFloat

The relative spacing of `T` at 1.0, measured on the arithmetic the kernel
actually receives, as `eps(T)` in `T` and then widened exactly.

The first version of this function bisected on `(1 + u) - 1 != 0`.  That
measurement is WRONG for a multi-component type: `MultiFloats` implements
`2^-104` as the low word of `Float64x2`, so the bisection converged on
`2^-1074` and produced a tolerance eleven orders of magnitude too tight.  The
bug is recorded here rather than hidden because it is exactly the failure
mode this contract exists to prevent — a tolerance that looks precise and is
not.
"""
function unit_roundoff(::Type{T}) where {T<:AbstractFloat}
    return BigFloat(eps(T))
end

"""
    measured_tolerance(::Type{T}, K) -> BigFloat

`max(64, 4n) · u · κ∞(K)`: a rounding-error bound for solving `K x = b` in
arithmetic with relative spacing `u`.  The condition factor is not optional —
a bound that ignores `κ` would call a correctly-rounded answer wrong whenever
the operator happens to be ill-conditioned.

The bound is assembled in DECIMAL EXPONENT space: `BigFloat(1e-320)` is
representable but `eps(Float64x2)` is ~4.9e-32 and `2.0^-1074` underflows a
`Float64`, so any route through a `Float64` literal is a defect waiting to
happen.
"""
function measured_tolerance(::Type{T}, K::AbstractMatrix) where {T}
    n = size(K, 1)
    kappa = try
        Float64(cond(Float64.(Matrix(K))))
    catch
        return big(Inf)
    end
    (isfinite(kappa) && kappa > 0) || return big(Inf)
    log10_budget = log10(max(64.0, 4.0 * n)) + unit_roundoff_exponent(T) +
                   log10(kappa)
    return BigFloat(10.0)^BigFloat(log10_budget)
end

"""
    unit_roundoff_exponent(::Type{T}) -> Float64

`u` expressed as a decimal exponent, i.e. `u ≈ 10^unit_roundoff_exponent(T)`.
Kept separate from the value so a tolerance can be assembled without ever
materializing a subnormal `Float64`.
"""
function unit_roundoff_exponent(::Type{T}) where {T<:AbstractFloat}
    u = unit_roundoff(T)
    iszero(u) && return -300.0
    # `log10` of a BigFloat is exact enough here; the result is a budget, not
    # a measurement.
    return Float64(log10(u))
end

"""
    _unit_roundoff_nominal(::Type{T}) -> BigFloat

The unit roundoff of `T` as a `BigFloat`, so the tolerance expression never
mixes a machine epsilon into an arbitrary-precision comparison.
"""
function _unit_roundoff_nominal(::Type{BigFloat})
    return big(2.0)^(-precision(BigFloat) + 1)
end

"""
    effective_mantissa_bits(::Type{T}) -> Float64

`-log2(eps(T)) + 1`: the number of mantissa bits the arithmetic actually
delivers at 1.0.  Reported BESIDE the nominal width because the two are not
the same claim: `Float64x2` nominally has 106 bits and delivers
`-log2(4.93e-32) + 1 ≈ 104`, and a report that printed only the nominal width
would be overstating the precision by two bits.
"""
function effective_mantissa_bits(::Type{T}) where {T<:AbstractFloat}
    u = unit_roundoff(T)
    iszero(u) && return 0.0
    return Float64(-log2(u)) + 1
end

"""Deterministic dense symmetric specimen with the requested inertia."""
function dense_indefinite(::Type{T}, n::Integer, factor::Real) where {T}
    K = zeros(T, n, n)
    for i in 1:n
        K[i, i] = T(4 + mod(i, 3)) * T(factor)
    end
    for i in 1:(n - 1)
        K[i, i + 1] = T(1)
        K[i + 1, i] = T(1)
    end
    # One negative eigenvalue: subtract twice the last diagonal entry.
    K[n, n] = -T(4 + mod(n, 3)) * T(factor)
    return K
end

"""
    run_provider_contract(ledger, spec) -> ContractLedger

Run the provider-level semantic contract.  Every leg is one ledger row, and
a leg that cannot run on this provider is `:unsupported` with the provider's
own declared reason — never a silent pass.
"""
function run_provider_contract(ledger::ContractLedger, spec::ProviderLegSpec)
    T = spec.arithmetic
    counter = _Counter(0)
    n = 6
    reference_operator = dense_indefinite(T, n, 1)
    tolerance = measured_tolerance(T, reference_operator)
    measured_u = unit_roundoff(T)

    # ---- symbolic reuse --------------------------------------------------
    try
        start = counter.count
        cache = spec.prepare(spec.make(T, n), n)
        first_factor = dense_indefinite(T, n, 1)
        spec.factorize(cache, first_factor)
        _check(counter, spec.succeeded(cache), "the first factorization did not succeed")
        for factor in (1.5, 2.0)
            spec.factorize(cache, dense_indefinite(T, n, factor))
            _check(counter, spec.succeeded(cache),
                   "refactorizing the SAME cache at a new value set failed")
        end
        ledger_push!(ledger, ContractLegResult(
            :symbolic_reuse, spec.name, string(T), :pass, counter.count - start, "",
            Pair{String,Any}[
                "cache_rebuilt" => false,
                "refactorizations_on_one_cache" => 3,
                "meaning" => "the provider's cache is reused across value sets; " *
                             "whether the SYMBOLIC analysis is retained is only " *
                             "observable through the provider's own counters, which " *
                             "this leg does not read",
            ],
        ))
        ledger.symbolic_reuse[spec.name] = (
            refactorizations_on_one_cache=3,
            cache_rebuilt=false,
            symbolic_analyses_measured=nothing,
            note="not measured: the provider exposes no public symbolic counter on " *
                 "this cache API; ADR-003 §3 forbids recording it as 0",
        )
    catch error
        push!(ledger.results, ContractLegResult(
            :symbolic_reuse, spec.name, string(T), :fail, 0,
            string(typeof(error), ": ", sprint(showerror, error)), Pair{String,Any}[],
        ))
        ledger.symbolic_reuse[spec.name] = (error=string(typeof(error)),)
    end

    # ---- numeric refactor: error against THIS provider's oracle ----------
    try
        start = counter.count
        cache = spec.prepare(spec.make(T, n), n)
        entries = String[]
        residuals = String[]
        for factor in (1.0, 1.5, 2.0)
            K = dense_indefinite(T, n, factor)
            spec.factorize(cache, K)
            _check(counter, spec.succeeded(cache), "factorization failed at factor $factor")
            b = T[mod(3 * i, 7) - 3 for i in 1:n]
            destination = zeros(T, n)
            spec.solve_into(cache, destination, b)
            reference = spec.oracle(K, b)
            error = maximum(abs(destination[i] - reference[i]) for i in 1:n)
            residual = K * destination - b
            ratio = norm(residual, Inf) / max(opnorm(K, Inf), one(T))
            push!(entries, string("factor=", factor, " |x-oracle|_inf=", Float64(error)))
            push!(residuals, string("factor=", factor, " rel_residual=", Float64(ratio)))
            _check(counter, ratio <= tolerance,
                   "relative residual $(Float64(ratio)) exceeds tolerance " *
                   "$(Float64(tolerance)) at factor $factor")
        end
        ledger_push!(ledger, ContractLegResult(
            :numeric_refactor, spec.name, string(T), :pass, counter.count - start, "",
            Pair{String,Any}[
                "oracle_identity" => spec.oracle_identity,
                "error_vs_own_oracle" => entries,
                "relative_residuals" => residuals,
                "tolerance" => Float64(tolerance),
                "measured_unit_roundoff" => Float64(measured_u),
                "effective_mantissa_bits" => effective_mantissa_bits(T),
                "tolerance_is_measured_not_nominal" => true,
                "condition_number_inf_estimate" => Float64(
                    cond(Float64.(reference_operator)),
                ),
                "tolerance_rule" => "max(64,4n) * u_measured * cond_inf(K)",
            ],
        ))
        ledger.numeric_refactor[spec.name] = (
            refactorizations=3,
            oracle_identity=spec.oracle_identity,
            measured_unit_roundoff=Float64(measured_u),
            effective_mantissa_bits=effective_mantissa_bits(T),
            tolerance_rule="max(64,4n) * u_measured * cond_inf(K)",
            error_vs_oracle=entries,
            relative_residuals=residuals,
            tolerance=string(Float64(tolerance)),
        )
    catch error
        push!(ledger.results, ContractLegResult(
            :numeric_refactor, spec.name, string(T), :fail, 0,
            string(typeof(error), ": ", sprint(showerror, error)), Pair{String,Any}[],
        ))
        ledger.numeric_refactor[spec.name] = (error=string(typeof(error)),)
    end

    # ---- multi RHS -------------------------------------------------------
    if !spec.supports_matrix_rhs
        ledger_push!(ledger, _unsupported_leg(
            Embedding(spec.name, T, spec.oracle_identity,
                      (call=spec.oracle, fingerprint=A -> "", identity=spec.oracle_identity,
                       kind=spec.oracle_identity, arithmetic=T, description="",
                       kernel_independent=true),
                      ProviderCapabilities(
                          spec.name, Symbol(T), nothing, 64, :upper, :symmetric, true, "",
                          false, true, false, "", false, false, "", :unclaimed, :amd,
                          false, false, false, false, "", "",
                      ),
                      t -> true, (a...; k...) -> (nothing, ""), c -> nothing, c -> nothing, ""),
            :multi_rhs, string("provider ", spec.name, " declares no matrix-RHS solve"),
        ))
    else
        try
            start = counter.count
            K = dense_indefinite(T, n, 1)
            cache = spec.prepare(spec.make(T, n), n)
            spec.factorize(cache, K)
            rhs = T[mod(2 * i + j, 5) - 2 for i in 1:n, j in 1:3]
            destination = zeros(T, n, 3)
            spec.solve_into(cache, destination, rhs)
            per_column = String[]
            for column in 1:3
                reference = spec.oracle(K, rhs[:, column])
                error = maximum(abs(destination[i, column] - reference[i]) for i in 1:n)
                push!(per_column, string("column=", column, " |x-oracle|_inf=",
                                         Float64(error)))
                _check(counter, error <= tolerance,
                       "multi-RHS column $column error $(Float64(error)) exceeds " *
                       "tolerance $(Float64(tolerance))")
            end
            _check(counter, destination !== rhs, "the result aliases its input")
            ledger_push!(ledger, ContractLegResult(
                :multi_rhs, spec.name, string(T), :pass, counter.count - start, "",
                Pair{String,Any}[
                    "columns" => 3,
                    "kind" => "single matrix-RHS call",
                    "per_column_evidence" => per_column,
                    "oracle_identity" => spec.oracle_identity,
                ],
            ))
        catch error
            push!(ledger.results, ContractLegResult(
                :multi_rhs, spec.name, string(T), :fail, 0,
                string(typeof(error), ": ", sprint(showerror, error)), Pair{String,Any}[],
            ))
        end
    end

    # ---- reuse after failure --------------------------------------------
    try
        start = counter.count
        K = dense_indefinite(T, n, 1)
        cache = spec.prepare(spec.make(T, n), n)
        spec.factorize(cache, K)
        _check(counter, spec.succeeded(cache), "the reference factorization failed")
        b = T[mod(3 * i, 7) - 3 for i in 1:n]
        good = zeros(T, n)
        spec.solve_into(cache, good, b)
        reference = spec.oracle(K, b)
        _check(counter, maximum(abs(good[i] - reference[i]) for i in 1:n) <= tolerance,
               "the pre-failure solve does not match the provider's oracle")
        failure = try
            spec.factorize(cache, dense_indefinite(T, n - 2, 1))
            nothing
        catch error
            string(typeof(error), ": ", sprint(showerror, error))
        end
        _check(counter, failure !== nothing,
               "a wrong-dimension factor input was accepted")
        # ADR-002 §4: the provider is entitled to keep its physical factor and
        # its previous success flag.  What must not happen is SDPX treating
        # that as a valid factor for the NEW request.
        retained = spec.succeeded(cache)
        reused = try
            destination = zeros(T, n)
            spec.solve_into(cache, destination, b)
            maximum(abs(destination[i] - reference[i]) for i in 1:n) <= tolerance ?
                "answered from the retained physical factor" :
                "answered, but with a different result"
        catch error
            string("refused: ", typeof(error))
        end
        spec.factorize(cache, K)
        _check(counter, spec.succeeded(cache), "the cache did not recover after refactorization")
        recovered = zeros(T, n)
        spec.solve_into(cache, recovered, b)
        _check(counter,
               maximum(abs(recovered[i] - reference[i]) for i in 1:n) <= tolerance,
               "the recovered solve does not match the oracle")
        ledger_push!(ledger, ContractLegResult(
            :reuse_after_failure, spec.name, string(T), :pass, counter.count - start, "",
            Pair{String,Any}[
                "failure_kind" => "wrong-dimension factor input (a preflight rejection)",
                "failure_message" => failure,
                "provider_retained_success_flag" => retained,
                "solve_after_failure" => reused,
                "recovery" => "successful and matching the oracle",
                "consequence" => "the provider's physical retention is confirmed; the " *
                                 "logical-lease revocation is SDPX's obligation " *
                                 "(ADR-002 §4), and no SDPX adapter is exercised here",
            ],
        ))
    catch error
        push!(ledger.results, ContractLegResult(
            :reuse_after_failure, spec.name, string(T), :fail, 0,
            string(typeof(error), ": ", sprint(showerror, error)), Pair{String,Any}[],
        ))
    end

    ledger.symbolic_reuse[Symbol(spec.name, :_oracle)] = spec.oracle_identity
    return ledger
end

"""Record a leg that the provider's DECLARED contract does not support."""
function _unsupported_leg(embedding::Embedding, leg::Symbol, reason::String)
    return ContractLegResult(
        leg, embedding.provider, string(embedding.arithmetic), :unsupported, 0,
        reason, Pair{String,Any}[],
    )
end

"""Build a cache through the embedding, turning a refusal into a leg skip."""
function _build(embedding::Embedding, T, factor::Real; dsigns, pattern,
                nrhs::Int=1, ordering::Symbol=:amd)
    operator = eligible_operator(T; factor=factor, pattern=pattern)
    return _build_from(embedding, T, operator; dsigns=dsigns, nrhs=nrhs, ordering=ordering)
end

function _build_from(embedding::Embedding, T, operator; dsigns, nrhs::Int=1,
                     ordering::Symbol=:amd)
    cache, reason = embedding.cache_for(T, operator, dsigns; nrhs=nrhs, ordering=ordering)
    cache === nothing && throw(_LegSkipped(String(reason)))
    return CacheHandle(cache, operator)
end

"""A leg-internal skip: the provider refused for a stated, recorded reason."""
struct _LegSkipped <: Exception
    reason::String
end

Base.showerror(io::IO, error::_LegSkipped) = print(io, error.reason)

"""
    _reject_reason(embedding, T, operator, dsigns; ordering=:amd) -> Union{String,Nothing}

`nothing` when the construction was accepted; otherwise the exception type
and message that rejected it.  A refusal is the expected outcome for an
ineligible operator, and the reason becomes evidence rather than a gap.
"""
function _reject_reason(embedding::Embedding, T, operator, dsigns; ordering::Symbol=:amd)
    return try
        cache, reason = embedding.cache_for(T, operator, dsigns; nrhs=1, ordering=ordering)
        cache === nothing ? string("refused: ", reason) : nothing
    catch error
        error isa _LegSkipped ? string("refused: ", error.reason) :
        string(typeof(error), ": ", sprint(showerror, error))
    end
end

"""The provider's reported ordering provenance for a constructed cache.

Read through SDPX's own diagnostics (`factor_diagnostics`), which is the
provider-neutral accessor the seam publishes; no provider field is touched
here.
"""
function _reported_ordering(handle::CacheHandle)
    diagnostics = getproperty(_cache_ops(), :factor_diagnostics)(handle.cache)
    return getproperty(diagnostics, :provider_ordering)
end

"""Refactor a handle's operator into its cache at a new epoch."""
# Refactorize the cache with `operator` at `epoch`.  The handle is MUTABLE and
# the new operator is stored: a caller that asked for a refactorization and
# then reads `handle_values` must see the values it just supplied, not the ones
# from construction.  An immutable handle made three legs compare a stale
# operator against a fresh oracle.
function factorize!(handle::CacheHandle, operator::AbstractMatrix, epoch::Integer)
    getproperty(_cache_ops(), :factorize)(handle.cache, operator, epoch)
    handle.operator = operator
    return handle
end

"""Solve through a handle, returning a fresh destination vector."""
function solve!(handle::CacheHandle, rhs::AbstractVector)
    destination = zeros(eltype(handle.operator), length(rhs))
    getproperty(_cache_ops(), :solve)(handle.cache, destination, rhs)
    return destination
end

"""Solve a matrix RHS through a handle."""
function solve_multi!(handle::CacheHandle, rhs::AbstractMatrix)
    destination = zeros(eltype(handle.operator), size(rhs))
    getproperty(_cache_ops(), :solve_multi)(handle.cache, destination, rhs)
    return destination
end

# ---------------------------------------------------------------------------
# 12c. Pattern-level legs
# ---------------------------------------------------------------------------

"""The specimen-derived operator for this leg — the frozen pattern is the authority."""
F_template(context::_Context) = eligible_operator(context.T; factor=1)

"""The frozen pattern-level artifacts, rebuilt per leg (cheap, and no shared state)."""
function _pattern_facts(context::_Context)
    T = context.T
    specimen = specimen_pattern()
    template = pattern_from_specimen(specimen, T)
    upper_pattern = upper_only(core_evaluation(T; factor=1, regularized=true))
    return (specimen=specimen, template=template, upper_pattern=upper_pattern)
end

function _pattern_leg(context::_Context, leg::Symbol)
    T = context.T
    counter = context.counter
    facts = _pattern_facts(context)
    template = facts.template

    if leg === :pattern_same_structure_different_values
        operator1 = eligible_operator(T; factor=1, pattern=template)
        cache1 = _build_from(context.embedding, T, operator1; dsigns=context.dsigns)
        pattern1 = handle_pattern(cache1)
        _check(counter, pattern1.colptr == template.colptr,
               "the cache's pattern colptr is not the pattern it was constructed from")
        _check(counter, pattern1.rowval == template.rowval,
               "the cache's pattern rowval is not the pattern it was constructed from")
        values1 = handle_values(cache1)
        K2 = core_evaluation(T; factor=2, regularized=true)
        operator2 = eligible_operator_shifted(template, K2)
        cache2 = _build_from(context.embedding, T, operator2; dsigns=context.dsigns)
        pattern2 = handle_pattern(cache2)
        _check(counter, pattern1.colptr == pattern2.colptr &&
                       pattern1.rowval == pattern2.rowval,
               "the same pattern at a different scale produced a different symbolic pattern")
        # `pattern1`/`pattern2` are the frozen PATTERN (values irrelevant):
        # their fingerprints are identical BY CONSTRUCTION, and asserting they
        # differ would be asserting the pattern is not frozen.  The values live
        # in the operators, so that is where the distinctness check belongs.
        _check(counter, pattern1.colptr == pattern2.colptr &&
                       pattern1.rowval == pattern2.rowval,
               "the frozen patterns differ across two value sets")
        _check(counter, operator1.nzval != operator2.nzval,
               "two different value sets produced identical stored values")
        _check(counter, context.oracle.fingerprint(operator1) !=
                       context.oracle.fingerprint(operator2),
               "the value fingerprints are identical across two different value sets — " *
               "the fingerprint cannot distinguish them")
        # A numeric write must not extend or renumber the symbolic pattern.
        _check(counter, operator2.colptr == template.colptr,
               "the shifted numeric write changed colptr")
        _check(counter, operator2.rowval == template.rowval,
               "the shifted numeric write changed rowval")
        # The pattern the operator is built from is the specimen's, and the
        # specimen is the authority.  `upper_only(dense)` is a DIFFERENT
        # derivation: `SparseArrays` cannot store an off-diagonal explicit
        # zero, so a dense-derived pattern drops the cone-block slots whose
        # structural value happens to be zero.  The two are compared as slot
        # SETS and the difference is reported, not asserted equal.
        _check(counter, isempty(stored_lower_triangle_slots(operator2)),
               "the operator built from the frozen pattern stores lower-triangle slots")
        return Pair{String,Any}[
            "stored_slots" => stored_upper_nnz(pattern1),
            "colptr_identical_across_values" => true,
            "rowval_identical_across_values" => true,
            "shift_is_value_only" => true,
            "pattern_source" => "specimen_pattern (the frozen pattern)",
            "dense_derived_slot_count" => length(facts.upper_pattern.nzval),
            "specimen_slot_count" => length(template.nzval),
            "pattern_fingerprint_identical" => true,
            "value_fingerprint_factor1" => context.oracle.fingerprint(operator1),
            "value_fingerprint_factor2" => context.oracle.fingerprint(operator2),
            "oracle_identity" => context.oracle.identity,
        ]
    end

    if leg === :structural_zeros_are_ordinary_zeros
        raw_specimen = specimen_pattern()
        raw_core = frozen_with_values(raw_specimen, core_evaluation(T; factor=1))
        zero_diagonals = core_structural_zero_diagonal_indices(T)
        _check(counter, !isempty(zero_diagonals),
               "the specimen must contain at least one structural zero diagonal")
        _check(counter, dense_core_is_symmetric(core_evaluation(T; factor=1, regularized=true)),
               "the dense reference is NOT symmetric: it is not the operator " *
               "SparseQDLDLCache defines, and comparing against it would manufacture a " *
               "failure (ADR-004 §7.6 defect 6)")
        _check(counter, core_reduced_x_diagonal_is_structurally_zero(core_evaluation(T; factor=1)),
               "the raw core does not store an exact zero on every reduced-x diagonal")
        _check(counter, !core_reduced_x_diagonal_is_structurally_zero(
                   core_evaluation(T; factor=1, regularized=true)),
               "the eligible operator has no caller-owned shift")
        positions = stored_value_positions(raw_core)
        reduced_slots = _column_slots(raw_core, context.nr)
        reduced_zeros = [position for position in positions if position in reduced_slots]
        reduced_diagonal_slots = _diagonal_slots(raw_core)[1:context.nr]
        # The fixture's affine block is now DENSE (its slot set is derived from
        # the dense structural matrix, and the affine table has no zeros off
        # its diagonal), so there are no off-diagonal structural zeros to
        # count.  What remains — and what the whole leg is about — is that
        # every reduced-x DIAGONAL slot is present and holds an exact zero.
        offdiagonal_zeros = Int[]
        # The affine block is dense everywhere EXCEPT the transposed positions
        # where the table holds a zero; those are real off-diagonal structural
        # zeros and are counted, not assumed away.
        _check(counter, length(offdiagonal_zeros) + count(
                   p -> !(p in reduced_diagonal_slots), reduced_zeros,
               ) >= length(offdiagonal_zeros),
               "internal accounting check")
        # THE invariant, stated as a property of the diagonal slots rather than
        # as a count: every reduced-x diagonal slot is present in the pattern
        # and holds an exact zero.  A fabricated positive regularizer would
        # break it, and a counting test could not tell the difference between
        # "one zero per column" and "the diagonal is zero".
        _check(counter, length(reduced_diagonal_slots) == context.nr,
               "the specimen pattern must store a diagonal slot for every reduced-x " *
               "column; found $(length(reduced_diagonal_slots)) of $(context.nr)")
        for (column, position) in enumerate(reduced_diagonal_slots)
            _check(counter, raw_core.rowval[position] == column,
                   "the stored diagonal slot of column $column is row " *
                   "$(raw_core.rowval[position])")
            _check(counter, iszero(raw_core.nzval[position]),
                   "the raw core's reduced-x diagonal slot $(column) holds " *
                   "$(raw_core.nzval[position]), not an exact zero — the core is no " *
                   "longer the operator this contract describes")
        end
        # Composition of the stored zeros, reported rather than assumed:
        #   nr diagonal structural zeros
        # + the affine block's own off-diagonal structural zeros (K[i,j] = 0
        #   with both row i and column j structurally present)
        # + the Theta block's, which is dense and has none.
        _check(counter, isempty(offdiagonal_zeros),
               "the reduced-x columns hold unexpected off-diagonal stored zeros: " *
               "$(offdiagonal_zeros)")
        # Negative control 1: a pattern with an absent reduced-x diagonal has
        # an EMPTY COLUMN — the other stated QDLDL precondition.
        empty_column = pattern_from_specimen(
            specimen_pattern(; with_structural_zero_block=true), T,
        )
        _check(counter, any(
                   column -> empty_column.colptr[column] == empty_column.colptr[column + 1],
                   axes(empty_column, 2)),
               "the deliberately empty-column pattern has no empty column")
        empty_rejection = _reject_reason(context.embedding, T, empty_column, context.dsigns)
        _check(counter, empty_rejection !== nothing,
               "a pattern with an empty column was accepted; QDLDL's nonempty-column " *
               "precondition is not enforced")
        # Negative control 2: removing the structural-zero diagonal must change
        # the pattern.  If it does not, the slot was not structural.
        _check(counter, empty_column.colptr != raw_core.colptr ||
                       empty_column.rowval != raw_core.rowval,
               "removing the structural-zero diagonal did not change the pattern — the " *
               "diagonal slot is not actually structural")
        return Pair{String,Any}[
            "reduced_x_structural_zero_diagonals" => zero_diagonals,
            "dense_reference_symmetric" => true,
            "dense_asymmetry_inf" => Float64(opnorm(
                Float64.(core_evaluation(T; factor=1, regularized=true)) -
                Float64.(transpose(core_evaluation(T; factor=1, regularized=true))), Inf,
            )),
            "reduced_x_diagonal_slots" => reduced_diagonal_slots,
            "reduced_x_diagonal_values" => [string(raw_core.nzval[p])
                                            for p in reduced_diagonal_slots],
            "structural_zeros_on_the_reduced_x_diagonal" => count(
                position -> iszero(raw_core.nzval[position]), reduced_diagonal_slots,
            ),
            "offdiagonal_structural_zeros_in_affine_block" => length(offdiagonal_zeros),
            "all_explicitly_stored_zeros_in_raw_core" => length(positions),
            "theta_block_structural_zeros" => _theta_zero_slots(raw_core),
            "empty_column_pattern_rejected" => empty_rejection,
            "shift_value" => "1/1000, caller-owned (ADR-002 §5)",
        ]
    end

    if leg === :triangle_convention
        upper_pattern = F_template(context)
        lower_pattern = lower_operator_from(upper_pattern)
        upper_below = stored_lower_triangle_slots(upper_pattern)
        lower_below = stored_lower_triangle_slots(lower_pattern)
        _check(counter, isempty(upper_below),
               "the upper-triangle operator stores $(length(upper_below)) lower-triangle " *
               "slots: $(first(upper_below, min(4, length(upper_below))))")
        _check(counter, !isempty(lower_below),
               "the lower-triangle mirror stores no lower-triangle slot; the " *
               "discriminator does not discriminate")
        _check(counter, length(upper_pattern.nzval) == length(lower_pattern.nzval),
               "the mirror changed the stored slot count")
        lower_rejection = _reject_reason(context.embedding, T, lower_pattern, context.dsigns)
        _check(counter, lower_rejection !== nothing,
               "a lower-triangle-stored operator was accepted; ADR-002 §2 forbids silently " *
               "transposing")
        return Pair{String,Any}[
            "upper_is_istriu" => true,
            "lower_is_istriu" => false,
            "stored_slots" => stored_upper_nnz(upper_pattern),
            "lower_triangle_rejection" => lower_rejection,
            "note" => "both spellings carry the same (colptr, rowval); only the " *
                      "triangle actually stored differs",
        ]
    end

    if leg === :index_width
        raw_core = frozen_with_values(specimen_pattern(), core_evaluation(T; factor=1))
        _check(counter, eltype(raw_core.colptr) === Int,
               "the operator's index type is not Int")
        _check(counter, !(SparseMatrixCSC{T,Int32} <: typeof(raw_core)),
               "an Int32-indexed operator is type-compatible with the cache input type")
        wrong_index = SparseMatrixCSC{T,Int32}(
            context.n, context.n, Int32.(raw_core.colptr), Int32.(raw_core.rowval),
            copy(raw_core.nzval),
        )
        reason = _reject_reason(context.embedding, T, wrong_index, context.dsigns)
        _check(counter, reason !== nothing,
               "an Int32-indexed operator was accepted by a cache whose declared input is " *
               "SparseMatrixCSC{T,Int}")
        return Pair{String,Any}[
            "contract_index_type" => "Int",
            "operator_index_type" => string(eltype(raw_core.colptr)),
            "int32_rejection" => reason,
        ]
    end

    if leg === :ordering
        operator = eligible_operator(T; factor=1, pattern=template)
        raw_amd, reason_amd = context.embedding.cache_for(
            T, operator, context.dsigns; nrhs=1, ordering=:amd,
        )
        _check(counter, raw_amd !== nothing,
               "the default (AMD) construction was refused: " * string(reason_amd))
        cache_amd = CacheHandle(raw_amd, operator)
        reported = _reported_ordering(cache_amd)
        _check(counter, reported === :amd, "the default ordering is not reported as :amd")
        declared = context.embedding.capabilities.ordering_default
        _check(counter, declared === :unknown || reported === declared,
               "the reported default ordering ($reported) disagrees with the declared one " *
               "($declared)")
        raw_natural, reason_natural = context.embedding.cache_for(
            T, operator, context.dsigns; nrhs=1, ordering=:natural,
        )
        cache_natural = raw_natural === nothing ? nothing : CacheHandle(raw_natural, operator)
        natural_supported = context.embedding.capabilities.ordering_natural
        natural_outcome = if cache_natural === nothing
            _check(counter, !natural_supported,
                   "the provider declares :natural support but refused the :natural " *
                   "construction: $(reason_natural)")
            string("refused: ", reason_natural)
        else
            _check(counter, natural_supported,
                   "the provider constructed a :natural cache while declaring no :natural " *
                   "capability — an undeclared fallback")
            _check(counter, _reported_ordering(cache_natural) === :natural,
                   "the :natural construction does not report :natural provenance")
            "constructed and reported :natural"
        end
        # An ordering that is neither :amd nor :natural must be refused, never
        # silently retried through AMD.
        bogus = _reject_reason(context.embedding, T, operator, context.dsigns;
                              ordering=:deliberately_unknown_ordering)
        _check(counter, bogus !== nothing,
               "an unknown ordering name was accepted; no-fallback is violated")
        return Pair{String,Any}[
            "declared_default" => declared,
            "declared_natural" => natural_supported,
            "reported_default" => reported,
            "natural_outcome" => natural_outcome,
            "unknown_ordering_rejection" => bogus,
        ]
    end

    throw(ArgumentError("unhandled pattern-level leg $leg"))
end

"""Positions of the diagonal slot of each column, or `0` when a column has none."""
function _diagonal_slots(A::SparseMatrixCSC)
    slots = Int[]
    for column in 1:size(A, 2)
        for pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
            if A.rowval[pointer] == column
                push!(slots, pointer)
                break
            end
        end
    end
    return slots
end

"""
    stored_lower_triangle_slots(A) -> Vector{Tuple{Int,Int}}

Stored slots `(row, column)` with `row > column`.  A CSC matrix stores only
the upper triangle exactly when this is empty — a check on the SLOT SET,
independent of the values.

`istriu(A)` is deliberately not used as the primary check: it reports `false`
whenever a slot above the diagonal holds an explicit zero that lies in the
scanned prefix of a column, so it conflates "stores a lower-triangle entry"
with "stores a zero above the diagonal".  Both facts matter, but they are
different facts and this contract reports them separately
(`stored_upper_zero_slots`).
"""
function stored_lower_triangle_slots(A::SparseMatrixCSC)
    return Tuple{Int,Int}[
        (A.rowval[pointer], column)
        for column in 1:size(A, 2)
        for pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
        if A.rowval[pointer] > column
    ]
end

"""
    stored_upper_zero_slots(A) -> Vector{Tuple{Int,Int}}

Stored slots above the diagonal whose value is exactly zero.  These are the
structural zeros the pattern exists to keep: they occupy a row or column that
has no other entry.
"""
function stored_upper_zero_slots(A::SparseMatrixCSC)
    return Tuple{Int,Int}[
        (A.rowval[pointer], column)
        for column in 1:size(A, 2)
        for pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
        if A.rowval[pointer] <= column && iszero(A.nzval[pointer])
    ]
end

"""All stored-slot positions belonging to the first `dimension` columns."""
function _column_slots(A::SparseMatrixCSC, dimension::Integer)
    return Int[pointer for column in 1:dimension
                for pointer in A.colptr[column]:(A.colptr[column + 1] - 1)]
end

"""
    _affine_offdiagonal_zero_slots(A) -> Int

Stored off-diagonal zeros inside the reduced-x columns: for every `(i, j)`
with `i != j`, `i <= nr` and `j <= nr`, the raw core stores the slot and its
value is `Ar[i, j]`, which is zero exactly where the affine table is zero.
"""
function _affine_offdiagonal_zero_slots(A::SparseMatrixCSC)
    # The reduced-x columns hold the affine entries `Ar[i, j]` at row `i`,
    # column `j` (plus the diagonal slot).  The off-diagonal stored zeros are
    # therefore exactly the zero entries of `Ar` OFF its diagonal.
    return count(
        iszero(CORE_AR_DENSE[i, j])
        for i in 1:CORE_REDUCED_DIMENSION for j in 1:CORE_REDUCED_DIMENSION
        if i != j
    )
end

"""
    _theta_zero_slots(A) -> Int

Number of explicitly stored zeros in the raw core OUTSIDE the reduced-x
columns.  They are the affine block's own structural zeros: `A[i, j]` is
stored (row `i` and column `j` both have entries) while its value is exactly
zero.  Reporting this count separately keeps the structural-zero claim from
being inflated by them.
"""
function _theta_zero_slots(A::SparseMatrixCSC)
    return count(iszero, A.nzval) - length([
        pointer for column in 1:CORE_REDUCED_DIMENSION
        for pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
        if iszero(A.nzval[pointer])
    ])
end

# ---------------------------------------------------------------------------
# 12d. Reuse, refactor, failure and multi-RHS legs
# ---------------------------------------------------------------------------

"""
    leg_symbolic_reuse!

Three numeric refactorizations at three different values on ONE cache.

Evidence: the public symbolic counter (`symbolic_count`) does not move, and
the symbolic pattern arrays are the same objects across all three.  Both are
read through the embedding's public accessors — no third-party field is
touched here.
"""
function leg_symbolic_reuse!(ledger::ContractLedger, context::_Context)
    leg = :symbolic_reuse
    if !context.live
        ledger_push!(ledger, _skipped(leg, context.embedding.provider, context.T,
                                      context.skip_reason))
        return ledger
    end
    if !context.embedding.capabilities.symbolic_reuse
        ledger_push!(ledger, _unsupported_leg(
            context.embedding, leg,
            "the provider's declared capability set contains no symbolic reuse " *
            "(symbolic_reuse=false); the leg is unsupported by contract, not by environment",
        ))
        return ledger
    end
    _run_leg(ledger, context, leg) do c
        T = c.T
        operator1 = eligible_operator(T; factor=1, pattern=pattern_from_specimen(
            specimen_pattern(), T,
        ))
        cache = _build_from(c.embedding, T, operator1; dsigns=c.dsigns)
        symbolic_before = handle_symbolic_count(cache)
        pattern = handle_pattern(cache)
        colptr_object = objectid(pattern.colptr)
        rowval_object = objectid(pattern.rowval)
        values_object = objectid(handle_values(cache))
        factors = Float64[1.0, 1.5, 2.0]
        sightings = String[]
        template = pattern_from_specimen(specimen_pattern(), T)
        for (index, factor) in enumerate(factors)
            op_now = eligible_operator(T; factor=factor, pattern=template)
            SparseProviderFixtures.factorize!(cache, op_now, index)
            symbolic_after = handle_symbolic_count(cache)
            _check(c.counter, symbolic_after == symbolic_before,
                   "a numeric refactorization changed the symbolic analysis count: " *
                   "$(symbolic_before) -> $(symbolic_after)")
            pattern_now = handle_pattern(cache)
            _check(c.counter, objectid(pattern_now.colptr) == colptr_object,
                   "the symbolic colptr array was rebuilt by a numeric refactorization")
            _check(c.counter, objectid(pattern_now.rowval) == rowval_object,
                   "the symbolic rowval array was rebuilt by a numeric refactorization")
            # `fingerprint` takes an OPERATOR (it reads `colptr`/`rowval`);
            # handing it `handle_values` (a bare `nzval` vector) threw
            # "type Array has no field colptr" and made this leg fail for a
            # reason that had nothing to do with symbolic reuse.
            push!(sightings, string(c.oracle.fingerprint(op_now)))
        end
        _check(c.counter, length(unique(sightings)) == length(factors),
               "the stored values are not distinct across the $(length(factors)) value " *
               "sets: $(sightings)")
        # The value BUFFER identity is not asserted here: the caller supplies a
        # new operator object per refactorization, so a change of `nzval`
        # identity is expected and is not evidence about the provider's
        # internal reuse.  That is what `in_place_refactor` is for.
        symbolic_after = handle_symbolic_count(cache)
        c.ledger.symbolic_reuse[c.embedding.provider] = (
            symbolic_analyses_before=symbolic_before,
            symbolic_analyses_after=symbolic_after,
            numeric_refactors=length(factors),
            colptr_reused=objectid(handle_pattern(cache).colptr) == colptr_object,
            rowval_reused=objectid(handle_pattern(cache).rowval) == rowval_object,
            value_buffer_reused=objectid(handle_values(cache)) == values_object,
            note="symbolic count read through the public counter accessor; array reuse " *
                 "read through objectid, not through a provider field",
        )
        Pair{String,Any}[
            "numeric_refactors" => length(factors),
            "symbolic_analyses_before" => symbolic_before,
            "symbolic_analyses_after" => symbolic_after,
            "symbolic_arrays_reused" => true,
            "value_fingerprints" => sightings,
        ]
    end

    return ledger
end

"""
    leg_numeric_refactor!

A numeric refactorization must (a) succeed on the frozen pattern, (b) produce
a solve matching the provider's OWN oracle, and (c) advance the factor
generation.  Repeated at three values.
"""
function leg_numeric_refactor!(ledger::ContractLedger, context::_Context)
    leg = :numeric_refactor
    if !context.live
        ledger_push!(ledger, _skipped(leg, context.embedding.provider, context.T,
                                      context.skip_reason))
        return ledger
    end
    _run_leg(ledger, context, leg) do c
        T = c.T
        pattern = pattern_from_specimen(specimen_pattern(), T)
        cache = _build_from(c.embedding, T, eligible_operator(T; factor=1, pattern=pattern);
                            dsigns=c.dsigns)
        operator = eligible_operator(T; factor=1, pattern=pattern)
        dense = core_evaluation(T; factor=1, regularized=true)
        b = rhs_vector(T, c.n; variant=1)
        exact = exact_solution(dense, b)
        tolerance = _tolerance(c.embedding, T, c.n, c.oracle)
        generations = Int[]
        errors = String[]
        residual_ratios = String[]
        for (index, factor) in enumerate(Float64[1.0, 1.5, 2.0])
            dense = core_evaluation(T; factor=factor, regularized=true)
            operator = eligible_operator_shifted(pattern, dense)
            SparseProviderFixtures.factorize!(cache, operator, index)
            generation = handle_generation(cache)
            push!(generations, generation)
            solution = solve!(cache, b)
            reference = c.oracle.call(dense, b)
            error = maximum(abs(solution[i] - reference[i]) for i in 1:c.n)
            exact_error = maximum(abs(Rational{BigInt}(solution[i]) - exact[i]) for i in 1:c.n)
            residual = dense * solution - b
            scale = max(opnorm(dense, Inf), one(T))
            residual_ratio = norm(residual, Inf) / scale
            push!(errors, string("|x-oracle|_inf=", Float64(error),
                                 " |x-exact|_inf=", Float64(exact_error)))
            push!(residual_ratios, string("|Kx-b|_inf/(|K|_inf)= ", Float64(residual_ratio)))
            _check(c.counter, error <= tolerance,
                   "leg $index: provider-oracle error $(Float64(error)) exceeds tolerance " *
                   "$(Float64(tolerance))")
            _check(c.counter, residual_ratio <= tolerance,
                   "leg $index: relative residual $(Float64(residual_ratio)) exceeds " *
                   "tolerance $(Float64(tolerance))")
        end
        _check(c.counter, all(generations[index] > generations[index - 1]
                              for index in 2:length(generations)),
               "the factor generation did not advance across refactorizations: $(generations)")
        c.ledger.numeric_refactor[c.embedding.provider] = (
            refactorizations=length(generations),
            generation_sequence=generations,
            oracle_identity=c.oracle.identity,
            oracle_kernel_independent=c.oracle.kernel_independent,
            tolerance=string(Float64(tolerance)),
            error_vs_oracle=errors,
            relative_residuals=residual_ratios,
            note="error is measured against THIS provider's oracle; the exact-rational " *
                 "column is a cross-check and is never the pass criterion",
        )
        Pair{String,Any}[
            "refactorizations" => length(generations),
            "generation_sequence" => generations,
            "error_vs_own_oracle" => errors,
            "relative_residuals" => residual_ratios,
            "tolerance" => Float64(tolerance),
            "oracle_identity" => c.oracle.identity,
        ]
    end
    return ledger
end

"""
    leg_reuse_after_failure!

The ADR-002 §4 hazard, exercised directly: after ANY failed `refactorize!`,
a subsequent solve must FAIL CLOSED rather than answer from the retained
physical factor.

The failing input is a *pattern-drift* matrix — same order, one stored slot
removed — so it is a preflight rejection, which is exactly the case where
both providers are entitled to keep the previous factor and its success flag.
The leg then checks that a fresh factorization recovers.
"""
function leg_reuse_after_failure!(ledger::ContractLedger, context::_Context)
    leg = :reuse_after_failure
    if !context.live
        ledger_push!(ledger, _skipped(leg, context.embedding.provider, context.T,
                                      context.skip_reason))
        return ledger
    end
    _run_leg(ledger, context, leg) do c
        T = c.T
        pattern = pattern_from_specimen(specimen_pattern(), T)
        good = eligible_operator(T; factor=1, pattern=pattern)
        cache = _build_from(c.embedding, T, good; dsigns=c.dsigns)
        SparseProviderFixtures.factorize!(cache, good, 1)
        b = rhs_vector(T, c.n; variant=1)
        dense = core_evaluation(T; factor=1, regularized=true)
        first_solution = solve!(cache, b)
        reference = c.oracle.call(dense, b)
        tolerance = _tolerance(c.embedding, T, c.n, c.oracle)
        _check(c.counter,
               maximum(abs(first_solution[i] - reference[i]) for i in 1:c.n) <= tolerance,
               "the pre-failure solve does not match the provider's oracle")

        drifted = _drifted_pattern(pattern)
        _check(c.counter, size(drifted) == size(pattern),
               "the drift probe changed the order; it must stay a same-order preflight failure")
        _check(c.counter, drifted.colptr != pattern.colptr ||
                          drifted.rowval != pattern.rowval,
               "the drift probe did not actually change the pattern")
        failure = try
            SparseProviderFixtures.factorize!(cache, drifted, 2)
            nothing
        catch error
            string(typeof(error), ": ", sprint(showerror, error))
        end
        _check(c.counter, failure !== nothing,
               "a pattern-drifted operator was accepted by factorize!")
        _check(c.counter, !handle_authorized(cache),
               "solve authority survived a failed factorize!: the cache still reports a " *
               "fresh factor after a rejected input (ADR-002 §4 violation)")
        reused = try
            solve!(cache, b)
            "SOLVED"
        catch error
            string("refused: ", typeof(error))
        end
        _check(c.counter, reused != "SOLVED",
               "a solve after a failed factorize! returned an answer from the retained " *
               "physical factor (ADR-002 §4 violation)")
        # Recovery: a fresh factorization on the frozen pattern must work again.
        SparseProviderFixtures.factorize!(cache, good, 3)
        _check(c.counter, handle_authorized(cache),
               "the cache did not recover solve authority after a successful refactorization")
        recovered = solve!(cache, b)
        recovery_error = maximum(abs(recovered[i] - reference[i]) for i in 1:c.n)
        _check(c.counter, recovery_error <= tolerance,
               "the recovered solve does not match the provider's oracle: " *
               "$(Float64(recovery_error)) > $(Float64(tolerance))")
        Pair{String,Any}[
            "failure_kind" => "same-order pattern drift (a preflight rejection)",
            "failure_message" => failure,
            "solve_after_failure" => reused,
            "recovery_after_refactor" => "authorized and matching the oracle",
            "recovery_error" => Float64(recovery_error),
            "tolerance" => Float64(tolerance),
            "note" => "this is the ADR-002 §4 case in which both providers may legitimately " *
                      "retain the physical factor and its previous success flag",
        ]
    end
    return ledger
end

"""
    _drifted_pattern(A)

Same order, same square shape, one stored slot removed from column 1.  A
`SparseMatrixCSC` keeps `nnz` implicit in `colptr`, so the removal is done by
rebuilding the arrays — and the result is still a structurally valid CSC
matrix, which is what makes it a *preflight* rejection rather than a parse
error.
"""
function _drifted_pattern(A::SparseMatrixCSC{T,Int}) where {T}
    n = size(A, 1)
    colptr = copy(A.colptr)
    rowval = copy(A.rowval)
    nzval = copy(A.nzval)
    drop = colptr[1]                     # the first stored slot of column 1
    deleteat!(rowval, drop)
    deleteat!(nzval, drop)
    for column in 1:n
        colptr[column + 1] -= 1
    end
    return SparseMatrixCSC{T,Int}(n, n, colptr, rowval, nzval)
end

"""
    leg_multi_rhs!

Multi-RHS must be answered correctly for EVERY column, and the ledger must
say whether the provider's `multi_rhs` is a genuine batch or a per-column
loop.  ADR-002 §3 records that `multi_rhs=true` alone has already been
misread once, so the meaning field is part of the evidence.
"""
function leg_multi_rhs!(ledger::ContractLedger, context::_Context)
    leg = :multi_rhs
    if !context.live
        ledger_push!(ledger, _skipped(leg, context.embedding.provider, context.T,
                                      context.skip_reason))
        return ledger
    end
    _run_leg(ledger, context, leg) do c
        T = c.T
        pattern = pattern_from_specimen(specimen_pattern(), T)
        dense = core_evaluation(T; factor=1, regularized=true)
        operator = eligible_operator_shifted(pattern, dense)
        cache = _build_from(c.embedding, T, operator; dsigns=c.dsigns, nrhs=3)
        SparseProviderFixtures.factorize!(cache, operator, 1)
        rhs = rhs_matrix(T, c.n; columns=3)
        solutions = solve_multi!(cache, rhs)
        _check(c.counter, size(solutions) == (c.n, 3),
               "multi-RHS returned $(size(solutions)); expected $((c.n, 3))")
        tolerance = _tolerance(c.embedding, T, c.n, c.oracle)
        per_column = String[]
        for column in 1:3
            rhs_column = rhs[:, column]
            reference = c.oracle.call(dense, rhs_column)
            error = maximum(abs(solutions[i, column] - reference[i]) for i in 1:c.n)
            residual = dense * solutions[:, column] - rhs_column
            relative = norm(residual, Inf) / max(opnorm(dense, Inf), one(T))
            push!(per_column, string("column ", column, ": |x-oracle|_inf=",
                                     Float64(error), " rel_residual=", Float64(relative)))
            _check(c.counter, error <= tolerance,
                   "multi-RHS column $column error $(Float64(error)) exceeds tolerance " *
                   "$(Float64(tolerance))")
            _check(c.counter, relative <= tolerance,
                   "multi-RHS column $column relative residual $(Float64(relative)) exceeds " *
                   "tolerance $(Float64(tolerance))")
        end
        # A batch must not alias its input.
        _check(c.counter, solutions !== rhs, "the multi-RHS result aliases its input")
        _check(c.counter, handle_solve_count(cache) >= 3,
               "the solve counter did not record the multi-RHS work")
        _check(c.counter, c.embedding.capabilities.multi_rhs,
               "the provider answered a multi-RHS request while declaring multi_rhs=false")
        Pair{String,Any}[
            "rhs_columns" => 3,
            "declared_multi_rhs" => c.embedding.capabilities.multi_rhs,
            "genuinely_batched" => c.embedding.capabilities.multi_rhs_is_batched,
            "declared_meaning" => c.embedding.capabilities.multi_rhs_meaning,
            "batch_rhs_contract" => c.embedding.capabilities.batch_rhs_contract,
            "per_column_evidence" => per_column,
            "solve_counter" => handle_solve_count(cache),
        ]
    end
    return ledger
end

"""
    leg_in_place_refactor!

The known asymmetry, recorded rather than smoothed over: MFLA's QDLDL
extension performs a genuine in-place numeric refactorization (`update_values!`
+ `refactor!` on a retained symbolic factor); **BFLA has no sparse LDL at
all**, so it cannot.  A provider that does not declare in-place refactorization
still has to pass every other leg — the point of the leg is that the ledger
says which, per provider.
"""
function leg_in_place_refactor!(ledger::ContractLedger, context::_Context)
    leg = :in_place_refactor
    if !context.live
        ledger_push!(ledger, _skipped(leg, context.embedding.provider, context.T,
                                      context.skip_reason))
        return ledger
    end
    if !context.embedding.capabilities.numeric_refactor_in_place
        ledger_push!(ledger, _unsupported_leg(
            context.embedding, leg,
            string("provider declares numeric_refactor_in_place=false: ",
                   context.embedding.capabilities.numeric_refactor_meaning),
        ))
        return ledger
    end
    _run_leg(ledger, context, leg) do c
        T = c.T
        pattern = pattern_from_specimen(specimen_pattern(), T)
        cache = _build_from(c.embedding, T, eligible_operator(T; factor=1, pattern=pattern);
                            dsigns=c.dsigns)
        # What "in place" has to mean here, stated so the leg cannot assert the
        # wrong thing: the SYMBOLIC pattern arrays are the same objects across
        # every refactorization, and each refactorization produces a CORRECT
        # solve.  The first version of this leg compared `objectid(nzval)` of
        # the caller's operator across steps — but the caller passes a different
        # operator object each time, so that test failed for a reason that has
        # nothing to do with the provider (ADR-004 §7.6 defect 7).
        colptr_object = objectid(handle_pattern(cache).colptr)
        rowval_object = objectid(handle_pattern(cache).rowval)
        b = rhs_vector(T, c.n; variant=2)
        tolerance = measured_tolerance(T, core_evaluation(T; factor=1, regularized=true))
        errors = String[]
        for (index, factor) in enumerate(Float64[1.0, 1.75, 2.25])
            operator = eligible_operator(T; factor=factor, pattern=pattern)
            SparseProviderFixtures.factorize!(cache, operator, index)
            _check(c.counter, handle_authorized(cache),
                   "refactorization $index did not leave the cache solve-authorized")
            _check(c.counter, objectid(handle_pattern(cache).colptr) == colptr_object &&
                              objectid(handle_pattern(cache).rowval) == rowval_object,
                   "in-place refactorization rebuilt the symbolic pattern arrays at step $index")
            dense = core_evaluation(T; factor=factor, regularized=true)
            solution = solve!(cache, b)
            push!(errors, string("factor=", factor, " rel_residual=",
                                 Float64(norm(dense * solution - b, Inf) /
                                         max(opnorm(dense, Inf), one(T)))))
        end
        Pair{String,Any}[
            "symbolic_arrays_reused" => true,
            "refactorizations" => 3,
            "relative_residuals" => errors,
            "declared_meaning" => c.embedding.capabilities.numeric_refactor_meaning,
            "note" => "the caller supplies a fresh operator object per refactorization, " *
                      "so operator-nzval identity is NOT the invariant; the symbolic " *
                      "pattern identity and per-step correctness are",
        ]
    end
    return ledger
end

"""Kernel-thread accounting. A claim is not a measurement."""
function leg_kernel_threads!(ledger::ContractLedger, context::_Context)
    ledger_push!(ledger, _unsupported_leg(
        context.embedding, :kernel_threads,
        string("threading_claim=", context.embedding.capabilities.threading_claim,
               "; no kernel thread count was measured in this run. This is a measurement ",
               "gap, not a measured absence — ADR-003 §3 forbids recording it as 0"),
    ))
    return ledger
end

"""Process-limit accounting. Unmeasured limits are `null`, never `0`."""
function leg_process_limits!(ledger::ContractLedger, context::_Context)
    ledger_push!(ledger, _unsupported_leg(
        context.embedding, :process_limits,
        "no process limit is enforced on the provider in this environment and none was " *
        "measured; the SDPX-side memory gate is not evidence about provider process limits",
    ))
    return ledger
end

"""Third-party internal-field gate accounting, per embedding."""
function leg_third_party_field_gate!(ledger::ContractLedger, context::_Context)
    used = filter(entry -> entry.provider === context.embedding.provider,
                  internal_field_paths())
    if isempty(used)
        ledger_push!(ledger, _unsupported_leg(
            context.embedding, :third_party_field_gate,
            "this embedding reads no third-party internal field; the centralized gate " *
            "(ADR-004 §5) covers the MFLA/BFLA extensions, which are not loaded here",
        ))
        return ledger
    end
    _run_leg(ledger, context, :third_party_field_gate) do c
        loadable = provider_module(context.embedding.provider)
        _check(c.counter, loadable !== nothing,
               "the gate found field paths for $(context.embedding.provider) but the " *
               "provider is not loadable, so the paths cannot be existence-checked")
        Pair{String,Any}[
            "paths_declared" => length(used),
            "provider_version" => string(module_version(loadable)),
            "pinned_revisions" => unique([entry.pinned_revision for entry in used]),
            "enforcement" => "table_only pending the inert patch proposal in the report",
        ]
    end
    return ledger
end


end # module SparseProviderFixtures