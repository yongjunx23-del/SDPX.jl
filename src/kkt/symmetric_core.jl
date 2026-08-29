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


"""
    symmetric_core_lower_sparse(pattern) -> SparseMatrixCSC{T,Int}

Return a lower-triangle CSC matrix that shares the frozen colptr/rowval and
references the owned numeric `nzval` buffer.  This is the factor-view input
consumed by the Float64 CHOLMOD lifecycle cache.  Callers must not mutate
the returned structure.
"""
function symmetric_core_lower_sparse(pattern::SymmetricCorePattern{T}) where {T}
    return SparseMatrixCSC{T, Int}(
        pattern.dimension, pattern.dimension,
        Vector{Int}(pattern.colptr), Vector{Int}(pattern.rowval),
        pattern.nzval,
    )
end

"""
    symmetric_core_dsigns(pattern) -> Vector{Int}

Signed diagonal descriptor for the augmented core: `+1` for the reduced-x
rows and `-1` for the y / cone rows.  This is the sign vector consumed by
the signed static regularization of the CHOLMOD lifecycle.
"""
function symmetric_core_dsigns(pattern::SymmetricCorePattern{T}) where {T}
    dsigns = Vector{Int}(undef, pattern.dimension)
    @inbounds for j in 1:pattern.nr
        dsigns[j] = 1
    end
    @inbounds for j in (pattern.nr + 1):pattern.dimension
        dsigns[j] = -1
    end
    return dsigns
end

"""Frozen lower-triangle CSC colptr."""
symmetric_core_colptr(pattern::SymmetricCorePattern) = pattern.colptr

"""Frozen lower-triangle CSC rowval."""
symmetric_core_rowval(pattern::SymmetricCorePattern) = pattern.rowval

# `BigFloat` is mutable. Assignment of one scalar into another array slot
# aliases the MPFR object, so values copied into setup/workspace storage need
# owned destination objects. The generic branch remains a plain store for
# bitstypes and other immutable arithmetic types.
@inline _core_owned_value(value::BigFloat) = MA.mutable_copy(value)
@inline _core_owned_value(value) = value

@inline function _core_store_owned!(
    destination::AbstractArray{T}, index::Int, value,
) where {T}
    destination[index] = _core_owned_value(value)
    return destination
end

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
    # Every cross-declared-block entry of Theta must be exactly zero.  The
    # frozen CSC pattern stores one dense lower triangle per declared cone
    # block range; any nonzero entry between distinct block ranges has no
    # structural slot and would be silently omitted by `refill!`.  Reject
    # it here so a semantic cone-coupling never disappears from the core.
    for (index, rows) in enumerate(pattern.block_ranges)
        for column in 1:(first(rows) - 1)
            for row in rows
                iszero(Theta[row, column]) &&
                iszero(Theta[column, row]) || throw(ArgumentError(
                    "symmetric core Theta has a non-zero cross-block entry " *
                    "at ($row, $column) between declared cone block ranges",
                ))
            end
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
        index = pattern.x_diag_slots[j]
        zero_owned!(view(nzval, index:index))
    end
    slot_index = 0
    @inbounds for j in 1:pattern.nr
        for pointer in nzrange(Ar, j)
            slot_index += 1
            _core_store_owned!(nzval, pattern.ar_slots[slot_index], Ar.nzval[pointer])
        end
    end
    slot_index = 0
    for (index, rows) in enumerate(pattern.block_ranges)
        last_row = last(rows)
        for column in first(rows):last_row
            for row in column:last_row
                slot_index += 1
                _core_store_owned!(
                    nzval, pattern.theta_slots[slot_index], -Theta[row, column],
                )
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
            K[row, j] = _core_owned_value(nzval[pointer])
        end
    end
    @inbounds for j in 1:d
        for row in (j + 1):d
            K[j, row] = _core_owned_value(K[row, j])
        end
    end
    return K
end

#=====================================================================#
#    SymmetricCoreWorkspace — type-stable test-only HSD direction
#    recovery on the symmetric augmented core.
#
#        K = [ 0    Ar' ]
#            [ Ar  -Theta ]
#
#    Consumes the frozen `SymmetricCorePattern`, a concrete
#    `AbstractFactorCache` (Float64 CHOLMOD lifecycle), the orthonormal
#    rank-reduction basis `V`, and a `NewtonSystem`.  Per factor epoch it
#    solves the homogeneous core once, then solves each variable core RHS
#    sequentially (predictor, then dependent corrector) with the same
#    factor and no refactor.
#
#    CHOLMOD's nonpivoting LDL cannot factor the structurally-zero primal
#    diagonal block of `K`, so the caller factors a signed static-shifted
#    core `Kε = K + δ*diag(dsigns)` (nonzero δ).  Every core solve is then
#    refined against the retained *original* `K` with the same `Kε` factor:
#    at most two correction solves per RHS, each accepted only on strict
#    normalized original-core residual contraction.  The recovered HSD
#    direction is accepted only through the frozen `newton_residual!`.
#
#    This is a test-only API: it does not dispatch the production HSD
#    route, does not own a sign convention, and never regularizes the
#    acceptance equations.
#=====================================================================#

"""
    SymmetricCoreWorkspace{T}

Preallocated direction-recovery workspace over the symmetric augmented
core.  `pattern` owns the frozen CSC structure and numeric buffer; `cache`
is the concrete factor cache (Float64 CHOLMOD lifecycle).  `V` is the
orthonormal `n × nr` rank-reduction basis and `system` is the semantic
`NewtonSystem`.  Owned buffers:

- `rhs_core` / `sol_core`: full `(nr+m)`-vector core RHS/solution buffers;
- `dr`: reduced dual affine `V' * system.rhs.dual_affine`;
- `wx`, `wy`, `ux`, `uy`: reduced-x and y halves of the variable and
  homogeneous solutions;
- `core_work`, `core_residual`, `core_correction`: original-core `K*x`,
  backward residual `rhs - K*x`, and the `Kε⁻¹ * residual` correction;
- `dxr`, `dx`, `dy`, `ds`, `dkappa`: recovered direction buffers;
- `residual`: preallocated `NewtonResidual`;
- counters: `factor_epoch`, `homogeneous_solves`, `variable_solves`,
  `directions`, `refinements`, and the last scalar denominator.

`nr = size(V, 2)`, `n = length(system.c)`, `m = length(system.b)`,
`dimension = nr + m`; the constructor validates every dimension and the
isometry `V'V = I`.
"""
mutable struct SymmetricCoreWorkspace{
    T<:AbstractFloat,
    P<:SymmetricCorePattern{T},
    FC<:AbstractFactorCache{T},
    VV<:AbstractMatrix{T},
    S<:NewtonSystem{T},
}
    pattern::P
    cache::FC
    V::VV
    system::S
    nr::Int
    n::Int
    m::Int
    dimension::Int
    cr::Vector{T}          # V' * system.c
    rhs_core::Vector{T}
    sol_core::Vector{T}
    dr::Vector{T}          # V' * system.rhs.dual_affine (preallocated)
    wx::Vector{T}
    wy::Vector{T}
    ux::Vector{T}
    uy::Vector{T}
    core_work::Vector{T}          # K_original * x
    core_residual::Vector{T}      # rhs - K_original * x
    core_correction::Vector{T}    # Kε⁻¹ * core_residual
    dxr::Vector{T}
    dx::Vector{T}
    dy::Vector{T}
    ds::Vector{T}
    dkappa::T
    residual::NewtonResidual{T}
    original_nzval::Vector{T}  # independent snapshot of the unregularized K
    row_sums::Vector{T}        # scratch for the exact symmetric row-sum norm
    pattern_signature::UInt64  # frozen structural Ar/block signature
    structure_signature::UInt64 # full frozen CSC structure signature
    operator_signature::UInt64 # static numeric operator signature (no RHS)
    cache_signature::UInt64    # provider symbolic/factor signature, if exposed
    matrix_epoch::Int
    factor_epoch::Int
    homogeneous_epoch::Int
    synchronized::Bool
    homogeneous_solves::Int
    variable_solves::Int
    directions::Int
    refinements::Int
    denominator::T
    original_scale::T   # max(1, ‖K_original‖_∞) fixed at synchronization
end

function _core_newton_residual(
    ::Type{T}, m::Int, n::Int,
) where {T<:AbstractFloat}
    return NewtonResidual{T}(
        alloc_zeros(T, m), alloc_zeros(T, n), zero(T), alloc_zeros(T, m),
        zero(T), alloc_zeros(T, m),
    )
end

@inline function _core_mix_uint(signature::UInt64, value::UInt64)
    return (signature ⊻ value) * UInt64(0x100000001b3)
end

function _core_mix_values(
    signature::UInt64, values::AbstractArray,
)
    signature = _core_mix_uint(signature, UInt64(ndims(values)))
    for axis in axes(values)
        signature = _core_mix_uint(signature, UInt64(length(axis)))
    end
    for value in values
        signature = _core_mix_uint(signature, UInt64(hash(value)))
    end
    return signature
end

function _core_mix_ranges(
    signature::UInt64, ranges::AbstractVector{<:UnitRange{Int}},
)
    signature = _core_mix_uint(signature, UInt64(length(ranges)))
    for rows in ranges
        signature = _core_mix_uint(signature, UInt64(first(rows)))
        signature = _core_mix_uint(signature, UInt64(last(rows)))
    end
    return signature
end

"""Include every mutable frozen CSC structure field, not only its public hash."""
function _core_structure_signature(pattern::SymmetricCorePattern)
    signature = UInt64(0xcbf29ce484222325)
    signature = _core_mix_uint(signature, pattern.signature)
    signature = _core_mix_values(signature, pattern.ar_colptr)
    signature = _core_mix_values(signature, pattern.ar_rowval)
    signature = _core_mix_values(signature, pattern.colptr)
    signature = _core_mix_values(signature, pattern.rowval)
    signature = _core_mix_ranges(signature, pattern.block_ranges)
    for shape in pattern.block_shapes
        signature = _core_mix_uint(signature, UInt64(hash(shape)))
    end
    return signature
end

function _core_mix_cone_signature(
    signature::UInt64, cone::AbstractConeLinearization,
)
    signature = _core_mix_uint(signature, UInt64(hash(typeof(cone))))
    if hasproperty(cone, :operator)
        signature = _core_mix_values(signature, getproperty(cone, :operator))
    elseif hasproperty(cone, :operators)
        operators = getproperty(cone, :operators)
        signature = _core_mix_uint(signature, UInt64(length(operators)))
        for operator in operators
            signature = _core_mix_values(signature, operator)
        end
    end
    if hasproperty(cone, :block_ranges)
        signature = _core_mix_ranges(signature, getproperty(cone, :block_ranges))
    elseif hasproperty(cone, :rows)
        rows = getproperty(cone, :rows)
        signature = _core_mix_uint(signature, UInt64(first(rows)))
        signature = _core_mix_uint(signature, UInt64(last(rows)))
    end
    return signature
end

"""Content-based signature of the static Newton operator; deliberately no RHS."""
function _core_operator_signature(
    pattern::SymmetricCorePattern,
    V::AbstractMatrix,
    system::NewtonSystem,
)
    signature = _core_structure_signature(pattern)
    signature = _core_mix_values(signature, pattern.nzval)
    signature = _core_mix_values(signature, V)
    signature = _core_mix_values(signature, system.A)
    signature = _core_mix_values(signature, system.b)
    signature = _core_mix_values(signature, system.c)
    signature = _core_mix_cone_signature(signature, system.cone)
    signature = _core_mix_uint(signature, UInt64(hash(system.tau)))
    signature = _core_mix_uint(signature, UInt64(hash(system.kappa)))
    return signature
end

@inline _core_cache_signature(cache::SparseSymbolicNumericCache) = cache.signature
@inline _core_cache_signature(::AbstractFactorCache) = UInt64(0)

@inline function _core_factor_matches_pattern(
    cache::SparseSymbolicNumericCache, pattern::SymmetricCorePattern,
)
    return length(cache.original_values) == length(pattern.nzval) &&
           cache.original_values == pattern.nzval
end
@inline _core_factor_matches_pattern(::AbstractFactorCache, ::SymmetricCorePattern) = true

function _core_projection_tolerance(::Type{T}, dimension::Int, scale::T) where {T}
    return T(256 * max(1, dimension)) * eps(one(T)) * max(one(T), scale)
end

function _core_vector_in_range(
    V::AbstractMatrix{T}, vector::AbstractVector{T}, work::AbstractVector{T},
) where {T}
    zero_owned!(work)
    @inbounds for r in axes(V, 2)
        coefficient = zero(T)
        for j in axes(V, 1)
            coefficient += V[j, r] * vector[j]
        end
        for j in axes(V, 1)
            work[j] += V[j, r] * coefficient
        end
    end
    residual = zero(T)
    scale = zero(T)
    @inbounds for j in eachindex(vector, work)
        residual = max(residual, abs(vector[j] - work[j]))
        scale = max(scale, abs(vector[j]))
    end
    return residual <= _core_projection_tolerance(T, length(vector), scale)
end

function _core_operator_in_range(
    V::AbstractMatrix{T}, A::AbstractMatrix{T}, work::AbstractVector{T},
) where {T}
    n = size(A, 2)
    size(V, 1) == n || return false
    @inbounds for i in axes(A, 1)
        row = @view A[i, :]
        _core_vector_in_range(V, row, work) || return false
    end
    return true
end

function SymmetricCoreWorkspace(
    pattern::SymmetricCorePattern{T},
    cache::FC,
    V::AbstractMatrix{T},
    system::NewtonSystem{T},
) where {T<:AbstractFloat,FC<:AbstractFactorCache{T}}
    nr = size(V, 2)
    n = length(system.c)
    m = length(system.b)
    size(V, 1) == n || throw(DimensionMismatch(
        "rank-reduction basis rows $n do not match V rows $(size(V, 1))",
    ))
    pattern.nr == nr || throw(DimensionMismatch(
        "pattern reduced dimension $(pattern.nr) disagrees with V columns $nr",
    ))
    pattern.m == m || throw(DimensionMismatch(
        "pattern cone dimension $(pattern.m) disagrees with system rows $m",
    ))
    # V'V = I (orthonormal rank-reduction basis to roundoff).
    VV = Matrix{T}(V') * Matrix{T}(V)
    isometry_tol = T(64) * eps(one(T))
    for j in 1:nr, i in 1:nr
        expected = i == j ? one(T) : zero(T)
        abs(VV[i, j] - expected) <= isometry_tol || throw(ArgumentError(
            "symmetric core workspace requires V'V = I " *
            "(deviation $(abs(VV[i, j] - expected)))",
        ))
    end
    dimension = nr + m
    range_work = alloc_zeros(T, n)
    _core_operator_in_range(V, system.A, range_work) || throw(ArgumentError(
        "symmetric core requires every row of A to lie in range(V)",
    ))
    _core_vector_in_range(V, system.c, range_work) || throw(ArgumentError(
        "symmetric core requires c to lie in range(V)",
    ))
    V_owned = alloc_zeros(T, size(V, 1), size(V, 2))
    copy_owned!(V_owned, V)
    cr = alloc_zeros(T, nr)
    for j in 1:n
        for r in 1:nr
            cr[r] += V_owned[j, r] * system.c[j]
        end
    end
    original_nzval = alloc_zeros(T, length(pattern.nzval))
    copy_owned!(original_nzval, pattern.nzval)
    row_sums = alloc_zeros(T, dimension)
    original_scale = _core_original_scale!(
        row_sums, pattern.colptr, pattern.rowval, original_nzval,
    )
    pattern_signature = symmetric_core_signature(pattern)
    structure_signature = _core_structure_signature(pattern)
    operator_signature = _core_operator_signature(pattern, V_owned, system)
    return SymmetricCoreWorkspace{
        T,typeof(pattern),FC,typeof(V_owned),typeof(system),
    }(
        pattern, cache, V_owned, system,
        nr, n, m, dimension, cr,
        alloc_zeros(T, dimension), alloc_zeros(T, dimension), alloc_zeros(T, nr),
        alloc_zeros(T, nr), alloc_zeros(T, m), alloc_zeros(T, nr), alloc_zeros(T, m),
        alloc_zeros(T, dimension), alloc_zeros(T, dimension), alloc_zeros(T, dimension),
        alloc_zeros(T, nr), alloc_zeros(T, n), alloc_zeros(T, m), alloc_zeros(T, m),
        zero(T), _core_newton_residual(T, m, n), original_nzval, row_sums,
        pattern_signature, structure_signature, operator_signature,
        _core_cache_signature(cache), 0, 0, -1, false,
        0, 0, 0, 0, zero(T), original_scale,
    )
end

"""Compute `max(1, ‖K‖∞)` from the lower CSC triangle exactly."""
function _core_original_scale!(
    row_sums::AbstractVector{T},
    colptr::AbstractVector{Int}, rowval::AbstractVector{Int},
    nzval::AbstractVector{T},
) where {T}
    dimension = length(row_sums)
    length(colptr) == dimension + 1 || throw(DimensionMismatch(
        "core row-sum colptr dimension mismatch",
    ))
    zero_owned!(row_sums)
    @inbounds for column in 1:dimension
        for pointer in colptr[column]:(colptr[column + 1] - 1)
            row = rowval[pointer]
            value = abs(nzval[pointer])
            row_sums[row] += value
            row == column || (row_sums[column] += value)
        end
    end
    return max(one(T), maximum(row_sums; init=zero(T)))
end

"""Synchronize factor/matrix stamps and retain an owned original-core snapshot.

Synchronization is legal only for a fresh factor.  A first call captures the
current factor epoch, matrix epoch, structural signatures, original `K` values,
and exact symmetric infinity row-sum norm.  Later calls with the same factor
epoch are idempotent only if none of those identities changed; a refilled
pattern without a new factorization therefore fails closed.  A successful new
factor epoch resets the homogeneous solve seam.
"""
function sync_core_factor_epoch!(
    workspace::SymmetricCoreWorkspace{T},
) where {T}
    cache = workspace.cache
    SDPX.factor_status(cache) === Fresh || throw(FactorCacheStateError(
        :sync_core_factor_epoch, Fresh, SDPX.factor_status(cache),
    ))
    pattern = workspace.pattern
    pattern_signature = symmetric_core_signature(pattern)
    structure_signature = _core_structure_signature(pattern)
    operator_signature = _core_operator_signature(
        pattern, workspace.V, workspace.system,
    )
    matrix_epoch = SDPX.factor_matrix_epoch(cache)
    factor_epoch = SDPX.factor_epoch(cache)
    cache_signature = _core_cache_signature(cache)
    _core_factor_matches_pattern(cache, pattern) || throw(ArgumentError(
        "symmetric core pattern values do not match the fresh factor operator",
    ))

    if workspace.synchronized && factor_epoch == workspace.factor_epoch
        pattern_signature == workspace.pattern_signature || throw(ArgumentError(
            "symmetric core pattern signature changed without a new factor epoch",
        ))
        structure_signature == workspace.structure_signature || throw(ArgumentError(
            "symmetric core CSC structure changed without a new factor epoch",
        ))
        operator_signature == workspace.operator_signature || throw(ArgumentError(
            "symmetric core operator changed without a new factor epoch",
        ))
        matrix_epoch == workspace.matrix_epoch || throw(ArgumentError(
            "symmetric core matrix epoch changed without a new factor epoch",
        ))
        cache_signature == workspace.cache_signature ||
            cache_signature == UInt64(0) || workspace.cache_signature == UInt64(0) ||
            throw(ArgumentError(
                "symmetric core factor signature changed without a new factor epoch",
            ))
        return workspace
    end

    # A new numeric factor may be synchronized only when it still represents
    # the same static operator.  A changed pattern/operator requires a new
    # workspace (and, in particular, a new `cr`/homogeneous seam), rather than
    # silently pairing an old NewtonSystem with a new matrix.
    if workspace.synchronized
        operator_signature == workspace.operator_signature || throw(ArgumentError(
            "symmetric core static operator changed; construct a new workspace",
        ))
        pattern_signature == workspace.pattern_signature || throw(ArgumentError(
            "symmetric core pattern signature changed; construct a new workspace",
        ))
        structure_signature == workspace.structure_signature || throw(ArgumentError(
            "symmetric core CSC structure changed; construct a new workspace",
        ))
    end

    length(workspace.original_nzval) == length(pattern.nzval) || throw(
        DimensionMismatch("symmetric core original-value snapshot dimension mismatch"),
    )
    copy_owned!(workspace.original_nzval, pattern.nzval)
    workspace.original_scale = _core_original_scale!(
        workspace.row_sums, pattern.colptr, pattern.rowval,
        workspace.original_nzval,
    )
    workspace.pattern_signature = pattern_signature
    workspace.structure_signature = structure_signature
    workspace.operator_signature = operator_signature
    workspace.cache_signature = cache_signature
    workspace.matrix_epoch = matrix_epoch
    workspace.factor_epoch = factor_epoch
    workspace.homogeneous_epoch = -1
    workspace.synchronized = true
    return workspace
end

"""Evaluate `destination = K_original * source` over the frozen lower
CSC triangle, expanding the symmetry so the product is exact for the
stored original core values (never the regularized factor view)."""
function _core_apply!(
    workspace::SymmetricCoreWorkspace{T},
    destination::AbstractVector{T},
    source::AbstractVector{T},
) where {T}
    d = workspace.dimension
    length(destination) == d || throw(DimensionMismatch(
        "core apply destination dimension mismatch",
    ))
    length(source) == d || throw(DimensionMismatch(
        "core apply source dimension mismatch",
    ))
    colptr = workspace.pattern.colptr
    rowval = workspace.pattern.rowval
    nzval = workspace.original_nzval
    zero_owned!(destination)
    # Lower triangle: for column j, rows i >= j with entry K[i,j];
    # the symmetric upper entry is K[j,i] = K[i,j].
    @inbounds for j in 1:d
        value = source[j]
        for pointer in colptr[j]:(colptr[j + 1] - 1)
            row = rowval[pointer]
            entry = nzval[pointer]
            destination[row] += entry * value
            row == j || (destination[j] += entry * source[row])
        end
    end
    return destination
end

"""Normalized original-core backward error `‖rhs - K*x‖ / (‖K‖‖x‖ + ‖rhs‖)`.
The `‖K‖` factor is fixed once per workspace from the original CSC values."""

"""Normalized residual `‖r‖_∞ / (K_scale * ‖x‖_∞ + ‖rhs‖_∞)`."""
function _core_normalized_residual(
    workspace::SymmetricCoreWorkspace{T},
    rhs::AbstractVector{T},
    x::AbstractVector{T},
    residual::AbstractVector{T},
) where {T}
    norm_r = maximum(abs, residual; init=zero(T))
    norm_x = maximum(abs, x; init=zero(T))
    norm_rhs = maximum(abs, rhs; init=zero(T))
    scale = workspace.original_scale * norm_x + norm_rhs
    iszero(norm_r) && iszero(scale) && return zero(T)
    isfinite(scale) && scale > zero(T) || return T(Inf)
    return norm_r / scale
end

"""Solve `Kε⁻¹ * rhs` into `x` through the (regularized) factor cache."""
function _core_solve_with_cache!(
    workspace::SymmetricCoreWorkspace{T},
    x::AbstractVector{T},
    rhs::AbstractVector{T},
) where {T}
    solve!(workspace.cache, x, rhs)
    all(isfinite, x) || throw(ArgumentError(
        "symmetric core refined solve produced non-finite data",
    ))
    return x
end

"""Iterative refinement of `x` against the original core `K`.

    x₀ = Kε⁻¹ rhs
    for k in 1:2
        r = rhs - K*xₖ₋₁
        ηₖ = ‖r‖ / (K_scale*‖xₖ₋₁‖ + ‖rhs‖)
        ηₖ == 0 → accept
        ηₖ ≥ ηₖ₋₁ → reject (no contraction, with η₀ the unrefined error)
        xₖ = xₖ₋₁ + Kε⁻¹ r
    end

Returns `(x, refinements)`; the caller then re-evaluates the frozen
five-equation residual on the refined direction.  The final direction is
accepted only if that production residual passes.  A correction that does
not strictly contract the normalized original-core residual fails closed.
"""
function _core_refine!(
    workspace::SymmetricCoreWorkspace{T},
    x::AbstractVector{T},
    rhs::AbstractVector{T},
) where {T}
    # initial normalized error
    _core_apply!(workspace, workspace.core_residual, x)
    @inbounds for i in 1:workspace.dimension
        _core_store_owned!(
            workspace.core_residual, i, rhs[i] - workspace.core_residual[i],
        )
    end
    previous = _core_normalized_residual(
        workspace, rhs, x, workspace.core_residual,
    )
    isfinite(previous) || throw(ArgumentError(
        "symmetric core refinement initial residual is non-finite",
    ))
    # Do not solve a correction once the original-core residual is already at
    # the arithmetic floor.  The recovered direction still faces the frozen
    # five-equation gate; this only avoids demanding strict contraction of
    # roundoff that cannot be represented.
    residual_floor = T(256) * eps(one(T))
    previous <= residual_floor && return (x, 0)
    corrections = 0
    for _ in 1:2
        SDPX.refine_once!(
            workspace.cache, workspace.core_residual, workspace.core_correction,
        )
        @inbounds for i in 1:workspace.dimension
            x[i] += workspace.core_correction[i]
        end
        corrections += 1
        workspace.refinements += 1
        _core_apply!(workspace, workspace.core_residual, x)
        @inbounds for i in 1:workspace.dimension
            _core_store_owned!(
                workspace.core_residual, i, rhs[i] - workspace.core_residual[i],
            )
        end
        current = _core_normalized_residual(
            workspace, rhs, x, workspace.core_residual,
        )
        isfinite(current) || throw(ArgumentError(
            "symmetric core refinement produced a non-finite residual",
        ))
        if current > residual_floor
            current < previous || throw(ArgumentError(
                "symmetric core refinement did not strictly contract the " *
                "original-core residual (η=$(current) >= η_prev=$(previous))",
            ))
        end
        previous = current
        previous <= residual_floor && break
    end
    return (x, corrections)
end

"""Reduced dual affine `V' * system.rhs.dual_affine` into `dr`."""
function _core_dr!(
    workspace::SymmetricCoreWorkspace{T}, system::NewtonSystem{T},
) where {T}
    _core_vector_in_range(
        workspace.V, system.rhs.dual_affine, workspace.dx,
    ) || throw(ArgumentError(
        "symmetric core dual RHS has a discarded nullspace component",
    ))
    dr = workspace.dr
    zero_owned!(dr)
    V = workspace.V
    @inbounds for j in 1:workspace.n
        value = system.rhs.dual_affine[j]
        for r in 1:workspace.nr
            dr[r] += V[j, r] * value
        end
    end
    return dr
end

"""Pack the full core RHS `[ dr; p - h ]`."""
function _core_pack_rhs!(
    rhs_core::AbstractVector{T}, dr::AbstractVector{T},
    system, nr::Int,
) where {T}
    m = length(system.b)
    @inbounds for r in 1:nr
        _core_store_owned!(rhs_core, r, dr[r])
    end
    @inbounds for i in 1:m
        _core_store_owned!(
            rhs_core, nr + i,
            system.rhs.primal_affine[i] - system.rhs.cone_corrector[i],
        )
    end
    return rhs_core
end

"""Split a core solution into its reduced-x and y halves."""
function _core_split!(
    xr::AbstractVector{T}, y::AbstractVector{T}, sol::AbstractVector{T},
    nr::Int, m::Int,
) where {T}
    @inbounds for r in 1:nr
        _core_store_owned!(xr, r, sol[r])
    end
    @inbounds for i in 1:m
        _core_store_owned!(y, i, sol[nr + i])
    end
    return (xr, y)
end

function _core_guard_ready!(
    workspace::SymmetricCoreWorkspace{T}, system::NewtonSystem{T},
) where {T}
    cache = workspace.cache
    status = SDPX.factor_status(cache)
    status === Fresh || throw(FactorCacheStateError(
        :solve_core_direction, Fresh, status,
    ))
    workspace.synchronized || throw(ArgumentError(
        "symmetric core workspace has not been synchronized with its factor",
    ))
    factor_epoch_now = SDPX.factor_epoch(cache)
    factor_epoch_now == workspace.factor_epoch || throw(ArgumentError(
        "symmetric core factor epoch changed: got $factor_epoch_now, " *
        "expected $(workspace.factor_epoch)",
    ))
    matrix_epoch_now = SDPX.factor_matrix_epoch(cache)
    matrix_epoch_now == workspace.matrix_epoch || throw(ArgumentError(
        "symmetric core matrix epoch changed: got $matrix_epoch_now, " *
        "expected $(workspace.matrix_epoch)",
    ))
    pattern_signature_now = symmetric_core_signature(workspace.pattern)
    pattern_signature_now == workspace.pattern_signature || throw(ArgumentError(
        "symmetric core pattern signature changed after synchronization",
    ))
    structure_signature_now = _core_structure_signature(workspace.pattern)
    structure_signature_now == workspace.structure_signature || throw(ArgumentError(
        "symmetric core CSC structure changed after synchronization",
    ))
    operator_signature_now = _core_operator_signature(
        workspace.pattern, workspace.V, system,
    )
    operator_signature_now == workspace.operator_signature || throw(ArgumentError(
        "symmetric core static operator changed; only RHS may change",
    ))
    cache_signature_now = _core_cache_signature(cache)
    (cache_signature_now == workspace.cache_signature ||
     cache_signature_now == UInt64(0) || workspace.cache_signature == UInt64(0)) ||
        throw(ArgumentError(
            "symmetric core factor signature changed after synchronization",
        ))
    return workspace
end

"""Solve the homogeneous core at most once for the synchronized factor epoch."""
function solve_core_homogeneous!(
    workspace::SymmetricCoreWorkspace{T},
) where {T}
    system = workspace.system
    _core_guard_ready!(workspace, system)
    workspace.homogeneous_epoch == workspace.factor_epoch && return workspace
    nr = workspace.nr
    # RHS: [ -V'c ; b ].
    @inbounds for r in 1:nr
        _core_store_owned!(workspace.rhs_core, r, -workspace.cr[r])
    end
    @inbounds for i in 1:workspace.m
        _core_store_owned!(workspace.rhs_core, nr + i, system.b[i])
    end
    all(isfinite, workspace.rhs_core) || throw(ArgumentError(
        "symmetric core homogeneous RHS is non-finite",
    ))
    SDPX.solve!(workspace.cache, workspace.sol_core, workspace.rhs_core)
    all(isfinite, workspace.sol_core) || throw(ArgumentError(
        "symmetric core homogeneous solve produced non-finite data",
    ))
    # Refine against the independent original K using the same factor.
    _core_refine!(workspace, workspace.sol_core, workspace.rhs_core)
    _core_split!(workspace.ux, workspace.uy, workspace.sol_core, nr,
        workspace.m)
    workspace.homogeneous_epoch = workspace.factor_epoch
    workspace.homogeneous_solves += 1
    return workspace
end

"""Scalar denominator `κ + τ*(cr'ux + b'uy)` for the current epoch."""
function _core_denominator(
    workspace::SymmetricCoreWorkspace{T}, system::NewtonSystem{T},
) where {T}
    eta = zero(T)
    eta_work = zero(T)
    @inbounds for r in 1:workspace.nr
        term = workspace.cr[r] * workspace.ux[r]
        eta += term
        eta_work += abs(term)
    end
    @inbounds for i in 1:workspace.m
        term = system.b[i] * workspace.uy[i]
        eta += term
        eta_work += abs(term)
    end
    denominator = system.kappa + system.tau * eta
    work = abs(system.kappa) + abs(system.tau) * eta_work
    return denominator, work
end

function _core_snapshot(source::AbstractVector{T}) where {T<:AbstractFloat}
    destination = alloc_zeros(T, length(source))
    copy_owned!(destination, source)
    return destination
end

function _core_snapshot(residual::NewtonResidual{T}) where {T<:AbstractFloat}
    return NewtonResidual{T}(
        _core_snapshot(residual.primal_affine),
        _core_snapshot(residual.dual_affine),
        _core_owned_value(residual.homogeneous_gap),
        _core_snapshot(residual.cone_complementarity),
        _core_owned_value(residual.tau_kappa),
        _core_snapshot(residual.cone_work),
    )
end

"""Solve one variable RHS and recover an independent direction snapshot."""
function solve_core_direction!(
    workspace::SymmetricCoreWorkspace{T}, system::NewtonSystem{T},
) where {T}
    _core_guard_ready!(workspace, system)
    workspace.homogeneous_epoch == workspace.factor_epoch || throw(ArgumentError(
        "symmetric core homogeneous solution is stale or unavailable",
    ))
    nr = workspace.nr
    cache = workspace.cache
    _core_dr!(workspace, system)
    _core_pack_rhs!(workspace.rhs_core, workspace.dr, system, nr)
    all(isfinite, workspace.rhs_core) || throw(ArgumentError(
        "symmetric core variable RHS is non-finite",
    ))
    SDPX.solve!(cache, workspace.sol_core, workspace.rhs_core)
    all(isfinite, workspace.sol_core) || throw(ArgumentError(
        "symmetric core variable solve produced non-finite data",
    ))
    _core_refine!(workspace, workspace.sol_core, workspace.rhs_core)
    _core_split!(workspace.wx, workspace.wy, workspace.sol_core, nr,
        workspace.m)

    # Scalar recovery.
    eta_w = zero(T)
    @inbounds for r in 1:nr
        eta_w += workspace.cr[r] * workspace.wx[r]
    end
    @inbounds for i in 1:workspace.m
        eta_w += system.b[i] * workspace.wy[i]
    end
    denominator, denominator_work = _core_denominator(workspace, system)
    isfinite(denominator) && isfinite(denominator_work) || throw(ArgumentError(
        "symmetric core scalar denominator or absolute work is non-finite",
    ))
    iszero(denominator) && throw(ArgumentError(
        "symmetric core scalar denominator is exactly zero",
    ))
    denominator_scale = max(one(T), denominator_work)
    sqrt_eps = sqrt(eps(one(T)))
    abs(denominator) <= sqrt_eps * denominator_scale && throw(
        ArgumentError(
            "symmetric core scalar denominator is type-scaled near-zero",
        ),
    )
    workspace.denominator = denominator
    numerator = system.rhs.tau_kappa - system.tau *
                (system.rhs.homogeneous_gap + eta_w)
    dtau = numerator / denominator
    isfinite(dtau) || throw(ArgumentError(
        "symmetric core scalar recovery produced non-finite dtau",
    ))
    @inbounds for r in 1:nr
        _core_store_owned!(workspace.dxr, r,
            workspace.wx[r] + dtau * workspace.ux[r])
    end
    @inbounds for i in 1:workspace.m
        _core_store_owned!(workspace.dy, i,
            workspace.wy[i] + dtau * workspace.uy[i])
    end
    # dx = V * dxr.
    zero_owned!(workspace.dx)
    @inbounds for r in 1:nr
        for j in 1:workspace.n
            workspace.dx[j] += workspace.V[j, r] * workspace.dxr[r]
        end
    end
    # ds = p - A*dx + b*dtau (frozen primal affine equation).
    for i in 1:workspace.m
        acc = system.rhs.primal_affine[i] + system.b[i] * dtau
        for j in 1:workspace.n
            acc -= system.A[i, j] * workspace.dx[j]
        end
        _core_store_owned!(workspace.ds, i, acc)
    end
    # dkappa = g + c'*dx + b'*dy.
    dk = system.rhs.homogeneous_gap
    @inbounds for j in 1:workspace.n
        dk += system.c[j] * workspace.dx[j]
    end
    @inbounds for i in 1:workspace.m
        dk += system.b[i] * workspace.dy[i]
    end
    workspace.dkappa = _core_owned_value(dk)
    candidate = NewtonDirection(
        workspace.dx, workspace.dy, workspace.ds, dtau, workspace.dkappa,
    )
    newton_residual!(workspace.residual, system, candidate)
    workspace.variable_solves += 1
    workspace.directions += 1

    # Never return workspace-owned arrays: predictor/corrector results must
    # remain immutable snapshots when the next RHS reuses all buffers.
    direction = NewtonDirection(
        _core_snapshot(workspace.dx), _core_snapshot(workspace.dy),
        _core_snapshot(workspace.ds), _core_owned_value(dtau),
        _core_owned_value(workspace.dkappa),
    )
    return (direction, _core_snapshot(workspace.residual))
end

"""Compatibility wrapper consuming the workspace's original NewtonSystem."""
function solve_core_direction!(
    workspace::SymmetricCoreWorkspace{T},
) where {T}
    return solve_core_direction!(workspace, workspace.system)
end

#=====================================================================#
#    C5: dense symmetric-core factor factory (MFLA/BFLA).
#
#    The direction workspace is arithmetic-agnostic.  This seam builds a
#    dense pivoted-LDL `AbstractFactorCache{T}` for the same `K` operator
#    through the existing MFLA/BFLA provider adapters, after a conservative
#    memory gate.  SDPX owns no dense LDL kernel and no new backend.
#=====================================================================#

"""Conservative dense symmetric-core byte estimate for `dimension^2` K.

The augmented core `K = [0 Ar'; Ar -Theta]` is stored once as a dense
`dimension × dimension` matrix at the exact arithmetic `T`; the factor
cache additionally owns its numeric factor/scratch.  This is an upper bound
using the existing saturating product and margin helper, so it can be
compared against a memory budget without overflow.
"""
function symmetric_core_dense_bytes(
    ::Type{T}, dimension::Integer,
) where {T<:AbstractFloat}
    dimension < 0 && throw(ArgumentError(
        "symmetric core dimension must be nonnegative",
    ))
    dimension > typemax(Int) && return typemax(Int)
    d = Int(dimension)
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    counted = saturating_sum_bytes(
        saturating_bytes(2, scalar_bytes, d, d),  # dense K + factor storage
        saturating_bytes(8, scalar_bytes, d),
    )
    return _workspace_estimate_with_margin(counted, 1)
end

"""Eligibility gate for a dense symmetric-core solve.

Runs before any dense allocation or factorization.  `memory_limit_bytes` may
be `nothing` (unknown → fail closed) or a recorded budget; `current_rss_bytes`
is the process peak RSS (unknown → fail closed).  Uses the existing
`conservative_memory_upper_bound_eligibility` semantics unchanged.
"""
function symmetric_core_dense_eligibility(
    ::Type{T},
    dimension::Integer,
    memory_limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer},
) where {T<:AbstractFloat}
    estimate = symmetric_core_dense_bytes(T, dimension)
    return conservative_memory_upper_bound_eligibility(
        estimate, memory_limit_bytes, current_rss_bytes,
    )
end

"""Provider-agnostic build seam; concrete providers implement this.

`precision_bits` is exact for BigFloat and advisory-but-required for
MultiFloat (the storage width is a type property, checked against the actual
`T`).  Returns an already-prepared `AbstractFactorCache{T}` that owns a dense
pivoted-LDL factor of the same `K` operator.
"""
function build_symmetric_core_ldlt_cache(
    ::Type{T},
    pattern::SymmetricCorePattern{T},
    precision_bits::Int,
    memory_limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer},
) where {T<:AbstractFloat}
    eligibility = symmetric_core_dense_eligibility(
        T, pattern.dimension, memory_limit_bytes, current_rss_bytes,
    )
    eligibility.eligible || throw(ArgumentError(
        "symmetric core dense factor ineligible: $(eligibility.reason)",
    ))
    return _build_symmetric_core_ldlt_cache_provider(
        T, pattern, precision_bits,
    )
end

"""Dispatch to the provider extension; absent provider fails closed."""
function _build_symmetric_core_ldlt_cache_provider(
    ::Type{T},
    pattern::SymmetricCorePattern{T},
    precision_bits::Int,
) where {T<:AbstractFloat}
    throw(ArgumentError(
        "symmetric core dense LDL has no provider for arithmetic $(T); " *
        "load MFLA for MultiFloat or BFLA for BigFloat",
    ))
end

#=====================================================================#
#    C6a: semantic/provider shadow bridge (cold/test seam).
#
#    Builds the existing `SymmetricCorePattern` + provider cache +
#    `SymmetricCoreWorkspace` directly from a frozen `NewtonSystem` and a
#    rank-reduction basis `V`.  This is deliberately not a production
#    dispatch: the caller keeps the old route authoritative and compares
#    recovered directions in validation only.
#=====================================================================#

"""Extract ordered cone block ranges from a semantic cone linearization.

Supports `ProductConeLinearization` and `BlockProductConeLinearization`.
The returned ranges must exactly cover `1:m`; a linearization without an
ordered block structure fails closed rather than guessing.
"""
function symmetric_core_block_ranges(
    cone::Union{ProductConeLinearization,BlockProductConeLinearization},
)
    ranges = UnitRange{Int}[rows for rows in product_cone_block_ranges(cone)]
    m = cone_dimension(cone)
    validate_product_cone_block_ranges(m, ranges)
    return ranges
end
symmetric_core_block_ranges(::AbstractConeLinearization) = throw(ArgumentError(
    "symmetric core requires a product or block-product cone linearization",
))

"""Materialize the full symmetric `Theta` operator of a semantic cone.

For a `ProductConeLinearization` the dense operator is used directly.  For a
`BlockProductConeLinearization` the per-block operators are placed on the
corresponding diagonal blocks (cross-block entries must be exactly zero, as
enforced by `validate_symmetric_core`).  Returns an owned `Matrix{T}`.
"""
function symmetric_core_theta(
    cone::ProductConeLinearization{T},
) where {T<:AbstractFloat}
    theta = alloc_zeros(T, size(cone.operator, 1), size(cone.operator, 2))
    copy_owned!(theta, cone.operator)
    return theta
end

function symmetric_core_theta(
    cone::BlockProductConeLinearization{T},
) where {T<:AbstractFloat}
    m = cone_dimension(cone)
    theta = alloc_zeros(T, m, m)
    for (index, rows) in enumerate(cone.block_ranges)
        block = cone.operators[index]
        # Owned copy: BigFloat blocks are mutable, so a block view must not
        # alias the returned operator or share objects within/across slots.
        copy_owned!(view(theta, rows, rows), block)
    end
    return theta
end

"""Validate documented rank-reduction preconditions before any materialization.

Checks V dimensions, `V'V = I`, `range(A') = range(V)` (every row of `A` in
`range(V)`), and `c in range(V)`.  Uses only owned temporary work; fails
closed on any violated precondition.  This is the single cold-path gate shared
by `symmetric_core_pattern_from_system` and `build_symmetric_core_workspace`
so a high-precision provider route never constructs/allocates `Theta`, `A*V`,
or a pattern before the operator is provably representable.
"""
function _validate_core_preconditions(
    system::NewtonSystem{T},
    V::AbstractMatrix{T},
) where {T<:AbstractFloat}
    n = length(system.c)
    nr = size(V, 2)
    size(V, 1) == n || throw(DimensionMismatch(
        "rank-reduction basis rows $n do not match V rows $(size(V, 1))",
    ))
    nr >= 0 || throw(ArgumentError(
        "rank-reduction basis column count must be nonnegative",
    ))
    # V'V = I to roundoff, evaluated with scalar dots so preflight never
    # allocates dense copies or an nr×nr Gram matrix.
    isometry_tol = T(64) * eps(one(T))
    @inbounds for j in 1:nr, i in 1:nr
        value = zero(T)
        for k in 1:n
            value += V[k, i] * V[k, j]
        end
        expected = i == j ? one(T) : zero(T)
        abs(value - expected) <= isometry_tol || throw(ArgumentError(
            "symmetric core requires V'V = I " *
            "(deviation $(abs(value - expected)))",
        ))
    end
    range_work = alloc_zeros(T, n)
    _core_operator_in_range(V, system.A, range_work) || throw(ArgumentError(
        "symmetric core requires every row of A to lie in range(V)",
    ))
    _core_vector_in_range(V, system.c, range_work) || throw(ArgumentError(
        "symmetric core requires c to lie in range(V)",
    ))
    return nothing
end

"""Base provider availability/precision gate for the dense symmetric core.

Concrete MFLA/BFLA extensions override this hook; the base implementation
fails closed so an absent provider is rejected before any dense allocation.
"""
function symmetric_core_provider_available(
    ::Type{T}, precision_bits::Int,
) where {T<:AbstractFloat}
    throw(ArgumentError(
        "symmetric core dense LDL has no provider for arithmetic $(T); " *
        "load MFLA for MultiFloat or BFLA for BigFloat",
    ))
end

"""Build a frozen `SymmetricCorePattern` from a semantic system + rank basis.

Validates the documented rank/range preconditions, forms `Ar = A*V`, and
freezes the block-diagonal lower-triangle pattern from the cone block ranges.
The pattern is then refilled with the exact `Theta` blocks.
"""
function _symmetric_core_pattern_from_validated(
    system::NewtonSystem{T}, V::AbstractMatrix{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    nr = size(V, 2)
    size(V, 1) == n || throw(DimensionMismatch(
        "rank-reduction basis rows $n do not match V rows $(size(V, 1))",
    ))
    ranges = symmetric_core_block_ranges(system.cone)
    theta = symmetric_core_theta(system.cone)
    ar = sparse(system.A * V)
    pattern = SymmetricCorePattern{T}(
        ar, ranges, [:dense_lower for _ in ranges],
    )
    refill!(pattern, ar, theta)
    return pattern
end

function symmetric_core_pattern_from_system(
    system::NewtonSystem{T}, V::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _validate_core_preconditions(system, V)
    return _symmetric_core_pattern_from_validated(system, V)
end

"""Build a synchronized symmetric-core workspace for a frozen system + basis.

Float64 uses the signed-shifted CHOLMOD lifecycle cache; MultiFloat/BigFloat
run the dense memory eligibility gate before the provider LDL factory.  The
workspace is then synchronized with the factor and the homogeneous solution is
solved once.  The caller compares directions against its own route.
"""
function build_symmetric_core_workspace(
    system::NewtonSystem{T},
    V::AbstractMatrix{T},
    matrix_epoch::Integer,
    precision_bits::Int,
    memory_limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer},
    regularization::Real;
    symbolic_epoch::Integer=0,
) where {T<:AbstractFloat}
    if T !== Float64
        # Provider and dimension-only memory eligibility are checked before
        # even the bounded rank/isometry scratch is allocated.
        symmetric_core_provider_available(T, precision_bits)
        eligibility = symmetric_core_dense_eligibility(
            T, size(V, 2) + length(system.b), memory_limit_bytes,
            current_rss_bytes,
        )
        eligibility.eligible || throw(ArgumentError(
            "symmetric core dense factor ineligible: $(eligibility.reason)",
        ))
    end
    # Rank/range preconditions precede Theta, A*V, pattern, and factor work.
    _validate_core_preconditions(system, V)
    pattern = _symmetric_core_pattern_from_validated(system, V)
    cache = if T === Float64
        isfinite(regularization) && regularization >= 0 || throw(ArgumentError(
            "symmetric core Float64 regularization must be finite and nonnegative",
        ))
        k = symmetric_core_lower_sparse(pattern)
        requirements = SparseSymbolicRequirements(
            k;
            symbolic_epoch=Int(symbolic_epoch),
            dsigns=symmetric_core_dsigns(pattern),
            regularization=Float64(regularization),
        )
        cache = SparseSymbolicNumericCache{Float64}()
        prepare!(cache, requirements)
        factorize!(cache, k, Int(matrix_epoch))
        cache
    else
        cache = build_symmetric_core_ldlt_cache(
            T, pattern, precision_bits, memory_limit_bytes, current_rss_bytes,
        )
        factorize!(cache, materialize_dense(pattern), Int(matrix_epoch))
        cache
    end
    workspace = SymmetricCoreWorkspace(pattern, cache, V, system)
    sync_core_factor_epoch!(workspace)
    solve_core_homogeneous!(workspace)
    return workspace
end
