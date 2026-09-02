#=====================================================================#
#    IdentityRankBasis — zero-payload identity rank-reduction basis.
#
#    The full-rank sparse path preserves original coordinates, so its
#    reduction basis is the identity with zero stored numerical values and
#    zero stored indices: the struct holds only the dimension `n` and carries
#    no Vector/Matrix payload at all.  Defined here (included before the HSD
#    state files) so both `symmetric_core.jl` and the later-included sparse
#    HSD files share one concrete type.
#=====================================================================#

"""
    IdentityRankBasis{T} <: AbstractMatrix{T}

Zero-payload typed identity rank-reduction basis.  The full-rank sparse
path preserves original coordinates, so the reduction basis is the identity
with zero stored numerical values and zero stored indices: this object holds
only the dimension `n` and carries no `Vector`/`Matrix` payload at all.  It
implements the minimal `AbstractMatrix` surface so generic code can call
`size`, `axes`, and `getindex` without materializing an identity.

Every hot/cold path that would otherwise iterate the `n×n` identity must
branch on `_hsd_is_identity_basis` first and never touch `getindex` in a
loop.  Concrete consumers (residual projection, direction scatter, `V'c` /
`V'd`, range/isometry validation, operator signatures, `Ar` reuse, core
recovery) use the identity fast paths in `hsd.jl`, `common_runtime.jl`, and
`symmetric_core.jl`.
"""
struct IdentityRankBasis{T<:AbstractFloat} <: AbstractMatrix{T}
    dimension::Int
end

IdentityRankBasis(::Type{T}, dimension::Integer) where {T<:AbstractFloat} =
    IdentityRankBasis{T}(Int(dimension))

Base.size(basis::IdentityRankBasis) = (basis.dimension, basis.dimension)
Base.size(basis::IdentityRankBasis, d::Integer) =
    d == 1 || d == 2 ? basis.dimension : 1
Base.axes(basis::IdentityRankBasis) =
    (Base.OneTo(basis.dimension), Base.OneTo(basis.dimension))
Base.axes(basis::IdentityRankBasis, d::Integer) =
    d == 1 || d == 2 ? Base.OneTo(basis.dimension) : Base.OneTo(1)
@inline function Base.getindex(
    ::IdentityRankBasis{T}, i::Integer, j::Integer,
) where {T}
    i == j && return one(T)
    return zero(T)
end

Base.IndexStyle(::Type{<:IdentityRankBasis}) = IndexCartesian()

@inline _hsd_is_identity_basis(::IdentityRankBasis) = true

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
    theta_nnz=0
    for rows in block_ranges
        block_size=length(rows)
        triangle=div(Base.checked_mul(block_size,block_size+1),2)
        theta_nnz=Base.checked_add(theta_nnz,triangle)
    end
    ar_nnz=nnz(Ar)
    structural_nnz=Base.checked_add(Base.checked_add(nr,ar_nnz),theta_nnz)

    signature = _symmetric_core_structure_signature(
        nr, m, Ar.colptr, Ar.rowval, block_ranges, block_shapes,
    )
    # Review slice 2: cross-solve structure cache.  The key mixes the
    # arithmetic type and the full CSC structural signature (dimensions,
    # Ar pattern, block ranges, block shapes), so any change in dimension,
    # cone partition, sparsity pattern, or formulation misses.  A hit
    # shares ONLY the frozen structural arrays; the numeric buffer is a
    # FRESH zero alloczation so no value can survive a reuse.
    key = (T, signature)
    cached = lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
        get(_SYMMETRIC_CORE_STRUCTURE_CACHE.patterns, key, nothing)
    end
    if cached isa NamedTuple && haskey(cached, :colptr)
        _structure_cache_record_hit!()
        nzval = alloc_zeros(T, structural_nnz)
        return SymmetricCorePattern{T}(
            nr, m, dimension, cached.a_colptr, cached.a_rowval,
            cached.block_ranges, cached.block_shapes,
            cached.colptr, cached.rowval, cached.ar_slots,
            cached.theta_slots, cached.x_diag_slots, nzval, signature,
        )
    elseif _SYMMETRIC_CORE_STRUCTURE_CACHE.enabled
        _structure_cache_record_miss!()
    end

    # ---- Frozen lower-triangle CSC structure ---------------------
    colptr = Vector{Int}(undef, dimension + 1)
    rowval = Int[]
    sizehint!(rowval,structural_nnz)
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
    sizehint!(ar_slots,ar_nnz)
    x_diag_slots = Vector{Int}(undef, nr)
    theta_slots = Int[]
    sizehint!(theta_slots,theta_nnz)
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

    if _SYMMETRIC_CORE_STRUCTURE_CACHE.enabled
        # Store the frozen structural content (no values).  The arrays are
        # immutable by contract: colptr/rowval are never rewritten after
        # construction, so sharing them across patterns is ownership-safe.
        lock(_SYMMETRIC_CORE_STRUCTURE_LOCK) do
            _SYMMETRIC_CORE_STRUCTURE_CACHE.patterns[key] = (
                a_colptr=Vector{Int}(Ar.colptr),
                a_rowval=Vector{Int}(Ar.rowval),
                block_ranges=UnitRange{Int}[rows for rows in block_ranges],
                block_shapes=Symbol[shape for shape in block_shapes],
                colptr=colptr, rowval=rowval,
                ar_slots=ar_slots, theta_slots=theta_slots,
                x_diag_slots=x_diag_slots,
            )
        end
    end
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
        pattern.colptr, pattern.rowval, pattern.nzval,
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

function _core_write_ar!(
    pattern::SymmetricCorePattern{T}, Ar::SparseMatrixCSC{T},
) where {T<:AbstractFloat}
    size(Ar) == (pattern.m, pattern.nr) || throw(DimensionMismatch(
        "symmetric core Ar dimensions $(size(Ar)) disagree with frozen " *
        "$(pattern.m)×$(pattern.nr)",
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
    nzval = pattern.nzval
    @inbounds for index in pattern.x_diag_slots
        zero_owned!(view(nzval, index:index))
    end
    @inbounds for (slot_index, pointer) in enumerate(eachindex(Ar.nzval))
        _core_store_owned!(
            nzval, pattern.ar_slots[slot_index], Ar.nzval[pointer],
        )
    end
    return pattern
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
    _core_write_ar!(pattern, Ar)
    nzval = pattern.nzval
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
    # Production raw-solve buffers: owned negated residual storage (semantic
    # Newton RHS) and the last recovered dtau.  The public snapshot wrapper
    # uses these too; the raw production path writes them in place.
    negated_primal::Vector{T}
    negated_dual::Vector{T}
    last_dtau::T
    residual::NewtonResidual{T}
    original_nzval::Vector{T}  # independent snapshot of the unregularized K
    row_sums::Vector{T}        # scratch for the exact symmetric row-sum norm
    pattern_signature::UInt64  # frozen structural Ar/block signature
    structure_signature::UInt64 # full frozen CSC structure signature
    operator_signature::UInt64 # static operator identity (A,b,c,V,structure)
    theta_signature::UInt64    # numeric Theta signature frozen per factor epoch
    epoch_tau::T               # tau frozen at the factor epoch
    epoch_kappa::T             # kappa frozen at the factor epoch
    cache_signature::UInt64    # provider symbolic/factor signature, if exposed
    factor_receipt::Union{Nothing,FactorReceipt{T}}
    receipt_build_count::Int
    matrix_epoch::Int
    factor_epoch::Int
    homogeneous_epoch::Int
    synchronized::Bool
    homogeneous_solves::Int
    variable_solves::Int
    directions::Int
    refinements::Int
    denominator::T
    scalar_closure::Symbol
    dy_gauged::Bool
    original_scale::T   # max(1, ‖K_original‖_∞) fixed at synchronization
    # Review slice 1: phase-level timing accumulator owned by the product-HSD
    # state; the raw core solve writes the refinement wall bucket into it.
    # `nothing` for setups outside the product-HSD hot route (zero overhead).
    phase_timings::Union{Nothing,ProductHSDPhaseTimings}
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
    isempty(values) && return signature
    len = length(values)
    step = max(1, len ÷ 64)
    @inbounds for idx in 1:step:len
        val = values[idx]
        signature = _core_mix_uint(signature, UInt64(hash(Float64(val))))
    end
    return signature
end

function _core_mix_values(
    signature::UInt64, basis::IdentityRankBasis,
)
    signature = _core_mix_uint(signature, UInt64(2))
    signature = _core_mix_uint(signature, UInt64(size(basis, 1)))
    signature = _core_mix_uint(signature, UInt64(size(basis, 2)))
    # A zero-payload identity contributes no numerical values; its static
    # identity is fully described by the type and the dimension.
    return signature
end

function _core_mix_values(
    signature::UInt64, values::SparseMatrixCSC,
)
    signature = _core_mix_uint(signature, UInt64(2))
    signature = _core_mix_uint(signature, UInt64(size(values, 1)))
    signature = _core_mix_uint(signature, UInt64(size(values, 2)))
    signature = _core_mix_values(signature, values.colptr)
    signature = _core_mix_values(signature, values.rowval)
    return _core_mix_values(signature, values.nzval)
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

"""Content-based signature of the full Newton operator; deliberately no RHS.

Includes the current numeric `pattern.nzval` (Theta blocks), `tau`/`kappa`, and
the cone numeric values.  Used by tests to prove a numeric mutation is visible.
"""
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

"""Static symmetric-core identity: A, b, c, V and the frozen CSC structure.

Deliberately excludes the numeric `Theta` (pattern.nzval), `tau`/`kappa`, and
every RHS field.  This identity must be invariant across scaling/matrix epochs;
only the Theta numeric values, tau/kappa and RHS may change.
"""
function _core_static_signature(
    pattern::SymmetricCorePattern,
    V::AbstractMatrix,
    system::NewtonSystem,
)
    signature = _core_structure_signature(pattern)
    # The semantic cone block partition is part of the static identity: a
    # changed partition (with the same dense operator values) must be
    # rejected at factor/guard/refill even though the CSC structure is
    # unchanged.
    signature = _core_mix_uint(
        signature, _core_cone_partition_signature(system.cone),
    )
    signature = _core_mix_values(signature, V)
    signature = _core_mix_values(signature, system.A)
    signature = _core_mix_values(signature, system.b)
    signature = _core_mix_values(signature, system.c)
    return signature
end

"""Content signature of the current numeric Theta operator (K numeric values)."""
function _core_pattern_theta_signature(pattern::SymmetricCorePattern)
    return _core_mix_values(UInt64(0xcbf29ce484222325), pattern.nzval)
end

"""Block partition of a semantic product/block cone."""
function _core_cone_block_ranges(
    cone::Union{ProductConeLinearization,BlockProductConeLinearization},
)
    return product_cone_block_ranges(cone)
end
_core_cone_block_ranges(::AbstractConeLinearization) = throw(ArgumentError(
    "symmetric core requires a product or block-product cone linearization",
))

"""Structural signature of a semantic cone block partition.

Independent of numeric Theta values.  A changed partition with the same dense
operator must be detected as a static identity change, so this signature is
mixed into the static operator identity and compared at factor/guard/refill.
"""
function _core_cone_partition_signature(
    cone::Union{ProductConeLinearization,BlockProductConeLinearization},
)
    return _core_mix_ranges(
        UInt64(0xcbf29ce484222325), _core_cone_block_ranges(cone),
    )
end
_core_cone_partition_signature(::AbstractConeLinearization) = throw(ArgumentError(
    "symmetric core requires a product or block-product cone linearization",
))

"""Content signature of the Theta numeric values of a semantic cone.

The block partition is mixed in first so a changed partition with the same
operator values (e.g. merging two blocks) fails closed even though the dense
numeric values are unchanged.
"""
function _core_cone_theta_signature(
    cone::Union{ProductConeLinearization,BlockProductConeLinearization},
)
    signature = _core_cone_partition_signature(cone)
    if cone isa ProductConeLinearization
        return _core_mix_values(signature, cone.operator)
    end
    signature = _core_mix_uint(signature, UInt64(length(cone.operators)))
    for operator in cone.operators
        signature = _core_mix_values(signature, operator)
    end
    return signature
end
_core_cone_theta_signature(::AbstractConeLinearization) = throw(ArgumentError(
    "symmetric core requires a product or block-product cone linearization",
))

@inline _core_cache_signature(cache::SparseSymbolicNumericCache) = cache.signature
@inline _core_cache_signature(::AbstractFactorCache) = UInt64(0)

@inline function _core_factor_matches_pattern(
    cache::SparseSymbolicNumericCache, pattern::SymmetricCorePattern,
)
    return length(cache.factor_view.nzval)==length(pattern.nzval) &&
           cache.factor_view.nzval==pattern.nzval
end
@inline _core_factor_matches_pattern(::AbstractFactorCache, ::SymmetricCorePattern) = true

function _core_projection_tolerance(::Type{T}, dimension::Int, scale::T) where {T}
    return T(256 * max(1, dimension)) * eps(one(T)) * max(one(T), scale)
end

function _core_vector_in_range(
    V::AbstractMatrix{T}, vector::AbstractVector{T}, work::AbstractVector{T},
) where {T}
    _hsd_is_identity_basis(V) && return true
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
    pattern::SymmetricCorePattern{T}, cache::FC,
    V::AbstractMatrix{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat,FC<:AbstractFactorCache{T}}
    _validate_core_preconditions(system, V)
    return _symmetric_core_workspace_prevalidated(pattern, cache, V, system)
end

# Internal path for callers that already ran `_validate_core_preconditions`
# before any materialization.  It owns/copies workspace storage only and does
# not repeat Gram/range validation.
function _symmetric_core_workspace_prevalidated(
    pattern::SymmetricCorePattern{T}, cache::FC,
    V::AbstractMatrix{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat,FC<:AbstractFactorCache{T}}
    nr = size(V, 2)
    n = length(system.c)
    m = length(system.b)
    pattern.nr == nr || throw(DimensionMismatch(
        "pattern reduced dimension $(pattern.nr) disagrees with V columns $nr",
    ))
    pattern.m == m || throw(DimensionMismatch(
        "pattern cone dimension $(pattern.m) disagrees with system rows $m",
    ))
    dimension = nr + m
    # A zero-payload identity basis is already immutable and owns no storage;
    # keep it as-is instead of copying a payload that does not exist.  Dense
    # RRQR bases are owned copies so a mutable BigFloat basis cannot be
    # corrupted by the caller.
    V_owned = V isa IdentityRankBasis ? V :
        (V isa SparseMatrixCSC ?
         owned_sparse_copy(T, V; precision_bits=factor_receipt_precision(T)) :
         copy_owned!(alloc_zeros(T, size(V, 1), size(V, 2)), V))
    cr = alloc_zeros(T, nr)
    if _hsd_is_identity_basis(V_owned)
        copy_owned!(cr, system.c)
    else
        for j in 1:n
            for r in 1:nr
                cr[r] += V_owned[j, r] * system.c[j]
            end
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
    operator_signature = _core_static_signature(pattern, V_owned, system)
    return SymmetricCoreWorkspace{
        T,typeof(pattern),FC,typeof(V_owned),typeof(system),
    }(
        pattern, cache, V_owned, system,
        nr, n, m, dimension, cr,
        alloc_zeros(T, dimension), alloc_zeros(T, dimension), alloc_zeros(T, nr),
        alloc_zeros(T, nr), alloc_zeros(T, m), alloc_zeros(T, nr), alloc_zeros(T, m),
        alloc_zeros(T, dimension), alloc_zeros(T, dimension), alloc_zeros(T, dimension),
        alloc_zeros(T, nr), alloc_zeros(T, n), alloc_zeros(T, m), alloc_zeros(T, m),
        zero(T), alloc_zeros(T, m), alloc_zeros(T, n), zero(T),
        _core_newton_residual(T, m, n), original_nzval, row_sums,
        pattern_signature, structure_signature, operator_signature,
        _core_cone_theta_signature(system.cone), system.tau, system.kappa,
        _core_cache_signature(cache), nothing, 0, 0, 0, -1, false,
        0, 0, 0, 0, zero(T), :regular, false, original_scale,
        nothing,
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

"""Current exact symmetric infinity row-sum norm of the live pattern values.

Used *before* synchronization so a new numeric epoch's regularization is
scaled by the new Theta, not by the previously accepted epoch's scale.
"""
function _core_current_original_scale(
    workspace::SymmetricCoreWorkspace{T},
) where {T}
    return _core_original_scale!(
        workspace.row_sums, workspace.pattern.colptr,
        workspace.pattern.rowval, workspace.pattern.nzval,
    )
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
    workspace::SymmetricCoreWorkspace{T};
    system::Union{Nothing,NewtonSystem{T}}=nothing,
) where {T}
    cache = workspace.cache
    SDPX.factor_status(cache) === Fresh || throw(FactorCacheStateError(
        :sync_core_factor_epoch, Fresh, SDPX.factor_status(cache),
    ))
    epoch_system = system === nothing ? workspace.system : system
    pattern = workspace.pattern
    pattern_signature = symmetric_core_signature(pattern)
    structure_signature = _core_structure_signature(pattern)
    operator_signature = _core_static_signature(pattern, workspace.V, epoch_system)
    theta_signature = _core_cone_theta_signature(epoch_system.cone)
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
            "symmetric core static operator changed without a new factor epoch",
        ))
        theta_signature == workspace.theta_signature || throw(ArgumentError(
            "symmetric core Theta numeric values changed without a new factor epoch",
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
    workspace.theta_signature = theta_signature
    workspace.epoch_tau = epoch_system.tau
    workspace.epoch_kappa = epoch_system.kappa
    workspace.cache_signature = cache_signature
    workspace.matrix_epoch = matrix_epoch
    workspace.factor_epoch = factor_epoch
    workspace.homogeneous_epoch = -1
    workspace.factor_receipt = _core_build_factor_receipt(workspace)
    workspace.receipt_build_count += 1
    workspace.synchronized = true
    return workspace
end

"""Build the minimal immutable FactorReceipt for one successful core epoch.

Provider facts only: actual route/provider/scalar/precision/regularization and
the pattern signature.  `proof_valid = false` because a provider receipt is
implementation evidence, never a mathematical certificate; the direction is
accepted only through the frozen five-equation residual and the original-K
refinement gates.
"""
function _core_build_factor_receipt(
    workspace::SymmetricCoreWorkspace{T},
) where {T<:AbstractFloat}
    cache = workspace.cache
    diag = try
        SDPX.factor_diagnostics(cache)
    catch
        nothing
    end
    provider = diag === nothing ? :provider_unknown :
        get(diag, :provider, :provider_unknown)
    kind = diag === nothing ? :unknown : get(diag, :kind, :unknown)
    precision_bits = diag === nothing ? factor_receipt_precision(T) :
        Int(get(diag, :precision_bits, factor_receipt_precision(T)))
    regularization = diag === nothing ? zero(T) :
        T(get(diag, :regularization, zero(T)))
    regularization_kind = iszero(regularization) ? :none : :signed_diagonal
    return FactorReceipt(
        workspace.matrix_epoch,
        workspace.factor_epoch,
        workspace.pattern_signature,
        :symmetric_augmented_core,
        provider,
        T,
        precision_bits,
        regularization,
        regularization_kind,
        :factored,
        zero(T),
        false,
        0,
        0,
    )
end

"""Refactor the symmetric core for a new scaling/matrix epoch.

Validates that the static operator identity (A,b,c,V and the frozen CSC
structure) is unchanged, refills the numeric `Theta` from the semantic cone,
chooses/updates the Float64 signed δ from the current original-K scale,
factors exactly once, syncs the frozen original snapshot, and solves the
homogeneous core once.  MultiFloat/BigFloat use the unregularized pivoted-LDL
cache path unchanged.  Predictor→corrector RHS changes within one epoch are
handled by `solve_core_direction!` without this seam.
"""
function _core_revoke_epoch!(
    workspace::SymmetricCoreWorkspace{T},
) where {T<:AbstractFloat}
    # Transaction start: before any pattern/operator mutation or numeric
    # factor, revoke every acceptance/ownership artifact of the previous
    # epoch.  A failure anywhere in the epoch leaves no usable receipt, no
    # Fresh solve, and no stale homogeneous solution.
    workspace.factor_receipt = nothing
    workspace.synchronized = false
    workspace.homogeneous_epoch = -1
    cache = workspace.cache
    # Revoke numeric validity while retaining prepared capacity and any
    # symbolic structure. Provider-specific methods keep MFLA/BFLA workspaces
    # Prepared and CHOLMOD's symbolic object reusable.
    SDPX.revoke_numeric!(cache)
    return workspace
end

function factorize_symmetric_core_pattern!(
    cache::AbstractFactorCache{T}, pattern::SymmetricCorePattern{T},
    matrix_epoch::Integer,
) where {T<:AbstractFloat}
    return factorize!(cache, materialize_dense(pattern), Int(matrix_epoch))
end

function factor_symmetric_core_epoch!(
    workspace::SymmetricCoreWorkspace{T},
    system::NewtonSystem{T},
    matrix_epoch::Integer,
) where {T<:AbstractFloat}
    cache = workspace.cache
    _core_validate_static_identity(workspace, system)
    _core_revoke_epoch!(workspace)
    try
        _core_refill_from_system!(workspace, system)
        epoch = Int(matrix_epoch)
        if T === Float64
            # Scale δ from the *current* refilled original-K infinity norm.
            # The prior accepted epoch's scale is deliberately not reused:
            # the Theta numeric values may change arbitrarily between epochs.
            current_scale = _core_current_original_scale(workspace)
            delta = T(64) * eps(one(T)) * current_scale
            isfinite(delta) && delta > zero(T) || throw(ArgumentError(
                "symmetric core regularization scale is not usable",
            ))
            SDPX.set_regularization!(cache, delta)
            SDPX.factorize!(
                cache, symmetric_core_lower_sparse(workspace.pattern), epoch,
            )
        else
            factorize_symmetric_core_pattern!(
                cache, workspace.pattern, epoch,
            )
        end
        sync_core_factor_epoch!(workspace; system=system)
        solve_core_homogeneous!(workspace, system)
    catch
        # Fail closed: the epoch never completes, so no receipt, no Fresh
        # solve and no homogeneous seam may survive.
        workspace.factor_receipt = nothing
        workspace.synchronized = false
        workspace.homogeneous_epoch = -1
        rethrow()
    end
    return workspace
end

"""Validate that a system may reuse this workspace: static identity only.

The numeric Theta, tau/kappa and all RHS fields may change per epoch; the
static A,b,c,V and the frozen CSC structure must be identical.
"""
function _core_validate_static_identity(
    workspace::SymmetricCoreWorkspace{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    pattern = workspace.pattern
    symmetric_core_signature(pattern) == workspace.pattern_signature ||
        throw(ArgumentError(
            "symmetric core pattern signature changed across epochs",
        ))
    _core_structure_signature(pattern) == workspace.structure_signature ||
        throw(ArgumentError(
            "symmetric core CSC structure changed across epochs",
        ))
    _core_static_signature(pattern, workspace.V, system) ==
        workspace.operator_signature || throw(ArgumentError(
            "symmetric core static operator changed across epochs",
        ))
    return true
end

"""Refill the owned numeric K from a semantic cone without global Theta.

`ProductConeLinearization` refills per declared dense block from its dense
operator; `BlockProductConeLinearization` writes each block directly into the
frozen theta slots (ownership-safe for BigFloat), never forming a global
`Theta` matrix.
"""
function _core_refill_from_system!(
    workspace::SymmetricCoreWorkspace{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    pattern = workspace.pattern
    cone = system.cone
    if cone isa ProductConeLinearization{T}
        _core_validate_theta_blocks(pattern, cone.operator)
        _core_write_theta_lower!(pattern, cone.operator)
    elseif cone isa BlockProductConeLinearization{T}
        _core_write_block_thetas!(pattern, cone)
    else
        throw(ArgumentError(
            "symmetric core requires a product or block-product cone linearization",
        ))
    end
    return pattern
end

"""Validate a dense Theta against the frozen block layout (no global copy).

Enforces symmetry, finiteness, cross-block zero structure, and dimensions.
"""
function _core_validate_theta_blocks(
    pattern::SymmetricCorePattern{T}, Theta::AbstractMatrix{T},
) where {T}
    size(Theta) == (pattern.m, pattern.m) || throw(DimensionMismatch(
        "symmetric core Theta dimension $(size(Theta)) disagrees with frozen " *
        "cone dimension $(pattern.m)",
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

"""Write only the Theta lower-triangle slots from a dense operator.

The structural zero x diagonal and the static Ar values are left untouched
(Ar depends only on the frozen A and V).  Each BigFloat store is an
independent object.
"""
function _core_write_theta_lower!(
    pattern::SymmetricCorePattern{T}, Theta::AbstractMatrix{T},
) where {T}
    nzval = pattern.nzval
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
    slot_index == length(pattern.theta_slots) || throw(ArgumentError(
        "symmetric core theta slot coverage mismatch",
    ))
    return pattern
end

"""Write each declared block's lower triangle into the frozen theta slots.

Never materializes a global `Theta`; each BigFloat store is an independent
object.  Cross-block zeros are structural in the pattern and are enforced by
the cone constructor.
"""
function _core_write_block_thetas!(
    pattern::SymmetricCorePattern{T}, cone::BlockProductConeLinearization{T},
) where {T}
    length(cone.block_ranges) == length(pattern.block_ranges) || throw(ArgumentError(
        "symmetric core block range count changed across epochs",
    ))
    nzval = pattern.nzval
    slot_index = 0
    for (index, rows) in enumerate(cone.block_ranges)
        rows == pattern.block_ranges[index] || throw(ArgumentError(
            "symmetric core block range changed across epochs",
        ))
        block = cone.operators[index]
        length(rows) == size(block, 1) == size(block, 2) || throw(ArgumentError(
            "symmetric core block operator dimension mismatch",
        ))
        all(isfinite, block) || throw(ArgumentError(
            "symmetric core block Theta contains non-finite data",
        ))
        @inbounds for local_column in 1:length(rows)
            for local_row in local_column:length(rows)
                block[local_row, local_column] == block[local_column, local_row] ||
                    throw(ArgumentError(
                        "symmetric core block Theta is not symmetric",
                    ))
            end
        end
        for column in first(rows):last(rows)
            for row in column:last(rows)
                slot_index += 1
                value = -block[row - first(rows) + 1, column - first(rows) + 1]
                _core_store_owned!(nzval, pattern.theta_slots[slot_index], value)
            end
        end
    end
    slot_index == length(pattern.theta_slots) || throw(ArgumentError(
        "symmetric core block refill slot coverage mismatch",
    ))
    return pattern
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
When `phase_timings !== nothing` the wall time of this function is added
to its `refinement_seconds` bucket (review slice 1 telemetry).
"""
function _core_refine!(
    workspace::SymmetricCoreWorkspace{T},
    x::AbstractVector{T},
    rhs::AbstractVector{T},
    phase_timings::Union{Nothing,ProductHSDPhaseTimings}=nothing,
) where {T}
    t_refine = time_ns()
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
    previous <= residual_floor && begin
        phase_timings === nothing || (phase_timings.refinement_seconds +=
            Float64(time_ns() - t_refine) * 1.0e-9)
        return (x, 0)
    end
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
    phase_timings === nothing || (phase_timings.refinement_seconds +=
        Float64(time_ns() - t_refine) * 1.0e-9)
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
    V = workspace.V
    if _hsd_is_identity_basis(V)
        copy_owned!(dr, system.rhs.dual_affine)
        return dr
    end
    zero_owned!(dr)
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
    # Static identity only: A,b,c,V and the frozen structure.  The numeric
    # Theta/tau/kappa may change only through a new `factor_symmetric_core_epoch!`,
    # which freezes a fresh theta_signature; RHS fields may change freely within
    # one factor epoch.
    operator_signature_now = _core_static_signature(
        workspace.pattern, workspace.V, system,
    )
    operator_signature_now == workspace.operator_signature || throw(ArgumentError(
        "symmetric core static operator changed; only RHS may change",
    ))
    # The live numeric K buffer must still equal the frozen original snapshot:
    # any refill/Theta mutation without a new factor epoch is stale.
    workspace.pattern.nzval == workspace.original_nzval || throw(ArgumentError(
        "symmetric core Theta numeric values changed without a new factor epoch",
    ))
    # The Theta signature is taken from the semantic cone of the passed
    # system.  Within one factor epoch only the RHS fields may change; a
    # changed cone Theta (even before any refill) is rejected.  The
    # pattern.nzval == original_nzval check above independently rejects an
    # external mutation of the owned K buffer.
    theta_signature_now = _core_cone_theta_signature(system.cone)
    theta_signature_now == workspace.theta_signature || throw(ArgumentError(
        "symmetric core Theta numeric values changed within one factor epoch",
    ))
    tau_kappa_changed = workspace.epoch_tau != system.tau ||
                        workspace.epoch_kappa != system.kappa
    tau_kappa_changed && throw(ArgumentError(
        "symmetric core tau/kappa changed without a new factor epoch",
    ))
    cache_signature_now = _core_cache_signature(cache)
    (cache_signature_now == workspace.cache_signature ||
     cache_signature_now == UInt64(0) || workspace.cache_signature == UInt64(0)) ||
        throw(ArgumentError(
            "symmetric core factor signature changed after synchronization",
        ))
    return workspace
end

"""Solve the homogeneous core at most once for the synchronized factor epoch.

`system` should be the epoch system whose `b` defines the homogeneous RHS; it
defaults to the workspace's frozen static system.  Only `b`/`c` enter the
homogeneous RHS, and both are static across epochs.
"""
function solve_core_homogeneous!(
    workspace::SymmetricCoreWorkspace{T},
    system::NewtonSystem{T}=workspace.system,
) where {T}
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

"""Solve one variable RHS in place; returns (candidate, residual, dtau).

Internal production/raw path: writes `workspace.dx/dy/ds/dkappa` and records
`workspace.last_dtau`.  The returned `candidate` references workspace-owned
buffers; callers that need independent snapshots must use the public wrapper.
Counters (`variable_solves`, `directions`, `refinements`) are updated here so
both the raw production path and the snapshot wrapper share the same accounting.
"""
function _core_solve_raw!(
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
    _core_refine!(
        workspace, workspace.sol_core, workspace.rhs_core,
        workspace.phase_timings,
    )
    _core_split!(workspace.wx, workspace.wy, workspace.sol_core, nr,
        workspace.m)

    # Scalar recovery.
    eta_w = zero(T)
    eta_w_work = zero(T)
    @inbounds for r in 1:nr
        term = workspace.cr[r] * workspace.wx[r]
        eta_w += term
        eta_w_work += abs(term)
    end
    @inbounds for i in 1:workspace.m
        term = system.b[i] * workspace.wy[i]
        eta_w += term
        eta_w_work += abs(term)
    end
    denominator, denominator_work = _core_denominator(workspace, system)
    numerator = system.rhs.tau_kappa - system.tau *
                (system.rhs.homogeneous_gap + eta_w)
    numerator_work = abs(system.rhs.tau_kappa) +
        abs(system.tau) * (abs(system.rhs.homogeneous_gap) + eta_w_work)
    classification = classify_scalar_closure(
        denominator, numerator;
        denominator_work=denominator_work,
        numerator_work=numerator_work,
    )
    classification === :insufficient_precision && throw(ArgumentError(
        "symmetric core scalar closure is non-finite",
    ))
    classification === :incompatible_singular && throw(ArgumentError(
        "symmetric core scalar closure is incompatible rank-deficient",
    ))
    dtau = scalar_closure_resolution(classification, denominator, numerator)
    workspace.denominator = denominator
    workspace.scalar_closure = classification
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
    workspace.dy_gauged = false
    # In the exact compatible scalar gauge, dy=0 is a deterministic valid
    # dual representative when the frozen dual RHS is exactly zero.  This is
    # not a tolerance relaxation: the unchanged five-equation gate below
    # rejects the reconstructed direction if any other equation disagrees.
    if scalar_closure_zero_dual_gauge(
        classification, dtau, system.rhs.dual_affine,
    )
        @inbounds for i in 1:workspace.m
            _core_store_owned!(workspace.dy, i, zero(T))
        end
        workspace.dy_gauged = true
    end
    # dx = V * dxr.
    if _hsd_is_identity_basis(workspace.V)
        copy_owned!(workspace.dx, workspace.dxr)
    else
        zero_owned!(workspace.dx)
        @inbounds for r in 1:nr
            for j in 1:workspace.n
                workspace.dx[j] += workspace.V[j, r] * workspace.dxr[r]
            end
        end
    end
    # ds = p - A*dx + b*dtau (frozen primal affine equation).
    mul!(workspace.ds, system.A, workspace.dx)
    for i in 1:workspace.m
        _core_store_owned!(
            workspace.ds, i,
            system.rhs.primal_affine[i] - workspace.ds[i] +
            system.b[i] * dtau,
        )
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
    workspace.last_dtau = dtau
    candidate = NewtonDirection(
        workspace.dx, workspace.dy, workspace.ds, dtau, workspace.dkappa,
    )
    newton_residual!(workspace.residual, system, candidate)
    workspace.variable_solves += 1
    workspace.directions += 1
    return (candidate, workspace.residual, dtau)
end

"""Solve one variable RHS and recover an independent direction snapshot."""
function solve_core_direction!(
    workspace::SymmetricCoreWorkspace{T}, system::NewtonSystem{T},
) where {T}
    candidate, residual, dtau = _core_solve_raw!(workspace, system)
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

# -------------------------------------------------------------------
# State-owned block preparation (C7.1b): allocate pattern, per-block
# operators, cone RHS, and a semantic BlockProduct linearization from
# canonical cone block ranges/sizes without any global m×m Theta, and
# prepare (but do not factor) the provider cache.  The prepared
# workspace is numeric-factor-free: factor_epoch/homogeneous/variable
# stay 0.  BigFloat per-block objects are ownership-safe.
# -------------------------------------------------------------------

"""Total conservative bytes for a prepared symmetric core + block Theta.

Counts the sparse lower-triangle core values, the dense factor storage, and
every per-block `Theta` matrix, with each term saturated at `typemax(Int)`.
Negative dimensions/sizes are rejected rather than masked to zero, because a
masked negative could approve an allocation that must fail closed.  A
saturated `typemax(Int)` result means the estimate is not a valid upper bound
and the caller must treat the route as ineligible.
"""
function symmetric_core_state_prepare_bytes(
    ::Type{T}, dimension::Integer, block_sizes::AbstractVector{Int};
    ar_nnz::Integer=0,
    variable_dimension::Integer=0,
    basis_nnz::Integer=0,
    canonical_nnz::Integer=0,
) where {T<:AbstractFloat}
    dimension < 0 && throw(ArgumentError(
        "symmetric core state dimension must be nonnegative",
    ))
    dimension > typemax(Int) && return typemax(Int)
    d = Int(dimension)
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    block_theta = 0
    for block_size in block_sizes
        block_size < 0 && throw(ArgumentError(
            "symmetric core state block sizes must be nonnegative",
        ))
        bs = Int(block_size)
        block_theta = saturating_sum_bytes(
            block_theta, saturating_bytes(scalar_bytes, bs, bs),
        )
    end
    ar_nnz >= 0 && variable_dimension >= 0 && basis_nnz >= 0 &&
    canonical_nnz >= 0 || throw(ArgumentError(
        "symmetric core structural counts must be nonnegative",
    ))
    core = if T === Float64
        cone_dimension = sum(block_sizes; init=0)
        reduced_dimension = d - cone_dimension
        reduced_dimension >= 0 || throw(DimensionMismatch(
            "symmetric core dimension is smaller than its cone blocks",
        ))
        theta_lower = 0
        for block_size in block_sizes
            theta_lower = saturating_sum_bytes(
                theta_lower, div(block_size * (block_size + 1), 2),
            )
        end
        structural_nnz = saturating_sum_bytes(
            reduced_dimension, Int(ar_nnz), theta_lower,
        )
        # Original/pattern/factor-view snapshots plus a conservative CHOLMOD
        # symbolic+numeric fill allowance tied to the actual frozen pattern.
        csc_bytes = saturating_sum_bytes(
            saturating_bytes(scalar_bytes + sizeof(Int), structural_nnz),
            saturating_bytes(sizeof(Int), d + 1),
        )
        factor_and_snapshots = saturating_bytes(32, csc_bytes)
        vectors = saturating_bytes(24, scalar_bytes, d)
        basis_bytes = saturating_sum_bytes(
            saturating_bytes(scalar_bytes + sizeof(Int), Int(basis_nnz)),
            saturating_bytes(sizeof(Int), Int(variable_dimension) + 1),
        )
        canonical_bytes = saturating_bytes(
            scalar_bytes + sizeof(Int), Int(canonical_nnz),
        )
        _workspace_estimate_with_margin(
            saturating_sum_bytes(
                factor_and_snapshots, vectors, basis_bytes, canonical_bytes,
            ),
            1,
        )
    else
        symmetric_core_dense_bytes(T, d)
    end
    return saturating_sum_bytes(core, block_theta)
end

"""Dimension-only symmetric-core state preflight.

Validates arithmetic/provider availability and the conservative memory upper
bound using only the dimensions, canonical block sizes, the scalar type, and
the supplied budget/RSS facts.  It allocates nothing.  It throws fail-closed
on a negative or out-of-range dimension, an absent provider, an unknown
memory fact, an over-budget request, or a saturated (overflowing) byte
estimate.  `dimension` must already be a checked/saturating `nr + m` sum; the
caller computes it from base facts before this gate.
"""
function symmetric_core_state_preflight(
    ::Type{T},
    dimension::Integer,
    block_sizes::AbstractVector{Int},
    precision_bits::Int,
    memory_limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer};
    ar_nnz::Integer=0,
    variable_dimension::Integer=0,
    basis_nnz::Integer=0,
    canonical_nnz::Integer=0,
) where {T<:AbstractFloat}
    dimension < 0 && throw(ArgumentError(
        "symmetric core state dimension must be nonnegative",
    ))
    dimension > typemax(Int) && throw(ArgumentError(
        "symmetric core state dimension exceeds the addressable range",
    ))
    T === Float64 || symmetric_core_provider_available(T, precision_bits)
    estimate = symmetric_core_state_prepare_bytes(
        T, dimension, block_sizes;
        ar_nnz, variable_dimension, basis_nnz, canonical_nnz,
    )
    # A saturated estimate cannot certify an upper bound, even against a
    # typemax(Int) budget (which would otherwise compare as eligible).
    estimate >= typemax(Int) && throw(ArgumentError(
        "symmetric core state byte estimate saturated; route ineligible",
    ))
    eligibility = conservative_memory_upper_bound_eligibility(
        estimate, memory_limit_bytes, current_rss_bytes,
    )
    eligibility.eligible || throw(ArgumentError(
        "symmetric core state ineligible: $(eligibility.reason)",
    ))
    return nothing
end

function _build_float64_core_cache(
    pattern::SymmetricCorePattern{Float64},
    symbolic_epoch::Integer, regularization::Real,
)
    isfinite(regularization) && regularization>=0 || throw(ArgumentError(
        "symmetric core Float64 regularization must be finite and nonnegative",
    ))
    k=symmetric_core_lower_sparse(pattern)
    dsigns=symmetric_core_dsigns(pattern)
    disconnected=DisconnectedLDLTCache(
        k,dsigns;symbolic_epoch,regularization,max_size=4,
    )
    disconnected===nothing || return disconnected
    requirements=SparseSymbolicRequirements(k;
        symbolic_epoch=Int(symbolic_epoch),dsigns,
        regularization=Float64(regularization))
    cache=SparseSymbolicNumericCache{Float64}()
    _prepare_owned_requirements!(cache,requirements)
    return cache
end

"""Build a numeric-factor-free prepared state workspace.

Allocates an owned `BlockProductConeLinearization` from ordered canonical
block ranges and per-block owned operator/RHS storage (no global m×m
Theta), freezes the core pattern, and prepares the provider cache only.
`memory_limit_bytes`/`current_rss_bytes` must be known; an unknown or
over-budget request fails closed before allocation.  The returned
workspace is unsynchronized and unfactored (cache Prepared,
factor_epoch/homogeneous_solves/variable_solves/directions == 0).
"""
function prepare_symmetric_core_state(
    system::NewtonSystem{T},
    V::AbstractMatrix{T},
    block_ranges::AbstractVector{<:UnitRange{Int}},
    block_sizes::AbstractVector{Int},
    precision_bits::Int,
    memory_limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer},
    regularization::Real;
    symbolic_epoch::Integer=0,
    take_cone_ownership::Bool=false,
) where {T<:AbstractFloat}
    length(block_ranges) == length(block_sizes) || throw(ArgumentError(
        "symmetric core state block ranges/sizes counts disagree",
    ))
    m = length(system.b)
    validate_product_cone_block_ranges(m, block_ranges)
    for (index, rows) in enumerate(block_ranges)
        length(rows) == block_sizes[index] || throw(DimensionMismatch(
            "symmetric core block size $(block_sizes[index]) disagrees " *
            "with rows $rows",
        ))
    end
    # Dimension-only preflight (provider + memory, checked/saturating nr+m)
    # runs before ANY allocation below.
    dimension = saturating_sum_bytes(size(V, 2), m)
    symmetric_core_state_preflight(
        T, dimension, block_sizes, precision_bits,
        memory_limit_bytes, current_rss_bytes,
    )
    # Range/isometry preconditions before materialization.
    _validate_core_preconditions(system, V)

    # State-owned per-block Theta operators and cone RHS. The internal HSD
    # caller may transfer the just-created semantic cone; public/default
    # preparation retains the isolating copy behavior.
    cone = if take_cone_ownership
        system.cone isa BlockProductConeLinearization{T} || throw(ArgumentError(
            "owned symmetric-core cone must be block-product",
        ))
        system.cone.block_ranges==block_ranges || throw(ArgumentError(
            "owned symmetric-core cone block ranges disagree",
        ))
        system.cone
    else
        operators = Matrix{T}[]
        sizehint!(operators,length(block_ranges))
        for rows in block_ranges
            dimension_block = length(rows)
            push!(operators, alloc_zeros(T, dimension_block, dimension_block))
        end
        corrector_rhs = alloc_zeros(T, m)
        BlockProductConeLinearization{T}(
            operators, corrector_rhs,
            UnitRange{Int}[rows for rows in block_ranges],
        )
    end

    pattern = _symmetric_core_pattern_from_validated(system, V)
    pattern.block_ranges == block_ranges || throw(ArgumentError(
        "symmetric core state block ranges drifted after pattern build",
    ))
    cache = if T === Float64
        _build_float64_core_cache(pattern,symbolic_epoch,regularization)
    else
        # QDLDL-backed sparse signed-LDL provider is NOT used for the
        # symmetric augmented core: K = [0 Ar'; Ar -Theta] stores a structural
        # zero in every reduced-x diagonal, which violates QDLDL's
        # quasi-definite precondition (and MFLA/BFLA adapters deliberately
        # disable dynamic regularization).  Fabricating a positive x diagonal
        # would change the Newton operator, so the dense MFLA/BFLA LDL stays
        # the sole high-precision path; QDLDL remains an independently
        # usable optional provider (see SparseQDLDLCache) for operators that
        # are already strictly quasi-definite.
        build_symmetric_core_ldlt_cache(
            T, pattern, precision_bits, memory_limit_bytes,
            current_rss_bytes,
        )
    end
    # Build the state-owned block-cone NewtonSystem first so the workspace
    # type parameter is the block-cone system from construction (no
    # concrete-field reassignment) and no global Theta is ever referenced.
    block_system = take_cone_ownership ? system : NewtonSystem(
        system.A, system.b, system.c, cone,
        system.tau, system.kappa, system.rhs,
    )
    workspace = _symmetric_core_workspace_prevalidated(
        pattern, cache, V, block_system,
    )
    return workspace
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
    # A structurally verified sparse identity is an exact isometry whose range
    # is the full variable space.  Avoid O(n²) scalar sparse indexing and the
    # O(n) projection scratch on the production full-rank sparse path.
    _hsd_is_identity_basis(V) && return nothing
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
    ar = _hsd_is_identity_basis(V) ? SparseArrays.sparse(system.A) :
         SparseArrays.sparse(system.A * V)
    pattern = SymmetricCorePattern{T}(
        ar, ranges, [:dense_lower for _ in ranges],
    )
    _core_write_ar!(pattern, ar)
    cone = system.cone
    if cone isa ProductConeLinearization{T}
        _core_validate_theta_blocks(pattern, cone.operator)
        _core_write_theta_lower!(pattern, cone.operator)
    elseif cone isa BlockProductConeLinearization{T}
        _core_write_block_thetas!(pattern, cone)
    else
        throw(ArgumentError(
            "symmetric core requires a product or block-product cone linearization",
        ))
    end
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
        cache=_build_float64_core_cache(pattern,symbolic_epoch,regularization)
        factorize!(cache,symmetric_core_lower_sparse(pattern),Int(matrix_epoch))
        cache
    else
        cache = build_symmetric_core_ldlt_cache(
            T, pattern, precision_bits, memory_limit_bytes, current_rss_bytes,
        )
        factorize!(cache, materialize_dense(pattern), Int(matrix_epoch))
        cache
    end
    workspace = _symmetric_core_workspace_prevalidated(
        pattern, cache, V, system,
    )
    sync_core_factor_epoch!(workspace)
    solve_core_homogeneous!(workspace)
    return workspace
end
