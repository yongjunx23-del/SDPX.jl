
#=====================================================================#
#=====================================================================#
#    INTERNAL EXPERIMENTAL sparse symmetric core (R3 bounded).
#
#    BigFloat-only, small LP systems (scalar orthant rows only), identity
#    coordinates, explicit caller-owned precision-matched positive finite δ.
#    NOT native routing: the default path, public Settings and Newton
#    equations are unchanged. Natural provider ordering is explicit opt-in;
#    no sparse-scalability or memory-bound qualification is implied.
#
#    The seams below reuse the existing machinery without duplicating
#    numerical kernels and without changing any math or tolerance:
#      1. `prepare_experimental_sparse_core_state` refuses unavailable
#         memory admission before pattern/workspace/provider construction.
#         The separately named private `_research_prepare_*` primitive
#         constructs only UNADMITTED numerical-research workspaces after
#         checking the original-A witness, LP scope and semantic SPD Theta.
#         There is no fallback to research construction from admission.
#         SOC and other cones remain unsupported.
#      2. (in `src/factor_cache/routes/experimental_sparse_core.jl`) the thin
#         core-specific wrapper: independent frozen map/diagonal/sign/shift
#         authority, shifted upper factor values, independent original
#         snapshot, last-successful epoch evidence (survives revocation),
#         exact owned static snapshots (original A, b, c), genuinely sparse
#         `factorize_symmetric_core_pattern!` with same-epoch conflict
#         refusal, specialized `_core_factor_matches_pattern` /
#         `_core_cache_signature`, truthful provider/precision/shift
#         diagnostics with `proof_valid=false`;
#      3. `factor_experimental_sparse_core_epoch!` — the whole epoch
#         (static validation, exact static binding, refill, SPD admission,
#         factorize, sync, homogeneous solve) inside one revocation
#         transaction (existing `_core_refine!` targets and two-correction
#         caps unchanged).  `solve_experimental_sparse_core_direction!`
#         runs the complete solve/acceptance inside the same transaction
#         with no manual cleanup.
#
#    The acceptance predicate `experimental_sparse_core_accept` is genuinely
#    shared production logic: every residual/work number comes from the
#    `_shared_*` acceptance helpers in `src/hsd/product_cone_hsd.jl` that the
#    production `_product_hsd_newton_residual_ok` gate calls itself.  Only
#    the input sourcing differs (standalone vectors instead of state
#    vectors).  No new tolerance is introduced.
#=====================================================================#

"""Experimental admission bound: larger systems are unsupported, not slower."""
const EXPERIMENTAL_SPARSE_CORE_MAX_DIMENSION = 64

"""
    ExperimentalSparseCoreContext

INTERNAL EXPERIMENTAL numerical-research facts for one workspace: declared
cone families, checked witness rows/orientation, shift and precision.
`memory_estimate_bytes` is a diagnostic research estimate, NOT an admission
proof or authorization. The enforced direction caller checks the acceptance
families and δ identity; resource admission is separately unavailable.
"""
struct ExperimentalSparseCoreContext
    families::Vector{Symbol}
    witness_rows::Vector{Int}
    witness_orientation::Symbol
    delta::BigFloat
    precision_bits::Int
    memory_estimate_bytes::Int
end

"""Validate declared LP families against the frozen block structure.

LP-only admission: every block must be declared `:lp` and hold exactly one
scalar orthant row.  `:soc` (and every other family) is UNSUPPORTED and
fails closed here: production SOC acceptance requires certified runtime
scaling context (SOC bounds, `w`/determinant work) that a standalone
dense-Theta block cannot supply, so calling an arbitrary SPD block "SOC"
would not be the production gate.  Unsupported families are rejected,
never approximated.
"""
function _experimental_validate_families(
    block_ranges::AbstractVector{<:UnitRange{Int}},
    cone_families::AbstractVector{Symbol},
)
    length(block_ranges) == length(cone_families) || throw(ArgumentError(
        "experimental sparse core cone family count " *
        "$(length(cone_families)) disagrees with block count " *
        "$(length(block_ranges))",
    ))
    for (index, rows) in enumerate(block_ranges)
        family = cone_families[index]
        family === :lp || throw(ArgumentError(
            "experimental sparse core supports only :lp families " *
            "(got $(family) for block $index); :soc and other cones " *
            "need production runtime scaling context and are unsupported",
        ))
        length(rows) == 1 || throw(ArgumentError(
            "experimental sparse core :lp block $index must be a " *
            "scalar orthant row, got $rows",
        ))
    end
    return true
end

"""
Check the caller-specified rows of the ORIGINAL `A` for a triangular square
minor with finite nonzero diagonal, reading actual coefficients (structural
zeros count as exact zeros).  Accepts lower- or upper-triangular form in the
natural column order and returns the orientation.  A triangular minor with
nonzero diagonal proves full column rank independently of any shifted
inertia; identity coordinates waive reduction machinery, never this proof.
Anything else — wrong row count, out-of-range/duplicated rows, zero or
non-finite diagonal, non-triangular minor — fails closed.
"""
function _experimental_check_witness(
    A::AbstractMatrix{BigFloat}, witness_rows::AbstractVector{<:Integer},
)
    m, n = size(A)
    rows = Int[Int(row) for row in witness_rows]
    length(rows) == n || throw(ArgumentError(
        "experimental sparse core witness needs exactly $n rows for a " *
        "square minor, got $(length(rows))",
    ))
    for row in rows
        1 <= row <= m || throw(ArgumentError(
            "experimental sparse core witness row $row out of 1:$m",
        ))
    end
    length(unique(rows)) == n || throw(ArgumentError(
        "experimental sparse core witness rows must be distinct",
    ))
    @inbounds for i in 1:n
        diagonal = A[rows[i], i]
        isfinite(diagonal) || throw(ArgumentError(
            "experimental sparse core witness diagonal ($i, $i) is non-finite",
        ))
        iszero(diagonal) && throw(ArgumentError(
            "experimental sparse core witness diagonal ($i, $i) is zero",
        ))
    end
    lower_ok = true
    upper_ok = true
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        value = A[rows[i], j]
        isfinite(value) || throw(ArgumentError(
            "experimental sparse core witness entry ($i, $j) is non-finite",
        ))
        if j > i
            iszero(value) || (lower_ok = false)
        else
            iszero(value) || (upper_ok = false)
        end
    end
    lower_ok || upper_ok || throw(ArgumentError(
        "experimental sparse core witness rows $rows do not form a " *
        "triangular minor of the original A",
    ))
    return lower_ok ? :lower : :upper
end

"""Unpivoted LDLᵀ SPD admission check on one owned scratch block.

Copies `block` into `scratch` (sized at least `n×n`), verifies symmetry and
finiteness, then runs Doolittle LDL without pivoting at the ambient BigFloat
precision.  Returns `true` only when every pivot is finite and strictly
positive — a sufficient semantic-SPD certificate for the admission gate.
Near-singular blocks may fail this gate; that is conservative rejection, not
a conditioning claim.
"""
function _experimental_block_spd!(
    scratch::Matrix{BigFloat}, block::AbstractMatrix{BigFloat},
)
    n = size(block, 1)
    size(block, 2) == n || return false
    size(scratch, 1) >= n && size(scratch, 2) >= n || throw(ArgumentError(
        "experimental sparse core SPD scratch is smaller than the block",
    ))
    @inbounds for j in 1:n, i in 1:n
        value = block[i, j]
        isfinite(value) || return false
        block[i, j] == block[j, i] || return false
        scratch[i, j] = MA.mutable_copy(value)
    end
    pivots = Vector{BigFloat}(undef, n)
    @inbounds for k in 1:n
        pivots[k] = MA.mutable_copy(scratch[k, k])
    end
    @inbounds for k in 1:n
        pivot = scratch[k, k]
        for s in 1:(k - 1)
            pivot -= scratch[k, s]^2 * pivots[s]
        end
        isfinite(pivot) && pivot > zero(BigFloat) || return false
        pivots[k] = pivot
        for i in (k + 1):n
            value = scratch[i, k]
            for s in 1:(k - 1)
                value -= scratch[i, s] * scratch[k, s] * pivots[s]
            end
            scratch[i, k] = value / pivot
        end
    end
    return true
end

"""Block operators of a product/block-product cone (read-only)."""
function _experimental_cone_blocks(
    cone::ProductConeLinearization{BigFloat},
)
    return AbstractMatrix{BigFloat}[
        @view(cone.operator[rows, rows]) for rows in cone.block_ranges
    ]
end
function _experimental_cone_blocks(
    cone::BlockProductConeLinearization{BigFloat},
)
    return AbstractMatrix{BigFloat}[operator for operator in cone.operators]
end

"""Require finite positive-definite semantic Theta blocks at this epoch.

Reads the same cone operators the refill wrote into the pattern, so the
check binds the about-to-be-factored values.  A failing block throws
fail-closed before any numeric factor is attempted.
"""
function _experimental_require_spd_theta!(
    workspace::SymmetricCoreWorkspace{BigFloat},
    system::NewtonSystem{BigFloat},
)
    cone = system.cone
    cone isa Union{ProductConeLinearization{BigFloat},
                   BlockProductConeLinearization{BigFloat}} ||
        throw(ArgumentError(
            "experimental sparse core requires a product or block-product " *
            "cone linearization",
        ))
    validate_product_cone_block_ranges(length(system.b), cone.block_ranges)
    blocks = _experimental_cone_blocks(cone)
    max_block = maximum(length(rows) for rows in cone.block_ranges; init=0)
    scratch = alloc_zeros(BigFloat, max_block, max_block)
    for (index, block) in enumerate(blocks)
        _experimental_block_spd!(scratch, block) || throw(ArgumentError(
            "experimental sparse core Theta block $index is not " *
            "positive-definite",
        ))
    end
    return workspace
end

"""
Candidate owned-storage inventory used before experimental allocations.
A complete simultaneous-live upper bound is NOT established: ordering and
solve allowances plus returned-direction/acceptance temporaries still need
reconciliation. Gate arithmetic alone does not prove storage sufficiency;
this remains an integration blocker for the research-only route. JIT, GC
fragmentation, BLAS and total-process RSS are outside this inventory.

Named allocation shapes and provisional allowances below include:
the frozen pattern (`SymmetricCorePattern`: colptr/rowval/nzval, admitted-Ar
structure copies, slot maps), the wrapper (upper CSC copy, slot map and
diagonal locations plus frozen copies, three unshifted snapshots, exact
static A/b/c snapshots), the direction workspace vectors counted exactly off
`SymmetricCoreWorkspace` fields (`6d + 5nr + 8m + 3n` BigFloats plus 4
scalars), the admitted block structure, and the provider chain inventoried
field by field:

- inner `SparseQDLDLCache`: structural colptr/rowval copies, D-sign copy;
- BFLA `BFLASparseLDLCache`: owned `matrix` copy, owned `factored_values`
  copy, value-index list, D-sign copy, frozen colptr/rowval copies;
- QDLDL `QDLDLWorkspace` (order `d`, upper nonzeros `nnz`, `Ti == Int`):
  `etree`/`Lnz`/`iwork(3d)`/`bwork`/`fwork`, `Lp`/`Li`/`Lx` with the `L` fill
  bounded by the dense worst case `d^2` (the single estimated fill term, and
  it dominates realistic AMD fill at admitted dimensions), `D`/`Dinv`,
  `triuA` (exactly `nnz` permuted entries), `AtoPAPt` (exactly `nnz`
  mappings), regularization scalars;
- QDLDL `QDLDLFactorisation`: AMD `perm`/`iperm` (exactly `d` each on this
  path); `L`/`Dinv` share workspace storage (counted once, noted here);
- construction transients (peak-counted, freed afterwards): the QDLDL input
  `deepcopy`, the symmetric-permutation scratch (`Pr`/`Pc`/`Pv`/`AtoPAPt`/`P`
  output), an AMD ordering worst-case dense-graph allowance, and BFLA owned
  copies;
- per-factorize transient (owned value copy) and per-solve transient
  (triangular-pass scratch allowance);
- the admission context (families, witness rows, frozen shift, estimate).

BigFloat elements count at `_element_storage_bytes(BigFloat)`, itself an
upper bound over MPFR limbs plus allocator rounding.  Saturating arithmetic:
a saturated estimate cannot certify an upper bound and the caller must treat
it as ineligible.
"""
function experimental_sparse_core_bytes(
    nr::Integer, m::Integer, ar_nnz::Integer,
    block_sizes::AbstractVector{Int},
)
    nr >= 0 && m >= 0 && ar_nnz >= 0 || throw(ArgumentError(
        "experimental sparse core structural counts must be nonnegative",
    ))
    d = Int(nr) + Int(m)
    scalar = ExtendedPrecisionBLAS._element_storage_bytes(BigFloat)
    INTSIZE = sizeof(Int)
    theta_lower = 0
    max_block = 0
    for block_size in block_sizes
        block_size < 0 && throw(ArgumentError(
            "experimental sparse core block sizes must be nonnegative",
        ))
        bs = Int(block_size)
        max_block = max(max_block, bs)
        theta_lower = saturating_sum_bytes(
            theta_lower, saturating_bytes(bs, bs + 1) ÷ 2,
        )
    end
    lower_nnz = saturating_sum_bytes(Int(nr), Int(ar_nnz), theta_lower)
    upper_nnz = lower_nnz  # transpose bijection: identical entry count
    # One CSC triangle shape (values + row indices + column pointers).
    csc_shape(nnz) = saturating_sum_bytes(
        saturating_bytes(scalar, nnz),
        saturating_bytes(nnz, INTSIZE),
        saturating_bytes(d + 1, INTSIZE),
    )
    lower_csc = csc_shape(lower_nnz)
    upper_csc = csc_shape(upper_nnz)
    value_bytes = saturating_bytes(scalar, lower_nnz)
    # Frozen pattern extras beyond the CSC triangle: admitted-Ar structure
    # copies and the Ar/Theta/x-diagonal slot maps.
    pattern_extra = saturating_sum_bytes(
        saturating_bytes(Int(ar_nnz) + Int(nr) + 1, INTSIZE),
        saturating_bytes(Int(ar_nnz), INTSIZE),
        saturating_bytes(lower_nnz, INTSIZE),
        saturating_bytes(Int(nr), INTSIZE),
    )
    # Wrapper owned storage: slot map, diagonal locations, Ar slot mapping
    # (each live plus frozen), three unshifted snapshots, frozen shift.
    wrapper_maps = saturating_sum_bytes(
        saturating_bytes(2, lower_nnz, INTSIZE),
        saturating_bytes(2, d, INTSIZE),
        saturating_bytes(2, Int(ar_nnz), INTSIZE),
    )
    wrapper_snapshots = saturating_bytes(3, value_bytes)
    # Exact static snapshots: original A (values + structure), b, c.
    static_owned = saturating_sum_bytes(
        saturating_bytes(scalar, Int(ar_nnz)),
        saturating_bytes(Int(ar_nnz) + Int(nr) + 1, INTSIZE),
        saturating_bytes(scalar, Int(m)),
        saturating_bytes(scalar, Int(nr)),
        saturating_bytes(scalar),
    )
    # Provider chain, inventoried field by field (d = order, nnz = upper
    # nonzeros, L fill dense worst case d^2).
    qdldl_workspace = saturating_sum_bytes(
        saturating_bytes(6, d, INTSIZE),      # etree + Lnz + iwork(3d)
        saturating_bytes(d, 1),               # bwork booleans
        saturating_bytes(scalar, d),          # fwork
        saturating_bytes(d + 1, INTSIZE),     # Lp
        saturating_bytes(d, d, INTSIZE),      # Li, dense worst case
        saturating_bytes(scalar, d, d),       # Lx, dense worst case
        saturating_bytes(2, scalar, d),       # D + Dinv
        csc_shape(upper_nnz),                 # triuA permuted copy
        saturating_bytes(upper_nnz, INTSIZE), # AtoPAPt mappings
        saturating_bytes(2, scalar),          # regularization scalars
        saturating_bytes(INTSIZE),            # regularize counter
    )
    qdldl_factor = saturating_sum_bytes(
        saturating_bytes(2, d, INTSIZE),      # AMD perm + iperm
    )
    bfla_cache = saturating_sum_bytes(
        csc_shape(upper_nnz),                 # owned matrix copy
        value_bytes,                          # factored-values copy
        saturating_bytes(upper_nnz, INTSIZE), # value-index list
        saturating_bytes(d, INTSIZE),         # D-sign copy
        saturating_bytes(d + 1, INTSIZE),     # frozen colptr copy
        saturating_bytes(upper_nnz, INTSIZE), # frozen rowval copy
    )
    inner_cache = saturating_sum_bytes(
        saturating_bytes(d + 1, INTSIZE),     # structural colptr copy
        saturating_bytes(upper_nnz, INTSIZE), # structural rowval copy
        saturating_bytes(d, INTSIZE),         # D-sign copy
    )
    # Construction transients (peak-counted): QDLDL input deepcopy,
    # symmetric-permutation scratch, AMD ordering dense-graph worst case,
    # BFLA owned copies (construction + first symbolic values).
    construction_peak = saturating_sum_bytes(
        csc_shape(upper_nnz),                 # input deepcopy
        saturating_sum_bytes(                 # permutation scratch
            saturating_bytes(upper_nnz, INTSIZE),
            saturating_bytes(d + 1, INTSIZE),
            saturating_bytes(scalar, upper_nnz),
            saturating_bytes(upper_nnz, INTSIZE),
            csc_shape(upper_nnz),
        ),
        saturating_bytes(d, d, INTSIZE),      # AMD dense-graph worst case
        saturating_bytes(upper_nnz, INTSIZE), # BFLA value-index list
        saturating_bytes(2, value_bytes),     # BFLA owned copies
    )
    # Per-factorize transient (owned value copy) and per-solve transient
    # (triangular-pass scratch allowance).
    operation_peak = saturating_sum_bytes(
        value_bytes,
        saturating_bytes(scalar, d),
        saturating_bytes(d, INTSIZE),
    )
    provider = saturating_sum_bytes(
        qdldl_workspace, qdldl_factor, bfla_cache, inner_cache,
        construction_peak, operation_peak,
    )
    # Direction workspace vectors, counted exactly off the
    # `SymmetricCoreWorkspace` fields: 6 length-d buffers (rhs/solution,
    # apply/residual/correction, row sums), 5 length-nr buffers, 8 length-m
    # buffers, 3 length-nr buffers (n == nr under identity coordinates),
    # and 4 scalars.
    workspace_vectors = saturating_sum_bytes(
        saturating_bytes(6, scalar, d),
        saturating_bytes(5, scalar, Int(nr)),
        saturating_bytes(8, scalar, Int(m)),
        saturating_bytes(3, scalar, Int(nr)),
        saturating_bytes(4, scalar),
    )
    counted = saturating_sum_bytes(
        lower_csc,                       # lower pattern copy
        upper_csc,                       # upper factor copy
        pattern_extra,                   # pattern structure/slot extras
        wrapper_maps,                    # live + frozen maps/locations
        wrapper_snapshots,               # wrapper/workspace/last snapshots
        static_owned,                    # exact static A/b/c snapshots
        provider,                        # inventoried provider chain
        workspace_vectors,               # exact workspace vector count
        saturating_sum_bytes(            # small SPD scratch
            saturating_bytes(scalar, max_block, max_block),
            saturating_bytes(max_block, scalar),
        ),
        saturating_sum_bytes(            # admission context
            saturating_bytes(
                length(block_sizes) + Int(nr) + d, INTSIZE,
            ),
            saturating_bytes(scalar),
            saturating_bytes(1024),
        ),
    )
    return _workspace_estimate_with_margin(counted, length(block_sizes) + 1)
end


"""Diagnostic scalar-LP payload inventory; never a memory admission proof.

Counts are conditioned on explicit natural ordering through a capable provider;
this diagnostic does not establish availability or the caller's actual mode.
Scalar fields, headers/capacities and unresolved simultaneous-live storage are
excluded. No byte bound is inferred from counts or empirical allocation margins.
"""
function experimental_sparse_core_memory_inventory(nr::Integer, m::Integer, a::Integer)
    0 <= nr <= EXPERIMENTAL_SPARSE_CORE_MAX_DIMENSION &&
    0 <= m <= EXPERIMENTAL_SPARSE_CORE_MAX_DIMENSION ||
        throw(ArgumentError("experimental sparse core inventory dimensions outside scope"))
    n, rows = Int(nr), Int(m)
    d = Base.checked_add(n, rows)
    d <= EXPERIMENTAL_SPARSE_CORE_MAX_DIMENSION ||
        throw(ArgumentError("experimental sparse core inventory dimension limit"))
    0 <= a <= Base.checked_mul(n, rows) ||
        throw(ArgumentError("experimental sparse core inventory nonzero count outside scope"))
    q = Base.checked_add(Int(a), d)
    l = Base.checked_mul(d, d - 1) ÷ 2
    return (
        proven=false,
        reason=:owned_storage_bound_unavailable,
        assumed_ordering=:natural,
        dimension=d,
        array_payload_bigfloat_slots=8q + Int(a) + 18d + l,
        returned_pair_bigfloat_slots=2n + 5rows + 4,
        acceptance_array_bigfloat_slots=n + 5rows,
        unresolved=(:container_layout, :operation_scratch,
                    :construction_overlap, :retained_results),
    )
end

# Private research pattern allocation is independent of the shared cache.
# These are checked logical lengths, NOT capacity/byte/RSS admission bounds.
function _research_private_lp_counts(n::Integer, m::Integer, a::Integer)
    n >= 0 && m >= 0 && a >= 0 || throw(ArgumentError("negative private LP dimensions"))
    nr, rows, entries = Int(n), Int(m), Int(a)
    d = Base.checked_add(nr, rows)
    d <= EXPERIMENTAL_SPARSE_CORE_MAX_DIMENSION || throw(ArgumentError("private LP dimension outside research scope"))
    entries <= Base.checked_mul(nr, rows) || throw(ArgumentError("private LP nonzero count outside scope"))
    q = Base.checked_add(d, entries)
    return (n=nr, m=rows, d=d, a=entries, q=q)
end
function _research_private_lp_counts(A::SparseMatrixCSC)
    A isa SparseMatrixCSC{BigFloat,Int} || throw(ArgumentError("private LP requires BigFloat/Int CSC storage"))
    m, n = size(A)
    counts = _research_private_lp_counts(n, m, length(A.nzval))
    length(A.rowval) == counts.a && length(A.colptr) == n + 1 || throw(DimensionMismatch("private LP CSC lengths"))
    A.colptr[1] == 1 && A.colptr[end] == counts.a + 1 || throw(ArgumentError("private LP CSC endpoints"))
    for j in 1:n
        1 <= A.colptr[j] <= A.colptr[j+1] <= counts.a + 1 || throw(ArgumentError("private LP CSC pointers"))
        previous = 0
        for k in A.colptr[j]:A.colptr[j+1]-1
            row = A.rowval[k]
            previous < row <= m || throw(ArgumentError("private LP CSC rows must be sorted and unique"))
            previous = row
        end
    end
    return counts
end

"""Private exact-length LP pattern: no cache lookup/publication or Ar copy.
Only the final pattern owns new arrays. Existing refill/signature semantics are
reused; visible lengths do not certify backing capacity or total live bytes.
"""
function _research_private_lp_pattern(system::NewtonSystem{BigFloat})
    A = system.A
    A isa SparseMatrixCSC || throw(ArgumentError("private LP requires sparse A"))
    n, m, d, a, q = _research_private_lp_counts(A)
    cone = system.cone
    cone isa Union{ProductConeLinearization{BigFloat},BlockProductConeLinearization{BigFloat}} ||
        throw(ArgumentError("private LP requires a product cone"))
    length(cone.block_ranges) == m && all(i -> cone.block_ranges[i] == (i:i), 1:m) ||
        throw(ArgumentError("private LP requires ordered scalar blocks"))
    ar_colptr = Vector{Int}(undef, n+1); copyto!(ar_colptr, A.colptr)
    ar_rowval = Vector{Int}(undef, a); copyto!(ar_rowval, A.rowval)
    colptr = Vector{Int}(undef, d+1)
    rowval = Vector{Int}(undef, q)
    ar_slots = Vector{Int}(undef, a)
    theta_slots = Vector{Int}(undef, m)
    x_diag_slots = Vector{Int}(undef, n)
    ranges = Vector{UnitRange{Int}}(undef, m)
    shapes = fill(:dense_lower, m)
    slot = 1
    for j in 1:n
        colptr[j] = slot; rowval[slot] = j; x_diag_slots[j] = slot; slot += 1
        for k in A.colptr[j]:A.colptr[j+1]-1
            rowval[slot] = n + A.rowval[k]; ar_slots[k] = slot; slot += 1
        end
    end
    for i in 1:m
        ranges[i] = i:i; colptr[n+i] = slot
        rowval[slot] = n+i; theta_slots[i] = slot; slot += 1
    end
    colptr[d+1] = slot
    slot == q+1 || error("private LP slot coverage")
    signature = _symmetric_core_structure_signature(n, m, ar_colptr, ar_rowval, ranges, shapes)
    pattern = SymmetricCorePattern{BigFloat}(n,m,d,ar_colptr,ar_rowval,ranges,shapes,
        colptr,rowval,ar_slots,theta_slots,x_diag_slots,alloc_zeros(BigFloat,q),signature)
    _core_write_ar!(pattern, A)
    if cone isa ProductConeLinearization{BigFloat}
        _core_validate_theta_blocks(pattern, cone.operator)
        _core_write_theta_lower!(pattern, cone.operator)
    else
        _core_write_block_thetas!(pattern, cone)
    end
    return pattern
end

"""Memory-admitting experimental preparation: currently fails closed.

No complete owned-live byte bound is available. Unknown or invalid capacity
facts also reject. No pattern, workspace, provider, or global cache entry is
constructed; a large requested capacity never turns an unproved estimate
into an admission. Numerical research uses a separately named private
primitive, not a fallback from this function.
"""
function prepare_experimental_sparse_core_state(
    system::NewtonSystem{BigFloat}, V::AbstractMatrix{BigFloat},
    cone_families::AbstractVector{Symbol}, witness_rows::AbstractVector{<:Integer},
    delta::BigFloat, precision_bits::Int,
    memory_limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer}; symbolic_epoch::Integer=0,
    ordering::Symbol=:amd,
)
    system.A isa SparseMatrixCSC || throw(ArgumentError("experimental sparse core requires sparse A"))
    m, n = size(system.A)
    inventory = experimental_sparse_core_memory_inventory(n, m, nnz(system.A))
    memory_limit_bytes !== nothing && current_rss_bytes !== nothing ||
        throw(ArgumentError("experimental sparse core capacity_unknown; memory admission refused"))
    0 <= current_rss_bytes < memory_limit_bytes ||
        throw(ArgumentError("experimental sparse core invalid_capacity; memory admission refused"))
    throw(ArgumentError("experimental sparse core $(inventory.reason); memory admission refused"))
end

"""
    _research_prepare_experimental_sparse_core_state(system, V, cone_families,
        witness_rows, delta, precision_bits, memory_limit_bytes,
        current_rss_bytes; symbolic_epoch=0) -> (workspace, context)

PRIVATE UNADMITTED numerical-research primitive for controlled small tests.
It is never called as a fallback by memory-admitting preparation and grants
no resource qualification. Constructs only small BigFloat LP systems
(every cone block a scalar orthant row) with identity coordinates,
a verified triangular original-A minor witness, per-block SPD semantic
Theta, an explicit caller-owned precision-matched positive finite δ, a
sparse original A, and a known budget/RSS feeding the provisional inventory
gate. The gate runs BEFORE pattern/workspace/wrapper/provider allocations,
but does not yet establish a complete simultaneous-live upper bound; that
remains an integration blocker. Unknown capacity or inputs outside the
admitted mathematical scope reject before those allocations.  Returns the standard
`SymmetricCoreWorkspace` (holding an `ExperimentalSparseCoreCache`) plus
the bound admission context.
"""
function _research_prepare_experimental_sparse_core_state(
    system::NewtonSystem{BigFloat},
    V::AbstractMatrix{BigFloat},
    cone_families::AbstractVector{Symbol},
    witness_rows::AbstractVector{<:Integer},
    delta::BigFloat,
    precision_bits::Int,
    memory_limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer};
    symbolic_epoch::Integer=0,
    ordering::Symbol=:amd,
)
    SparseQDLDLProviderOrderingAvailable(BigFloat, ordering) || throw(ArgumentError(
        "experimental sparse core ordering $ordering is unavailable; no fallback",
    ))
    bits = precision(BigFloat)
    precision_bits == bits || throw(ArgumentError(
        "experimental sparse core precision $precision_bits disagrees " *
        "with ambient BigFloat precision $bits",
    ))
    precision(delta) == bits || throw(ArgumentError(
        "experimental sparse core delta precision $(precision(delta)) " *
        "disagrees with ambient BigFloat precision $bits",
    ))
    isfinite(delta) && delta > zero(BigFloat) || throw(ArgumentError(
        "experimental sparse core delta must be positive and finite",
    ))
    _hsd_is_identity_basis(V) || throw(ArgumentError(
        "experimental sparse core requires identity coordinates " *
        "(got $(typeof(V)))",
    ))
    # Sparse original A throughout: the shared production acceptance terms
    # iterate `nzrange`, and exact static binding compares sparse slots.
    system.A isa SparseMatrixCSC || throw(ArgumentError(
        "experimental sparse core requires a sparse original A " *
        "(got $(typeof(system.A)))",
    ))
    m, n = size(system.A)
    nr = size(V, 2)
    size(V, 1) == n || throw(DimensionMismatch(
        "experimental sparse core basis rows $(size(V, 1)) disagree " *
        "with A columns $n",
    ))
    nr == n || throw(ArgumentError(
        "experimental sparse core identity coordinates require nr == n, " *
        "got nr=$nr, n=$n",
    ))
    # Validate counts/CSC structure before witness indexing or structural allocation.
    counts = _research_private_lp_counts(system.A)
    d = counts.d
    cone = system.cone
    cone isa Union{ProductConeLinearization{BigFloat},
                   BlockProductConeLinearization{BigFloat}} ||
        throw(ArgumentError(
            "experimental sparse core requires a product or block-product " *
            "cone linearization",
        ))
    block_ranges = symmetric_core_block_ranges(cone)
    _experimental_validate_families(block_ranges, cone_families)
    block_sizes = Int[length(rows) for rows in block_ranges]
    ar_nnz = counts.a
    # Dimension-only budget gate BEFORE any allocation below.
    estimate = experimental_sparse_core_bytes(nr, m, ar_nnz, block_sizes)
    estimate >= typemax(Int) && throw(ArgumentError(
        "experimental sparse core byte estimate saturated; route ineligible",
    ))
    # Rejection heuristic for controlled research ONLY, never a proof or an
    # input to the certified-memory eligibility function. The public
    # memory-admitting entry above always refuses the unavailable bound.
    memory_limit_bytes !== nothing && current_rss_bytes !== nothing ||
        throw(ArgumentError("unknown local research capacity"))
    0 <= current_rss_bytes < memory_limit_bytes &&
    estimate <= memory_limit_bytes - current_rss_bytes ||
        throw(ArgumentError("local research capacity heuristic rejected"))
    # Range/isometry preconditions before materialization (identity fast
    # path: no allocation, no Gram matrix).
    _validate_core_preconditions(system, V)
    # Independent rank evidence from actual original coefficients — never a
    # caller Boolean or an identity-coordinate shortcut.
    orientation = _experimental_check_witness(system.A, witness_rows)
    # SPD admission on the semantic Theta blocks (owned small scratch only).
    max_block = maximum(block_sizes; init=0)
    scratch = alloc_zeros(BigFloat, max_block, max_block)
    for (index, block) in enumerate(_experimental_cone_blocks(cone))
        _experimental_block_spd!(scratch, block) || throw(ArgumentError(
            "experimental sparse core Theta block $index is not " *
            "positive-definite",
        ))
    end
    # Research-only allocations. The inventory above is not yet a certified
    # simultaneous-live bound; do not promote this gate to production policy.
    # The private research path allocates its final owned pattern directly:
    # no global cache side effects, cached-template overlap, or sparse(A) copy.
    # Default/shared constructors and memory admission remain unchanged.
    pattern = _research_private_lp_pattern(system)
    wrapper = ExperimentalSparseCoreCache(
        pattern, system, delta; symbolic_epoch=Int(symbolic_epoch), ordering=ordering,
    )
    workspace = _symmetric_core_workspace_prevalidated(
        pattern, wrapper, V, system,
    )
    context = ExperimentalSparseCoreContext(
        Symbol[family for family in cone_families],
        Int[row for row in witness_rows],
        orientation,
        MA.mutable_copy(delta),
        bits,
        estimate,
    )
    return (workspace, context)
end

"""Require exact static-operator identity against owned snapshots.

Compares the request `system` EXACTLY against the wrapper's owned static
snapshots: original-A dimensions, sparse structure, every stored
coefficient (BigFloat value equality, no Float64 conversion), and every
coefficient precision; same for `b` and `c`; plus identity coordinates of
the admitted shape on the workspace basis.  A stored-slot mutation that is
invisible to sampled hashes (e.g. `BigFloat("1e-400")` replaced by zero,
where `Float64` sees zero both times) is caught here because the comparison
never converts.  Anything outside the frozen static identity fails closed.
"""
function _experimental_require_exact_static!(
    workspace::SymmetricCoreWorkspace{BigFloat},
    system::NewtonSystem{BigFloat},
)
    wrapper = workspace.cache
    wrapper isa ExperimentalSparseCoreCache || throw(ArgumentError(
        "experimental sparse core static check requires an " *
        "ExperimentalSparseCoreCache (got $(typeof(wrapper)))",
    ))
    bits = wrapper.precision_bits
    _hsd_is_identity_basis(workspace.V) || throw(ArgumentError(
        "experimental sparse core coordinates are no longer identity",
    ))
    size(workspace.V, 1) == workspace.n &&
    size(workspace.V, 2) == workspace.nr || throw(DimensionMismatch(
        "experimental sparse core basis shape changed",
    ))
    static = wrapper.static_A
    system.A isa SparseMatrixCSC || throw(ArgumentError(
        "experimental sparse core requires a sparse request A",
    ))
    size(system.A) == size(static) || throw(DimensionMismatch(
        "experimental sparse core original-A dimensions changed",
    ))
    system.A.colptr == static.colptr &&
    system.A.rowval == static.rowval || throw(ArgumentError(
        "experimental sparse core original-A structure changed",
    ))
    length(system.A.nzval) == length(static.nzval) || throw(DimensionMismatch(
        "experimental sparse core original-A value storage length changed",
    ))
    @inbounds for column in 1:size(static, 2)
        for pointer in static.colptr[column]:(static.colptr[column + 1] - 1)
            value = system.A.nzval[pointer]
            value == static.nzval[pointer] || throw(ArgumentError(
                "experimental sparse core original-A coefficient changed",
            ))
            precision(value) == bits || throw(ArgumentError(
                "experimental sparse core original-A coefficient " *
                "precision changed",
            ))
        end
    end
    length(system.b) == length(wrapper.static_b) || throw(DimensionMismatch(
        "experimental sparse core static b dimension changed",
    ))
    @inbounds for i in eachindex(wrapper.static_b)
        value = system.b[i]
        value == wrapper.static_b[i] || throw(ArgumentError(
            "experimental sparse core static b changed",
        ))
        precision(value) == bits || throw(ArgumentError(
            "experimental sparse core static b precision changed",
        ))
    end
    length(system.c) == length(wrapper.static_c) || throw(DimensionMismatch(
        "experimental sparse core static c dimension changed",
    ))
    @inbounds for i in eachindex(wrapper.static_c)
        value = system.c[i]
        value == wrapper.static_c[i] || throw(ArgumentError(
            "experimental sparse core static c changed",
        ))
        precision(value) == bits || throw(ArgumentError(
            "experimental sparse core static c precision changed",
        ))
    end
    return true
end

"""
    factor_experimental_sparse_core_epoch!(workspace, system, matrix_epoch)

INTERNAL EXPERIMENTAL epoch driver over the existing seams: validates the
static identity, revokes the previous epoch, refills Theta from the semantic
cone, re-requires per-block SPD Theta at this epoch, factors genuinely
sparse through the wrapper, syncs the frozen original snapshot, and solves
the homogeneous core once (existing `_core_refine!` targets and
two-correction caps unchanged).  ANY failure — including a post-factor
synchronization or homogeneous-refinement failure — revokes wrapper+inner
solve authority and clears the workspace receipt, synchronization, and
homogeneous authority before rethrowing.
"""
function factor_experimental_sparse_core_epoch!(
    workspace::SymmetricCoreWorkspace{BigFloat},
    system::NewtonSystem{BigFloat},
    matrix_epoch::Integer,
)
    # No generic dense K/Theta/RRQR fallback: this driver factors only
    # through the thin experimental wrapper.  The check is a runtime `isa`
    # (rather than a signature constraint) because the wrapper type is
    # defined after this file in include order.
    workspace.cache isa ExperimentalSparseCoreCache || throw(ArgumentError(
        "experimental sparse core epoch requires an " *
        "ExperimentalSparseCoreCache (got $(typeof(workspace.cache)))",
    ))
    _core_revoke_epoch!(workspace)
    try
        # The whole epoch — static validation, exact static binding,
        # refill, SPD admission, factorize, sync, homogeneous solve — runs
        # inside one fail-closed revocation transaction: any failure
        # revokes wrapper+inner authority and clears receipts below.
        _experimental_require_frozen!(workspace.cache)
        _core_validate_static_identity(workspace, system)
        _experimental_require_exact_static!(workspace, system)
        _experimental_require_static_pattern!(workspace.cache, workspace.pattern)
        _core_refill_from_system!(workspace, system)
        _experimental_require_spd_theta!(workspace, system)
        factorize_symmetric_core_pattern!(
            workspace.cache, workspace.pattern, Int(matrix_epoch),
        )
        sync_core_factor_epoch!(workspace; system=system)
        solve_core_homogeneous!(workspace, system)
    catch
        # Fail closed: no receipt, no Fresh solve, and no stale homogeneous
        # solution may survive any epoch failure.
        _core_revoke_epoch!(workspace)
        rethrow()
    end
    return workspace
end

"""
    experimental_sparse_core_accept(system, direction, cone_families) -> Bool

INTERNAL EXPERIMENTAL shared five-equation acceptance predicate.  This is
genuinely shared production logic, not a copy: every residual/work number
below is computed by the `_shared_*` acceptance helpers in
`src/hsd/product_cone_hsd.jl` that the production
`_product_hsd_newton_residual_ok` gate (ordinary
`conditioned_authority=false` path) calls itself — same formulas, same
accumulation order, same `_product_hsd_newton_close` threshold.  Only the
input sourcing differs (a standalone `NewtonSystem`/direction instead of
the product-HSD state vectors), and only LP (scalar orthant) cone rows are
admitted: production SOC acceptance needs certified runtime scaling context
that no standalone block can supply, so `:soc` and every other family
return `false` (rejected, never approximated).  RHS signs follow the
semantic `HSDNewtonRHS` convention (`rhs = -residual`).  Any dimension
mismatch or non-finite term returns `false`.
"""
function experimental_sparse_core_accept(
    system::NewtonSystem{BigFloat},
    direction::NewtonDirection{BigFloat},
    cone_families::AbstractVector{Symbol},
)::Bool
    A = system.A
    A isa SparseMatrixCSC || return false
    m, n = size(A)
    length(direction.dx) == n || return false
    length(direction.dy) == m || return false
    length(direction.ds) == m || return false
    isfinite(direction.dtau) && isfinite(direction.dkappa) || return false
    cone = system.cone
    cone isa Union{ProductConeLinearization{BigFloat},
                   BlockProductConeLinearization{BigFloat}} || return false
    ranges = cone.block_ranges
    length(ranges) == length(cone_families) || return false
    # LP-only: every declared family must be `:lp` (scalar orthant row).
    # Anything else lacks production acceptance context and is rejected.
    for family in cone_families
        family === :lp || return false
    end
    for rows in ranges
        length(rows) == 1 || return false
    end
    rhs = system.rhs
    b = system.b
    c = system.c
    dx = direction.dx
    dy = direction.dy
    ds = direction.ds
    dtau = direction.dtau
    dkappa = direction.dkappa

    # Production-sign residuals as owned vectors (`rhs = -residual`).
    rP = alloc_zeros(BigFloat, m)
    rD = alloc_zeros(BigFloat, n)
    @inbounds for k in 1:m
        rP[k] = MA.mutable_copy(-rhs.primal_affine[k])
    end
    @inbounds for j in 1:n
        rD[j] = MA.mutable_copy(-rhs.dual_affine[j])
    end
    rG = -rhs.homogeneous_gap
    scalar_rhs = rhs.tau_kappa
    h = rhs.cone_corrector

    # Primal/dual `A` actions and the scalar orthant `Theta` map, evaluated
    # exactly once (same sourcing role as the production `base.ax`/`base.e`).
    ax = alloc_zeros(BigFloat, m)
    mul!(ax, A, dx)
    theta = alloc_zeros(BigFloat, m)
    e = alloc_zeros(BigFloat, m)
    for (block_index, rows) in enumerate(ranges)
        i = first(rows)
        value = cone isa ProductConeLinearization{BigFloat} ?
            cone.operator[i, i] : cone.operators[block_index][1, 1]
        isfinite(value) && isfinite(dy[i]) || return false
        theta[i] = MA.mutable_copy(value)
        e[i] = MA.mutable_copy(value * dy[i])
    end

    # Primal: shared scale + shared residuals, production group fallback.
    row_sums = alloc_zeros(BigFloat, m)
    operator_norm, direction_norm, rhs_norm = _shared_primal_scale(
        A, b, ds, dx, dtau, rP, row_sums,
    )
    primal_componentwise, primal_group, primal_work = _shared_primal_residuals(
        ax, ds, b, dtau, rP, operator_norm, direction_norm, rhs_norm,
    )
    (primal_componentwise ||
     _product_hsd_newton_close(primal_group, primal_work)) || return false

    # Dual: shared columnwise stats, production group fallback.
    dual_componentwise, dual_group, dual_work = _shared_dual_stats(
        A, c, dy, dtau, rD,
    )
    (dual_componentwise ||
     _product_hsd_newton_close(dual_group, dual_work)) || return false

    # Gap and scalar: shared terms, production close.
    gap_residual, gap_work = _shared_gap_terms(
        rG, dkappa, c, dx, b, dy,
    )
    _product_hsd_newton_close(gap_residual, gap_work) || return false

    # Cone: one shared scalar-orthant row evaluation per LP row, production
    # componentwise + group fallback over the same close.
    cone_componentwise = true
    cone_group = zero(BigFloat)
    cone_work_max = zero(BigFloat)
    @inbounds for k in 1:m
        residual, work = _shared_orthant_row_terms(
            ds[k], e[k], h[k], theta[k], dy[k],
        )
        cone_componentwise &= _product_hsd_cone_newton_close(residual, work)
        cone_group = max(cone_group, abs(residual))
        cone_work_max = max(cone_work_max, work)
    end
    (cone_componentwise ||
     _product_hsd_cone_newton_close(cone_group, cone_work_max)) || return false

    scalar_residual, scalar_work = _shared_scalar_terms(
        system.kappa, dtau, system.tau, dkappa, scalar_rhs,
    )
    _product_hsd_newton_close(scalar_residual, scalar_work) || return false
    return true
end

"""
    solve_experimental_sparse_core_direction!(workspace, system, context)

INTERNAL EXPERIMENTAL enforced direction caller.  The complete operation —
context checks, exact static binding, guard, homogeneous reuse, original-K
refinement, and shared five-equation acceptance — runs inside one
fail-closed revocation transaction: any failure (guard, solve, refinement,
or acceptance) revokes wrapper+inner authority and clears receipts before
rethrowing, so no manual cleanup is ever needed after a failed solve.  A
failing direction is never published as accepted.
"""
function solve_experimental_sparse_core_direction!(
    workspace::SymmetricCoreWorkspace{BigFloat},
    system::NewtonSystem{BigFloat},
    context::ExperimentalSparseCoreContext,
)
    workspace.cache isa ExperimentalSparseCoreCache || throw(ArgumentError(
        "experimental sparse core solve requires an " *
        "ExperimentalSparseCoreCache (got $(typeof(workspace.cache)))",
    ))
    try
        context.precision_bits == precision(BigFloat) || throw(ArgumentError(
            "experimental sparse core ambient BigFloat precision " *
            "$(precision(BigFloat)) disagrees with admitted precision " *
            "$(context.precision_bits)",
        ))
        context.delta == workspace.cache.delta || throw(ArgumentError(
            "experimental sparse core delta identity changed between " *
            "prepare and solve",
        ))
        _experimental_require_frozen!(workspace.cache)
        _experimental_require_exact_static!(workspace, system)
        _experimental_require_static_pattern!(workspace.cache, workspace.pattern)
        direction, residual = solve_core_direction!(workspace, system)
        experimental_sparse_core_accept(
            system, direction, context.families,
        ) || throw(ArgumentError(
            "experimental sparse core direction failed the five-equation " *
            "acceptance gate",
        ))
        return (direction, residual)
    catch
        # Fail closed with no manual cleanup: a failed solve leaves no
        # usable factor, receipt, synchronization, or homogeneous state.
        # Last-successful epoch evidence in the wrapper survives (it must:
        # wiping it would authorize conflicting same-epoch operators).
        _core_revoke_epoch!(workspace)
        rethrow()
    end
end
