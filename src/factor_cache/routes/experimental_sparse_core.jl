#=====================================================================#
#    ExperimentalSparseCoreCache — INTERNAL EXPERIMENTAL thin wrapper
#    over SparseQDLDLCache{BigFloat} for the symmetric augmented core.
#
#    Scope (BigFloat-only, internal, explicitly experimental; NOT native
#    routing, NOT public Settings, NOT a scalability claim):
#      * Wraps the existing `SparseQDLDLCache{BigFloat}` without duplicating
#        any numerical kernel: symbolic/numeric LDL, solves, and correction
#        solves are delegated to the loaded BFLA/QDLDL provider through the
#        inner cache.  This file owns only the core-specific CSC translation:
#        the frozen lower-to-upper slot map, diagonal locations, the owned
#        shifted upper factor buffer, the independent unshifted original
#        snapshot, D-signs, the frozen caller shift δ, precision, and the
#        operator identity.
#      * The raw symmetric augmented core K = [0 Ar'; Ar -Theta] is NOT
#        quasi-definite (structural zeros on the reduced-x diagonal), so the
#        wrapper factors only the explicitly shifted operator
#        Kδ = K + δ*diag(dsigns) with a caller-owned, precision-matched,
#        positive finite δ.  No shift ladder is fabricated here: δ is frozen
#        at construction and any change requires a new wrapper.
#      * The setup-owned lower pattern buffer (`pattern.nzval`) is NEVER
#        shifted: shifts are applied only inside the wrapper-owned upper
#        buffer at diagonal slots.  Structural zeros are preserved exactly
#        in both the snapshot and the factor input (off-diagonal slots copy
#        the original values, including zeros, unchanged).
#      * Included AFTER `kkt/symmetric_core.jl` (see `src/SDPX.jl`) because
#        the constructor consumes `SymmetricCorePattern{BigFloat}`.
#
#    Fail-closed contract:
#      * construction requires a loaded QDLDL provider (via the inner
#        cache), ambient BigFloat precision agreement, and an explicit
#        positive finite precision-matched δ; anything else throws before
#        any factorization;
#      * the ONLY numeric factor entry is `factorize_symmetric_core_pattern!`
#        (genuinely sparse refill through the frozen map); every direct
#        `factorize!` call — dense or sparse — revokes all solve authority
#        and throws, so an accidental dense dispatch can never silently
#        factorize;
#      * same-epoch reuse without any operator/map/sign/δ change returns the
#        existing factor; any conflicting same-epoch call revokes and throws;
#      * any failed factor attempt leaves the wrapper AND the inner cache
#        non-`Fresh`, so no stale solve survives.
#=====================================================================#

"""
    ExperimentalSparseCoreCache <: AbstractFactorCache{BigFloat}

INTERNAL EXPERIMENTAL thin core-specific wrapper over one
`SparseQDLDLCache{BigFloat}`.  Owns the frozen lower→upper CSC slot map
(`lower_to_upper`), the upper diagonal locations (`upper_diag`), the owned
shifted upper factor buffer (`upper_nzval`), an independent owned snapshot
of the unshifted lower original (`snapshot_lower`), the frozen D-signs, the
frozen caller shift (`delta`), and the frozen precision/operator identity.
The numeric factor itself lives in `inner`.
"""
mutable struct ExperimentalSparseCoreCache <: AbstractFactorCache{BigFloat}
    inner::SparseQDLDLCache{BigFloat}
    n::Int
    lower_colptr::Vector{Int}
    lower_rowval::Vector{Int}
    snapshot_lower::Vector{BigFloat}
    upper_colptr::Vector{Int}
    upper_rowval::Vector{Int}
    lower_to_upper::Vector{Int}
    upper_diag::Vector{Int}
    upper_nzval::Vector{BigFloat}
    dsigns::Vector{Int}
    delta::BigFloat
    precision_bits::Int
    formation_rounding::RoundingMode
    const ordering::Symbol
    operator_signature::UInt64
    # P1 frozen authority: independent exact copies of the map, diagonal
    # locations, signs, and shift taken at construction.  The live arrays
    # above are mutable buffers; every reuse/refactor/sync-match/solve
    # verifies them against these frozen copies first, so a mutated live
    # array can never authorize itself.
    frozen_lower_to_upper::Vector{Int}
    frozen_upper_diag::Vector{Int}
    frozen_dsigns::Vector{Int}
    frozen_delta::BigFloat
    # Static slot mapping: pattern buffer slots holding the admitted Ar
    # coefficients (in admitted-A CSC order) plus its frozen copy.  Verified
    # against the admitted original A before every factorization — never
    # against the mutable request system or a mutable self-snapshot.
    ar_slots::Vector{Int}
    frozen_ar_slots::Vector{Int}
    # P1 last-successful epoch evidence: the matrix epoch and the exact
    # unshifted operator values of the most recent successful numeric
    # factor.  Revocation (`invalidate!`/revoke paths) clears live solve
    # validity but NEVER this evidence, so driver revocation cannot erase
    # the record and authorize a conflicting same-epoch refactor.
    last_matrix_epoch::Int
    last_valid::Bool
    last_snapshot_lower::Vector{BigFloat}
    # P1 exact static-operator snapshots: owned exact BigFloat copies of the
    # original A (structure + values), b, and c taken at preparation.  Rank
    # and static authority bind to these — never to lossy sampled hashes.
    static_A::SparseMatrixCSC{BigFloat,Int}
    static_b::Vector{BigFloat}
    static_c::Vector{BigFloat}
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    solve_count::Int
    refine_count::Int
    status::FactorCacheState
end

"""Transpose a lower-triangle CSC structure to upper, owning the slot map.

Returns `(upper_colptr, upper_rowval, lower_to_upper)` where
`lower_to_upper[s]` is the upper `nzval` slot holding the transpose of lower
slot `s`.  The transpose is a bijection, so both triangles hold the same
entry count; any deviation throws.  Upper columns are emitted in ascending
row order.  Every entry must satisfy `row >= column` (lower-triangle input)
and every upper column must be nonempty (the caller guarantees the
structural diagonal slots).
"""
function _experimental_upper_structure(
    colptr::AbstractVector{Int}, rowval::AbstractVector{Int}, d::Int,
)
    d >= 0 || throw(ArgumentError(
        "experimental sparse core dimension must be nonnegative",
    ))
    length(colptr) == d + 1 || throw(DimensionMismatch(
        "experimental sparse core lower colptr length disagrees with order $d",
    ))
    nnz_lower = colptr[d + 1] - 1
    nnz_lower == length(rowval) || throw(ArgumentError(
        "experimental sparse core lower CSC structure is inconsistent",
    ))
    counts = zeros(Int, d)
    @inbounds for c in 1:d
        for p in colptr[c]:(colptr[c + 1] - 1)
            r = rowval[p]
            1 <= r <= d || throw(ArgumentError(
                "experimental sparse core lower row index $r out of range",
            ))
            r >= c || throw(ArgumentError(
                "experimental sparse core lower structure holds an entry " *
                "above the diagonal at ($r, $c)",
            ))
            counts[r] += 1
        end
    end
    upper_colptr = Vector{Int}(undef, d + 1)
    upper_colptr[1] = 1
    @inbounds for c in 1:d
        upper_colptr[c + 1] = upper_colptr[c] + counts[c]
    end
    nnz_upper = upper_colptr[d + 1] - 1
    nnz_upper == nnz_lower || throw(ArgumentError(
        "experimental sparse core transpose entry count $nnz_upper " *
        "disagrees with lower count $nnz_lower",
    ))
    upper_rowval = Vector{Int}(undef, nnz_upper)
    lower_to_upper = Vector{Int}(undef, nnz_lower)
    next = copy(upper_colptr)
    @inbounds for c in 1:d
        for p in colptr[c]:(colptr[c + 1] - 1)
            r = rowval[p]
            slot = next[r]
            next[r] = slot + 1
            upper_rowval[slot] = c
            lower_to_upper[p] = slot
        end
    end
    @inbounds for c in 1:d
        upper_colptr[c] < upper_colptr[c + 1] || throw(ArgumentError(
            "experimental sparse core upper column $c is structurally empty",
        ))
        for p in upper_colptr[c]:(upper_colptr[c + 1] - 1)
            upper_rowval[p] <= c || throw(ArgumentError(
                "experimental sparse core upper structure holds an entry " *
                "below the diagonal",
            ))
        end
    end
    return (upper_colptr, upper_rowval, lower_to_upper)
end

"""Locate the lower slot of each diagonal entry `(j, j)`.

The symmetric-core pattern structurally owns every diagonal (the x diagonal
as numerical zeros, the y diagonal from the dense Theta triangles), so a
missing diagonal is a structural violation and throws.
"""
function _experimental_lower_diag_slots(
    colptr::AbstractVector{Int}, rowval::AbstractVector{Int}, d::Int,
)
    lower_diag = Vector{Int}(undef, d)
    @inbounds for c in 1:d
        found = 0
        for p in colptr[c]:(colptr[c + 1] - 1)
            if rowval[p] == c
                found = p
                break
            end
        end
        found == 0 && throw(ArgumentError(
            "experimental sparse core lower structure misses diagonal $c",
        ))
        lower_diag[c] = found
    end
    return lower_diag
end

"""
    ExperimentalSparseCoreCache(pattern, system, delta; symbolic_epoch=0, nrhs=1, ordering=:amd)

INTERNAL EXPERIMENTAL constructor from a refilled lower
`SymmetricCorePattern{BigFloat}`, the admitted `NewtonSystem{BigFloat}`
(frozen for exact static snapshots), and an explicit caller shift `delta`.
Freezes the ambient BigFloat precision AND rounding mode, the D-signs, the
upper CSC structure, and an owned copy of δ. This internal route requires the
same arithmetic context for refactor, reuse, matching, and solves. Ordering is
also frozen; explicit natural ordering requires provider capability, while
omission retains AMD. Fills the owned upper buffer with the
shifted operator `Kδ = K + δ*diag(dsigns)` (shifts only at diagonal slots;
every off-diagonal slot — including structural zeros — copies the original
value unchanged) and constructs the inner `SparseQDLDLCache{BigFloat}` over
it, which performs the one symbolic analysis for this frozen pattern.
Throws fail-closed when no QDLDL provider is loaded, on any precision
mismatch, or unless δ is positive and finite.
"""
function ExperimentalSparseCoreCache(
    pattern::SymmetricCorePattern{BigFloat},
    system::NewtonSystem{BigFloat},
    delta::BigFloat;
    symbolic_epoch::Integer=0,
    nrhs::Integer=1,
    ordering::Symbol=:amd,
)
    SparseQDLDLProviderOrderingAvailable(BigFloat, ordering) || throw(ArgumentError(
        "experimental sparse core ordering $ordering is unavailable; no fallback",
    ))
    bits = precision(BigFloat)
    formation_rounding = rounding(BigFloat)
    precision(delta) == bits || throw(ArgumentError(
        "experimental sparse core delta precision $(precision(delta)) " *
        "disagrees with ambient BigFloat precision $bits",
    ))
    isfinite(delta) && delta > zero(BigFloat) || throw(ArgumentError(
        "experimental sparse core delta must be positive and finite",
    ))
    d = pattern.dimension
    for value in pattern.nzval
        precision(value) == bits || throw(ArgumentError(
            "experimental sparse core pattern precision disagrees with " *
            "ambient BigFloat precision $bits",
        ))
    end
    all(isfinite, pattern.nzval) || throw(ArgumentError(
        "experimental sparse core pattern contains non-finite data",
    ))
    upper_colptr, upper_rowval, lower_to_upper =
        _experimental_upper_structure(pattern.colptr, pattern.rowval, d)
    lower_diag = _experimental_lower_diag_slots(
        pattern.colptr, pattern.rowval, d,
    )
    upper_diag = Vector{Int}(undef, d)
    @inbounds for j in 1:d
        upper_diag[j] = lower_to_upper[lower_diag[j]]
    end
    dsigns = symmetric_core_dsigns(pattern)
    upper_nzval = Vector{BigFloat}(undef, length(pattern.nzval))
    @inbounds for s in eachindex(pattern.nzval)
        upper_nzval[lower_to_upper[s]] = MA.mutable_copy(pattern.nzval[s])
    end
    frozen_delta = MA.mutable_copy(delta)
    @inbounds for j in 1:d
        slot = upper_diag[j]
        upper_nzval[slot] = upper_nzval[slot] + BigFloat(dsigns[j]) * frozen_delta
    end
    # The inner cache freezes its own structural copies and the provider
    # owns its value copies, so sharing these construction buffers with the
    # inner input retains no aliasing: every stored buffer below stays
    # wrapper-owned.
    upper = SparseMatrixCSC{BigFloat,Int}(
        d, d, copy(upper_colptr), copy(upper_rowval), copy(upper_nzval),
    )
    inner = SparseQDLDLCache{BigFloat}(
        upper, dsigns;
        symbolic_epoch=Int(symbolic_epoch), nrhs=Int(nrhs), ordering=ordering,
    )
    system.A isa SparseMatrixCSC || throw(ArgumentError(
        "experimental sparse core requires a sparse admitted A",
    ))
    length(pattern.ar_slots) == nnz(system.A) || throw(DimensionMismatch(
        "experimental sparse core admitted-A nonzeros disagree " *
        "with the frozen pattern Ar slots",
    ))
    ar_slots = copy(pattern.ar_slots)
    snapshot = Vector{BigFloat}(undef, length(pattern.nzval))
    @inbounds for i in eachindex(pattern.nzval)
        snapshot[i] = MA.mutable_copy(pattern.nzval[i])
    end
    last_snapshot = Vector{BigFloat}(undef, length(pattern.nzval))
    @inbounds for i in eachindex(pattern.nzval)
        last_snapshot[i] = MA.mutable_copy(pattern.nzval[i])
    end
    static_A = _experimental_owned_sparse(system.A, bits)
    static_b = Vector{BigFloat}(undef, length(system.b))
    @inbounds for i in eachindex(system.b)
        isfinite(system.b[i]) || throw(ArgumentError(
            "experimental sparse core static b is non-finite",
        ))
        static_b[i] = MA.mutable_copy(system.b[i])
    end
    static_c = Vector{BigFloat}(undef, length(system.c))
    @inbounds for i in eachindex(system.c)
        isfinite(system.c[i]) || throw(ArgumentError(
            "experimental sparse core static c is non-finite",
        ))
        static_c[i] = MA.mutable_copy(system.c[i])
    end
    size(static_A) == (pattern.m, pattern.nr) || throw(DimensionMismatch(
        "experimental sparse core static A dimensions disagree with " *
        "the frozen pattern",
    ))
    return ExperimentalSparseCoreCache(
        inner, d,
        copy(pattern.colptr), copy(pattern.rowval), snapshot,
        upper_colptr, upper_rowval, lower_to_upper, upper_diag, upper_nzval,
        dsigns, frozen_delta, bits, formation_rounding, ordering, pattern.signature,
        copy(lower_to_upper), copy(upper_diag), copy(dsigns),
        MA.mutable_copy(frozen_delta),
        copy(ar_slots), copy(ar_slots),
        0, false, last_snapshot,
        static_A, static_b, static_c,
        Int(symbolic_epoch), 0, 0, 0, 0, Prepared,
    )
end

"""Independent owned sparse copy of a static operator matrix.

Copies structure and every BigFloat value into fresh objects at the frozen
precision; non-finite or precision-drifted values fail closed.  Accepts any
`AbstractMatrix{BigFloat}` (sparse input keeps its slots; dense input is
normalized through `sparse`)."""
function _experimental_owned_sparse(
    A::AbstractMatrix{BigFloat}, bits::Int,
)
    S = A isa SparseMatrixCSC ? A : sparse(A)
    m, n = size(S)
    owned_nzval = Vector{BigFloat}(undef, nnz(S))
    @inbounds for i in eachindex(S.nzval)
        value = S.nzval[i]
        isfinite(value) || throw(ArgumentError(
            "experimental sparse core static A is non-finite",
        ))
        precision(value) == bits || throw(ArgumentError(
            "experimental sparse core static A precision disagrees " *
            "with frozen construction precision $bits",
        ))
        owned_nzval[i] = MA.mutable_copy(value)
    end
    return SparseMatrixCSC{BigFloat,Int}(
        m, n, copy(S.colptr), copy(S.rowval), owned_nzval,
    )
end

"""Verify static pattern slots against the admitted original A.

Reads every Ar coefficient back through the static slot mapping
(`ar_slots`, in admitted-A CSC order) and compares exactly — BigFloat value
equality plus precision — against the admitted snapshot; every structural x
diagonal must be exactly zero at the frozen precision.  This binds the
pattern buffer to the admitted original, not to the mutable request system
or a mutable self-snapshot: a poisoned pattern slot (or a poisoned x
zero) is caught here before any factorization.  Returns `true` or `false`
(query form for matching); the seam uses the throwing form below.
"""
function _experimental_pattern_static_ok(
    wrapper::ExperimentalSparseCoreCache,
    pattern::SymmetricCorePattern{BigFloat},
)
    _experimental_structure_ok(wrapper) || return false
    bits = wrapper.precision_bits
    A = wrapper.static_A
    m, nr = size(A)
    pattern.nr == nr && pattern.m == m && pattern.dimension == wrapper.n || return false
    pattern.colptr == wrapper.lower_colptr && pattern.rowval == wrapper.lower_rowval || return false
    length(pattern.nzval) == length(wrapper.snapshot_lower) || return false
    pattern.ar_colptr == A.colptr && pattern.ar_rowval == A.rowval || return false
    wrapper.ar_slots == wrapper.frozen_ar_slots || return false
    pattern.ar_slots == wrapper.ar_slots || return false
    length(pattern.ar_slots) == length(A.nzval) || return false
    for c in 1:nr
        for k in A.colptr[c]:(A.colptr[c + 1] - 1)
            slot = pattern.ar_slots[k]
            1 <= slot <= length(pattern.nzval) || return false
            u, r = wrapper.lower_to_upper[slot], nr + A.rowval[k]
            wrapper.upper_rowval[u] == c || return false
            wrapper.upper_colptr[r] <= u < wrapper.upper_colptr[r + 1] || return false
            value = pattern.nzval[slot]
            value == A.nzval[k] && precision(value) == bits || return false
        end
    end
    # The admitted layout is scalar LP. Bind every refill map to its actual
    # diagonal before any @inbounds refill can follow a malformed map.
    length(pattern.x_diag_slots) == nr || return false
    length(pattern.theta_slots) == m || return false
    length(pattern.block_ranges) == length(pattern.block_shapes) == m || return false
    for j in 1:nr
        slot = pattern.x_diag_slots[j]
        1 <= slot <= length(pattern.nzval) || return false
        wrapper.lower_to_upper[slot] == wrapper.frozen_upper_diag[j] || return false
        value = pattern.nzval[slot]
        iszero(value) && precision(value) == bits || return false
    end
    for i in 1:m
        pattern.block_ranges[i] == (i:i) || return false
        pattern.block_shapes[i] === :dense_lower || return false
        slot = pattern.theta_slots[i]
        1 <= slot <= length(pattern.nzval) || return false
        wrapper.lower_to_upper[slot] == wrapper.frozen_upper_diag[nr + i] || return false
    end
    return true
end

function _experimental_require_static_pattern!(
    wrapper::ExperimentalSparseCoreCache,
    pattern::SymmetricCorePattern{BigFloat},
)
    _experimental_pattern_static_ok(wrapper, pattern) || begin
        _experimental_revoke!(wrapper)
        throw(ArgumentError(
            "experimental sparse core static pattern slots disagree " *
            "with the admitted original A (or a structural x zero " *
            "moved); solve authority revoked",
        ))
    end
    return wrapper
end

"""Diagnostic `hash(BigFloat)` mixer, without Float64 narrowing.

The retained 1e-30 and 1e-40 perturbation tests distinguish their values.
Hashes can collide; exact operator checks, not this signature, are authority.
"""
@inline function _experimental_mix_bigfloat(
    signature::UInt64, value::BigFloat,
)
    return _core_mix_uint(signature, UInt64(hash(value)))
end

function _experimental_mix_bigfloats(
    signature::UInt64, values::AbstractVector{BigFloat},
)
    signature = _core_mix_uint(signature, UInt64(length(values)))
    @inbounds for value in values
        signature = _experimental_mix_bigfloat(signature, value)
    end
    return signature
end

# This internal route requires BOTH the creation precision and creation
# rounding mode at every numeric entry. It never changes either ambient
# setting. In particular, unchanged precision does not authorize a new
# rounding of K + delta*D under an already-used matrix epoch.
@inline function _experimental_formation_ok(wrapper::ExperimentalSparseCoreCache)
    return precision(BigFloat) == wrapper.precision_bits &&
           rounding(BigFloat) == wrapper.formation_rounding
end

"""Bind live coordinates to the inner cache's independently owned CSC copies.

Check lengths before any indexing, including malformed-array requests. The
lower-to-upper coordinate relation is also checked without allocating marker
arrays. Inner structural copies are construction authority, not snapshots of
these live buffers taken during validation.
"""
function _experimental_structure_ok(wrapper::ExperimentalSparseCoreCache)
    inner = wrapper.inner
    wrapper.ordering === inner.ordering || return false
    _qdldl_provider_ordering(BigFloat, inner.provider) === wrapper.ordering || return false
    d = inner.n
    d >= 0 && wrapper.n == d && inner.prepared_shape == (d, d) || return false
    wrapper.upper_colptr == inner.colptr || return false
    wrapper.upper_rowval == inner.rowval || return false
    nvals = length(inner.rowval)
    length(wrapper.upper_colptr) == d + 1 || return false
    length(wrapper.upper_nzval) == nvals || return false
    length(wrapper.lower_colptr) == d + 1 || return false
    length(wrapper.lower_rowval) == nvals || return false
    length(wrapper.lower_to_upper) == nvals || return false
    length(wrapper.snapshot_lower) == nvals || return false
    length(wrapper.last_snapshot_lower) == nvals || return false
    length(wrapper.upper_diag) == d || return false
    length(wrapper.dsigns) == d || return false
    wrapper.lower_to_upper == wrapper.frozen_lower_to_upper || return false
    wrapper.upper_diag == wrapper.frozen_upper_diag || return false
    wrapper.lower_colptr[1] == wrapper.upper_colptr[1] == 1 || return false
    wrapper.lower_colptr[end] == wrapper.upper_colptr[end] == nvals + 1 || return false
    for c in 1:d
        lo, hi = wrapper.lower_colptr[c], wrapper.lower_colptr[c + 1]
        1 <= lo <= hi <= nvals + 1 || return false
        ulo, uhi = wrapper.upper_colptr[c], wrapper.upper_colptr[c + 1]
        1 <= ulo < uhi <= nvals + 1 || return false
        for s in lo:(hi - 1)
            r, u = wrapper.lower_rowval[s], wrapper.lower_to_upper[s]
            c <= r <= d && 1 <= u <= nvals || return false
            wrapper.upper_colptr[r] <= u < wrapper.upper_colptr[r + 1] || return false
            wrapper.upper_rowval[u] == c || return false
            r == c && wrapper.upper_diag[c] != u && return false
        end
    end
    return true
end

"""Verify exact coordinates and values of the declared factor operator.

Creation precision AND rounding must match before forming any expected
shift. Every diagonal is compared with its always-computed signed shift.
The primal snapshot zero is checked separately: rounded `v + delta == delta`
does NOT imply `v == 0`. No factor-precision or certificate-budget change.
"""
function _experimental_upper_ok(wrapper::ExperimentalSparseCoreCache)
    _experimental_formation_ok(wrapper) || return false
    _experimental_structure_ok(wrapper) || return false
    bits = wrapper.precision_bits
    wrapper.dsigns == wrapper.frozen_dsigns || return false
    wrapper.delta == wrapper.frozen_delta || return false
    precision(wrapper.delta) == bits || return false
    isfinite(wrapper.delta) && wrapper.delta > 0 || return false
    wrapper.ar_slots == wrapper.frozen_ar_slots || return false
    length(wrapper.ar_slots) == nnz(wrapper.static_A) || return false
    wrapper.last_valid && wrapper.snapshot_lower != wrapper.last_snapshot_lower && return false
    for c in 1:wrapper.n
        for s in wrapper.lower_colptr[c]:(wrapper.lower_colptr[c + 1] - 1)
            u = wrapper.lower_to_upper[s]
            expected = wrapper.snapshot_lower[s]
            isfinite(expected) && precision(expected) == bits || return false
            if wrapper.lower_rowval[s] == c
                wrapper.dsigns[c] == 1 && !iszero(expected) && return false
                # Same two-step multiply/add order as actual shift formation.
                expected = expected + BigFloat(wrapper.dsigns[c]) * wrapper.delta
            end
            isfinite(expected) || return false
            upper_value = wrapper.upper_nzval[u]
            upper_value == expected && precision(upper_value) == bits || return false
        end
    end
    for (k, slot) in enumerate(wrapper.ar_slots)
        1 <= slot <= length(wrapper.snapshot_lower) || return false
        value = wrapper.snapshot_lower[slot]
        value == wrapper.static_A.nzval[k] || return false
        precision(wrapper.static_A.nzval[k]) == bits || return false
    end
    return true
end

function _experimental_require_upper!(wrapper::ExperimentalSparseCoreCache)
    _experimental_upper_ok(wrapper) || begin
        _experimental_revoke!(wrapper)
        throw(ArgumentError(
            "experimental sparse core factor operator disagrees with " *
            "the expected numeric operator; solve authority revoked",
        ))
    end
    return wrapper
end

"""Refill the owned upper buffer from unshifted lower values plus the shift.

Off-diagonal slots copy the original values exactly (structural zeros stay
exactly zero); diagonal slots add `dsigns[j]*delta` to the original diagonal
value. The frozen precision/rounding and structure are checked before any
write; every stored value is an independent BigFloat object.
"""
function _experimental_refill_upper!(
    wrapper::ExperimentalSparseCoreCache,
    lower_nzval::AbstractVector{BigFloat},
)
    _experimental_require_frozen!(wrapper)
    length(lower_nzval) == length(wrapper.lower_to_upper) ||
        throw(DimensionMismatch(
            "experimental sparse core refill dimension mismatch",
        ))
    upper = wrapper.upper_nzval
    @inbounds for s in eachindex(lower_nzval)
        upper[wrapper.lower_to_upper[s]] = MA.mutable_copy(lower_nzval[s])
    end
    @inbounds for j in 1:wrapper.n
        slot = wrapper.upper_diag[j]
        upper[slot] = upper[slot] + BigFloat(wrapper.dsigns[j]) * wrapper.delta
    end
    return wrapper
end

"""Upper-triangle factor input sharing the wrapper-owned shifted buffer.

The inner cache validates the structure and copies values into provider
storage, so sharing this buffer with the inner `factorize!` call retains no
aliasing.
"""
function _experimental_upper_input(wrapper::ExperimentalSparseCoreCache)
    return SparseMatrixCSC{BigFloat,Int}(
        wrapper.n, wrapper.n,
        wrapper.upper_colptr, wrapper.upper_rowval, wrapper.upper_nzval,
    )
end

"""Revoke wrapper solve authority; the inner cache is revoked alongside.

Live validity (`status`, `matrix_epoch`, inner authority) dies here, but the
frozen authority, the last-successful epoch evidence, and the exact static
snapshots are NEVER cleared by revocation: wiping evidence would let a later
call authorize a conflicting operator."""
function _experimental_revoke!(wrapper::ExperimentalSparseCoreCache)
    wrapper.status = Failed
    try
        invalidate!(wrapper.inner)
    catch
    end
    return wrapper
end

"""Verify live buffers against the independent frozen authority.

Checks the mutable map, diagonal locations, signs, and shift against the
construction-frozen copies (exact integer equality; exact BigFloat value
plus precision for δ).  Any drift revokes all live solve authority and
throws — a mutated live array can never authorize reuse, refactor, or
solve.  Frozen copies, last-successful evidence, and static snapshots
survive this revocation."""
function _experimental_require_frozen!(wrapper::ExperimentalSparseCoreCache)
    _experimental_formation_ok(wrapper) || begin
        _experimental_revoke!(wrapper)
        throw(ArgumentError(
            "experimental sparse core formation precision/rounding context changed; " *
            "solve authority revoked",
        ))
    end
    _experimental_structure_ok(wrapper) || begin
        _experimental_revoke!(wrapper)
        throw(ArgumentError(
            "experimental sparse core CSC/ordering identity changed; " *
            "solve authority revoked",
        ))
    end
    wrapper.lower_to_upper == wrapper.frozen_lower_to_upper || begin
        _experimental_revoke!(wrapper)
        throw(ArgumentError(
            "experimental sparse core lower-to-upper map changed; " *
            "solve authority revoked",
        ))
    end
    wrapper.upper_diag == wrapper.frozen_upper_diag || begin
        _experimental_revoke!(wrapper)
        throw(ArgumentError(
            "experimental sparse core diagonal locations changed; " *
            "solve authority revoked",
        ))
    end
    wrapper.dsigns == wrapper.frozen_dsigns || begin
        _experimental_revoke!(wrapper)
        throw(ArgumentError(
            "experimental sparse core D-signs changed; " *
            "solve authority revoked",
        ))
    end
    wrapper.ar_slots == wrapper.frozen_ar_slots || begin
        _experimental_revoke!(wrapper)
        throw(ArgumentError(
            "experimental sparse core Ar slot mapping changed; " *
            "solve authority revoked",
        ))
    end
    (wrapper.delta == wrapper.frozen_delta &&
     precision(wrapper.delta) == wrapper.precision_bits) || begin
        _experimental_revoke!(wrapper)
        throw(ArgumentError(
            "experimental sparse core shift changed; " *
            "solve authority revoked",
        ))
    end
    return wrapper
end

function prepare!(
    wrapper::ExperimentalSparseCoreCache,
    requirements::AbstractFactorRequirements,
)
    try
        _experimental_require_frozen!(wrapper)
        n = getproperty(requirements, :n)
        n == wrapper.n || throw(ArgumentError(
            "experimental sparse core shape change requires a new cache; " *
            "got $n, frozen $(wrapper.n)",
        ))
        # Revoke live validity while retaining the last-successful evidence.
        wrapper.matrix_epoch = 0
        wrapper.status = Prepared
        prepare!(wrapper.inner, requirements)
    catch
        _experimental_revoke!(wrapper)
        rethrow()
    end
    return wrapper
end

# No direct `factorize!` entry exists on this wrapper — dense or sparse.  Any
# direct call revokes all solve authority and throws, so an accidental dense
# dispatch (`factorize!(cache, materialize_dense(pattern), epoch)`) can never
# silently factorize.  The only numeric factor entry is the genuinely sparse
# `factorize_symmetric_core_pattern!` method below.
function factorize!(
    wrapper::ExperimentalSparseCoreCache,
    A,
    matrix_epoch::Integer,
)
    wrapper.status = Factoring
    try
        throw(ArgumentError(
            "experimental sparse core has no direct factorize! entry " *
            "(got $(typeof(A))); use factorize_symmetric_core_pattern!",
        ))
    catch
        _experimental_revoke!(wrapper)
        rethrow()
    end
    return wrapper
end

function solve!(
    wrapper::ExperimentalSparseCoreCache,
    destination::AbstractVector{BigFloat},
    rhs::AbstractVector{BigFloat},
)
    try
        _experimental_require_frozen!(wrapper)
        _experimental_require_upper!(wrapper)
        _require_fresh(wrapper.status)
        solve!(wrapper.inner, destination, rhs)
        wrapper.solve_count += 1
        return destination
    catch
        _experimental_revoke!(wrapper)
        rethrow()
    end
end

function solve_multi!(
    wrapper::ExperimentalSparseCoreCache,
    destination::AbstractMatrix{BigFloat},
    rhs::AbstractMatrix{BigFloat},
)
    try
        _experimental_require_upper!(wrapper)
        _require_fresh(wrapper.status)
        solve_multi!(wrapper.inner, destination, rhs)
        wrapper.solve_count += size(rhs, 2)
        return destination
    catch
        _experimental_revoke!(wrapper)
        rethrow()
    end
end

function refine_once!(
    wrapper::ExperimentalSparseCoreCache,
    residual::AbstractVector{BigFloat},
    correction::AbstractVector{BigFloat},
)
    try
        _experimental_require_upper!(wrapper)
        _require_fresh_for_refine(wrapper.status)
        refine_once!(wrapper.inner, residual, correction)
        wrapper.refine_count += 1
        return correction
    catch
        _experimental_revoke!(wrapper)
        rethrow()
    end
end

function invalidate!(wrapper::ExperimentalSparseCoreCache)
    wrapper.matrix_epoch = 0
    wrapper.status = Invalid
    try
        invalidate!(wrapper.inner)
    catch
    end
    return wrapper
end

function revoke_numeric!(wrapper::ExperimentalSparseCoreCache)
    return invalidate!(wrapper)
end

factor_status(wrapper::ExperimentalSparseCoreCache) = wrapper.status
factor_matrix_epoch(wrapper::ExperimentalSparseCoreCache) =
    wrapper.matrix_epoch
factor_symbolic_epoch(wrapper::ExperimentalSparseCoreCache) =
    wrapper.symbolic_epoch
factor_epoch(wrapper::ExperimentalSparseCoreCache) = wrapper.factor_epoch

"""
Truthful diagnostics for the experimental factor: the actual numeric
provenance (`:qdldl` via the BFLA adapter), the frozen precision, and the
declared static shift.  `proof_valid` stays `false` in the receipt built
from these facts: a provider receipt is implementation evidence, never a
mathematical certificate.
"""
function factor_diagnostics(wrapper::ExperimentalSparseCoreCache)
    return (
        provider=:qdldl,
        kind=:sparse_ldlt,
        n=wrapper.n,
        precision_bits=wrapper.precision_bits,
        formation_rounding=string(wrapper.formation_rounding),
        ordering=wrapper.ordering,
        provider_ordering=_qdldl_provider_ordering(BigFloat, wrapper.inner.provider),
        symbolic_epoch=wrapper.symbolic_epoch,
        matrix_epoch=wrapper.matrix_epoch,
        factor_epoch=wrapper.factor_epoch,
        status=wrapper.status,
        regularization=MA.mutable_copy(wrapper.delta),
        regularization_kind=:signed_diagonal,
        pattern_signature=wrapper.operator_signature,
        upper_nnz=length(wrapper.upper_nzval),
        solve_count=wrapper.solve_count,
        refine_count=wrapper.refine_count,
    )
end

#=====================================================================#
#    Genuinely sparse core factorization + operator matching.
#
#    `factorize_symmetric_core_pattern!` for this wrapper refills the owned
#    upper buffer through the frozen lower→upper map (shifts only at diagonal
#    slots, structural zeros preserved) and delegates the numeric factor to
#    the inner QDLDL cache.  No dense `K`, no dense Theta, and no RRQR path
#    is ever consulted here.
#=====================================================================#

function factorize_symmetric_core_pattern!(
    wrapper::ExperimentalSparseCoreCache,
    pattern::SymmetricCorePattern{BigFloat},
    matrix_epoch::Integer,
)
    # Revoke solve authority on entry, BEFORE any preflight: a rejected or
    # failed attempt can never leave a stale `Fresh` behind.
    previous_status = wrapper.status
    previous_epoch = wrapper.matrix_epoch
    wrapper.status = Factoring
    try
        # Frozen authority first: a drifted live map/diagonal/sign/shift
        # revokes before any reuse decision or refactor.
        _experimental_require_frozen!(wrapper)
        epoch = Int(matrix_epoch)
        # Frozen structural identity: the setup-owned lower pattern must be
        # exactly the structure this wrapper was built for.
        pattern.colptr == wrapper.lower_colptr &&
        pattern.rowval == wrapper.lower_rowval || throw(ArgumentError(
            "experimental sparse core pattern drift: factor input must " *
            "reuse the frozen lower-triangle pattern exactly",
        ))
        pattern.signature == wrapper.operator_signature || throw(ArgumentError(
            "experimental sparse core pattern signature changed",
        ))
        all(isfinite, pattern.nzval) || throw(ArgumentError(
            "experimental sparse core pattern contains non-finite data",
        ))
        precision(BigFloat) == wrapper.precision_bits || throw(ArgumentError(
            "experimental sparse core ambient BigFloat precision " *
            "$(precision(BigFloat)) disagrees with frozen construction " *
            "precision $(wrapper.precision_bits)",
        ))
        for value in pattern.nzval
            precision(value) == wrapper.precision_bits || throw(ArgumentError(
                "experimental sparse core pattern BigFloat precision " *
                "disagrees with frozen construction precision " *
                "$(wrapper.precision_bits)",
            ))
        end
        # Static pattern slots against the admitted original A (and x
        # zeros) BEFORE any epoch decision: a poisoned pattern never
        # reaches factorization and can never become a new "original".
        _experimental_require_static_pattern!(wrapper, pattern)
        # Last-successful epoch evidence (independent of live solve
        # validity): a request for an already-factored epoch with different
        # unshifted values is a conflict and is REFUSED — never silently
        # refactored.  Live authority dies; the evidence survives.
        if wrapper.last_valid && epoch == wrapper.last_matrix_epoch &&
           pattern.nzval != wrapper.last_snapshot_lower
            _experimental_revoke!(wrapper)
            throw(ArgumentError(
                "experimental sparse core conflicting same-epoch " *
                "operator change for matrix epoch $epoch: changed values " *
                "require a new matrix epoch; refactor refused",
            ))
        end
        if previous_status === Fresh && previous_epoch == epoch
            # Same-epoch unchanged-operator promise: reuse is legal only when
            # the unshifted original values are exactly unchanged (frozen
            # map/sign/shift identity was verified above).  Any conflict
            # revokes and throws.
            pattern.nzval == wrapper.snapshot_lower || begin
                _experimental_revoke!(wrapper)
                throw(ArgumentError(
                    "experimental sparse core conflicting same-epoch " *
                    "reuse: operator values changed for matrix epoch $epoch",
                ))
            end
            _experimental_require_upper!(wrapper)
            _require_fresh(wrapper.inner.status)
            wrapper.status = Fresh
            return wrapper
        end
        _experimental_refill_upper!(wrapper, pattern.nzval)
        factorize!(wrapper.inner, _experimental_upper_input(wrapper), epoch)
        @inbounds for i in eachindex(pattern.nzval)
            wrapper.snapshot_lower[i] = MA.mutable_copy(pattern.nzval[i])
            wrapper.last_snapshot_lower[i] = MA.mutable_copy(pattern.nzval[i])
        end
        wrapper.matrix_epoch = epoch
        wrapper.last_matrix_epoch = epoch
        wrapper.last_valid = true
        wrapper.factor_epoch += 1
        wrapper.status = Fresh
    catch
        _experimental_revoke!(wrapper)
        rethrow()
    end
    return wrapper
end

"""Operator matching against the successfully factored original snapshot.

Authorizes the factor only when the live lower pattern values exactly equal
the independent unshifted snapshot taken at the successful factor epoch AND
the frozen lower structure, pattern signature, D-signs, shift, and precision
all agree.  Any tampering with values, map identity, signs, or δ reads as a
mismatch here (white-box mutation tests assert exactly this binding).
"""
function _core_factor_matches_pattern(
    wrapper::ExperimentalSparseCoreCache,
    pattern::SymmetricCorePattern{BigFloat},
)
    # Exact expected-operator matching (query: refusal, no revocation).
    # Both formation settings and exact CSC coordinates precede value checks.
    _experimental_formation_ok(wrapper) || return false
    _experimental_structure_ok(wrapper) || return false
    # Frozen authority: live map/diagonals/signs/shift/Ar-slots equal their
    # independent frozen copies; a mutated live array mismatches even when
    # values agree.
    wrapper.lower_to_upper == wrapper.frozen_lower_to_upper || return false
    wrapper.upper_diag == wrapper.frozen_upper_diag || return false
    wrapper.dsigns == wrapper.frozen_dsigns || return false
    wrapper.delta == wrapper.frozen_delta || return false
    precision(wrapper.delta) == wrapper.precision_bits || return false
    wrapper.ar_slots == wrapper.frozen_ar_slots || return false
    pattern.dimension == wrapper.n || return false
    length(pattern.nzval) == length(wrapper.snapshot_lower) || return false
    pattern.nzval == wrapper.snapshot_lower || return false
    pattern.colptr == wrapper.lower_colptr || return false
    pattern.rowval == wrapper.lower_rowval || return false
    pattern.signature == wrapper.operator_signature || return false
    length(wrapper.dsigns) == wrapper.n || return false
    @inbounds for j in 1:wrapper.n
        expected = j <= pattern.nr ? 1 : -1
        wrapper.dsigns[j] == expected || return false
    end
    isfinite(wrapper.delta) && wrapper.delta > zero(BigFloat) || return false
    for value in pattern.nzval
        precision(value) == wrapper.precision_bits || return false
    end
    # Static pattern slots against the admitted original A (never the
    # mutable request or a mutable self-snapshot).
    _experimental_pattern_static_ok(wrapper, pattern) || return false
    # Exact expected numeric operator (always-expected diagonals: a missing
    # shift reads as a mismatch, never a match).
    _experimental_upper_ok(wrapper) || return false
    return true
end

function _core_cache_signature(wrapper::ExperimentalSparseCoreCache)
    signature = _core_mix_uint(
        UInt64(0xcbf29ce484222325), wrapper.operator_signature,
    )
    signature = _core_mix_values(signature, wrapper.upper_colptr)
    signature = _core_mix_values(signature, wrapper.upper_rowval)
    # Frozen map/diagonal/sign/Ar-slot authority plus the exact live
    # shifted upper values and the exact unshifted snapshot. This diagnostic
    # hash includes sub-Float64 changes without narrowing, but is not a proof
    # of identity: hash collisions remain possible. Exact checks above are
    # the authority.
    signature = _core_mix_values(signature, wrapper.frozen_lower_to_upper)
    signature = _core_mix_values(signature, wrapper.frozen_upper_diag)
    signature = _core_mix_values(signature, wrapper.frozen_dsigns)
    signature = _core_mix_values(signature, wrapper.frozen_ar_slots)
    signature = _experimental_mix_bigfloats(signature, wrapper.upper_nzval)
    signature = _experimental_mix_bigfloats(signature, wrapper.snapshot_lower)
    signature = _experimental_mix_bigfloat(signature, wrapper.delta)
    signature = _experimental_mix_bigfloat(signature, wrapper.frozen_delta)
    signature = _core_mix_uint(signature, UInt64(hash(wrapper.formation_rounding)))
    signature = _core_mix_uint(signature, UInt64(hash(wrapper.ordering)))
    signature = _core_mix_uint(
        signature, reinterpret(UInt64, Int64(wrapper.precision_bits)),
    )
    signature = _core_mix_uint(
        signature, reinterpret(UInt64, Int64(wrapper.matrix_epoch)),
    )
    signature = _core_mix_uint(
        signature, reinterpret(UInt64, Int64(wrapper.factor_epoch)),
    )
    return signature
end
