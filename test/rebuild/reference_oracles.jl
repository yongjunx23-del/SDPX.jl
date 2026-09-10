#=====================================================================#
#    A01 — independent reference oracles for the SDPX rebuild packet.
#
#    Task card: agents/A01.md (write allow-list:
#    test/rebuild/{fixtures.jl,reference_oracles.jl,A01.jl}).
#
#    Every function in this file is an INDEPENDENT reference.  None of them
#    calls SDPX, MFLA, BFLA, CHOLMOD or LAPACK to obtain the value it is
#    used to check.  The oracles are built from:
#
#      * exact `Rational{BigInt}` arithmetic (no rounding at all), or
#      * `BigFloat` at an explicitly requested precision with
#        round-to-nearest-even, or
#      * the defining algebraic identities themselves (for the block-LDLᵀ
#        grammar the reference is `P A Pᵀ = L D Lᵀ` plus the 2×2 defining
#        system, never a kernel's stored answer).
#
#    `AGENTS.md`: "同一个错误内核不能既生成答案又验证答案".  Keeping the
#    reference arithmetic and the kernel arithmetic in different modules,
#    different types and different association orders is how that rule is
#    enforced here.
#
#    Provenance of every oracle is returned as a `ReferenceProvenance`
#    record (derivation, arithmetic, precision, rounding, tolerance label
#    and input hash) — acceptance item 2 of the card.
#=====================================================================#

module A01Oracles

using LinearAlgebra
using SHA

export ReferenceProvenance,
    oracle_provenance,
    fixture_sha256,
    sha256_text,
    hash_rational_matrix,
    hash_rational_vector,
    hash_float_vector,
    EXACT,
    exact_value,
    exact_matrix,
    exact_vector,
    rational_from_float,
    oracle_svec_packed_index,
    oracle_svec_is_scaled,
    oracle_svec_raw_packed,
    oracle_svec_highprec,
    oracle_rsoc_map_highprec,
    oracle_rsoc_apply_highprec,
    oracle_residual_exact,
    oracle_residual_highprec,
    LDLTRecord,
    oracle_compact_from_raw,
    oracle_raw_from_compact,
    oracle_grammar_is_wellformed,
    oracle_perm_from_pivots,
    oracle_lower_factors,
    oracle_bk_ldlt_exact,
    oracle_ldlt_identity_residual,
    oracle_ldlt_solve,
    oracle_ldlt_solve_exact,
    oracle_block_inertia_exact,
    oracle_accepted_pivots,
    oracle_characteristic_polynomial,
    oracle_inertia_from_characteristic_polynomial,
    oracle_2x2_solve,
    oracle_2x2_solve_mfla,
    oracle_2x2_solve_bfla,
    oracle_2x2_defining_residual,
    oracle_2x2_normalized_determinant,
    decode_lapack_ipiv_blocks

# ---------------------------------------------------------------------------
# 1. Provenance record
# ---------------------------------------------------------------------------

"""
    ReferenceProvenance

Everything the packet requires a reference to be able to state about
itself: where the value comes from, in which arithmetic, at which
precision, with which rounding, under which tolerance label, over which
input hash, and whether it is independent of the kernel under test.
"""
struct ReferenceProvenance
    name::String
    derivation::String
    anchor::String
    arithmetic::Symbol
    precision_bits::Union{Int,Nothing}
    rounding::Symbol
    tolerance::String
    input_sha256::String
    kernel_independent::Bool
    notes::String
end

function Base.show(io::IO, p::ReferenceProvenance)
    print(io, "ReferenceProvenance(", p.name, ", arithmetic=", p.arithmetic)
    print(io, ", precision_bits=", p.precision_bits, ", rounding=", p.rounding)
    print(io, ", input_sha256=", first(p.input_sha256, 16), "…)")
    return nothing
end

"""One deterministic, greppable line of provenance for a log."""
function oracle_provenance(p::ReferenceProvenance)
    return string(
        "reference=", p.name,
        " | derivation=", p.derivation,
        " | anchor=", p.anchor,
        " | arithmetic=", p.arithmetic,
        " | precision_bits=", p.precision_bits,
        " | rounding=", p.rounding,
        " | tolerance=", p.tolerance,
        " | input_sha256=", p.input_sha256,
        " | kernel_independent=", p.kernel_independent,
        " | notes=", p.notes,
    )
end

# The only arithmetic label used by the exact references.
const EXACT = :rational_bigint

# ---------------------------------------------------------------------------
# 2. Canonical input hashing
# ---------------------------------------------------------------------------
#
# Hashing is over an explicit, endian-free textual serialization so that the
# hash is reproducible on any host.  Floats are hashed by their raw bit
# pattern (never by a decimal rendering), so a 1-ulp input change changes
# the hash.

sha256_text(text::AbstractString) = bytes2hex(SHA.sha256(String(text)))

function _hash_fragment!(parts::Vector{String}, value::Rational{BigInt})
    push!(parts, string(numerator(value), "/", denominator(value)))
    return parts
end

_hash_fragment!(parts::Vector{String}, value::Integer) =
    (push!(parts, string(value)); parts)

_hash_fragment!(parts::Vector{String}, value::AbstractString) =
    (push!(parts, "str:", String(value)); parts)

_hash_fragment!(parts::Vector{String}, value::Symbol) =
    (push!(parts, "sym:", String(value)); parts)

_hash_fragment!(parts::Vector{String}, value::Bool) =
    (push!(parts, "bool:", string(value)); parts)

function _hash_fragment!(parts::Vector{String}, value::Tuple)
    push!(parts, "tup(", string(length(value)), ")")
    for entry in value
        _hash_fragment!(parts, entry)
    end
    return parts
end

function _hash_fragment!(parts::Vector{String}, value::AbstractDict)
    push!(parts, "dict(", string(length(value)), ")")
    for key in sort!(collect(keys(value)); by=string)
        _hash_fragment!(parts, key)
        _hash_fragment!(parts, value[key])
    end
    return parts
end

function _hash_fragment!(parts::Vector{String}, value::AbstractFloat)
    if value isa BigFloat
        # BigFloat is hashed through its exact rational value at its own
        # precision, not through a rounded decimal rendering.
        _hash_fragment!(parts, rational_from_float(value))
    else
        push!(parts, string(reinterpret(unsigned(typeof(value)), value)))
    end
    return parts
end

function _hash_fragment!(parts::Vector{String}, value::AbstractMatrix)
    push!(parts, string(size(value, 1), "x", size(value, 2)))
    for j in axes(value, 2), i in axes(value, 1)
        _hash_fragment!(parts, value[i, j])
    end
    return parts
end

function _hash_fragment!(parts::Vector{String}, value::AbstractVector)
    push!(parts, string("len=", length(value)))
    for i in eachindex(value)
        _hash_fragment!(parts, value[i])
    end
    return parts
end

_hashable(value) = (parts = String[]; _hash_fragment!(parts, value); parts)

"""Hash of an ordered list of values/matrices/vectors."""
function fixture_sha256(values...)
    parts = String[]
    for value in values
        append!(parts, _hashable(value))
    end
    return sha256_text(join(parts, ";"))
end

hash_rational_matrix(A) = fixture_sha256(A)
hash_rational_vector(v) = fixture_sha256(v)
hash_float_vector(v) = fixture_sha256(v)

# ---------------------------------------------------------------------------
# 3. Exact arithmetic helpers
# ---------------------------------------------------------------------------

"""
    rational_from_float(x) -> Rational{BigInt}

The EXACT rational value of a binary float.  `Rational{BigInt}(0.1)` is
`3602879701896397//36028797018963968`, not `1//10`.  This is the
conversion the residual oracle uses: the reference is exact *for the
inputs the kernel actually received*, which is the only comparison that
separates kernel rounding from input rounding.
"""
rational_from_float(x::Rational{BigInt}) = x
rational_from_float(x::Integer) = Rational{BigInt}(x)
rational_from_float(x::AbstractFloat) = Rational{BigInt}(x)

exact_value(x) = rational_from_float(x)
exact_matrix(A) = Rational{BigInt}[rational_from_float(A[i, j])
                                   for i in axes(A, 1), j in axes(A, 2)]
exact_vector(v) = Rational{BigInt}[rational_from_float(v[i])
                                   for i in eachindex(v)]

# ---------------------------------------------------------------------------
# 4. PSD `svec` oracle
# ---------------------------------------------------------------------------
#
# Frozen definition (src/ir/storage.jl `PSDCoordinateMap`, and the packed
# order `psd_packed_row`/`psd_packed_column`): lower-triangle
# column-major packing of a symmetric matrix, with the OFF-DIAGONAL
# entries multiplied by √2:
#
#     svec_k = M[i,i]                 for i == j
#     svec_k = √2 · M[i,j]            for i > j
#
# The oracle never calls `PSDCoordinateMap`; it re-derives the packing from
# the index definition and verifies the coefficient by an exact ratio test.

"""Packed length `n(n+1)/2` of `svec(P)`, `P ∈ S^n`."""
oracle_svec_length(n::Integer) = div(n * (n + 1), 2)

"""
    oracle_svec_packed_index(n) -> (rows, cols)

The `(row, column)` of every packed position in lower-column-major order,
derived directly from the loop `for j in 1:n, for i in j:n`.
"""
function oracle_svec_packed_index(n::Integer)
    rows = Vector{Int}(undef, oracle_svec_length(n))
    cols = Vector{Int}(undef, oracle_svec_length(n))
    position = 1
    for j in 1:n, i in j:n
        rows[position] = i
        cols[position] = j
        position += 1
    end
    return rows, cols
end

"""`true` at the packed positions that carry the √2 factor (off-diagonal)."""
function oracle_svec_is_scaled(n::Integer)
    rows, cols = oracle_svec_packed_index(n)
    return [rows[k] != cols[k] for k in eachindex(rows)]
end

"""Exact raw lower packing of `M` without the √2 scaling (rounded nowhere)."""
function oracle_svec_raw_packed(M, n::Integer)
    rows, cols = oracle_svec_packed_index(n)
    return Rational{BigInt}[rational_from_float(M[rows[k], cols[k]])
                            for k in eachindex(rows)]
end

"""
    oracle_svec_highprec(M, n; bits) -> Vector{BigFloat}

`svec` evaluated at `bits`-bit BigFloat precision: the off-diagonal
entries are multiplied by a √2 computed inside the same precision scope.
"""
function oracle_svec_highprec(M, n::Integer; bits::Integer=512)
    rows, cols = oracle_svec_packed_index(n)
    return setprecision(BigFloat, bits) do
        root = sqrt(BigFloat(2))
        BigFloat[
            rows[k] == cols[k] ? BigFloat(M[rows[k], cols[k]]) :
            BigFloat(M[rows[k], cols[k]]) * root
            for k in eachindex(rows)
        ]
    end
end

# ---------------------------------------------------------------------------
# 5. RSOC → SOC oracle
# ---------------------------------------------------------------------------
#
# Frozen definition (src/program/transforms_rsoc.jl header):
#
#     M(u, v, w) = ( (u+v)/√2, (u-v)/√2, w )
#
# so `M = [a a 0…; a -a 0…; 0 0 I]` with `a = 1/√2`.  Built here from the
# definition with a high-precision `a`; the kernel's construction order
# (`inv(sqrt(T(2)))`) is deliberately NOT reproduced.

function oracle_rsoc_map_highprec(n::Integer; bits::Integer=512, T::Type=BigFloat)
    n >= 3 || throw(ArgumentError("RSOC dimension must be >= 3, got $n"))
    return setprecision(BigFloat, bits) do
        a = one(BigFloat) / sqrt(BigFloat(2))
        M = zeros(T, n, n)
        M[1, 1] = T(a)
        M[1, 2] = T(a)
        M[2, 1] = T(a)
        M[2, 2] = T(-a)
        for k in 3:n
            M[k, k] = one(T)
        end
        M
    end
end

function oracle_rsoc_apply_highprec(u, v, tail; bits::Integer=512)
    return setprecision(BigFloat, bits) do
        a = one(BigFloat) / sqrt(BigFloat(2))
        first = (BigFloat(u) + BigFloat(v)) * a
        second = (BigFloat(u) - BigFloat(v)) * a
        return (first, second, BigFloat[BigFloat(w) for w in tail])
    end
end

# ---------------------------------------------------------------------------
# 6. HSD residual oracle
# ---------------------------------------------------------------------------
#
# Frozen equations (src/hsd/hsd.jl):
#     rP = A x + s − b·τ
#     rD = Aᵀ y + c·τ
#     rG = cᵀ x + bᵀ y + κ
#     complementarity = sᵀ y + τ·κ
#     μ  = complementarity / (ν + 1)
#
# The exact oracle returns the true value together with the per-row term
# magnitudes.  Those magnitudes are what a *scale-aware* tolerance must use:
# a strongly cancelling residual can be O(1) relative to itself while being
# O(u) relative to its terms, and no test may pretend otherwise.

function oracle_residual_exact(A, b, c, x, y, s, tau, kappa, nu::Integer)
    m = length(b)
    n = length(c)
    rP = Vector{Rational{BigInt}}(undef, m)
    rD = Vector{Rational{BigInt}}(undef, n)
    rP_terms = Vector{Rational{BigInt}}(undef, m)
    rD_terms = Vector{Rational{BigInt}}(undef, n)
    for k in 1:m
        ax = zero(Rational{BigInt})
        for j in 1:n
            ax += A[k, j] * x[j]
        end
        rP[k] = s[k] - b[k] * tau + ax
        rP_terms[k] = abs(s[k]) + abs(b[k] * tau) + abs(ax)
    end
    for j in 1:n
        aty = zero(Rational{BigInt})
        for i in 1:m
            aty += A[i, j] * y[i]
        end
        rD[j] = aty + c[j] * tau
        rD_terms[j] = abs(aty) + abs(c[j] * tau)
    end
    rG = zero(Rational{BigInt})
    rG_terms = zero(Rational{BigInt})
    for j in 1:n
        rG += c[j] * x[j]
        rG_terms += abs(c[j] * x[j])
    end
    for i in 1:m
        rG += b[i] * y[i]
        rG_terms += abs(b[i] * y[i])
    end
    rG += kappa
    rG_terms += abs(kappa)
    complementarity = tau * kappa
    complementarity_terms = abs(tau * kappa)
    for i in 1:m
        complementarity += s[i] * y[i]
        complementarity_terms += abs(s[i] * y[i])
    end
    return (
        rP=rP, rD=rD, rG=rG,
        complementarity=complementarity,
        mu=complementarity / Rational{BigInt}(nu + 1),
        rP_terms=rP_terms, rD_terms=rD_terms, rG_terms=rG_terms,
        complementarity_terms=complementarity_terms,
    )
end

"""The same residual at `bits`-bit BigFloat, for a second association."""
function oracle_residual_highprec(A, b, c, x, y, s, tau, kappa, nu::Integer;
                                  bits::Integer=512)
    return setprecision(BigFloat, bits) do
        m = length(b)
        n = length(c)
        rP = zeros(BigFloat, m)
        rD = zeros(BigFloat, n)
        for k in 1:m
            ax = zero(BigFloat)
            for j in 1:n
                ax += BigFloat(A[k, j]) * BigFloat(x[j])
            end
            rP[k] = BigFloat(s[k]) - BigFloat(b[k]) * BigFloat(tau) + ax
        end
        for j in 1:n
            aty = zero(BigFloat)
            for i in 1:m
                aty += BigFloat(A[i, j]) * BigFloat(y[i])
            end
            rD[j] = aty + BigFloat(c[j]) * BigFloat(tau)
        end
        rG = zero(BigFloat)
        for j in 1:n
            rG += BigFloat(c[j]) * BigFloat(x[j])
        end
        for i in 1:m
            rG += BigFloat(b[i]) * BigFloat(y[i])
        end
        rG += BigFloat(kappa)
        complementarity = BigFloat(tau) * BigFloat(kappa)
        for i in 1:m
            complementarity += BigFloat(s[i]) * BigFloat(y[i])
        end
        (
            rP=rP, rD=rD, rG=rG,
            complementarity=complementarity,
            mu=complementarity / BigFloat(nu + 1),
        )
    end
end

# ---------------------------------------------------------------------------
# 7. Block-LDLᵀ grammar and exact Bunch–Kaufman reference
# ---------------------------------------------------------------------------
#
# Storage conventions, taken from the *specification* (MFLA
# `factor_blocks`/`factor_permutation` docstrings, BFLA `BFLALDLTFactor`
# docstring) and then verified through the defining identity alone:
#
#   * `blocks_raw[k] ∈ {1, 2, 0}`: `1` starts a 1×1 pivot, `2` starts a 2×2
#     pivot, `0` marks the consumed second row of a 2×2 pivot.
#   * `blocks_compact` is the projection of `blocks_raw` to block sizes;
#     `sum(blocks_compact) == n`.
#   * `pivots[k]` is the step swap partner of position `k` (for a 2×2 block
#     the partner of position `k+1`).
#   * `perm[i]` is the ORIGINAL index sitting at factored position `i`;
#     the defining identity is `A[perm, perm] = L D Lᵀ`.
#   * packed lower storage: `packed[i, i]` holds `D[i,i]`; `packed[i, j]`
#     (i > j) holds `L[i,j]` except `packed[k+1, k]` for a 2×2 block at
#     `k`, which holds `D[k+1,k]`.  `dsub[k]` records the same `D[k+1,k]`.
#   * `mirroring`: the upper triangle of `packed` is written from the lower
#     triangle by the producer (MFLA `_mirror_lower_to_upper!`) and is
#     never an independent source of truth.
#
# Nothing below reads a kernel's answer: `oracle_bk_ldlt_exact` computes its
# own factorization in exact rational arithmetic, and
# `oracle_ldlt_identity_residual` checks the identity itself.

"""
    LDLTRecord{T}

A block-LDLᵀ record in the convention above.  `T` is the coefficient type
of `packed`, so the same struct carries an exact reference
(`Rational{BigInt}`) and a kernel-produced Float64/BigFloat record.
"""
struct LDLTRecord{T}
    n::Int
    blocks_raw::Vector{UInt8}
    blocks_compact::Vector{Int}
    pivots::Vector{Int}
    perm::Vector{Int}
    packed::Matrix{T}
    dsub::Vector{T}
end

function LDLTRecord(
    blocks_raw::AbstractVector{UInt8},
    compact::AbstractVector{Int},
    pivots::AbstractVector{Int},
    perm::AbstractVector{Int},
    packed::AbstractMatrix{T},
    dsub::AbstractVector,
) where {T}
    n = size(packed, 1)
    return LDLTRecord{T}(
        n, Vector{UInt8}(blocks_raw), Vector{Int}(compact),
        Vector{Int}(pivots), Vector{Int}(perm),
        Matrix{T}(packed), Vector{T}(dsub),
    )
end

"""Project the raw 1/2/0 grammar to compact block sizes."""
function oracle_compact_from_raw(raw::AbstractVector{UInt8}, n::Int)
    compact = Int[]
    k = 1
    while k <= n
        marker = raw[k]
        if marker == UInt8(1)
            push!(compact, 1)
            k += 1
        elseif marker == UInt8(2) && k < n && raw[k + 1] == UInt8(0)
            push!(compact, 2)
            k += 2
        else
            throw(ArgumentError(
                "invalid 1/2/0 block grammar at position $k: $(raw)",
            ))
        end
    end
    return compact
end

"""Expand compact block sizes back to the raw 1/2/0 grammar."""
function oracle_raw_from_compact(compact::AbstractVector{Int})
    raw = UInt8[]
    for block in compact
        block == 1 && push!(raw, UInt8(1))
        block == 2 && push!(raw, UInt8(2), UInt8(0))
        block in (1, 2) || throw(ArgumentError(
            "invalid compact block size $block",
        ))
    end
    return raw
end

function oracle_grammar_is_wellformed(record::LDLTRecord)
    n = record.n
    length(record.blocks_raw) == n || return false
    length(record.pivots) == n || return false
    length(record.perm) == n || return false
    sort(record.perm) == collect(1:n) || return false
    sum(record.blocks_compact) == n || return false
    try
        oracle_raw_from_compact(record.blocks_compact) == record.blocks_raw ||
            return false
    catch
        return false
    end
    k = 1
    while k <= n
        block = record.blocks_raw[k]
        if block == UInt8(1)
            k += 1
        elseif block == UInt8(2)
            k + 1 <= n || return false
            record.blocks_raw[k + 1] == UInt8(0) || return false
            k += 2
        else
            return false
        end
    end
    return true
end

"""
    oracle_perm_from_pivots(blocks_raw, pivots) -> Vector{Int}

Replay the step-swap record into `perm[i] = original index at factored
position i`, exactly as the specification states it.  The result is only
accepted because `oracle_ldlt_identity_residual` then verifies
`A[perm, perm] = L D Lᵀ`; the rule is not trusted on its own.
"""
function oracle_perm_from_pivots(blocks_raw::AbstractVector{UInt8},
                                 pivots::AbstractVector{Int})
    n = length(blocks_raw)
    perm = collect(1:n)
    k = 1
    while k <= n
        block = blocks_raw[k]
        if block == UInt8(1)
            pivot = pivots[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]
            k += 1
        elseif block == UInt8(2) && k < n
            pivot = pivots[k]
            perm[k + 1], perm[pivot] = perm[pivot], perm[k + 1]
            k += 2
        else
            break
        end
    end
    return perm
end

"""
    oracle_lower_factors(record) -> (L, D)

Materialize the unit-lower `L` and the block-diagonal `D` from the packed
record, using only the storage convention above.  For a 2×2 block at `k`
the entry `packed[k+1, k]` is `D[k+1,k]` (not a multiplier).  The result is
dense `Rational{BigInt}` when the record is exact.
"""
function oracle_lower_factors(record::LDLTRecord{T}) where {T}
    n = record.n
    L = Matrix{T}(undef, n, n)
    D = zeros(T, n, n)
    fill!(L, zero(T))
    for i in 1:n
        L[i, i] = one(T)
    end
    k = 1
    for block in record.blocks_compact
        if block == 1
            D[k, k] = record.packed[k, k]
            k += 1
        else
            d11 = record.packed[k, k]
            d21 = record.packed[k + 1, k]
            d22 = record.packed[k + 1, k + 1]
            D[k, k] = d11
            D[k, k + 1] = d21
            D[k + 1, k] = d21
            D[k + 1, k + 1] = d22
            k += 2
        end
    end
    # Multipliers strictly below the diagonal, skipping the 2×2 subdiagonal
    # slot, which is D's off-diagonal.
    k = 1
    for block in record.blocks_compact
        if block == 1
            for i in (k + 1):n
                L[i, k] = record.packed[i, k]
            end
            k += 1
        else
            for i in (k + 2):n
                L[i, k] = record.packed[i, k]
                L[i, k + 1] = record.packed[i, k + 1]
            end
            k += 2
        end
    end
    return L, D
end

"""
    oracle_ldlt_identity_residual(A, record) -> (residual, L, D)

`A[perm, perm] − L D Lᵀ` in exact arithmetic.  This is the defining
identity of the block-LDLᵀ grammar; it is the arbiter for the permutation,
the block structure, the 2×2 subdiagonal placement and the mirroring
convention all at once.
"""
function oracle_ldlt_identity_residual(A::AbstractMatrix{Rational{BigInt}},
                                       record::LDLTRecord{Rational{BigInt}})
    n = record.n
    size(A) == (n, n) || throw(DimensionMismatch("A is not $(n)×$(n)"))
    L, D = oracle_lower_factors(record)
    permuted = A[record.perm, record.perm]
    residual = permuted - L * D * transpose(L)
    return residual, L, D
end

# --- exact Bunch–Kaufman pivot rule ---------------------------------------
#
# α = (1 + √17)/8 is irrational, so every comparison against `α·t` is
# decided exactly by clearing the denominator and squaring:
#
#   |a| >= α·t            ⟺  8|a| − t >= √17·t
#                         ⟺  (8|a| − t >= 0)  and  (8|a| − t)² >= 17 t²
#
# This makes the reference pivot selection exact and reproducible without
# ever forming α in floating point.

const BK_ALPHA_DESCRIPTION =
    "(1+sqrt(17))/8, compared exactly by squaring 8|a| >= (1+sqrt(17))t"

function _bk_ge_alpha_times(a::Rational{BigInt}, t::Rational{BigInt})
    t >= 0 || throw(ArgumentError("comparison target must be nonnegative"))
    lhs = 8 * abs(a) - t
    lhs >= 0 || return false
    return lhs * lhs >= 17 * t * t
end

"""Exact Bunch–Kaufman step decision, mirroring the published rule only."""
function _bk_select_exact(S::AbstractMatrix{Rational{BigInt}}, k::Int)
    n = size(S, 1)
    absakk = abs(S[k, k])
    k == n && return iszero(absakk) ? (0, k) : (1, k)
    imax = k + 1
    colmax = abs(S[imax, k])
    for row in (k + 2):n
        candidate = abs(S[row, k])
        if candidate > colmax
            colmax = candidate
            imax = row
        end
    end
    max(absakk, colmax) == 0 && return (0, k)
    _bk_ge_alpha_times(absakk, colmax) && return (1, k)
    rowmax = zero(Rational{BigInt})
    for column in k:(imax - 1)
        rowmax = max(rowmax, abs(S[imax, column]))
    end
    for row in (imax + 1):n
        rowmax = max(rowmax, abs(S[row, imax]))
    end
    if _bk_ge_alpha_times(absakk * rowmax, colmax * colmax)
        return (1, k)
    elseif _bk_ge_alpha_times(abs(S[imax, imax]), rowmax)
        return (1, imax)
    end
    return (2, imax)
end

@inline function _exact_sym_swap!(S::AbstractMatrix{Rational{BigInt}},
                                  r::Int, s::Int)
    r == s && return S
    n = size(S, 1)
    for c in 1:n
        S[r, c], S[s, c] = S[s, c], S[r, c]
    end
    for i in 1:n
        S[i, r], S[i, s] = S[i, s], S[i, r]
    end
    return S
end

"""
    oracle_bk_ldlt_exact(A) -> (record, status, L, D)

Exact rational Bunch–Kaufman block-LDLᵀ.  The Schur complement is kept as
a pure symmetric matrix (multipliers are accumulated separately), so the
symmetric swaps are exact row/column transpositions and the resulting
`perm` satisfies `A[perm, perm] = L D Lᵀ` by construction — which
`oracle_ldlt_identity_residual` then verifies rather than assumes.

`status` is `:success`, `:zero_pivot` (1×1 pivot exactly zero) or
`:singular_2x2` (2×2 pivot with exactly zero determinant).  When the
factorization stops early, `blocks_compact` describes only the accepted
pivots and the trailing positions are `:unfactored`.
"""
function oracle_bk_ldlt_exact(A::AbstractMatrix{Rational{BigInt}})
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A must be square"))
    # `S` is the pure symmetric Schur complement (never holds multipliers),
    # so a symmetric swap is an exact row/column transposition of `S`.
    S = Matrix{Rational{BigInt}}(A)
    # `multipliers` accumulates the L columns separately.  A symmetric swap
    # of active positions (k, r) must also exchange the already-computed
    # multipliers of those two rows: that is what keeps the final
    # `A[perm, perm] = L D Lᵀ` identity true.
    multipliers = zeros(Rational{BigInt}, n, n)
    blocks = Vector{UInt8}(undef, n)
    fill!(blocks, UInt8(0))
    pivots = collect(1:n)
    dsub = zeros(Rational{BigInt}, n)
    status = :success
    k = 1
    while k <= n
        block_size, pivot = _bk_select_exact(S, k)
        if block_size == 0
            status = :zero_pivot
            break
        elseif block_size == 1
            pivots[k] = pivot
            blocks[k] = UInt8(1)
            if pivot != k
                _exact_sym_swap!(S, k, pivot)
                for j in 1:(k - 1)
                    multipliers[k, j], multipliers[pivot, j] =
                        multipliers[pivot, j], multipliers[k, j]
                end
            end
            d = S[k, k]
            if iszero(d)
                status = :zero_pivot
                break
            end
            for i in (k + 1):n
                multipliers[i, k] = S[i, k] / d
            end
            for j in (k + 1):n, i in j:n
                S[i, j] = S[i, j] -
                          multipliers[i, k] * d * multipliers[j, k]
                S[j, i] = S[i, j]
            end
            k += 1
        else
            k < n || (status = :zero_pivot; break)
            pivots[k] = pivot
            if pivot != k + 1
                _exact_sym_swap!(S, k + 1, pivot)
                for j in 1:(k - 1)
                    multipliers[k + 1, j], multipliers[pivot, j] =
                        multipliers[pivot, j], multipliers[k + 1, j]
                end
            end
            d11 = S[k, k]
            d21 = S[k + 1, k]
            d22 = S[k + 1, k + 1]
            determinant = d11 * d22 - d21 * d21
            if iszero(determinant)
                status = :singular_2x2
                break
            end
            for i in (k + 2):n
                first = S[i, k]
                second = S[i, k + 1]
                multipliers[i, k] = (d22 * first - d21 * second) / determinant
                multipliers[i, k + 1] =
                    (d11 * second - d21 * first) / determinant
            end
            for j in (k + 2):n, i in j:n
                coefficient_first =
                    d11 * multipliers[j, k] + d21 * multipliers[j, k + 1]
                coefficient_second =
                    d21 * multipliers[j, k] + d22 * multipliers[j, k + 1]
                S[i, j] = S[i, j] - multipliers[i, k] * coefficient_first -
                          multipliers[i, k + 1] * coefficient_second
                S[j, i] = S[i, j]
            end
            # The kernel explicitly clears the consumed subdiagonal slots.
            S[k + 1, k] = zero(Rational{BigInt})
            S[k, k + 1] = zero(Rational{BigInt})
            blocks[k] = UInt8(2)
            blocks[k + 1] = UInt8(0)
            dsub[k] = d21
            k += 2
        end
    end

    accepted = k - 1
    raw = copy(blocks)
    if accepted < n
        for index in (accepted + 1):n
            raw[index] = UInt8(0)
        end
    end
    pivots_raw = copy(pivots)
    # Pack: unit lower `multipliers`, with D's 1×1 diagonals and 2×2 blocks
    # overlaid, and D's 2×2 subdiagonal written into the one slot that is
    # not a multiplier.
    packed = zeros(Rational{BigInt}, n, n)
    k = 1
    while k <= accepted
        if raw[k] == UInt8(1)
            packed[k, k] = S[k, k]
            for i in (k + 1):n
                packed[i, k] = multipliers[i, k]
            end
            k += 1
        else
            packed[k, k] = S[k, k]
            packed[k + 1, k] = dsub[k]
            packed[k + 1, k + 1] = S[k + 1, k + 1]
            for i in (k + 2):n
                packed[i, k] = multipliers[i, k]
                packed[i, k + 1] = multipliers[i, k + 1]
            end
            k += 2
        end
    end
    # Mirror the lower triangle into the upper triangle: the producers write
    # both triangles (`_mirror_lower_to_upper!`), and the identity check must
    # not depend on which triangle is authoritative.
    for j in 1:n, i in (j + 1):n
        packed[j, i] = packed[i, j]
    end
    compact = accepted == n ? oracle_compact_from_raw(raw, n) : Int[]
    perm = oracle_perm_from_pivots(raw, pivots_raw)
    record = LDLTRecord(raw, compact, pivots_raw, perm, packed, dsub)
    L, D = oracle_lower_factors(record)
    return record, status, L, D
end

# --- N / T solves from the defining identity ------------------------------

"""
    exact_dense_solve(M, rhs) -> Vector{Rational{BigInt}}

Exact Gaussian elimination with partial pivoting on `Rational{BigInt}`.
Used as a SECOND, structurally different exact solver: it never touches the
block-LDLᵀ factor, so agreement between it and the factor sequence is real
evidence rather than a restatement.
"""
function exact_dense_solve(M::AbstractMatrix{Rational{BigInt}},
                           rhs::AbstractVector{Rational{BigInt}})
    n = size(M, 1)
    size(M, 2) == n || throw(DimensionMismatch("M must be square"))
    length(rhs) == n || throw(DimensionMismatch("rhs length != n"))
    A = Matrix{Rational{BigInt}}(M)
    b = Vector{Rational{BigInt}}(rhs)
    for column in 1:n
        pivot = column
        for row in (column + 1):n
            if abs(A[row, column]) > abs(A[pivot, column])
                pivot = row
            end
        end
        A[pivot, column] == 0 && throw(ArgumentError("singular system"))
        if pivot != column
            for j in 1:n
                A[column, j], A[pivot, j] = A[pivot, j], A[column, j]
            end
            b[column], b[pivot] = b[pivot], b[column]
        end
        for row in (column + 1):n
            factor = A[row, column] / A[column, column]
            iszero(factor) && continue
            for j in column:n
                A[row, j] -= factor * A[column, j]
            end
            b[row] -= factor * b[column]
        end
    end
    x = zeros(Rational{BigInt}, n)
    for row in n:-1:1
        acc = b[row]
        for j in (row + 1):n
            acc -= A[row, j] * x[j]
        end
        x[row] = acc / A[row, row]
    end
    return x
end

"""
    oracle_ldlt_solve_exact(record, rhs, op) -> Vector{Rational{BigInt}}

Solve `A x = rhs` (`op === :N`) or `Aᵀ x = rhs` (`op === :T`) exactly.

`A[perm, perm] = L D Lᵀ` is the defining identity, so the two systems are

  * `:N` — `(L D Lᵀ) z = P b`, solved through the factor sequence
    (forward `L`, block `D`, backward `Lᵀ`, unpermute);
  * `:T` — `(L D Lᵀ)ᵀ z = P b`, solved by an entirely different exact
    algorithm (dense elimination on the explicitly transposed product).

For a symmetric `A` the two right-hand sides are equal and so must the two
answers be, exactly.  Deriving them by two different algorithms is what
makes their agreement a test rather than a tautology.
"""
function oracle_ldlt_solve_exact(record::LDLTRecord{Rational{BigInt}},
                                 rhs::AbstractVector{Rational{BigInt}},
                                 op::Symbol)
    op in (:N, :T) || throw(ArgumentError("solve op must be :N or :T"))
    n = record.n
    length(rhs) == n || throw(DimensionMismatch("rhs length != n"))
    L, D = oracle_lower_factors(record)
    z = Vector{Rational{BigInt}}(undef, n)
    for i in 1:n
        z[i] = rhs[record.perm[i]]
    end
    if op === :T
        # (L D Lᵀ)ᵀ as an explicit matrix, solved by dense elimination.
        product = transpose(L * D * transpose(L))
        solution = exact_dense_solve(product, z)
        out = Vector{Rational{BigInt}}(undef, n)
        for i in 1:n
            out[record.perm[i]] = solution[i]
        end
        return out
    end
    # :N — the factor sequence.
    y = zeros(Rational{BigInt}, n)
    for i in 1:n
        acc = z[i]
        for j in 1:(i - 1)
            acc -= L[i, j] * y[j]
        end
        y[i] = acc
    end
    w = copy(y)
    k = 1
    for block in record.blocks_compact
        if block == 1
            w[k] = y[k] / D[k, k]
            k += 1
        else
            d11 = D[k, k]
            d21 = D[k + 1, k]
            d22 = D[k + 1, k + 1]
            determinant = d11 * d22 - d21 * d21
            w[k] = (d22 * y[k] - d21 * y[k + 1]) / determinant
            w[k + 1] = (d11 * y[k + 1] - d21 * y[k]) / determinant
            k += 2
        end
    end
    x = zeros(Rational{BigInt}, n)
    for i in n:-1:1
        acc = w[i]
        for j in (i + 1):n
            acc -= L[j, i] * x[j]
        end
        x[i] = acc
    end
    out = Vector{Rational{BigInt}}(undef, n)
    for i in 1:n
        out[record.perm[i]] = x[i]
    end
    return out
end

"""
    oracle_ldlt_solve(record, rhs, op)

Dispatch wrapper used by the tests; `rhs` may be a vector or a matrix
(matrix RHS is one solve per column, exactly as the contract states).
"""
function oracle_ldlt_solve(record::LDLTRecord{Rational{BigInt}},
                           rhs::AbstractVector, op::Symbol)
    return oracle_ldlt_solve_exact(record, exact_vector(rhs), op)
end

function oracle_ldlt_solve(record::LDLTRecord{Rational{BigInt}},
                           rhs::AbstractMatrix, op::Symbol)
    n = record.n
    size(rhs, 1) == n || throw(DimensionMismatch("rhs rows != n"))
    columns = [oracle_ldlt_solve_exact(record, exact_vector(rhs[:, c]), op)
               for c in axes(rhs, 2)]
    return reduce(hcat, columns)
end

# --- exact inertia from the block grammar ---------------------------------

"""Number of factored positions in a possibly partial record."""
oracle_accepted_pivots(record::LDLTRecord) = sum(_accepted_blocks(record))

"""The accepted block sizes of a possibly partial record, in order."""
function _accepted_blocks(record::LDLTRecord)
    blocks = Int[]
    k = 1
    n = record.n
    while k <= n
        marker = record.blocks_raw[k]
        if marker == UInt8(1)
            push!(blocks, 1)
            k += 1
        elseif marker == UInt8(2) && k < n && record.blocks_raw[k + 1] == UInt8(0)
            push!(blocks, 2)
            k += 2
        else
            break
        end
    end
    return blocks
end

"""
    oracle_block_inertia_exact(record) -> (positive, negative, zero)

Inertia of `D`, hence of `A[perm, perm]`, from the defining 2×2 spectral
facts: with `D₂ = [d11 d21; d21 d22]`, `det = d11·d22 − d21²` and
`tr = d11 + d22`,

  * `det < 0`  → one positive and one negative eigenvalue;
  * `det > 0`  → two eigenvalues with the sign of `tr`;
  * `det == 0` → one zero eigenvalue plus one with the sign of `tr`.

No square root and no floating point is used, so the count is exact.
For a partial (rank-deficient) record the accepted pivots are counted and
the caller adds the unfactored trailing positions as zeros.
"""
function oracle_block_inertia_exact(record::LDLTRecord{Rational{BigInt}})
    positive = 0
    negative = 0
    zeros_count = 0
    k = 1
    # Walk the RAW grammar, not the compact one: a partial factorization
    # (rank-deficient input) has an empty `blocks_compact` but still has a
    # well-defined inertia for the pivots that were accepted.  A `0` marker
    # either continues a 2×2 block or terminates the walk.
    for block in _accepted_blocks(record)
        if block == 1
            value = record.packed[k, k]
            if value > 0
                positive += 1
            elseif value < 0
                negative += 1
            else
                zeros_count += 1
            end
            k += 1
        else
            d11 = record.packed[k, k]
            d21 = record.packed[k + 1, k]
            d22 = record.packed[k + 1, k + 1]
            determinant = d11 * d22 - d21 * d21
            trace = d11 + d22
            if determinant < 0
                positive += 1
                negative += 1
            elseif determinant > 0
                trace > 0 ? (positive += 2) : (negative += 2)
            else
                zeros_count += 1
                trace > 0 && (positive += 1)
                trace < 0 && (negative += 1)
                iszero(trace) && (zeros_count += 1)
            end
            k += 2
        end
    end
    return positive, negative, zeros_count
end

"""
    oracle_characteristic_polynomial(A) -> Vector{Rational{BigInt}}

`det(λI − A)` in exact arithmetic, highest degree first, by the
Faddeev–LeVerrier recurrence
`M_k = A·M_{k−1} + c_k·I`, `c_{k+1} = −tr(A·M_k)/k`, `M_0 = 0`.

This is a completely different route to the spectrum than any
factorization: it touches no pivots, no blocks and no ordering.
"""
function oracle_characteristic_polynomial(A::AbstractMatrix{Rational{BigInt}})
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A must be square"))
    identity_matrix = zeros(Rational{BigInt}, n, n)
    for i in 1:n
        identity_matrix[i, i] = one(Rational{BigInt})
    end
    M = zeros(Rational{BigInt}, n, n)
    coefficients = zeros(Rational{BigInt}, n + 1)
    coefficients[1] = one(Rational{BigInt})
    for k in 1:n
        M = A * M + coefficients[k] * identity_matrix
        trace_value = zero(Rational{BigInt})
        product = A * M
        for i in 1:n
            trace_value += product[i, i]
        end
        coefficients[k + 1] = -trace_value / k
    end
    return coefficients
end

"""Number of sign changes in a coefficient sequence, ignoring zeros."""
function _sign_changes(coefficients)
    changes = 0
    previous = 0
    for coefficient in coefficients
        iszero(coefficient) && continue
        sign_value = coefficient > 0 ? 1 : -1
        previous != 0 && sign_value != previous && (changes += 1)
        previous = sign_value
    end
    return changes
end

"""
    oracle_inertia_from_characteristic_polynomial(A)

Exact inertia from `det(λI − A)` alone.

`A` is symmetric, so every root of the characteristic polynomial is real;
for a real-rooted polynomial **Descartes' rule of signs is exact**, not a
bound.  The number of positive roots is the number of sign changes in the
coefficient sequence, and the number of negative roots is the number of
sign changes in `p(−λ)`.  The remainder are zero eigenvalues.

This shares no code path with the block-LDLᵀ grammar: it is the
independent arbiter for every inertia claim in the fixtures.
"""
function oracle_inertia_from_characteristic_polynomial(
    A::AbstractMatrix{Rational{BigInt}},
)
    n = size(A, 1)
    coefficients = oracle_characteristic_polynomial(A)
    positive = _sign_changes(coefficients)
    negated = Rational{BigInt}[
        iseven(n - (index - 1)) ? coefficients[index] : -coefficients[index]
        for index in eachindex(coefficients)
    ]
    negative = _sign_changes(negated)
    zeros_count = n - positive - negative
    return (positive, negative, zeros_count)
end

# ---------------------------------------------------------------------------
# 8. 2×2 pivot normalization oracle
# ---------------------------------------------------------------------------
#
# Both providers normalize the 2×2 pivot before solving it, with different
# (and both valid) row scalings:
#
#   S12 (MFLA, `_ldlt_solve_2x2`):   scale = max(|d11|, |d21|, |d22|)
#   S17 (BFLA, `_ldlt_2x2_normalize_rows!`): row scales
#                                    s1 = max(|d11|, |e|), s2 = max(|e|, |d22|)
#
# The reference below is the DEFINING SYSTEM
#
#       [d11  e ; e  d22] [x1; x2] = [y1; y2]
#
# solved exactly in rationals through each documented normalization.  The
# normalization is therefore verified against the identity it exists to
# serve, not against a kernel's output.

"""
    oracle_2x2_solve_mfla(d11, e, d22, y1, y2)

MFLA/S12 normalization: one common ∞-norm scale, then a two-branch
|a| >= |b| elimination.  Returns `(x1, x2)` or `nothing` when the
normalized pivot is exactly singular.
"""
function oracle_2x2_solve_mfla(d11::T, e::T, d22::T, y1::T, y2::T) where {T}
    scale = max(abs(d11), abs(e), abs(d22))
    iszero(scale) && return nothing
    a = d11 / scale
    b = e / scale
    c = d22 / scale
    r1 = y1 / scale
    r2 = y2 / scale
    if abs(a) >= abs(b)
        iszero(a) && return nothing
        t = b / a
        u = c - t * b
        iszero(u) && return nothing
        x2 = (r2 - t * r1) / u
        x1 = (r1 - b * x2) / a
        return (x1, x2)
    end
    iszero(b) && return nothing
    t = a / b
    u = b - t * c
    iszero(u) && return nothing
    x2 = (r1 - t * r2) / u
    x1 = (r2 - c * x2) / b
    return (x1, x2)
end

"""
    oracle_2x2_solve_bfla(d11, e, d22, y1, y2)

BFLA/S17 normalization: independent positive row scales, then Cramer on
the scaled system `[a e1; e2 c] [x1; x2] = [y1/s1; y2/s2]`.  Returns
`(x1, x2)` or `nothing` when the normalized determinant is exactly zero.
"""
function oracle_2x2_solve_bfla(d11::T, e::T, d22::T, y1::T, y2::T) where {T}
    row_scale_1 = max(abs(d11), abs(e))
    row_scale_2 = max(abs(e), abs(d22))
    (iszero(row_scale_1) || iszero(row_scale_2)) && return nothing
    a = d11 / row_scale_1
    e1 = e / row_scale_1
    e2 = e / row_scale_2
    c = d22 / row_scale_2
    determinant = a * c - e1 * e2
    iszero(determinant) && return nothing
    scaled_y1 = y1 / row_scale_1
    scaled_y2 = y2 / row_scale_2
    x1 = (c * scaled_y1 - e1 * scaled_y2) / determinant
    x2 = (a * scaled_y2 - e2 * scaled_y1) / determinant
    return (x1, x2)
end

oracle_2x2_solve(::Val{:mfla}, d11, e, d22, y1, y2) =
    oracle_2x2_solve_mfla(d11, e, d22, y1, y2)
oracle_2x2_solve(::Val{:bfla}, d11, e, d22, y1, y2) =
    oracle_2x2_solve_bfla(d11, e, d22, y1, y2)

"""
    oracle_2x2_defining_residual(d11, e, d22, y1, y2, x1, x2)

The defining system's residual.  Exact zero in rational arithmetic is the
acceptance condition for a normalization formula.
"""
function oracle_2x2_defining_residual(d11, e, d22, y1, y2, x1, x2)
    return (d11 * x1 + e * x2 - y1, e * x1 + d22 * x2 - y2)
end

"""
    oracle_2x2_normalized_determinant(variant, d11, e, d22)
        -> (divisor, cofactor, true_determinant)

The quantity the normalization actually DIVIDES BY, together with the
cofactor that makes the identity

    cofactor * divisor == d11·d22 − e²

exact.  `agreement of signs` is deliberately NOT asserted: the two
documented normalizations legitimately produce divisors of opposite sign
in one of their branches (MFLA's `u` is `det_scaled/a` in the |a| ≥ |b|
branch and `−det_scaled/b` otherwise).  What must hold — and what a wrong
normalization breaks — is

  * `cofactor·divisor == true_determinant` exactly, and
  * `iszero(divisor) == iszero(true_determinant)`, i.e. the normalization
    detects exactly the singular pivots and no others.
"""
function oracle_2x2_normalized_determinant(::Val{:mfla}, d11::T, e::T,
                                           d22::T) where {T}
    true_determinant = d11 * d22 - e * e
    scale = max(abs(d11), abs(e), abs(d22))
    if iszero(scale)
        return zero(T), zero(T), true_determinant
    end
    a = d11 / scale
    b = e / scale
    c = d22 / scale
    if abs(a) >= abs(b)
        # u = c − (b/a)·b = det_scaled / a ; cofactor = a·scale²
        iszero(a) && return zero(T), zero(T), true_determinant
        u = c - (b / a) * b
        return u, a * scale * scale, true_determinant
    end
    # u = b − (a/b)·c = −det_scaled / b ; cofactor = −b·scale²
    iszero(b) && return zero(T), zero(T), true_determinant
    u = b - (a / b) * c
    return u, -b * scale * scale, true_determinant
end

function oracle_2x2_normalized_determinant(::Val{:bfla}, d11::T, e::T,
                                           d22::T) where {T}
    true_determinant = d11 * d22 - e * e
    row_scale_1 = max(abs(d11), abs(e))
    row_scale_2 = max(abs(e), abs(d22))
    if iszero(row_scale_1) || iszero(row_scale_2)
        return zero(T), zero(T), true_determinant
    end
    a = d11 / row_scale_1
    e1 = e / row_scale_1
    e2 = e / row_scale_2
    c = d22 / row_scale_2
    # det_normalized = det / (s1·s2)
    return a * c - e1 * e2, row_scale_1 * row_scale_2, true_determinant
end

# ---------------------------------------------------------------------------
# 9. Kernel-record adapter (LAPACK `bunchkaufman` ipiv)
# ---------------------------------------------------------------------------
#
# This reads a record PRODUCED BY A KERNEL; it is an adapter, not part of
# the reference arithmetic.  LAPACK encodes the block grammar in `ipiv`:
# `ipiv[k] > 0` is a 1×1 pivot, `ipiv[k] < 0` is a 2×2 pivot occupying rows
# k, k+1 and interchanged with row `|ipiv[k]|`.  Decoding it gives an
# independent grammar record whose `perm` can be checked against the
# kernel's own `p` and against the defining identity.

function decode_lapack_ipiv_blocks(ipiv::AbstractVector{<:Integer})
    n = length(ipiv)
    compact = Int[]
    k = 1
    while k <= n
        entry = ipiv[k]
        if entry > 0
            push!(compact, 1)
            k += 1
        elseif entry < 0 && k < n && ipiv[k + 1] == entry
            push!(compact, 2)
            k += 2
        else
            throw(ArgumentError(
                "invalid LAPACK ipiv block encoding at position $k: $(ipiv)",
            ))
        end
    end
    return compact
end

end # module A01Oracles
