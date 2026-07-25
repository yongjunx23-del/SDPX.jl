#=====================================================================
    BigFloat fast path (§1.5). Plain BigFloat arithmetic allocates a
    fresh MPFR object on every `+`/`*` (P1); MutableArithmetics wraps
    the mutating `mpfr_add`/`mpfr_mul`/`mpfr_fma` C calls as
    `operate!`/`operate_to!`/`buffered_operate!`, letting kdot!/kaxpby!
    run at O(1) allocations regardless of array size.

    API verified against the installed MutableArithmetics v1.8.0
    source (not assumed from memory): `buffered_operate!(buffer, op,
    accumulator, args...)` — note `op` precedes the accumulator, which
    is the operand order the package actually uses.

    kchol!/ktrsm!/ktrmm! below were *not* part of the first pass (they
    initially just routed to Base's generic cholesky!/ldiv!/rmul!,
    which — measured directly — allocates a fresh BigFloat on every
    scalar +,-,*,/ internally, since Base's generic dense linear
    algebra has no idea BigFloat happens to be mutable). A direct
    before/after benchmark against the original solver on a matched
    instance showed 321x fewer allocations and 184x faster wall-clock
    at Float64, but only 1.7x fewer allocations and ~1.0x wall-clock
    at BigFloat — i.e. eliminating the array-level allocation (SS
    panels, per-i slices, redundant factorizations) wasn't enough; the
    *scalar*-level BigFloat allocation inside cholesky!/ldiv!/rmul!
    dominates. Cholesky–Banachiewicz and forward/back substitution are
    themselves just sequences of dot products and one scalar
    sqrt/divide per entry, so they're written here directly in terms
    of the already-verified kdot!, giving the same O(1)-allocation
    property Base's generic path can't. This is exactly the kernel-seam
    hook the plan describes for these functions, filled in once
    profiling showed it was needed rather than optional polish.
=====================================================================#

import MutableArithmetics as MA

# ---- kdot: simple form (2 O(1) allocations per call: the returned
#     accumulator plus one scratch buffer) ----

function kdot(x::AbstractArray{BigFloat}, y::AbstractArray{BigFloat})
    acc = BigFloat()
    buf = BigFloat()
    return kdot!(acc, buf, x, y)
end

# ---- kdot!: fully buffered form — zero allocation once (acc, buf)
#     are supplied. Used inside the O(m²)-pair Schur accumulation loop
#     (schur.jl), where `acc`/`buf` are allocated once per block per
#     iteration and reused across every pair. ----

function kdot!(acc::BigFloat, buf::BigFloat, x::AbstractArray{BigFloat}, y::AbstractArray{BigFloat})
    MA.operate!(zero, acc)
    @inbounds for i in eachindex(x, y)
        MA.buffered_operate!(buf, MA.add_mul, acc, x[i], y[i])
    end
    return acc
end

function kdot_columns!(
    acc::BigFloat,
    buf::BigFloat,
    panel::AbstractMatrix{BigFloat},
    first_column::Int,
    second_column::Int,
    rows::Int,
)
    MA.operate!(zero, acc)
    @inbounds for row in 1:rows
        MA.buffered_operate!(
            buf,
            MA.add_mul,
            acc,
            panel[row, first_column],
            panel[row, second_column],
        )
    end
    return acc
end

# Generic fallback: for BLAS/bitstypes there is nothing to buffer —
# Float64 arithmetic doesn't heap-allocate at all, so this is already
# O(1) (in fact O(0)) without any special-casing.
kdot!(acc, buf, x, y) = kdot(x, y)

# ---- kaxpby! : Y = α·X + β·Y, one shared scratch BigFloat for the
#     whole call regardless of array size ----

"""
    kaxpby!(α, X, β, Y) -> Y   (BigFloat)

`Y = α·X + β·Y`, writing a **fresh object** into every slot of `Y`.

An earlier version mutated `Y[i]` in place through MutableArithmetics, which is
faster but silently wrong whenever `Y`'s slots are not distinct objects — and in
Julia they usually are not. `zeros(BigFloat, m, m)` and `fill!(A, zero(BigFloat))`
store *the same* `BigFloat` in every slot (verified: `Y[1,1] === Y[3,3]`), and
`copyto!`/`Matrix{BigFloat}(A)` copy references rather than values. Mutating one
slot then rewrites the whole array: `kaxpby!(1, X, 1, zeros(BigFloat,3,3))`
produced 45 in all nine entries — the running total accumulated into one shared
object — instead of `X`.

That is the actual reason threaded BigFloat gave results "off by orders of
magnitude with no error": the per-bin `zeros(BigFloat, m, m)` accumulators are
fully aliased, so the reduction destroyed them. It was never MPFR
thread-unsafety — concurrent MPFR arithmetic at fixed precision is bitwise
identical to serial (verified separately).

Writing a fresh object per slot costs a few allocations per element but is
correct for any input. This kernel is not on the dominant cost path
(`kdot!`/`kmul!` are, and both still use buffered in-place accumulation into
scratch they exclusively own).
"""
function kaxpby!(α::BigFloat, X::AbstractArray{BigFloat}, β::BigFloat, Y::AbstractArray{BigFloat})
    @inbounds for i in eachindex(X, Y)
        Y[i] = α * X[i] + β * Y[i]
    end
    return Y
end

"""
    alloc_zeros(T, dims...) -> Array{T}

Zero-filled array whose elements are **independent objects**.

For bitstypes this is exactly `zeros`. For `BigFloat` it is not: `zeros` and
`fill!` install one shared object in every slot, so any in-place mutation of a
single entry corrupts the whole array. Workspace buffers are allocated through
this so that in-place kernels have a well-defined destination.
"""
alloc_zeros(::Type{T}, dims::Integer...) where {T} = zeros(T, dims...)
function alloc_zeros(::Type{BigFloat}, dims::Integer...)
    A = Array{BigFloat}(undef, dims...)
    @inbounds for i in eachindex(A)
        A[i] = BigFloat(0)
    end
    return A
end

"""
    zero_distinct!(A) -> A

Zero `A` while keeping every slot an independent object (see
[`alloc_zeros`](@ref)). `fill!(A, zero(BigFloat))` would alias them all.
"""
zero_distinct!(A::AbstractArray) = fill!(A, zero(eltype(A)))
function zero_distinct!(A::AbstractArray{BigFloat})
    @inbounds for i in eachindex(A)
        A[i] = BigFloat(0)
    end
    return A
end

# ---- kchol! : in-place lower Cholesky via Cholesky–Banachiewicz.
#     Column-outer-loop order: by the time column j is processed,
#     every entry in columns 1..j-1 (any row) is already finalized, so
#     the row-slices `A[j,1:j-1]`/`A[i,1:j-1]` used in the dot products
#     below are always valid, already-computed L entries — standard
#     algorithm, just re-derived and checked against the textbook
#     formula (A[i,j] = Σ_{p≤min(i,j)} L[i,p]L[j,p]) during development
#     rather than assumed. Only the O(k) sqrt/divide calls per column
#     allocate (non-mutating, so always safe to store into A directly,
#     per this file's aliasing discipline); the O(k²)/O(k³) inner
#     products go through kdot!, which doesn't. ----

function kchol!(A::AbstractMatrix{BigFloat})
    k = size(A, 1)
    k == 0 && return true
    acc = BigFloat()
    buf = BigFloat()
    @inbounds for j in 1:k
        if j > 1
            Lj = view(A, j, 1:(j-1))
            kdot!(acc, buf, Lj, Lj)
            djj = A[j, j] - acc   # non-mutating -, fresh object: safe to store
        else
            djj = A[j, j]
        end
        djj <= 0 && return false
        Ljj = sqrt(djj)            # non-mutating, fresh
        A[j, j] = Ljj
        for i in (j+1):k
            if j > 1
                Li = view(A, i, 1:(j-1))
                Lj = view(A, j, 1:(j-1))
                kdot!(acc, buf, Li, Lj)
                num = A[i, j] - acc
            else
                num = A[i, j]
            end
            A[i, j] = num / Ljj    # non-mutating, fresh
        end
    end
    return true
end

# ---- ktrsm! : X ← L⁻¹X via dot-product forward substitution, one
#     column of X at a time. Safe: each Xc[i] assignment comes from
#     non-mutating -,/ (fresh object), and kdot! only mutates its own
#     acc/buf scratch, never Xc's entries. ----

function ktrsm!(L::AbstractMatrix{BigFloat}, X::AbstractMatrix{BigFloat})
    k = size(L, 1)
    k == 0 && return X
    p = size(X, 2)
    acc = BigFloat()
    buf = BigFloat()
    @inbounds for c in 1:p
        Xc = view(X, :, c)
        for i in 1:k
            if i > 1
                Li = view(L, i, 1:(i-1))
                yi = view(Xc, 1:(i-1))
                kdot!(acc, buf, Li, yi)
                val = Xc[i] - acc
            else
                val = Xc[i]
            end
            Xc[i] = val / L[i, i]
        end
    end
    return X
end

# ---- ktrmm! : X ← X·M, M square lower-triangular, X square (matches
#     the solver's only call site: a k×k block times the k×k Cholesky
#     factor of Y). (X·M)[i,j] = Σ_{p≥j} X[i,p]M[p,j] only involves
#     entries in columns ≥j of X and M's column j (contiguous, since M
#     is column-major) — processing j = 1..k in increasing order reads
#     only not-yet-overwritten columns of X (each iteration writes
#     only column j, and only ever reads columns ≥ its own j).
#
#     A fresh, independent value is needed for each store here (unlike
#     everywhere else in this file, nothing downstream of kdot! applies
#     a further non-mutating op before the store). The first attempt
#     used `copy(acc)` for this and was WRONG — caught by a direct
#     numerical test (a hand-computable 2×2 case gave every output
#     entry equal to the *last* one computed, i.e. every store ended up
#     aliasing the same object): `copy(::BigFloat)` is not a deep copy
#     in Julia — `copy(x) === x` — despite BigFloat being a mutable
#     struct. `acc + zero(BigFloat)` is the safe substitute: `+` is
#     non-mutating for BigFloat (confirmed throughout this file) and
#     always constructs a genuinely fresh MPFR object. ----

function ktrmm!(X::AbstractMatrix{BigFloat}, M::AbstractMatrix{BigFloat})
    k = size(M, 1)
    k == 0 && return X
    rows = size(X, 1)
    acc = BigFloat()
    buf = BigFloat()
    @inbounds for j in 1:k
        Mcol = view(M, j:k, j)
        for i in 1:rows
            Xrow = view(X, i, j:k)
            kdot!(acc, buf, Xrow, Mcol)
            X[i, j] = acc + zero(BigFloat)   # NOT copy(acc) — see note above
        end
    end
    return X
end

# The determinant-only 2x2 line-search test is excellent for immutable
# fixed-width arithmetic, but tiny last-bit differences can change the
# accepted step in long BigFloat trajectories. Retain the original
# Cholesky criterion for BigFloat so dense and sparse paths make the same
# acceptance decision at the requested MPFR precision.
function trial_isposdef!(
    scratch::AbstractMatrix{BigFloat},
    X::AbstractMatrix{BigFloat},
    t::BigFloat,
    dX::AbstractMatrix{BigFloat},
)
    trial_combine!(scratch, X, t, dX)
    return kchol!(scratch)
end

# ---- kmul! : C = α·A·B + β·C, gemm/gemv-shaped, via row·column kdot!.
#     Added for the same reason as kchol!/ktrsm!/ktrmm! above: it's the
#     other pervasive Base-generic call (buildP!, accumulate_v!, the
#     predictor/corrector's Z/dX/dY reconstruction) and, measured after
#     kchol!/ktrsm!/ktrmm! landed, was the next-largest remaining
#     allocation source. `A`/`B` may be `Transpose`-wrapped (both call
#     sites in schur.jl use `transpose(Av[l])`) — indexing/`view`
#     through a `Transpose` works transparently in Julia, so no separate
#     code path is needed for that case. β=0 is special-cased to avoid
#     ever reading C's old value (standard BLAS gemm convention) — this
#     also sidesteps any risk from a stale/uninitialized C containing a
#     non-finite value, where `0 * NaN = NaN` would otherwise corrupt
#     an intended overwrite. ----

function kmul!(C::AbstractVector{BigFloat}, A::AbstractMatrix{BigFloat}, B::AbstractVector{BigFloat},
    α::BigFloat, β::BigFloat)
    p = size(A, 1)
    acc = BigFloat()
    buf = BigFloat()
    @inbounds for i in 1:p
        Ai = view(A, i, :)
        kdot!(acc, buf, Ai, B)
        C[i] = β == 0 ? α * acc : α * acc + β * C[i]
    end
    return C
end

function kmul!(C::AbstractMatrix{BigFloat}, A::AbstractMatrix{BigFloat}, B::AbstractMatrix{BigFloat},
    α::BigFloat, β::BigFloat)
    p, r = size(C, 1), size(C, 2)
    acc = BigFloat()
    buf = BigFloat()
    @inbounds for j in 1:r
        Bj = view(B, :, j)
        for i in 1:p
            Ai = view(A, i, :)
            kdot!(acc, buf, Ai, Bj)
            C[i, j] = β == 0 ? α * acc : α * acc + β * C[i, j]
        end
    end
    return C
end

kmul!(C::AbstractVecOrMat{BigFloat}, A::AbstractMatrix{BigFloat}, B::AbstractVecOrMat{BigFloat}) =
    kmul!(C, A, B, one(BigFloat), zero(BigFloat))
