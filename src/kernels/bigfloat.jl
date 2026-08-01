#=====================================================================
    BigFloat fast path (§1.5). Plain BigFloat arithmetic allocates a
    fresh MPFR object on every `+`/`*` (P1); MutableArithmetics wraps
    the mutating `mpfr_add`/`mpfr_mul`/`mpfr_fma` C calls as
    `operate!`/`operate_to!`/`buffered_operate!`. This makes `kdot!`
    allocation-free after its two scratch values are supplied. Alias-safe
    array outputs still need one independent MPFR object per entry, but avoid
    all intermediate product and sum objects.

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

@inline _coo_owned_scalar(value::BigFloat) = MA.mutable_copy(value)

# MutableArithmetics intentionally does not expose in-place division for
# BigFloat yet. The second triangular pass in `kcholsolve!` owns independent
# destination objects created by the first pass, so MPFR may safely reuse that
# storage. Keep this tiny wrapper local to the BigFloat kernel layer; its
# signature mirrors Julia's own BigFloat `/` implementation.
@inline function _mpfr_divide!(
    output::BigFloat,
    numerator::BigFloat,
    denominator::BigFloat,
)
    ccall(
        (:mpfr_div, Base.MPFR.libmpfr),
        Cint,
        (
            Ref{BigFloat},
            Ref{BigFloat},
            Ref{BigFloat},
            Base.MPFR.MPFRRoundingMode,
        ),
        output,
        numerator,
        denominator,
        Base.MPFR.rounding_raw(BigFloat),
    )
    return output
end

@inline function _mpfr_sqrt!(
    output::BigFloat,
    input::BigFloat,
)
    ccall(
        (:mpfr_sqrt, Base.MPFR.libmpfr),
        Cint,
        (
            Ref{BigFloat},
            Ref{BigFloat},
            Base.MPFR.MPFRRoundingMode,
        ),
        output,
        input,
        Base.MPFR.rounding_raw(BigFloat),
    )
    return output
end

# Convert a binary64 value into an already-owned MPFR destination. Calling
# `BigFloat(value)` in a dense mixed-precision solve creates a temporary MPFR
# object for every vector entry and right-hand side; `mpfr_set_d` performs the
# same correctly rounded conversion directly into reusable solver storage.
@inline function _mpfr_set_float64!(
    output::BigFloat,
    input::Float64,
)
    ccall(
        (:mpfr_set_d, Base.MPFR.libmpfr),
        Cint,
        (
            Ref{BigFloat},
            Cdouble,
            Base.MPFR.MPFRRoundingMode,
        ),
        output,
        input,
        Base.MPFR.rounding_raw(BigFloat),
    )
    return output
end

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
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    beta_is_zero = iszero(β)
    @inbounds for i in eachindex(X, Y)
        MA.operate_to!(accumulator, *, α, X[i])
        if !beta_is_zero
            MA.operate_to!(multiplication_buffer, *, β, Y[i])
            MA.operate!(+, accumulator, multiplication_buffer)
        end
        # The destination may have been created with `zeros(BigFloat, ...)`
        # or may alias X. Store one independent MPFR object per entry instead
        # of mutating Y[i] directly.
        Y[i] = MA.mutable_copy(accumulator)
    end
    return Y
end

function kaxpby_owned!(
    α::BigFloat,
    X::AbstractArray{BigFloat},
    β::BigFloat,
    Y::AbstractArray{BigFloat},
)
    multiplication_buffer = BigFloat()
    alpha_is_one = isone(α)
    beta_is_zero = iszero(β)
    @inbounds for index in eachindex(X, Y)
        if !beta_is_zero
            MA.operate_to!(multiplication_buffer, *, β, Y[index])
        end
        if alpha_is_one
            MA.operate_to!(Y[index], copy, X[index])
        else
            MA.operate_to!(Y[index], *, α, X[index])
        end
        if !beta_is_zero
            MA.operate!(+, Y[index], multiplication_buffer)
        end
    end
    return Y
end

function copy_owned!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray{BigFloat},
)
    axes(destination) == axes(source) ||
        throw(DimensionMismatch("copy_owned! arrays must have matching axes"))
    @inbounds for index in eachindex(destination, source)
        MA.operate_to!(destination[index], copy, source[index])
    end
    return destination
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
    zero_value = BigFloat(0)
    @inbounds for i in eachindex(A)
        A[i] = MA.mutable_copy(zero_value)
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
    zero_value = BigFloat(0)
    @inbounds for i in eachindex(A)
        A[i] = MA.mutable_copy(zero_value)
    end
    return A
end

"""
    zero_owned!(A) -> A

Reset caller-owned workspace storage without replacing its elements.

For `BigFloat`, every entry must already be an initialized, independently
owned MPFR object at the current working precision, normally from
[`alloc_zeros`](@ref) or a previous alias-safe kernel store. Unlike
[`zero_distinct!`](@ref), this function does not repair aliased or
mixed-precision storage; in exchange, it performs zero heap allocations. Use
`zero_distinct!` for arbitrary user arrays and `zero_owned!` only for solver
workspaces whose ownership invariant is known.
"""
zero_owned!(A::AbstractArray) = fill!(A, zero(eltype(A)))
function zero_owned!(A::AbstractArray{BigFloat})
    @inbounds for value in A
        MA.operate!(zero, value)
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
    difference = BigFloat()
    @inbounds for j in 1:k
        if j > 1
            Lj = view(A, j, 1:(j-1))
            kdot!(acc, buf, Lj, Lj)
            MA.operate_to!(difference, -, A[j, j], acc)
            djj = difference
        else
            djj = A[j, j]
        end
        djj <= 0 && return false
        # `A` is solver-owned storage, so reuse its independent MPFR
        # destination. This preserves the exact `mpfr_sqrt`/`mpfr_div`
        # rounding of the former expressions without constructing one new
        # BigFloat object for every factor entry.
        _mpfr_sqrt!(A[j, j], djj)
        Ljj = A[j, j]
        for i in (j+1):k
            if j > 1
                Li = view(A, i, 1:(j-1))
                Lj = view(A, j, 1:(j-1))
                kdot!(acc, buf, Li, Lj)
                MA.operate_to!(difference, -, A[i, j], acc)
                num = difference
            else
                num = A[i, j]
            end
            _mpfr_divide!(A[i, j], num, Ljj)
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
    difference = BigFloat()
    @inbounds for c in 1:p
        Xc = view(X, :, c)
        for i in 1:k
            if i > 1
                Li = view(L, i, 1:(i-1))
                yi = view(Xc, 1:(i-1))
                kdot!(acc, buf, Li, yi)
                MA.operate_to!(difference, -, Xc[i], acc)
                val = difference
            else
                val = Xc[i]
            end
            Xc[i] = val / L[i, i]
        end
    end
    return X
end

# ---- ktrsv_lower!/ktrsv_transpose! : alias-safe vector triangular
#     solves. Each output is stored as a fresh MPFR object, so these helpers
#     are safe after `copyto!` and for arrays originally created by
#     `zeros(BigFloat, ...)`. ----

function ktrsv_lower!(
    L::AbstractMatrix{BigFloat},
    rhs::AbstractVector{BigFloat},
)
    dimension = size(L, 1)
    dimension == 0 && return rhs
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    difference = BigFloat()
    @inbounds for row in 1:dimension
        numerator = rhs[row]
        if row > 1
            kdot!(
                accumulator,
                multiplication_buffer,
                view(L, row, 1:(row - 1)),
                view(rhs, 1:(row - 1)),
            )
            MA.operate_to!(difference, -, rhs[row], accumulator)
            numerator = difference
        end
        rhs[row] = numerator / L[row, row]
    end
    return rhs
end

function ktrsv_transpose!(
    L::AbstractMatrix{BigFloat},
    rhs::AbstractVector{BigFloat},
)
    dimension = size(L, 1)
    dimension == 0 && return rhs
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    difference = BigFloat()
    @inbounds for row in dimension:-1:1
        numerator = rhs[row]
        if row < dimension
            kdot!(
                accumulator,
                multiplication_buffer,
                view(L, (row + 1):dimension, row),
                view(rhs, (row + 1):dimension),
            )
            MA.operate_to!(difference, -, rhs[row], accumulator)
            numerator = difference
        end
        rhs[row] = numerator / L[row, row]
    end
    return rhs
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
#     struct. `MutableArithmetics.mutable_copy(acc)` is the safe substitute:
#     it constructs a genuinely independent MPFR object without first
#     allocating a temporary zero. ----

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
            X[i, j] = MA.mutable_copy(acc)
        end
    end
    return X
end

# ---- kcholsolve! : M ← (L*L')⁻¹M. Base's generic triangular solves
#     repeatedly allocate intermediate BigFloat products and differences.
#     Reusing the dot and subtraction buffers leaves one independent result
#     object per entry in the forward pass; the backward pass then reuses that
#     now-independent storage through `_mpfr_divide!`. ----

function kcholsolve!(
    L::AbstractMatrix{BigFloat},
    rhs::AbstractVector{BigFloat},
)
    dimension = size(L, 1)
    dimension == 0 && return rhs
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    difference = BigFloat()

    @inbounds for row in 1:dimension
        numerator = rhs[row]
        if row > 1
            kdot!(
                accumulator,
                multiplication_buffer,
                view(L, row, 1:(row - 1)),
                view(rhs, 1:(row - 1)),
            )
            MA.operate_to!(difference, -, rhs[row], accumulator)
            numerator = difference
        end
        rhs[row] = numerator / L[row, row]
    end

    @inbounds for row in dimension:-1:1
        numerator = rhs[row]
        if row < dimension
            kdot!(
                accumulator,
                multiplication_buffer,
                view(L, (row + 1):dimension, row),
                view(rhs, (row + 1):dimension),
            )
            MA.operate_to!(difference, -, rhs[row], accumulator)
            numerator = difference
        end
        _mpfr_divide!(rhs[row], numerator, L[row, row])
    end
    return rhs
end

function kcholsolve_owned!(
    L::AbstractMatrix{BigFloat},
    rhs::AbstractVector{BigFloat},
    accumulator::BigFloat,
    multiplication_buffer::BigFloat,
    difference::BigFloat,
)
    dimension = size(L, 1)
    dimension == 0 && return rhs
    if dimension == 1
        _mpfr_divide!(rhs[1], rhs[1], L[1, 1])
        _mpfr_divide!(rhs[1], rhs[1], L[1, 1])
        return rhs
    end
    @inbounds for row in 1:dimension
        numerator = rhs[row]
        if row > 1
            kdot!(
                accumulator,
                multiplication_buffer,
                view(L, row, 1:(row - 1)),
                view(rhs, 1:(row - 1)),
            )
            MA.operate_to!(difference, -, rhs[row], accumulator)
            numerator = difference
        end
        _mpfr_divide!(rhs[row], numerator, L[row, row])
    end
    @inbounds for row in dimension:-1:1
        numerator = rhs[row]
        if row < dimension
            kdot!(
                accumulator,
                multiplication_buffer,
                view(L, (row + 1):dimension, row),
                view(rhs, (row + 1):dimension),
            )
            MA.operate_to!(difference, -, rhs[row], accumulator)
            numerator = difference
        end
        _mpfr_divide!(rhs[row], numerator, L[row, row])
    end
    return rhs
end

function kcholsolve_owned!(
    L::AbstractMatrix{BigFloat},
    rhs::AbstractVector{BigFloat},
)
    return kcholsolve_owned!(
        L,
        rhs,
        BigFloat(),
        BigFloat(),
        BigFloat(),
    )
end

function kcholsolve!(
    L::AbstractMatrix{BigFloat},
    rhs::AbstractMatrix{BigFloat},
)
    dimension = size(L, 1)
    dimension == 0 && return rhs
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    difference = BigFloat()

    @inbounds for column in axes(rhs, 2)
        rhs_column = view(rhs, :, column)
        for row in 1:dimension
            numerator = rhs_column[row]
            if row > 1
                kdot!(
                    accumulator,
                    multiplication_buffer,
                    view(L, row, 1:(row - 1)),
                    view(rhs_column, 1:(row - 1)),
                )
                MA.operate_to!(difference, -, rhs_column[row], accumulator)
                numerator = difference
            end
            rhs_column[row] = numerator / L[row, row]
        end

        for row in dimension:-1:1
            numerator = rhs_column[row]
            if row < dimension
                kdot!(
                    accumulator,
                    multiplication_buffer,
                    view(L, (row + 1):dimension, row),
                    view(rhs_column, (row + 1):dimension),
                )
                MA.operate_to!(difference, -, rhs_column[row], accumulator)
                numerator = difference
            end
            _mpfr_divide!(rhs_column[row], numerator, L[row, row])
        end
    end
    return rhs
end

function kcholsolve_owned!(
    L::AbstractMatrix{BigFloat},
    rhs::AbstractMatrix{BigFloat},
    accumulator::BigFloat,
    multiplication_buffer::BigFloat,
    difference::BigFloat,
)
    dimension = size(L, 1)
    dimension == 0 && return rhs
    if dimension == 1
        diagonal = L[1, 1]
        @inbounds for column in axes(rhs, 2)
            _mpfr_divide!(
                rhs[1, column],
                rhs[1, column],
                diagonal,
            )
            _mpfr_divide!(
                rhs[1, column],
                rhs[1, column],
                diagonal,
            )
        end
        return rhs
    end
    @inbounds for column in axes(rhs, 2)
        rhs_column = view(rhs, :, column)
        for row in 1:dimension
            numerator = rhs_column[row]
            if row > 1
                kdot!(
                    accumulator,
                    multiplication_buffer,
                    view(L, row, 1:(row - 1)),
                    view(rhs_column, 1:(row - 1)),
                )
                MA.operate_to!(
                    difference,
                    -,
                    rhs_column[row],
                    accumulator,
                )
                numerator = difference
            end
            _mpfr_divide!(
                rhs_column[row],
                numerator,
                L[row, row],
            )
        end
        for row in dimension:-1:1
            numerator = rhs_column[row]
            if row < dimension
                kdot!(
                    accumulator,
                    multiplication_buffer,
                    view(L, (row + 1):dimension, row),
                    view(rhs_column, (row + 1):dimension),
                )
                MA.operate_to!(
                    difference,
                    -,
                    rhs_column[row],
                    accumulator,
                )
                numerator = difference
            end
            _mpfr_divide!(
                rhs_column[row],
                numerator,
                L[row, row],
            )
        end
    end
    return rhs
end

function kcholsolve_owned!(
    L::AbstractMatrix{BigFloat},
    rhs::AbstractMatrix{BigFloat},
)
    return kcholsolve_owned!(
        L,
        rhs,
        BigFloat(),
        BigFloat(),
        BigFloat(),
    )
end

"""
    BigFloatCholeskyFactor(L)

Internal marker for a full-rank BigFloat Cholesky factor produced by
[`kchol!`](@ref). The KKT layer stores this lightweight wrapper in its
factorization cache and solves through [`kcholsolve!`](@ref), avoiding Base's
allocation-heavy generic `Cholesky` solve. `L` is borrowed, not copied.
"""
struct BigFloatCholeskyFactor{M<:AbstractMatrix{BigFloat}}
    L::M
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
    trial_combine_owned!(scratch, X, t, dX)
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
    alpha_is_one = isone(α)
    beta_is_zero = iszero(β)
    @inbounds for i in 1:p
        Ai = view(A, i, :)
        kdot!(acc, buf, Ai, B)
        alpha_is_one || MA.operate!(*, acc, α)
        if !beta_is_zero
            MA.operate_to!(buf, *, β, C[i])
            MA.operate!(+, acc, buf)
        end
        C[i] = MA.mutable_copy(acc)
    end
    return C
end

function kmul!(C::AbstractMatrix{BigFloat}, A::AbstractMatrix{BigFloat}, B::AbstractMatrix{BigFloat},
    α::BigFloat, β::BigFloat)
    p, r = size(C, 1), size(C, 2)
    acc = BigFloat()
    buf = BigFloat()
    alpha_is_one = isone(α)
    beta_is_zero = iszero(β)
    @inbounds for j in 1:r
        Bj = view(B, :, j)
        for i in 1:p
            Ai = view(A, i, :)
            kdot!(acc, buf, Ai, Bj)
            alpha_is_one || MA.operate!(*, acc, α)
            if !beta_is_zero
                MA.operate_to!(buf, *, β, C[i, j])
                MA.operate!(+, acc, buf)
            end
            C[i, j] = MA.mutable_copy(acc)
        end
    end
    return C
end

kmul!(C::AbstractVecOrMat{BigFloat}, A::AbstractMatrix{BigFloat}, B::AbstractVecOrMat{BigFloat}) =
    kmul!(C, A, B, one(BigFloat), zero(BigFloat))

@inline function _store_owned_bigfloat!(
    destination::BigFloat,
    accumulator::BigFloat,
    multiplication_buffer::BigFloat,
    α::BigFloat,
    β::BigFloat,
    alpha_is_one::Bool,
    beta_is_zero::Bool,
)
    if !beta_is_zero
        MA.operate_to!(multiplication_buffer, *, β, destination)
    end
    if alpha_is_one
        MA.operate_to!(destination, copy, accumulator)
    else
        MA.operate_to!(destination, *, α, accumulator)
    end
    if !beta_is_zero
        MA.operate!(+, destination, multiplication_buffer)
    end
    return nothing
end

function kmul_owned!(
    C::AbstractVector{BigFloat},
    A::AbstractMatrix{BigFloat},
    B::AbstractVector{BigFloat},
    α::BigFloat,
    β::BigFloat,
)
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    alpha_is_one = isone(α)
    beta_is_zero = iszero(β)
    @inbounds for row in axes(C, 1)
        kdot!(
            accumulator,
            multiplication_buffer,
            view(A, row, :),
            B,
        )
        _store_owned_bigfloat!(
            C[row],
            accumulator,
            multiplication_buffer,
            α,
            β,
            alpha_is_one,
            beta_is_zero,
        )
    end
    return C
end

function kmul_owned!(
    C::AbstractMatrix{BigFloat},
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    α::BigFloat,
    β::BigFloat,
)
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    alpha_is_one = isone(α)
    beta_is_zero = iszero(β)
    @inbounds for column in axes(C, 2)
        B_column = view(B, :, column)
        for row in axes(C, 1)
            kdot!(
                accumulator,
                multiplication_buffer,
                view(A, row, :),
                B_column,
            )
            _store_owned_bigfloat!(
                C[row, column],
                accumulator,
                multiplication_buffer,
                α,
                β,
                alpha_is_one,
                beta_is_zero,
            )
        end
    end
    return C
end

kmul_owned!(
    C::AbstractVecOrMat{BigFloat},
    A::AbstractMatrix{BigFloat},
    B::AbstractVecOrMat{BigFloat},
) = kmul_owned!(C, A, B, one(BigFloat), zero(BigFloat))

"""
    _sparse_bigfloat_gemv_owned!(destination, matrix, vector)

Compute `destination = matrix * vector` directly from CSC nonzeros. Output
rows receive contributions in ascending matrix-column order, matching the
dense owned kernel's reduction order while avoiding structural-zero MPFR
products. The routine is serial because different CSC columns may update the
same row.
"""
function _sparse_bigfloat_gemv_owned!(
    destination::AbstractVector{BigFloat},
    matrix::SparseMatrixCSC{BigFloat,Ti},
    vector::AbstractVector{BigFloat},
) where {Ti<:Integer}
    size(matrix, 1) == length(destination) ||
        throw(DimensionMismatch("sparse BigFloat GEMV output mismatch"))
    size(matrix, 2) == length(vector) ||
        throw(DimensionMismatch("sparse BigFloat GEMV input mismatch"))
    zero_owned!(destination)
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    multiplication_buffer = BigFloat()
    @inbounds for column in axes(matrix, 2)
        scalar = vector[column]
        for position in nzrange(matrix, column)
            row = rows[position]
            MA.buffered_operate!(
                multiplication_buffer,
                MA.add_mul,
                destination[row],
                destination[row],
                values[position],
                scalar,
            )
        end
    end
    return destination
end

function _sparse_bigfloat_transpose_workers(
    matrix::SparseMatrixCSC{BigFloat,Ti},
    requested_workers::Int,
) where {Ti<:Integer}
    useful_workers = max(1, nnz(matrix) ÷ 18_000)
    return min(
        max(requested_workers, 1),
        Threads.nthreads(),
        max(size(matrix, 2), 1),
        useful_workers,
    )
end

"""
    _sparse_bigfloat_transpose_gemv_owned!(
        destination, matrix, vector, requested_workers)

Compute `destination = matrix' * vector` with complete CSC columns assigned
to workers. A worker owns every destination scalar it writes and uses private
MPFR accumulation buffers; row indices within each column retain their serial
order. Small matrices stay serial through a nonzero-work crossover.
"""
function _sparse_bigfloat_transpose_gemv_owned!(
    destination::AbstractVector{BigFloat},
    matrix::SparseMatrixCSC{BigFloat,Ti},
    vector::AbstractVector{BigFloat},
    requested_workers::Int,
) where {Ti<:Integer}
    size(matrix, 2) == length(destination) ||
        throw(DimensionMismatch(
            "sparse BigFloat transpose GEMV output mismatch",
        ))
    size(matrix, 1) == length(vector) ||
        throw(DimensionMismatch(
            "sparse BigFloat transpose GEMV input mismatch",
        ))
    workers = _sparse_bigfloat_transpose_workers(
        matrix,
        requested_workers,
    )
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    chunk = cld(length(destination), workers)
    @sync for worker in 1:workers
        first_column = (worker - 1) * chunk + 1
        first_column > length(destination) && continue
        last_column = min(worker * chunk, length(destination))
        Threads.@spawn begin
            accumulator = BigFloat()
            multiplication_buffer = BigFloat()
            @inbounds for column in first_column:last_column
                MA.operate!(zero, accumulator)
                for position in nzrange(matrix, column)
                    MA.buffered_operate!(
                        multiplication_buffer,
                        MA.add_mul,
                        accumulator,
                        accumulator,
                        values[position],
                        vector[rows[position]],
                    )
                end
                MA.operate_to!(
                    destination[column],
                    copy,
                    accumulator,
                )
            end
        end
    end
    return destination
end

# ---- ksyrk! : S = α*P'*P + β*S. The generic implementation allocates
#     scratch for every pairwise dot and stores the same mutable BigFloat in
#     both symmetric positions. This specialization reuses one reduction
#     buffer and makes the two output entries independent. ----

function ksyrk!(
    S::AbstractMatrix{BigFloat},
    panel::AbstractMatrix{BigFloat},
    α::BigFloat,
    β::BigFloat,
)
    columns = size(panel, 2)
    size(S) == (columns, columns) ||
        throw(DimensionMismatch("ksyrk!: S must be c×c for P r×c"))
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    alpha_is_one = isone(α)
    beta_is_zero = iszero(β)

    @inbounds for column in 1:columns
        for row in column:columns
            kdot_columns!(
                accumulator,
                multiplication_buffer,
                panel,
                row,
                column,
                size(panel, 1),
            )
            alpha_is_one || MA.operate!(*, accumulator, α)
            if !beta_is_zero
                MA.operate_to!(
                    multiplication_buffer,
                    *,
                    β,
                    S[row, column],
                )
                MA.operate!(+, accumulator, multiplication_buffer)
            end
            value = MA.mutable_copy(accumulator)
            S[row, column] = value
            if row != column
                S[column, row] = MA.mutable_copy(value)
            end
        end
    end
    return S
end

ksyrk!(S::AbstractMatrix{BigFloat}, panel::AbstractMatrix{BigFloat}) =
    ksyrk!(S, panel, one(BigFloat), zero(BigFloat))

# ---- trial construction and infinity norms. These are called repeatedly by
#     line search and residual checks, so O(length(A)) scalar allocations are
#     visible even though each individual operation is small. ----

function trial_combine!(
    destination::AbstractArray{BigFloat},
    X::AbstractArray{BigFloat},
    step::BigFloat,
    direction::AbstractArray{BigFloat},
)
    accumulator = BigFloat()
    @inbounds for index in eachindex(destination, X, direction)
        MA.operate_to!(accumulator, *, step, direction[index])
        MA.operate!(+, accumulator, X[index])
        destination[index] = MA.mutable_copy(accumulator)
    end
    return destination
end

function trial_combine_owned!(
    destination::AbstractArray{BigFloat},
    X::AbstractArray{BigFloat},
    step::BigFloat,
    direction::AbstractArray{BigFloat},
    accumulator::BigFloat,
)
    @inbounds for index in eachindex(destination, X, direction)
        MA.operate_to!(accumulator, *, step, direction[index])
        # MPFR addition permits the output to alias `X[index]`. This matters
        # for the accepted-step update, where `destination === X`.
        MA.operate_to!(
            destination[index],
            +,
            X[index],
            accumulator,
        )
    end
    return destination
end

function trial_combine_owned!(
    destination::AbstractArray{BigFloat},
    X::AbstractArray{BigFloat},
    step::BigFloat,
    direction::AbstractArray{BigFloat},
)
    return trial_combine_owned!(
        destination,
        X,
        step,
        direction,
        BigFloat(),
    )
end

@inline function _update_bigfloat_abs_maximum!(
    maximum_value::BigFloat,
    negative_maximum::BigFloat,
    value::BigFloat,
)
    if signbit(value)
        if value < negative_maximum
            MA.operate_to!(negative_maximum, copy, value)
            MA.operate_to!(maximum_value, -, value)
        end
    elseif value > maximum_value
        MA.operate_to!(maximum_value, copy, value)
        MA.operate_to!(negative_maximum, -, value)
    end
    return nothing
end

function knrmInf(array::AbstractArray{BigFloat})
    maximum_value = BigFloat()
    negative_maximum = BigFloat()
    MA.operate!(zero, maximum_value)
    MA.operate!(zero, negative_maximum)
    @inbounds for value in array
        if isnan(value)
            return MA.mutable_copy(value)
        end
        _update_bigfloat_abs_maximum!(
            maximum_value,
            negative_maximum,
            value,
        )
    end
    return maximum_value
end

function knrmInf(blocks::AbstractVector{<:AbstractArray{BigFloat}})
    maximum_value = BigFloat()
    negative_maximum = BigFloat()
    MA.operate!(zero, maximum_value)
    MA.operate!(zero, negative_maximum)
    @inbounds for block in blocks
        for value in block
            if isnan(value)
                return MA.mutable_copy(value)
            end
            _update_bigfloat_abs_maximum!(
                maximum_value,
                negative_maximum,
                value,
            )
        end
    end
    return maximum_value
end
