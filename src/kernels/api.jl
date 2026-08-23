#=====================================================================
    Kernel API (§1.4). Solver linear algebra routes through this seam.
    `kernels/generic.jl` provides allocation-light Base LinearAlgebra
    fallbacks for every arithmetic type. `kernels/bigfloat.jl` adds
    buffered MPFR reductions, factorizations, triangular solves, norms,
    and explicitly named owned-workspace variants. Extended-precision
    blocked and threaded kernels can therefore change implementations
    without duplicating solver logic or weakening the public alias-safe
    operations.
=====================================================================#

"""
    kdot(A, B) -> scalar

Frobenius / vector inner product `⟨A,B⟩ = Σ A[i]*B[i]`, allocation-free
for BigFloat via MutableArithmetics (kernels/bigfloat.jl). Replaces
`tr(A*B)`-style full matrix products for symmetric `A,B` (P6) and the
sliced `dot(SS[l][i], A[l][j,:,:])` pattern (P5).
"""
function kdot end

"""
    kmul!(C, A, B, α=one(eltype(C)), β=zero(eltype(C))) -> C

`C = α·A·B + β·C`, gemm-shaped, in place.
"""
function kmul! end

"""
    kmul_owned!(C, A, B, α, β) -> C

Internal workspace variant of [`kmul!`](@ref). The destination entries must
already be initialized, independently owned at the current working precision,
and must not alias `A` or `B`. This stronger ownership contract lets mutable
arithmetic write directly into preallocated high-precision scalar storage.
"""
function kmul_owned! end

"""
    ksyrk!(S, P, α=one(eltype(S)), β=zero(eltype(S))) -> S

`S = α·Pᵀ·P + β·S` for `P` an `r×c` panel, `S` `c×c`; the default `β`
overwrites `S`. Used for the symmetric-square Schur contribution
`S += P̂P̂ᵀ` (§2.3) and `Q = B̃ᵀB̃` (§2.2); implemented via pairwise
`kdot` calls so it inherits the BigFloat fast path automatically.
"""
function ksyrk! end

"""
    ktrsm!(L, X) -> X

`X ← L⁻¹X` for `L` square lower-triangular, in place.
"""
function ktrsm! end

"""
    ktrsv_lower!(L, x) -> x

Solve `L*x = b` in place, where `L` is a square lower-triangular factor and
`x` initially contains `b`.
"""
function ktrsv_lower! end

"""
    ktrsv_transpose!(L, x) -> x

Solve `transpose(L)*x = b` in place using the lower-triangular factor `L`.
"""
function ktrsv_transpose! end

"""
    ktrmm!(X, M) -> X

`X ← X·M` for `M` square lower-triangular, applied on the right, in
place.
"""
function ktrmm! end

"""
    kchol!(A) -> success::Bool

In-place lower Cholesky of symmetric `A` (lower triangle read/written;
upper untouched). Returns `false` instead of throwing on failure, so
callers (regularization retry, line-search trials) can branch without
`try/catch` in the hot path.
"""
function kchol! end

"""
    kaxpby!(α, X, β, Y) -> Y

`Y = α·X + β·Y` elementwise, for matrices, vectors, or matching-shape
containers thereof.
"""
function kaxpby! end

"""
    kaxpby_owned!(α, X, β, Y) -> Y

Internal workspace variant of [`kaxpby!`](@ref). Every destination entry must
already own independent scalar storage at the current working precision;
arbitrary aliased user arrays must use `kaxpby!`.
"""
function kaxpby_owned! end

"""
    copy_owned!(destination, source) -> destination

Copy values into preallocated workspace storage. For mutable scalar types,
destination entries must be independently owned at the current working
precision and must not alias source entries.
"""
function copy_owned! end

"""
    zero_owned!(A) -> A

Reset preallocated workspace storage in place. For mutable scalar types,
entries must already be initialized and independently owned at the current
working precision. Use the alias-safe storage initializer for arbitrary user
arrays.
"""
function zero_owned! end

"""
    knrmInf(A) -> scalar

`maximum(abs, A)` without splatting (P7: `max(abs.(A)...)` allocates
and can overflow the call stack for large blocks).
"""
function knrmInf end
