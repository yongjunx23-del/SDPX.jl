#=====================================================================
    Kernel API (§1.4). Every linear-algebra operation in the solver
    goes through these eight functions. `kernels/generic.jl` gives a
    correct, allocation-light implementation for any T using Base
    LinearAlgebra; `kernels/bigfloat.jl` overrides the two scalar-
    reduction primitives (kdot, kaxpby!) with MutableArithmetics-based
    zero-allocation MPFR kernels, since every other kernel (ksyrk!,
    ktrsm!, ktrmm!, kchol!) is itself expressed in terms of kdot/axpby
    in the generic path and so inherits the speedup automatically.
    Later phases (threaded tiling, CRT/BLAS syrk) swap implementations
    behind this same seam without touching solver logic.
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
    ksyrk!(S, P, α=one(eltype(S)), β=one(eltype(S))) -> S

`S = α·Pᵀ·P + β·S` for `P` an `r×c` panel, `S` `c×c`. Used for the
symmetric-square Schur contribution `S += P̂P̂ᵀ` (§2.3) and `Q = B̃ᵀB̃`
(§2.2); implemented via pairwise `kdot` calls so it inherits the
BigFloat fast path automatically.
"""
function ksyrk! end

"""
    ktrsm!(L, X) -> X

`X ← L⁻¹X` for `L` square lower-triangular, in place.
"""
function ktrsm! end

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
    knrmInf(A) -> scalar

`maximum(abs, A)` without splatting (P7: `max(abs.(A)...)` allocates
and can overflow the call stack for large blocks).
"""
function knrmInf end
