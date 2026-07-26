#=====================================================================
    Generic kernel implementations (§1.4): correct for any T, built on
    Base LinearAlgebra so BLAS-backed types (Float64, ComplexF64, …)
    get real BLAS/LAPACK speed automatically, while non-BLAS types
    (BigFloat, MultiFloat, Double64, …) fall onto Base's generic dense
    linear algebra (verified to exist for cholesky!/lu!/qr!/ldiv! on
    BigFloat — see development notes). No array-level temporaries are
    created anywhere in this file: every kernel writes into a caller-
    supplied buffer. `kernels/bigfloat.jl` overrides `kdot`/`kaxpby!`
    with MutableArithmetics-backed zero-allocation MPFR kernels; since
    `ksyrk!`/`kchol!` route their scalar work through `kdot`, and the
    solver's per-iteration hot loops are themselves built from these
    eight functions, the speedup composes without touching solve.jl.
=====================================================================#

using LinearAlgebra: LowerTriangular, Symmetric, issuccess

# ---- kdot ----

kdot(A, B) = LinearAlgebra.dot(A, B)

function kdot_columns!(accumulator, buffer, panel, first_column, second_column, rows)
    accumulator = zero(eltype(panel))
    @inbounds for row in 1:rows
        accumulator += panel[row, first_column] * panel[row, second_column]
    end
    return accumulator
end

# ---- kmul! ----

@inline function _kmul2!(C, A, B, α, β)
    c11 = A[1, 1] * B[1, 1] + A[1, 2] * B[2, 1]
    c21 = A[2, 1] * B[1, 1] + A[2, 2] * B[2, 1]
    c12 = A[1, 1] * B[1, 2] + A[1, 2] * B[2, 2]
    c22 = A[2, 1] * B[1, 2] + A[2, 2] * B[2, 2]
    if iszero(β)
        C[1, 1] = α * c11
        C[2, 1] = α * c21
        C[1, 2] = α * c12
        C[2, 2] = α * c22
    else
        C[1, 1] = α * c11 + β * C[1, 1]
        C[2, 1] = α * c21 + β * C[2, 1]
        C[1, 2] = α * c12 + β * C[1, 2]
        C[2, 2] = α * c22 + β * C[2, 2]
    end
    return C
end

function kmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, β)
    if size(C) == (2, 2) && size(A) == (2, 2) && size(B) == (2, 2)
        return _kmul2!(C, A, B, α, β)
    end
    return LinearAlgebra.mul!(C, A, B, α, β)
end

function kmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    T = eltype(C)
    return kmul!(C, A, B, one(T), zero(T))
end

kmul!(C, A, B, α, β) = LinearAlgebra.mul!(C, A, B, α, β)
kmul!(C, A, B) = LinearAlgebra.mul!(C, A, B)

# Owned-storage variants matter only for mutable scalar types. The generic
# fallback preserves the existing implementation and behavior.
kmul_owned!(C, A, B, α, β) = kmul!(C, A, B, α, β)
kmul_owned!(C, A, B) = kmul!(C, A, B)

# ---- ksyrk! : S = α·Pᵀ·P + β·S, P is r×c, S is c×c. Column-wise Gram
#     matrix (used for Q = B̃ᵀB̃ in kkt.jl); the Schur build's block-
#     level analogue lives in schur.jl since its contraction is over
#     k×k blocks rather than single columns. ----

function ksyrk!(S::AbstractMatrix, P::AbstractMatrix, α, β)
    c = size(P, 2)
    @assert size(S) == (c, c) "ksyrk!: S must be c×c for P r×c"
    @inbounds for j in 1:c
        Pj = @view P[:, j]
        for i in j:c
            Pi = @view P[:, i]
            v = α * kdot(Pi, Pj) + β * S[i, j]
            S[i, j] = v
            i != j && (S[j, i] = v)
        end
    end
    return S
end

# `Q = B̃'B̃` is a level-3 BLAS operation for ordinary floating-point
# arithmetic.  The generic pairwise-dot implementation above launches one
# level-1 BLAS call per output entry, which is much slower once the equality
# space is more than a handful of columns (Task_Low08 has 394).  Compute the
# lower triangle with SYRK, then mirror it to preserve the public `ksyrk!`
# contract used by diagnostics and callers.
function ksyrk!(
    S::StridedMatrix{T},
    P::StridedMatrix{T},
    α::T,
    β::T,
) where {T<:Union{Float32,Float64}}
    c = size(P, 2)
    size(S) == (c, c) ||
        throw(DimensionMismatch("ksyrk!: S must be c×c for P r×c"))
    LinearAlgebra.BLAS.syrk!('L', 'T', α, P, β, S)
    @inbounds for column in axes(S, 2), row in (column + 1):c
        S[column, row] = S[row, column]
    end
    return S
end

ksyrk!(S::AbstractMatrix{T}, P::AbstractMatrix{T}) where {T} = ksyrk!(S, P, one(T), zero(T))

# ---- ktrsm! : X ← L⁻¹X, L square lower-triangular ----

ktrsm!(L::AbstractMatrix, X) = LinearAlgebra.ldiv!(LowerTriangular(L), X)

# ---- ktrsv_lower!/ktrsv_transpose! : vector triangular solves ----

ktrsv_lower!(L::AbstractMatrix, x::AbstractVector) =
    LinearAlgebra.ldiv!(LowerTriangular(L), x)

ktrsv_transpose!(L::AbstractMatrix, x::AbstractVector) =
    LinearAlgebra.ldiv!(UpperTriangular(transpose(L)), x)

# ---- ktrmm! : X ← X·M, M square lower-triangular, right multiply ----

ktrmm!(X::AbstractMatrix, M::AbstractMatrix) = LinearAlgebra.rmul!(X, LowerTriangular(M))

# ---- kchol! : in-place lower Cholesky, no throw ----

function kchol!(A::AbstractMatrix)
    k = size(A, 1)
    if k == 1
        d11 = A[1, 1]
        d11 > zero(d11) || return false
        A[1, 1] = sqrt(d11)
        return true
    elseif k == 2
        d11 = A[1, 1]
        d11 > zero(d11) || return false
        l11 = sqrt(d11)
        l21 = A[2, 1] / l11
        d22 = A[2, 2] - l21 * l21
        d22 > zero(d22) || return false
        A[1, 1] = l11
        A[2, 1] = l21
        A[2, 2] = sqrt(d22)
        return true
    end
    C = LinearAlgebra.cholesky!(Symmetric(A, :L); check=false)
    return issuccess(C)
end

# ---- kaxpby! : Y = α·X + β·Y, elementwise; recurses over Vector{<:AbstractArray} blocks ----

function kaxpby!(α, X::AbstractArray{T}, β, Y::AbstractArray{T}) where {T}
    @inbounds for i in eachindex(X, Y)
        Y[i] = α * X[i] + β * Y[i]
    end
    return Y
end
function kaxpby!(α, X::AbstractVector{<:AbstractArray}, β, Y::AbstractVector{<:AbstractArray})
    @inbounds for l in eachindex(X, Y)
        kaxpby!(α, X[l], β, Y[l])
    end
    return Y
end

kaxpby_owned!(α, X, β, Y) = kaxpby!(α, X, β, Y)
copy_owned!(destination, source) = copyto!(destination, source)

# ---- knrmInf : maximum(abs, ·) without splatting (P7) ----

knrmInf(A::AbstractArray) = isempty(A) ? zero(real(eltype(A))) : maximum(abs, A)
function knrmInf(A::AbstractVector{<:AbstractArray})
    isempty(A) && return 0.0
    m = knrmInf(A[1])
    @inbounds for l in 2:length(A)
        m = max(m, knrmInf(A[l]))
    end
    return m
end

# ---- kcholsolve! : M ← X⁻¹M given the lower Cholesky factor LX of X
#     (X = LX·LXᵀ), via two triangular solves. Pervasive in the
#     solver (Z, dY recovery, sparse-path invX formation) so it earns
#     a name even though it's not one of the original eight — it is
#     built from the same Base-generic ldiv! and inherits the same
#     BigFloat-aliasing safety (verified: Base's generic triangular
#     solve never mutates elements in place, only reassigns via
#     ordinary non-mutating arithmetic, so it cannot corrupt anything
#     that happens to alias LX or M from an earlier copyto!). ----

function kcholsolve!(LX::AbstractMatrix, M::AbstractVector)
    k = size(LX, 1)
    if k == 1
        l11 = LX[1, 1]
        M[1] = M[1] / l11 / l11
        return M
    elseif k == 2
        l11 = LX[1, 1]
        l21 = LX[2, 1]
        l22 = LX[2, 2]
        y1 = M[1] / l11
        y2 = (M[2] - l21 * y1) / l22
        x2 = y2 / l22
        M[1] = (y1 - l21 * x2) / l11
        M[2] = x2
        return M
    end
    LinearAlgebra.ldiv!(LowerTriangular(LX), M)
    LinearAlgebra.ldiv!(LowerTriangular(LX)', M)
    return M
end

function kcholsolve!(LX::AbstractMatrix, M::AbstractMatrix)
    k = size(LX, 1)
    if k == 1
        l11 = LX[1, 1]
        @inbounds for column in axes(M, 2)
            M[1, column] = M[1, column] / l11 / l11
        end
        return M
    elseif k == 2
        l11 = LX[1, 1]
        l21 = LX[2, 1]
        l22 = LX[2, 2]
        @inbounds for column in axes(M, 2)
            y1 = M[1, column] / l11
            y2 = (M[2, column] - l21 * y1) / l22
            x2 = y2 / l22
            M[1, column] = (y1 - l21 * x2) / l11
            M[2, column] = x2
        end
        return M
    end
    LinearAlgebra.ldiv!(LowerTriangular(LX), M)
    LinearAlgebra.ldiv!(LowerTriangular(LX)', M)
    return M
end

# ---- trial_combine! : dest ← X + t·dX, via ordinary (non-mutating)
#     arithmetic and direct assignment — deliberately NOT built from
#     copyto!+kaxpby!. For mutable-element types (BigFloat) copyto!
#     aliases dest's entries with X's (verified empirically), so a
#     subsequent in-place kaxpby! on dest would silently corrupt X —
#     exactly the bug this avoids: dest[i] is always assigned a fresh
#     object, never produced by mutating something dest came to alias.
#     Used for line-search trial construction, where X is the *live*
#     iterate and must survive a rejected trial untouched. ----

function trial_combine!(dest::AbstractArray{T}, X::AbstractArray{T}, t::T, dX::AbstractArray{T}) where {T}
    @inbounds for i in eachindex(dest, X, dX)
        dest[i] = X[i] + t * dX[i]
    end
    return dest
end

"""
    trial_isposdef!(scratch, X, t, dX)

Test whether `X + t*dX` is positive definite. The `1x1` and `2x2`
paths use Sylvester's criterion directly, avoiding a scratch-matrix
construction, square roots, divisions, and a generic Cholesky wrapper
inside the backtracking loop. Larger blocks retain the Cholesky test.
"""
function trial_isposdef!(scratch::AbstractMatrix{T}, X::AbstractMatrix{T},
    t::T, dX::AbstractMatrix{T}) where {T}
    k = size(X, 1)
    if k == 1
        return X[1, 1] + t * dX[1, 1] > zero(T)
    elseif k == 2
        a = X[1, 1] + t * dX[1, 1]
        a > zero(T) || return false
        b = X[2, 1] + t * dX[2, 1]
        d = X[2, 2] + t * dX[2, 2]
        return a * d > b * b
    end
    trial_combine!(scratch, X, t, dX)
    return kchol!(scratch)
end

"""
    fraction_to_boundary_bound!(scratch, X, dX)

Return the largest step in `[0, 1]` before `X + t*dX` reaches the PSD
boundary. The `1x1` and `2x2` paths use scalar formulas. Larger blocks use a
bounded bisection fallback and the same positive-definiteness test as the
backtracking rule.
"""
function fraction_to_boundary_bound!(
    scratch::AbstractMatrix{T},
    X::AbstractMatrix{T},
    dX::AbstractMatrix{T},
) where {T}
    k = size(X, 1)
    if k == 1
        d = dX[1, 1]
        return d < zero(T) ? min(one(T), -X[1, 1] / d) : one(T)
    elseif k == 2
        a = X[1, 1]
        b = X[1, 2]
        d = X[2, 2]
        da = dX[1, 1]
        db = (dX[1, 2] + dX[2, 1]) / (one(T) + one(T))
        dd = dX[2, 2]
        a1 = a + da
        b1 = b + db
        d1 = d + dd
        a1 > zero(T) && a1 * d1 > b1 * b1 && return one(T)

        c0 = a * d - b * b
        c1 = da * d + a * dd - (one(T) + one(T)) * b * db
        c2 = da * dd - db * db
        root = one(T)
        found = false
        if iszero(c2)
            if c1 < zero(T)
                candidate = -c0 / c1
                if zero(T) < candidate <= one(T)
                    root = candidate
                    found = true
                end
            end
        else
            discriminant = c1 * c1 - (one(T) + one(T) + one(T) + one(T)) * c2 * c0
            if discriminant >= zero(T)
                sqrt_discriminant = sqrt(discriminant)
                denominator = (one(T) + one(T)) * c2
                candidate1 = (-c1 - sqrt_discriminant) / denominator
                candidate2 = (-c1 + sqrt_discriminant) / denominator
                if zero(T) < candidate1 <= root
                    root = candidate1
                    found = true
                end
                if zero(T) < candidate2 <= root
                    root = candidate2
                    found = true
                end
            end
        end
        found && return root
        # A root is guaranteed between zero and one when the full step is not
        # positive definite. This fallback handles a nearly linear determinant
        # polynomial whose discriminant was rounded slightly below zero.
    elseif trial_isposdef!(scratch, X, one(T), dX)
        return one(T)
    end

    lower = zero(T)
    upper = one(T)
    for _ in 1:64
        midpoint = (lower + upper) / (one(T) + one(T))
        if trial_isposdef!(scratch, X, midpoint, dX)
            lower = midpoint
        else
            upper = midpoint
        end
    end
    return lower
end

# ---- symmetrize_inplace! : M ← (M + Mᵀ)/2, safe by the same
#     non-mutating-arithmetic-then-assign argument. ----

function symmetrize_inplace!(M::AbstractMatrix{T}) where {T}
    k = size(M, 1)
    @inbounds for j in 1:k, i in 1:(j-1)
        avg = (M[i, j] + M[j, i]) / 2
        M[i, j] = avg
        M[j, i] = avg
    end
    return M
end
