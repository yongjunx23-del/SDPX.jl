# src/cones/symmetric/psd.jl
#
# Positive-semidefinite cone `S_+^n` over `n×n` symmetric matrices, stored as
# the packed lower triangle of length `n(n+1)/2`. No Kronecker matrices are
# ever materialised: the Jordan product and NT scaling are built from plain
# symmetric matrix products (`mul!`, BLAS gemm on Float64).
#
# Jordan product: `X ∘ Y = (XY + YX)/2`; identity `I`; inverse `X^{-1}` via the
# eigendecomposition. The primitive idempotents of the Jordan algebra are the
# rank-one projections `E_k = v_k v_kᵀ` on the (unit-normalised) eigenvector
# columns `v_k`, which satisfy `E_k ∘ E_k = E_k` exactly.

# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------
"""
    membership(cone::PSDTriangleCone, v) -> Bool

`true` iff the packed lower triangle `v` represents a PSD matrix, i.e. every
eigenvalue of the symmetrised matrix is `≥ -tol` for a relative tolerance.
Uses the converged eigendecomposition on the preallocated workspace.
"""
function membership(cone::PSDTriangleCone{T}, v::AbstractVector) where {T}
    length(v) == cone.len || throw(DimensionMismatch())
    # Fail closed on non-finite coordinates before the eigensolver runs: an
    # eigendecomposition of a NaN/Inf matrix is not a cone-membership verdict.
    @inbounds for index in eachindex(v)
        isfinite(v[index]) || return false
    end
    s = cone.scratch
    _eigen!(s, v)
    n = cone.dim
    scale = zero(T)
    @inbounds for k in 1:n
        a = abs(s.w[k])
        scale = a > scale ? a : scale
    end
    tol = eps(T) * (scale > one(T) ? scale : one(T)) * T(n) * T(100)
    @inbounds for k in 1:n
        s.w[k] < -tol && return false
    end
    return true
end

"""The PSD cone is self-dual: `dual_membership == membership`."""
dual_membership(cone::PSDTriangleCone, v::AbstractVector) = membership(cone, v)

# ---------------------------------------------------------------------------
# Identity and Jordan product
# ---------------------------------------------------------------------------
"""Write the packed identity matrix `I` into `out` (ones on the diagonal)."""
function identity!(cone::PSDTriangleCone, out::AbstractVector)
    length(out) == cone.len || throw(DimensionMismatch())
    T = eltype(out)
    z = zero(T)
    o = one(T)
    n = cone.dim
    k = 0
    @inbounds for j in 1:n
        for i in j:n
            k += 1
            out[k] = i == j ? o : z
        end
    end
    return out
end

"""Allocating packed identity matrix (convenience)."""
identity_element(cone::PSDTriangleCone{T}, v::AbstractVector) where {T} =
    identity!(cone, Vector{T}(undef, cone.len))

"""`z = X ∘ Y = (XY + YX)/2` in packed storage, alias-safe."""
function jordan_product!(cone::PSDTriangleCone{T}, z::AbstractVector, x::AbstractVector, y::AbstractVector) where {T}
    length(z) == length(x) == length(y) == cone.len || throw(DimensionMismatch())
    n = cone.dim
    s = cone.scratch
    _unpack!(s.A, x, n)
    _unpack!(s.B, y, n)
    mul!(s.C, s.A, s.B)   # XY
    mul!(s.V, s.B, s.A)   # YX
    half = T(0.5)
    @inbounds for j in 1:n
        for i in 1:n
            s.C[i, j] = (s.C[i, j] + s.V[i, j]) * half
        end
    end
    _pack!(z, s.C, n)
    return z
end

# ---------------------------------------------------------------------------
# Inverse, sqrt, NT scaling
# ---------------------------------------------------------------------------
"""
`z = X^{-1}` via the eigendecomposition. For an interior (definite) `X` the
reciprocal eigenvalues are finite. Throws `DomainError` for a singular `X`.
"""
function inverse!(cone::PSDTriangleCone{T}, z::AbstractVector, x::AbstractVector) where {T}
    length(z) == length(x) == cone.len || throw(DimensionMismatch())
    n = cone.dim
    s = cone.scratch
    _eigen!(s, x)
    fill!(s.C, zero(T))
    @inbounds for k in 1:n
        wk = s.w[k]
        iszero(wk) && throw(DomainError(x, "PSD element is singular; inverse is undefined"))
        rk = one(T) / wk
        for j in 1:n
            for i in 1:n
                s.C[i, j] += rk * s.V[i, k] * s.V[j, k]
            end
        end
    end
    _pack!(z, s.C, n)
    return z
end

"""
`z = sqrt(X)` via the eigendecomposition. Tiny negative eigenvalues (round-off
on a PSD input) are clamped to zero.
"""
function sqrt!(cone::PSDTriangleCone{T}, z::AbstractVector, x::AbstractVector) where {T}
    n = cone.dim
    s = cone.scratch
    _eigen!(s, x)
    fill!(s.C, zero(T))
    @inbounds for k in 1:n
        wk = s.w[k]
        rk = wk <= zero(T) ? zero(T) : sqrt(wk)
        for j in 1:n
            for i in 1:n
                s.C[i, j] += rk * s.V[i, k] * s.V[j, k]
            end
        end
    end
    _pack!(z, s.C, n)
    return z
end

"""`W = X^{-1}` — the NT scaling point for the interior point `X`; `L_W` maps `X` to `I`."""
nt_scaling!(cone::PSDTriangleCone, W::AbstractVector, x::AbstractVector) = inverse!(cone, W, x)

@inline function _allfinite_matrix(A::AbstractMatrix)
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        isfinite(A[i, j]) || return false
    end
    return true
end

@inline function _psd_eigen_route!(
    A::AbstractMatrix,
    V::AbstractMatrix,
    values::AbstractVector,
    route::Symbol,
)
    route === :setup_jacobi || throw(ArgumentError(
        "PSD NT eigensolver route $(repr(route)) is unavailable; no fallback is permitted",
    ))
    return _jacobi_eigen!(A, V, values)
end

@inline function _psd_nt_close(A::AbstractMatrix{T}, B::AbstractMatrix, n::Int) where {T}
    residual = zero(T)
    scale = one(T)
    @inbounds for j in 1:n, i in 1:n
        aij = T(A[i, j])
        bij = T(B[i, j])
        rij = abs(aij - bij)
        residual = rij > residual ? rij : residual
        aa = abs(aij)
        ab = abs(bij)
        scale = aa > scale ? aa : scale
        scale = ab > scale ? ab : scale
    end
    # This is an internal same-precision orientation gate, not a solution or
    # certificate tolerance.  The factor covers the three eigensolver/
    # congruence stages used to construct P and Pinv; final Newton equations
    # and original-coordinate certificates remain independently verified.
    return residual <= eps(T) * scale * T(10000 * n)
end

"""In-place unpivoted Cholesky `L` with `L*L' = X` (lower factor stored in
`L`). Throws `DomainError` for a non-strictly-positive-definite input; the
HSD caller has already verified strict interiority, and failing closed is
the contract — no pivoted or fallback variant is permitted."""
function _chol_unpivoted!(L::AbstractMatrix{T}, X::AbstractMatrix, n::Int) where {T}
    @inbounds for j in 1:n
        for i in j:n
            value = X[i, j]
            for k in 1:(j - 1)
                value -= L[i, k] * L[j, k]
            end
            if i == j
                (isfinite(value) && value > zero(T)) || throw(DomainError(
                    value, "PSD NT Cholesky requires a strictly positive definite pair",
                ))
                L[j, j] = sqrt(value)
            else
                value /= L[j, j]
                isfinite(value) || throw(DomainError(
                    value, "PSD NT Cholesky produced a non-finite factor",
                ))
                L[i, j] = value
            end
        end
    end
    return L
end

"""`W = L⁻¹` for a lower-triangular `L`, via forward substitution against the
identity columns. No explicit inverse is formed anywhere else in the NT
construction; this is the single triangular inverse the Cholesky route needs."""
function _l_inverse!(W::AbstractMatrix{T}, L::AbstractMatrix, n::Int) where {T}
    @inbounds for j in 1:n
        for i in 1:n
            value = i == j ? one(T) : zero(T)
            for k in 1:(i - 1)
                value -= L[i, k] * W[k, j]
            end
            value /= L[i, i]
            isfinite(value) || throw(DomainError(
                value, "PSD NT Cholesky inverse produced a non-finite entry",
            ))
            W[i, j] = value
        end
    end
    return W
end

"""
    psd_congruence_factors!(LX, MY, X, Y)

Lower Cholesky factors `X = LX*LX'`, `Y = MY*MY'` used by the PSD Schur panel
congruence, computed through the same fail-closed unpivoted route as the NT
construction (`_chol_unpivoted!`).  This is the scalar/reference factor entry
point: the panelized kernels route the identical factors through the LA
provider seam (`la_chol!`) instead, and both sides of the parity comparison
share the same factor semantics.
"""
function psd_congruence_factors!(
    LX::AbstractMatrix{T},
    MY::AbstractMatrix{T},
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
) where {T}
    n = size(X, 1)
    size(X, 2) == n || throw(DimensionMismatch(
        "PSD congruence factor source X must be square",
    ))
    size(Y) == (n, n) || throw(DimensionMismatch(
        "PSD congruence factor source Y must be n×n",
    ))
    size(LX) == (n, n) || throw(DimensionMismatch(
        "PSD congruence factor buffer LX must be n×n",
    ))
    size(MY) == (n, n) || throw(DimensionMismatch(
        "PSD congruence factor buffer MY must be n×n",
    ))
    _chol_unpivoted!(LX, X, n)
    _chol_unpivoted!(MY, Y, n)
    return LX, MY
end

"""Symmetrize in place: `A = (A + A')/2`. Roundoff in chained triangular
congruences need not be bitwise symmetric; downstream eigensolvers and gates
require one authoritative symmetric matrix."""
function _symmetrize!(A::AbstractMatrix{T}, n::Int) where {T}
    half = one(T) / (one(T) + one(T))
    @inbounds for j in 1:n, i in (j + 1):n
        value = (A[i, j] + A[j, i]) * half
        A[i, j] = value
        A[j, i] = value
    end
    return A
end

"""Compute the SPD square root and inverse square root without allocation."""
function _spd_sqrt_invsqrt!(
    root::AbstractMatrix{T},
    invroot::AbstractMatrix{T},
    X::AbstractMatrix{T},
    work::AbstractMatrix{T},
    V::AbstractMatrix{T},
    values::AbstractVector{T},
    eigen_route::Symbol,
) where {T}
    n = size(X, 1)
    size(X, 2) == n || throw(DimensionMismatch())
    _allfinite_matrix(X) || throw(DomainError(X, "SPD matrix contains non-finite entries"))
    copyto!(work, X)
    _identity!(V, n)
    _psd_eigen_route!(work, V, values, eigen_route)
    _orthonormalize!(V, n)
    fill!(root, zero(T))
    fill!(invroot, zero(T))
    @inbounds for k in 1:n
        value = values[k]
        (isfinite(value) && value > zero(T)) ||
            throw(DomainError(value, "PSD NT pair must be strictly positive definite"))
        rk = sqrt(value)
        ik = one(T) / rk
        for j in 1:n
            vjk = V[j, k]
            for i in 1:n
                vv = V[i, k] * vjk
                root[i, j] += rk * vv
                invroot[i, j] += ik * vv
            end
        end
    end
    return root, invroot
end

"""
    nt_scaling!(cone, state::PSDNTScaling, s, y)

Update the pair-dependent PSD NT state from HSD `svec` coordinates. With
`Y = L*L'` and `M = L' S L`, computes the congruence-form NT scaling
`P = L^{-T} M^{1/2} L^{-1}`, so `Theta[Z]=P*Z*P`, `Theta(Y)=S`, and
`G=Theta^{-1}`. Any non-finite input, non-positive eigenvalue, Cholesky
failure, or Jacobi non-convergence throws and is never silently accepted.
"""
function nt_scaling!(
    cone::PSDTriangleCone{T},
    state::PSDNTScaling{T},
    svec_s::AbstractVector,
    svec_y::AbstractVector,
) where {T}
    state.valid[1] = false
    length(svec_s) == length(svec_y) == cone.len == state.len ||
        throw(DimensionMismatch())
    cone.dim == state.dim || throw(DimensionMismatch())
    n = state.dim
    _unpack_svec!(state.S, svec_s, n, state.invsqrt2)
    _unpack_svec!(state.Y, svec_y, n, state.invsqrt2)

    # Cholesky-stable NT construction.  The textbook form
    # `P = Y^{-1/2}(Y^{1/2} S Y^{1/2})^{1/2}Y^{-1/2}` builds `P` through an
    # explicit `Y^(-1/2)` eigensystem whose roundoff grows like `cond(Y)`;
    # near the PSD boundary that residual can cross the same-precision
    # orientation gate even for exactly representable iterates.  With
    # `Y = L*L'` and `M = L' S L`, the congruence form
    #     P = L^{-T} M^{1/2} L^{-1}
    # satisfies `P*Y*P = S` identically, replaces the ill-conditioned
    # eigensystem by backward-stable triangular solves, and keeps the same
    # fail-closed orientation gates below.
    _chol_unpivoted!(state.chol, state.Y, n)
    mul!(state.work1, state.chol', state.S)
    mul!(state.work2, state.work1, state.chol)
    # core = M^(1/2), coreinv = M^(-1/2) via the same eigen route as before.
    _spd_sqrt_invsqrt!(
        state.work4,
        state.work3,
        state.work2,
        state.work1,
        state.U,
        state.lambda,
        state.eigen_route,
    )
    _l_inverse!(state.chol_inv, state.chol, n)
    # P = W' core W with W = L^{-1}.
    mul!(state.work1, state.work4, state.chol_inv)
    mul!(state.P, state.chol_inv', state.work1)
    _symmetrize!(state.P, n)

    # Freeze the square-root automorphism and its inverse.
    _spd_sqrt_invsqrt!(
        state.Proot,
        state.Prootinv,
        state.P,
        state.work3,
        state.U,
        state.lambda,
        state.eigen_route,
    )
    mul!(state.Pinv, state.Prootinv, state.Prootinv)

    # Lambda = R(Y) = P^(1/2) Y P^(1/2), then freeze its eigenbasis.
    mul!(state.work3, state.Proot, state.Y)
    mul!(state.Lambda, state.work3, state.Proot)
    copyto!(state.work3, state.Lambda)
    _identity!(state.U, n)
    _psd_eigen_route!(state.work3, state.U, state.lambda, state.eigen_route)
    _orthonormalize!(state.U, n)
    @inbounds for k in 1:n
        lk = state.lambda[k]
        (isfinite(lk) && lk > zero(T)) ||
            throw(DomainError(lk, "PSD scaled Lambda must be positive definite"))
    end
    # Fail closed unless the frozen dual->primal orientation and its square
    # root are numerically verified in the same precision as the state.
    mul!(state.work1, state.P, state.Y)
    mul!(state.work2, state.work1, state.P)
    _psd_nt_close(state.work2, state.S, n) || throw(DomainError(
        state.P, "PSD NT Theta(Y)=S residual exceeded tolerance",
    ))
    mul!(state.work1, state.Pinv, state.S)
    mul!(state.work2, state.work1, state.Pinv)
    _psd_nt_close(state.work2, state.Y, n) || throw(DomainError(
        state.Pinv, "PSD NT G(S)=Y residual exceeded tolerance",
    ))
    mul!(state.work1, state.Prootinv, state.S)
    mul!(state.work2, state.work1, state.Prootinv)
    _psd_nt_close(state.work2, state.Lambda, n) || throw(DomainError(
        state.Proot, "PSD NT scaled-Lambda residual exceeded tolerance",
    ))
    mul!(state.work1, state.Proot, state.Proot)
    _psd_nt_close(state.work1, state.P, n) ||
        throw(DomainError(state.Proot, "PSD NT R^2=Theta residual exceeded tolerance"))
    state.valid[1] = true
    return state
end

@inline function _psd_congruence_svec!(
    out::AbstractVector,
    state::PSDNTScaling,
    P::AbstractMatrix,
    x::AbstractVector,
)
    _require_nt_valid(state)
    length(out) == length(x) == state.len || throw(DimensionMismatch())
    n = state.dim
    _unpack_svec!(state.work1, x, n, state.invsqrt2)
    mul!(state.work2, P, state.work1)
    mul!(state.work1, state.work2, P)
    _pack_svec!(out, state.work1, n, state.sqrt2)
    return out
end

theta_apply!(
    ::PSDTriangleCone,
    out::AbstractVector,
    state::PSDNTScaling,
    x::AbstractVector,
) = _psd_congruence_svec!(out, state, state.P, x)

g_apply!(
    ::PSDTriangleCone,
    out::AbstractVector,
    state::PSDNTScaling,
    x::AbstractVector,
) = _psd_congruence_svec!(out, state, state.Pinv, x)

r_apply!(
    ::PSDTriangleCone,
    out::AbstractVector,
    state::PSDNTScaling,
    x::AbstractVector,
) = _psd_congruence_svec!(out, state, state.Proot, x)

r_inverse_apply!(
    ::PSDTriangleCone,
    out::AbstractVector,
    state::PSDNTScaling,
    x::AbstractVector,
) = _psd_congruence_svec!(out, state, state.Prootinv, x)

"""Solve `L_Lambda(out)=rhs` in the frozen eigenbasis, in `svec` coordinates."""
function solve_Llambda!(
    cone::PSDTriangleCone{T},
    out::AbstractVector,
    state::PSDNTScaling{T},
    rhs::AbstractVector,
) where {T}
    _require_nt_valid(state)
    length(out) == length(rhs) == cone.len == state.len || throw(DimensionMismatch())
    n = state.dim
    two = one(T) + one(T)
    _unpack_svec!(state.work1, rhs, n, state.invsqrt2)
    mul!(state.work2, transpose(state.U), state.work1)
    mul!(state.work3, state.work2, state.U)
    @inbounds for j in 1:n
        for i in 1:n
            denom = state.lambda[i] + state.lambda[j]
            (isfinite(denom) && denom > zero(T)) ||
                throw(DomainError(denom, "L_Lambda is not positive definite"))
            state.work3[i, j] *= two / denom
        end
    end
    mul!(state.work2, state.U, state.work3)
    mul!(state.work1, state.work2, transpose(state.U))
    _pack_svec!(out, state.work1, n, state.sqrt2)
    return out
end

"""`y = W X = (W X + X W)/2` (Jordan product by the scaling point), packed."""
function scaling_apply!(cone::PSDTriangleCone{T}, y::AbstractVector, W::AbstractVector, x::AbstractVector) where {T}
    n = cone.dim
    s = cone.scratch
    _unpack!(s.A, W, n)
    _unpack!(s.B, x, n)
    mul!(s.C, s.A, s.B)   # WX
    mul!(s.V, s.B, s.A)   # XW
    half = T(0.5)
    @inbounds for j in 1:n
        for i in 1:n
            s.C[i, j] = (s.C[i, j] + s.V[i, j]) * half
        end
    end
    _pack!(y, s.C, n)
    return y
end

"""Solve `L_W(Y) = X`, where `L_W(Y) = (WY + YW)/2`, in raw packed coordinates."""
function scaling_inverse_apply!(cone::PSDTriangleCone{T}, y::AbstractVector, W::AbstractVector, x::AbstractVector) where {T}
    length(y) == length(W) == length(x) == cone.len || throw(DimensionMismatch())
    n = cone.dim
    s = cone.scratch
    _eigen!(s, W)
    @inbounds for k in 1:n
        wk = s.w[k]
        (isfinite(wk) && wk > zero(T)) ||
            throw(DomainError(wk, "L_W inverse requires a positive-definite W"))
    end
    _unpack!(s.B, x, n)
    mul!(s.C, transpose(s.V), s.B)
    mul!(s.A, s.C, s.V)
    two = one(T) + one(T)
    @inbounds for j in 1:n
        for i in 1:n
            s.A[i, j] *= two / (s.w[i] + s.w[j])
        end
    end
    mul!(s.C, s.V, s.A)
    mul!(s.B, s.C, transpose(s.V))
    _pack!(y, s.B, n)
    return y
end

# ---------------------------------------------------------------------------
# Boundary step
# ---------------------------------------------------------------------------
"""
    boundary_step!(cone, x, alpha, p)

Largest `α ≥ 0` such that `X + α p` stays PSD, using the generalized
eigenvalues of `(p, X)`.  For strictly positive-definite `X = L*L'`, these
are the ordinary eigenvalues of `L⁻¹*p*L⁻ᵀ`, and
`α = -1/min λ` (`Inf` when the direction never leaves).  The Cholesky frame is
used directly: unlike an explicit eigendecomposition of a near-singular `X`,
it never turns a positive pivot into a zero inverse square root. Stores the
value into `Ref` `alpha` and returns it.
"""
function boundary_step!(cone::PSDTriangleCone{T}, x::AbstractVector, alpha::Base.RefValue, p::AbstractVector) where {T}
    length(x) == length(p) == cone.len || throw(DimensionMismatch())
    n = cone.dim
    s = cone.scratch
    _unpack!(s.A, x, n)

    # In-place unpivoted Cholesky X=L*L'.  HSD calls this kernel only from a
    # strict-interior point; returning zero is the conservative boundary for
    # a direct caller that supplies a non-SPD point.
    @inbounds for j in 1:n
        for i in j:n
            value = s.A[i, j]
            for k in 1:(j - 1)
                value -= s.A[i, k] * s.A[j, k]
            end
            if i == j
                if !(isfinite(value) && value > zero(T))
                    alpha[] = zero(T)
                    return zero(T)
                end
                s.A[j, j] = sqrt(value)
            else
                value /= s.A[j, j]
                if !isfinite(value)
                    alpha[] = zero(T)
                    return zero(T)
                end
                s.A[i, j] = value
            end
        end
    end

    _unpack!(s.B, p, n)
    # C=L⁻¹*p (forward solve, one right-hand side per column).
    @inbounds for j in 1:n
        for i in 1:n
            value = s.B[i, j]
            for k in 1:(i - 1)
                value -= s.A[i, k] * s.C[k, j]
            end
            s.C[i, j] = value / s.A[i, i]
        end
    end
    # B=C*L⁻ᵀ.  Each row is another forward solve by L.
    @inbounds for i in 1:n
        for j in 1:n
            value = s.C[i, j]
            for k in 1:(j - 1)
                value -= s.B[i, k] * s.A[j, k]
            end
            s.B[i, j] = value / s.A[j, j]
        end
    end
    # Roundoff in the two triangular solves need not be bitwise symmetric.
    # Jacobi requires one authoritative symmetric matrix, so average the pair.
    half = one(T) / (one(T) + one(T))
    @inbounds for j in 1:n, i in 1:n
        s.A[i, j] = (s.B[i, j] + s.B[j, i]) * half
    end
    _identity!(s.V, n)
    _jacobi_eigen!(s.A, s.V, s.w)
    mu_min = s.w[1]
    @inbounds for k in 2:n
        mu_min = s.w[k] < mu_min ? s.w[k] : mu_min
    end
    step = mu_min >= zero(T) ? T(Inf) : -one(T) / mu_min
    alpha[] = step
    return step
end

# ---------------------------------------------------------------------------
# Barrier derivatives
# ---------------------------------------------------------------------------
"""
`g = ∇F(x)` for the barrier `F(X) = -log det(X)`, in packed coordinates:
`g = -X^{-1}` with every off-diagonal packed entry doubled. (Doubling matches
the packed-coordinate derivative `∂F/∂v_k`; see [`barrier_hessian_product!`](@ref).)
"""
function barrier_gradient!(cone::PSDTriangleCone{T}, g::AbstractVector, x::AbstractVector) where {T}
    length(g) == length(x) == cone.len || throw(DimensionMismatch())
    inverse!(cone, g, x)          # g = X^{-1} (packed)
    n = cone.dim
    k = 0
    @inbounds for j in 1:n
        for i in j:n
            k += 1
            g[k] = -(i > j ? g[k] * T(2) : g[k])
        end
    end
    return g
end

"""
`h = F''(x)[d] = X^{-1} d X^{-1}` in packed coordinates, with the off-diagonal
entries doubled to match the packed-coordinate derivative convention. Here `d`
is the symmetric matrix represented by the packed direction vector.
"""
function barrier_hessian_product!(cone::PSDTriangleCone{T}, h::AbstractVector, x::AbstractVector, d::AbstractVector) where {T}
    length(h) == length(x) == length(d) == cone.len || throw(DimensionMismatch())
    n = cone.dim
    s = cone.scratch
    _eigen!(s, x)
    fill!(s.C, zero(T))
    @inbounds for k in 1:n
        wk = s.w[k]
        rk = one(T) / wk
        for j in 1:n
            for i in 1:n
                s.C[i, j] += rk * s.V[i, k] * s.V[j, k]
            end
        end
    end
    _unpack!(s.B, d, n)
    mul!(s.A, s.C, s.B)   # X^{-1} d
    mul!(s.B, s.A, s.C)   # X^{-1} d X^{-1}
    _pack!(h, s.B, n)
    # double off-diagonal packed entries
    k = 0
    @inbounds for j in 1:n
        for i in j:n
            k += 1
            if i > j
                h[k] *= T(2)
            end
        end
    end
    return h
end

# ---------------------------------------------------------------------------
# Third-order correction
# ---------------------------------------------------------------------------
"""
Legacy raw-packed compatibility overload for `w = d1 ∘ (d2 ∘ d3)`. It
allocates one temporary and is not a hot API; production svec code must use
the `PSDNTScaling` overload below.
"""
function third_order_correction!(cone::PSDTriangleCone, w::AbstractVector, d1::AbstractVector, d2::AbstractVector, d3::AbstractVector)
    tmp = similar(d2)
    jordan_product!(cone, tmp, d2, d3)
    return jordan_product!(cone, w, d1, tmp)
end

@inline function _psd_jordan_svec!(
    out::AbstractVector,
    state::PSDNTScaling{T},
    x::AbstractVector,
    y::AbstractVector,
) where {T}
    n = state.dim
    _unpack_svec!(state.work1, x, n, state.invsqrt2)
    _unpack_svec!(state.work2, y, n, state.invsqrt2)
    mul!(state.work3, state.work1, state.work2)
    mul!(state.work4, state.work2, state.work1)
    half = one(T) / (one(T) + one(T))
    @inbounds for j in 1:n, i in 1:n
        state.work3[i, j] = (state.work3[i, j] + state.work4[i, j]) * half
    end
    _pack_svec!(out, state.work3, n, state.sqrt2)
    return out
end

"""Preallocated PSD third-order correction in HSD `svec` coordinates."""
function third_order_correction!(
    cone::PSDTriangleCone,
    state::PSDNTScaling,
    w::AbstractVector,
    d1::AbstractVector,
    d2::AbstractVector,
    d3::AbstractVector,
)
    _require_nt_valid(state)
    length(w) == length(d1) == length(d2) == length(d3) == state.len == cone.len ||
        throw(DimensionMismatch())
    _psd_jordan_svec!(w, state, d2, d3)
    return _psd_jordan_svec!(w, state, d1, w)
end

# ---------------------------------------------------------------------------
# Spectral decomposition (allocating, test/analysis oriented)
# ---------------------------------------------------------------------------
"""
    spectrum(cone::PSDTriangleCone, x) -> (eigenvalues, eigenvectors)

Eigenvalues (ascending) and eigenvectors (columns) of the packed symmetric
matrix `x`. Each eigenvector column is unit-normalised so the primitive
idempotent `E_k = v_k v_kᵀ` satisfies `E_k ∘ E_k = E_k`.
"""
function spectrum(cone::PSDTriangleCone{T}, x::AbstractVector) where {T}
    length(x) == cone.len || throw(DimensionMismatch())
    s = cone.scratch
    _eigen!(s, x)
    return copy(s.w), copy(s.V)
end

"""
    primitive_idempotents(cone::PSDTriangleCone, x) -> Vector{Matrix}

The `n` rank-one projections `E_k = v_k v_kᵀ` on the eigenvector columns. Each
is an exact idempotent of the PSD Jordan algebra: `E_k ∘ E_k = E_k`.
"""
function primitive_idempotents(cone::PSDTriangleCone{T}, x::AbstractVector) where {T}
    n = cone.dim
    w, V = spectrum(cone, x)
    return [let E = zeros(T, n, n)
        @inbounds for j in 1:n, i in 1:n
            E[i, j] += V[i, k] * V[j, k]
        end
        E
    end for k in 1:n]
end
