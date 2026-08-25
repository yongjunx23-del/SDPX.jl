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

"""
`y = W^{-1} x`. Since `W = L_{X^{-1}}`, `W^{-1} = L_X`, so `y = X ∘ X = X²`
(the Jordan square).
"""
function scaling_inverse_apply!(cone::PSDTriangleCone{T}, y::AbstractVector, W::AbstractVector, x::AbstractVector) where {T}
    n = cone.dim
    s = cone.scratch
    _unpack!(s.A, x, n)
    mul!(s.C, s.A, s.A)
    _pack!(y, s.C, n)
    return y
end

# ---------------------------------------------------------------------------
# Boundary step
# ---------------------------------------------------------------------------
"""
    boundary_step!(cone, x, alpha, p)

Largest `α ≥ 0` such that `X + α p` stays PSD, using
`α = -1/min λ(X^{-1/2} p X^{-1/2})` (`Inf` when the direction never leaves).
Stores the value into `Ref` `alpha` and returns it.
"""
function boundary_step!(cone::PSDTriangleCone{T}, x::AbstractVector, alpha::Base.RefValue, p::AbstractVector) where {T}
    length(x) == length(p) == cone.len || throw(DimensionMismatch())
    n = cone.dim
    s = cone.scratch
    # X^{-1/2} in s.C
    _eigen!(s, x)
    fill!(s.C, zero(T))
    @inbounds for k in 1:n
        wk = s.w[k]
        rk = wk <= zero(T) ? zero(T) : one(T) / sqrt(wk)
        for j in 1:n
            for i in 1:n
                s.C[i, j] += rk * s.V[i, k] * s.V[j, k]
            end
        end
    end
    _unpack!(s.B, p, n)
    mul!(s.A, s.C, s.B)   # X^{-1/2} p
    mul!(s.B, s.A, s.C)   # X^{-1/2} p X^{-1/2}  (symmetric)
    copyto!(s.A, s.B)
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
"""`w = d1 ∘ (d2 ∘ d3)` (allocates one temporary; not on the zero-alloc list)."""
function third_order_correction!(cone::PSDTriangleCone, w::AbstractVector, d1::AbstractVector, d2::AbstractVector, d3::AbstractVector)
    tmp = similar(d2)
    jordan_product!(cone, tmp, d2, d3)
    return jordan_product!(cone, w, d1, tmp)
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
