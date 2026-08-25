# src/cone_algebra.jl
#
# Wave A-5: mathematically-correct ConeAlgebra layer (orthant + SOC + PSD).
#
# This is a self-contained math-validation module. It is deliberately NOT wired
# into the solver: the solver continues to use its existing SOC/PSD kernels.
# The module provides the Jordan-algebra primitives for the three standard
# self-dual cones, a generic symmetric Jacobi eigensolver that works across
# Float64 / Float64x2 / Float64x4 / BigFloat, and both allocating and mutating
# (workspace) boundary-step APIs.
#
# Conventions
# -----------
# * ORTHANT  R_+^n : componentwise Jordan product; NT point w = sqrt.(x ./ s)
#   satisfies W*Y*W = X with Y = s (the dual).
# * SOC      Q^n   : (t,u) o (s,v) = (t*s + u·v, t*v + s*u); identity (1,0,..).
# * PSD      S^n_+ : X o Y = (XY + YX)/2; NT point
#   W = X^{1/2} (X^{1/2} Y X^{1/2})^{-1/2} X^{1/2} satisfies W*Y*W = X.

module ConeAlgebra

using LinearAlgebra
import Base: sqrt

# ---------------------------------------------------------------------------
# Cone tags
# ---------------------------------------------------------------------------
abstract type Cone end
struct OrthantCone <: Cone end
struct LorentzCone <: Cone end
struct PSDCone <: Cone end

# ---------------------------------------------------------------------------
# Generic symmetric Jacobi eigensolver.
#
# Works for any element type supporting +,-,*,/,sqrt,abs,sign,eps (Float64,
# Float64x2, Float64x4, BigFloat, ...). Returns (eigenvalues, eigenvectors)
# with the eigenvectors as columns of V (X = V diag(λ) Vᵀ).
# ---------------------------------------------------------------------------
function _jacobi_eigen(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("matrix must be square"))
    n == 0 && return T[], Matrix{T}(undef, 0, 0)
    z = zero(T)
    o = one(T)
    two = o + o
    W = Matrix{T}(undef, n, n)
    copyto!(W, A)
    V = Matrix{T}(I, n, n)
    n == 1 && return T[W[1, 1]], V
    tol = eps(T) * T(n) * T(10)
    max_iter = 100
    for _ in 1:max_iter
        # locate the largest off-diagonal entry
        p = 1
        q = 2
        maxval = abs(W[1, 2])
        for j in 1:n, i in 1:(j - 1)
            v = abs(W[i, j])
            if v > maxval
                maxval = v
                p = i
                q = j
            end
        end
        maxval < tol && break
        app = W[p, p]
        aqq = W[q, q]
        apq = W[p, q]
        theta = (aqq - app) / (two * apq)
        t = iszero(theta) ? o : sign(theta) / (abs(theta) + sqrt(theta * theta + o))
        c = o / sqrt(t * t + o)
        s = t * c
        for k in 1:n
            if k != p && k != q
                akp = W[k, p]
                akq = W[k, q]
                W[k, p] = c * akp - s * akq
                W[p, k] = W[k, p]
                W[k, q] = s * akp + c * akq
                W[q, k] = W[k, q]
            end
        end
        W[p, p] = app - t * apq
        W[q, q] = aqq + t * apq
        W[p, q] = z
        W[q, p] = z
        for k in 1:n
            vkp = V[k, p]
            vkq = V[k, q]
            V[k, p] = c * vkp - s * vkq
            V[k, q] = s * vkp + c * vkq
        end
    end
    eigenvalues = T[W[i, i] for i in 1:n]
    return eigenvalues, V
end

# ---------------------------------------------------------------------------
# ORTHANT  (nonnegative orthant R_+^n)
# ---------------------------------------------------------------------------
"""NT scaling point for the orthant: `w = sqrt.(x ./ s)`, so `W*Y*W = X`."""
nt_scaling(::OrthantCone, x, s) = sqrt.(x ./ s)

"""Componentwise Jordan product for the orthant."""
jordan_product(::OrthantCone, x, y) = x .* y

"""Componentwise square root for the orthant."""
sqrt(::OrthantCone, x) = sqrt.(x)

"""Componentwise inverse for the orthant."""
inverse(::OrthantCone, x) = 1 ./ x

"""
Largest `alpha >= 0` such that `x + alpha*dx` stays in the orthant.
Returns `0` if `x` is outside the cone, `Inf` if the ray never leaves.
"""
function boundary_step(::OrthantCone, x, dx)
    T = promote_type(eltype(x), eltype(dx))
    z = zero(T)
    step = T(Inf)
    @inbounds for i in eachindex(x, dx)
        xi = T(x[i])
        di = T(dx[i])
        xi < z && return z          # outside the cone
        if di < z
            step = min(step, -xi / di)
        end
    end
    return step
end

# convenience (allocating + mutating)
orthant_nt_scaling(x, s) = nt_scaling(OrthantCone(), x, s)
orthant_jordan_product(x, y) = jordan_product(OrthantCone(), x, y)
orthant_sqrt(x) = sqrt(OrthantCone(), x)
orthant_inverse(x) = inverse(OrthantCone(), x)
orthant_boundary_step(x, dx) = boundary_step(OrthantCone(), x, dx)
function orthant_boundary_step!(w, x, dx)
    step = boundary_step(OrthantCone(), x, dx)
    w[1] = step
    return step
end

# ---------------------------------------------------------------------------
# SOC  (Lorentz cone Q^n = {(t,u) : t >= ||u||})
# ---------------------------------------------------------------------------
"""Lorentz Jordan product `(t,u) o (s,v) = (t*s + u·v, t*v + s*u)`."""
function jordan_product(::LorentzCone, x, y)
    length(x) == length(y) || throw(DimensionMismatch())
    T = promote_type(eltype(x), eltype(y))
    out = Vector{T}(undef, length(x))
    head = x[1] * y[1]
    @inbounds for i in 2:length(x)
        head += x[i] * y[i]
    end
    out[1] = head
    @inbounds for i in 2:length(x)
        out[i] = x[1] * y[i] + y[1] * x[i]
    end
    return out
end

"""In-place Lorentz Jordan product (alias-safe)."""
function jordan_product!(out, x, y)
    length(out) == length(x) == length(y) || throw(DimensionMismatch())
    xh = x[1]
    yh = y[1]
    head = xh * yh
    @inbounds for i in 2:length(x)
        head += x[i] * y[i]
    end
    out[1] = head
    @inbounds for i in 2:length(x)
        out[i] = xh * y[i] + yh * x[i]
    end
    return out
end

"""
Spectral decomposition of a Lorentz element: `x = λ1*c1 + λ2*c2`.
When the tail is zero (`|u| == 0`) both idempotents are `(1/2, 0, ..., 0)`.
"""
function spectral(::LorentzCone, x)
    T = eltype(x)
    length(x) >= 2 || throw(ArgumentError("SOC element must have length >= 2"))
    t = T(x[1])
    u2 = zero(T)
    @inbounds for i in 2:length(x)
        u2 += x[i] * x[i]
    end
    nu = sqrt(u2)
    lam1 = t + nu
    lam2 = t - nu
    c1 = Vector{T}(undef, length(x))
    c2 = Vector{T}(undef, length(x))
    half = one(T) / 2
    if iszero(nu)
        # zero-tail: both idempotents are (1/2, 0, ..., 0)
        c1[1] = half
        c2[1] = half
        @inbounds for i in 2:length(x)
            c1[i] = zero(T)
            c2[i] = zero(T)
        end
    else
        two = one(T) + one(T)
        c1[1] = half
        c2[1] = half
        @inbounds for i in 2:length(x)
            c1[i] = x[i] / (two * nu)
            c2[i] = -x[i] / (two * nu)
        end
    end
    return lam1, lam2, c1, c2
end

"""Lorentz square root (for `x` in the cone)."""
function sqrt(::LorentzCone, x)
    T = eltype(x)
    t = T(x[1])
    u2 = zero(T)
    @inbounds for i in 2:length(x)
        u2 += x[i] * x[i]
    end
    nu = sqrt(u2)
    out = Vector{T}(undef, length(x))
    if iszero(nu)
        r = sqrt(t)
        out[1] = r
        @inbounds for i in 2:length(x)
            out[i] = zero(T)
        end
    else
        two = one(T) + one(T)
        lam1 = t + nu
        lam2 = t - nu
        r1 = sqrt(lam1)
        r2 = sqrt(lam2)
        out[1] = (r1 + r2) / 2
        @inbounds for i in 2:length(x)
            out[i] = (r1 - r2) * x[i] / (two * nu)
        end
    end
    return out
end

"""Lorentz inverse `x^{-1} = (t, -u)/det(x)`."""
function inverse(::LorentzCone, x)
    T = eltype(x)
    t = T(x[1])
    det = t * t
    @inbounds for i in 2:length(x)
        det -= x[i] * x[i]
    end
    out = Vector{T}(undef, length(x))
    out[1] = t / det
    @inbounds for i in 2:length(x)
        out[i] = -x[i] / det
    end
    return out
end

"""
Largest `alpha >= 0` such that `x + alpha*dx` stays in the Lorentz cone.

Handles the linear (`c2 == 0`) case, the repeated-root / tangential case
(`disc == 0`), outside-cone detection (returns `0`), and the head constraint
`t + alpha*dt >= 0`. Analytic checks: `x=(2,0), dx=(-1,0) -> 2`;
`x=(2,0), dx=(-1,1) -> 1`.
"""
function boundary_step(::LorentzCone, x, dx)
    T = promote_type(eltype(x), eltype(dx))
    z = zero(T)
    o = one(T)
    two = o + o
    four = two + two
    t = T(x[1])
    dt = T(dx[1])
    c0 = t * t
    c1 = t * dt
    c2 = dt * dt
    @inbounds for i in 2:length(x)
        u = T(x[i])
        du = T(dx[i])
        c0 -= u * u
        c1 -= u * du
        c2 -= du * du
    end
    c1 *= two
    # outside-cone detection
    if c0 < z || t < z
        return z
    end
    # head constraint: t + alpha*dt >= 0
    head_step = dt < z ? -t / dt : T(Inf)
    # determinant constraint: c0 + c1*alpha + c2*alpha^2 >= 0
    if c0 == z
        # x on the boundary
        if c1 < z
            det_step = z
        elseif c1 == z
            det_step = c2 >= z ? T(Inf) : z
        else # c1 > 0
            det_step = c2 < z ? -c1 / c2 : T(Inf)
        end
    else
        # x interior (c0 > 0)
        if c2 == z
            # linear case
            det_step = c1 < z ? -c0 / c1 : T(Inf)
        elseif c2 > z
            disc = c1 * c1 - four * c0 * c2
            if disc <= z
                # repeated root or no root: quadratic stays >= 0
                det_step = T(Inf)
            else
                r1 = (-c1 - sqrt(disc)) / (two * c2)
                det_step = r1 > z ? r1 : T(Inf)
            end
        else # c2 < 0
            disc = c1 * c1 - four * c0 * c2
            if disc <= z
                det_step = z
            else
                hi = (-c1 - sqrt(disc)) / (two * c2)
                det_step = hi > z ? hi : z
            end
        end
    end
    return min(head_step, det_step)
end

"""Mutating Lorentz boundary step; writes the step into `w[1]`."""
function boundary_step!(w, ::LorentzCone, x, dx)
    step = boundary_step(LorentzCone(), x, dx)
    w[1] = step
    return step
end

# convenience (allocating + mutating)
soc_jordan_product(x, y) = jordan_product(LorentzCone(), x, y)
soc_jordan_product!(out, x, y) = jordan_product!(out, x, y)
soc_spectral(x) = spectral(LorentzCone(), x)
soc_sqrt(x) = sqrt(LorentzCone(), x)
soc_inverse(x) = inverse(LorentzCone(), x)
soc_boundary_step(x, dx) = boundary_step(LorentzCone(), x, dx)
soc_boundary_step!(w, x, dx) = boundary_step!(w, LorentzCone(), x, dx)

# ---------------------------------------------------------------------------
# PSD  (positive semidefinite cone S^n_+)
# ---------------------------------------------------------------------------
"""Symmetrized Jordan product `X o Y = (XY + YX)/2`."""
jordan_product(::PSDCone, X, Y) = (X * Y + Y * X) / 2

"""PSD matrix square root via the eigendecomposition."""
function sqrt(::PSDCone, X)
    T = eltype(X)
    vals, vecs = _jacobi_eigen(X)
    n = size(X, 1)
    out = zeros(T, n, n)
    @inbounds for k in 1:n
        r = sqrt(vals[k])
        for i in 1:n, j in 1:n
            out[i, j] += r * vecs[i, k] * vecs[j, k]
        end
    end
    return out
end

"""PSD matrix inverse via the eigendecomposition."""
function inverse(::PSDCone, X)
    T = eltype(X)
    vals, vecs = _jacobi_eigen(X)
    n = size(X, 1)
    out = zeros(T, n, n)
    @inbounds for k in 1:n
        r = one(T) / vals[k]
        for i in 1:n, j in 1:n
            out[i, j] += r * vecs[i, k] * vecs[j, k]
        end
    end
    return out
end

"""`X^{-1/2}` via the eigendecomposition."""
function _psd_inverse_sqrt(X)
    T = eltype(X)
    vals, vecs = _jacobi_eigen(X)
    n = size(X, 1)
    out = zeros(T, n, n)
    @inbounds for k in 1:n
        r = one(T) / sqrt(vals[k])
        for i in 1:n, j in 1:n
            out[i, j] += r * vecs[i, k] * vecs[j, k]
        end
    end
    return out
end

"""
Largest `alpha >= 0` such that `X + alpha*dX` stays PSD, via the generalized
symmetric eigenproblem. Using the congruence
`X^{-1/2}(X + alpha dX)X^{-1/2} = I + alpha B` with `B = X^{-1/2} dX X^{-1/2}`,
the binding eigenvalue is the most negative one: `alpha = -1/min(λ(B))`.
"""
function boundary_step(::PSDCone, X, dX)
    T = promote_type(eltype(X), eltype(dX))
    Xis = _psd_inverse_sqrt(X)
    B = Xis * dX * Xis
    vals, _ = _jacobi_eigen(B)
    mu_min = minimum(vals)
    if mu_min >= zero(T)
        return T(Inf)
    else
        return -one(T) / mu_min
    end
end

"""
NT scaling point for the PSD cone:
`W = X^{1/2} (X^{1/2} Y X^{1/2})^{-1/2} X^{1/2}`, which satisfies `W*Y*W = X`.
"""
function nt_scaling(::PSDCone, X, Y)
    Xs = sqrt(PSDCone(), X)
    M = Xs * Y * Xs
    Mis = _psd_inverse_sqrt(M)
    return Xs * Mis * Xs
end

"""Mutating PSD boundary step; writes the step into `w[1]`."""
function boundary_step!(w, ::PSDCone, X, dX)
    step = boundary_step(PSDCone(), X, dX)
    w[1] = step
    return step
end

# convenience (allocating + mutating)
psd_jordan_product(X, Y) = jordan_product(PSDCone(), X, Y)
psd_sqrt(X) = sqrt(PSDCone(), X)
psd_inverse(X) = inverse(PSDCone(), X)
psd_boundary_step(X, dX) = boundary_step(PSDCone(), X, dX)
psd_boundary_step!(w, X, dX) = boundary_step!(w, PSDCone(), X, dX)
psd_nt_scaling(X, Y) = nt_scaling(PSDCone(), X, Y)

end # module ConeAlgebra
