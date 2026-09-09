# src/cones/symmetric/eigen.jl
#
# Zero-allocation symmetric eigendecomposition with a hard convergence check.
#
# The PSD kernels of this module run on a preallocated `PSDTriangleScratch` and
# must never allocate Julia heap memory on warm calls. LAPACK's public
# `syevr!`/`eigen` wrappers allocate their work vectors, so we cannot use them
# for the hot path. Instead we use a *cyclic Jacobi* iteration that:
#
#   * runs entirely on caller-provided buffers (matrix `A`, eigenvector `V`,
#     eigenvalue `w`) so warm calls allocate zero Julia bytes,
#   * carries a strict convergence check: it iterates up to a generous budget
#     and THROWS `EigFailed` if the off-diagonal residual has not dropped below
#     the tolerance, so a failed diagonalization is never silently accepted,
#   * normalises every eigenvector column, guaranteeing the rank-one primitive
#     idempotents `E_k = v_k v_kᵀ` satisfy the Jordan idempotence `E∘E = E`.
#
# It is generic over `T` and therefore also serves the extended-precision
# element types: MultiFloat (`Float64x2/3/4`) uses MultiFloat arithmetic and
# BigFloat uses MPFR arithmetic through this same generic backend (the MPFR
# work is native, not Julia heap allocation).
#
# For Float64 the matrix products in the PSD kernels go through BLAS gemm; the
# only matrix-valued primitive used here is the symmetrised full matrix copied
# out of the packed lower-triangle storage (no Kronecker matrices anywhere).

"""
    _SymmetricEigenFailed

Thrown by `_jacobi_eigen!` when the cyclic-Jacobi iteration budget is exhausted
before the off-diagonal residual reaches the requested tolerance. Carries the
matrix size `n` and the iteration sweep at which it gave up.
"""
struct _SymmetricEigenFailed <: Exception
    n::Int
    sweep::Int
end
Base.showerror(io::IO, e::_SymmetricEigenFailed) =
    print(io, "SymmetricCones: eigendecomposition failed to converge " *
              "(n = $(e.n), after $(e.sweep) sweeps)")

"""
    _jacobi_eigen!(A, V, w; max_sweeps=64)

In-place cyclic-Jacobi eigendecomposition of the symmetric matrix `A`
(read/written). On entry `V` must already hold the identity; on exit `V` holds
the eigenvectors as columns and `A` has been driven to a diagonal whose diagonal
is copied into `w` (both `A` and `V` are the caller's preallocated workspace).

Convergence criterion: the sum of `|A[i,j]|` over `i>j` must be at most
`tol = eps(T)*max(1, n)*10*norm_scale` where `norm_scale` is the maximum
absolute diagonal entry of the initial `A`.  A rotation may be skipped only
below `tol / (n*(n-1)/2)`, so skipped entries cannot collectively violate the
same aggregate convergence test. The iteration throws [`_SymmetricEigenFailed`](@ref)
if it has not converged after `maxiter` full sweeps.
"""
function _jacobi_eigen!(
    A::AbstractMatrix{T},
    V::AbstractMatrix{T},
    w::AbstractVector{T};
    maxsweeps::Int = 50,
) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("matrix must be square"))
    length(w) == n || throw(DimensionMismatch("eigenvalue buffer must have length n"))
    n == 0 && (return w, V)
    z = zero(T)
    o = one(T)
    two = o + o

    # scale the convergence tolerance by the largest diagonal magnitude
    scale = z
    @inbounds for i in 1:n
        aii = abs(A[i, i])
        scale = aii > scale ? aii : scale
    end
    scale = scale > o ? scale : o
    tol = eps(T) * scale * T(max(1, n)) * T(10)
    off_count = max(1, div(n * (n - 1), 2))
    rotation_tol = tol / T(off_count)

    # --- initial sweep detection of exact diagonal/zero matrix ---
    off = z
    @inbounds for j in 2:n, i in 1:(j - 1)
        off += abs(A[i, j])
    end
    if off <= tol
        @inbounds for i in 1:n
            w[i] = A[i, i]
        end
        return w, V
    end

    for sweep in 1:maxsweeps
        @inbounds for p in 1:(n - 1)
            for q in (p + 1):n
                apq = A[p, q]
                abs(apq) <= rotation_tol && continue
                app = A[p, p]
                aqq = A[q, q]
                theta = (aqq - app) / (two * apq)
                abs_theta = abs(theta)
                # The usual `sqrt(theta^2 + 1)` formula overflows for a tiny
                # off-diagonal beside separated diagonal entries.  The
                # reciprocal branch is algebraically identical and bounded.
                t = if iszero(theta)
                    o
                elseif abs_theta <= o
                    sign(theta) / (abs_theta + sqrt(abs_theta * abs_theta + o))
                else
                    inv_theta = o / abs_theta
                    sign(theta) * inv_theta /
                    (o + sqrt(o + inv_theta * inv_theta))
                end
                c = o / sqrt(t * t + o)
                s = t * c
                for k in 1:n
                    if k != p && k != q
                        akp = A[k, p]
                        akq = A[k, q]
                        A[k, p] = c * akp - s * akq
                        A[p, k] = A[k, p]
                        A[k, q] = s * akp + c * akq
                        A[q, k] = A[k, q]
                    end
                end
                A[p, p] = app - t * apq
                A[q, q] = aqq + t * apq
                A[p, q] = z
                A[q, p] = z
                for k in 1:n
                    vkp = V[k, p]
                    vkq = V[k, q]
                    V[k, p] = c * vkp - s * vkq
                    V[k, q] = s * vkp + c * vkq
                end
            end
        end
        # re-check the full off-diagonal residual after each sweep
        off = z
        @inbounds for j in 2:n, i in 1:(j - 1)
            off += abs(A[i, j])
        end
        if off <= tol
            @inbounds for i in 1:n
                w[i] = A[i, i]
            end
            return w, V
        end
    end
    throw(_SymmetricEigenFailed(n, maxsweeps))
end

"""
    _eigen!(scratch, packed) -> (w, V)

Run the in-place Jacobi eigendecomposition on the packed-lower symmetric matrix
`packed`, writing eigenvalues into `scratch.w` and eigenvectors (columns) into
`scratch.V`, using `scratch.A` as the full working matrix. Throws
[`_SymmetricEigenFailed`](@ref) on non-convergence.
"""
function _eigen!(scratch::PSDEigenScratch{T}, packed::AbstractVector) where {T}
    n = scratch.n
    _unpack!(scratch.A, packed, n)
    _identity!(scratch.V, n)
    _jacobi_eigen!(scratch.A, scratch.V, scratch.w)
    # defensive orthonormalisation of the eigenvector columns so the primitive
    # idempotents v vᵀ are exact idempotents of the Jordan algebra.
    _orthonormalize!(scratch.V, n)
    return scratch.w, scratch.V
end

# ---------------------------------------------------------------------------
# Experimental n=2 SPD-relative eigensolver route (Float64 only).
#
# The production cyclic Jacobi uses an ABSOLUTE rotation threshold
# (eps(T)*scale*max(1,n)*10/off_count). For a matrix with a tiny off-diagonal
# beside a large diagonal, e.g. M = [1 delta; delta 2*delta^2] with
# delta = 2^-50, |M12| = 8.9e-16 falls below the absolute threshold and the
# rotation is SKIPPED even though the relative correlation
# |M12|/sqrt(M11*M22) = 1/sqrt(2) is O(1). This experimental route (dimension
# two, Float64 only, explicit selection only) replaces the gate with a
# RELATIVE one, |b|/sqrt(a*c) <= tau_off, evaluated with a range-safe
# exponent-separated comparison, and applies a bounded rotation. It exists to
# measure whether a relative-accuracy eigensolver changes the downstream PSD
# NT scaling; it is NOT a production route and no fallback is permitted.
# ---------------------------------------------------------------------------

"""
    _relative2_offdiag_gate(a, b, c, tau) -> :pass | :fail | :unresolved

Range-safe proof of `|b|/sqrt(a*c) <= tau` for strictly positive `a, c`.
Uses exponent-separated mantissa bounds so no overflow/underflow occurs at
any exponent scale. Throws on nonfinite entries or non-positive diagonals.
Returns `:unresolved` when the mantissa interval straddles `tau`.
"""
function _relative2_offdiag_gate(a::T, b::T, c::T, tau::T) where {T}
    (isfinite(a) && isfinite(b) && isfinite(c)) ||
        throw(ArgumentError("experimental_relative2: nonfinite PSD entry"))
    (a > zero(T) && c > zero(T)) ||
        throw(DomainError((a, c), "experimental_relative2 requires strictly positive diagonals"))
    ma, ea = frexp(a)
    mb, eb = frexp(b)
    mc, ec = frexp(c)
    e = eb - div(ea + ec, 2)
    rem = (ea + ec) & 1
    f = rem == 1 ? one(T) / sqrt(one(T) + one(T)) : one(T)
    if e > 0
        # rho >= mb * 2^e * f >= 0.5 * 2^e >= 1 > tau (tau ~ 20*eps)
        return :fail
    end
    twoe = exp2(T(e))
    lo = mb * twoe * f        # strict lower bound of rho
    hi = (mb + mb) * twoe * f # upper bound of rho (ratio_m in (0.5, 2))
    if hi <= tau
        return :pass
    elseif lo > tau
        return :fail
    end
    return :unresolved
end

"""
    _relative2_jacobi_eigen!(A, V, w; tau_off=10*n*eps(T))

Experimental dimension-two SPD-relative cyclic-Jacobi route. `A` is the
2x2 symmetric working matrix, `V` the (identity-initialised) eigenvector
matrix, `w` the eigenvalue buffer. Uses the relative off-diagonal gate
`|A12|/sqrt(A11*A22) <= tau_off`; when the correlation is above `tau_off` it
applies the bounded rotation of the reviewed design
(`g = max(a,c,|b|)`, `d = (c/g - a/g)/2`, `beta = b/g`,
`t = beta/(d + copysign(hypot(d,beta), d))`, equal diagonals -> `t = 1`,
`c1 = 1/sqrt(1+t^2)`, `s = t*c1`). Refuses on nonfinite/nonpositive rotated
diagonals, unresolved gates, or any non-2x2 / non-Float64 input.
"""
function _relative2_jacobi_eigen!(
    A::AbstractMatrix{T},
    V::AbstractMatrix{T},
    w::AbstractVector{T};
    tau_off::T = T(10) * T(2) * eps(T),
) where {T}
    n = size(A, 1)
    n == 2 || throw(ArgumentError("experimental_relative2 is dimension-two only"))
    T === Float64 || throw(ArgumentError("experimental_relative2 is Float64-only"))
    size(A, 2) == 2 || throw(DimensionMismatch("A must be 2x2"))
    length(w) == 2 || throw(DimensionMismatch("w must have length 2"))
    a = A[1, 1]
    b = A[1, 2]
    c = A[2, 2]
    gate = _relative2_offdiag_gate(a, b, c, tau_off)
    if gate === :pass
        w[1] = a
        w[2] = c
        return w, V
    elseif gate === :unresolved
        throw(ArgumentError("experimental_relative2: off-diagonal gate unresolved"))
    end
    g = max(a, c, abs(b))
    g > zero(T) || throw(DomainError(g, "experimental_relative2: zero diagonal"))
    d = (c / g - a / g) / 2
    beta = b / g
    t = iszero(d) ? one(T) : beta / (d + copysign(hypot(d, beta), d))
    isfinite(t) || throw(ArgumentError("experimental_relative2: nonfinite rotation"))
    c1 = one(T) / sqrt(one(T) + t * t)
    s = t * c1
    app = a - t * b
    aqq = c + t * b
    (isfinite(app) && isfinite(aqq) && app > zero(T) && aqq > zero(T)) ||
        throw(ArgumentError("experimental_relative2: nonfinite/nonpositive rotated diagonal"))
    A[1, 1] = app
    A[2, 2] = aqq
    A[1, 2] = zero(T)
    A[2, 1] = zero(T)
    v11, v12 = V[1, 1], V[1, 2]
    v21, v22 = V[2, 1], V[2, 2]
    V[1, 1] = c1 * v11 - s * v12
    V[1, 2] = s * v11 + c1 * v12
    V[2, 1] = c1 * v21 - s * v22
    V[2, 2] = s * v21 + c1 * v22
    w[1] = app
    w[2] = aqq
    return w, V
end
