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
