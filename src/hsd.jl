#=
# Homogeneous Self-Dual embedding (Phase 6) - mathematical core.
#
# For an LP min c'x s.t. Ax=b, x>=0 with dual A'y+s=c, s>=0, the
# homogeneous self-dual embedding appends a homogenization variable tau and
# a complementarity gap variable kappa. The skew-symmetric matrix M below is
# the self-dual embedding; a solution (x, y, tau, s, kappa) of the HSD system
# yields an optimality or (primal/dual) infeasibility certificate depending
# on whether tau or kappa drives to zero.
#
# This module provides the embedding construction and the certificate
# mapping (Farkas / dual-ray certificates), which the solver can use to
# certify infeasibility and unboundedness in original coordinates. It does
# not (yet) drive the interior-point loop; that is the documented next step.
=#

"""Skew-symmetric self-dual embedding matrix for the LP.
Returns the (1+m+n)-square matrix acting on (x; y; tau) with the residual
rows ordered (primal; dual; objective-gap).  Always skew-symmetric."""
function hsd_skew_embedding(A::AbstractMatrix{T}, b::AbstractVector{T},
    c::AbstractVector{T}) where {T}
    m, n = size(A)
    N = 1 + m + n
    M = zeros(T, N, N)
    # Ordering: [tau; y (dual, m); x (primal, n)].
    # Block layout (skew-symmetric):
    #   M = [ 0     b'    -c' ]
    #       [ -b    0      A  ]
    #       [ c    -A'     0  ]
    for j in 1:n
        M[1, 1 + m + j] = -c[j]
        M[1 + m + j, 1] = c[j]
    end
    for j in 1:m
        M[1, 1 + j] = b[j]
        M[1 + j, 1] = -b[j]
        for k in 1:n
            M[1 + j, 1 + m + k] = A[j, k]
            M[1 + m + k, 1 + j] = -A[j, k]
        end
    end
    return M
end

"""Whether a matrix is skew-symmetric (within floating tolerance)."""
function is_skew_symmetric(M::AbstractMatrix{T}; tol::Real=eps(T)) where {T}
    size(M, 1) == size(M, 2) || return false
    n = size(M, 1)
    @inbounds for i in 1:n, j in 1:n
        abs(M[i, j] + M[j, i]) > tol && return false
    end
    return true
end

"""Farkas primal-infeasibility certificate: `A'y <= 0` and `b'y > 0`.
Verifies the certificate in original coordinates and returns whether it is
valid and the (positive) Farkas value."""
function primal_infeasibility_certificate(A::AbstractMatrix{T},
    b::AbstractVector{T}, y::AbstractVector{T}) where {T}
    m, n = size(A)
    length(y) == m || throw(DimensionMismatch)
    # A'y must be <= 0 on the nonnegative cone.
    dual_residual = transpose(A) * y
    all(v -> v <= zero(T) + T(1e-9) * max(one(T), abs(v)), dual_residual) ||
        return (valid=false, farkas_value=zero(T))
    value = dot(b, y)
    return (valid = value > T(1e-9) * max(one(T), abs(value)),
            farkas_value = value)
end

"""Dual-infeasibility / primal-unbounded certificate: `x >= 0`, `Ax = 0`,
`c'x < 0`."""
function dual_infeasibility_certificate(A::AbstractMatrix{T},
    c::AbstractVector{T}, x::AbstractVector{T}) where {T}
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch)
    all(v -> v >= -T(1e-9) * max(one(T), abs(v)), x) ||
        return (valid=false, objective_value=zero(T))
    norm(A * x, Inf) > T(1e-9) * max(one(T), norm(A * x, Inf)) &&
        return (valid=false, objective_value=zero(T))
    value = dot(c, x)
    return (valid = value < -T(1e-9) * max(one(T), abs(value)),
            objective_value = value)
end

"""Map an HSD solution (x, y, s, tau, kappa) to a certificate status.
Returns :optimal, :primal_infeasible, :dual_infeasible, or :undetermined."""
function hsd_status(x::AbstractVector{T}, y::AbstractVector{T},
    s::AbstractVector{T}, tau::T, kappa::T) where {T}
    if tau > T(1e-8) && kappa <= T(1e-8) * max(one(T), kappa)
        return :optimal
    elseif tau <= T(1e-8) && kappa > T(1e-8)
        return :infeasible
    end
    return :undetermined
end

"""Minimal HSD bordered-system prototype.  Builds the (1+m+n)-square
self-dual embedding and returns it with the diagonal barrier/centering term
`mu*I` added, so the bordered system is symmetric (skew part + positive
diagonal) and ready to be factorized by the main KKT path.  Pure prototype:
does not drive the interior-point loop."""
function hsd_bordered_system(A::AbstractMatrix{T}, b::AbstractVector{T},
    c::AbstractVector{T}; mu::T=one(T)) where {T}
    m, n = size(A)
    M = hsd_skew_embedding(A, b, c)
    N = size(M, 1)
    for i in 1:N
        M[i, i] += mu
    end
    return (matrix=M, m=m, n=n, dim=N)
end
