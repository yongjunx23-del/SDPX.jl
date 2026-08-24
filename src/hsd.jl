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

"""Classify an LP HSD candidate (x, y, s, tau, kappa) into a validated status
using the certificate functions.  Returns a NamedTuple with `status`
(:optimal / :primal_infeasible / :dual_infeasible / :undetermined) and the
matching certificate validity flags.  Does not solve; classifies a given
candidate in original coordinates."""
function hsd_classify(A::AbstractMatrix{T}, b::AbstractVector{T},
    c::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T},
    s::AbstractVector{T}, tau::T, kappa::T) where {T}
    m, n = size(A)
    if tau > T(1e-8)
        # Rescale to the original LP and check primal/dual feasibility.
        xp = x ./ tau
        yp = y ./ tau
        primal_res = norm(A * xp - b, Inf)
        dual_res = norm(transpose(A) * yp + s ./ tau - c, Inf)
        if primal_res < T(1e-8) && dual_res < T(1e-8) && all(v -> v >= -T(1e-8), xp)
            return (status=:optimal, valid=true, primal_residual=primal_res, dual_residual=dual_res)
        end
    end
    if tau <= T(1e-8) && kappa > T(1e-8)
        primal_cert = primal_infeasibility_certificate(A, b, y)
        dual_cert = dual_infeasibility_certificate(A, c, x)
        if primal_cert.valid
            return (status=:primal_infeasible, valid=true, primal_residual=zero(T), dual_residual=zero(T))
        elseif dual_cert.valid
            return (status=:dual_infeasible, valid=true, primal_residual=zero(T), dual_residual=zero(T))
        end
    end
    return (status=:undetermined, valid=false, primal_residual=zero(T), dual_residual=zero(T))
end

"""Standalone path-following HSD solver for a small LP (Phase-6 prototype).
Solves min c'x s.t. Ax=b, x>=0 via the homogeneous self-dual embedding
(tau, kappa) with a dense path-following Newton method.  Returns a NamedTuple
with the HSD solution (x, y, s, tau, kappa), the classified status via
`hsd_classify`, the primal objective c'x/tau, and the iteration count."""
function hsd_lp_solve(A::AbstractMatrix{T}, b::AbstractVector{T},
    c::AbstractVector{T}; iter_max::Int=300, tol::T=T(1e-10)) where {T}
    m, n = size(A)
    x = ones(T, n)
    s = ones(T, n)
    y = zeros(T, m)
    tau = one(T)
    kappa = one(T)
    iterations = 0
    for _ in 1:iter_max
        iterations += 1
        r_p = A * x - b * tau
        r_d = transpose(A) * y + s - c * tau
        r_g = -dot(c, x) + dot(b, y) - kappa
        mu = (dot(x, s) + tau * kappa) / (n + 1)
        residual = norm(r_p, Inf) + norm(r_d, Inf) + abs(r_g)
        if mu <= tol && residual < T(1e-6)
            break
        end
        sigma = T(0.1)
        dimension = 2 * n + m + 2
        M = zeros(T, dimension, dimension)
        rhs = zeros(T, dimension)
        tau_i = 2 * n + m + 1
        kap_i = 2 * n + m + 2
        # Primal rows: A dx - b dtau = -r_p.
        for j in 1:m
            for i in 1:n
                M[j, i] = A[j, i]
            end
            M[j, tau_i] = -b[j]
            rhs[j] = -r_p[j]
        end
        # Dual rows: A' dy + ds - c dtau = -r_d.
        for k in 1:n
            row = m + k
            for j in 1:m
                M[row, n + j] = A[j, k]
            end
            M[row, m + n + k] = one(T)
            M[row, tau_i] = -c[k]
            rhs[row] = -r_d[k]
        end
        # Gap row: -c' dx + b' dy - dkappa = -r_g.
        gap_row = m + n + 1
        for i in 1:n
            M[gap_row, i] = -c[i]
        end
        for j in 1:m
            M[gap_row, n + j] = b[j]
        end
        M[gap_row, kap_i] = -one(T)
        rhs[gap_row] = -r_g
        # Complementarity (x): S dx + X ds = -(X s - sigma mu e).
        for k in 1:n
            row = m + n + 1 + k
            M[row, k] = s[k]
            M[row, n + m + k] = x[k]
            rhs[row] = -(x[k] * s[k] - sigma * mu)
        end
        # Complementarity (tau kappa): kappa dtau + tau dkappa = -(tau kappa - sigma mu).
        cap_row = m + 2 * n + 2
        M[cap_row, tau_i] = kappa
        M[cap_row, kap_i] = tau
        rhs[cap_row] = -(tau * kappa - sigma * mu)
        delta = try
            M \ rhs
        catch err
            err isa LinearAlgebra.SingularException || rethrow()
            break  # degenerate system (e.g. rank-deficient A); classify current iterate
        end
        dx = delta[1:n]
        dy = delta[n+1:n+m]
        ds = delta[n+m+1:n+m+n]
        dtau = delta[tau_i]
        dkappa = delta[kap_i]
        # Step to keep all variables strictly positive.
        alpha = one(T)
        for i in 1:n
            dx[i] < zero(T) && (alpha = min(alpha, -x[i] / dx[i]))
            ds[i] < zero(T) && (alpha = min(alpha, -s[i] / ds[i]))
        end
        dtau < zero(T) && (alpha = min(alpha, -tau / dtau))
        dkappa < zero(T) && (alpha = min(alpha, -kappa / dkappa))
        alpha = min(one(T), T(0.995) * alpha)
        x .+= alpha .* dx
        s .+= alpha .* ds
        y .+= alpha .* dy
        tau += alpha * dtau
        kappa += alpha * dkappa
    end
    classified = hsd_classify(A, b, c, x, y, s, tau, kappa)
    pobj = tau > zero(T) ? dot(c, x) / tau : T(Inf)
    return (status=classified.status, valid=classified.valid, x=x, y=y, s=s,
            tau=tau, kappa=kappa, pobj=pobj, iterations=iterations)
end
