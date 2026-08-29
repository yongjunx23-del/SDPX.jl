# Production-shaped three-dimensional nonsymmetric primal-dual scaling.
#
# Orientation is frozen as
#
#     G      : primal -> dual,
#     Theta  : dual -> primal,
#     G * s = y,                 Theta * y = s,
#     G * stilde = ytilde,       Theta * ytilde = stilde.
#
# The default construction is the Tunçel double secant with the
# Dahl-Andersen block-BFGS axis coefficient.  A conjugate dual-Hessian route
# exists only behind an explicit policy and is repaired by a one-secant BFGS
# update before it can be consumed by the HSD complementarity equation.

@inline _ns_scaling_dot3(a, b) =
    a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

@inline function _ns_scaling_norm3(a)
    # Do not form ``a'a`` or a scale factor directly.  The vector can contain
    # finite values at either end of the floating-point range, while the
    # power-of-two scale itself (for example 2^1024 for Float64) is not
    # representable.  Carry the exponent separately and reconstruct only the
    # final norm; a nonzero norm which leaves the range is reported as Inf and
    # is rejected by the caller's terminal finite gate.
    magnitude = max(abs(a[1]), abs(a[2]), abs(a[3]))
    isfinite(magnitude) && magnitude > zero(magnitude) || return magnitude
    exponent_value = exponent(magnitude)
    ok1, a1 = _ns_scaling_checked_ldexp(a[1], -exponent_value)
    ok2, a2 = _ns_scaling_checked_ldexp(a[2], -exponent_value)
    ok3, a3 = _ns_scaling_checked_ldexp(a[3], -exponent_value)
    ok1 && ok2 && ok3 || return typeof(magnitude)(Inf)
    normalized = sqrt(
        a1 * a1 + a2 * a2 +
        a3 * a3,
    )
    ok, norm = _ns_scaling_checked_ldexp(normalized, exponent_value)
    ok ? norm : norm
end

@inline function _ns_scaling_safe_exponent_sum(a::Int, b::Int)
    if b > 0 && a > typemax(Int) - b
        return false, 0
    elseif b < 0 && a < typemin(Int) - b
        return false, 0
    end
    return true, a + b
end

@inline function _ns_scaling_safe_exponent_difference(a::Int, b::Int)
    if b < 0 && a > typemax(Int) + b
        return false, 0
    elseif b > 0 && a < typemin(Int) + b
        return false, 0
    end
    return true, a - b
end

@inline function _ns_scaling_normalize3(a)
    T = typeof(a[1])
    magnitude = max(abs(a[1]), abs(a[2]), abs(a[3]))
    isfinite(magnitude) && magnitude > zero(T) ||
        return false, 0, zero(T), zero(T), zero(T)
    exponent_value = exponent(magnitude)
    ok1, a1 = _ns_scaling_checked_ldexp(a[1], -exponent_value)
    ok2, a2 = _ns_scaling_checked_ldexp(a[2], -exponent_value)
    ok3, a3 = _ns_scaling_checked_ldexp(a[3], -exponent_value)
    ok1 && ok2 && ok3 ||
        return false, exponent_value, zero(T), zero(T), zero(T)
    return true, exponent_value, a1, a2, a3
end

@inline function _ns_scaling_scaled_absdot3(a, b)
    T = typeof(a[1])
    a_ok, a_exponent, a1, a2, a3 = _ns_scaling_normalize3(a)
    b_ok, b_exponent, b1, b2, b3 = _ns_scaling_normalize3(b)
    a_ok && b_ok || return false, T(Inf)
    exponent_ok, result_exponent = _ns_scaling_safe_exponent_sum(
        a_exponent, b_exponent,
    )
    exponent_ok || return false, T(Inf)
    normalized = abs(a1) * abs(b1) + abs(a2) * abs(b2) +
                 abs(a3) * abs(b3)
    ok, result = _ns_scaling_checked_ldexp(normalized, result_exponent)
    ok, result
end

@inline function _ns_scaling_scaled_absdot6(
    a1, a2, a3, b1, b2, b3,
)
    T = typeof(a1)
    a_magnitude = max(abs(a1), abs(a2), abs(a3))
    b_magnitude = max(abs(b1), abs(b2), abs(b3))
    isfinite(a_magnitude) && isfinite(b_magnitude) &&
        a_magnitude > zero(T) && b_magnitude > zero(T) ||
        return false, T(Inf)
    a_exponent = exponent(a_magnitude)
    b_exponent = exponent(b_magnitude)
    a1_ok, na1 = _ns_scaling_checked_ldexp(a1, -a_exponent)
    a2_ok, na2 = _ns_scaling_checked_ldexp(a2, -a_exponent)
    a3_ok, na3 = _ns_scaling_checked_ldexp(a3, -a_exponent)
    b1_ok, nb1 = _ns_scaling_checked_ldexp(b1, -b_exponent)
    b2_ok, nb2 = _ns_scaling_checked_ldexp(b2, -b_exponent)
    b3_ok, nb3 = _ns_scaling_checked_ldexp(b3, -b_exponent)
    a1_ok && a2_ok && a3_ok && b1_ok && b2_ok && b3_ok ||
        return false, T(Inf)
    exponent_ok, result_exponent = _ns_scaling_safe_exponent_sum(
        a_exponent, b_exponent,
    )
    exponent_ok || return false, T(Inf)
    normalized = abs(na1) * abs(nb1) + abs(na2) * abs(nb2) +
                 abs(na3) * abs(nb3)
    _ns_scaling_checked_ldexp(normalized, result_exponent)
end

@inline function _ns_scaling_abs_sum2(a, b)
    T = typeof(a)
    aa = abs(a)
    bb = abs(b)
    isfinite(aa) && isfinite(bb) || return false, T(Inf)
    scale = max(aa, bb)
    iszero(scale) && return true, zero(T)
    e = exponent(scale)
    ok1, n1 = _ns_scaling_checked_ldexp(aa, -e)
    ok2, n2 = _ns_scaling_checked_ldexp(bb, -e)
    ok1 && ok2 || return false, T(Inf)
    normalized = n1 + n2
    _ns_scaling_checked_ldexp(normalized, e)
end

@inline function _ns_scaling_abs_sum3(a, b, c)
    ok, ab = _ns_scaling_abs_sum2(a, b)
    ok || return false, ab
    return _ns_scaling_abs_sum2(ab, c)
end

@inline function _ns_scaling_abs_sum4(a, b, c, d)
    ok, ab = _ns_scaling_abs_sum2(a, b)
    ok || return false, ab
    ok, abc = _ns_scaling_abs_sum2(ab, c)
    ok || return false, abc
    return _ns_scaling_abs_sum2(abc, d)
end

@inline function _ns_scaling_power2_vector_scale(a)
    T = typeof(a[1])
    magnitude = max(abs(a[1]), abs(a[2]), abs(a[3]))
    isfinite(magnitude) && magnitude > zero(T) ||
        return false, 0, zero(T)
    e = exponent(magnitude)
    # The scale need not itself be representable: a finite Float64 value can
    # have exponent 1024 although `ldexp(1, 1024)` is Inf.  Keep the exact
    # integer gauge and let each transformed coordinate be range-checked.
    return true, e, zero(T)
end

@inline function _ns_scaling_checked_ldexp(value, exponent_value::Int)
    result = ldexp(value, exponent_value)
    isfinite(result) || return false, result
    # A nonzero input becoming zero is range loss, not a valid homogeneous
    # gauge.  Exact zero entries remain exact zero under the transform.
    !iszero(value) && iszero(result) && return false, result
    return true, result
end

@inline function _ns_scaling_gauged_dot3(a, b)
    T = typeof(a[1])
    a_ok, a_exponent, a1, a2, a3 = _ns_scaling_normalize3(a)
    b_ok, b_exponent, b1, b2, b3 = _ns_scaling_normalize3(b)
    a_ok && b_ok || return false, T(Inf)
    normalized = a1 * b1 + a2 * b2 + a3 * b3
    exponent_ok, result_exponent = _ns_scaling_safe_exponent_sum(
        a_exponent, b_exponent,
    )
    exponent_ok || return false, T(Inf)
    normalized_ok, result = _ns_scaling_checked_ldexp(normalized, result_exponent)
    normalized_ok || return false, result
    return true, result
end

@inline function _ns_scaling_gauged_cross!(destination, a, b)
    a_ok, a_exponent, _ = _ns_scaling_power2_vector_scale(a)
    b_ok, b_exponent, _ = _ns_scaling_power2_vector_scale(b)
    a_ok && b_ok || return false
    a1_ok, a1 = _ns_scaling_checked_ldexp(a[1], -a_exponent)
    a2_ok, a2 = _ns_scaling_checked_ldexp(a[2], -a_exponent)
    a3_ok, a3 = _ns_scaling_checked_ldexp(a[3], -a_exponent)
    b1_ok, b1 = _ns_scaling_checked_ldexp(b[1], -b_exponent)
    b2_ok, b2 = _ns_scaling_checked_ldexp(b[2], -b_exponent)
    b3_ok, b3 = _ns_scaling_checked_ldexp(b[3], -b_exponent)
    a1_ok && a2_ok && a3_ok && b1_ok && b2_ok && b3_ok || return false
    destination[1] = a2 * b3 - a3 * b2
    destination[2] = a3 * b1 - a1 * b3
    destination[3] = a1 * b2 - a2 * b1
    return _ns_scaling_finite_vector(destination)
end

@inline function _ns_scaling_absdot_work3(a, b)
    ok, work = _ns_scaling_scaled_absdot3(a, b)
    return ok ? work : typeof(a[1])(Inf)
end

@inline function _ns_scaling_relative_gate(residual, work, tolerance)
    isfinite(residual) && isfinite(work) && work >= zero(work) || return false
    if iszero(work)
        return iszero(residual)
    end
    ratio = abs(residual) / work
    return isfinite(ratio) && ratio <= tolerance
end

@inline function _ns_scaling_relative_difference(a, b, tolerance)
    T = typeof(a)
    isfinite(a) && isfinite(b) || return false
    scale = max(abs(a), abs(b))
    if iszero(scale)
        return true
    end
    e = exponent(scale)
    a_ok, na = _ns_scaling_checked_ldexp(a, -e)
    b_ok, nb = _ns_scaling_checked_ldexp(b, -e)
    a_ok && b_ok || return false
    residual = na - nb
    work = abs(na) + abs(nb)
    isfinite(residual) && isfinite(work) && work > zero(T) ||
        return iszero(residual)
    return abs(residual) <= tolerance * work
end

@inline function _ns_scaling_strict_relative_gate(value, work, tolerance)
    isfinite(value) && isfinite(work) && work > zero(work) || return false
    ratio = abs(value) / work
    return isfinite(ratio) && ratio > tolerance
end

# Invert a symmetric 2x2 Gram matrix in an independent exact exponent gauge.
# The diagonal entries are normalized separately and the off-diagonal is
# compared with their geometric mean through the balanced correlation
#
#     rho = c / sqrt(a*b),       gap = (1-rho)*(1+rho).
#
# No norm product or cancellation-sensitive determinant is formed.  The
# returned `range_failure` bit distinguishes an unrepresentable finite result
# from a genuinely degenerate Gram matrix; only the latter can be considered
# by the explicit fallback policy.
@inline function _ns_scaling_inverse2(m11, m12, m22, degeneracy)
    T = typeof(m11)
    isfinite(m11) && isfinite(m12) && isfinite(m22) ||
        return false, true, T(Inf), T(Inf), T(Inf)
    m11 > zero(T) && m22 > zero(T) ||
        return false, false, T(Inf), T(Inf), T(Inf)

    e1 = exponent(m11)
    e2 = exponent(m22)
    ok11, a = _ns_scaling_checked_ldexp(m11, -e1)
    ok22, b = _ns_scaling_checked_ldexp(m22, -e2)
    ok11 && ok22 && a > zero(T) && b > zero(T) ||
        return false, true, T(Inf), T(Inf), T(Inf)

    # `sum_e` is the exponent of a*b.  Use floor(sum_e/2) and retain one
    # exact sqrt(2) factor for odd sums, which also works for negative odd
    # exponents (e.g. 2^(-3/2) = 2^-2 * sqrt(2)).
    sum_ok, sum_e = _ns_scaling_safe_exponent_sum(e1, e2)
    sum_ok || return false, true, T(Inf), T(Inf), T(Inf)
    geometric_exponent = fld(sum_e, 2)
    odd = isodd(sum_e)
    sqrt_a = sqrt(a)
    sqrt_b = sqrt(b)
    isfinite(sqrt_a) && isfinite(sqrt_b) && sqrt_a > zero(T) &&
        sqrt_b > zero(T) || return false, true, T(Inf), T(Inf), T(Inf)
    root_product = sqrt_a * sqrt_b
    if odd
        root_product *= sqrt(T(2))
    end
    isfinite(root_product) && root_product > zero(T) ||
        return false, true, T(Inf), T(Inf), T(Inf)
    cross_ok, cross_normalized = _ns_scaling_checked_ldexp(
        m12, -geometric_exponent,
    )
    cross_ok || return false, true, T(Inf), T(Inf), T(Inf)
    rho = cross_normalized / root_product
    isfinite(rho) || return false, true, T(Inf), T(Inf), T(Inf)
    one_minus_rho = one(T) - rho
    one_plus_rho = one(T) + rho
    gap = one_minus_rho * one_plus_rho
    # The correlation form is the authority for degeneracy.  In particular,
    # do not replace it by `m11*m22-m12*m12`, whose two terms can overflow or
    # cancel before the same finite, well-conditioned answer is recoverable.
    isfinite(gap) && gap > degeneracy ||
        return false, false, T(Inf), T(Inf), T(Inf)

    i11_normalized = one(T) / (a * gap)
    i22_normalized = one(T) / (b * gap)
    i12_normalized = -rho / (root_product * gap)
    isfinite(i11_normalized) && isfinite(i12_normalized) &&
        isfinite(i22_normalized) || return false, true, T(Inf), T(Inf), T(Inf)
    i11_ok, i11 = _ns_scaling_checked_ldexp(i11_normalized, -e1)
    i22_ok, i22 = _ns_scaling_checked_ldexp(i22_normalized, -e2)
    i12_ok, i12 = _ns_scaling_checked_ldexp(
        i12_normalized, -geometric_exponent,
    )
    i11_ok && i12_ok && i22_ok ||
        return false, true, T(Inf), T(Inf), T(Inf)
    _ns_conjugate_finite3(i11, i12, i22) ||
        return false, true, T(Inf), T(Inf), T(Inf)
    return true, false, i11, i12, i22
end

@inline function _ns_scaling_finite_vector(a)
    return isfinite(a[1]) && isfinite(a[2]) && isfinite(a[3])
end

@inline function _ns_scaling_nonzero_vector(a)
    return !(iszero(a[1]) && iszero(a[2]) && iszero(a[3]))
end

@inline function _ns_scaling_finite_matrix(a)
    @inbounds for value in a
        isfinite(value) || return false
    end
    return true
end

@inline function _ns_scaling_cross!(destination, a, b)
    a1, a2, a3 = a[1], a[2], a[3]
    b1, b2, b3 = b[1], b[2], b[3]
    destination[1] = a2 * b3 - a3 * b2
    destination[2] = a3 * b1 - a1 * b3
    destination[3] = a1 * b2 - a2 * b1
    return destination
end

@inline function _ns_scaling_matvec!(destination, matrix, vector)
    v1, v2, v3 = vector[1], vector[2], vector[3]
    destination[1] = matrix[1, 1] * v1 + matrix[1, 2] * v2 + matrix[1, 3] * v3
    destination[2] = matrix[2, 1] * v1 + matrix[2, 2] * v2 + matrix[2, 3] * v3
    destination[3] = matrix[3, 1] * v1 + matrix[3, 2] * v2 + matrix[3, 3] * v3
    return destination
end

@inline function _ns_scaling_rank2!(matrix, a, b, q11, q12, q22)
    @inbounds for j in 1:3
        aj = a[j]
        bj = b[j]
        for i in 1:j
            value = q11 * a[i] * aj + q12 * (a[i] * bj + b[i] * aj) +
                    q22 * b[i] * bj
            matrix[i, j] = value
            matrix[j, i] = value
        end
    end
    return matrix
end

@inline function _ns_scaling_add_outer!(matrix, vector, coefficient)
    @inbounds for j in 1:3
        vj = vector[j]
        for i in 1:j
            value = matrix[i, j] + coefficient * vector[i] * vj
            matrix[i, j] = value
            matrix[j, i] = value
        end
    end
    return matrix
end

@inline function _ns_scaling_spd!(workspace, matrix)
    rhs = workspace.work1
    z = zero(eltype(rhs))
    rhs[1] = one(eltype(rhs))
    rhs[2] = z
    rhs[3] = z
    return _ns_conjugate_spd_solve!(
        workspace.work2, matrix, rhs, workspace.factor,
    )
end

@inline function _ns_scaling_inverse_spd!(destination, matrix, workspace)
    rhs = workspace.work1
    solution = workspace.work2
    T = eltype(rhs)
    z = zero(T)
    o = one(T)
    @inbounds for column in 1:3
        rhs[1] = column == 1 ? o : z
        rhs[2] = column == 2 ? o : z
        rhs[3] = column == 3 ? o : z
        _ns_conjugate_spd_solve!(
            solution, matrix, rhs, workspace.factor,
        ) || return false
        destination[1, column] = solution[1]
        destination[2, column] = solution[2]
        destination[3, column] = solution[3]
    end
    two = o + o
    h12 = (destination[1, 2] + destination[2, 1]) / two
    h13 = (destination[1, 3] + destination[3, 1]) / two
    h23 = (destination[2, 3] + destination[3, 2]) / two
    destination[1, 2] = h12
    destination[2, 1] = h12
    destination[1, 3] = h13
    destination[3, 1] = h13
    destination[2, 3] = h23
    destination[3, 2] = h23
    return _ns_scaling_finite_matrix(destination)
end

@inline function _ns_scaling_gamma(workspace, operations::Int)
    T = typeof(workspace.mu)
    numerator = T(operations) * eps(one(T))
    numerator < one(T) || return T(Inf)
    return numerator / (one(T) - numerator)
end

# Certify the accepted lower factor against the *stored* Theta, and retain
# E = Theta - L*L' in `work_matrix` for the inverse-column certificate.  This
# is deliberately separate from the generic SPD probe: fallback hot paths
# treat the accepted Theta and this factor as the metric authority.
@inline function _ns_scaling_theta_factor_certificate!(workspace)
    theta = workspace.theta
    factor = workspace.factor
    error_matrix = workspace.work_matrix
    gamma12 = _ns_scaling_gamma(workspace, 12)
    isfinite(gamma12) || return false, typeof(workspace.mu)(Inf)
    error = zero(workspace.mu)
    @inbounds for j in 1:3
        for i in 1:3
            reconstruction = zero(workspace.mu)
            reconstruction_work = zero(workspace.mu)
            for k in 1:min(i, j)
                product = factor[i, k] * factor[j, k]
                reconstruction += product
                reconstruction_work += abs(product)
            end
            residual = theta[i, j] - reconstruction
            work = abs(theta[i, j]) + reconstruction_work
            allowance = typeof(workspace.mu)(8) * gamma12 * work
            if !(isfinite(residual) && isfinite(work) && isfinite(allowance))
                return false, typeof(workspace.mu)(Inf)
            end
            if iszero(work)
                iszero(residual) || return false, typeof(workspace.mu)(Inf)
            else
                abs(residual) <= allowance ||
                    return false, abs(residual) / work
                error = max(error, abs(residual) / work)
            end
            error_matrix[i, j] = residual
        end
    end
    return true, error
end

@inline function _ns_scaling_factor_solve!(
    destination, factor, rhs, forward,
)
    l11 = factor[1, 1]
    l21 = factor[2, 1]
    l22 = factor[2, 2]
    l31 = factor[3, 1]
    l32 = factor[3, 2]
    l33 = factor[3, 3]
    isfinite(l11) && isfinite(l21) && isfinite(l22) &&
        isfinite(l31) && isfinite(l32) && isfinite(l33) &&
        l11 > zero(l11) && l22 > zero(l22) && l33 > zero(l33) ||
        return false

    forward[1] = rhs[1] / l11
    forward[2] = (rhs[2] - l21 * forward[1]) / l22
    forward[3] = (rhs[3] - l31 * forward[1] - l32 * forward[2]) / l33
    destination[3] = forward[3] / l33
    destination[2] = (forward[2] - l32 * destination[3]) / l22
    destination[1] = (
        forward[1] - l21 * destination[2] - l31 * destination[3]
    ) / l11
    return _ns_scaling_finite_vector(forward) &&
           _ns_scaling_finite_vector(destination)
end

# Rebuild the diagnostic explicit G from the final accepted Theta factor.
# The safe midpoint is followed by a certificate on the *symmetrized* columns;
# no pre-symmetrization solve is accepted as evidence for the returned G.
@inline function _ns_scaling_rebuild_fallback_g!(workspace)
    rhs = workspace.work1
    solution = workspace.work2
    forward = workspace.work3
    T = typeof(workspace.mu)
    z = zero(T)
    o = one(T)

    rhs[1] = o
    rhs[2] = z
    rhs[3] = z
    _ns_conjugate_spd_solve!(
        solution, workspace.theta, rhs, workspace.factor,
    ) || return NS_SCALING_FALLBACK_NOT_SPD
    factor_ok, _ = _ns_scaling_theta_factor_certificate!(workspace)
    factor_ok || return NS_SCALING_INVERSE_MISMATCH

    @inbounds for column in 1:3
        rhs[1] = column == 1 ? o : z
        rhs[2] = column == 2 ? o : z
        rhs[3] = column == 3 ? o : z
        _ns_scaling_factor_solve!(
            solution, workspace.factor, rhs, forward,
        ) || return NS_SCALING_INVERSE_MISMATCH
        workspace.g[1, column] = solution[1]
        workspace.g[2, column] = solution[2]
        workspace.g[3, column] = solution[3]
    end
    half = inv(T(2))
    h12 = half * workspace.g[1, 2] + half * workspace.g[2, 1]
    h13 = half * workspace.g[1, 3] + half * workspace.g[3, 1]
    h23 = half * workspace.g[2, 3] + half * workspace.g[3, 2]
    workspace.g[1, 2] = h12
    workspace.g[2, 1] = h12
    workspace.g[1, 3] = h13
    workspace.g[3, 1] = h13
    workspace.g[2, 3] = h23
    workspace.g[3, 2] = h23
    _ns_scaling_finite_matrix(workspace.g) ||
        return NS_SCALING_NONFINITE_RESULT

    # Theta and its certified factor are authoritative.  Near the cone
    # boundary, cond(Theta) can legitimately approach inv(eps(T)); the rounded
    # explicit inverse G can then lose a resolvable Cholesky pivot even though
    # it is a backward-stable inverse of the certified SPD Theta.  Do not make
    # a second, cancellation-sensitive Cholesky of diagnostic G authoritative.
    # `_ns_scaling_inverse_columns_certificate!` below certifies the matrix
    # actually returned, and the propagated fallback secant certifies G*s=y.
    return NS_SCALING_CONVERGED
end

# Apply a backward-error-sized correction only on the subspace orthogonal to
# the authoritative fallback secant y.  Since P*y == 0 mathematically,
#
#     Theta <- Theta + delta * (I - u*u'),  u = y / norm(y),
#
# preserves Theta*y=s while resolving a rounded final Cholesky pivot at the
# near-boundary conditioning limit.  Each candidate is rebuilt from the
# unmodified metric, and the normal factor/inverse/secant certificates remain
# mandatory.  The bounded 16-epsilon search is not a provider change and does
# not hide a genuinely indefinite fallback metric.
@inline function _ns_scaling_regularize_fallback_theta!(workspace)
    theta = workspace.theta
    original = workspace.g_bfgs
    y = workspace.dual
    T = typeof(workspace.mu)

    ok, _, y1, y2, y3 = _ns_scaling_normalize3(y)
    ok || return NS_SCALING_FALLBACK_NOT_SPD
    ynorm = sqrt(y1 * y1 + y2 * y2 + y3 * y3)
    isfinite(ynorm) && ynorm > zero(T) ||
        return NS_SCALING_FALLBACK_NOT_SPD
    u1 = y1 / ynorm
    u2 = y2 / ynorm
    u3 = y3 / ynorm
    _ns_conjugate_finite3(u1, u2, u3) ||
        return NS_SCALING_FALLBACK_NOT_SPD

    scale = zero(T)
    @inbounds for j in 1:3, i in 1:3
        value = theta[i, j]
        isfinite(value) || return NS_SCALING_NONFINITE_RESULT
        original[i, j] = value
        scale = max(scale, abs(value))
    end
    scale > zero(T) || return NS_SCALING_FALLBACK_NOT_SPD
    base_shift = eps(one(T)) * scale
    isfinite(base_shift) && base_shift > zero(T) ||
        return NS_SCALING_FALLBACK_NOT_SPD

    @inbounds for attempt in 1:16
        delta = T(attempt) * base_shift
        us = (u1, u2, u3)
        for j in 1:3
            uj = us[j]
            for i in 1:j
                projector = (i == j ? one(T) : zero(T)) - us[i] * uj
                value = original[i, j] + delta * projector
                theta[i, j] = value
                theta[j, i] = value
            end
        end
        reason = _ns_scaling_rebuild_fallback_g!(workspace)
        reason === NS_SCALING_CONVERGED && return reason
        reason === NS_SCALING_FALLBACK_NOT_SPD || return reason
    end
    return NS_SCALING_FALLBACK_NOT_SPD
end

@inline function _ns_scaling_result(
    workspace::NonsymmetricScalingWorkspace{T},
    status::NonsymmetricScalingStatus,
    reason::NonsymmetricScalingReason,
    fallback_reason::NonsymmetricScalingReason,
    conjugate_reason::NonsymmetricConjugateReason,
    secant_error::T,
    inverse_error::T,
) where {T}
    workspace.last_status = status
    workspace.last_reason = reason
    workspace.last_fallback_reason = fallback_reason
    workspace.valid = status !== NS_SCALING_FAILED
    workspace.used_fallback = status === NS_SCALING_DUAL_HESSIAN_FALLBACK
    return NonsymmetricScalingResult{T}(
        status, reason, fallback_reason, conjugate_reason, workspace.mu,
        secant_error, inverse_error,
    )
end

@inline function _ns_scaling_failure(
    workspace::NonsymmetricScalingWorkspace{T},
    reason::NonsymmetricScalingReason;
    fallback_reason::NonsymmetricScalingReason=NS_SCALING_NO_FALLBACK,
    conjugate_reason::NonsymmetricConjugateReason=NS_CONJUGATE_INVALID_PARAMETER,
) where {T}
    return _ns_scaling_result(
        workspace, NS_SCALING_FAILED, reason, fallback_reason,
        conjugate_reason, T(Inf), T(Inf),
    )
end

@inline function _ns_scaling_valid_policy(::StrictDoubleSecantScaling)
    return true
end
@inline function _ns_scaling_valid_policy(
    ::DoubleSecantWithDualHessianFallback,
)
    return true
end
@inline _ns_scaling_valid_policy(::ForcedDualHessianScaling) = true
@inline _ns_scaling_force_dual(::NonsymmetricScalingPolicy) = false
@inline _ns_scaling_force_dual(::ForcedDualHessianScaling) = true

@inline function _ns_scaling_fallback_allowed(::StrictDoubleSecantScaling)
    return false
end
@inline function _ns_scaling_fallback_allowed(
    ::DoubleSecantWithDualHessianFallback,
)
    return true
end
@inline _ns_scaling_fallback_allowed(::ForcedDualHessianScaling) = true

@inline function _ns_scaling_primal_gradient_hessian!(
    workspace, tag::NonsymmetricConjugateTag,
)
    s = workspace.primal
    scale_ok, scale_exponent, _ = _ns_scaling_power2_vector_scale(s)
    scale_ok || return false
    T = typeof(s[1])
    normalized = workspace.work3
    @inbounds for i in 1:3
        coordinate_ok, coordinate = _ns_scaling_checked_ldexp(
            s[i], -scale_exponent,
        )
        coordinate_ok || return false
        normalized[i] = coordinate
    end
    _ns_conjugate_gradient!(
        workspace.primal_gradient, tag,
        normalized[1], normalized[2], normalized[3],
    )
    _ns_conjugate_hessian!(
        workspace.primal_hessian, tag,
        normalized[1], normalized[2], normalized[3],
    )
    @inbounds for i in 1:3
        gradient_ok, gradient = _ns_scaling_checked_ldexp(
            workspace.primal_gradient[i], -scale_exponent,
        )
        gradient_ok || return false
        workspace.primal_gradient[i] = gradient
        workspace.dual_shadow[i] = -gradient
    end
    @inbounds for j in 1:3, i in 1:3
        hessian_ok, hessian = _ns_scaling_checked_ldexp(
            workspace.primal_hessian[i, j], -2 * scale_exponent,
        )
        hessian_ok || return false
        workspace.primal_hessian[i, j] = hessian
    end
    return _ns_scaling_finite_vector(workspace.dual_shadow) &&
           _ns_scaling_finite_matrix(workspace.primal_hessian)
end

@inline function _ns_scaling_inverse_error!(workspace)
    G = workspace.g
    theta = workspace.theta
    error = zero(workspace.mu)
    @inbounds for j in 1:3
        for i in 1:3
            p1 = G[i, 1] * theta[1, j]
            p2 = G[i, 2] * theta[2, j]
            p3 = G[i, 3] * theta[3, j]
            value = p1 + p2 + p3
            target = i == j ? one(value) : zero(value)
            work = abs(p1) + abs(p2) + abs(p3)
            denominator = one(value) + work + abs(target)
            error = max(error, abs(value - target) / denominator)
        end
    end
    return error
end

@inline function _ns_scaling_secant_equation_error!(
    workspace, matrix, source, target, destination,
)
    _ns_scaling_matvec!(destination, matrix, source)
    _ns_scaling_finite_vector(destination) ||
        return typeof(workspace.mu)(Inf)
    backward_error = zero(workspace.mu)
    @inbounds for i in 1:3
        residual = abs(destination[i] - target[i])
        work_ok, work = _ns_scaling_scaled_absdot6(
            matrix[i, 1], matrix[i, 2], matrix[i, 3],
            source[1], source[2], source[3],
        )
        work_ok || return typeof(workspace.mu)(Inf)
        denominator_ok, denominator = _ns_scaling_abs_sum2(
            work, abs(target[i]),
        )
        isfinite(residual) && denominator_ok && isfinite(denominator) ||
            return typeof(workspace.mu)(Inf)
        row_error = if iszero(denominator)
            iszero(residual) ? zero(workspace.mu) : typeof(workspace.mu)(Inf)
        else
            residual / denominator
        end
        backward_error = max(backward_error, row_error)
    end
    # This is a componentwise backward-error gate for one secant equation.
    # Each row uses only its own arithmetic work and target; no norm from a
    # different (possibly huge) Fenchel shadow can mask corruption.  A raw
    # forward-error gate is not meaningful for the strongly conditioned
    # metrics encountered near curved boundaries: even a backward-stable
    # matrix-vector product can lose many target-relative digits there.
    return backward_error
end

@inline function _ns_scaling_secant_error!(workspace, double_secant::Bool)
    s = workspace.primal
    y = workspace.dual
    stilde = workspace.conjugate.shadow
    ytilde = workspace.dual_shadow
    error = _ns_scaling_secant_equation_error!(
        workspace, workspace.g, s, y, workspace.work1,
    )
    error = max(error, _ns_scaling_secant_equation_error!(
        workspace, workspace.theta, y, s, workspace.work2,
    ))
    if double_secant
        error = max(error, _ns_scaling_secant_equation_error!(
            workspace, workspace.g, stilde, ytilde, workspace.work1,
        ))
        error = max(error, _ns_scaling_secant_equation_error!(
            workspace, workspace.theta, ytilde, stilde, workspace.work2,
        ))
    end
    return error
end

# Certify every column of the *final symmetrized* G against the accepted
# Theta factor.  If E = Theta - L*L', then
#
#   e_j - Theta*g_j = (e_j - L*L'*g_j) - E*g_j.
#
# The first term is bounded by the fixed 3x3 triangular-solve work and the
# second is evaluated explicitly.  This remains a backward certificate when
# the inverse is strongly conditioned and an output-relative G*s test does
# not.
@inline function _ns_scaling_inverse_columns_certificate!(workspace)
    theta = workspace.theta
    factor = workspace.factor
    inverse = workspace.g
    error_matrix = workspace.work_matrix
    transpose_work = workspace.work3
    gamma24 = _ns_scaling_gamma(workspace, 24)
    isfinite(gamma24) || return false, typeof(workspace.mu)(Inf)
    T = typeof(workspace.mu)
    error = zero(T)
    @inbounds for column in 1:3
        g1 = inverse[1, column]
        g2 = inverse[2, column]
        g3 = inverse[3, column]
        transpose_work[1] = abs(factor[1, 1]) * abs(g1) +
                            abs(factor[2, 1]) * abs(g2) +
                            abs(factor[3, 1]) * abs(g3)
        transpose_work[2] = abs(factor[2, 2]) * abs(g2) +
                            abs(factor[3, 2]) * abs(g3)
        transpose_work[3] = abs(factor[3, 3]) * abs(g3)
        for i in 1:3
            target = i == column ? one(T) : zero(T)
            action = theta[i, 1] * g1 + theta[i, 2] * g2 +
                     theta[i, 3] * g3
            residual = abs(target - action)
            theta_work = abs(theta[i, 1]) * abs(g1) +
                         abs(theta[i, 2]) * abs(g2) +
                         abs(theta[i, 3]) * abs(g3)
            error_work = abs(error_matrix[i, 1]) * abs(g1) +
                         abs(error_matrix[i, 2]) * abs(g2) +
                         abs(error_matrix[i, 3]) * abs(g3)
            factor_work = zero(T)
            for k in 1:i
                factor_work += abs(factor[i, k]) * transpose_work[k]
            end
            arithmetic_work = abs(target) + theta_work + factor_work
            allowance = error_work + T(8) * gamma24 * arithmetic_work
            if !(isfinite(residual) && isfinite(allowance))
                return false, T(Inf)
            end
            if iszero(allowance)
                iszero(residual) || return false, T(Inf)
            else
                residual <= allowance || return false, residual / allowance
            end
            denominator = abs(target) + theta_work
            if iszero(denominator)
                iszero(residual) || return false, T(Inf)
            else
                error = max(error, residual / denominator)
            end
        end
    end
    return true, error
end

# The one-secant fallback has only the authoritative Theta secant.  Validate
# the diagnostic G*s relation as a consequence of that secant and the
# certified inverse columns:
#
#   G*s - y = G*(s - Theta*y) + (G*Theta - I)*y.
#
# The allowance below is componentwise and homogeneous.  It has no absolute
# floor and therefore does not reject a backward-stable inverse merely
# because the true output is formed by severe cancellation.
@inline function _ns_scaling_fallback_secant_certificate!(workspace)
    T = typeof(workspace.mu)
    theta = workspace.theta
    inverse = workspace.g
    s = workspace.primal
    y = workspace.dual
    rtheta = workspace.work1
    rg = workspace.work2
    inverse_residual = workspace.work_matrix
    gamma24 = _ns_scaling_gamma(workspace, 24)
    isfinite(gamma24) || return false, T(Inf)

    theta_error = _ns_scaling_secant_equation_error!(
        workspace, theta, y, s, rtheta,
    )
    isfinite(theta_error) || return false, T(Inf)
    @inbounds for i in 1:3
        rtheta[i] -= s[i]
    end

    @inbounds for j in 1:3
        for i in 1:3
            action = inverse[i, 1] * theta[1, j] +
                     inverse[i, 2] * theta[2, j] +
                     inverse[i, 3] * theta[3, j]
            target = i == j ? one(T) : zero(T)
            inverse_residual[i, j] = action - target
        end
    end
    _ns_scaling_matvec!(rg, inverse, s)

    consequence_error = zero(T)
    @inbounds for i in 1:3
        residual = abs(rg[i] - y[i])
        propagated = zero(T)
        direct_work = abs(y[i])
        nested_theta_work = zero(T)
        inverse_theta_work = zero(T)
        for k in 1:3
            gik = abs(inverse[i, k])
            propagated += gik * abs(rtheta[k]) +
                          abs(inverse_residual[i, k]) * abs(y[k])
            direct_work += gik * abs(s[k])

            theta_y_row_work = abs(s[k])
            for j in 1:3
                theta_y_row_work += abs(theta[k, j]) * abs(y[j])
                inverse_theta_work +=
                    gik * abs(theta[k, j]) * abs(y[j])
            end
            nested_theta_work += gik * theta_y_row_work
        end
        inverse_theta_work += abs(y[i])
        work = direct_work + nested_theta_work + inverse_theta_work
        allowance = propagated + T(8) * gamma24 * work
        if !(isfinite(residual) && isfinite(propagated) &&
             isfinite(work) && isfinite(allowance))
            return false, T(Inf)
        end
        if iszero(allowance)
            iszero(residual) || return false, T(Inf)
        else
            residual <= allowance || return false, residual / allowance
        end
        excess = max(zero(T), residual - propagated)
        if iszero(work)
            iszero(excess) || return false, T(Inf)
        else
            consequence_error = max(consequence_error, excess / work)
        end
    end
    return true, max(theta_error, consequence_error)
end

@inline function _ns_scaling_validate_fallback_metric!(workspace)
    T = typeof(workspace.mu)
    _ns_scaling_finite_matrix(workspace.g) &&
        _ns_scaling_finite_matrix(workspace.theta) &&
        _ns_scaling_finite_matrix(workspace.factor) ||
        return NS_SCALING_NONFINITE_RESULT, T(Inf), T(Inf)

    factor_ok, factor_error =
        _ns_scaling_theta_factor_certificate!(workspace)
    factor_ok || return NS_SCALING_INVERSE_MISMATCH, T(Inf), factor_error

    # Do not require a standalone Cholesky factorization of diagnostic G here.
    # At the near-boundary conditioning limit that test can fail solely from a
    # rounded final pivot.  The accepted Theta factor plus the factor-aware
    # inverse-column certificate are the backward-error authority.
    inverse_ok, inverse_error =
        _ns_scaling_inverse_columns_certificate!(workspace)
    inverse_error = max(inverse_error, factor_error)
    inverse_ok ||
        return NS_SCALING_INVERSE_MISMATCH, T(Inf), inverse_error

    secant_ok, secant_error =
        _ns_scaling_fallback_secant_certificate!(workspace)
    secant_ok ||
        return NS_SCALING_SECANT_MISMATCH, secant_error, inverse_error
    isfinite(secant_error) &&
        secant_error <= workspace.settings.validation_tolerance ||
        return NS_SCALING_SECANT_MISMATCH, secant_error, inverse_error
    return NS_SCALING_CONVERGED, secant_error, inverse_error
end

@inline function _ns_scaling_validate_metric!(workspace, double_secant::Bool)
    _ns_scaling_finite_matrix(workspace.g) &&
        _ns_scaling_finite_matrix(workspace.theta) ||
        return NS_SCALING_NONFINITE_RESULT, typeof(workspace.mu)(Inf),
               typeof(workspace.mu)(Inf)
    _ns_scaling_spd!(workspace, workspace.g) &&
        _ns_scaling_spd!(workspace, workspace.theta) ||
        return NS_SCALING_METRIC_NOT_SPD, typeof(workspace.mu)(Inf),
               typeof(workspace.mu)(Inf)

    secant_error = _ns_scaling_secant_error!(workspace, double_secant)
    tolerance = workspace.settings.validation_tolerance
    isfinite(secant_error) && secant_error <= tolerance ||
        return NS_SCALING_SECANT_MISMATCH, secant_error,
               _ns_scaling_inverse_error!(workspace)

    # The runtime consumes `factor` as the authoritative lower factor of
    # Theta.  A small forward product error in `G*Theta` is not sufficient to
    # certify the inverse when the double-secant G is strongly conditioned:
    # the exact same metric can pass that weak test while its factor-aware
    # inverse columns fail the runtime preflight.  Make the production
    # double-secant provider publish only a metric that the runtime can
    # independently replay.  The explicit fallback policy may handle this
    # typed inverse mismatch below; strict policy remains fail-closed.
    factor_ok, factor_error =
        _ns_scaling_theta_factor_certificate!(workspace)
    factor_ok || return NS_SCALING_INVERSE_MISMATCH,
        typeof(workspace.mu)(Inf), factor_error
    inverse_ok, inverse_error =
        _ns_scaling_inverse_columns_certificate!(workspace)
    inverse_error = max(inverse_error, factor_error)
    inverse_ok || return NS_SCALING_INVERSE_MISMATCH,
        typeof(workspace.mu)(Inf), inverse_error

    isfinite(inverse_error) && inverse_error <= tolerance ||
        return NS_SCALING_INVERSE_MISMATCH, secant_error, inverse_error
    return NS_SCALING_CONVERGED, secant_error, inverse_error
end

# Return the terminal reason for the strict double-secant attempt.  On
# success, G and Theta are fully populated and independently validated.
@inline function _ns_scaling_double_secant!(
    workspace::NonsymmetricScalingWorkspace{T},
) where {T}
    s = workspace.primal
    y = workspace.dual
    stilde = workspace.conjugate.shadow
    ytilde = workspace.dual_shadow
    tolerance = workspace.settings.validation_tolerance
    degeneracy = workspace.settings.degeneracy_tolerance
    degree = T(3)

    m11_ok, m11 = _ns_scaling_gauged_dot3(y, s)
    m12_ok, m12 = _ns_scaling_gauged_dot3(y, stilde)
    m21_ok, m21 = _ns_scaling_gauged_dot3(ytilde, s)
    m22_ok, m22 = _ns_scaling_gauged_dot3(ytilde, stilde)
    _ns_scaling_finite_vector(s) && _ns_scaling_finite_vector(y) &&
        _ns_scaling_finite_vector(stilde) &&
        _ns_scaling_finite_vector(ytilde) &&
        m11_ok && m12_ok && m21_ok && m22_ok &&
        isfinite(m11) && isfinite(m12) && isfinite(m21) && isfinite(m22) ||
        return NS_SCALING_NONFINITE_RESULT
    _ns_scaling_relative_difference(m12, m21, tolerance) ||
        return NS_SCALING_GRAM_NONSYMMETRIC
    _ns_scaling_relative_difference(m12, degree, tolerance) &&
        _ns_scaling_relative_difference(m21, degree, tolerance) ||
        return NS_SCALING_SHADOW_IDENTITY_FAILED
    half = inv(T(2))
    mcross = half * m12 + half * m21
    isfinite(mcross) || return NS_SCALING_NONFINITE_RESULT
    inverse_ok, inverse_range_failure, i11, i12, i22 = _ns_scaling_inverse2(
        m11, mcross, m22, degeneracy,
    )
    inverse_ok || return inverse_range_failure ?
        NS_SCALING_NONFINITE_RESULT : NS_SCALING_SECOND_SECANT_DEGENERATE

    _ns_scaling_gauged_cross!(workspace.axis_z, s, stilde) ||
        return NS_SCALING_NONFINITE_RESULT
    znorm = _ns_scaling_norm3(workspace.axis_z)
    # Work in the normalized cross gauge.  The terms that cancel in the cross
    # product are exactly what expose an almost-parallel axis; retaining the
    # normalized pair-products avoids forming a^2 or b^2 on reciprocal orbits.
    zwork = zero(T)
    s_ok, s_exponent, _ = _ns_scaling_power2_vector_scale(s)
    st_ok, st_exponent, _ = _ns_scaling_power2_vector_scale(stilde)
    s_ok && st_ok || return NS_SCALING_NONFINITE_RESULT
    @inbounds for i in 1:3, j in 1:3
        si_ok, si = _ns_scaling_checked_ldexp(s[i], -s_exponent)
        stj_ok, stj = _ns_scaling_checked_ldexp(stilde[j], -st_exponent)
        si_ok && stj_ok || return NS_SCALING_NONFINITE_RESULT
        zwork += abs(si) * abs(stj)
    end
    isfinite(znorm) && isfinite(zwork) ||
        return NS_SCALING_NONFINITE_RESULT
    zwork > zero(T) ||
        return NS_SCALING_AXIS_DEGENERATE
    # Use the actual cross-product work for the cancellation-sensitive test.
    _ns_scaling_strict_relative_gate(znorm, zwork, degeneracy) ||
        return NS_SCALING_AXIS_DEGENERATE
    @inbounds for i in 1:3
        workspace.axis_z[i] /= znorm
    end
    _ns_scaling_gauged_cross!(workspace.axis_r, y, ytilde) ||
        return NS_SCALING_NONFINITE_RESULT
    axis_pairing = _ns_scaling_dot3(workspace.axis_z, workspace.axis_r)
    # A large component orthogonal to z still consumes arithmetic work in the
    # cross-product and can make z'axis_r unresolved.  Use the componentwise
    # cross-vector work, not only the three products retained by the dot.
    rwork_ok, rwork = _ns_scaling_abs_sum3(
        workspace.axis_r[1], workspace.axis_r[2], workspace.axis_r[3],
    )
    isfinite(axis_pairing) && rwork_ok && isfinite(rwork) ||
        return NS_SCALING_NONFINITE_RESULT

    _ns_scaling_strict_relative_gate(axis_pairing, rwork, degeneracy) ||
        return NS_SCALING_AXIS_PAIRING_DEGENERATE
    @inbounds for i in 1:3
        workspace.axis_r[i] /= axis_pairing
    end

    _ns_scaling_rank2!(workspace.g, y, ytilde, i11, i12, i22)
    _ns_scaling_rank2!(workspace.theta, s, stilde, i11, i12, i22)
    _ns_scaling_finite_matrix(workspace.g) &&
        _ns_scaling_finite_matrix(workspace.theta) ||
        return NS_SCALING_NONFINITE_RESULT

    H = workspace.primal_hessian
    G0 = workspace.g0
    # Form the BFGS reference in an exact power-of-two gauge.  The primal
    # Hessian scales as s_scale^-2 and the metric scales as y_scale/s_scale.
    # Keep only the integer exponents: constructing s_scale*y_scale or a
    # floating norm ratio would itself overflow on a perfectly valid common
    # orbit (and would underflow on its reciprocal).
    s_scale_ok, s_exponent, _ = _ns_scaling_power2_vector_scale(s)
    y_scale_ok, y_exponent, _ = _ns_scaling_power2_vector_scale(y)
    s_scale_ok && y_scale_ok || return NS_SCALING_NONFINITE_RESULT
    exponent_sum = s_exponent + y_exponent
    normalized_mu_ok, normalized_mu = _ns_scaling_checked_ldexp(
        workspace.mu, -exponent_sum,
    )
    normalized_mu_ok || return NS_SCALING_NONFINITE_RESULT
    metric_exponent = y_exponent - s_exponent
    metric_ratio_ok, metric_ratio = _ns_scaling_checked_ldexp(
        one(T), metric_exponent,
    )
    metric_ratio_ok && metric_ratio > zero(T) ||
        return NS_SCALING_NONFINITE_RESULT
    @inbounds for j in 1:3, i in 1:3
        h1_ok, h1 = _ns_scaling_checked_ldexp(H[i, j], s_exponent)
        h1_ok || return NS_SCALING_NONFINITE_RESULT
        h2_ok, normalized_hessian = _ns_scaling_checked_ldexp(
            h1, s_exponent,
        )
        h2_ok || return NS_SCALING_NONFINITE_RESULT
        g0_ok, g0_value = _ns_scaling_checked_ldexp(
            normalized_mu * normalized_hessian, metric_exponent,
        )
        g0_ok || return NS_SCALING_NONFINITE_RESULT
        G0[i, j] = g0_value
    end
    _ns_scaling_finite_matrix(G0) || return NS_SCALING_NONFINITE_RESULT
    _ns_scaling_matvec!(workspace.work1, G0, s)
    _ns_scaling_matvec!(workspace.work2, G0, stilde)
    _ns_scaling_finite_vector(workspace.work1) &&
        _ns_scaling_finite_vector(workspace.work2) ||
        return NS_SCALING_NONFINITE_RESULT
    # A reference metric whose action on the secant plane vanishes is a
    # degenerate BFGS denominator, not a non-finite computation.  The gauged
    # dot products below cannot normalize an exactly zero vector and would
    # report `NONFINITE_RESULT`, which is not fallback-eligible -- so an
    # entirely benign degenerate reference would become fail-closed instead
    # of routing to the explicit dual-Hessian fallback.
    _ns_scaling_nonzero_vector(workspace.work1) &&
        _ns_scaling_nonzero_vector(workspace.work2) ||
        return NS_SCALING_BFGS_DENOMINATOR
    k11_ok, k11 = _ns_scaling_gauged_dot3(s, workspace.work1)
    k12_ok, k12 = _ns_scaling_gauged_dot3(s, workspace.work2)
    k21_ok, k21 = _ns_scaling_gauged_dot3(stilde, workspace.work1)
    k22_ok, k22 = _ns_scaling_gauged_dot3(stilde, workspace.work2)
    k11_ok && k12_ok && k21_ok && k22_ok &&
        isfinite(k11) && isfinite(k12) && isfinite(k21) && isfinite(k22) ||
        return NS_SCALING_NONFINITE_RESULT
    _ns_scaling_relative_difference(k12, k21, tolerance) ||
        return NS_SCALING_BFGS_DENOMINATOR
    kcross = half * k12 + half * k21
    isfinite(kcross) || return NS_SCALING_NONFINITE_RESULT
    inverse_ok, inverse_range_failure, q11, q12, q22 = _ns_scaling_inverse2(
        k11, kcross, k22, degeneracy,
    )
    inverse_ok || return inverse_range_failure ?
        NS_SCALING_NONFINITE_RESULT : NS_SCALING_BFGS_DENOMINATOR
    _ns_scaling_rank2!(
        workspace.work_matrix, workspace.work1, workspace.work2,
        q11, q12, q22,
    )
    @inbounds for j in 1:3, i in 1:3
        workspace.g_bfgs[i, j] = workspace.g[i, j] + G0[i, j] -
                                  workspace.work_matrix[i, j]
    end
    _ns_scaling_spd!(workspace, workspace.g_bfgs) ||
        return NS_SCALING_BFGS_NOT_SPD

    _ns_scaling_matvec!(workspace.work1, workspace.g_bfgs, workspace.axis_z)
    tG = _ns_scaling_dot3(workspace.axis_z, workspace.work1)
    tgwork_ok, tgwork = _ns_scaling_scaled_absdot3(
        workspace.axis_z, workspace.work1,
    )
    if !tgwork_ok || !isfinite(tG) || !isfinite(tgwork)
        return NS_SCALING_NONFINITE_RESULT
    end
    tG > zero(T) &&
        _ns_scaling_strict_relative_gate(tG, tgwork, degeneracy) ||
        return NS_SCALING_AXIS_COEFFICIENT
    _ns_scaling_add_outer!(workspace.g, workspace.axis_z, tG)
    _ns_scaling_add_outer!(workspace.theta, workspace.axis_r, inv(tG))
    _ns_scaling_finite_matrix(workspace.g) &&
        _ns_scaling_finite_matrix(workspace.theta) ||
        return NS_SCALING_NONFINITE_RESULT

    reason, _, _ = _ns_scaling_validate_metric!(workspace, true)
    return reason
end

@inline function _ns_scaling_dual_hessian_fallback!(
    workspace::NonsymmetricScalingWorkspace{T},
) where {T}
    theta0 = workspace.work_matrix
    workspace.conjugate.inverse_valid ||
        return NS_SCALING_FALLBACK_NOT_SPD
    hstar = workspace.conjugate.inverse_hessian
    @inbounds for j in 1:3, i in 1:3
        theta0[i, j] = workspace.mu * hstar[i, j]
    end
    _ns_scaling_finite_matrix(theta0) || return NS_SCALING_NONFINITE_RESULT
    # `inverse_valid` is the factor-aware certificate for Hstar.  Multiplying
    # that SPD operator by positive mu preserves SPD mathematically, while a
    # second Cholesky of the explicitly scaled, near-boundary matrix can lose
    # its final pivot solely from rounding.  The repaired final Theta is still
    # factored and certified below; theta0 itself is never consumed by runtime.

    s = workspace.primal
    y = workspace.dual
    _ns_scaling_matvec!(workspace.work1, theta0, y)
    denominator_ok, denominator = _ns_scaling_gauged_dot3(y, workspace.work1)
    pairing_ok, pairing = _ns_scaling_gauged_dot3(s, y)
    denominator_work = _ns_scaling_absdot_work3(y, workspace.work1)
    pairing_work = _ns_scaling_absdot_work3(s, y)
    tolerance = workspace.settings.degeneracy_tolerance
    denominator_ok && pairing_ok && isfinite(denominator) && isfinite(pairing) &&
        isfinite(denominator_work) && isfinite(pairing_work) ||
        return NS_SCALING_NONFINITE_RESULT
    denominator > zero(T) && pairing > zero(T) &&
        _ns_scaling_strict_relative_gate(
            denominator, denominator_work, tolerance,
        ) &&
        _ns_scaling_strict_relative_gate(pairing, pairing_work, tolerance) ||
        return NS_SCALING_FALLBACK_DENOMINATOR

    @inbounds for j in 1:3
        for i in 1:j
            value = theta0[i, j] -
                    workspace.work1[i] * workspace.work1[j] / denominator +
                    s[i] * s[j] / pairing
            workspace.theta[i, j] = value
            workspace.theta[j, i] = value
        end
    end
    _ns_scaling_finite_matrix(workspace.theta) ||
        return NS_SCALING_NONFINITE_RESULT

    rebuild_reason = _ns_scaling_rebuild_fallback_g!(workspace)
    if rebuild_reason === NS_SCALING_FALLBACK_NOT_SPD
        rebuild_reason = _ns_scaling_regularize_fallback_theta!(workspace)
    end
    rebuild_reason === NS_SCALING_CONVERGED || return rebuild_reason
    reason, _, _ = _ns_scaling_validate_fallback_metric!(workspace)
    reason === NS_SCALING_SECANT_MISMATCH &&
        return NS_SCALING_FALLBACK_SECANT_MISMATCH
    return reason
end

@inline function _ns_scaling_fallback_eligible(reason)
    return reason === NS_SCALING_GRAM_NONSYMMETRIC ||
           reason === NS_SCALING_SECOND_SECANT_DEGENERATE ||
           reason === NS_SCALING_AXIS_DEGENERATE ||
           reason === NS_SCALING_AXIS_PAIRING_DEGENERATE ||
           reason === NS_SCALING_BFGS_DENOMINATOR ||
           reason === NS_SCALING_BFGS_NOT_SPD ||
           reason === NS_SCALING_AXIS_COEFFICIENT ||
           reason === NS_SCALING_METRIC_NOT_SPD ||
           reason === NS_SCALING_SECANT_MISMATCH ||
           reason === NS_SCALING_INVERSE_MISMATCH ||
           reason === NS_SCALING_FORCED_DUAL_HESSIAN
end

"""
    try_update_nonsymmetric_scaling!(workspace, policy, tag, s, y)
    try_update_nonsymmetric_scaling!(workspace, policy, tag, s, y, mu)

Update the local degree-three primal-dual metric.  Strict policy never changes
provider.  The fallback policy may use the Fenchel conjugate Hessian only
after recording why the double-secant route was rejected; the fallback is
then repaired to satisfy `Theta*y=s` and validated as a finite SPD inverse
pair.  The five-argument method retains the standalone block convention
`mu=dot(s,y)/3`.  Product-HSD production code must use the six-argument
method so every block BFGS reference metric uses the embedding's single
global complementarity observable.  Expected numerical failures return
`NS_SCALING_FAILED`.
"""
function _try_update_nonsymmetric_scaling!(
    workspace::NonsymmetricScalingWorkspace{T},
    policy::NonsymmetricScalingPolicy,
    tag::NonsymmetricConjugateTag,
    primal,
    dual,
    target_mu,
) where {T<:AbstractFloat}
    workspace.valid = false
    workspace.used_fallback = false
    workspace.mu = zero(T)
    length(primal) == 3 && length(dual) == 3 ||
        return _ns_scaling_failure(workspace, NS_SCALING_INVALID_PARAMETER)
    _ns_scaling_valid_policy(policy) ||
        return _ns_scaling_failure(workspace, NS_SCALING_INVALID_PARAMETER)
    settings = workspace.settings
    isfinite(settings.validation_tolerance) &&
        settings.validation_tolerance > zero(T) &&
        isfinite(settings.degeneracy_tolerance) &&
        settings.degeneracy_tolerance > zero(T) ||
        return _ns_scaling_failure(workspace, NS_SCALING_INVALID_PARAMETER)

    @inbounds for i in 1:3
        workspace.primal[i] = convert(T, primal[i])
        workspace.dual[i] = convert(T, dual[i])
    end
    _ns_scaling_finite_vector(workspace.primal) &&
        _ns_scaling_finite_vector(workspace.dual) ||
        return _ns_scaling_failure(workspace, NS_SCALING_NONFINITE_INPUT)
    _ns_conjugate_valid_tag(tag) ||
        return _ns_scaling_failure(workspace, NS_SCALING_INVALID_PARAMETER)
    _ns_conjugate_primal_interior(
        tag, workspace.primal[1], workspace.primal[2], workspace.primal[3],
    ) || return _ns_scaling_failure(
        workspace, NS_SCALING_PRIMAL_NOT_INTERIOR,
    )
    _ns_conjugate_dual_interior(
        tag, workspace.dual[1], workspace.dual[2], workspace.dual[3],
    ) || return _ns_scaling_failure(
        workspace, NS_SCALING_DUAL_NOT_INTERIOR,
    )
    pairing_ok, pairing = _ns_scaling_gauged_dot3(
        workspace.primal, workspace.dual,
    )
    pairing_work = _ns_scaling_absdot_work3(
        workspace.primal, workspace.dual,
    )
    pairing_ok && isfinite(pairing) && isfinite(pairing_work) ||
        return _ns_scaling_failure(workspace, NS_SCALING_NONFINITE_RESULT)
    pairing_work > zero(T) ||
        return _ns_scaling_failure(workspace, NS_SCALING_NONFINITE_RESULT)
    pairing > zero(T) &&
        _ns_scaling_strict_relative_gate(
            pairing, pairing_work, settings.degeneracy_tolerance,
        ) ||
        return _ns_scaling_failure(
            workspace, NS_SCALING_NONPOSITIVE_PAIRING,
        )
    if target_mu === nothing
        workspace.mu = pairing / T(3)
    else
        isfinite(target_mu) && target_mu > zero(T) ||
            return _ns_scaling_failure(
                workspace, NS_SCALING_INVALID_PARAMETER,
            )
        workspace.mu = target_mu
    end

    # The strict double-secant construction needs only the certified Fenchel
    # shadow and its primal Hessian.  Do not make that provider depend on the
    # much more ill-conditioned inverse Hessian used exclusively by the
    # explicitly permitted fallback.
    conjugate_result = _ns_conjugate_shadow_hessian_candidate!(
        workspace.conjugate, tag, workspace.dual,
    )
    conjugate_result.status === NS_CONJUGATE_SUCCESS ||
        return _ns_scaling_failure(
            workspace, NS_SCALING_CONJUGATE_FAILED;
            conjugate_reason=conjugate_result.reason,
        )
    barrier_ok = try
        _ns_scaling_primal_gradient_hessian!(workspace, tag)
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false
    end
    if !(barrier_ok && _ns_scaling_finite_vector(workspace.dual_shadow) &&
         _ns_scaling_finite_matrix(workspace.primal_hessian))
        _ns_conjugate_restore_accepted!(workspace.conjugate)
        return _ns_scaling_failure(workspace, NS_SCALING_NONFINITE_RESULT)
    end

    primary_reason = _ns_scaling_force_dual(policy) ?
        NS_SCALING_FORCED_DUAL_HESSIAN :
        _ns_scaling_double_secant!(workspace)
    if primary_reason === NS_SCALING_CONVERGED
        terminal_reason, secant_error, inverse_error =
            _ns_scaling_validate_metric!(workspace, true)
        terminal_reason === NS_SCALING_CONVERGED || begin
            # This is the final provider boundary.  Never publish a
            # successful double-secant metric while discarding a second
            # factor/inverse certificate failure.
            _ns_conjugate_restore_accepted!(workspace.conjugate)
            return _ns_scaling_failure(
                workspace, terminal_reason;
                conjugate_reason=conjugate_result.reason,
            )
        end
        accepted = _ns_conjugate_accept!(
            workspace.conjugate,
            workspace.dual[1], workspace.dual[2], workspace.dual[3], false,
        )
        accepted || begin
            _ns_conjugate_restore_accepted!(workspace.conjugate)
            return _ns_scaling_failure(
                workspace, NS_SCALING_CONJUGATE_FAILED;
                conjugate_reason=conjugate_result.reason,
            )
        end
        return _ns_scaling_result(
            workspace, NS_SCALING_DOUBLE_SECANT, NS_SCALING_CONVERGED,
            NS_SCALING_NO_FALLBACK, conjugate_result.reason,
            secant_error, inverse_error,
        )
    end
    if !_ns_scaling_fallback_allowed(policy) ||
       !_ns_scaling_fallback_eligible(primary_reason)
        _ns_conjugate_restore_accepted!(workspace.conjugate)
        return _ns_scaling_failure(
            workspace, primary_reason;
            conjugate_reason=conjugate_result.reason,
        )
    end

    if !_ns_conjugate_ensure_inverse_hessian!(workspace.conjugate)
        _ns_conjugate_restore_accepted!(workspace.conjugate)
        return _ns_scaling_failure(
            workspace, NS_SCALING_CONJUGATE_FAILED;
            fallback_reason=primary_reason,
            conjugate_reason=NS_CONJUGATE_INVERSE_HESSIAN_FAILED,
        )
    end
    fallback_terminal = _ns_scaling_dual_hessian_fallback!(workspace)
    if fallback_terminal !== NS_SCALING_CONVERGED
        _ns_conjugate_restore_accepted!(workspace.conjugate)
        return _ns_scaling_failure(
            workspace, fallback_terminal;
            fallback_reason=primary_reason,
            conjugate_reason=conjugate_result.reason,
        )
    end
    terminal_reason, secant_error, inverse_error =
        _ns_scaling_validate_fallback_metric!(workspace)
    terminal_reason === NS_SCALING_CONVERGED || begin
        # The fallback's final validation is authoritative as well.  Restore
        # the prior accepted conjugate state before returning a typed failure
        # so a caller cannot consume a partially published candidate.
        _ns_conjugate_restore_accepted!(workspace.conjugate)
        return _ns_scaling_failure(
            workspace, terminal_reason;
            fallback_reason=primary_reason,
            conjugate_reason=conjugate_result.reason,
        )
    end
    accepted = _ns_conjugate_accept!(
        workspace.conjugate,
        workspace.dual[1], workspace.dual[2], workspace.dual[3], true,
    )
    accepted || begin
        _ns_conjugate_restore_accepted!(workspace.conjugate)
        return _ns_scaling_failure(
            workspace, NS_SCALING_CONJUGATE_FAILED;
            fallback_reason=primary_reason,
            conjugate_reason=conjugate_result.reason,
        )
    end
    return _ns_scaling_result(
        workspace, NS_SCALING_DUAL_HESSIAN_FALLBACK,
        NS_SCALING_CONVERGED, primary_reason, conjugate_result.reason,
        secant_error, inverse_error,
    )
end

function try_update_nonsymmetric_scaling!(
    workspace::NonsymmetricScalingWorkspace{T},
    policy::NonsymmetricScalingPolicy,
    tag::NonsymmetricConjugateTag,
    primal,
    dual,
) where {T<:AbstractFloat}
    return _try_update_nonsymmetric_scaling!(
        workspace, policy, tag, primal, dual, nothing,
    )
end

function try_update_nonsymmetric_scaling!(
    workspace::NonsymmetricScalingWorkspace{T},
    policy::NonsymmetricScalingPolicy,
    tag::NonsymmetricConjugateTag,
    primal,
    dual,
    mu,
) where {T<:AbstractFloat}
    muT = try
        mu isa T ? mu : convert(T, mu)
    catch exception
        exception isa InexactError || exception isa MethodError ||
            exception isa OverflowError || rethrow(exception)
        workspace.valid = false
        workspace.used_fallback = false
        workspace.mu = zero(T)
        return _ns_scaling_failure(
            workspace, NS_SCALING_INVALID_PARAMETER,
        )
    end
    return _try_update_nonsymmetric_scaling!(
        workspace, policy, tag, primal, dual, muT,
    )
end

"""Apply the latest valid primal-to-dual metric without allocating."""
@inline function apply_nonsymmetric_G!(destination, workspace, source)
    workspace.valid || throw(ArgumentError("nonsymmetric scaling is invalid"))
    return _ns_scaling_matvec!(destination, workspace.g, source)
end

"""Apply the latest valid dual-to-primal metric without allocating."""
@inline function apply_nonsymmetric_Theta!(destination, workspace, source)
    workspace.valid || throw(ArgumentError("nonsymmetric scaling is invalid"))
    return _ns_scaling_matvec!(destination, workspace.theta, source)
end

# ---------------------------------------------------------------------------
# Fixed-size Exp/Power 3x3 scaling/contribution path (GPT Pro plan P4/C3).
#
# Each scaling epoch freezes the accepted 3x3 Theta and its certified lower
# factor `workspace.factor`.  The local KKT contribution is assembled from
# that accepted state only: no generic dense factor call per block and no
# global inverse are ever formed.  The generic reference path
# (`apply_nonsymmetric_Theta!` / `apply_nonsymmetric_G!`) remains callable
# unchanged.
# ---------------------------------------------------------------------------

"""
    nonsymmetric_scaling_contribution_symbol() -> Symbol

Specialization symbol of the fixed-size Exp/Power 3x3 contribution path,
registered in the KKT specialization registry.
"""
@inline function nonsymmetric_scaling_contribution_symbol()
    return _nonsymmetric_scaling_contribution_symbol()
end

"""
    nonsymmetric_scaling_accepted_factor(workspace) -> Matrix{T}

Return the accepted lower 3x3 Cholesky factor of the accepted block Theta
(block-owned storage; not copied, not re-factorized).  Throws on an invalid
scaling state, mirroring `apply_nonsymmetric_Theta!`.
"""
@inline function nonsymmetric_scaling_accepted_factor(
    workspace::NonsymmetricScalingWorkspace{T},
) where {T<:AbstractFloat}
    workspace.valid || throw(ArgumentError("nonsymmetric scaling is invalid"))
    return workspace.factor
end

"""
    _ns_scaling_factor_g_backward_ok(workspace, destination, source) -> Bool

Componentwise, scale-free backward-error gate for the factor-based `G` action:
certifies `Theta * destination == source` against the stored accepted Theta
with the same forcing constant as the runtime Theta-solve gate.
"""
@inline function _ns_scaling_factor_g_backward_ok(
    workspace::NonsymmetricScalingWorkspace{T},
    destination::AbstractVector{T},
    source::AbstractVector{T},
) where {T<:AbstractFloat}
    theta = workspace.theta
    three_eps = T(3) * eps(one(T))
    three_eps < one(T) || return false
    forcing = T(128) * three_eps / (one(T) - three_eps)
    isfinite(forcing) || return false
    @inbounds for i in 1:3
        action = theta[i, 1] * destination[1] +
                 theta[i, 2] * destination[2] +
                 theta[i, 3] * destination[3]
        work = abs(source[i]) +
               abs(theta[i, 1]) * abs(destination[1]) +
               abs(theta[i, 2]) * abs(destination[2]) +
               abs(theta[i, 3]) * abs(destination[3])
        residual = action - source[i]
        isfinite(residual) && isfinite(work) || return false
        if iszero(work)
            iszero(residual) || return false
        elseif abs(residual) > forcing * work
            return false
        end
    end
    return true
end

"""
    try_apply_nonsymmetric_factor_G_reason!(destination, workspace, source)

Apply `G = Theta^-1` for one accepted Exp/Power block through the accepted
block factor only: `(L*L') x = source` with the certified lower factor, then
a componentwise backward-error check against the stored accepted Theta.  No
re-factorization of Theta and no generic dense factor call per block occur;
`source` may alias `destination` (the forward solve captures the RHS first).
"""
@inline function try_apply_nonsymmetric_factor_G_reason!(
    destination::Vector{T},
    workspace::NonsymmetricScalingWorkspace{T},
    source::Vector{T},
) where {T<:AbstractFloat}
    workspace.valid || return NS_SCALING_INVALID_PARAMETER
    length(destination) == length(source) == 3 ||
        return NS_SCALING_INVALID_PARAMETER
    isfinite(source[1]) && isfinite(source[2]) && isfinite(source[3]) ||
        return NS_SCALING_NONFINITE_INPUT
    _ns_scaling_factor_solve!(
        destination, workspace.factor, source, workspace.work3,
    ) || return NS_SCALING_INVERSE_MISMATCH
    _ns_scaling_factor_g_backward_ok(workspace, destination, source) ||
        return NS_SCALING_INVERSE_MISMATCH
    return NS_SCALING_CONVERGED
end

"""
    try_apply_nonsymmetric_factor_G!(destination, workspace, source) -> Bool

Boolean wrapper of the reason-returning factor-based `G` action.
"""
@inline function try_apply_nonsymmetric_factor_G!(
    destination::Vector{T},
    workspace::NonsymmetricScalingWorkspace{T},
    source::Vector{T},
) where {T<:AbstractFloat}
    return try_apply_nonsymmetric_factor_G_reason!(
        destination, workspace, source,
    ) === NS_SCALING_CONVERGED
end

"""
    nonsymmetric_scaling_contribution3!(
        operator, corrector, workspace, corrector_rhs,
    ) -> NonsymmetricScalingReason

Assemble the fixed-size 3x3 local KKT contribution for one accepted Exp/Power
block into caller-owned block scratch: the self-adjoint local linearization
`operator` is the accepted Theta and `corrector` receives the block slice of
the caller's corrector right-hand side.  Uses only the accepted block scaling
state; no factorization and no global inverse are formed.
"""
@inline function nonsymmetric_scaling_contribution3!(
    operator::AbstractMatrix{T},
    corrector::AbstractVector{T},
    workspace::NonsymmetricScalingWorkspace{T},
    corrector_rhs::AbstractVector{T},
) where {T<:AbstractFloat}
    workspace.valid || return NS_SCALING_INVALID_PARAMETER
    size(operator) == (3, 3) || return NS_SCALING_INVALID_PARAMETER
    length(corrector) == 3 && length(corrector_rhs) == 3 ||
        return NS_SCALING_INVALID_PARAMETER
    theta = workspace.theta
    @inbounds for j in 1:3, i in 1:3
        value = theta[i, j]
        isfinite(value) || return NS_SCALING_NONFINITE_RESULT
        operator[i, j] = value
    end
    @inbounds for i in 1:3
        value = corrector_rhs[i]
        isfinite(value) || return NS_SCALING_NONFINITE_RESULT
        corrector[i] = value
    end
    return NS_SCALING_CONVERGED
end
