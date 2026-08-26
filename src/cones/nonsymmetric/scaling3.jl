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
    return sqrt(_ns_scaling_dot3(a, a))
end

@inline function _ns_scaling_finite_vector(a)
    return isfinite(a[1]) && isfinite(a[2]) && isfinite(a[3])
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

@inline function _ns_scaling_fallback_allowed(::StrictDoubleSecantScaling)
    return false
end
@inline function _ns_scaling_fallback_allowed(
    ::DoubleSecantWithDualHessianFallback,
)
    return true
end

@inline function _ns_scaling_primal_gradient_hessian!(
    workspace, tag::NonsymmetricConjugateTag,
)
    s = workspace.primal
    _ns_conjugate_gradient!(
        workspace.primal_gradient, tag, s[1], s[2], s[3],
    )
    _ns_conjugate_hessian!(
        workspace.primal_hessian, tag, s[1], s[2], s[3],
    )
    workspace.dual_shadow[1] = -workspace.primal_gradient[1]
    workspace.dual_shadow[2] = -workspace.primal_gradient[2]
    workspace.dual_shadow[3] = -workspace.primal_gradient[3]
    return nothing
end

@inline function _ns_scaling_inverse_error!(workspace)
    G = workspace.g
    theta = workspace.theta
    error = zero(workspace.mu)
    @inbounds for j in 1:3
        for i in 1:3
            value = G[i, 1] * theta[1, j] +
                    G[i, 2] * theta[2, j] +
                    G[i, 3] * theta[3, j]
            target = i == j ? one(value) : zero(value)
            error = max(error, abs(value - target))
        end
    end
    return error
end

@inline function _ns_scaling_secant_error!(workspace, double_secant::Bool)
    s = workspace.primal
    y = workspace.dual
    stilde = workspace.conjugate.shadow
    ytilde = workspace.dual_shadow
    error = zero(workspace.mu)
    scale = max(
        one(workspace.mu), _ns_scaling_norm3(s), _ns_scaling_norm3(y),
        _ns_scaling_norm3(stilde), _ns_scaling_norm3(ytilde),
    )

    _ns_scaling_matvec!(workspace.work1, workspace.g, s)
    _ns_scaling_matvec!(workspace.work2, workspace.theta, y)
    @inbounds for i in 1:3
        error = max(error, abs(workspace.work1[i] - y[i]))
        error = max(error, abs(workspace.work2[i] - s[i]))
    end
    if double_secant
        _ns_scaling_matvec!(workspace.work1, workspace.g, stilde)
        _ns_scaling_matvec!(workspace.work2, workspace.theta, ytilde)
        @inbounds for i in 1:3
            error = max(error, abs(workspace.work1[i] - ytilde[i]))
            error = max(error, abs(workspace.work2[i] - stilde[i]))
        end
    end
    return error / scale
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
    inverse_error = _ns_scaling_inverse_error!(workspace)
    tolerance = workspace.settings.validation_tolerance
    isfinite(secant_error) && secant_error <= tolerance ||
        return NS_SCALING_SECANT_MISMATCH, secant_error, inverse_error
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

    m11 = _ns_scaling_dot3(y, s)
    m12 = _ns_scaling_dot3(y, stilde)
    m21 = _ns_scaling_dot3(ytilde, s)
    m22 = _ns_scaling_dot3(ytilde, stilde)
    mscale = max(one(T), abs(m11), abs(m12), abs(m21), abs(m22))
    abs(m12 - m21) <= tolerance * mscale ||
        return NS_SCALING_GRAM_NONSYMMETRIC
    max(abs(m12 - degree), abs(m21 - degree)) <= tolerance * mscale ||
        return NS_SCALING_SHADOW_IDENTITY_FAILED
    mcross = (m12 + m21) / T(2)
    determinant = m11 * m22 - mcross * mcross
    determinant_scale = max(one(T), abs(m11 * m22), abs(mcross * mcross))
    isfinite(determinant) && determinant > degeneracy * determinant_scale ||
        return NS_SCALING_SECOND_SECANT_DEGENERATE
    i11 = m22 / determinant
    i12 = -mcross / determinant
    i22 = m11 / determinant
    _ns_conjugate_finite3(i11, i12, i22) ||
        return NS_SCALING_SECOND_SECANT_DEGENERATE

    _ns_scaling_cross!(workspace.axis_z, s, stilde)
    znorm = _ns_scaling_norm3(workspace.axis_z)
    zscale = max(one(T), _ns_scaling_norm3(s) * _ns_scaling_norm3(stilde))
    isfinite(znorm) && znorm > degeneracy * zscale ||
        return NS_SCALING_AXIS_DEGENERATE
    @inbounds for i in 1:3
        workspace.axis_z[i] /= znorm
    end
    _ns_scaling_cross!(workspace.axis_r, y, ytilde)
    axis_pairing = _ns_scaling_dot3(workspace.axis_z, workspace.axis_r)
    rscale = max(one(T), _ns_scaling_norm3(workspace.axis_r))
    isfinite(axis_pairing) && abs(axis_pairing) > degeneracy * rscale ||
        return NS_SCALING_AXIS_PAIRING_DEGENERATE
    @inbounds for i in 1:3
        workspace.axis_r[i] /= axis_pairing
    end

    _ns_scaling_rank2!(workspace.g, y, ytilde, i11, i12, i22)
    _ns_scaling_rank2!(workspace.theta, s, stilde, i11, i12, i22)

    H = workspace.primal_hessian
    G0 = workspace.g0
    @inbounds for j in 1:3, i in 1:3
        G0[i, j] = workspace.mu * H[i, j]
    end
    _ns_scaling_matvec!(workspace.work1, G0, s)
    _ns_scaling_matvec!(workspace.work2, G0, stilde)
    k11 = _ns_scaling_dot3(s, workspace.work1)
    k12 = _ns_scaling_dot3(s, workspace.work2)
    k21 = _ns_scaling_dot3(stilde, workspace.work1)
    k22 = _ns_scaling_dot3(stilde, workspace.work2)
    kscale = max(one(T), abs(k11), abs(k12), abs(k21), abs(k22))
    abs(k12 - k21) <= tolerance * kscale ||
        return NS_SCALING_BFGS_DENOMINATOR
    kcross = (k12 + k21) / T(2)
    kdet = k11 * k22 - kcross * kcross
    kdscale = max(one(T), abs(k11 * k22), abs(kcross * kcross))
    isfinite(kdet) && kdet > degeneracy * kdscale ||
        return NS_SCALING_BFGS_DENOMINATOR
    q11 = k22 / kdet
    q12 = -kcross / kdet
    q22 = k11 / kdet
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
    tgscale = one(T)
    @inbounds for value in workspace.g_bfgs
        tgscale = max(tgscale, abs(value))
    end
    isfinite(tG) && tG > degeneracy * tgscale ||
        return NS_SCALING_AXIS_COEFFICIENT
    _ns_scaling_add_outer!(workspace.g, workspace.axis_z, tG)
    _ns_scaling_add_outer!(workspace.theta, workspace.axis_r, inv(tG))

    reason, _, _ = _ns_scaling_validate_metric!(workspace, true)
    return reason
end

@inline function _ns_scaling_dual_hessian_fallback!(
    workspace::NonsymmetricScalingWorkspace{T},
) where {T}
    theta0 = workspace.work_matrix
    hstar = workspace.conjugate.inverse_hessian
    @inbounds for j in 1:3, i in 1:3
        theta0[i, j] = workspace.mu * hstar[i, j]
    end
    _ns_scaling_spd!(workspace, theta0) ||
        return NS_SCALING_FALLBACK_NOT_SPD

    s = workspace.primal
    y = workspace.dual
    _ns_scaling_matvec!(workspace.work1, theta0, y)
    denominator = _ns_scaling_dot3(y, workspace.work1)
    pairing = _ns_scaling_dot3(s, y)
    scale = max(one(T), abs(denominator), abs(pairing))
    tolerance = workspace.settings.degeneracy_tolerance
    isfinite(denominator) && isfinite(pairing) &&
        denominator > tolerance * scale && pairing > tolerance * scale ||
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
    _ns_scaling_inverse_spd!(workspace.g, workspace.theta, workspace) ||
        return NS_SCALING_FALLBACK_NOT_SPD
    reason, _, _ = _ns_scaling_validate_metric!(workspace, false)
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
           reason === NS_SCALING_INVERSE_MISMATCH
end

"""
    try_update_nonsymmetric_scaling!(workspace, policy, tag, s, y)

Update the local degree-three primal-dual metric.  Strict policy never changes
provider.  The fallback policy may use the Fenchel conjugate Hessian only
after recording why the double-secant route was rejected; the fallback is
then repaired to satisfy `Theta*y=s` and validated as a finite SPD inverse
pair.  Expected numerical failures return `NS_SCALING_FAILED`.
"""
function try_update_nonsymmetric_scaling!(
    workspace::NonsymmetricScalingWorkspace{T},
    policy::NonsymmetricScalingPolicy,
    tag::NonsymmetricConjugateTag,
    primal,
    dual,
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
    pairing = _ns_scaling_dot3(workspace.primal, workspace.dual)
    pairing_scale = max(
        one(T), _ns_scaling_norm3(workspace.primal) *
                _ns_scaling_norm3(workspace.dual),
    )
    isfinite(pairing) &&
        pairing > settings.degeneracy_tolerance * pairing_scale ||
        return _ns_scaling_failure(
            workspace, NS_SCALING_NONPOSITIVE_PAIRING,
        )
    workspace.mu = pairing / T(3)

    conjugate_result = conjugate_shadow!(workspace.conjugate, tag, workspace.dual)
    conjugate_result.status === NS_CONJUGATE_SUCCESS ||
        return _ns_scaling_failure(
            workspace, NS_SCALING_CONJUGATE_FAILED;
            conjugate_reason=conjugate_result.reason,
        )
    barrier_ok = try
        _ns_scaling_primal_gradient_hessian!(workspace, tag)
        true
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false
    end
    barrier_ok && _ns_scaling_finite_vector(workspace.dual_shadow) &&
        _ns_scaling_finite_matrix(workspace.primal_hessian) ||
        return _ns_scaling_failure(workspace, NS_SCALING_NONFINITE_RESULT)

    primary_reason = _ns_scaling_double_secant!(workspace)
    if primary_reason === NS_SCALING_CONVERGED
        _, secant_error, inverse_error =
            _ns_scaling_validate_metric!(workspace, true)
        return _ns_scaling_result(
            workspace, NS_SCALING_DOUBLE_SECANT, NS_SCALING_CONVERGED,
            NS_SCALING_NO_FALLBACK, conjugate_result.reason,
            secant_error, inverse_error,
        )
    end
    if !_ns_scaling_fallback_allowed(policy) ||
       !_ns_scaling_fallback_eligible(primary_reason)
        return _ns_scaling_failure(
            workspace, primary_reason;
            conjugate_reason=conjugate_result.reason,
        )
    end

    fallback_terminal = _ns_scaling_dual_hessian_fallback!(workspace)
    if fallback_terminal !== NS_SCALING_CONVERGED
        return _ns_scaling_failure(
            workspace, fallback_terminal;
            fallback_reason=primary_reason,
            conjugate_reason=conjugate_result.reason,
        )
    end
    _, secant_error, inverse_error =
        _ns_scaling_validate_metric!(workspace, false)
    return _ns_scaling_result(
        workspace, NS_SCALING_DUAL_HESSIAN_FALLBACK,
        NS_SCALING_CONVERGED, primary_reason, conjugate_result.reason,
        secant_error, inverse_error,
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
