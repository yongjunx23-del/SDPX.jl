# Higher-order correction for three-dimensional nonsymmetric barriers.
#
# With the frozen primal barrier F and affine directions (ds,dy), compute
#
#     u   = inv(hessian(F,s)) * dy,
#     chi = -1/2 * F'''(s)[ds,u].
#
# The third-order contraction is evaluated as the exact first directional
# derivative of the analytic Hessian formula.  No finite difference, AD tape,
# closure, or full third-order tensor appears on the production path.

@enum NonsymmetricCorrectorStatus::UInt8 begin
    NS_CORRECTOR_AFFINE_READY = 0x00
    NS_CORRECTOR_COMBINED_READY = 0x01
    NS_CORRECTOR_FAILED = 0x02
end

@enum NonsymmetricCorrectorReason::UInt8 begin
    NS_CORRECTOR_CONVERGED = 0x00
    NS_CORRECTOR_INVALID_PARAMETER = 0x01
    NS_CORRECTOR_NONFINITE_INPUT = 0x02
    NS_CORRECTOR_PRIMAL_NOT_INTERIOR = 0x03
    NS_CORRECTOR_SCALING_INVALID = 0x04
    NS_CORRECTOR_SCALING_POINT_MISMATCH = 0x05
    NS_CORRECTOR_HESSIAN_FAILED = 0x06
    NS_CORRECTOR_HESSIAN_NOT_SPD = 0x07
    NS_CORRECTOR_THIRD_DERIVATIVE_FAILED = 0x08
    NS_CORRECTOR_EULER_MISMATCH = 0x09
    NS_CORRECTOR_METRIC_FAILED = 0x0a
    NS_CORRECTOR_LINEARIZATION_MISMATCH = 0x0b
    NS_CORRECTOR_LINEAR_SOLVE_MISMATCH = 0x0c
    NS_CORRECTOR_THIRD_SYMMETRY_MISMATCH = 0x0d
    NS_CORRECTOR_PROJECTION_TOO_LARGE = 0x0e
    NS_CORRECTOR_FACTOR_FAILED = 0x0f
    NS_CORRECTOR_FACTOR_MISMATCH = 0x10
end

struct NonsymmetricCorrectorResult{T}
    status::NonsymmetricCorrectorStatus
    reason::NonsymmetricCorrectorReason
    euler_error::T
    linearization_error::T
end


mutable struct NonsymmetricCorrectorWorkspace{T}
    validation_tolerance::T
    hessian::Matrix{T}
    factor::Matrix{T}
    u::Vector{T}
    chi::Vector{T}
    rho::Vector{T}
    h::Vector{T}
    work::Vector{T}
    swap::Vector{T}
    natural_bound::Vector{T}
    solve_error::T
    symmetry_error::T
    raw_euler_error::T
    projection_error::T
    factor_error::T
    factor_valid::Bool
    valid::Bool
    last_status::NonsymmetricCorrectorStatus
    last_reason::NonsymmetricCorrectorReason
end

function NonsymmetricCorrectorWorkspace(
    ::Type{T}; validation_tolerance=T(32768) * eps(one(T)),
) where {T<:AbstractFloat}
    tolerance = convert(T, validation_tolerance)
    return NonsymmetricCorrectorWorkspace{T}(
        tolerance,
        zeros(T, 3, 3),
        zeros(T, 3, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        T(Inf),
        T(Inf),
        T(Inf),
        T(Inf),
        T(Inf),
        false,
        false,
        NS_CORRECTOR_FAILED,
        NS_CORRECTOR_INVALID_PARAMETER,
    )
end

NonsymmetricCorrectorWorkspace{T}(; kwargs...) where {T<:AbstractFloat} =
    NonsymmetricCorrectorWorkspace(T; kwargs...)

@inline function _ns_corrector_result(
    workspace::NonsymmetricCorrectorWorkspace{T},
    status::NonsymmetricCorrectorStatus,
    reason::NonsymmetricCorrectorReason,
    euler_error::T,
    linearization_error::T,
) where {T}
    failed = status === NS_CORRECTOR_FAILED
    workspace.valid = !failed
    if failed
        # A post-factor jet/Euler failure must never leave a factor-valid flag
        # that a later caller could mistake for an accepted corrector epoch.
        workspace.factor_valid = false
        workspace.factor_error = T(Inf)
    end
    workspace.last_status = status
    workspace.last_reason = reason
    return NonsymmetricCorrectorResult{T}(
        status, reason, euler_error, linearization_error,
    )
end

@inline function _ns_corrector_failure(
    workspace::NonsymmetricCorrectorWorkspace{T},
    reason::NonsymmetricCorrectorReason,
) where {T}
    return _ns_corrector_result(
        workspace, NS_CORRECTOR_FAILED, reason, T(Inf), T(Inf),
    )
end

# A value and its derivative in one caller-supplied direction.  The type is
# isbits whenever T is isbits, and every operation below is scalar/inlined.
struct _NSDirectional3{T}
    value::T
    derivative::T
end

@inline _nsd_constant(value::T) where {T} = _NSDirectional3{T}(value, zero(T))
@inline _nsd_add(a::_NSDirectional3, b::_NSDirectional3) =
    _NSDirectional3(a.value + b.value, a.derivative + b.derivative)
@inline _nsd_sub(a::_NSDirectional3, b::_NSDirectional3) =
    _NSDirectional3(a.value - b.value, a.derivative - b.derivative)
@inline _nsd_neg(a::_NSDirectional3) =
    _NSDirectional3(-a.value, -a.derivative)
@inline _nsd_scale(value, a::_NSDirectional3) =
    _NSDirectional3(value * a.value, value * a.derivative)
@inline _nsd_mul(a::_NSDirectional3, b::_NSDirectional3) =
    _NSDirectional3(
        a.value * b.value,
        a.derivative * b.value + a.value * b.derivative,
    )
@inline function _nsd_inv(a::_NSDirectional3)
    inverse = inv(a.value)
    return _NSDirectional3(inverse, -a.derivative * inverse * inverse)
end
@inline _nsd_div(a::_NSDirectional3, b::_NSDirectional3) =
    _nsd_mul(a, _nsd_inv(b))
@inline function _nsd_exp(a::_NSDirectional3)
    value = exp(a.value)
    return _NSDirectional3(value, value * a.derivative)
end
@inline _nsd_log(a::_NSDirectional3) =
    _NSDirectional3(log(a.value), a.derivative / a.value)
@inline _nsd_log_abs(a::_NSDirectional3) =
    _NSDirectional3(log(abs(a.value)), a.derivative / a.value)
@inline function _nsd_negative_expm1(a::_NSDirectional3)
    exponential = exp(a.value)
    return _NSDirectional3(
        -_nonsymmetric_stable_expm1(a.value),
        -exponential * a.derivative,
    )
end
@inline function _nsd_log1p(a::_NSDirectional3)
    return _NSDirectional3(
        _nonsymmetric_stable_log1p(a.value),
        a.derivative / (one(a.value) + a.value),
    )
end
@inline function _nsd_positive_log_ratio(
    numerator::_NSDirectional3, denominator::_NSDirectional3,
)
    relative = _nsd_div(_nsd_sub(numerator, denominator), denominator)
    if isfinite(relative.value) && relative.value > -one(relative.value)
        return _nsd_log1p(relative)
    end
    return _nsd_sub(_nsd_log(numerator), _nsd_log(denominator))
end
@inline function _nsd_abs_value(a::_NSDirectional3)
    sign_a = copysign(one(a.value), a.value)
    return _NSDirectional3(abs(a.value), sign_a * a.derivative)
end

@inline function _ns_corrector_contract_from_hessian_derivative!(
    destination, h11, h12, h13, h22, h23, h33, u,
)
    half = inv(one(h11.derivative) + one(h11.derivative))
    u1, u2, u3 = u[1], u[2], u[3]
    c1 = h11.derivative * u1 + h12.derivative * u2 + h13.derivative * u3
    c2 = h12.derivative * u1 + h22.derivative * u2 + h23.derivative * u3
    c3 = h13.derivative * u1 + h23.derivative * u2 + h33.derivative * u3
    destination[1] = -half * c1
    destination[2] = -half * c2
    destination[3] = -half * c3
    return _ns_scaling_finite_vector(destination)
end

@inline function _ns_exp_third_contraction!(destination, s, ds, u)
    T = eltype(destination)
    x = _NSDirectional3{T}(s[1], ds[1])
    y = _NSDirectional3{T}(s[2], ds[2])
    z = _NSDirectional3{T}(s[3], ds[3])
    one_d = _nsd_constant(one(T))

    t = _nsd_div(x, y)
    log_ratio = _nsd_add(t, _nsd_positive_log_ratio(y, z))
    rho = _nsd_exp(log_ratio)
    delta = _nsd_negative_expm1(log_ratio)
    c = _nsd_div(rho, delta)
    c2 = _nsd_mul(c, c)
    inv_y = _nsd_inv(y)
    inv_z = _nsd_inv(z)
    inv_y2 = _nsd_mul(inv_y, inv_y)
    inv_z2 = _nsd_mul(inv_z, inv_z)
    inv_delta = _nsd_inv(delta)
    omt = _nsd_sub(one_d, t)

    h11 = _nsd_mul(_nsd_add(c, c2), inv_y2)
    h12 = _nsd_mul(
        _nsd_add(_nsd_neg(_nsd_mul(c, t)), _nsd_mul(c2, omt)), inv_y2,
    )
    h13 = _nsd_neg(_nsd_mul(_nsd_mul(_nsd_mul(c, inv_delta), inv_y), inv_z))
    h22 = _nsd_mul(
        _nsd_add(
            _nsd_add(_nsd_mul(c, _nsd_mul(t, t)),
                     _nsd_mul(c2, _nsd_mul(omt, omt))),
            one_d,
        ),
        inv_y2,
    )
    h23 = _nsd_neg(
        _nsd_mul(_nsd_mul(_nsd_mul(_nsd_mul(c, omt), inv_delta), inv_y), inv_z),
    )
    h33 = _nsd_mul(
        _nsd_add(_nsd_mul(inv_delta, inv_delta), one_d), inv_z2,
    )
    return _ns_corrector_contract_from_hessian_derivative!(
        destination, h11, h12, h13, h22, h23, h33, u,
    )
end

@inline function _ns_power_third_contraction!(destination, s, ds, u, alpha)
    T = eltype(destination)
    a = convert(T, alpha)
    b = one(T) - a
    x = _NSDirectional3{T}(s[1], ds[1])
    y = _NSDirectional3{T}(s[2], ds[2])
    z = _NSDirectional3{T}(s[3], ds[3])
    one_d = _nsd_constant(one(T))

    log_w = _nsd_add(_nsd_scale(a, _nsd_log(x)),
                     _nsd_scale(b, _nsd_log(y)))
    inv_w = _nsd_exp(_nsd_neg(log_w))
    rho, delta = if iszero(z.value)
        local_rho = _nsd_mul(z, inv_w)
        local_rho, _nsd_sub(one_d, _nsd_mul(local_rho, local_rho))
    else
        absolute_z = _nsd_abs_value(z)
        log_abs_rho = _nsd_add(
            _nsd_scale(a, _nsd_positive_log_ratio(absolute_z, x)),
            _nsd_scale(b, _nsd_positive_log_ratio(absolute_z, y)),
        )
        abs_rho = _nsd_exp(log_abs_rho)
        sign_z = copysign(one(T), z.value)
        local_rho = _nsd_scale(sign_z, abs_rho)
        two_log_abs_rho = _nsd_scale(one(T) + one(T), log_abs_rho)
        local_rho, _nsd_negative_expm1(two_log_abs_rho)
    end
    inv_delta = _nsd_inv(delta)
    inv_delta2 = _nsd_mul(inv_delta, inv_delta)
    inv_x = _nsd_inv(x)
    inv_y = _nsd_inv(y)
    two_a = a + a
    two_b = b + b

    h11_inner = _nsd_add(
        _nsd_sub(_nsd_scale(two_a * two_a, inv_delta2),
                 _nsd_scale(two_a * (two_a - one(T)), inv_delta)),
        _nsd_constant(b),
    )
    h11 = _nsd_mul(h11_inner, _nsd_mul(inv_x, inv_x))
    h12 = _nsd_mul(
        _nsd_scale((a + a + a + a) * b,
                   _nsd_sub(inv_delta2, inv_delta)),
        _nsd_mul(inv_x, inv_y),
    )
    h13 = _nsd_neg(
        _nsd_scale(
            a + a + a + a,
            _nsd_mul(_nsd_mul(_nsd_mul(rho, inv_w), inv_delta2), inv_x),
        ),
    )
    h22_inner = _nsd_add(
        _nsd_sub(_nsd_scale(two_b * two_b, inv_delta2),
                 _nsd_scale(two_b * (two_b - one(T)), inv_delta)),
        _nsd_constant(a),
    )
    h22 = _nsd_mul(h22_inner, _nsd_mul(inv_y, inv_y))
    h23 = _nsd_neg(
        _nsd_scale(
            b + b + b + b,
            _nsd_mul(_nsd_mul(_nsd_mul(rho, inv_w), inv_delta2), inv_y),
        ),
    )
    two_rho = _nsd_add(rho, rho)
    h33 = _nsd_mul(
        _nsd_add(_nsd_mul(_nsd_mul(two_rho, two_rho), inv_delta2),
                 _nsd_scale(one(T) + one(T), inv_delta)),
        _nsd_mul(inv_w, inv_w),
    )
    return _ns_corrector_contract_from_hessian_derivative!(
        destination, h11, h12, h13, h22, h23, h33, u,
    )
end

@inline function _ns_corrector_third_contraction!(
    destination, ::ExpConjugateTag, s, ds, u,
)
    return _ns_exp_third_contraction!(destination, s, ds, u)
end
@inline function _ns_corrector_third_contraction!(
    destination, tag::PowerConjugateTag, s, ds, u,
)
    return _ns_power_third_contraction!(destination, s, ds, u, tag.alpha)
end

@inline function _ns_corrector_input_finite(vector)
    length(vector) == 3 || return false
    return isfinite(vector[1]) && isfinite(vector[2]) && isfinite(vector[3])
end

@inline function _ns_corrector_hessian_solve_gate!(
    workspace::NonsymmetricCorrectorWorkspace{T}, dy,
) where {T}
    workspace.factor_valid || return false
    ok, error = _ns_structural_hessian_solve_certificate!(
        workspace.factor, workspace.u, dy, workspace.swap, workspace.work,
    )
    workspace.solve_error = error
    return ok
end

@inline function _ns_corrector_third_symmetry_average!(
    workspace::NonsymmetricCorrectorWorkspace{T},
) where {T}
    workspace.factor_valid || return false
    # The structural factor is the Hessian authority.  Its analytic identities
    # were independently certified before this routine; never reconstruct a
    # cancellation-prone dense H merely to re-certify the same factor.
    l = workspace.factor

    # Standard self-concordance gives
    # |chi_i| <= ||ds||_H ||u||_H sqrt(H_ii).  Evaluate the two local norms
    # through the already-certified H=L*L' factor, avoiding a cancellation-
    # small output-relative denominator for the swapped contractions.
    d1 = l[1, 1] * workspace.work[1] +
         l[2, 1] * workspace.work[2] +
         l[3, 1] * workspace.work[3]
    d2 = l[2, 2] * workspace.work[2] +
         l[3, 2] * workspace.work[3]
    d3 = l[3, 3] * workspace.work[3]
    u1 = l[1, 1] * workspace.u[1] +
         l[2, 1] * workspace.u[2] +
         l[3, 1] * workspace.u[3]
    u2 = l[2, 2] * workspace.u[2] + l[3, 2] * workspace.u[3]
    u3 = l[3, 3] * workspace.u[3]
    dscale = max(abs(d1), abs(d2), abs(d3))
    uscale = max(abs(u1), abs(u2), abs(u3))
    dnorm = iszero(dscale) ? zero(T) :
            dscale * sqrt((d1 / dscale)^2 +
                          (d2 / dscale)^2 + (d3 / dscale)^2)
    unorm = iszero(uscale) ? zero(T) :
            uscale * sqrt((u1 / uscale)^2 +
                          (u2 / uscale)^2 + (u3 / uscale)^2)
    isfinite(dnorm) && isfinite(unorm) || return false

    forcing = T(512) * sqrt(eps(one(T)))
    magnitude_factor = one(T) + forcing
    worst = zero(T)
    @inbounds for i in 1:3
        hii = zero(T)
        for k in 1:i
            hii += l[i, k] * l[i, k]
        end
        isfinite(hii) && hii > zero(T) || return false
        bound = dnorm * unorm * sqrt(hii)
        isfinite(bound) || return false
        workspace.natural_bound[i] = bound
        primary = abs(workspace.chi[i])
        swapped = abs(workspace.swap[i])
        difference = abs(workspace.chi[i] - workspace.swap[i])
        if iszero(bound)
            iszero(primary) && iszero(swapped) && iszero(difference) ||
                return false
        else
            primary <= magnitude_factor * bound &&
                swapped <= magnitude_factor * bound &&
                difference <= forcing * bound || return false
            worst = max(worst, difference / bound)
        end
    end
    half = inv(one(T) + one(T))
    @inbounds for i in 1:3
        # Halve before adding so two same-sign finite contractions cannot
        # overflow in an otherwise representable midpoint.
        workspace.chi[i] = workspace.chi[i] * half +
                           workspace.swap[i] * half
        isfinite(workspace.chi[i]) || return false
    end
    workspace.symmetry_error = worst
    return true
end

@inline function _ns_corrector_euler_projection!(
    workspace::NonsymmetricCorrectorWorkspace{T},
    s1::T,
    s2::T,
    s3::T,
    ds_aff,
) where {T}
    d1 = convert(T, ds_aff[1])
    d2 = convert(T, ds_aff[2])
    d3 = convert(T, ds_aff[3])
    q1 = d1 * workspace.work[1]
    q2 = d2 * workspace.work[2]
    q3 = d3 * workspace.work[3]
    q = q1 + q2 + q3
    p1 = s1 * workspace.chi[1]
    p2 = s2 * workspace.chi[2]
    p3 = s3 * workspace.chi[3]
    p = p1 + p2 + p3
    # The raw structural gate uses the self-concordant natural bounds, so
    # cancellation-small chi coordinates do not define their own acceptance
    # scale. The post-projection gate below still checks the actual dot-product
    # arithmetic work at O(gamma_9).
    work = abs(s1) * workspace.natural_bound[1] +
           abs(s2) * workspace.natural_bound[2] +
           abs(s3) * workspace.natural_bound[3] +
           abs(q1) + abs(q2) + abs(q3)
    residual = abs(p - q)
    raw_error = if iszero(work)
        iszero(residual) ? zero(T) : T(Inf)
    else
        residual / work
    end
    raw_limit = T(1024) * sqrt(eps(one(T)))
    isfinite(raw_error) && raw_error <= raw_limit ||
        return NS_CORRECTOR_PROJECTION_TOO_LARGE
    workspace.raw_euler_error = raw_error

    a1, a2, a3 = abs(s1), abs(s2), abs(s3)
    k = a1 >= a2 && a1 >= a3 ? 1 : a2 >= a3 ? 2 : 3
    sk = k == 1 ? s1 : k == 2 ? s2 : s3
    isfinite(sk) && !iszero(sk) ||
        return NS_CORRECTOR_PROJECTION_TOO_LARGE
    correction = (q - p) / sk
    correction_scale = iszero(work) ? zero(T) : work / abs(sk)
    correction_error = if iszero(correction_scale)
        iszero(correction) ? zero(T) : T(Inf)
    else
        abs(correction) / correction_scale
    end
    isfinite(correction) && isfinite(correction_error) &&
        correction_error <= raw_limit ||
        return NS_CORRECTOR_PROJECTION_TOO_LARGE
    workspace.chi[k] += correction
    isfinite(workspace.chi[k]) ||
        return NS_CORRECTOR_PROJECTION_TOO_LARGE
    workspace.projection_error = correction_error

    p1 = s1 * workspace.chi[1]
    p2 = s2 * workspace.chi[2]
    p3 = s3 * workspace.chi[3]
    p = p1 + p2 + p3
    post_work = abs(p1) + abs(p2) + abs(p3) +
                abs(q1) + abs(q2) + abs(q3)
    post_residual = abs(p - q)
    nine_eps = T(9) * eps(one(T))
    nine_eps < one(T) || return NS_CORRECTOR_EULER_MISMATCH
    gamma9 = nine_eps / (one(T) - nine_eps)
    post_allowance = T(128) * gamma9 * post_work
    isfinite(post_residual) && isfinite(post_allowance) &&
        (iszero(post_work) ? iszero(post_residual) :
         post_residual <= post_allowance) ||
        return NS_CORRECTOR_EULER_MISMATCH
    return NS_CORRECTOR_CONVERGED
end

"""
    try_nonsymmetric_higher_correction!(workspace, tag, s, ds_aff, dy_aff)

Compute `u=H_F(s)\\dy_aff` and
`chi=-F'''(s)[ds_aff,u]/2`.  The Euler identity
`dot(s,chi)=dot(ds_aff,H_F(s)u)` is enforced only after four independent,
scale-free gates: a componentwise solve posterior, swapped third-derivative
symmetry in the self-concordant local norm, a bounded single-coordinate
structure projection, and an O(gamma) post-projection dot-product check.
`raw_euler_error` records the pre-projection defect; the projection is never
reported as raw jet accuracy.
"""
function try_nonsymmetric_higher_correction!(
    workspace::NonsymmetricCorrectorWorkspace{T},
    tag::NonsymmetricConjugateTag,
    primal,
    ds_aff,
    dy_aff,
) where {T<:AbstractFloat}
    workspace.valid = false
    workspace.solve_error = T(Inf)
    workspace.symmetry_error = T(Inf)
    workspace.raw_euler_error = T(Inf)
    workspace.projection_error = T(Inf)
    workspace.factor_error = T(Inf)
    workspace.factor_valid = false
    _ns_conjugate_valid_tag(tag) ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_INVALID_PARAMETER)
    isfinite(workspace.validation_tolerance) &&
        workspace.validation_tolerance > zero(T) ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_INVALID_PARAMETER)
    _ns_corrector_input_finite(primal) &&
        _ns_corrector_input_finite(ds_aff) &&
        _ns_corrector_input_finite(dy_aff) ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_NONFINITE_INPUT)
    s1 = convert(T, primal[1])
    s2 = convert(T, primal[2])
    s3 = convert(T, primal[3])
    _ns_conjugate_primal_interior(tag, s1, s2, s3) ||
        return _ns_corrector_failure(
            workspace, NS_CORRECTOR_PRIMAL_NOT_INTERIOR,
        )

    barrier_ok = try
        _ns_conjugate_hessian!(workspace.hessian, tag, s1, s2, s3)
        true
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false
    end
    barrier_ok && _ns_scaling_finite_matrix(workspace.hessian) ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_HESSIAN_FAILED)

    factor_built = try
        _ns_structural_hessian_factor!(
            workspace.factor, tag, s1, s2, s3,
        )
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false
    end
    factor_built || return _ns_corrector_failure(
        workspace, NS_CORRECTOR_FACTOR_FAILED,
    )
    factor_ok, factor_error = try
        _ns_structural_hessian_factor_certificate!(
            workspace.factor, tag, s1, s2, s3,
        )
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false, T(Inf)
    end
    workspace.factor_error = factor_error
    factor_ok || return _ns_corrector_failure(
        workspace, NS_CORRECTOR_FACTOR_MISMATCH,
    )
    workspace.factor_valid = true

    workspace.work[1] = convert(T, dy_aff[1])
    workspace.work[2] = convert(T, dy_aff[2])
    workspace.work[3] = convert(T, dy_aff[3])
    _ns_structural_hessian_solve!(
        workspace.u, workspace.factor, workspace.work,
        workspace.natural_bound,
    ) || return _ns_corrector_failure(
        workspace, NS_CORRECTOR_FACTOR_FAILED,
    )
    workspace.work[1] = convert(T, ds_aff[1])
    workspace.work[2] = convert(T, ds_aff[2])
    workspace.work[3] = convert(T, ds_aff[3])
    _ns_corrector_third_contraction!(
        workspace.chi, tag, (s1, s2, s3), workspace.work, workspace.u,
    ) || return _ns_corrector_failure(
        workspace, NS_CORRECTOR_THIRD_DERIVATIVE_FAILED,
    )

    # Third-derivative symmetry supplies an independent vector-valued check,
    # unlike the single scalar Euler identity.  The swapped contraction is
    # `-D H(s)[u]ds/2`; average only after the fail-closed comparison.
    _ns_corrector_third_contraction!(
        workspace.swap, tag, (s1, s2, s3), workspace.u, workspace.work,
    ) || return _ns_corrector_failure(
        workspace, NS_CORRECTOR_THIRD_DERIVATIVE_FAILED,
    )
    _ns_corrector_third_symmetry_average!(workspace) ||
        return _ns_corrector_failure(
            workspace, NS_CORRECTOR_THIRD_SYMMETRY_MISMATCH,
        )

    # The LH identity applies to the computed solve:
    #     s'chi = ds'*(H*u).
    # First certify H*u≈dy componentwise, then project one largest-|s_i|
    # coordinate onto the exact computed-u identity.  Raw jet quality is
    # recorded before the single projection and remains the reported error.
    _ns_corrector_hessian_solve_gate!(workspace, dy_aff) ||
        return _ns_corrector_failure(
            workspace, NS_CORRECTOR_LINEAR_SOLVE_MISMATCH,
        )
    projection_reason = _ns_corrector_euler_projection!(
        workspace, s1, s2, s3, ds_aff,
    )
    projection_reason === NS_CORRECTOR_CONVERGED ||
        return _ns_corrector_failure(workspace, projection_reason)
    return _ns_corrector_result(
        workspace, NS_CORRECTOR_COMBINED_READY, NS_CORRECTOR_CONVERGED,
        workspace.raw_euler_error, zero(T),
    )
end

@inline function _ns_corrector_scaling_matches(scaling, primal, dual)
    length(primal) == 3 && length(dual) == 3 || return false
    @inbounds for i in 1:3
        scaling.primal[i] == primal[i] || return false
        scaling.dual[i] == dual[i] || return false
    end
    return true
end

@inline function _ns_corrector_composed_map_gate!(workspace, scaling)
    T = eltype(workspace.rho)
    G = scaling.g
    theta = scaling.theta
    rho = workspace.rho
    h = workspace.h
    z = workspace.work
    E = workspace.factor

    gnorm = zero(T)
    theta_norm = zero(T)
    @inbounds for i in 1:3
        grow = abs(G[i, 1]) + abs(G[i, 2]) + abs(G[i, 3])
        theta_row = abs(theta[i, 1]) + abs(theta[i, 2]) + abs(theta[i, 3])
        gnorm = max(gnorm, grow)
        theta_norm = max(theta_norm, theta_row)
    end
    condition_bound = gnorm * theta_norm
    three_eps = T(3) * eps(one(T))
    gamma3 = three_eps / (one(T) - three_eps)
    isfinite(condition_bound) && isfinite(gamma3) &&
        gamma3 * condition_bound < T(1) / T(100) || return false, T(Inf)

    inverse_norm = zero(T)
    @inbounds for i in 1:3
        row_norm = zero(T)
        for j in 1:3
            value = G[i, 1] * theta[1, j] +
                    G[i, 2] * theta[2, j] +
                    G[i, 3] * theta[3, j] - (i == j ? one(T) : zero(T))
            E[i, j] = value
            row_norm += abs(value)
        end
        inverse_norm = max(inverse_norm, row_norm)
    end
    inverse_allowance = T(16) * scaling.settings.validation_tolerance *
                        max(one(T), condition_bound)
    isfinite(inverse_norm) && inverse_norm <= inverse_allowance ||
        return false, T(Inf)

    reported_error = zero(T)
    @inbounds for i in 1:3
        gh_work = zero(T)
        composed_work = zero(T)
        e_action = zero(T)
        for j in 1:3
            gij = abs(G[i, j])
            gh_work += gij * abs(h[j])
            theta_rho_work = abs(theta[j, 1]) * abs(rho[1]) +
                             abs(theta[j, 2]) * abs(rho[2]) +
                             abs(theta[j, 3]) * abs(rho[3])
            composed_work += gij * theta_rho_work
            e_action += abs(E[i, j]) * abs(rho[j])
        end
        arithmetic_work = abs(rho[i]) + gh_work + composed_work
        allowance = T(128) * gamma3 * arithmetic_work + T(8) * e_action
        residual = abs(z[i] - rho[i])
        isfinite(allowance) && isfinite(residual) && residual <= allowance ||
            return false, T(Inf)
        reported_error = max(
            reported_error,
            residual / (one(T) + arithmetic_work),
        )
    end
    return true, reported_error
end

"""Build the frozen affine RHS `rho=-y`, `h=-s`."""
function nonsymmetric_affine_shift!(
    workspace::NonsymmetricCorrectorWorkspace{T},
    scaling::NonsymmetricScalingWorkspace{T},
    primal,
    dual,
) where {T<:AbstractFloat}
    workspace.valid = false
    scaling.valid ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_SCALING_INVALID)
    _ns_corrector_input_finite(primal) &&
        _ns_corrector_input_finite(dual) ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_NONFINITE_INPUT)
    _ns_corrector_scaling_matches(scaling, primal, dual) ||
        return _ns_corrector_failure(
            workspace, NS_CORRECTOR_SCALING_POINT_MISMATCH,
        )
    _ns_scaling_finite_matrix(scaling.g) &&
        _ns_scaling_finite_matrix(scaling.theta) ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_METRIC_FAILED)
    @inbounds for i in 1:3
        workspace.rho[i] = -convert(T, dual[i])
        workspace.h[i] = -convert(T, primal[i])
    end
    # This repeats the already validated primary secant at the corrector
    # seam, so use the same per-row componentwise backward error.  A raw
    # target-relative residual is numerically impossible once `Theta` is
    # strongly conditioned near a curved boundary, even though the matvec is
    # backward stable.  Keep the frozen affine target `h=-s`; do not replace
    # it by the computed action.
    linearization_error = _ns_scaling_secant_equation_error!(
        scaling, scaling.theta, workspace.rho, workspace.h, workspace.work,
    )
    isfinite(linearization_error) &&
        linearization_error <= workspace.validation_tolerance ||
        return _ns_corrector_failure(
            workspace, NS_CORRECTOR_LINEARIZATION_MISMATCH,
        )
    zero_distinct!(workspace.chi)
    return _ns_corrector_result(
        workspace, NS_CORRECTOR_AFFINE_READY, NS_CORRECTOR_CONVERGED,
        zero(T), linearization_error,
    )
end

"""
Build the combined RHS
`rho=sigma_mu*ytilde-y-chi`, `h=Theta*rho` without changing provider.
"""
function nonsymmetric_combined_shift!(
    workspace::NonsymmetricCorrectorWorkspace{T},
    scaling::NonsymmetricScalingWorkspace{T},
    tag::NonsymmetricConjugateTag,
    primal,
    dual,
    ds_aff,
    dy_aff,
    sigma_mu,
) where {T<:AbstractFloat}
    workspace.valid = false
    scaling.valid ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_SCALING_INVALID)
    _ns_corrector_scaling_matches(scaling, primal, dual) ||
        return _ns_corrector_failure(
            workspace, NS_CORRECTOR_SCALING_POINT_MISMATCH,
        )
    _ns_scaling_finite_matrix(scaling.g) &&
        _ns_scaling_finite_matrix(scaling.theta) ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_METRIC_FAILED)
    target = convert(T, sigma_mu)
    isfinite(target) && target >= zero(T) ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_INVALID_PARAMETER)
    correction = try_nonsymmetric_higher_correction!(
        workspace, tag, primal, ds_aff, dy_aff,
    )
    correction.status === NS_CORRECTOR_COMBINED_READY || return correction

    @inbounds for i in 1:3
        workspace.rho[i] = target * scaling.dual_shadow[i] -
                           convert(T, dual[i]) - workspace.chi[i]
    end
    _ns_scaling_matvec!(workspace.h, scaling.theta, workspace.rho)
    _ns_scaling_matvec!(workspace.work, scaling.g, workspace.h)
    composed_ok, linearization_error =
        _ns_corrector_composed_map_gate!(workspace, scaling)
    composed_ok ||
        return _ns_corrector_failure(
            workspace, NS_CORRECTOR_LINEARIZATION_MISMATCH,
        )
    return _ns_corrector_result(
        workspace, NS_CORRECTOR_COMBINED_READY, NS_CORRECTOR_CONVERGED,
        correction.euler_error, linearization_error,
    )
end
