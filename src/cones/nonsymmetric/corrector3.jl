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
    workspace.valid = status !== NS_CORRECTOR_FAILED
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
    log_ratio = _nsd_sub(_nsd_add(_nsd_log(y), t), _nsd_log(z))
    rho = _nsd_exp(log_ratio)
    delta = _nsd_sub(one_d, rho)
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
    rho = _nsd_mul(z, inv_w)
    delta = _nsd_sub(one_d, _nsd_mul(rho, rho))
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

"""
    try_nonsymmetric_higher_correction!(workspace, tag, s, ds_aff, dy_aff)

Compute `u=H_F(s)\\dy_aff` and
`chi=-F'''(s)[ds_aff,u]/2`.  The Euler identity
`dot(s,chi)=dot(ds_aff,dy_aff)` is a runtime acceptance gate.
"""
function try_nonsymmetric_higher_correction!(
    workspace::NonsymmetricCorrectorWorkspace{T},
    tag::NonsymmetricConjugateTag,
    primal,
    ds_aff,
    dy_aff,
) where {T<:AbstractFloat}
    workspace.valid = false
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

    workspace.work[1] = convert(T, dy_aff[1])
    workspace.work[2] = convert(T, dy_aff[2])
    workspace.work[3] = convert(T, dy_aff[3])
    _ns_conjugate_spd_solve!(
        workspace.u, workspace.hessian, workspace.work, workspace.factor,
    ) || return _ns_corrector_failure(
        workspace, NS_CORRECTOR_HESSIAN_NOT_SPD,
    )
    workspace.work[1] = convert(T, ds_aff[1])
    workspace.work[2] = convert(T, ds_aff[2])
    workspace.work[3] = convert(T, ds_aff[3])
    _ns_corrector_third_contraction!(
        workspace.chi, tag, (s1, s2, s3), workspace.work, workspace.u,
    ) || return _ns_corrector_failure(
        workspace, NS_CORRECTOR_THIRD_DERIVATIVE_FAILED,
    )

    lhs = s1 * workspace.chi[1] + s2 * workspace.chi[2] +
          s3 * workspace.chi[3]
    rhs = convert(T, ds_aff[1]) * convert(T, dy_aff[1]) +
          convert(T, ds_aff[2]) * convert(T, dy_aff[2]) +
          convert(T, ds_aff[3]) * convert(T, dy_aff[3])
    scale = max(one(T), abs(lhs), abs(rhs))
    euler_error = abs(lhs - rhs) / scale
    isfinite(euler_error) && euler_error <= workspace.validation_tolerance ||
        return _ns_corrector_failure(workspace, NS_CORRECTOR_EULER_MISMATCH)
    return _ns_corrector_result(
        workspace, NS_CORRECTOR_COMBINED_READY, NS_CORRECTOR_CONVERGED,
        euler_error, zero(T),
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
    _ns_scaling_matvec!(workspace.work, scaling.theta, workspace.rho)
    error = zero(T)
    scale = one(T)
    @inbounds for i in 1:3
        error = max(error, abs(workspace.work[i] - workspace.h[i]))
        scale = max(scale, abs(workspace.h[i]))
    end
    linearization_error = error / scale
    isfinite(linearization_error) &&
        linearization_error <= workspace.validation_tolerance ||
        return _ns_corrector_failure(
            workspace, NS_CORRECTOR_LINEARIZATION_MISMATCH,
        )
    fill!(workspace.chi, zero(T))
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
    error = zero(T)
    scale = one(T)
    @inbounds for i in 1:3
        error = max(error, abs(workspace.work[i] - workspace.rho[i]))
        scale = max(scale, abs(workspace.rho[i]))
    end
    linearization_error = error / scale
    isfinite(linearization_error) &&
        linearization_error <= workspace.validation_tolerance ||
        return _ns_corrector_failure(
            workspace, NS_CORRECTOR_LINEARIZATION_MISMATCH,
        )
    return _ns_corrector_result(
        workspace, NS_CORRECTOR_COMBINED_READY, NS_CORRECTOR_CONVERGED,
        correction.euler_error, linearization_error,
    )
end
