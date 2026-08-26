#=====================================================================#
#    Power-cone primal/dual degree-3 barrier oracle.
#
#    K_pow(alpha) = { (x, y, z) : x^alpha y^(1-alpha) >= |z|,
#                     x >= 0, y >= 0 },  alpha in (0, 1).
#
#    Validated 3-logarithmically-homogeneous barrier, beta=1-alpha:
#        F = -log(x^(2alpha)y^(2beta)-z^2)
#            - beta*log(x) - alpha*log(y).
#
#    Evaluation uses the signed normalized coordinate
#        rho = z / (x^alpha y^beta), delta = 1-rho^2,
#    so the weighted monomial is never formed in the membership/barrier
#    path.  The exact map L(u,v,w)=(u/alpha,v/beta,w) sends the dual
#    cone to the primal cone.
#=====================================================================#

const POWER_BARRIER_DEGREE = 3

@inline _power_finite3(x, y, z) = isfinite(x) && isfinite(y) && isfinite(z)

@inline function _power_alpha(alpha, x, y, z)
    T = promote_type(typeof(x), typeof(y), typeof(z))
    a = convert(T, alpha)
    isfinite(a) && zero(a) < a < one(a) || throw(ArgumentError(
        "power-cone alpha must be finite and in (0,1)",
    ))
    return a, one(a) - a
end

@inline function _power_primal_terms(x, y, z, alpha)
    _power_finite3(x, y, z) || throw(ArgumentError(
        "power-cone oracle requires finite coordinates",
    ))
    a, b = _power_alpha(alpha, x, y, z)
    x > zero(x) && y > zero(y) || throw(ArgumentError(
        "power-cone oracle requires x > 0 and y > 0",
    ))
    log_w = a * log(x) + b * log(y)
    isfinite(log_w) || throw(ArgumentError(
        "power-cone oracle produced a non-finite weighted log",
    ))
    rho = if iszero(z)
        zero(log_w)
    else
        log_abs_rho = log(abs(z)) - log_w
        tolerance = _nonsymmetric_log_tolerance(
            log(abs(z)), log_w,
        )
        isfinite(log_abs_rho) && log_abs_rho < -tolerance ||
            throw(ArgumentError(
                "power-cone oracle requires a resolvable |z| < x^alpha*y^(1-alpha) gap",
            ))
        copysign(exp(log_abs_rho), z)
    end
    delta = one(rho) - rho * rho
    isfinite(rho) && delta > zero(delta) || throw(ArgumentError(
        "power-cone oracle could not resolve a strict interior gap",
    ))
    return a, b, log_w, rho, delta
end

"""
    power_membership(x, y, z, alpha) -> Bool

Whether `(x, y, z)` lies in the power cone of parameter `alpha`.
"""
function power_membership(x, y, z, alpha)
    _power_finite3(x, y, z) || return false
    a, b = try
        _power_alpha(alpha, x, y, z)
    catch
        return false
    end
    x >= zero(x) && y >= zero(y) || return false
    iszero(z) && return true
    (iszero(x) || iszero(y)) && return false
    log_w = a * log(x) + b * log(y)
    log_abs_z = log(abs(z))
    tolerance = _nonsymmetric_log_tolerance(log_w, log_abs_z)
    return !isnan(log_w) && log_w - log_abs_z >= -tolerance
end

"""Whether `(x,y,z)` lies in the strict primal power-cone interior."""
function power_primal_interior(x, y, z, alpha)
    _power_finite3(x, y, z) || return false
    a, b = try
        _power_alpha(alpha, x, y, z)
    catch
        return false
    end
    x > zero(x) && y > zero(y) || return false
    iszero(z) && return true
    log_w = a * log(x) + b * log(y)
    log_abs_z = log(abs(z))
    tolerance = _nonsymmetric_log_tolerance(log_w, log_abs_z)
    return !isnan(log_w) && log_w - log_abs_z > tolerance
end

"""
    power_barrier(x, y, z, alpha)

Value of the smooth self-concordant barrier of the power cone at an
interior point.
"""
function power_barrier(x, y, z, alpha)
    a, b, log_w, rho, _ = _power_primal_terms(x, y, z, alpha)
    return -(one(log_w) + one(log_w)) * log_w -
           log1p(-(rho * rho)) - b * log(x) - a * log(y)
end

power_primal_barrier(x, y, z, alpha) = power_barrier(x, y, z, alpha)

"""
    power_barrier_gradient(x, y, z, alpha) -> (gx, gy, gz)

Gradient of the power-cone barrier.
"""
function power_barrier_gradient(x, y, z, alpha)
    a, b, log_w, rho, delta = _power_primal_terms(x, y, z, alpha)
    inv_delta = inv(delta)
    inv_w = exp(-log_w)
    gx = -(a + a) * inv_delta / x - b / x
    gy = -(b + b) * inv_delta / y - a / y
    gz = (one(rho) + one(rho)) * rho * inv_w * inv_delta
    result = (gx, gy, gz)
    all(isfinite, result) || throw(ArgumentError(
        "power-cone gradient is non-finite",
    ))
    return result
end

function power_primal_gradient!(gradient, x, y, z, alpha)
    _require_dense3_vector(gradient, "gradient")
    gx, gy, gz = power_barrier_gradient(x, y, z, alpha)
    gradient[1] = gx
    gradient[2] = gy
    gradient[3] = gz
    return gradient
end

"""
    power_barrier_hessian(x, y, z, alpha) -> Matrix

Hessian of the power-cone barrier (3×3 symmetric).
"""
function power_barrier_hessian(x, y, z, alpha)
    T = promote_type(typeof(x), typeof(y), typeof(z))
    hessian = Matrix{T}(undef, 3, 3)
    return power_primal_hessian!(hessian, x, y, z, alpha)
end

"""
    power_dual_membership(u, v, w, alpha; tol=0) -> Bool

Whether `(u, v, w)` lies in the dual power cone of parameter `alpha`:
    K_pow(alpha)* = { (u, v, w) : (u/alpha)^alpha * (v/(1-alpha))^(1-alpha) >= |w|,
                      u >= 0, v >= 0 }.
"""
function power_dual_membership(u, v, w, alpha; tol=zero(u))
    _power_finite3(u, v, w) && isfinite(tol) && tol >= zero(tol) || return false
    a, b = try
        _power_alpha(alpha, u, v, w)
    catch
        return false
    end
    (u < -tol || v < -tol) && return false
    abs(w) <= tol && return true
    uc = u < zero(u) ? zero(u) : u
    vc = v < zero(v) ? zero(v) : v
    (iszero(uc) || iszero(vc)) && return false
    required = abs(w) - tol
    log_dual_width = a * (log(uc) - log(a)) +
                     b * (log(vc) - log(b))
    log_required = log(required)
    tolerance = _nonsymmetric_log_tolerance(log_dual_width, log_required)
    return !isnan(log_dual_width) &&
           log_dual_width - log_required >= -tolerance
end

"""
    power_dual_interior(u, v, w, alpha) -> Bool

Whether `(u,v,w)` lies in the strict dual power-cone interior.
"""
function power_dual_interior(u, v, w, alpha)
    _power_finite3(u, v, w) || return false
    a, b = try
        _power_alpha(alpha, u, v, w)
    catch
        return false
    end
    return power_primal_interior(u / a, v / b, w, a)
end

"""
    power_barrier_hessian!(H, x, y, z, alpha)

In-place 3×3 Hessian of the power-cone barrier at interior point `(x, y, z)`.
"""
function power_primal_hessian!(hessian, x, y, z, alpha)
    _require_dense3_matrix(hessian, "hessian")
    a, b, log_w, rho, delta = _power_primal_terms(x, y, z, alpha)
    inv_delta = inv(delta)
    inv_delta2 = inv_delta * inv_delta
    inv_w = exp(-log_w)
    inv_x = inv(x)
    inv_y = inv(y)
    two_a = a + a
    two_b = b + b
    h11 = ((two_a * two_a) * inv_delta2 -
           two_a * (two_a - one(a)) * inv_delta + b) * inv_x * inv_x
    h12 = ((a + a + a + a) * b) *
          (inv_delta2 - inv_delta) * inv_x * inv_y
    h13 = -(a + a + a + a) * rho * inv_w *
          inv_delta2 * inv_x
    h22 = ((two_b * two_b) * inv_delta2 -
           two_b * (two_b - one(b)) * inv_delta + a) * inv_y * inv_y
    h23 = -(b + b + b + b) * rho * inv_w *
          inv_delta2 * inv_y
    h33 = ((rho + rho) * (rho + rho) * inv_delta2 +
           (one(delta) + one(delta)) * inv_delta) * inv_w * inv_w
    all(isfinite, (h11, h12, h13, h22, h23, h33)) || throw(ArgumentError(
        "power-cone Hessian is non-finite",
    ))
    hessian[1, 1] = h11
    hessian[1, 2] = h12
    hessian[1, 3] = h13
    hessian[2, 1] = h12
    hessian[2, 2] = h22
    hessian[2, 3] = h23
    hessian[3, 1] = h13
    hessian[3, 2] = h23
    hessian[3, 3] = h33
    return hessian
end

power_barrier_hessian!(hessian, x, y, z, alpha) =
    power_primal_hessian!(hessian, x, y, z, alpha)

"""The degree-3 dual barrier `F_primal(u/alpha,v/(1-alpha),w)`."""
function power_dual_barrier(u, v, w, alpha)
    a, b = _power_alpha(alpha, u, v, w)
    return power_primal_barrier(u / a, v / b, w, a)
end

function power_dual_gradient(u, v, w, alpha)
    a, b = _power_alpha(alpha, u, v, w)
    gx, gy, gz = power_barrier_gradient(u / a, v / b, w, a)
    return (gx / a, gy / b, gz)
end

function power_dual_gradient!(gradient, u, v, w, alpha)
    _require_dense3_vector(gradient, "gradient")
    gu, gv, gw = power_dual_gradient(u, v, w, alpha)
    gradient[1] = gu
    gradient[2] = gv
    gradient[3] = gw
    return gradient
end

function power_dual_hessian!(hessian, u, v, w, alpha)
    _require_dense3_matrix(hessian, "hessian")
    a, b = _power_alpha(alpha, u, v, w)
    power_primal_hessian!(hessian, u / a, v / b, w, a)
    hessian[1, 1] /= a * a
    hessian[1, 2] /= a * b
    hessian[2, 1] = hessian[1, 2]
    hessian[1, 3] /= a
    hessian[3, 1] = hessian[1, 3]
    hessian[2, 2] /= b * b
    hessian[2, 3] /= b
    hessian[3, 2] = hessian[2, 3]
    return hessian
end

function power_dual_hessian(u, v, w, alpha)
    T = promote_type(typeof(u), typeof(v), typeof(w))
    hessian = Matrix{T}(undef, 3, 3)
    return power_dual_hessian!(hessian, u, v, w, alpha)
end

function power_primal_hessian_product!(
    destination, x, y, z, alpha, vector, hessian,
)
    power_primal_hessian!(hessian, x, y, z, alpha)
    return nonsymmetric_hessian_product!(destination, hessian, vector)
end

function power_dual_hessian_product!(
    destination, u, v, w, alpha, vector, hessian,
)
    power_dual_hessian!(hessian, u, v, w, alpha)
    return nonsymmetric_hessian_product!(destination, hessian, vector)
end

function power_primal_hessian_solve!(
    destination, x, y, z, alpha, rhs, hessian, cholesky_storage,
)
    power_primal_hessian!(hessian, x, y, z, alpha)
    return nonsymmetric_hessian_solve!(
        destination, hessian, rhs, cholesky_storage,
    )
end

function power_dual_hessian_solve!(
    destination, u, v, w, alpha, rhs, hessian, cholesky_storage,
)
    power_dual_hessian!(hessian, u, v, w, alpha)
    return nonsymmetric_hessian_solve!(
        destination, hessian, rhs, cholesky_storage,
    )
end
