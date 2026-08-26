#=====================================================================#
#    Exponential-cone primal/dual degree-3 barrier oracle.
#
#    K_exp = { (x, y, z) : y * exp(x / y) <= z, y > 0 }
#           ∪ { (x, 0, z) : x <= 0, z >= 0 }
#
#    Primal 3-logarithmically-homogeneous self-concordant barrier:
#        f(x, y, z) = -log(z - y*exp(x/y)) - log(y) - log(z)
#
#    Evaluation uses rho = exp(log(y)+x/y-log(z)) and delta = 1-rho,
#    avoiding direct formation of exp(x/y).  The exact map
#        L(u,v,w) = (u-v,-u,w)
#    sends the dual exponential cone to the primal cone, so the dual
#    barrier and derivatives are F o L, L'grad(F), and L'H L.
#=====================================================================#

const EXPONENTIAL_BARRIER_DEGREE = 3

@inline _exp_finite3(x, y, z) = isfinite(x) && isfinite(y) && isfinite(z)

@inline function _exp_primal_terms(x, y, z)
    _exp_finite3(x, y, z) || throw(ArgumentError(
        "exponential-cone oracle requires finite coordinates",
    ))
    y > zero(y) && z > zero(z) || throw(ArgumentError(
        "exponential-cone oracle requires y > 0 and z > 0",
    ))
    t = x / y
    log_ratio = log(y) + t - log(z)
    margin_tolerance = _nonsymmetric_log_tolerance(log_ratio, zero(log_ratio))
    isfinite(t) && isfinite(log_ratio) && log_ratio < -margin_tolerance ||
        throw(ArgumentError(
            "exponential-cone oracle requires a resolvable y*exp(x/y) < z gap",
        ))
    rho = exp(log_ratio)
    delta = one(rho) - rho
    isfinite(rho) && delta > zero(delta) || throw(ArgumentError(
        "exponential-cone oracle could not resolve a strict interior gap",
    ))
    return t, rho, delta
end

"""
    exp_membership(x, y, z) -> Bool

Whether `(x, y, z)` lies in the exponential cone (including the
limit face `(x, 0, z)` with `x <= 0, z >= 0`).
"""
function exp_primal_membership(x, y, z)
    _exp_finite3(x, y, z) || return false
    if y > zero(y)
        z > zero(z) || return false
        log_ratio = log(y) + x / y - log(z)
        tolerance = _nonsymmetric_log_tolerance(log_ratio, zero(log_ratio))
        return !isnan(log_ratio) && log_ratio <= tolerance
    end
    # Limit face: x <= 0, y == 0, z >= 0.
    return iszero(y) && x <= zero(x) && z >= zero(z)
end

exp_membership(x, y, z) = exp_primal_membership(x, y, z)

"""Whether `(x,y,z)` lies in the strict primal exponential-cone interior."""
function exp_primal_interior(x, y, z)
    _exp_finite3(x, y, z) || return false
    y > zero(y) && z > zero(z) || return false
    log_ratio = log(y) + x / y - log(z)
    tolerance = _nonsymmetric_log_tolerance(log_ratio, zero(log_ratio))
    return !isnan(log_ratio) && log_ratio < -tolerance
end

"""
    exp_barrier(x, y, z)

Value of the self-concordant barrier of the exponential cone at an
interior point `(x, y, z)` (i.e. `y > 0` and `y*exp(x/y) < z`).
"""
function exp_primal_barrier(x, y, z)
    _, rho, _ = _exp_primal_terms(x, y, z)
    return -log1p(-rho) - log(y) - (one(z) + one(z)) * log(z)
end

exp_barrier(x, y, z) = exp_primal_barrier(x, y, z)

"""
    exp_barrier_gradient(x, y, z) -> (gx, gy, gz)

Gradient of the exponential-cone barrier.
"""
function exp_barrier_gradient(x, y, z)
    t, rho, delta = _exp_primal_terms(x, y, z)
    c = rho / delta
    inv_y = inv(y)
    inv_z = inv(z)
    result = (
        c * inv_y,
        (c * (one(t) - t) - one(t)) * inv_y,
        -(inv(delta) + one(delta)) * inv_z,
    )
    all(isfinite, result) || throw(ArgumentError(
        "exponential-cone gradient is non-finite",
    ))
    return result
end

function exp_primal_gradient!(gradient, x, y, z)
    _require_dense3_vector(gradient, "gradient")
    gx, gy, gz = exp_barrier_gradient(x, y, z)
    gradient[1] = gx
    gradient[2] = gy
    gradient[3] = gz
    return gradient
end

"""
    exp_barrier_hessian(x, y, z) -> Matrix

Hessian of the exponential-cone barrier (3×3 symmetric).
"""
function exp_barrier_hessian(x, y, z)
    T = promote_type(typeof(x), typeof(y), typeof(z))
    hessian = Matrix{T}(undef, 3, 3)
    return exp_primal_hessian!(hessian, x, y, z)
end

"""
    exp_dual_membership(u, v, w; tol=0) -> Bool

Whether `(u, v, w)` lies in the dual exponential cone:
    K_exp* = { (u, v, w) : -u * exp(v / u - 1) <= w, u < 0 }
           ∪ { (0, v, w) : v >= 0, w >= 0 }
"""
function exp_dual_membership(u, v, w; tol=zero(u))
    _exp_finite3(u, v, w) && isfinite(tol) && tol >= zero(tol) || return false
    if u < -tol
        rhs = w + tol
        rhs > zero(rhs) || return false
        log_lhs = log(-u) + v / u - one(u)
        log_rhs = log(rhs)
        tolerance = _nonsymmetric_log_tolerance(log_lhs, log_rhs)
        return !isnan(log_lhs) && log_lhs - log_rhs <= tolerance
    elseif abs(u) <= tol
        return v >= -tol && w >= -tol
    end
    return false
end

"""
    exp_dual_interior(u, v, w) -> Bool

Whether `(u,v,w)` lies in the strict dual exponential-cone interior.
"""
function exp_dual_interior(u, v, w)
    _exp_finite3(u, v, w) || return false
    return exp_primal_interior(u - v, -u, w)
end

"""
    exp_barrier_hessian!(H, x, y, z)

In-place 3×3 Hessian of the exponential-cone barrier at interior point `(x, y, z)`.
"""
function exp_primal_hessian!(hessian, x, y, z)
    _require_dense3_matrix(hessian, "hessian")
    t, rho, delta = _exp_primal_terms(x, y, z)
    c = rho / delta
    c2 = c * c
    inv_y = inv(y)
    inv_z = inv(z)
    inv_y2 = inv_y * inv_y
    inv_z2 = inv_z * inv_z
    inv_delta = inv(delta)
    omt = one(t) - t
    h11 = (c + c2) * inv_y2
    h12 = (-c * t + c2 * omt) * inv_y2
    h13 = -c * inv_delta * inv_y * inv_z
    h22 = (c * t * t + c2 * omt * omt + one(t)) * inv_y2
    h23 = -c * omt * inv_delta * inv_y * inv_z
    h33 = (inv_delta * inv_delta + one(delta)) * inv_z2
    all(isfinite, (h11, h12, h13, h22, h23, h33)) || throw(ArgumentError(
        "exponential-cone Hessian is non-finite",
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

exp_barrier_hessian!(hessian, x, y, z) =
    exp_primal_hessian!(hessian, x, y, z)

"""The degree-3 dual barrier `F_primal(u-v,-u,w)`."""
exp_dual_barrier(u, v, w) = exp_primal_barrier(u - v, -u, w)

function exp_dual_gradient(u, v, w)
    gx, gy, gz = exp_barrier_gradient(u - v, -u, w)
    return (gx - gy, -gx, gz)
end

function exp_dual_gradient!(gradient, u, v, w)
    _require_dense3_vector(gradient, "gradient")
    gu, gv, gw = exp_dual_gradient(u, v, w)
    gradient[1] = gu
    gradient[2] = gv
    gradient[3] = gw
    return gradient
end

function exp_dual_hessian!(hessian, u, v, w)
    _require_dense3_matrix(hessian, "hessian")
    exp_primal_hessian!(hessian, u - v, -u, w)
    hxx = hessian[1, 1]
    hxy = hessian[1, 2]
    hxz = hessian[1, 3]
    hyy = hessian[2, 2]
    hyz = hessian[2, 3]
    hzz = hessian[3, 3]
    h11 = hxx - (hxy + hxy) + hyy
    h12 = hxy - hxx
    h13 = hxz - hyz
    h22 = hxx
    h23 = -hxz
    hessian[1, 1] = h11
    hessian[1, 2] = h12
    hessian[1, 3] = h13
    hessian[2, 1] = h12
    hessian[2, 2] = h22
    hessian[2, 3] = h23
    hessian[3, 1] = h13
    hessian[3, 2] = h23
    hessian[3, 3] = hzz
    return hessian
end

function exp_dual_hessian(u, v, w)
    T = promote_type(typeof(u), typeof(v), typeof(w))
    hessian = Matrix{T}(undef, 3, 3)
    return exp_dual_hessian!(hessian, u, v, w)
end

function exp_primal_hessian_product!(destination, x, y, z, vector, hessian)
    exp_primal_hessian!(hessian, x, y, z)
    return nonsymmetric_hessian_product!(destination, hessian, vector)
end


function exp_dual_hessian_product!(destination, u, v, w, vector, hessian)
    exp_dual_hessian!(hessian, u, v, w)
    return nonsymmetric_hessian_product!(destination, hessian, vector)
end

function exp_primal_hessian_solve!(
    destination, x, y, z, rhs, hessian, cholesky_storage,
)
    exp_primal_hessian!(hessian, x, y, z)
    return nonsymmetric_hessian_solve!(
        destination, hessian, rhs, cholesky_storage,
    )
end

function exp_dual_hessian_solve!(
    destination, u, v, w, rhs, hessian, cholesky_storage,
)
    exp_dual_hessian!(hessian, u, v, w)
    return nonsymmetric_hessian_solve!(
        destination, hessian, rhs, cholesky_storage,
    )
end
