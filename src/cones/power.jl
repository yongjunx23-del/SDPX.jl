#=====================================================================#
#    Power cone primitives (Subagent E).
#
#    K_pow(alpha) = { (x, y, z) : x^alpha y^(1-alpha) >= |z|,
#                     x >= 0, y >= 0 },  alpha in (0, 1).
#
#    Smooth self-concordant barrier (handles both z >= 0 and z <= 0):
#        f(x, y, z) = -log(w - z) - log(w + z) - log(x) - log(y),
#        w = x^alpha y^(1-alpha).
#
#    Generic in the element type (Float64, Float64x2, Float64x4,
#    BigFloat, ...).
#=====================================================================#

"""
    power_membership(x, y, z, alpha) -> Bool

Whether `(x, y, z)` lies in the power cone of parameter `alpha`.
"""
function power_membership(x, y, z, alpha)
    x >= zero(x) && y >= zero(y) || return false
    if iszero(x) && iszero(y)
        return iszero(z)
    end
    w = x^alpha * y^(one(alpha) - alpha)
    return w >= abs(z)
end

"""
    power_barrier(x, y, z, alpha)

Value of the smooth self-concordant barrier of the power cone at an
interior point.
"""
function power_barrier(x, y, z, alpha)
    w = x^alpha * y^(one(alpha) - alpha)
    w > abs(z) || throw(ArgumentError("power_barrier requires x^alpha y^(1-alpha) > |z|"))
    return -log(w - z) - log(w + z) - log(x) - log(y)
end

"""
    power_barrier_gradient(x, y, z, alpha) -> (gx, gy, gz)

Gradient of the power-cone barrier.
"""
function power_barrier_gradient(x, y, z, alpha)
    w = x^alpha * y^(one(alpha) - alpha)
    w > abs(z) || throw(ArgumentError("power_barrier_gradient requires x^alpha y^(1-alpha) > |z|"))
    A = w - z
    B = w + z
    wx = alpha * w / x
    wy = (one(alpha) - alpha) * w / y
    gx = -wx / A - wx / B - one(x) / x
    gy = -wy / A - wy / B - one(y) / y
    gz = one(z) / A - one(z) / B
    return (gx, gy, gz)
end

"""
    power_barrier_hessian(x, y, z, alpha) -> Matrix

Hessian of the power-cone barrier (3×3 symmetric).
"""
function power_barrier_hessian(x, y, z, alpha)
    w = x^alpha * y^(one(alpha) - alpha)
    w > abs(z) || throw(ArgumentError("power_barrier_hessian requires x^alpha y^(1-alpha) > |z|"))
    A = w - z
    B = w + z
    wx = alpha * w / x
    wy = (one(alpha) - alpha) * w / y
    wxx = alpha * (alpha - one(alpha)) * w / (x * x)
    wyy = (one(alpha) - alpha) * (-alpha) * w / (y * y)
    wxy = alpha * (one(alpha) - alpha) * w / (x * y)
    h11 = -wxx / A + wx * wx / (A * A) - wxx / B + wx * wx / (B * B) + one(x) / (x * x)
    h12 = -wxy / A + wx * wy / (A * A) - wxy / B + wx * wy / (B * B)
    inv_sq_diff = one(z) / (A * A) - one(z) / (B * B)
    h13 = -wx * inv_sq_diff
    h22 = -wyy / A + wy * wy / (A * A) - wyy / B + wy * wy / (B * B) + one(y) / (y * y)
    h23 = -wy * inv_sq_diff
    h33 = one(z) / (A * A) + one(z) / (B * B)
    return [h11 h12 h13; h12 h22 h23; h13 h23 h33]
end

"""
    power_scaling_point(x, y, z, alpha) -> (sx, sy, sz)

A scaling point for the power cone at an interior point: the gradient
of the barrier (used as the dual scaling in the primal-dual method).
"""
power_scaling_point(x, y, z, alpha) = power_barrier_gradient(x, y, z, alpha)
