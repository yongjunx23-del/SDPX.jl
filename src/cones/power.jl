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
    power_dual_membership(u, v, w, alpha; tol=0) -> Bool

Whether `(u, v, w)` lies in the dual power cone of parameter `alpha`:
    K_pow(alpha)* = { (u, v, w) : (u/alpha)^alpha * (v/(1-alpha))^(1-alpha) >= |w|,
                      u >= 0, v >= 0 }.
"""
function power_dual_membership(u, v, w, alpha; tol=zero(u))
    T = typeof(u)
    (u < -tol || v < -tol) && return false
    uc = u < zero(T) ? zero(T) : u
    vc = v < zero(T) ? zero(T) : v
    if iszero(uc) && iszero(vc)
        return abs(w) <= tol
    end
    a = convert(T, alpha)
    oma = one(T) - a
    w_dual = (uc / a)^a * (vc / oma)^oma
    return w_dual + tol >= abs(w)
end

"""
    power_scaling_point(x, y, z, alpha) -> (sx, sy, sz)

A scaling point for the power cone at an interior point: the gradient
of the barrier (used as the dual scaling in the primal-dual method).
"""
power_scaling_point(x, y, z, alpha) = power_barrier_gradient(x, y, z, alpha)

"""
    power_barrier_hessian!(H, x, y, z, alpha)

In-place 3×3 Hessian of the power-cone barrier at interior point `(x, y, z)`.
"""
function power_barrier_hessian!(H::AbstractMatrix{T}, x::T, y::T, z::T, alpha) where {T}
    a = convert(T, alpha)
    w = x^a * y^(one(T) - a)
    w > abs(z) || throw(ArgumentError("power_barrier_hessian! requires x^alpha y^(1-alpha) > |z|"))
    A = w - z
    B = w + z
    wx = a * w / x
    wy = (one(T) - a) * w / y
    wxx = a * (a - one(T)) * w / (x * x)
    wyy = (one(T) - a) * (-a) * w / (y * y)
    wxy = a * (one(T) - a) * w / (x * y)
    h11 = -wxx / A + wx * wx / (A * A) - wxx / B + wx * wx / (B * B) + one(T) / (x * x)
    h12 = -wxy / A + wx * wy / (A * A) - wxy / B + wx * wy / (B * B)
    inv_sq_diff = one(T) / (A * A) - one(T) / (B * B)
    h13 = -wx * inv_sq_diff
    h22 = -wyy / A + wy * wy / (A * A) - wyy / B + wy * wy / (B * B) + one(T) / (y * y)
    h23 = -wy * inv_sq_diff
    h33 = one(T) / (A * A) + one(T) / (B * B)
    H[1, 1] = h11; H[1, 2] = h12; H[1, 3] = h13
    H[2, 1] = h12; H[2, 2] = h22; H[2, 3] = h23
    H[3, 1] = h13; H[3, 2] = h23; H[3, 3] = h33
    return H
end
