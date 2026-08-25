#=====================================================================#
#    Exponential cone primitives (Subagent E).
#
#    K_exp = { (x, y, z) : y * exp(x / y) <= z, y > 0 }
#           ∪ { (0, y, z) : y >= 0, z >= 0 }
#
#    Self-concordant barrier:
#        f(x, y, z) = -log(z - y*exp(x/y)) - log(y) - log(z)
#
#    Let A = z - u, u = y*exp(x/y), e = exp(x/y).
#    u_x = e, u_y = e*(1 - x/y), u_xx = e/y,
#    u_xy = -e*x/y^2, u_yy = e*x^2/y^3.
#    f_x = u_x/A, f_y = u_y/A - 1/y, f_z = -1/A - 1/z.
#=====================================================================#

"""
    exp_membership(x, y, z) -> Bool

Whether `(x, y, z)` lies in the exponential cone (including the
limit face `(0, y, z)` with `y >= 0, z >= 0`).
"""
function exp_membership(x, y, z)
    if y > zero(y)
        return y * exp(x / y) <= z
    else
        # limit face: x == 0, y >= 0, z >= 0
        return iszero(x) && y >= zero(y) && z >= zero(z)
    end
end

"""
    exp_barrier(x, y, z)

Value of the self-concordant barrier of the exponential cone at an
interior point `(x, y, z)` (i.e. `y > 0` and `y*exp(x/y) < z`).
"""
function exp_barrier(x, y, z)
    u = y * exp(x / y)
    u < z || throw(ArgumentError("exp_barrier requires y*exp(x/y) < z"))
    return -log(z - u) - log(y) - log(z)
end

"""
    exp_barrier_gradient(x, y, z) -> (gx, gy, gz)

Gradient of the exponential-cone barrier.
"""
function exp_barrier_gradient(x, y, z)
    u = y * exp(x / y)
    u < z || throw(ArgumentError("exp_barrier_gradient requires y*exp(x/y) < z"))
    A = z - u
    e = exp(x / y)
    ux = e
    uy = e * (one(x) - x / y)
    gx = ux / A
    gy = uy / A - one(y) / y
    gz = -one(z) / A - one(z) / z
    return (gx, gy, gz)
end

"""
    exp_barrier_hessian(x, y, z) -> Matrix

Hessian of the exponential-cone barrier (3×3 symmetric).
"""
function exp_barrier_hessian(x, y, z)
    u = y * exp(x / y)
    u < z || throw(ArgumentError("exp_barrier_hessian requires y*exp(x/y) < z"))
    A = z - u
    e = exp(x / y)
    ux = e
    uy = e * (one(x) - x / y)
    uxx = e / y
    uxy = -e * x / (y * y)
    uyy = e * x * x / (y * y * y)
    A2 = A * A
    h11 = (uxx * A + ux * ux) / A2
    h12 = (uxy * A + ux * uy) / A2
    h13 = -ux / A2
    h22 = (uyy * A + uy * uy) / A2 + one(y) / (y * y)
    h23 = -uy / A2
    h33 = one(z) / A2 + one(z) / (z * z)
    return [h11 h12 h13; h12 h22 h23; h13 h23 h33]
end

"""
    exp_scaling_point(x, y, z) -> (sx, sy, sz)

A scaling point for the exponential cone at an interior point: the
gradient of the barrier (used as the dual scaling in the primal-dual
method).
"""
exp_scaling_point(x, y, z) = exp_barrier_gradient(x, y, z)
