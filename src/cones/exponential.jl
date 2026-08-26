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

# MultiFloats deliberately does not provide `Base.expm1`.  The generic branch
# therefore evaluates the cancellation-sensitive small-argument regime with a
# type-native Taylor recurrence and uses `exp(x)-1` only away from zero.  The
# fixed bound is a fail-safe; normal convergence stops when the target type can
# no longer distinguish the next partial sum.  All state is scalar/isbits for
# fixed-width arithmetic.
@inline _nonsymmetric_stable_expm1(x::Float16) = Base.expm1(x)
@inline _nonsymmetric_stable_expm1(x::Float32) = Base.expm1(x)
@inline _nonsymmetric_stable_expm1(x::Float64) = Base.expm1(x)
@inline _nonsymmetric_stable_expm1(x::BigFloat) = Base.expm1(x)
@inline function _nonsymmetric_stable_expm1(x)
    iszero(x) && return zero(x)
    abs(x) > inv(one(x) + one(x)) && return exp(x) - one(x)
    term = x
    total = x
    @inbounds for order in 2:512
        term *= x / typeof(x)(order)
        next_total = total + term
        next_total == total && return total
        total = next_total
    end
    return total
end

@inline _nonsymmetric_stable_log1p(x::Float16) = Base.log1p(x)
@inline _nonsymmetric_stable_log1p(x::Float32) = Base.log1p(x)
@inline _nonsymmetric_stable_log1p(x::Float64) = Base.log1p(x)
@inline _nonsymmetric_stable_log1p(x::BigFloat) = Base.log1p(x)
@inline function _nonsymmetric_stable_log1p(x)
    iszero(x) && return zero(x)
    abs(x) > inv(one(x) + one(x)) && return log(one(x) + x)
    term = x
    total = x
    @inbounds for order in 2:1024
        term *= -x * typeof(x)(order - 1) / typeof(x)(order)
        next_total = total + term
        next_total == total && return total
        total = next_total
    end
    return total
end

@inline function _nonsymmetric_positive_log_ratio(numerator, denominator)
    relative = (numerator - denominator) / denominator
    if isfinite(relative) && relative > -one(relative)
        return _nonsymmetric_stable_log1p(relative)
    end
    return log(numerator) - log(denominator)
end

# The value and its arithmetic-work certificate deliberately travel together.
# A small logarithm can be the result of subtracting two O(1) positive
# quantities, so `abs(value)` alone is not a backward-error scale.  The first
# branch records the work in `(numerator-denominator)/denominator`; the second
# records the two logarithms whose difference is returned.
@inline function _nonsymmetric_positive_log_ratio_terms(
    numerator, denominator,
)
    relative = (numerator - denominator) / denominator
    if isfinite(relative) && relative > -one(relative)
        value = _nonsymmetric_stable_log1p(relative)
        ratio = numerator / denominator
        arithmetic_work = abs(ratio) + one(ratio)
        kernel_work = abs(relative) + abs(value)
        return value, arithmetic_work, kernel_work
    end
    log_numerator = log(numerator)
    log_denominator = log(denominator)
    value = log_numerator - log_denominator
    kernel_work = abs(log_numerator) + abs(log_denominator) + abs(value)
    return value, zero(value), kernel_work
end

@inline function _nonsymmetric_positive_log_ratio_with_work(
    numerator, denominator,
)
    value, arithmetic_work, kernel_work =
        _nonsymmetric_positive_log_ratio_terms(numerator, denominator)
    return value, arithmetic_work + kernel_work
end

@inline _exp_log_ratio(x, y, z) =
    x / y + _nonsymmetric_positive_log_ratio(y, z)

@inline function _exp_dual_centered_ratio_terms(u, v)
    difference = v - u
    if isfinite(difference)
        centered = difference / u
        # `abs(v / u) + 1` is the finite-safe evaluation of
        # `(abs(v) + abs(u)) / abs(u)`.  It retains the subtraction work when
        # `v` and `u` nearly cancel without first overflowing the numerator.
        quotient = v / u
        work = abs(quotient) + one(u) + abs(centered)
        return centered, work
    end

    # `v-u` can overflow even though the algebraically equivalent quotient is
    # representable (for example, opposite-sign coordinates near floatmax).
    quotient = v / u
    centered = quotient - one(u)
    work = abs(quotient) + one(u) + abs(centered)
    return centered, work
end
@inline _exp_dual_centered_ratio_with_work(u, v) =
    _exp_dual_centered_ratio_terms(u, v)

@inline function _exp_dual_log_ratio_terms(u, v, w)
    centered, centered_work = _exp_dual_centered_ratio_terms(u, v)
    log_wu, log_arithmetic_work, log_kernel_work =
        _nonsymmetric_positive_log_ratio_terms(w, -u)
    value = centered - log_wu
    arithmetic_work = centered_work + log_arithmetic_work +
                      abs(centered) + abs(log_wu)
    return value, arithmetic_work, log_kernel_work
end

@inline function _exp_dual_log_ratio_with_work(u, v, w)
    value, arithmetic_work, kernel_work =
        _exp_dual_log_ratio_terms(u, v, w)
    return value, arithmetic_work + kernel_work
end

@inline function _exp_dual_log_ratio(u, v, w)
    value, _ = _exp_dual_log_ratio_with_work(u, v, w)
    return value
end

@inline function _exp_primal_terms(x, y, z)
    _exp_finite3(x, y, z) || throw(ArgumentError(
        "exponential-cone oracle requires finite coordinates",
    ))
    y > zero(y) && z > zero(z) || throw(ArgumentError(
        "exponential-cone oracle requires y > 0 and z > 0",
    ))
    t = x / y
    log_ratio = _exp_log_ratio(x, y, z)
    margin_tolerance = _nonsymmetric_log_tolerance(log_ratio, zero(log_ratio))
    isfinite(t) && isfinite(log_ratio) && log_ratio < -margin_tolerance ||
        throw(ArgumentError(
            "exponential-cone oracle requires a resolvable y*exp(x/y) < z gap",
        ))
    rho = exp(log_ratio)
    delta = -_nonsymmetric_stable_expm1(log_ratio)
    isfinite(rho) && delta > zero(delta) || throw(ArgumentError(
        "exponential-cone oracle could not resolve a strict interior gap",
    ))
    return t, rho, delta
end

"""
    exp_primal_residual(x, y, z)

Degree-one violation of the closed exponential cone.  The curved branch uses
`y * log(y*exp(x/y)/z)`, so the residual has the same units and homogeneity as
the cone coordinates while avoiding overflow in `exp(x/y)`.  It is zero on
the cone and finite/strictly positive for every finite point outside it.
"""
function exp_primal_residual(x, y, z)
    T = promote_type(typeof(float(x)), typeof(float(y)), typeof(float(z)))
    infinity = T(Inf)
    _exp_finite3(x, y, z) || return infinity
    zero_value = zero(T)
    if y > zero_value && z > zero_value
        log_ratio = _exp_log_ratio(x, y, z)
        isfinite(log_ratio) || return infinity
        margin = y * log_ratio
        isfinite(margin) || return infinity
        return max(zero_value, margin)
    end
    # Outside the curved branch, the nearest possible closed-cone branch is
    # the limit face y=0, x<=0, z>=0.  Including x here is essential: a point
    # with tiny y,z but large positive x is not tolerance-close to that face.
    return max(zero_value, abs(y), x, -z)
end

"""
    exp_primal_membership(x, y, z; tol=0) -> Bool

Whether `(x, y, z)` lies in the exponential cone (including the limit face
`(x, 0, z)` with `x <= 0, z >= 0`).  `tol` is an optional absolute,
degree-one certificate tolerance.  Arithmetic roundoff at the logarithmic
boundary is accounted for separately and cannot mask a macroscopic cone
violation.
"""
function exp_primal_membership(x, y, z; tol=zero(x))
    T = promote_type(
        typeof(float(x)), typeof(float(y)), typeof(float(z)),
        typeof(float(tol)),
    )
    tolerance = convert(T, tol)
    _exp_finite3(x, y, z) && isfinite(tolerance) &&
        tolerance >= zero(T) || return false
    residual = exp_primal_residual(x, y, z)
    isfinite(residual) || return false
    allowance = tolerance
    if y > zero(T) && z > zero(T)
        log_ratio = _exp_log_ratio(x, y, z)
        roundoff = abs(y) *
            _nonsymmetric_log_tolerance(log_ratio, zero(log_ratio))
        isfinite(roundoff) || return false
        allowance += roundoff
    end
    return isfinite(allowance) && residual <= allowance
end

exp_membership(x, y, z; tol=zero(x)) =
    exp_primal_membership(x, y, z; tol=tol)

"""Whether `(x,y,z)` lies in the strict primal exponential-cone interior."""
function exp_primal_interior(x, y, z)
    _exp_finite3(x, y, z) || return false
    y > zero(y) && z > zero(z) || return false
    log_ratio = _exp_log_ratio(x, y, z)
    tolerance = _nonsymmetric_log_tolerance(log_ratio, zero(log_ratio))
    return !isnan(log_ratio) && log_ratio < -tolerance
end

"""
    exp_barrier(x, y, z)

Value of the self-concordant barrier of the exponential cone at an
interior point `(x, y, z)` (i.e. `y > 0` and `y*exp(x/y) < z`).
"""
function exp_primal_barrier(x, y, z)
    _, _, delta = _exp_primal_terms(x, y, z)
    return -log(delta) - log(y) - (one(z) + one(z)) * log(z)
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
        log_ratio = _exp_dual_log_ratio(u, v, rhs)
        tolerance = _nonsymmetric_log_tolerance(log_ratio, zero(log_ratio))
        return !isnan(log_ratio) && log_ratio <= tolerance
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
@inline function _exp_primal_hessian_from_terms!(
    hessian, t, rho, delta, y, z,
)
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

function exp_primal_hessian!(hessian, x, y, z)
    _require_dense3_matrix(hessian, "hessian")
    t, rho, delta = _exp_primal_terms(x, y, z)
    return _exp_primal_hessian_from_terms!(
        hessian, t, rho, delta, y, z,
    )
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
