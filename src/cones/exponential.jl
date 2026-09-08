#=====================================================================#
#    Exponential-cone membership and public degree-3 barrier interface.
#
#    Membership uses the logarithmic cone inequality without forming exp(x/y).
#    The production barrier, gradient and Hessian below delegate to the
#    proved logarithmic LHSCB in exp_logarithmic.jl.  The linear dual map is
#    retained only for the public pulled-back dual-oracle compatibility API;
#    nonsymmetric scaling uses the actual Fenchel inverse.
#=====================================================================#

const EXPONENTIAL_BARRIER_DEGREE = 3

@inline _exp_finite3(x, y, z) = isfinite(x) && isfinite(y) && isfinite(z)

@inline _nonsymmetric_stable_fma(a::Float16,b::Float16,c::Float16) = fma(a,b,c)
@inline _nonsymmetric_stable_fma(a::Float32,b::Float32,c::Float32) = fma(a,b,c)
@inline _nonsymmetric_stable_fma(a::Float64,b::Float64,c::Float64) = fma(a,b,c)
@inline _nonsymmetric_stable_fma(a::BigFloat,b::BigFloat,c::BigFloat) = fma(a,b,c)
@inline _nonsymmetric_stable_fma(a,b,c) = a*b+c

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

# Subtraction can destroy relative information when numerator << denominator:
# e.g. Float64 numerator=3*2^-54, denominator=1 produces -1+2^-52, so log1p
# returns log(2^-52), not log(3*2^-54). Keep the computed argument away from -1;
# the bounds are exact binary constants, approximately operand ratios [1/2,2].
@inline _nonsymmetric_log_ratio_uses_log1p(relative) =
    isfinite(relative) && oftype(relative, -0.5) <= relative <= one(relative)

@inline function _nonsymmetric_positive_log_ratio(numerator, denominator)
    relative = (numerator - denominator) / denominator
    if _nonsymmetric_log_ratio_uses_log1p(relative)
        return _nonsymmetric_stable_log1p(relative)
    end
    return log(numerator) - log(denominator)
end

# Value and arithmetic-work accounting deliberately use the same branch.
# `abs(value)` alone misses cancellation. The first branch records the work
# in `(numerator-denominator)/denominator`; the second records the two logs
# whose difference is returned. These quantities alone are NOT a rigorous
# transcendental-error or complete Phi-enclosure certificate.
@inline function _nonsymmetric_positive_log_ratio_terms(
    numerator, denominator,
)
    relative = (numerator - denominator) / denominator
    if _nonsymmetric_log_ratio_uses_log1p(relative)
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

"""The degree-three logarithmic barrier on the primal exponential cone."""
function exp_primal_barrier(x, y, z)
    return exp_logarithmic_barrier((x, y, z))
end

exp_barrier(x, y, z) = exp_primal_barrier(x, y, z)

"""
    exp_barrier_gradient(x, y, z) -> (gx, gy, gz)

Gradient of the exponential-cone barrier.
"""
function exp_barrier_gradient(x, y, z)
    return _exp_logarithmic_gradient_values((x, y, z))
end

function exp_primal_gradient!(gradient, x, y, z)
    _require_dense3_vector(gradient, "gradient")
    values = _exp_logarithmic_gradient_values((x, y, z))
    for i in 1:3
        _owned_setindex!(gradient, i, values[i])
    end
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

In-place 3×3 Hessian of the logarithmic exponential-cone barrier at interior point `(x, y, z)`.
"""

function exp_primal_hessian!(hessian, x, y, z)
    _require_dense3_matrix(hessian, "hessian")
    return exp_logarithmic_hessian!(hessian, (x, y, z))
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
    _owned_setindex!(gradient, 1, gu)
    _owned_setindex!(gradient, 2, gv)
    _owned_setindex!(gradient, 3, gw)
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
    _owned_setindex!(hessian, CartesianIndex(1, 1), h11)
    _owned_setindex!(hessian, CartesianIndex(1, 2), h12)
    _owned_setindex!(hessian, CartesianIndex(1, 3), h13)
    _owned_setindex!(hessian, CartesianIndex(2, 1), h12)
    _owned_setindex!(hessian, CartesianIndex(2, 2), h22)
    _owned_setindex!(hessian, CartesianIndex(2, 3), h23)
    _owned_setindex!(hessian, CartesianIndex(3, 1), h13)
    _owned_setindex!(hessian, CartesianIndex(3, 2), h23)
    _owned_setindex!(hessian, CartesianIndex(3, 3), hzz)
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
