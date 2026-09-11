"""
Allocation-conscious kernels for the standard Lorentz cone

    Q_q = {(t,u) : t >= norm(u)}.

All functions operate directly in Lorentz coordinates.  They neither build
the historical PSD arrow representation nor depend on a linear-algebra
provider.  Q3 is therefore the ordinary `q == 3` case; its 2x2 matrix image
remains a validation identity, not an execution representation.
"""

@inline function _soc_require_equal_dimensions(label::AbstractString, vectors...)
    isempty(vectors) && return nothing
    dimension = length(first(vectors))
    dimension > 0 || throw(DimensionMismatch("$label must be nonempty"))
    all(length(vector) == dimension for vector in vectors) ||
        throw(DimensionMismatch("$label must have equal dimensions"))
    return dimension
end

"""Stable tail norm used by margins, interior checks, and NT scaling."""
@inline function _soc_tail_norm(vector)
    scale = zero(eltype(vector))
    @inbounds for index in 2:length(vector)
        scale = max(scale, abs(vector[index]))
    end
    iszero(scale) && return scale
    sum_squares = zero(scale)
    @inbounds for index in 2:length(vector)
        value = vector[index] / scale
        sum_squares += value * value
    end
    return scale * sqrt(sum_squares)
end

@inline _soc_margin(vector) = vector[1] - _soc_tail_norm(vector)

"""
Return `(ok, sqrt(det(x)))` without squaring unscaled coordinates.
`ok=false` covers nonfinite values and either sheet outside the strict cone.
"""
function _soc_sqrt_determinant(vector)
    isempty(vector) && throw(DimensionMismatch("Lorentz vectors must be nonempty"))
    scale = zero(eltype(vector))
    @inbounds for value in vector
        isfinite(value) || return false, zero(value)
        scale = max(scale, abs(value))
    end
    iszero(scale) && return false, zero(scale)
    head_value = vector[1]
    tail_value = _soc_tail_norm(vector)
    margin = head_value - tail_value
    margin > zero(margin) || return false, zero(scale)
    # Form the small factor in the original scale before normalization.  If
    # an iterate is only one ulp inside the cone, separately normalizing head
    # and tail can round both to one and erase that representable margin.
    residual = (margin / scale) *
               (head_value / scale + tail_value / scale)
    residual > zero(residual) || return false, zero(scale)
    result = scale * sqrt(residual)
    return isfinite(result) && result > zero(result), result
end

"""
    _soc_nt_scaling!(w, lambda, s, z) -> (ok, eta, eta_squared)

Construct the Nesterov--Todd scaling point for arbitrary Lorentz dimension.
The implementation is the dimension-generic form of the established Q3
kernel and uses max-norm normalization before determinant calculations.
Outputs are committed only after every numerical check succeeds.
"""
function _soc_nt_scaling!(w, lambda, s, z)
    dimension = _soc_require_equal_dimensions(
        "Lorentz NT vectors", w, lambda, s, z,
    )
    (Base.mightalias(w, s) || Base.mightalias(w, z) ||
     Base.mightalias(lambda, s) || Base.mightalias(lambda, z) ||
     Base.mightalias(w, lambda)) && throw(ArgumentError(
        "Lorentz NT outputs must not alias inputs or each other",
    ))

    sok, sscale = _soc_sqrt_determinant(s)
    zok, zscale = _soc_sqrt_determinant(z)
    failure = (false, zero(sscale), zero(sscale))
    sok && zok || return failure

    eta = sqrt(sscale) / sqrt(zscale)
    eta_squared = eta * eta
    isfinite(eta_squared) && eta_squared > zero(eta_squared) || return failure

    T = promote_type(eltype(s), eltype(z))
    a = Vector{T}(undef, dimension)
    b = Vector{T}(undef, dimension)
    raw = Vector{T}(undef, dimension)
    @inbounds begin
        a[1] = s[1] / sscale
        b[1] = z[1] / zscale
        raw[1] = a[1] + b[1]
        for index in 2:dimension
            a[index] = s[index] / sscale
            b[index] = z[index] / zscale
            raw[index] = a[index] - b[index]
        end
    end
    all(isfinite, a) && all(isfinite, b) && all(isfinite, raw) || return failure
    # Since det(a)=det(b)=1, the raw midpoint has
    #
    #   det(a + J*b) = 2 * (1 + a0*b0 + dot(a_tail,b_tail)).
    #
    # Evaluating `head(raw)^2 - norm(tail(raw))^2` loses every useful bit
    # when both normalized cone points are close to the boundary.  Compute
    # the strictly positive cross term after a common scaling instead.  This
    # is algebraically identical, but it avoids a spurious NT failure near an
    # otherwise well-resolved optimal boundary point.
    midpoint_scale = one(T)
    @inbounds for index in 1:dimension
        midpoint_scale = max(
            midpoint_scale, abs(a[index]), abs(b[index]),
        )
    end
    inverse_midpoint_scale = one(T) / midpoint_scale
    cross_scaled =
        (a[1] * inverse_midpoint_scale) *
        (b[1] * inverse_midpoint_scale)
    @inbounds for index in 2:dimension
        cross_scaled +=
            (a[index] * inverse_midpoint_scale) *
            (b[index] * inverse_midpoint_scale)
    end
    radicand = inverse_midpoint_scale * inverse_midpoint_scale + cross_scaled
    isfinite(radicand) && radicand > zero(T) || return failure
    two = one(T) + one(T)
    wscale = sqrt(two) * midpoint_scale * sqrt(radicand)
    isfinite(wscale) && wscale > zero(T) || return failure

    normalized_tail_scale = one(T)
    @inbounds for index in 2:dimension
        normalized_tail_scale = max(
            normalized_tail_scale,
            abs(raw[index] / wscale),
        )
    end
    inv_tail_scale = one(T) / normalized_tail_scale
    head_accumulator = inv_tail_scale * inv_tail_scale
    @inbounds for index in 2:dimension
        value = (raw[index] / wscale) / normalized_tail_scale
        head_accumulator += value * value
    end
    normalized_head = normalized_tail_scale * sqrt(head_accumulator)
    isfinite(normalized_head) && normalized_head > zero(T) || return failure

    gamma = wscale / two
    lambda_scale = max(one(T), abs(a[1]), abs(b[1]), abs(gamma))
    @inbounds for index in 2:dimension
        lambda_scale = max(lambda_scale, abs(a[index]), abs(b[index]))
    end
    inverse_lambda_scale = one(T) / lambda_scale
    aa_head = a[1] * inverse_lambda_scale
    bb_head = b[1] * inverse_lambda_scale
    gg = gamma * inverse_lambda_scale
    denominator = aa_head + bb_head + two * gg
    isfinite(denominator) && denominator > zero(denominator) || return failure
    geometric = sqrt(sscale) * sqrt(zscale)
    lambda_head = geometric * gamma
    isfinite(lambda_head) || return failure

    lambda_tail = Vector{T}(undef, max(dimension - 1, 0))
    @inbounds for index in 2:dimension
        aa = a[index] * inverse_lambda_scale
        bb = b[index] * inverse_lambda_scale
        scaled = (gg + bb_head) * aa + (gg + aa_head) * bb
        value = geometric * (scaled / denominator) * lambda_scale
        isfinite(value) || return failure
        lambda_tail[index - 1] = value
    end

    w[1] = normalized_head
    lambda[1] = lambda_head
    @inbounds for index in 2:dimension
        w[index] = raw[index] / wscale
        lambda[index] = lambda_tail[index - 1]
    end
    return true, eta, eta_squared
end

"""Apply the symmetric Lorentz NT scaling map `W` in place or out of place."""

"""Apply `W^-1` in place or out of place."""

"""Apply `(W'W)^-1` without materializing a dense cone block."""

"""Lorentz determinant of the scalar triple `(x0, x1, x2)`."""
@inline _soc_lorentz_determinant(x0, x1, x2) = x0 * x0 - x1 * x1 - x2 * x2

"""
    _soc_sym2_inverse_entries(x0, x1, x2) -> (i11, i12, i22)

Inverse entries of the symmetric 2×2 matrix with packed coordinates
`(x11, x12, x22) = (x0 + x1, x2, x0 - x1)`, returned as the `(11, 12, 22)`
triple divided by the Lorentz determinant `x0² - x1² - x2²`. Callers
guarantee strict interiority, so the determinant is positive. One shared
formulation for the sign-sensitive triple that every 2×2 X⁻¹ contraction
needs.
"""
@inline function _soc_fixed_trace_determinant(x0, x1, x2)
    tail_norm = sqrt(x1 * x1 + x2 * x2)
    return (x0 - tail_norm) * (x0 + tail_norm)
end

@inline function _soc_sym2_inverse_entries(x0, x1, x2)
    determinant = _soc_fixed_trace_determinant(x0, x1, x2)
    return (x0 - x1) / determinant, -x2 / determinant, (x0 + x1) / determinant
end






"""Form the complete Q3 HKM map `M` in direct Lorentz coordinates.

For the Sym2 representatives `X` of `primal`, `Y=Z/2` of `dual`, and a
Lorentz direction `d`, this is the self-adjoint map

    M*d = symcoord(X^-1 * D * Y).

The fixed-trace local Schur uses its tail restriction `M[2:3,2:3]`; the full
map is retained for the homogeneous `b*dτ` head coupling.  The formulas are
provider-neutral and allocation-free.
"""
@inline function _soc_fixed_trace_hkm_full_metric!(destination, primal, dual)
    size(destination) == (3, 3) || throw(DimensionMismatch(
        "fixed-trace HKM metric destination must be 3×3",
    ))
    x0, x1, x2 = primal
    z0, z1, z2 = dual
    x_tail = sqrt(x1 * x1 + x2 * x2)
    z_tail = sqrt(z1 * z1 + z2 * z2)
    x0 > x_tail && z0 > z_tail || return false
    determinant = (x0 - x_tail) * (x0 + x_tail)
    isfinite(determinant) && determinant > zero(determinant) || return false

    m11 = (x0 * z0 - x1 * z1 - x2 * z2) / determinant
    m12 = (x0 * z1 - x1 * z0) / determinant
    m13 = (x0 * z2 - x2 * z0) / determinant
    m22 = (x0 * z0 - x1 * z1 + x2 * z2) / determinant
    m23 = -(x1 * z2 + x2 * z1) / determinant
    m33 = (x0 * z0 + x1 * z1 - x2 * z2) / determinant
    all(isfinite, (m11, m12, m13, m22, m23, m33)) || return false
    destination[1, 1] = m11
    destination[2, 1] = m12
    destination[3, 1] = m13
    destination[1, 2] = m12
    destination[2, 2] = m22
    destination[3, 2] = m23
    destination[1, 3] = m13
    destination[2, 3] = m23
    destination[3, 3] = m33
    return true
end

"""Form the affine HKM recovery term `r` in `dy = r - M*ds`."""
@inline function _soc_fixed_trace_hkm_rhs!(
    destination,
    primal,
    dual,
    affine_primal,
    affine_dual,
    target,
    include_affine_product::Bool,
)
    x0, x1, x2 = primal
    z0, z1, z2 = dual
    x_tail = sqrt(x1 * x1 + x2 * x2)
    x0 > x_tail || return false
    determinant = (x0 - x_tail) * (x0 + x_tail)
    isfinite(determinant) && determinant > zero(determinant) || return false
    x11, x12, x22 = x0 + x1, x2, x0 - x1
    y11, y12, y22 = (z0 + z1) / 2, z2 / 2, (z0 - z1) / 2

    r11 = target - (x11 * y11 + x12 * y12)
    r12 = -(x11 * y12 + x12 * y22)
    r21 = -(x12 * y11 + x22 * y12)
    r22 = target - (x12 * y12 + x22 * y22)
    if include_affine_product
        ax0, ax1, ax2 = affine_primal
        az0, az1, az2 = affine_dual
        ax11, ax12, ax22 = ax0 + ax1, ax2, ax0 - ax1
        ay11, ay12, ay22 =
            (az0 + az1) / 2, az2 / 2, (az0 - az1) / 2
        r11 -= ax11 * ay11 + ax12 * ay12
        r12 -= ax11 * ay12 + ax12 * ay22
        r21 -= ax12 * ay11 + ax22 * ay12
        r22 -= ax12 * ay12 + ax22 * ay22
    end

    inverse11 = (x0 - x1) / determinant
    inverse12 = -x2 / determinant
    inverse22 = (x0 + x1) / determinant
    q11 = inverse11 * r11 + inverse12 * r21
    q12 = inverse11 * r12 + inverse12 * r22
    q21 = inverse12 * r11 + inverse22 * r21
    q22 = inverse12 * r12 + inverse22 * r22
    destination[1] = q11 + q22
    destination[2] = q11 - q22
    destination[3] = q12 + q21
    return all(isfinite, destination)
end



"""Solve `left o result = right` in the Lorentz Jordan algebra."""
