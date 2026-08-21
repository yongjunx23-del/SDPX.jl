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
function _soc_nt_apply_w!(destination, w, eta, source)
    dimension = _soc_require_equal_dimensions(
        "Lorentz W vectors", destination, w, source,
    )
    head = source[1]
    dot_tail = zero(promote_type(eltype(w), eltype(source)))
    @inbounds for index in 2:dimension
        dot_tail += w[index] * source[index]
    end
    Base.mightalias(destination, w) && throw(ArgumentError(
        "Lorentz W destination must not alias its scaling point",
    ))
    correction = head + dot_tail / (one(w[1]) + w[1])
    destination[1] = eta * (w[1] * head + dot_tail)
    @inbounds for index in 2:dimension
        destination[index] = eta * (source[index] + correction * w[index])
    end
    return destination
end

"""Apply `W^-1` in place or out of place."""
function _soc_nt_apply_winv!(destination, w, eta, source)
    dimension = _soc_require_equal_dimensions(
        "Lorentz W inverse vectors", destination, w, source,
    )
    head = source[1]
    dot_tail = zero(promote_type(eltype(w), eltype(source)))
    @inbounds for index in 2:dimension
        dot_tail += w[index] * source[index]
    end
    Base.mightalias(destination, w) && throw(ArgumentError(
        "Lorentz W inverse destination must not alias its scaling point",
    ))
    correction = -head + dot_tail / (one(w[1]) + w[1])
    destination[1] = (w[1] * head - dot_tail) / eta
    @inbounds for index in 2:dimension
        destination[index] = (source[index] + correction * w[index]) / eta
    end
    return destination
end

"""Apply `(W'W)^-1` without materializing a dense cone block."""
function _soc_nt_apply_hs_inverse!(destination, w, eta_squared, source)
    dimension = _soc_require_equal_dimensions(
        "Lorentz Hs inverse vectors", destination, w, source,
    )
    Base.mightalias(destination, w) && throw(ArgumentError(
        "Lorentz Hs inverse destination must not alias its scaling point",
    ))
    head = source[1]
    lorentz_dot = w[1] * head
    @inbounds for index in 2:dimension
        lorentz_dot -= w[index] * source[index]
    end
    two = one(w[1]) + one(w[1])
    destination[1] = (two * w[1] * lorentz_dot - head) / eta_squared
    @inbounds for index in 2:dimension
        destination[index] =
            (-two * w[index] * lorentz_dot + source[index]) / eta_squared
    end
    return destination
end

"""
Fixed-trace Q3 block kernels.

The planner supplies the two active variable ids and the owned 2x2 tail map,
so these routines never touch a sparse/dense cone matrix.  They are kept in
the Lorentz-kernel file deliberately: the arithmetic is provider-independent
and can be exercised directly for Float64, MultiFloat, and BigFloat.
"""
@inline function _soc_fixed_trace_primal_residual!(
    destination,
    x,
    slack,
    first::Int,
    second::Int,
    a11,
    a12,
    a21,
    a22,
    head,
    offset1,
    offset2,
)
    # The fixed head row of A is zero by construction.
    destination[1] = head - slack[1]
    destination[2] = a11 * x[first] + a12 * x[second] + offset1 - slack[2]
    destination[3] = a21 * x[first] + a22 * x[second] + offset2 - slack[3]
    return destination
end

@inline function _soc_fixed_trace_dual_scatter!(
    destination,
    dual,
    first::Int,
    second::Int,
    a11,
    a12,
    a21,
    a22,
)
    u1 = dual[2]
    u2 = dual[3]
    destination[first] -= a11 * u1 + a21 * u2
    destination[second] -= a12 * u1 + a22 * u2
    return destination
end

@inline function _soc_fixed_trace_transpose_scatter!(
    destination,
    values,
    first::Int,
    second::Int,
    a11,
    a12,
    a21,
    a22,
)
    u1 = values[2]
    u2 = values[3]
    destination[first] += a11 * u1 + a21 * u2
    destination[second] += a12 * u1 + a22 * u2
    return destination
end

@inline function _soc_fixed_trace_primal_map!(
    destination,
    dx,
    first::Int,
    second::Int,
    a11,
    a12,
    a21,
    a22,
)
    # Match the generic `mul!(destination, A, dx, 1, 1)` contract: the
    # destination already contains the primal residual, and the fixed-head
    # row contributes zero while the two tail rows are accumulated.
    destination[2] += a11 * dx[first] + a12 * dx[second]
    destination[3] += a21 * dx[first] + a22 * dx[second]
    return destination
end

@inline function _soc_fixed_trace_hkm_metric!(destination, primal, dual)
    x0, x1, x2 = primal
    z0, z1, z2 = dual
    # Interiority is decided by the stable head-versus-tail-norm comparison,
    # not the raw determinant difference (which cancels near the boundary);
    # the raw determinant is still what the metric divides by, so it must
    # also be strictly positive in floating point. A reflected head
    # (x0 < -||tail||) has a positive determinant but is outside the cone —
    # the old determinant-only test accepted it.
    tail_norm = sqrt(x1 * x1 + x2 * x2)
    x0 > tail_norm || return false
    determinant = x0 * x0 - x1 * x1 - x2 * x2
    determinant > zero(determinant) || return false
    destination[1] = (x0 * z0 - x1 * z1 + x2 * z2) / determinant
    destination[2] = -(x1 * z2 + x2 * z1) / determinant
    destination[3] = (x0 * z0 + x1 * z1 - x2 * z2) / determinant
    return true
end

@inline function _soc_fixed_trace_hkm_rhs_coordinates(
    primal,
    dual,
    primal_residual,
    affine_primal,
    affine_dual,
    target,
    include_affine_product::Bool,
)
    x0, x1, x2 = primal
    z0, z1, z2 = dual
    p0, p1, p2 = primal_residual
    x11, x12, x22 = x0 + x1, x2, x0 - x1
    y11, y12, y22 = (z0 + z1) / 2, z2 / 2, (z0 - z1) / 2
    p11, p12, p22 = p0 + p1, p2, p0 - p1

    r11 = target - (x11 * y11 + x12 * y12)
    r12 = -(x11 * y12 + x12 * y22)
    r21 = -(x12 * y11 + x22 * y12)
    r22 = target - (x12 * y12 + x22 * y22)
    if include_affine_product
        dx0, dx1, dx2 = affine_primal
        dz0, dz1, dz2 = affine_dual
        dx11, dx12, dx22 = dx0 + dx1, dx2, dx0 - dx1
        dy11, dy12, dy22 =
            (dz0 + dz1) / 2, dz2 / 2, (dz0 - dz1) / 2
        r11 -= dx11 * dy11 + dx12 * dy12
        r12 -= dx11 * dy12 + dx12 * dy22
        r21 -= dx12 * dy11 + dx22 * dy12
        r22 -= dx12 * dy12 + dx22 * dy22
    end

    w11 = p11 * y11 + p12 * y12 - r11
    w12 = p11 * y12 + p12 * y22 - r12
    w21 = p12 * y11 + p22 * y12 - r21
    w22 = p12 * y12 + p22 * y22 - r22
    determinant = x0 * x0 - x1 * x1 - x2 * x2
    inverse11 = (x0 - x1) / determinant
    inverse12 = -x2 / determinant
    inverse22 = (x0 + x1) / determinant
    q11 = inverse11 * w11 + inverse12 * w21
    q12 = inverse11 * w12 + inverse12 * w22
    q21 = inverse12 * w11 + inverse22 * w21
    q22 = inverse12 * w12 + inverse22 * w22
    # The direct Lorentz dual uses Euclidean pairing, whereas the historical
    # 2x2 representation stores Y=Z/2.  These are exactly the coordinates of
    # twice the symmetric projection, which is the direct-coordinate
    # contraction consumed by A_tail'.
    return q11 + q22, q11 - q22, q12 + q21
end

@inline function _soc_fixed_trace_hkm_recovery!(
    destination,
    primal,
    dual,
    primal_direction,
    affine_primal,
    affine_dual,
    target,
    include_affine_product::Bool,
)
    x0, x1, x2 = primal
    z0, z1, z2 = dual
    dx0, dx1, dx2 = primal_direction
    x11, x12, x22 = x0 + x1, x2, x0 - x1
    y11, y12, y22 = (z0 + z1) / 2, z2 / 2, (z0 - z1) / 2
    dx11, dx12, dx22 = dx0 + dx1, dx2, dx0 - dx1

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
    w11 = r11 - (dx11 * y11 + dx12 * y12)
    w12 = r12 - (dx11 * y12 + dx12 * y22)
    w21 = r21 - (dx12 * y11 + dx22 * y12)
    w22 = r22 - (dx12 * y12 + dx22 * y22)
    determinant = x0 * x0 - x1 * x1 - x2 * x2
    inverse11 = (x0 - x1) / determinant
    inverse12 = -x2 / determinant
    inverse22 = (x0 + x1) / determinant
    q11 = inverse11 * w11 + inverse12 * w21
    q12 = inverse11 * w12 + inverse12 * w22
    q21 = inverse12 * w11 + inverse22 * w21
    q22 = inverse12 * w12 + inverse22 * w22
    destination[1] = q11 + q22
    destination[2] = q11 - q22
    destination[3] = q12 + q21
    return destination
end

"""Solve `left o result = right` in the Lorentz Jordan algebra."""
function _soc_jordan_solve!(destination, left, right)
    dimension = _soc_require_equal_dimensions(
        "Lorentz Jordan solve vectors", destination, left, right,
    )
    _soc_is_interior(left) || throw(ArgumentError(
        "the Lorentz Jordan divisor must be interior",
    ))
    left_head = left[1]
    right_head = right[1]
    determinant = _soc_determinant(left)
    tail_dot = zero(promote_type(eltype(left), eltype(right)))
    @inbounds for index in 2:dimension
        tail_dot += left[index] * right[index]
    end
    result_head = (left_head * right_head - tail_dot) / determinant
    tail_scale = (tail_dot / left_head - right_head) / determinant
    destination[1] = result_head
    @inbounds for index in 2:dimension
        destination[index] = tail_scale * left[index] + right[index] / left_head
    end
    return destination
end
