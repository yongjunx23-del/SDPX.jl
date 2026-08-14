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
