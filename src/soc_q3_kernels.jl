"""
Allocation-conscious kernels for the isomorphism

    (m₀,m₁,m₂) ↦ [m₀ + m₁  m₂; m₂  m₀ - m₁]

between `Q₃` coordinates and real symmetric `2×2` matrices.  The file is
intentionally self contained: it can be included by the native Q3 backend
without requiring the rest of SDPX.  Every mutating entry point writes into a
caller-owned vector or matrix and uses only scalar temporaries.  Full (not
necessarily symmetric) matrices represented by a length-four vector always
use the documented row-major order `(11, 12, 21, 22)`.

The arithmetic is deliberately expressed in terms of `one(T)`, `zero(T)`, and
scalar conversions rather than Float64 constants.  Consequently the same
code works for `Float64`, `BigFloat`, and the package's extended scalar types.
"""

@inline _q3_owned_scalar(value) = value
@inline _q3_owned_scalar(value::BigFloat) = deepcopy(value)

"""Check that a Q3 coordinate container has at least its three coordinates."""
@inline function _q3_require_coords(v)
    length(v) >= 3 || throw(DimensionMismatch("Q3 coordinates must have length 3"))
    return nothing
end

@inline function _q3_coordinate_values(v)
    length(v) in (2, 3) ||
        throw(DimensionMismatch("Q3 coordinates must have length 2 or 3"))
    if length(v) == 2
        return zero(v[1]), v[1], v[2]
    end
    return v[1], v[2], v[3]
end

"""Check that a full-4 coordinate container has four entries."""
@inline function _q3_require_full4(v)
    length(v) >= 4 || throw(DimensionMismatch("full 2×2 coordinates must have length 4"))
    return nothing
end

@inline function _q3_require_matrix(M)
    size(M) == (2, 2) || throw(DimensionMismatch("the Q3 matrix image must be 2×2"))
    return nothing
end

"""
    _q3_to_sym2!(destination, coordinates)

Write the symmetric matrix image of Q3 coordinates into `destination`.
"""
function _q3_to_sym2!(destination::AbstractMatrix, coordinates)
    _q3_require_matrix(destination)
    m0, m1, m2 = _q3_coordinate_values(coordinates)
    destination[1, 1] = m0 + m1
    destination[1, 2] = _q3_owned_scalar(m2)
    destination[2, 1] = _q3_owned_scalar(m2)
    destination[2, 2] = m0 - m1
    return destination
end

"""
    _sym2_to_q3!(destination, matrix)

Write Q3 coordinates for a symmetric `2×2` matrix into `destination`.
Off-diagonal entries are averaged, making the conversion robust to a matrix
whose two stored off-diagonal values differ by roundoff.
"""
function _sym2_to_q3!(destination, matrix::AbstractMatrix)
    length(destination) in (2, 3) ||
        throw(DimensionMismatch("Q3 coordinates must have length 2 or 3"))
    _q3_require_matrix(matrix)
    two = one(eltype(destination)) + one(eltype(destination))
    if length(destination) == 2
        destination[1] = (matrix[1, 1] - matrix[2, 2]) / two
        destination[2] = (matrix[1, 2] + matrix[2, 1]) / two
    else
        destination[1] = (matrix[1, 1] + matrix[2, 2]) / two
        destination[2] = (matrix[1, 1] - matrix[2, 2]) / two
        destination[3] = (matrix[1, 2] + matrix[2, 1]) / two
    end
    return destination
end

"""Allocate the matrix image (the mutating form is preferred in hot paths)."""
function _q3_to_sym2(coordinates)
    length(coordinates) in (2, 3) ||
        throw(DimensionMismatch("Q3 coordinates must have length 2 or 3"))
    T = eltype(coordinates)
    destination = Matrix{T}(undef, 2, 2)
    return _q3_to_sym2!(destination, coordinates)
end

"""Allocate Q3 coordinates for a symmetric matrix."""
function _sym2_to_q3(matrix::AbstractMatrix)
    _q3_require_matrix(matrix)
    T = eltype(matrix)
    destination = Vector{T}(undef, 3)
    return _sym2_to_q3!(destination, matrix)
end

# Descriptive aliases used by the native backend and by older experiments.
const _q3_to_matrix! = _q3_to_sym2!
const _q3_from_matrix! = _sym2_to_q3!
const _q3_to_matrix = _q3_to_sym2
const _q3_from_matrix = _sym2_to_q3
const _q3_from_sym2! = _sym2_to_q3!
const _sym2_from_q3! = _q3_to_sym2!
const _q3_coords_to_sym2! = _q3_to_sym2!
const _sym2_to_q3_coords! = _sym2_to_q3!

@inline function _q3_determinant(coordinates)
    m0, m1, m2 = _q3_coordinate_values(coordinates)
    return m0 * m0 - m1 * m1 - m2 * m2
end

@inline function _q3_determinant(matrix::AbstractMatrix)
    _q3_require_matrix(matrix)
    return matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]
end

"""The smallest eigenvalue of the symmetric matrix image."""
@inline function _q3_margin(coordinates)
    m0, m1, m2 = _q3_coordinate_values(coordinates)
    return m0 - sqrt(m1 * m1 + m2 * m2)
end

@inline function _q3_margin(matrix::AbstractMatrix)
    _q3_require_matrix(matrix)
    two = one(eltype(matrix)) + one(eltype(matrix))
    diagonal_difference = matrix[1, 1] - matrix[2, 2]
    off_diagonal_sum = matrix[1, 2] + matrix[2, 1]
    return (matrix[1, 1] + matrix[2, 2] -
            sqrt(diagonal_difference * diagonal_difference +
                 off_diagonal_sum * off_diagonal_sum)) / two
end

@inline function _q3_isposdef(coordinates)
    m0, _, _ = _q3_coordinate_values(coordinates)
    z = zero(m0)
    return m0 > z && _q3_determinant(coordinates) > z
end

@inline function _q3_isposdef(matrix::AbstractMatrix)
    _q3_require_matrix(matrix)
    z = zero(matrix[1, 1])
    return matrix[1, 1] > z && _q3_determinant(matrix) > z
end

@inline function _q3_is_psd(coordinates)
    m0, _, _ = _q3_coordinate_values(coordinates)
    z = zero(m0)
    return m0 >= z && _q3_determinant(coordinates) >= z
end

@inline function _q3_is_psd(matrix::AbstractMatrix)
    _q3_require_matrix(matrix)
    z = zero(matrix[1, 1])
    return matrix[1, 1] >= z && _q3_determinant(matrix) >= z
end

const _q3_pd = _q3_isposdef
const _q3_is_pd = _q3_isposdef
const _q3_is_interior = _q3_isposdef
const _q3_det = _q3_determinant
const _q3_psd_margin = _q3_margin
const _q3_min_eigenvalue = _q3_margin

@inline function _q3_trial_isposdef(coordinates, alpha, direction)
    s0, s1, s2 = _q3_coordinate_values(coordinates)
    d0, d1, d2 = _q3_coordinate_values(direction)
    first = s0 + alpha * d0
    second = s1 + alpha * d1
    third = s2 + alpha * d2
    z = zero(first)
    return first > z && first * first - second * second - third * third > z
end

@inline function _q3_trial_is_psd(coordinates, alpha, direction)
    s0, s1, s2 = _q3_coordinate_values(coordinates)
    d0, d1, d2 = _q3_coordinate_values(direction)
    first = s0 + alpha * d0
    second = s1 + alpha * d1
    third = s2 + alpha * d2
    z = zero(first)
    return first >= z && first * first - second * second - third * third >= z
end

@inline function _q3_consider_boundary(root, candidate, zero_value)
    return candidate > zero_value && candidate <= root ? candidate : root
end

"""
    _q3_fraction_to_boundary(s, ds)

Return the largest `α ∈ [0,1]` for which `s + α*ds` stays in the closed
Q3 cone, assuming `s` is interior.  The determinant is a quadratic.  Roots
are evaluated with the `q`-formula (`q = -(b ± √Δ)/2`) and the product of
the roots, avoiding the severe cancellation of the conventional `(-b+√Δ)`
root near a tangential boundary.  The head-coordinate root is considered as
well, since a ray can leave the cone through `m₀ = 0` before its determinant
vanishes.
"""
function _q3_fraction_to_boundary(coordinates, direction)
    s0, s1, s2 = _q3_coordinate_values(coordinates)
    d0, d1, d2 = _q3_coordinate_values(direction)
    T = promote_type(eltype(coordinates), eltype(direction))
    z = zero(T)
    o = one(T)
    two = o + o
    four = two + two

    c0 = s0 * s0 - s1 * s1 - s2 * s2
    c1 = two * (s0 * d0 - s1 * d1 - s2 * d2)
    c2 = d0 * d0 - d1 * d1 - d2 * d2

    # The usual caller starts strictly inside the cone, but handling an exact
    # boundary is useful for diagnostics and keeps the closed-cone contract
    # literal.  A negative first derivative of either active inequality means
    # that no positive step is feasible.
    if s0 == z && d0 < z
        return z
    elseif c0 == z && (c1 < z || (iszero(c1) && c2 < z))
        return z
    end

    # A full step is cheap to certify and avoids root calculations in the
    # overwhelming common case.
    full0 = s0 + d0
    full1 = s1 + d1
    full2 = s2 + d2
    if full0 >= z && full0 * full0 - full1 * full1 - full2 * full2 >= z
        return o
    end

    root = o

    if iszero(c2)
        c1 < z && (root = _q3_consider_boundary(root, -c0 / c1, z))
    else
        discriminant = c1 * c1 - four * c2 * c0
        # Rounding may produce a tiny negative discriminant at a tangent;
        # clamping is mathematically harmless for the closed-cone boundary.
        discriminant < z && (discriminant = z)
        square_root = sqrt(discriminant)
        # q is chosen with the sign that avoids subtracting nearly equal
        # numbers.  The other root follows from r₁r₂ = c₀/c₂.
        q = c1 >= z ? -(c1 + square_root) / two :
            -(c1 - square_root) / two
        if !iszero(q)
            root = _q3_consider_boundary(root, q / c2, z)
            root = _q3_consider_boundary(root, c0 / q, z)
        else
            # This is only reached for a repeated zero numerator.  The
            # direct expression is safe here because both terms coincide.
            root = _q3_consider_boundary(
                root,
                (-c1 - square_root) / (two * c2),
                z,
            )
            root = _q3_consider_boundary(
                root,
                (-c1 + square_root) / (two * c2),
                z,
            )
        end
    end
    d0 < z && (root = _q3_consider_boundary(root, -s0 / d0, z))
    return clamp(root, z, o)
end

const _q3_fraction_to_boundary_bound = _q3_fraction_to_boundary
const _q3_fraction_to_boundary_exact = _q3_fraction_to_boundary

"""Frobenius inner product of two Q3 coordinate vectors."""
@inline function _q3_frobenius_dot(left, right)
    l0, l1, l2 = _q3_coordinate_values(left)
    r0, r1, r2 = _q3_coordinate_values(right)
    two = one(left[1]) + one(left[1])
    return two * (l0 * r0 + l1 * r1 + l2 * r2)
end

@inline function _q3_frobenius_dot(left::AbstractMatrix, right::AbstractMatrix)
    _q3_require_matrix(left)
    _q3_require_matrix(right)
    return left[1, 1] * right[1, 1] + left[1, 2] * right[1, 2] +
           left[2, 1] * right[2, 1] + left[2, 2] * right[2, 2]
end

@inline function _q3_frobenius_dot(left, right::AbstractMatrix)
    left11, left12, left21, left22 = _q3_full_components(left)
    _q3_require_matrix(right)
    return left11 * right[1, 1] + left12 * right[1, 2] +
           left21 * right[2, 1] + left22 * right[2, 2]
end

@inline function _q3_frobenius_dot(left::AbstractMatrix, right)
    _q3_require_matrix(left)
    right11, right12, right21, right22 = _q3_full_components(right)
    return left[1, 1] * right11 + left[1, 2] * right12 +
           left[2, 1] * right21 + left[2, 2] * right22
end

const _q3_frobenius_inner = _q3_frobenius_dot
const _q3_dot = _q3_frobenius_dot
const _q3_frob_dot = _q3_frobenius_dot
const _q3_inner = _q3_frobenius_dot

@inline function _q3_vector_components(coordinates)
    return _q3_coordinate_values(coordinates)
end

@inline function _q3_matrix_components(matrix::AbstractMatrix)
    _q3_require_matrix(matrix)
    return matrix[1, 1], matrix[1, 2], matrix[2, 1], matrix[2, 2]
end

@inline function _q3_symmetric_coordinates(value)
    if value isa AbstractMatrix
        _q3_require_matrix(value)
        two = one(value[1, 1]) + one(value[1, 1])
        return (value[1, 1] + value[2, 2]) / two,
               (value[1, 1] - value[2, 2]) / two,
               (value[1, 2] + value[2, 1]) / two
    end
    return _q3_coordinate_values(value)
end

"""Read a symmetric Q3 value or a full-4 value as `(11,12,21,22)`."""
@inline function _q3_full_components(value)
    if value isa AbstractMatrix
        return _q3_matrix_components(value)
    end
    length(value) in (2, 3) && begin
        m0, m1, m2 = _q3_coordinate_values(value)
        return m0 + m1, m2, m2, m0 - m1
    end
    _q3_require_full4(value)
    return value[1], value[2], value[3], value[4]
end

"""
    _q3_product_full4!(destination, left, right)

Compute `left*right` and store the full result in row-major coordinates
`(11,12,21,22)`.  `left` and `right` may be Q3 vectors, full-4 vectors, or
`2×2` matrices; the output may be a length-four vector or a `2×2` matrix.
"""
function _q3_product_full4!(destination::AbstractVector, left, right)
    _q3_require_full4(destination)
    a11, a12, a21, a22 = _q3_full_components(left)
    b11, b12, b21, b22 = _q3_full_components(right)
    destination[1] = a11 * b11 + a12 * b21
    destination[2] = a11 * b12 + a12 * b22
    destination[3] = a21 * b11 + a22 * b21
    destination[4] = a21 * b12 + a22 * b22
    return destination
end

function _q3_product_full4!(destination::AbstractMatrix, left, right)
    _q3_require_matrix(destination)
    a11, a12, a21, a22 = _q3_full_components(left)
    b11, b12, b21, b22 = _q3_full_components(right)
    destination[1, 1] = a11 * b11 + a12 * b21
    destination[1, 2] = a11 * b12 + a12 * b22
    destination[2, 1] = a21 * b11 + a22 * b21
    destination[2, 2] = a21 * b12 + a22 * b22
    return destination
end

function _q3_product_full4(left, right)
    a11, a12, a21, a22 = _q3_full_components(left)
    b11, b12, b21, b22 = _q3_full_components(right)
    T = promote_type(typeof(a11), typeof(b11))
    destination = Vector{T}(undef, 4)
    destination[1] = a11 * b11 + a12 * b21
    destination[2] = a11 * b12 + a12 * b22
    destination[3] = a21 * b11 + a22 * b21
    destination[4] = a21 * b12 + a22 * b22
    return destination
end

const _q3_mul_full4! = _q3_product_full4!
const _q3_product4! = _q3_product_full4!
const _q3_product_to_full4! = _q3_product_full4!

"""Compute `X⁻¹ * F`, writing a full result in row-major coordinates."""
function _q3_inverse_left_multiply_full4!(destination::AbstractVector, x, full)
    _q3_require_full4(destination)
    x0, x1, x2 = _q3_symmetric_coordinates(x)
    determinant = x0 * x0 - x1 * x1 - x2 * x2
    inverse11 = (x0 - x1) / determinant
    inverse12 = -x2 / determinant
    inverse22 = (x0 + x1) / determinant
    f11, f12, f21, f22 = _q3_full_components(full)
    destination[1] = inverse11 * f11 + inverse12 * f21
    destination[2] = inverse11 * f12 + inverse12 * f22
    destination[3] = inverse12 * f11 + inverse22 * f21
    destination[4] = inverse12 * f12 + inverse22 * f22
    return destination
end

function _q3_inverse_left_multiply_full4!(destination::AbstractMatrix, x, full)
    _q3_require_matrix(destination)
    x0, x1, x2 = _q3_symmetric_coordinates(x)
    determinant = x0 * x0 - x1 * x1 - x2 * x2
    inverse11 = (x0 - x1) / determinant
    inverse12 = -x2 / determinant
    inverse22 = (x0 + x1) / determinant
    f11, f12, f21, f22 = _q3_full_components(full)
    destination[1, 1] = inverse11 * f11 + inverse12 * f21
    destination[1, 2] = inverse11 * f12 + inverse12 * f22
    destination[2, 1] = inverse12 * f11 + inverse22 * f21
    destination[2, 2] = inverse12 * f12 + inverse22 * f22
    return destination
end

const _q3_left_inverse_full4! = _q3_inverse_left_multiply_full4!
const _q3_inverse_left_full4! = _q3_inverse_left_multiply_full4!
const _q3_left_multiply_inverse! = _q3_inverse_left_multiply_full4!

@inline function _q3_coefficient_count(coefficients)
    if coefficients isa AbstractMatrix
        rows = size(coefficients, 1)
        (rows == 2 || rows == 3) ||
            throw(DimensionMismatch("Q3 coefficient matrix must have two or three rows"))
        return size(coefficients, 2)
    end
    isempty(coefficients) && return 0
    first = coefficients[1]
    if first isa AbstractMatrix
        _q3_require_matrix(first)
        return length(coefficients)
    end
    if first isa Number
        length(coefficients) in (2, 3) ||
            throw(DimensionMismatch("one traceless/Q3 coefficient must have length 2 or 3"))
        return 1
    end
    length(first) in (2, 3) ||
        throw(DimensionMismatch("traceless/Q3 coefficients must have length 2 or 3"))
    return length(coefficients)
end

@inline function _q3_coefficient_value(coefficients, coefficient::Int, coordinate::Int)
    if coefficients isa AbstractMatrix
        rows = size(coefficients, 1)
        rows == 3 && return coefficients[coordinate, coefficient]
        coordinate == 1 && return zero(coefficients[1, coefficient])
        return coefficients[coordinate - 1, coefficient]
    end
    first = coefficients[1]
    if first isa AbstractMatrix
        matrix = coefficients[coefficient]
        _q3_require_matrix(matrix)
        two = one(matrix[1, 1]) + one(matrix[1, 1])
        if coordinate == 1
            return (matrix[1, 1] + matrix[2, 2]) / two
        elseif coordinate == 2
            return (matrix[1, 1] - matrix[2, 2]) / two
        end
        return (matrix[1, 2] + matrix[2, 1]) / two
    end
    if first isa Number
        length(coefficients) == 3 && return coefficients[coordinate]
        coordinate == 1 && return zero(coefficients[1])
        return coefficients[coordinate - 1]
    end
    value = coefficients[coefficient]
    length(value) == 3 && return value[coordinate]
    coordinate == 1 && return zero(value[1])
    return value[coordinate - 1]
end

"""
    _q3_schur_metric!(H, coefficients, X, Y)

Build the fixed-trace traceless 2×2 metric

    H[i,j] = tr(Aⱼ * Y * Aᵢ * X⁻¹).

`coefficients` is either a `3×p` Q3-coordinate matrix (one coefficient per
column), a `2×p` traceless-coordinate matrix, or a vector of length-3/length-2
coordinate vectors.  Length-two inputs mean `(m₁,m₂)` with `m₀ = 0`.  The implementation forms
`Y*Aᵢ*X⁻¹` in scalar registers and contracts it directly against every `Aⱼ`;
no full matrix or pairwise panel is allocated.
"""
function _q3_schur_metric!(H::AbstractMatrix, coefficients, X, Y)
    count = _q3_coefficient_count(coefficients)
    size(H) == (count, count) || throw(DimensionMismatch("Schur metric must be p×p"))
    coefficients isa AbstractArray && Base.mightalias(H, coefficients) &&
        throw(ArgumentError(
            "Q3 Schur output must not alias its coefficient storage",
        ))
    x0, x1, x2 = _q3_symmetric_coordinates(X)
    determinant = x0 * x0 - x1 * x1 - x2 * x2
    inverse11 = (x0 - x1) / determinant
    inverse12 = -x2 / determinant
    inverse22 = (x0 + x1) / determinant

    y0, y1, y2 = _q3_symmetric_coordinates(Y)
    y11 = y0 + y1
    y12 = y2
    y22 = y0 - y1

    @inbounds for i in 1:count
        ai0 = _q3_coefficient_value(coefficients, i, 1)
        ai1 = _q3_coefficient_value(coefficients, i, 2)
        ai2 = _q3_coefficient_value(coefficients, i, 3)
        ai11 = ai0 + ai1
        ai12 = ai2
        ai22 = ai0 - ai1

        # W = Y*Aᵢ.
        w11 = y11 * ai11 + y12 * ai12
        w12 = y11 * ai12 + y12 * ai22
        w21 = y12 * ai11 + y22 * ai12
        w22 = y12 * ai12 + y22 * ai22

        # Q = W*X⁻¹.
        q11 = w11 * inverse11 + w12 * inverse12
        q12 = w11 * inverse12 + w12 * inverse22
        q21 = w21 * inverse11 + w22 * inverse12
        q22 = w21 * inverse12 + w22 * inverse22

        for j in 1:count
            aj0 = _q3_coefficient_value(coefficients, j, 1)
            aj1 = _q3_coefficient_value(coefficients, j, 2)
            aj2 = _q3_coefficient_value(coefficients, j, 3)
            aj11 = aj0 + aj1
            aj12 = aj2
            aj22 = aj0 - aj1
            H[i, j] = aj11 * q11 + aj12 * q21 + aj12 * q12 + aj22 * q22
        end
    end
    return H
end

const _q3_fixed_trace_schur_metric! = _q3_schur_metric!
const _q3_traceless_schur_metric! = _q3_schur_metric!
const _q3_schur! = _q3_schur_metric!

@inline function _q3_predictor_components(x, p, y, residual)
    x0, x1, x2 = _q3_symmetric_coordinates(x)
    p11, p12, p21, p22 = _q3_full_components(p)
    y11, y12, y21, y22 = _q3_full_components(y)
    r11, r12, r21, r22 = _q3_full_components(residual)
    py11 = p11 * y11 + p12 * y21
    py12 = p11 * y12 + p12 * y22
    py21 = p21 * y11 + p22 * y21
    py22 = p21 * y12 + p22 * y22
    w11 = py11 - r11
    w12 = py12 - r12
    w21 = py21 - r21
    w22 = py22 - r22

    determinant = x0 * x0 - x1 * x1 - x2 * x2
    inverse11 = (x0 - x1) / determinant
    inverse12 = -x2 / determinant
    inverse22 = (x0 + x1) / determinant
    return inverse11 * w11 + inverse12 * w21,
           inverse11 * w12 + inverse12 * w22,
           inverse12 * w11 + inverse22 * w21,
           inverse12 * w12 + inverse22 * w22
end

function _q3_predictor_rhs_contraction!(destination::AbstractVector, x, p, y, residual)
    _q3_require_full4(destination)
    destination[1], destination[2], destination[3], destination[4] =
        _q3_predictor_components(x, p, y, residual)
    return destination
end

function _q3_predictor_rhs_contraction!(destination::AbstractMatrix, x, p, y, residual)
    _q3_require_matrix(destination)
    q11, q12, q21, q22 = _q3_predictor_components(x, p, y, residual)
    destination[1, 1] = q11
    destination[1, 2] = q12
    destination[2, 1] = q21
    destination[2, 2] = q22
    return destination
end

function _q3_predictor_rhs_contraction(x, p, y, residual)
    values = _q3_predictor_components(x, p, y, residual)
    T = typeof(values[1])
    destination = Vector{T}(undef, 4)
    destination[1], destination[2], destination[3], destination[4] = values
    return destination
end

const _q3_predictor_rhs! = _q3_predictor_rhs_contraction!
const _q3_predictor_contract! = _q3_predictor_rhs_contraction!

@inline function _q3_direction_components(x, direction_x, y, residual)
    q11, q12, q21, q22 = _q3_predictor_components(x, direction_x, y, residual)
    two = one(q11) + one(q11)
    # `_q3_predictor_components` evaluates X^-1(dX*Y - R). SDPX's Newton
    # direction convention is the negative of that quantity:
    #
    #     dY = sym(X^-1(R - dX*Y)).
    #
    # Keeping this sign synchronized with `threaded_direction_blocks!` is
    # essential; the opposite convention makes the affine predictor increase
    # complementarity unless the KKT right-hand side is also negated.
    return -(q11 + q22) / two,
           -(q11 - q22) / two,
           -(q12 + q21) / two
end

"""
    _q3_direction_recovery!(destination, X, dX, Y, R)

Recover the symmetric direction `sym(X⁻¹*(R - dX*Y))`.  A length-two
destination receives traceless coordinates `(m₁,m₂)`, a length-three
destination receives Q3 coordinates, and a length-four destination or a `2×2`
destination receives row-major/full matrix coordinates.  `R` may be symmetric
Q3/full-4 coordinates or a `2×2` matrix.
"""
function _q3_direction_recovery!(destination::AbstractVector, x, direction_x, y, residual)
    length(destination) in (2, 3, 4) ||
        throw(DimensionMismatch("direction output must have length 2, 3, or 4"))
    m0, m1, m2 = _q3_direction_components(x, direction_x, y, residual)
    if length(destination) == 2
        destination[1], destination[2] = m1, m2
    elseif length(destination) == 3
        destination[1], destination[2], destination[3] = m0, m1, m2
    else
        destination[1] = m0 + m1
        destination[2] = _q3_owned_scalar(m2)
        destination[3] = _q3_owned_scalar(m2)
        destination[4] = m0 - m1
    end
    return destination
end

function _q3_direction_recovery!(destination::AbstractMatrix, x, direction_x, y, residual)
    _q3_require_matrix(destination)
    m0, m1, m2 = _q3_direction_components(x, direction_x, y, residual)
    destination[1, 1] = m0 + m1
    destination[1, 2] = _q3_owned_scalar(m2)
    destination[2, 1] = _q3_owned_scalar(m2)
    destination[2, 2] = m0 - m1
    return destination
end

const _q3_recover_direction! = _q3_direction_recovery!
const _q3_direction! = _q3_direction_recovery!

"""Compute `target*I - X*Y - dX*dY` in row-major full-4 coordinates."""
function _q3_corrector_residual!(destination::AbstractVector, target, x, y, direction_x, direction_y)
    _q3_require_full4(destination)
    x11, x12, x21, x22 = _q3_full_components(x)
    y11, y12, y21, y22 = _q3_full_components(y)
    dx11, dx12, dx21, dx22 = _q3_full_components(direction_x)
    dy11, dy12, dy21, dy22 = _q3_full_components(direction_y)
    xy11 = x11 * y11 + x12 * y21
    xy12 = x11 * y12 + x12 * y22
    xy21 = x21 * y11 + x22 * y21
    xy22 = x21 * y12 + x22 * y22
    dd11 = dx11 * dy11 + dx12 * dy21
    dd12 = dx11 * dy12 + dx12 * dy22
    dd21 = dx21 * dy11 + dx22 * dy21
    dd22 = dx21 * dy12 + dx22 * dy22
    destination[1] = target - xy11 - dd11
    destination[2] = -xy12 - dd12
    destination[3] = -xy21 - dd21
    destination[4] = target - xy22 - dd22
    return destination
end

function _q3_corrector_residual!(destination::AbstractMatrix, target, x, y, direction_x, direction_y)
    _q3_require_matrix(destination)
    x11, x12, x21, x22 = _q3_full_components(x)
    y11, y12, y21, y22 = _q3_full_components(y)
    dx11, dx12, dx21, dx22 = _q3_full_components(direction_x)
    dy11, dy12, dy21, dy22 = _q3_full_components(direction_y)
    xy11 = x11 * y11 + x12 * y21
    xy12 = x11 * y12 + x12 * y22
    xy21 = x21 * y11 + x22 * y21
    xy22 = x21 * y12 + x22 * y22
    dd11 = dx11 * dy11 + dx12 * dy21
    dd12 = dx11 * dy12 + dx12 * dy22
    dd21 = dx21 * dy11 + dx22 * dy21
    dd22 = dx21 * dy12 + dx22 * dy22
    destination[1, 1] = target - xy11 - dd11
    destination[1, 2] = -xy12 - dd12
    destination[2, 1] = -xy21 - dd21
    destination[2, 2] = target - xy22 - dd22
    return destination
end

const _q3_corrector! = _q3_corrector_residual!
const _q3_corrector_rhs! = _q3_corrector_residual!
const _q3_corrector_residual_full4! = _q3_corrector_residual!

# ---------------------------------------------------------------------------
# Q3 Nesterov--Todd scaling
# ---------------------------------------------------------------------------

"""
    _q3_nt_scaling!(w, lambda, s, z) -> (ok, eta, eta_squared)

Construct the Nesterov--Todd point for two interior Q3/Lorentz points.  The
three coordinates use the Lorentz convention `(m₀,m₁,m₂)` and `J` is
`diag(1,-1,-1)`.  `w` and `lambda` are caller-owned, length-three vectors;
the function writes them only after all checks have succeeded.  A `false`
status denotes a non-interior, non-finite, or numerically degenerate input
(`eta` and `eta_squared` are zero in that case).

The implementation follows Clarabel's SOC construction.  In particular,

```
eta = sqrt(sqrt(det(s)) / sqrt(det(z)))
w   = normalize(s / sqrt(det(s)) + J*z / sqrt(det(z)))
```

where `normalize` is with respect to the Lorentz quadratic form.  The
scaling point is then obtained from the same normalized vectors.  Residuals
are evaluated after a max-norm rescaling, so ordinary Float64 values spanning
many orders of magnitude do not overflow merely while checking interiority.
"""
@inline function _q3_nt_require_vector(v, label::AbstractString)
    length(v) == 3 || throw(DimensionMismatch("Q3 $label must have length 3"))
    return nothing
end

@inline function _q3_nt_coordinates(v, label::AbstractString)
    if v isa AbstractMatrix
        _q3_require_matrix(v)
        return _q3_symmetric_coordinates(v)
    end
    _q3_nt_require_vector(v, label)
    return v[1], v[2], v[3]
end

"""Stable `sqrt(x₀²-x₁²-x₂²)` together with an interiority flag."""
@inline function _q3_nt_sqrt_residual(x0, x1, x2)
    # Scaling by the largest coordinate prevents overflow in both the norm
    # and the factored residual.  In an interior point the normalized head is
    # at most one, hence the final multiplication cannot overflow unless the
    # requested residual itself is unrepresentable.
    scale = max(abs(x0), abs(x1), abs(x2))
    z = zero(scale)
    if !(isfinite(scale) && scale > z)
        return false, z, z
    end
    u0 = x0 / scale
    u1 = x1 / scale
    u2 = x2 / scale
    tail = sqrt(u1 * u1 + u2 * u2)
    if !(isfinite(tail) && u0 > tail)
        return false, z, z
    end
    residual_unit = (u0 - tail) * (u0 + tail)
    if !(isfinite(residual_unit) && residual_unit > z)
        return false, z, z
    end
    residual = scale * sqrt(residual_unit)
    if !(isfinite(residual) && residual > z)
        return false, z, z
    end
    return true, scale, residual
end

@inline function _q3_nt_allfinite(x0, x1, x2)
    return isfinite(x0) && isfinite(x1) && isfinite(x2)
end

@inline _q3_nt_failure(value) = (false, zero(value), zero(value))

@inline function _q3_nt_allfinite(x0, x1)
    return isfinite(x0) && isfinite(x1)
end

@inline function _q3_nt_writeable_outputs(w, lambda)
    _q3_nt_require_vector(w, "scaling output w")
    _q3_nt_require_vector(lambda, "scaling output lambda")
    # The arithmetic below is alias-safe only for the input vectors.  Reject
    # exact and view-based overlap rather than silently overwriting an input.
    return nothing
end

@inline function _q3_nt_scaling_failure_coordinates(value)
    z = zero(value)
    return false, z, z, z, z, z, z, z, z
end

"""Scalar-register NT construction used by the batched fixed-trace backend."""
@inline function _q3_nt_scaling_coordinates(s0, s1, s2, z0, z1, z2)
    # Use the first input's scalar type for failure values.  Inputs are
    # promoted naturally by the arithmetic below, while output storage does
    # the final conversion when mixed scalar types are used.
    if !(_q3_nt_allfinite(s0, s1, s2) && _q3_nt_allfinite(z0, z1, z2))
        return _q3_nt_scaling_failure_coordinates(s0)
    end

    sok, _, sscale = _q3_nt_sqrt_residual(s0, s1, s2)
    zok, _, zscale = _q3_nt_sqrt_residual(z0, z1, z2)
    if !(sok && zok)
        return _q3_nt_scaling_failure_coordinates(s0)
    end

    # Normalized primal/dual points.  Forming these ratios rather than
    # multiplying by the reciprocal keeps BigFloat values independent and
    # avoids an extra mutable scalar being shared by output entries.
    a0 = s0 / sscale
    a1 = s1 / sscale
    a2 = s2 / sscale
    b0 = z0 / zscale
    b1 = z1 / zscale
    b2 = z2 / zscale
    if !_q3_nt_allfinite(a0, a1, a2) || !_q3_nt_allfinite(b0, b1, b2)
        return _q3_nt_scaling_failure_coordinates(s0)
    end

    # eta = sqrt(sscale/zscale).  Taking square roots first avoids an
    # intermediate overflow/underflow when the ratio itself is representable.
    eta = sqrt(sscale) / sqrt(zscale)
    eta_squared = eta * eta
    if !(isfinite(eta) && eta > zero(eta) &&
         isfinite(eta_squared) && eta_squared > zero(eta_squared))
        return _q3_nt_scaling_failure_coordinates(s0)
    end

    # Raw normalized w = a + J*b.
    raw0 = a0 + b0
    raw1 = a1 - b1
    raw2 = a2 - b2
    if !_q3_nt_allfinite(raw0, raw1, raw2)
        return _q3_nt_scaling_failure_coordinates(s0)
    end
    wok, _, wscale = _q3_nt_sqrt_residual(raw0, raw1, raw2)
    if !wok
        return _q3_nt_scaling_failure_coordinates(s0)
    end

    nw1 = raw1 / wscale
    nw2 = raw2 / wscale
    if !_q3_nt_allfinite(nw1, nw2)
        return _q3_nt_scaling_failure_coordinates(s0)
    end

    # Force w'Jw = 1 as Clarabel does.  The scaled branch avoids squaring an
    # enormous tail (which can overflow even when its norm is representable).
    one_w = one(nw1)
    tail_scale = max(one_w, abs(nw1), abs(nw2))
    inv_tail_scale = one_w / tail_scale
    wn1 = nw1 * inv_tail_scale
    wn2 = nw2 * inv_tail_scale
    nw0 = tail_scale * sqrt(inv_tail_scale * inv_tail_scale +
                            (nw1 / tail_scale) * (nw1 / tail_scale) +
                            (nw2 / tail_scale) * (nw2 / tail_scale))
    if !_q3_nt_allfinite(nw0, nw1, nw2) || !(nw0 > zero(nw0))
        return _q3_nt_scaling_failure_coordinates(s0)
    end

    # Compute the lambda tail in a scaled register.  Near the cone boundary
    # the normalized coordinates can be large; scaling all of them before the
    # two products prevents an otherwise harmless common factor from
    # overflowing.  This is algebraically identical to Clarabel's formula.
    gamma = wscale / (one(wscale) + one(wscale))
    lambda_scale = max(one(gamma), abs(a0), abs(a1), abs(a2),
                       abs(b0), abs(b1), abs(b2), abs(gamma))
    inv_lambda_scale = one(gamma) / lambda_scale
    aa0 = a0 * inv_lambda_scale
    aa1 = a1 * inv_lambda_scale
    aa2 = a2 * inv_lambda_scale
    bb0 = b0 * inv_lambda_scale
    bb1 = b1 * inv_lambda_scale
    bb2 = b2 * inv_lambda_scale
    gg = gamma * inv_lambda_scale
    denominator = aa0 + bb0 + gg + gg
    if !(isfinite(denominator) && denominator > zero(denominator))
        return _q3_nt_scaling_failure_coordinates(s0)
    end
    lambda_tail1_scaled = (gg + bb0) * aa1 + (gg + aa0) * bb1
    lambda_tail2_scaled = (gg + bb0) * aa2 + (gg + aa0) * bb2
    lambda_tail1 = (lambda_tail1_scaled / denominator) * lambda_scale
    lambda_tail2 = (lambda_tail2_scaled / denominator) * lambda_scale

    geometric = sqrt(sscale) * sqrt(zscale)
    lambda0 = geometric * gamma
    lambda1 = geometric * lambda_tail1
    lambda2 = geometric * lambda_tail2
    if !_q3_nt_allfinite(lambda0, lambda1, lambda2)
        return _q3_nt_scaling_failure_coordinates(s0)
    end

    return true, nw0, nw1, nw2, lambda0, lambda1, lambda2, eta, eta_squared
end

function _q3_nt_scaling!(w::AbstractVector, lambda::AbstractVector, s, z)
    _q3_nt_writeable_outputs(w, lambda)
    # `mightalias` is cheap for ordinary vectors and catches SubArray views as
    # well.  Keeping this check outside the hot arithmetic makes accidental
    # output/input aliasing an explicit error instead of a silent corruption.
    (Base.mightalias(w, s) || Base.mightalias(w, z) ||
     Base.mightalias(lambda, s) || Base.mightalias(lambda, z) ||
     Base.mightalias(w, lambda)) &&
        throw(ArgumentError("Q3 NT scaling outputs must not alias inputs or each other"))

    s0, s1, s2 = _q3_nt_coordinates(s, "scaling input s")
    z0, z1, z2 = _q3_nt_coordinates(z, "scaling input z")
    ok, nw0, nw1, nw2, lambda0, lambda1, lambda2, eta, eta_squared =
        _q3_nt_scaling_coordinates(s0, s1, s2, z0, z1, z2)
    ok || return _q3_nt_failure(s0)

    # Commit only now: a failed scaling leaves caller-owned workspaces intact.
    w[1] = nw0
    w[2] = nw1
    w[3] = nw2
    lambda[1] = lambda0
    lambda[2] = lambda1
    lambda[3] = lambda2
    return true, eta, eta_squared
end

"""Build a six-entry upper-triangular Q3 `Hₛ` block.

The packing order is Clarabel's dense SOC order
`(h₁₁,h₁₂,h₂₂,h₁₃,h₂₃,h₃₃)`.  The block equals
`eta_squared * (2*w*w' - J)` with `J = diag(1,-1,-1)`.
"""
function _q3_nt_hs!(packed::AbstractVector, w::AbstractVector, eta_squared)
    length(packed) == 6 ||
        throw(DimensionMismatch("packed Q3 Hs output must have length 6"))
    _q3_nt_require_vector(w, "normalized w")
    Base.mightalias(packed, w) &&
        throw(ArgumentError("packed Q3 Hs output must not alias w"))
    w0, w1, w2 = w[1], w[2], w[3]
    z = zero(eta_squared)
    (isfinite(eta_squared) && eta_squared > z &&
     _q3_nt_allfinite(w0, w1, w2)) ||
        throw(ArgumentError("Q3 Hs requires finite positive eta_squared and w"))
    two = one(w0) + one(w0)
    h11 = two * w0 * w0 - one(w0)
    h12 = two * w0 * w1
    h22 = two * w1 * w1 + one(w0)
    h13 = two * w0 * w2
    h23 = two * w1 * w2
    h33 = two * w2 * w2 + one(w0)
    packed[1] = eta_squared * h11
    packed[2] = eta_squared * h12
    packed[3] = eta_squared * h22
    packed[4] = eta_squared * h13
    packed[5] = eta_squared * h23
    packed[6] = eta_squared * h33
    return packed
end

"""Apply `Hₛ = eta_squared*(2*w*w' - J)` without materializing `Hₛ`."""
@inline function _q3_nt_apply_hs!(destination::AbstractVector,
                                  w::AbstractVector,
                                  eta_squared,
                                  x::AbstractVector)
    _q3_nt_require_vector(destination, "Hs destination")
    _q3_nt_require_vector(w, "normalized w")
    _q3_nt_require_vector(x, "Hs input")
    # Load all input coordinates before writing so an in-place call is safe.
    x0, x1, x2 = x[1], x[2], x[3]
    w0, w1, w2 = w[1], w[2], w[3]
    two = one(w0) + one(w0)
    c = two * (w0 * x0 + w1 * x1 + w2 * x2)
    destination[1] = eta_squared * (c * w0 - x0)
    destination[2] = eta_squared * (c * w1 + x1)
    destination[3] = eta_squared * (c * w2 + x2)
    return destination
end

"""Apply the symmetric Q3 Nesterov--Todd scaling map `W`.

The normalized scaling point satisfies `w'Jw=1`, and `eta>0`.  This is the
three-coordinate specialization of the ECOS/Clarabel SOC product.  The input
is loaded before any output is written, so `destination === x` is supported.
"""
@inline function _q3_nt_apply_w_coordinates(
    w0, w1, w2, eta, x0, x1, x2,
)
    tail_dot = w1 * x1 + w2 * x2
    correction = x0 + tail_dot / (one(w0) + w0)
    return (
        eta * (w0 * x0 + tail_dot),
        eta * (x1 + correction * w1),
        eta * (x2 + correction * w2),
    )
end

function _q3_nt_apply_w!(destination::AbstractVector,
                          w::AbstractVector,
                          eta,
                          x::AbstractVector)
    _q3_nt_require_vector(destination, "W destination")
    _q3_nt_require_vector(w, "normalized w")
    _q3_nt_require_vector(x, "W input")
    y0, y1, y2 = _q3_nt_apply_w_coordinates(
        w[1], w[2], w[3], eta, x[1], x[2], x[3],
    )
    destination[1] = y0
    destination[2] = y1
    destination[3] = y2
    return destination
end

"""Apply the inverse symmetric Q3 Nesterov--Todd scaling map `W^-1`."""
@inline function _q3_nt_apply_winv_coordinates(
    w0, w1, w2, eta, x0, x1, x2,
)
    tail_dot = w1 * x1 + w2 * x2
    correction = -x0 + tail_dot / (one(w0) + w0)
    return (
        (w0 * x0 - tail_dot) / eta,
        (x1 + correction * w1) / eta,
        (x2 + correction * w2) / eta,
    )
end

function _q3_nt_apply_winv!(destination::AbstractVector,
                             w::AbstractVector,
                             eta,
                             x::AbstractVector)
    _q3_nt_require_vector(destination, "W inverse destination")
    _q3_nt_require_vector(w, "normalized w")
    _q3_nt_require_vector(x, "W inverse input")
    y0, y1, y2 = _q3_nt_apply_winv_coordinates(
        w[1], w[2], w[3], eta, x[1], x[2], x[3],
    )
    destination[1] = y0
    destination[2] = y1
    destination[3] = y2
    return destination
end

"""Apply `H_s^-1=(W'W)^-1` without forming a dense matrix."""
@inline function _q3_nt_apply_hs_inverse_coordinates(
    w0, w1, w2, eta_squared, x0, x1, x2,
)
    # For w'Jw=1,
    #   (2ww' - J)^-1 = 2Jw*w'J - J.
    # Keeping this closed form avoids two scaling-map passes in every local
    # RHS and direction recovery.
    lorentz_dot = w0 * x0 - w1 * x1 - w2 * x2
    return (
        (2 * w0 * lorentz_dot - x0) / eta_squared,
        (-2 * w1 * lorentz_dot + x1) / eta_squared,
        (-2 * w2 * lorentz_dot + x2) / eta_squared,
    )
end

function _q3_nt_apply_hs_inverse!(destination::AbstractVector,
                                   w::AbstractVector,
                                   eta_squared,
                                   x::AbstractVector)
    _q3_nt_require_vector(destination, "Hs inverse destination")
    _q3_nt_require_vector(w, "normalized w")
    _q3_nt_require_vector(x, "Hs inverse input")
    y0, y1, y2 = _q3_nt_apply_hs_inverse_coordinates(
        w[1], w[2], w[3], eta_squared, x[1], x[2], x[3],
    )
    destination[1] = y0
    destination[2] = y1
    destination[3] = y2
    return destination
end

"""Lorentz Jordan product `(a0,a)*(b0,b)`."""
@inline function _q3_jordan_product_coordinates(
    a0, a1, a2, b0, b1, b2,
)
    return (
        a0 * b0 + a1 * b1 + a2 * b2,
        a0 * b1 + b0 * a1,
        a0 * b2 + b0 * a2,
    )
end

function _q3_jordan_product!(destination::AbstractVector,
                              left::AbstractVector,
                              right::AbstractVector)
    _q3_nt_require_vector(destination, "Jordan-product destination")
    _q3_nt_require_vector(left, "Jordan-product left input")
    _q3_nt_require_vector(right, "Jordan-product right input")
    y0, y1, y2 = _q3_jordan_product_coordinates(
        left[1], left[2], left[3], right[1], right[2], right[3],
    )
    destination[1] = y0
    destination[2] = y1
    destination[3] = y2
    return destination
end

"""Solve `left o result = right` in the Q3 Jordan algebra."""
@inline function _q3_jordan_solve_coordinates(
    left0, left1, left2, right0, right1, right2,
)
    determinant = left0 * left0 - left1 * left1 - left2 * left2
    tail_dot = left1 * right1 + left2 * right2
    result0 = (left0 * right0 - tail_dot) / determinant
    tail_scale = (tail_dot / left0 - right0) / determinant
    return (
        result0,
        tail_scale * left1 + right1 / left0,
        tail_scale * left2 + right2 / left0,
    )
end

function _q3_jordan_solve!(destination::AbstractVector,
                            left::AbstractVector,
                            right::AbstractVector)
    _q3_nt_require_vector(destination, "Jordan-solve destination")
    _q3_nt_require_vector(left, "Jordan-solve left input")
    _q3_nt_require_vector(right, "Jordan-solve right input")
    y0, y1, y2 = _q3_jordan_solve_coordinates(
        left[1], left[2], left[3], right[1], right[2], right[3],
    )
    destination[1] = y0
    destination[2] = y1
    destination[3] = y2
    return destination
end

# Descriptive aliases retained for callers experimenting with the standalone
# kernels.  The mutating forms above are the allocation-conscious API.
const _q3_nt_scale! = _q3_nt_scaling!
const _q3_nesterov_todd_scaling! = _q3_nt_scaling!
const _q3_soc_nt_scaling! = _q3_nt_scaling!
const _q3_nt_build_hs! = _q3_nt_hs!
const _q3_hs! = _q3_nt_hs!
const _q3_apply_hs! = _q3_nt_apply_hs!
const _q3_apply_hs_inverse! = _q3_nt_apply_hs_inverse!
