#=====================================================================#
#    Allocation-free dense 3x3 reference kernels for nonsymmetric cones.
#
#    These kernels are deliberately independent of the production HSD
#    scaling implementation.  They provide a small, auditable full-Hessian
#    reference for the exponential- and power-cone barrier oracles.
#=====================================================================#

@inline function _require_dense3_matrix(matrix, name::AbstractString)
    size(matrix) == (3, 3) || throw(DimensionMismatch(
        "$name must be exactly 3x3, got $(size(matrix))",
    ))
    return nothing
end

@inline function _require_dense3_vector(vector, name::AbstractString)
    length(vector) == 3 || throw(DimensionMismatch(
        "$name must have length 3, got $(length(vector))",
    ))
    return nothing
end

"""Roundoff band for comparisons of two logarithmic cone boundaries."""
@inline function _nonsymmetric_log_tolerance(left, right)
    scale = max(one(left), abs(left), abs(right))
    eight = one(scale) + one(scale)
    eight += eight
    eight += eight
    return eight * eps(one(scale)) * scale
end

"""
    nonsymmetric_hessian_product!(destination, hessian, vector)

Allocation-free product of an explicit 3x3 nonsymmetric-cone barrier Hessian
with a three-vector. `destination` may alias `vector`.
"""
function nonsymmetric_hessian_product!(destination, hessian, vector)
    _require_dense3_vector(destination, "destination")
    _require_dense3_matrix(hessian, "hessian")
    _require_dense3_vector(vector, "vector")
    v1, v2, v3 = vector[1], vector[2], vector[3]
    r1 = hessian[1, 1] * v1 + hessian[1, 2] * v2 + hessian[1, 3] * v3
    r2 = hessian[2, 1] * v1 + hessian[2, 2] * v2 + hessian[2, 3] * v3
    r3 = hessian[3, 1] * v1 + hessian[3, 2] * v2 + hessian[3, 3] * v3
    destination[1] = r1
    destination[2] = r2
    destination[3] = r3
    return destination
end

"""
    nonsymmetric_hessian_solve!(destination, hessian, rhs, cholesky_storage)

Solve the SPD 3x3 system `hessian * destination = rhs` using an explicit
lower Cholesky factor stored in caller-owned `cholesky_storage`.  No factor or
work vector is allocated. `destination` may alias `rhs`. Non-finite or
non-positive pivots fail closed with `ArgumentError`.
"""
function nonsymmetric_hessian_solve!(
    destination,
    hessian,
    rhs,
    cholesky_storage,
)
    _require_dense3_vector(destination, "destination")
    _require_dense3_matrix(hessian, "hessian")
    _require_dense3_vector(rhs, "rhs")
    _require_dense3_matrix(cholesky_storage, "cholesky_storage")

    h11 = hessian[1, 1]
    isfinite(h11) && h11 > zero(h11) || throw(ArgumentError(
        "nonsymmetric Hessian is not finite SPD at pivot 1",
    ))
    l11 = sqrt(h11)
    l21 = hessian[2, 1] / l11
    l31 = hessian[3, 1] / l11

    pivot2 = hessian[2, 2] - l21 * l21
    isfinite(pivot2) && pivot2 > zero(pivot2) || throw(ArgumentError(
        "nonsymmetric Hessian is not finite SPD at pivot 2",
    ))
    l22 = sqrt(pivot2)
    l32 = (hessian[3, 2] - l31 * l21) / l22

    pivot3 = hessian[3, 3] - l31 * l31 - l32 * l32
    isfinite(pivot3) && pivot3 > zero(pivot3) || throw(ArgumentError(
        "nonsymmetric Hessian is not finite SPD at pivot 3",
    ))
    l33 = sqrt(pivot3)

    z = zero(l11)
    cholesky_storage[1, 1] = l11
    cholesky_storage[1, 2] = z
    cholesky_storage[1, 3] = z
    cholesky_storage[2, 1] = l21
    cholesky_storage[2, 2] = l22
    cholesky_storage[2, 3] = z
    cholesky_storage[3, 1] = l31
    cholesky_storage[3, 2] = l32
    cholesky_storage[3, 3] = l33

    b1, b2, b3 = rhs[1], rhs[2], rhs[3]
    y1 = b1 / l11
    y2 = (b2 - l21 * y1) / l22
    y3 = (b3 - l31 * y1 - l32 * y2) / l33
    x3 = y3 / l33
    x2 = (y2 - l32 * x3) / l22
    x1 = (y1 - l21 * x2 - l31 * x3) / l11
    all(isfinite, (x1, x2, x3)) || throw(ArgumentError(
        "nonsymmetric Hessian solve produced a non-finite result",
    ))
    destination[1] = x1
    destination[2] = x2
    destination[3] = x3
    return destination
end
