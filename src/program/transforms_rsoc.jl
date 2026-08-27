#=====================================================================#
#    Typed rotated second-order cone -> Lorentz cone transform (P1b).
#
#    This is the orthogonal convention used by `ir/canonical.jl`:
#
#       M(u,v,w) = ((u+v)/sqrt(2), (u-v)/sqrt(2), w).
#
#    M is symmetric, orthogonal, and involutory, so primal, inverse-primal,
#    dual inverse-adjoint, and dual-adjoint application share one O(n) map.
#=====================================================================#

"""A typed orthogonal map from `RotatedLorentzCone` coordinates to SOC."""
struct RotatedSOCToSOC{T<:AbstractFloat} <: AbstractCoordinateTransform{T}
    dimension::Int
    precision_bits::Int
    inv_sqrt_two::T
    pairing_scale::T
end

@inline function _rsoc_transform_precision_bits(::Type{T}) where {T<:AbstractFloat}
    return precision(T)
end

"""
    RotatedSOCToSOC{T}(dimension; precision_bits=...)

Construct the orthogonal RSOC-to-SOC isometry at the selected arithmetic
precision. Since `M=M⁻¹=Mᵀ`, all primal and dual directions use the same
allocation-free map and pairing scale is exactly one.
"""
function RotatedSOCToSOC{T}(
    dimension::Integer;
    precision_bits::Integer=_rsoc_transform_precision_bits(T),
) where {T<:AbstractFloat}
    dimension_int = Int(dimension)
    dimension_int >= 3 || throw(ArgumentError(
        "RotatedSOCToSOC requires dimension >= 3, got $dimension_int",
    ))
    bits = Int(precision_bits)
    bits > 0 || throw(ArgumentError("precision_bits must be positive, got $bits"))
    inv_sqrt_two, pairing_scale = if T === BigFloat
        setprecision(BigFloat, bits) do
            inv(sqrt(BigFloat(2))), BigFloat(1)
        end
    else
        inv(sqrt(T(2))), one(T)
    end
    pairing_scale == one(pairing_scale) || throw(AssertionError(
        "orthogonal RSOC transform must preserve pairing with scale one",
    ))
    return RotatedSOCToSOC{T}(
        dimension_int, bits, inv_sqrt_two, pairing_scale,
    )
end

RotatedSOCToSOC(dimension::Integer, ::Type{T}; kwargs...) where {T<:AbstractFloat} =
    RotatedSOCToSOC{T}(dimension; kwargs...)

SDPX.dimension(transform::RotatedSOCToSOC) = transform.dimension

@inline function _rsoc_transform_apply!(
    destination, source, dimension::Int, inv_sqrt_two,
)
    length(source) == dimension || throw(DimensionMismatch(
        "RSOC transform source length $(length(source)) != $dimension",
    ))
    length(destination) == dimension || throw(DimensionMismatch(
        "RSOC transform destination length $(length(destination)) != $dimension",
    ))
    # Save the coupled coordinates before either output write so in-place
    # application (`destination === source`) is valid.
    @inbounds begin
        u = source[1]
        v = source[2]
        destination[1] = (u + v) * inv_sqrt_two
        destination[2] = (u - v) * inv_sqrt_two
        for index in 3:dimension
            destination[index] = source[index]
        end
    end
    return destination
end

@inline function _rsoc_transform_apply!(destination, transform::RotatedSOCToSOC, source)
    return _rsoc_transform_apply!(
        destination, source, transform.dimension, transform.inv_sqrt_two,
    )
end

"""Materialize the setup-time row map for canonical sparse assembly only."""
function _rsoc_transform_matrix(transform::RotatedSOCToSOC{T}) where {T}
    matrix = zeros(T, transform.dimension, transform.dimension)
    @inbounds begin
        a = transform.inv_sqrt_two
        matrix[1, 1] = a
        matrix[1, 2] = a
        matrix[2, 1] = a
        matrix[2, 2] = -a
        for index in 3:transform.dimension
            matrix[index, index] = one(T)
        end
    end
    return matrix
end

"""Map an original RSOC primal vector into canonical SOC coordinates."""
function forward_primal!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform, source)
end

"""Reconstruct an original RSOC primal vector from canonical SOC coordinates."""
function backward_primal!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform, source)
end

"""Map an original RSOC dual vector into canonical SOC dual coordinates."""
function forward_dual!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform, source)
end

"""Reconstruct an original RSOC dual vector from canonical SOC coordinates."""
function backward_dual!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform, source)
end

"""Reconstruct an original primal infeasibility ray."""
function backward_primal_ray!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform, source)
end

"""Reconstruct an original dual infeasibility ray."""
function backward_dual_ray!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform, source)
end

"""RSOC coordinate changes do not add a scalar objective constant."""
function objective_shift(transform::RotatedSOCToSOC{T}) where {T<:AbstractFloat}
    return zero(transform.pairing_scale)
end

function objective_shift(
    transform::RotatedSOCToSOC{T}, original_constant::T,
) where {T<:AbstractFloat}
    return original_constant + objective_shift(transform)
end

@inline function _rsoc_transform_isapprox(a, b; atol=nothing, rtol=nothing)
    T = promote_type(eltype(a), eltype(b))
    atol_value = atol === nothing ? sqrt(eps(T)) : atol
    rtol_value = rtol === nothing ? sqrt(eps(T)) : rtol
    return isapprox(a, b; atol=atol_value, rtol=rtol_value)
end

"""
    verify_pairing_invariant(transform, primal, dual; atol, rtol)

Verify `<s,y> == pairing_scale * <T*s,T⁻ᵀ*y>`.  The scale is explicit even
though it is one for the orthogonal convention.
"""
function verify_pairing_invariant(
    transform::RotatedSOCToSOC{T}, primal::AbstractVector{T}, dual::AbstractVector{T};
    atol=nothing, rtol=nothing, tol=nothing,
) where {T<:AbstractFloat}
    canonical_primal = similar(primal)
    canonical_dual = similar(dual)
    forward_primal!(transform, canonical_primal, primal)
    forward_dual!(transform, canonical_dual, dual)
    original_pairing = dot(primal, dual)
    canonical_pairing = transform.pairing_scale * dot(canonical_primal, canonical_dual)
    if tol !== nothing
        atol = atol === nothing ? tol : atol
        rtol = rtol === nothing ? tol : rtol
    end
    return _rsoc_transform_isapprox(
        [original_pairing], [canonical_pairing]; atol=atol, rtol=rtol,
    )
end

function verify_pairing_invariant(
    transform::RotatedSOCToSOC{T}, primal::AbstractVector{T}, dual::AbstractVector{T},
    transformed_primal::AbstractVector{T}, transformed_dual::AbstractVector{T};
    atol=nothing, rtol=nothing, tol=nothing,
) where {T<:AbstractFloat}
    expected_primal = similar(primal)
    expected_dual = similar(dual)
    forward_primal!(transform, expected_primal, primal)
    forward_dual!(transform, expected_dual, dual)
    size(expected_primal) == size(transformed_primal) || throw(DimensionMismatch(
        "transformed primal size mismatch",
    ))
    size(expected_dual) == size(transformed_dual) || throw(DimensionMismatch(
        "transformed dual size mismatch",
    ))
    atol_value, rtol_value = _transform_tolerances(T;
        atol=atol, rtol=rtol, tol=tol,
    )
    all(isapprox.(expected_primal, transformed_primal;
                  atol=atol_value, rtol=rtol_value)) || return false
    all(isapprox.(expected_dual, transformed_dual;
                  atol=atol_value, rtol=rtol_value)) || return false
    return verify_pairing_invariant(
        transform, primal, dual; atol=atol, rtol=rtol, tol=tol,
    )
end

"""
    verify_stationarity_invariant(transform, A, dual, objective; atol, rtol)

For a row transform, compare stationarity before and after replacing
`A` with `T*A` and `y` with `T⁻ᵀ*y`.  This checks the invariant for both
satisfied and deliberately nonzero residuals.
"""
function verify_stationarity_invariant(
    transform::RotatedSOCToSOC{T},
    A::AbstractMatrix{T},
    dual::AbstractVector{T},
    objective::AbstractVector{T};
    atol=nothing,
    rtol=nothing,
) where {T<:AbstractFloat}
    size(A, 1) == transform.dimension || throw(DimensionMismatch(
        "stationarity matrix has $(size(A, 1)) rows, expected $(transform.dimension)",
    ))
    length(dual) == transform.dimension || throw(DimensionMismatch(
        "stationarity dual length $(length(dual)) != $(transform.dimension)",
    ))
    size(A, 2) == length(objective) || throw(DimensionMismatch(
        "stationarity objective length $(length(objective)) != $(size(A, 2))",
    ))
    canonical_A = _rsoc_transform_matrix(transform) * A
    canonical_dual = similar(dual)
    forward_dual!(transform, canonical_dual, dual)
    original_residual = transpose(A) * dual + objective
    canonical_residual = transpose(canonical_A) * canonical_dual + objective
    return _rsoc_transform_isapprox(
        original_residual, canonical_residual; atol=atol, rtol=rtol,
    )
end

function verify_stationarity_invariant(
    transform::RotatedSOCToSOC{T},
    A, c, y, tau, A_hat, c_hat, y_hat;
    atol=nothing, rtol=nothing, tol=nothing,
) where {T<:AbstractFloat}
    size(A, 1) == transform.dimension || throw(DimensionMismatch(
        "stationarity matrix has $(size(A, 1)) rows, expected $(transform.dimension)",
    ))
    size(A_hat) == size(A) || throw(DimensionMismatch(
        "transformed stationarity matrix size mismatch",
    ))
    length(y) == transform.dimension == length(y_hat) || throw(DimensionMismatch(
        "stationarity dual lengths do not match transform dimension",
    ))
    size(A, 2) == length(c) == length(c_hat) || throw(DimensionMismatch(
        "stationarity objective lengths do not match matrix columns",
    ))
    original_residual = transpose(A) * y + tau .* c
    canonical_residual = transpose(A_hat) * y_hat + tau .* c_hat
    return _rsoc_transform_isapprox(
        original_residual, canonical_residual; atol=atol === nothing ? tol : atol,
        rtol=rtol === nothing ? tol : rtol,
    )
end
