#=====================================================================#
#    Typed rotated second-order cone -> Lorentz cone transform (P1b).
#
#    This is the orthogonal convention used by `ir/canonical.jl`:
#
#       M(u,v,w) = ((u+v)/sqrt(2), (u-v)/sqrt(2), w).
#
#    M is symmetric and involutory.  We nevertheless store the primal map,
#    its inverse, the dual inverse-adjoint, and the dual adjoint as explicit
#    fields.  This makes the pairing contract visible and prevents a future
#    non-orthogonal convention from accidentally reusing the wrong map.
#=====================================================================#

"""A typed orthogonal map from `RotatedLorentzCone` coordinates to SOC."""
struct RotatedSOCToSOC{T<:AbstractFloat} <: AbstractConeTransform{T}
    dimension::Int
    precision_bits::Int
    primal_map::Matrix{T}
    inverse_primal_map::Matrix{T}
    dual_inverse_adjoint::Matrix{T}
    dual_adjoint::Matrix{T}
    pairing_scale::T
end

@inline function _rsoc_transform_precision_bits(::Type{T}) where {T<:AbstractFloat}
    return T === BigFloat ? precision(BigFloat) : 53
end

function _rsoc_transform_map(::Type{T}, dimension::Int, bits::Int) where {T<:AbstractFloat}
    dimension >= 3 || throw(ArgumentError(
        "RotatedSOCToSOC requires dimension >= 3, got $dimension",
    ))
    if T === BigFloat
        return setprecision(BigFloat, bits) do
            inv_sqrt_two = inv(sqrt(BigFloat(2)))
            one_value = BigFloat(1)
            zero_value = BigFloat(0)
            map = fill(zero_value, dimension, dimension)
            map[1, 1] = inv_sqrt_two
            map[1, 2] = inv_sqrt_two
            map[2, 1] = inv_sqrt_two
            map[2, 2] = -inv_sqrt_two
            for index in 3:dimension
                map[index, index] = one_value
            end
            map
        end
    end
    inv_sqrt_two = inv(sqrt(T(2)))
    map = zeros(T, dimension, dimension)
    map[1, 1] = inv_sqrt_two
    map[1, 2] = inv_sqrt_two
    map[2, 1] = inv_sqrt_two
    map[2, 2] = -inv_sqrt_two
    for index in 3:dimension
        map[index, index] = one(T)
    end
    return map
end

"""
    RotatedSOCToSOC{T}(dimension; precision_bits=...)

Construct the exact (at the selected arithmetic precision) orthogonal
RSOC-to-SOC isometry.  The canonical primal and dual maps are
`M` and `M⁻ᵀ`, respectively.  Since this convention has `M=M⁻¹=Mᵀ`,
all four stored maps currently have the same entries, but they remain
separate named fields to make their mathematical roles explicit.
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
    map = _rsoc_transform_map(T, dimension_int, bits)
    # These copies are intentional: each field is an independently owned
    # contract object, so later non-orthogonal transforms cannot accidentally
    # alias and mutate another side of the pairing map.
    inverse = copy(map)
    dual_inverse_adjoint = copy(map)
    dual_adjoint = copy(map)
    pairing_scale = if T === BigFloat
        setprecision(BigFloat, bits) do
            BigFloat(1)
        end
    else
        one(T)
    end
    return RotatedSOCToSOC{T}(
        dimension_int,
        bits,
        map,
        inverse,
        dual_inverse_adjoint,
        dual_adjoint,
        pairing_scale,
    )
end

RotatedSOCToSOC(dimension::Integer, ::Type{T}; kwargs...) where {T<:AbstractFloat} =
    RotatedSOCToSOC{T}(dimension; kwargs...)

@inline function _rsoc_transform_apply!(destination, map, source, dimension::Int)
    length(source) == dimension || throw(DimensionMismatch(
        "RSOC transform source length $(length(source)) != $dimension",
    ))
    length(destination) == dimension || throw(DimensionMismatch(
        "RSOC transform destination length $(length(destination)) != $dimension",
    ))
    @inbounds for row in 1:dimension
        value = zero(eltype(map))
        for column in 1:dimension
            coefficient = map[row, column]
            iszero(coefficient) && continue
            value += coefficient * source[column]
        end
        destination[row] = value
    end
    return destination
end

"""Map an original RSOC primal vector into canonical SOC coordinates."""
function forward_primal!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform.primal_map, source, transform.dimension)
end

"""Reconstruct an original RSOC primal vector from canonical SOC coordinates."""
function backward_primal!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform.inverse_primal_map, source, transform.dimension)
end

"""Map an original RSOC dual vector into canonical SOC dual coordinates."""
function forward_dual!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(
        destination, transform.dual_inverse_adjoint, source, transform.dimension,
    )
end

"""Reconstruct an original RSOC dual vector from canonical SOC dual coordinates."""
function backward_dual!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform.dual_adjoint, source, transform.dimension)
end

"""Reconstruct an original primal infeasibility ray."""
function backward_primal_ray!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform.inverse_primal_map, source, transform.dimension)
end

"""Reconstruct an original dual infeasibility ray."""
function backward_dual_ray!(
    transform::RotatedSOCToSOC{T}, destination::AbstractVector{T}, source::AbstractVector{T},
) where {T<:AbstractFloat}
    return _rsoc_transform_apply!(destination, transform.dual_adjoint, source, transform.dimension)
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
    atol=nothing, rtol=nothing,
) where {T<:AbstractFloat}
    canonical_primal = similar(primal)
    canonical_dual = similar(dual)
    forward_primal!(transform, canonical_primal, primal)
    forward_dual!(transform, canonical_dual, dual)
    original_pairing = dot(primal, dual)
    canonical_pairing = transform.pairing_scale * dot(canonical_primal, canonical_dual)
    return _rsoc_transform_isapprox(
        [original_pairing], [canonical_pairing]; atol=atol, rtol=rtol,
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
    canonical_A = transform.primal_map * A
    canonical_dual = similar(dual)
    forward_dual!(transform, canonical_dual, dual)
    original_residual = transpose(A) * dual + objective
    canonical_residual = transpose(canonical_A) * canonical_dual + objective
    return _rsoc_transform_isapprox(
        original_residual, canonical_residual; atol=atol, rtol=rtol,
    )
end
