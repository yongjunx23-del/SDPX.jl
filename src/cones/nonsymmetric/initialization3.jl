# Type-native initialization for one three-dimensional nonsymmetric block.
#
# The default seeds are deliberately noncentral.  Besides lying strictly in
# the primal and dual cones, they exercise the strict double-secant scaling
# path instead of its centered-point dual-Hessian fallback:
#
#   Exp:       s = (0,1,2),       y = (-1,1,1),
#   Power(a):  s = (1,1,0),       y = (a,1-a,1/2).
#
# Every scalar is formed from zero(T), one(T), the target-typed alpha, and
# arithmetic in T.  In particular, a Power tag whose alpha has another scalar
# type is rejected rather than converted.  The workspace aliases the primal
# and dual buffers owned by its scaling workspace, so a warmed fixed-width
# initialization performs no allocation and leaves a directly consumable
# validated scaling state.

@enum NonsymmetricInitializationStatus::UInt8 begin
    NS_INITIALIZATION_READY = 0x00
    NS_INITIALIZATION_FAILED = 0x01
end

@enum NonsymmetricInitializationReason::UInt8 begin
    NS_INITIALIZATION_CONVERGED = 0x00
    NS_INITIALIZATION_INVALID_TAG = 0x01
    NS_INITIALIZATION_TYPE_MISMATCH = 0x02
    NS_INITIALIZATION_INVALID_ALPHA = 0x03
    NS_INITIALIZATION_INVALID_STORAGE = 0x04
    NS_INITIALIZATION_NONFINITE_INPUT = 0x05
    NS_INITIALIZATION_PRIMAL_NOT_INTERIOR = 0x06
    NS_INITIALIZATION_DUAL_NOT_INTERIOR = 0x07
    NS_INITIALIZATION_NONPOSITIVE_PAIRING = 0x08
    NS_INITIALIZATION_SCALING_FAILED = 0x09
    NS_INITIALIZATION_INVALID_SETTINGS = 0x0a
end

struct NonsymmetricInitializationResult{T}
    status::NonsymmetricInitializationStatus
    reason::NonsymmetricInitializationReason
    scaling_status::NonsymmetricScalingStatus
    scaling_reason::NonsymmetricScalingReason
    pairing::T
    mu::T
end

mutable struct NonsymmetricInitializationWorkspace{
    T,
    SW<:NonsymmetricScalingWorkspace{T},
}
    primal::Vector{T}
    dual::Vector{T}
    scaling::SW
    valid::Bool
    last_status::NonsymmetricInitializationStatus
    last_reason::NonsymmetricInitializationReason
end

@inline _ns_initialization_default_bisections(::Type{T}) where {T} = 512
@inline _ns_initialization_default_bisections(::Type{BigFloat}) =
    max(512, precision(BigFloat) + precision(BigFloat))

function NonsymmetricInitializationWorkspace(
    ::Type{T};
    max_bisections::Integer=_ns_initialization_default_bisections(T),
    scaling_kwargs...,
) where {T<:AbstractFloat}
    scaling = NonsymmetricScalingWorkspace(
        T; max_bisections=max_bisections, scaling_kwargs...,
    )
    return NonsymmetricInitializationWorkspace{T,typeof(scaling)}(
        scaling.primal,
        scaling.dual,
        scaling,
        false,
        NS_INITIALIZATION_FAILED,
        NS_INITIALIZATION_INVALID_TAG,
    )
end

NonsymmetricInitializationWorkspace{T}(; kwargs...) where {T<:AbstractFloat} =
    NonsymmetricInitializationWorkspace(T; kwargs...)

@inline function _ns_initialization_result(
    workspace::NonsymmetricInitializationWorkspace{T},
    status::NonsymmetricInitializationStatus,
    reason::NonsymmetricInitializationReason,
    scaling_status::NonsymmetricScalingStatus,
    scaling_reason::NonsymmetricScalingReason,
    pairing::T,
    mu::T,
) where {T}
    workspace.valid = status === NS_INITIALIZATION_READY
    workspace.last_status = status
    workspace.last_reason = reason
    return NonsymmetricInitializationResult{T}(
        status,
        reason,
        scaling_status,
        scaling_reason,
        pairing,
        mu,
    )
end

@inline function _ns_initialization_failure(
    workspace::NonsymmetricInitializationWorkspace{T},
    reason::NonsymmetricInitializationReason;
    scaling_status::NonsymmetricScalingStatus=NS_SCALING_FAILED,
    scaling_reason::NonsymmetricScalingReason=NS_SCALING_INVALID_PARAMETER,
    pairing::T=zero(T),
) where {T}
    workspace.scaling.valid = false
    return _ns_initialization_result(
        workspace,
        NS_INITIALIZATION_FAILED,
        reason,
        scaling_status,
        scaling_reason,
        pairing,
        zero(T),
    )
end

@inline _ns_initialization_tag_reason(
    ::Type{T}, ::ExpConjugateTag,
) where {T} = NS_INITIALIZATION_CONVERGED

@inline function _ns_initialization_tag_reason(
    ::Type{T}, tag::PowerConjugateTag{T},
) where {T}
    alpha = tag.alpha
    return isfinite(alpha) && zero(T) < alpha < one(T) ?
           NS_INITIALIZATION_CONVERGED : NS_INITIALIZATION_INVALID_ALPHA
end

@inline _ns_initialization_tag_reason(
    ::Type, ::PowerConjugateTag,
) = NS_INITIALIZATION_TYPE_MISMATCH

@inline _ns_initialization_tag_reason(
    ::Type, ::Any,
) = NS_INITIALIZATION_INVALID_TAG

@inline function _ns_initialization_seed!(
    workspace::NonsymmetricInitializationWorkspace{T},
    ::ExpConjugateTag,
) where {T}
    zero_t = zero(T)
    one_t = one(T)
    two_t = one_t + one_t
    _store_owned_scalar!(workspace.primal, 1, zero_t)
    _store_owned_scalar!(workspace.primal, 2, one_t)
    _store_owned_scalar!(workspace.primal, 3, two_t)
    _store_owned_scalar!(workspace.dual, 1, -one_t)
    _store_owned_scalar!(workspace.dual, 2, one_t)
    _store_owned_scalar!(workspace.dual, 3, one_t)
    return nothing
end

@inline function _ns_initialization_seed!(
    workspace::NonsymmetricInitializationWorkspace{T},
    tag::PowerConjugateTag{T},
) where {T}
    zero_t = zero(T)
    one_t = one(T)
    alpha = tag.alpha
    _store_owned_scalar!(workspace.primal, 1, one_t)
    _store_owned_scalar!(workspace.primal, 2, one_t)
    _store_owned_scalar!(workspace.primal, 3, zero_t)
    _store_owned_scalar!(workspace.dual, 1, alpha)
    _store_owned_scalar!(workspace.dual, 2, one_t - alpha)
    _store_owned_scalar!(workspace.dual, 3, inv(one_t + one_t))
    return nothing
end

@inline _ns_initialization_storage3(::Any) = false
@inline _ns_initialization_storage3(values::Tuple) = length(values) == 3
@inline _ns_initialization_storage3(values::AbstractVector) = length(values) == 3

@inline function _ns_initialization_target_coordinates(
    ::Type{T}, values,
) where {T}
    return values[1] isa T && values[2] isa T && values[3] isa T
end

@inline function _ns_initialization_finite3(values)
    return isfinite(values[1]) && isfinite(values[2]) && isfinite(values[3])
end

@inline function _ns_initialization_primal_interior(
    ::ExpConjugateTag, primal,
)
    return exp_primal_interior(primal[1], primal[2], primal[3])
end

@inline function _ns_initialization_primal_interior(
    tag::PowerConjugateTag, primal,
)
    return power_primal_interior(
        primal[1], primal[2], primal[3], tag.alpha,
    )
end

@inline function _ns_initialization_dual_interior(
    ::ExpConjugateTag, dual,
)
    return exp_dual_interior(dual[1], dual[2], dual[3])
end

@inline function _ns_initialization_dual_interior(
    tag::PowerConjugateTag, dual,
)
    return power_dual_interior(dual[1], dual[2], dual[3], tag.alpha)
end

@inline function _ns_initialization_validate_loaded!(
    workspace::NonsymmetricInitializationWorkspace{T},
    tag::NonsymmetricConjugateTag,
) where {T}
    primal = workspace.primal
    dual = workspace.dual
    _ns_initialization_finite3(primal) &&
        _ns_initialization_finite3(dual) ||
        return _ns_initialization_failure(
            workspace, NS_INITIALIZATION_NONFINITE_INPUT,
        )
    _ns_initialization_primal_interior(tag, primal) ||
        return _ns_initialization_failure(
            workspace, NS_INITIALIZATION_PRIMAL_NOT_INTERIOR,
        )
    _ns_initialization_dual_interior(tag, dual) ||
        return _ns_initialization_failure(
            workspace, NS_INITIALIZATION_DUAL_NOT_INTERIOR,
        )
    pairing = _ns_scaling_dot3(primal, dual)
    pairing_scale = max(
        one(T), _ns_scaling_norm3(primal) * _ns_scaling_norm3(dual),
    )
    tolerance = workspace.scaling.settings.degeneracy_tolerance
    isfinite(tolerance) && tolerance > zero(T) ||
        return _ns_initialization_failure(
            workspace, NS_INITIALIZATION_INVALID_SETTINGS;
            pairing=pairing,
        )
    isfinite(pairing) && pairing > tolerance * pairing_scale ||
        return _ns_initialization_failure(
            workspace, NS_INITIALIZATION_NONPOSITIVE_PAIRING;
            pairing=pairing,
        )

    scaling_result = try_update_nonsymmetric_scaling!(
        workspace.scaling,
        StrictDoubleSecantScaling(),
        tag,
        primal,
        dual,
    )
    scaling_result.status === NS_SCALING_DOUBLE_SECANT ||
        return _ns_initialization_failure(
            workspace,
            NS_INITIALIZATION_SCALING_FAILED;
            scaling_status=scaling_result.status,
            scaling_reason=scaling_result.reason,
            pairing=pairing,
        )
    return _ns_initialization_result(
        workspace,
        NS_INITIALIZATION_READY,
        NS_INITIALIZATION_CONVERGED,
        scaling_result.status,
        scaling_result.reason,
        pairing,
        scaling_result.mu,
    )
end

"""
    try_initialize_nonsymmetric_block!(workspace, tag)

Construct a target-type-native strict-interior primal/dual seed and validate a
strict double-secant scaling state.  A Power alpha of another scalar type is
rejected rather than converted.
"""
function try_initialize_nonsymmetric_block!(
    workspace::NonsymmetricInitializationWorkspace{T},
    tag,
) where {T<:AbstractFloat}
    workspace.valid = false
    workspace.scaling.valid = false
    reason = _ns_initialization_tag_reason(T, tag)
    reason === NS_INITIALIZATION_CONVERGED ||
        return _ns_initialization_failure(workspace, reason)
    _ns_initialization_seed!(workspace, tag)
    return _ns_initialization_validate_loaded!(workspace, tag)
end

"""
    try_initialize_nonsymmetric_block!(workspace, tag, primal, dual)

Runtime block entry for a caller-supplied pair.  Coordinates and Power alpha
must already have the workspace scalar type; no implicit precision-losing
conversion is performed.  On success the pair is copied into the workspace and
the same validation/scaling gates as the native default seed are applied.
"""
function try_initialize_nonsymmetric_block!(
    workspace::NonsymmetricInitializationWorkspace{T},
    tag,
    primal,
    dual,
) where {T<:AbstractFloat}
    workspace.valid = false
    workspace.scaling.valid = false
    reason = _ns_initialization_tag_reason(T, tag)
    reason === NS_INITIALIZATION_CONVERGED ||
        return _ns_initialization_failure(workspace, reason)
    _ns_initialization_storage3(primal) &&
        _ns_initialization_storage3(dual) ||
        return _ns_initialization_failure(
            workspace, NS_INITIALIZATION_INVALID_STORAGE,
        )
    _ns_initialization_target_coordinates(T, primal) &&
        _ns_initialization_target_coordinates(T, dual) ||
        return _ns_initialization_failure(
            workspace, NS_INITIALIZATION_TYPE_MISMATCH,
        )
    @inbounds for index in 1:3
        _store_owned_scalar!(workspace.primal, index, primal[index])
        _store_owned_scalar!(workspace.dual, index, dual[index])
    end
    return _ns_initialization_validate_loaded!(workspace, tag)
end
