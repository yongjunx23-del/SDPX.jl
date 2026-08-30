# Typed state shared by the three-dimensional Fenchel-conjugate oracles.
#
# The existing `exp_dual_*` and `power_dual_*` functions evaluate valid
# barriers obtained by pulling the primal barrier back through an exact
# linear cone map.  They are deliberately not represented by these types:
# the scaling theory needs the Fenchel conjugate, whose shadow is defined by
#
#     -gradient(F, shadow) = dual_point.
#
# Expected numerical failures use the isbits status/reason values below.  A
# fixed-width workspace owns every vector and 3x3 matrix needed by the damped
# Newton solve, so a warmed successful call does not allocate.

abstract type NonsymmetricConjugateTag end

"""Fenchel-conjugate tag for the primal exponential-cone barrier."""
struct ExpConjugateTag <: NonsymmetricConjugateTag end

"""Fenchel-conjugate tag for the primal power-cone barrier."""
struct PowerConjugateTag{T} <: NonsymmetricConjugateTag
    alpha::T
end

@enum NonsymmetricConjugateStatus::UInt8 begin
    NS_CONJUGATE_SUCCESS = 0x00
    NS_CONJUGATE_FAILED = 0x01
end

@enum NonsymmetricConjugateReason::UInt8 begin
    NS_CONJUGATE_CONVERGED = 0x00
    NS_CONJUGATE_INVALID_PARAMETER = 0x01
    NS_CONJUGATE_NONFINITE_DUAL = 0x02
    NS_CONJUGATE_DUAL_NOT_INTERIOR = 0x03
    NS_CONJUGATE_PRIMAL_SEED_FAILED = 0x04
    NS_CONJUGATE_BARRIER_FAILED = 0x05
    NS_CONJUGATE_HESSIAN_NOT_SPD = 0x06
    NS_CONJUGATE_STEP_BOUND_FAILED = 0x07
    NS_CONJUGATE_BACKTRACK_LIMIT = 0x08
    NS_CONJUGATE_ITERATION_LIMIT = 0x09
    NS_CONJUGATE_INVERSE_HESSIAN_FAILED = 0x0a
    NS_CONJUGATE_NONFINITE_RESULT = 0x0b
    NS_CONJUGATE_ROOT_RESOLUTION_LIMIT = 0x0c
    NS_CONJUGATE_FACTOR_FAILED = 0x0d
    NS_CONJUGATE_FACTOR_MISMATCH = 0x0e
end

@enum NonsymmetricConjugateSeedMode::UInt8 begin
    NS_CONJUGATE_MAPPED_COLD_SEED = 0x00
    NS_CONJUGATE_PREDICTED_WARM_SEED = 0x01
end

"""Allocation-free result record returned by `conjugate_shadow!`."""
struct NonsymmetricConjugateResult{T}
    status::NonsymmetricConjugateStatus
    reason::NonsymmetricConjugateReason
    iterations::Int
    backtracks::Int
    residual::T
    step::T
    seed_mode::NonsymmetricConjugateSeedMode
    restored::Bool
end

"""Numerical policy for one deterministic three-dimensional Newton solve."""
struct NonsymmetricConjugateSettings{T}
    residual_tolerance::T
    armijo::T
    step_safety::T
    max_iterations::Int
    max_backtracks::Int
    max_bisections::Int
end

@inline _ns_conjugate_default_bisections(::Type{T}) where {T} = 512
@inline _ns_conjugate_default_bisections(::Type{BigFloat}) =
    max(512, precision(BigFloat) + precision(BigFloat))

function NonsymmetricConjugateSettings(
    ::Type{T};
    residual_tolerance=T(256) * eps(one(T)),
    armijo=inv(T(10_000)),
    step_safety=T(99) / T(100),
    max_iterations::Integer=64,
    max_backtracks::Integer=64,
    max_bisections::Integer=_ns_conjugate_default_bisections(T),
) where {T<:AbstractFloat}
    return NonsymmetricConjugateSettings{T}(
        convert(T, residual_tolerance),
        convert(T, armijo),
        convert(T, step_safety),
        Int(max_iterations),
        Int(max_backtracks),
        Int(max_bisections),
    )
end

"""
    NonsymmetricConjugateWorkspace{T}

Preallocated inverse-gradient state.  `valid` certifies the current `shadow`,
diagnostic primal `hessian`, and authoritative `hessian_factor`;
`inverse_valid` separately certifies `inverse_hessian`.  A successful public
`conjugate_shadow!` sets both flags, whereas strict double-secant scaling
intentionally commits only the former.  The corresponding accepted flags
prevent a restored shadow or factor from ever silently belonging to a
different dual point.
`mapped_gradient` exists only to form a primal-interior cold seed from the
already validated mapped dual barrier; it is never returned as a Fenchel
shadow and never supplies the conjugate Hessian.
"""
mutable struct NonsymmetricConjugateWorkspace{T}
    settings::NonsymmetricConjugateSettings{T}
    shadow::Vector{T}
    gradient::Vector{T}
    mapped_gradient::Vector{T}
    residual::Vector{T}
    direction::Vector{T}
    trial::Vector{T}
    trial_gradient::Vector{T}
    hessian::Matrix{T}
    hessian_factor::Matrix{T}
    factor::Matrix{T}
    inverse_hessian::Matrix{T}
    accepted_dual::Vector{T}
    accepted_shadow::Vector{T}
    warm_shadow::Vector{T}
    accepted_hessian::Matrix{T}
    accepted_hessian_factor::Matrix{T}
    accepted_inverse_hessian::Matrix{T}
    hessian_factor_error::T
    accepted_hessian_factor_error::T
    gap::T
    accepted_gap::T
    accepted_valid::Bool
    accepted_hessian_factor_valid::Bool
    accepted_inverse_valid::Bool
    last_seed_mode::NonsymmetricConjugateSeedMode
    valid::Bool
    hessian_factor_valid::Bool
    inverse_valid::Bool
    root_resolution_limited::Bool
    root_certified_limited::Bool
end

function NonsymmetricConjugateWorkspace(
    ::Type{T}; kwargs...,
) where {T<:AbstractFloat}
    settings = NonsymmetricConjugateSettings(T; kwargs...)
    return NonsymmetricConjugateWorkspace{T}(
        settings,
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3, 3),
        zeros(T, 3, 3),
        zeros(T, 3, 3),
        zeros(T, 3, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3),
        zeros(T, 3, 3),
        zeros(T, 3, 3),
        zeros(T, 3, 3),
        T(Inf),
        T(Inf),
        zero(T),
        zero(T),
        false,
        false,
        false,
        NS_CONJUGATE_MAPPED_COLD_SEED,
        false,
        false,
        false,
        false,
        false,
    )
end

NonsymmetricConjugateWorkspace{T}(; kwargs...) where {T<:AbstractFloat} =
    NonsymmetricConjugateWorkspace(T; kwargs...)

# ---------------------------------------------------------------------------
# Three-dimensional primal-dual scaling state
# ---------------------------------------------------------------------------

abstract type NonsymmetricScalingPolicy end

"""Require the full double-secant/BFGS construction; never change provider."""
struct StrictDoubleSecantScaling <: NonsymmetricScalingPolicy end

"""Explicitly permit a recorded one-secant conjugate-Hessian fallback."""
struct DoubleSecantWithDualHessianFallback <: NonsymmetricScalingPolicy end

"""Skip double-secant construction and require the certified dual-Hessian path."""
struct ForcedDualHessianScaling <: NonsymmetricScalingPolicy end

@enum NonsymmetricScalingStatus::UInt8 begin
    NS_SCALING_DOUBLE_SECANT = 0x00
    NS_SCALING_DUAL_HESSIAN_FALLBACK = 0x01
    NS_SCALING_FAILED = 0x02
end

@enum NonsymmetricScalingReason::UInt8 begin
    NS_SCALING_CONVERGED = 0x00
    NS_SCALING_NO_FALLBACK = 0x01
    NS_SCALING_INVALID_PARAMETER = 0x02
    NS_SCALING_NONFINITE_INPUT = 0x03
    NS_SCALING_PRIMAL_NOT_INTERIOR = 0x04
    NS_SCALING_DUAL_NOT_INTERIOR = 0x05
    NS_SCALING_NONPOSITIVE_PAIRING = 0x06
    NS_SCALING_CONJUGATE_FAILED = 0x07
    NS_SCALING_SHADOW_IDENTITY_FAILED = 0x08
    NS_SCALING_GRAM_NONSYMMETRIC = 0x09
    NS_SCALING_SECOND_SECANT_DEGENERATE = 0x0a
    NS_SCALING_AXIS_DEGENERATE = 0x0b
    NS_SCALING_AXIS_PAIRING_DEGENERATE = 0x0c
    NS_SCALING_BFGS_DENOMINATOR = 0x0d
    NS_SCALING_BFGS_NOT_SPD = 0x0e
    NS_SCALING_AXIS_COEFFICIENT = 0x0f
    NS_SCALING_METRIC_NOT_SPD = 0x10
    NS_SCALING_SECANT_MISMATCH = 0x11
    NS_SCALING_INVERSE_MISMATCH = 0x12
    NS_SCALING_FALLBACK_DENOMINATOR = 0x13
    NS_SCALING_FALLBACK_NOT_SPD = 0x14
    NS_SCALING_FALLBACK_SECANT_MISMATCH = 0x15
    NS_SCALING_NONFINITE_RESULT = 0x16
    NS_SCALING_FORCED_DUAL_HESSIAN = 0x17
end

struct NonsymmetricScalingResult{T}
    status::NonsymmetricScalingStatus
    reason::NonsymmetricScalingReason
    fallback_reason::NonsymmetricScalingReason
    conjugate_reason::NonsymmetricConjugateReason
    mu::T
    secant_error::T
    inverse_error::T
end

struct NonsymmetricScalingSettings{T}
    validation_tolerance::T
    degeneracy_tolerance::T
end

function NonsymmetricScalingSettings(
    ::Type{T};
    validation_tolerance=T(8192) * eps(one(T)),
    degeneracy_tolerance=T(128) * eps(one(T)),
) where {T<:AbstractFloat}
    return NonsymmetricScalingSettings{T}(
        convert(T, validation_tolerance),
        convert(T, degeneracy_tolerance),
    )
end

"""Preallocated state for one Exp/Power double-secant scaling block."""
mutable struct NonsymmetricScalingWorkspace{
    T,
    CW<:NonsymmetricConjugateWorkspace{T},
}
    settings::NonsymmetricScalingSettings{T}
    conjugate::CW
    primal::Vector{T}
    dual::Vector{T}
    dual_shadow::Vector{T}
    primal_gradient::Vector{T}
    axis_z::Vector{T}
    axis_r::Vector{T}
    work1::Vector{T}
    work2::Vector{T}
    work3::Vector{T}
    primal_hessian::Matrix{T}
    g0::Matrix{T}
    g_bfgs::Matrix{T}
    g::Matrix{T}
    theta::Matrix{T}
    work_matrix::Matrix{T}
    factor::Matrix{T}
    mu::T
    valid::Bool
    used_fallback::Bool
    last_status::NonsymmetricScalingStatus
    last_reason::NonsymmetricScalingReason
    last_fallback_reason::NonsymmetricScalingReason
end

function NonsymmetricScalingWorkspace(
    ::Type{T};
    validation_tolerance=T(8192) * eps(one(T)),
    degeneracy_tolerance=T(128) * eps(one(T)),
    conjugate_kwargs...,
) where {T<:AbstractFloat}
    settings = NonsymmetricScalingSettings(
        T;
        validation_tolerance=validation_tolerance,
        degeneracy_tolerance=degeneracy_tolerance,
    )
    conjugate = NonsymmetricConjugateWorkspace(T; conjugate_kwargs...)
    z3() = zeros(T, 3)
    m3() = zeros(T, 3, 3)
    return NonsymmetricScalingWorkspace{T,typeof(conjugate)}(
        settings,
        conjugate,
        z3(), z3(), z3(), z3(), z3(), z3(), z3(), z3(), z3(),
        m3(), m3(), m3(), m3(), m3(), m3(), m3(),
        zero(T), false, false, NS_SCALING_FAILED,
        NS_SCALING_INVALID_PARAMETER, NS_SCALING_NO_FALLBACK,
    )
end

NonsymmetricScalingWorkspace{T}(; kwargs...) where {T<:AbstractFloat} =
    NonsymmetricScalingWorkspace(T; kwargs...)
