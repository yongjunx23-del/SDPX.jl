# Product-cone runtime storage.
#
# This file intentionally contains only setup-time data structures.  A
# runtime groups blocks by cone family so the execution methods in
# `symmetric_api.jl` never dispatch through an abstract cone or construct a
# temporary `view` for a block.

"""One setup-built nonnegative block in a `ProductConeRuntime`."""
struct OrthantRuntimeBlock{T}
    offset::Int
    dim::Int
    cone::SymmetricCones.NonnegativeCone
    state::SymmetricCones.OrthantNTScaling{T}
    primal::Vector{T}
    dual::Vector{T}
    input::Vector{T}
    output::Vector{T}
    direction::Vector{T}
    alpha::Base.RefValue{T}
end
"""One setup-built SOC block in a `ProductConeRuntime`."""
struct SOCRuntimeBlock{T}
    offset::Int
    dim::Int
    cone::SymmetricCones.SOCone
    state::SymmetricCones.SOCNTScaling{T}
    primal::Vector{T}
    dual::Vector{T}
    input::Vector{T}
    output::Vector{T}
    direction::Vector{T}
    alpha::Base.RefValue{T}
end

"""One setup-built PSD block in a `ProductConeRuntime`.

The execution coordinates are `svec` (diagonal scale one, off-diagonal
scale `sqrt(2)`).  The two raw-lower buffers are used only by the legacy PSD
boundary primitive, which consumes raw packed coordinates.
"""
struct PSDRuntimeBlock{T}
    offset::Int
    dim::Int
    len::Int
    cone::SymmetricCones.PSDTriangleCone{T}
    state::SymmetricCones.PSDNTScaling{T}
    primal::Vector{T}
    dual::Vector{T}
    input::Vector{T}
    output::Vector{T}
    direction::Vector{T}
    raw_primal::Vector{T}
    raw_direction::Vector{T}
    alpha::Base.RefValue{T}
end

@enum NonsymmetricRuntimeStatus::UInt8 begin
    NS_RUNTIME_READY = 0x00
    NS_RUNTIME_FAILED = 0x01
end

@enum NonsymmetricRuntimeReason::UInt8 begin
    NS_RUNTIME_CONVERGED = 0x00
    NS_RUNTIME_INITIALIZATION_FAILED = 0x01
    NS_RUNTIME_SCALING_FAILED = 0x02
    NS_RUNTIME_CORRECTOR_FAILED = 0x03
    NS_RUNTIME_STEP_FAILED = 0x04
    NS_RUNTIME_NONFINITE_INPUT = 0x05
    NS_RUNTIME_POINT_MISMATCH = 0x06
    NS_RUNTIME_INVALID_PARAMETER = 0x07
end

"""Typed, allocation-free diagnostic for a nonsymmetric runtime operation."""
struct NonsymmetricRuntimeResult{T}
    status::NonsymmetricRuntimeStatus
    reason::NonsymmetricRuntimeReason
    block_offset::Int
    initialization_reason::NonsymmetricInitializationReason
    scaling_status::NonsymmetricScalingStatus
    scaling_reason::NonsymmetricScalingReason
    fallback_reason::NonsymmetricScalingReason
    conjugate_reason::NonsymmetricConjugateReason
    corrector_reason::NonsymmetricCorrectorReason
    step_status::NonsymmetricStepStatus
    value::T
end

@inline function _runtime_nonsymmetric_default_result(::Type{T}) where {T}
    return NonsymmetricRuntimeResult{T}(
        NS_RUNTIME_READY,
        NS_RUNTIME_CONVERGED,
        0,
        NS_INITIALIZATION_CONVERGED,
        NS_SCALING_DOUBLE_SECANT,
        NS_SCALING_CONVERGED,
        NS_SCALING_NO_FALLBACK,
        NS_CONJUGATE_CONVERGED,
        NS_CORRECTOR_CONVERGED,
        NS_STEP_FULL_LIMIT,
        zero(T),
    )
end

"""Owned primal/dual line-search state for one nonsymmetric block."""
mutable struct NonsymmetricRuntimeLineSearchWorkspace{T,P,D}
    primal_tag::P
    dual_tag::D
    safety::T
    alpha_limit::T
    max_backtracks::Int
    max_bisections::Int
    point::Vector{T}
    direction::Vector{T}
    last_primal::NonsymmetricStepResult{T}
    last_dual::NonsymmetricStepResult{T}
end

"""Second-level checkpoint owned across an outer HSD line search."""
mutable struct NonsymmetricRuntimeScalingCheckpoint{T}
    valid::Bool
    primal::Vector{T}
    dual::Vector{T}
    dual_shadow::Vector{T}
    g::Matrix{T}
    theta::Matrix{T}
    scaling_factor::Matrix{T}
    mu::T
    scaling_valid::Bool
    used_fallback::Bool
    scaling_status::NonsymmetricScalingStatus
    scaling_reason::NonsymmetricScalingReason
    fallback_reason::NonsymmetricScalingReason
    conjugate_reason::NonsymmetricConjugateReason
    conjugate_shadow::Vector{T}
    conjugate_inverse_hessian::Matrix{T}
    accepted_dual::Vector{T}
    accepted_shadow::Vector{T}
    conjugate_hessian::Matrix{T}
    conjugate_hessian_factor::Matrix{T}
    accepted_hessian::Matrix{T}
    accepted_hessian_factor::Matrix{T}
    accepted_inverse_hessian::Matrix{T}
    conjugate_hessian_factor_error::T
    accepted_hessian_factor_error::T
    conjugate_gap::T
    accepted_gap::T
    accepted_valid::Bool
    accepted_hessian_factor_valid::Bool
    accepted_inverse_valid::Bool
    conjugate_valid::Bool
    conjugate_hessian_factor_valid::Bool
    conjugate_inverse_valid::Bool
    seed_mode::NonsymmetricConjugateSeedMode
end

const _RuntimeConjugateWorkspace{T} = NonsymmetricConjugateWorkspace{T}
const _RuntimeScalingWorkspace{T} =
    NonsymmetricScalingWorkspace{T,_RuntimeConjugateWorkspace{T}}
const _RuntimeInitializationWorkspace{T} =
    NonsymmetricInitializationWorkspace{T,_RuntimeScalingWorkspace{T}}

"""One Exp block with an explicit, recorded dual-Hessian fallback policy."""
mutable struct ExpRuntimeBlock{T}
    offset::Int
    dim::Int
    tag::ExpConjugateTag
    policy::DoubleSecantWithDualHessianFallback
    initialization::_RuntimeInitializationWorkspace{T}
    scaling::_RuntimeScalingWorkspace{T}
    corrector::NonsymmetricCorrectorWorkspace{T}
    line_search::NonsymmetricRuntimeLineSearchWorkspace{
        T,ExpPrimalStepTag,ExpDualStepTag,
    }
    primal::Vector{T}
    dual::Vector{T}
    input::Vector{T}
    output::Vector{T}
    direction::Vector{T}
    checkpoint::NonsymmetricRuntimeScalingCheckpoint{T}
    last_scaling_status::NonsymmetricScalingStatus
    last_scaling_reason::NonsymmetricScalingReason
    last_fallback_reason::NonsymmetricScalingReason
    last_conjugate_reason::NonsymmetricConjugateReason
end

"""One setup-built power-cone block with target-typed alpha ownership."""
mutable struct PowerRuntimeBlock{T}
    offset::Int
    dim::Int
    tag::PowerConjugateTag{T}
    policy::DoubleSecantWithDualHessianFallback
    initialization::_RuntimeInitializationWorkspace{T}
    scaling::_RuntimeScalingWorkspace{T}
    corrector::NonsymmetricCorrectorWorkspace{T}
    line_search::NonsymmetricRuntimeLineSearchWorkspace{
        T,PowerPrimalStepTag{T},PowerDualStepTag{T},
    }
    primal::Vector{T}
    dual::Vector{T}
    input::Vector{T}
    output::Vector{T}
    direction::Vector{T}
    checkpoint::NonsymmetricRuntimeScalingCheckpoint{T}
    force_dual_hessian::Bool
    forced_dual_hessian_updates::Int
    last_scaling_status::NonsymmetricScalingStatus
    last_scaling_reason::NonsymmetricScalingReason
    last_fallback_reason::NonsymmetricScalingReason
    last_conjugate_reason::NonsymmetricConjugateReason
end

"""A typed, setup-built runtime for canonical symmetric and 3D blocks.

Every nonsymmetric block owns its tag, explicit double-secant/fallback policy,
initialization, corrector, line-search state, and scratch. Provider changes are
accepted only through the typed fallback status and recorded primary reason.
"""
mutable struct ProductConeRuntime{T,O,S,P,E,W}
    orthant::O
    soc::S
    psd::P
    exp::E
    power::W
    dimension::Int
    valid::Bool
    last_mu::T
    last_nonsymmetric::NonsymmetricRuntimeResult{T}
    checkpoint_mu::T
    checkpoint_nonsymmetric::NonsymmetricRuntimeResult{T}
    checkpoint_valid::Bool
end

"""Setup-owned work vectors for `assemble_schur!`.

The scratch is deliberately separate from the cone runtime: callers may own
more than one Schur assembly context while every hot call remains concrete
and allocation-free.
"""
struct ProductSchurScratch{T}
    input::Vector{T}
    output::Vector{T}
end

ProductSchurScratch(runtime::ProductConeRuntime{T}) where {T} =
    ProductSchurScratch{T}(zeros(T, runtime.dimension), zeros(T, runtime.dimension))
