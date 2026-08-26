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

"""A typed, setup-built runtime for canonical nonnegative/SOC/PSD blocks.

`exp` and `power` are retained as concretely typed empty vectors in the
struct layout to make unsupported families explicit.  The constructor rejects
such blocks; they are not silently routed through an asymmetric fallback.
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
