#=====================================================================#
#    Typed canonical-coordinate transformations (P1).
#
#    A transform owns both sides of a coordinate change.  In particular,
#    it must not be possible to implement a primal map without also
#    specifying the dual inverse-adjoint and the reconstruction maps.
#
#    Coordinate convention:
#
#       canonical primal = T * original primal
#       canonical dual   = T⁻ᵀ * original dual
#
#    Therefore the pairing is checked as
#
#       <original primal, original dual>
#         == pairing_scale * <canonical primal, canonical dual>.
#
#    This file intentionally contains only the shared interface.  Concrete
#    cone transformations live in sibling files and are included by SDPX.jl.
#=====================================================================#

"""Abstract interface for an owned canonical-coordinate transform."""
abstract type AbstractConeTransform{T<:AbstractFloat} end

# The bang methods use transform-first argument order.  Keeping the generic
# methods here makes an incompletely implemented transformation fail at the
# boundary rather than silently falling back to an unrelated map.
function forward_primal!(transform::AbstractConeTransform, destination, source)
    throw(MethodError(forward_primal!, (transform, destination, source)))
end

function backward_primal!(transform::AbstractConeTransform, destination, source)
    throw(MethodError(backward_primal!, (transform, destination, source)))
end

function forward_dual!(transform::AbstractConeTransform, destination, source)
    throw(MethodError(forward_dual!, (transform, destination, source)))
end

function backward_dual!(transform::AbstractConeTransform, destination, source)
    throw(MethodError(backward_dual!, (transform, destination, source)))
end

function backward_primal_ray!(transform::AbstractConeTransform, destination, source)
    throw(MethodError(backward_primal_ray!, (transform, destination, source)))
end

function backward_dual_ray!(transform::AbstractConeTransform, destination, source)
    throw(MethodError(backward_dual_ray!, (transform, destination, source)))
end

function objective_shift(transform::AbstractConeTransform)
    throw(MethodError(objective_shift, (transform,)))
end

function verify_pairing_invariant(transform::AbstractConeTransform, primal, dual; kwargs...)
    throw(MethodError(verify_pairing_invariant, (transform, primal, dual)))
end

function verify_stationarity_invariant(transform::AbstractConeTransform, A, dual, objective; kwargs...)
    throw(MethodError(
        verify_stationarity_invariant,
        (transform, A, dual, objective),
    ))
end
