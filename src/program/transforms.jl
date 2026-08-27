#=====================================================================#
# Typed canonical program transforms (P1).
#
# A transform maps one program coordinate system to the next canonical
# coordinate system.  `forward_*` maps source -> canonical and `backward_*`
# maps canonical -> source.  The latter direction is the authoritative
# reconstruction direction used for results and certificates.
#
# The interface is deliberately independent of a model, solver route, or
# cone implementation.  In particular, a transform owns the primal and
# dual maps together, so a row-sign change cannot silently omit the inverse
# adjoint or a certificate-ray map.
#=====================================================================#

"""
    AbstractProgramTransform{T}

A typed, reversible coordinate transform for a canonical conic program.
The required interface is:

```julia
forward_primal!(transform, dest, src)
backward_primal!(transform, dest, src)
forward_dual!(transform, dest, src)
backward_dual!(transform, dest, src)
backward_primal_ray!(transform, dest, src)
backward_dual_ray!(transform, dest, src)
objective_shift(transform)
verify_pairing_invariant(transform, primal, dual, transformed_primal,
                         transformed_dual; atol, rtol)
verify_stationarity_invariant(transform, A, c, y, tau, A_hat, c_hat,
                              y_hat; atol, rtol)
```

A transform is immutable and owns no iterate or factorization state.
"""
abstract type AbstractProgramTransform{T} end

# Required interface declarations.  Keeping these as explicit methods gives
# incomplete transform implementations an immediate, source-named error
# instead of allowing an unimplemented operation to pass through a stack.
function forward_primal!(transform::AbstractProgramTransform, dest, src)
    throw(MethodError(forward_primal!, (transform, dest, src)))
end
function backward_primal!(transform::AbstractProgramTransform, dest, src)
    throw(MethodError(backward_primal!, (transform, dest, src)))
end
function forward_dual!(transform::AbstractProgramTransform, dest, src)
    throw(MethodError(forward_dual!, (transform, dest, src)))
end
function backward_dual!(transform::AbstractProgramTransform, dest, src)
    throw(MethodError(backward_dual!, (transform, dest, src)))
end
function backward_primal_ray!(transform::AbstractProgramTransform, dest, src)
    throw(MethodError(backward_primal_ray!, (transform, dest, src)))
end
function backward_dual_ray!(transform::AbstractProgramTransform, dest, src)
    throw(MethodError(backward_dual_ray!, (transform, dest, src)))
end
function objective_shift(transform::AbstractProgramTransform)
    throw(MethodError(objective_shift, (transform,)))
end
function verify_pairing_invariant(
    transform::AbstractProgramTransform, primal, dual,
    transformed_primal, transformed_dual; atol=nothing, rtol=nothing, tol=nothing,
)
    throw(MethodError(verify_pairing_invariant,
                      (transform, primal, dual, transformed_primal, transformed_dual)))
end
function verify_stationarity_invariant(
    transform::AbstractProgramTransform, A, c, y, tau, A_hat, c_hat, y_hat;
    atol=nothing, rtol=nothing, tol=nothing,
)
    throw(MethodError(verify_stationarity_invariant,
                      (transform, A, c, y, tau, A_hat, c_hat, y_hat)))
end

# A compact tolerance helper.  `tol` is the caller-facing single tolerance;
# explicit atol/rtol take precedence when supplied.  The default is only for
# arithmetic roundoff and never changes a supplied certificate tolerance.
function _transform_tolerances(::Type{T}; atol=nothing, rtol=nothing, tol=nothing) where {T}
    if tol !== nothing
        atol = atol === nothing ? tol : atol
        rtol = rtol === nothing ? tol : rtol
    end
    default = try
        sqrt(eps(T))
    catch
        zero(T)
    end
    return (atol === nothing ? default : atol,
            rtol === nothing ? default : rtol)
end

"""Check two scalar values under an explicitly caller-controlled tolerance."""
@inline function _transform_isapprox(a, b, ::Type{T}; atol=nothing, rtol=nothing, tol=nothing) where {T}
    a_tol, r_tol = _transform_tolerances(T; atol=atol, rtol=rtol, tol=tol)
    return isapprox(a, b; atol=a_tol, rtol=r_tol)
end

# ---------------------------------------------------------------------------
# Reconstruction stack
# ---------------------------------------------------------------------------

"""
    ReconstructionStack{T}

Ordered transforms from source coordinates to a normalized program.  A
canonical result is reconstructed by applying the inverse (`backward_*`)
operations in reverse push order.  The stack owns only transform metadata;
it never owns numerical iterates.
"""
mutable struct ReconstructionStack{T}
    transforms::Vector{AbstractProgramTransform{T}}

    function ReconstructionStack{T}() where {T}
        return new{T}(AbstractProgramTransform{T}[])
    end

    function ReconstructionStack{T}(
        transforms::AbstractVector{<:AbstractProgramTransform{T}},
    ) where {T}
        return new{T}(AbstractProgramTransform{T}[transforms...])
    end
end

ReconstructionStack(transforms::AbstractVector{<:AbstractProgramTransform{T}}) where {T} =
    ReconstructionStack{T}(transforms)

Base.length(stack::ReconstructionStack) = length(stack.transforms)
Base.isempty(stack::ReconstructionStack) = isempty(stack.transforms)
Base.iterate(stack::ReconstructionStack, state...) = iterate(stack.transforms, state...)

"""Push one transform onto the source-to-canonical chain."""
function push_transform!(stack::ReconstructionStack{T}, transform::AbstractProgramTransform{T}) where {T}
    push!(stack.transforms, transform)
    return stack
end

Base.push!(stack::ReconstructionStack, transform::AbstractProgramTransform) =
    push_transform!(stack, transform)

"""Pop and return the most recently pushed transform."""
function pop_transform!(stack::ReconstructionStack)
    isempty(stack) && throw(ArgumentError("cannot pop an empty ReconstructionStack"))
    return pop!(stack.transforms)
end

Base.pop!(stack::ReconstructionStack) = pop_transform!(stack)

# Forward application is useful while assembling a normalized program.
function _stack_forward!(operation!, stack::ReconstructionStack, dest, src)
    isempty(stack) && (copyto!(dest, src); return dest)
    current = src
    buffer_a = similar(dest)
    buffer_b = similar(dest)
    count = length(stack)
    for (index, transform) in enumerate(stack.transforms)
        target = index == count ? dest : (isodd(index) ? buffer_a : buffer_b)
        operation!(transform, target, current)
        current = target
    end
    return dest
end

forward_primal!(stack::ReconstructionStack, dest, src) =
    _stack_forward!(forward_primal!, stack, dest, src)

forward_dual!(stack::ReconstructionStack, dest, src) =
    _stack_forward!(forward_dual!, stack, dest, src)

# All reconstruction operations walk the full chain backwards.  The stack
# methods intentionally use the same names as the transform interface: code
# consuming either one transform or a complete chain has identical semantics.
function _stack_backward!(operation!, stack::ReconstructionStack, dest, src)
    isempty(stack) && (copyto!(dest, src); return dest)
    current = src
    buffer_a = similar(dest)
    buffer_b = similar(dest)
    count = length(stack)
    for index in count:-1:1
        transform = stack.transforms[index]
        target = index == 1 ? dest : (isodd(index) ? buffer_a : buffer_b)
        operation!(transform, target, current)
        current = target
    end
    return dest
end

function backward_primal!(stack::ReconstructionStack, dest, src)
    return _stack_backward!(backward_primal!, stack, dest, src)
end

function backward_dual!(stack::ReconstructionStack, dest, src)
    return _stack_backward!(backward_dual!, stack, dest, src)
end

function backward_primal_ray!(stack::ReconstructionStack, dest, src)
    return _stack_backward!(backward_primal_ray!, stack, dest, src)
end

function backward_dual_ray!(stack::ReconstructionStack, dest, src)
    return _stack_backward!(backward_dual_ray!, stack, dest, src)
end

# Explicit reconstruction names make the result/certificate boundary
# discoverable while retaining the exact backward_* interface required by
# each transform.
reconstruct_primal!(stack::ReconstructionStack, dest, src) =
    backward_primal!(stack, dest, src)
reconstruct_dual!(stack::ReconstructionStack, dest, src) =
    backward_dual!(stack, dest, src)
reconstruct_primal_ray!(stack::ReconstructionStack, dest, src) =
    backward_primal_ray!(stack, dest, src)
reconstruct_dual_ray!(stack::ReconstructionStack, dest, src) =
    backward_dual_ray!(stack, dest, src)

"""Reconstruct a canonical primal and dual optimum through the full chain."""
function reconstruct_optima!(stack::ReconstructionStack, primal_dest, dual_dest,
                             primal_src, dual_src)
    backward_primal!(stack, primal_dest, primal_src)
    backward_dual!(stack, dual_dest, dual_src)
    return primal_dest, dual_dest
end

"""Reconstruct primal and dual certificate rays through the full chain."""
function reconstruct_rays!(stack::ReconstructionStack, primal_ray_dest, dual_ray_dest,
                           primal_ray_src, dual_ray_src)
    backward_primal_ray!(stack, primal_ray_dest, primal_ray_src)
    backward_dual_ray!(stack, dual_ray_dest, dual_ray_src)
    return primal_ray_dest, dual_ray_dest
end

# ---------------------------------------------------------------------------
# Nonpositive -> Nonnegative
# ---------------------------------------------------------------------------

"""
    NonpositiveToNonnegative{T}

Exact sign transform for one nonpositive vector block.  Forward maps are
`ŝ = -s`, `ŷ = -y`, and the affine row map is `Â = -A`, `b̂ = -b`.
The inverse adjoint is the same map because `T = -I`; this preserves the
Euclidean pairing and the HSD stationarity expression exactly.
"""
struct NonpositiveToNonnegative{T<:AbstractFloat} <: AbstractProgramTransform{T}
end

NonpositiveToNonnegative(::Type{T}) where {T<:AbstractFloat} = NonpositiveToNonnegative{T}()

@inline function _nonpositive_sign_map!(dest, src)
    length(dest) == length(src) || throw(DimensionMismatch(
        "transform destination length $(length(dest)) != source length $(length(src))",
    ))
    @inbounds for index in eachindex(dest, src)
        dest[index] = -src[index]
    end
    return dest
end

forward_primal!(::NonpositiveToNonnegative, dest, src) = _nonpositive_sign_map!(dest, src)
backward_primal!(::NonpositiveToNonnegative, dest, src) = _nonpositive_sign_map!(dest, src)
forward_dual!(::NonpositiveToNonnegative, dest, src) = _nonpositive_sign_map!(dest, src)
backward_dual!(::NonpositiveToNonnegative, dest, src) = _nonpositive_sign_map!(dest, src)
backward_primal_ray!(::NonpositiveToNonnegative, dest, src) = _nonpositive_sign_map!(dest, src)
backward_dual_ray!(::NonpositiveToNonnegative, dest, src) = _nonpositive_sign_map!(dest, src)

objective_shift(::NonpositiveToNonnegative{T}) where {T} = zero(T)

"""Apply the Nonpositive row map `Â=-A`, `b̂=-b` in-place."""
function forward_affine!(::NonpositiveToNonnegative, A_dest, b_dest, A, b)
    size(A_dest) == size(A) || throw(DimensionMismatch("A destination size mismatch"))
    length(b_dest) == length(b) || throw(DimensionMismatch("b destination length mismatch"))
    A_dest .= -A
    @inbounds for index in eachindex(b_dest, b)
        b_dest[index] = -b[index]
    end
    return A_dest, b_dest
end

"""Inverse of [`forward_affine!`](@ref), also `A=-Â`, `b=-b̂`."""
backward_affine!(transform::NonpositiveToNonnegative, A_dest, b_dest, A, b) =
    forward_affine!(transform, A_dest, b_dest, A, b)

function verify_pairing_invariant(
    ::NonpositiveToNonnegative{T}, primal, dual,
    transformed_primal, transformed_dual;
    atol=nothing, rtol=nothing, tol=nothing,
) where {T<:AbstractFloat}
    length(primal) == length(dual) == length(transformed_primal) == length(transformed_dual) ||
        throw(DimensionMismatch("pairing vectors must have equal lengths"))
    lhs = dot(primal, dual)
    rhs = dot(transformed_primal, transformed_dual)
    return _transform_isapprox(lhs, rhs, T; atol=atol, rtol=rtol, tol=tol)
end

function verify_stationarity_invariant(
    ::NonpositiveToNonnegative{T}, A, c, y, tau, A_hat, c_hat, y_hat;
    atol=nothing, rtol=nothing, tol=nothing,
) where {T<:AbstractFloat}
    size(A, 1) == length(y) || throw(DimensionMismatch("A/y dimensions mismatch"))
    size(A_hat) == size(A) || throw(DimensionMismatch("transformed A dimensions mismatch"))
    length(c) == size(A, 2) == length(c_hat) ||
        throw(DimensionMismatch("A/c dimensions mismatch"))
    length(y_hat) == length(y) || throw(DimensionMismatch("transformed y length mismatch"))
    original = transpose(A) * y .+ tau .* c
    transformed = transpose(A_hat) * y_hat .+ tau .* c_hat
    length(original) == length(transformed) || throw(DimensionMismatch("stationarity result mismatch"))
    @inbounds for index in eachindex(original, transformed)
        _transform_isapprox(original[index], transformed[index], T;
                            atol=atol, rtol=rtol, tol=tol) || return false
    end
    return true
end

# Convenience overload for the common unchanged-objective case.
function verify_stationarity_invariant(
    transform::NonpositiveToNonnegative{T}, A, c, y, tau, A_hat, y_hat;
    atol=nothing, rtol=nothing, tol=nothing,
) where {T<:AbstractFloat}
    return verify_stationarity_invariant(
        transform, A, c, y, tau, A_hat, c, y_hat;
        atol=atol, rtol=rtol, tol=tol,
    )
end

#=====================================================================#
#    P1b compatibility interface (provisional).
#
#    The RSOC transform (src/program/transforms_rsoc.jl) was authored
#    against this leaner abstract during the parallel P1a/P1b work.  Both
#    hierarchies share identical method names and semantics; the leaner one
#    carries an explicit `pairing_scale` field convention.  Unify into the
#    single AbstractProgramTransform hierarchy during the P1 lowerer
#    integration pass (tracked for that task, not silently dropped).
#=====================================================================#

"""Provisional leaner transform interface (see integration TODO above)."""
abstract type AbstractConeTransform{T<:AbstractFloat} end

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
