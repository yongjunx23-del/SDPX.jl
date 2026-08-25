# Wave E (setup-only): ReductionPlan types for structural reduction.
#
# These types are the frozen interface for symmetry + chordal reduction. They
# are NOT wired into the solver yet (Wave E is setup-only until Wave C/D
# stabilize). The planner compares original / symmetry / chordal /
# symmetry+chordal and produces reversible forward/backward maps plus a
# certificate map so reduced solutions/certificates round-trip to original
# coordinates.

"""A reversible map from original coordinates to reduced coordinates."""
struct ForwardMap{T}
    permutation::Vector{Int}   # reduced[k] = original[perm[k]]
    block_sizes::Vector{Int}
    basis::Matrix{T}           # columns are the block-diagonalizing basis
end

"""The inverse of a ForwardMap (reconstruct original from reduced)."""
struct BackwardMap{T}
    inverse_permutation::Vector{Int}
    block_sizes::Vector{Int}
    basis::Matrix{T}
end

"""Maps a reduced-coordinate certificate back to original coordinates."""
struct CertificateMap{T}
    forward::ForwardMap{T}
    backward::BackwardMap{T}
end

"""A structural reduction plan: which reduction to apply and its maps."""
struct ReductionPlan{T}
    kind::Symbol                 # :none | :symmetry | :chordal | :symmetry_chordal
    forward::ForwardMap{T}
    backward::BackwardMap{T}
    certificate::CertificateMap{T}
    original_dimension::Int
    reduced_dimension::Int
end

"""Identity reduction plan (no reduction)."""
function identity_reduction_plan(::Type{T}, n::Int) where {T}
    perm = collect(1:n)
    fwd = ForwardMap{T}(perm, [n], Matrix{T}(I, n, n))
    bwd = BackwardMap{T}(perm, [n], Matrix{T}(I, n, n))
    return ReductionPlan{T}(:none, fwd, bwd, CertificateMap{T}(fwd, bwd), n, n)
end

"""Apply the forward map to a vector (reduced = forward(original))."""
function apply_forward(plan::ReductionPlan{T}, x::AbstractVector{T}) where {T}
    return x[plan.forward.permutation]
end

"""Apply the backward map to a reduced vector (original = backward(reduced))."""
function apply_backward(plan::ReductionPlan{T}, xr::AbstractVector{T}) where {T}
    return xr[plan.backward.inverse_permutation]
end
