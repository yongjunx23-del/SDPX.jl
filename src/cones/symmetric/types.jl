# src/cones/symmetric/types.jl
#
# Cone descriptor types and packed-lower PSD storage helpers.

# ---------------------------------------------------------------------------
# Cone descriptors
# ---------------------------------------------------------------------------

"""Nonnegative orthant cone `R_+^dim`. Self-dual; identity `(1, …, 1)`."""
struct NonnegativeCone
    dim::Int
end

"""Lorentz second-order cone `{(t, u) : t ≥ ‖u‖}`, `dim` scalars total."""
struct SOCone
    dim::Int
end

"""
    PSDEigenScratch{T}

Preallocated workspace for the PSD kernels. Holds the full (symmetrised)
working matrices, the eigenvector matrix, the eigenvalue buffer and the
matrix-multiplication scratch. Because it lives inside the
[`PSDTriangleCone`](@ref) descriptor, the hot PSD operations (`membership`,
`jordan_product!`, `inverse!`, `sqrt!`, `nt_scaling!`, `scaling_apply!`) run
with **zero** Julia heap allocation on warm calls.
"""
struct PSDEigenScratch{T}
    n::Int
    A::Matrix{T}          # full n×n working matrix (symmetrised)
    B::Matrix{T}          # scratch
    C::Matrix{T}          # scratch
    V::Matrix{T}          # eigenvector matrix (columns)
    w::Vector{T}          # eigenvalues
end

function PSDEigenScratch{T}(n::Int) where {T}
    PSDEigenScratch{T}(
        n,
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n),
    )
end

"""
    PSDTriangleCone{T}(dim)

Positive-semidefinite cone over `dim×dim` symmetric matrices, stored internally
as the packed lower triangle of length `dim*(dim+1)/2` (column-major, matching
the `:packed_lower` IR storage: for `n = 2` the vector is
`(A[1,1], A[2,1], A[2,2])`). Carries the preallocated [`PSDEigenScratch`](@ref).
"""
struct PSDTriangleCone{T}
    dim::Int
    len::Int
    scratch::PSDEigenScratch{T}
end

function PSDTriangleCone{T}(dim::Int) where {T}
    dim >= 1 || throw(ArgumentError("PSD dimension must be >= 1"))
    PSDTriangleCone{T}(dim, div(dim * (dim + 1), 2), PSDEigenScratch{T}(dim))
end

# convenience constructor without an explicit element type, inferred from a
# sample element so callers do not have to name `T`
PSDTriangleCone(dim::Int, ::Type{T}) where {T} = PSDTriangleCone{T}(dim)

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------
dim(cone::NonnegativeCone) = cone.dim
dim(cone::SOCone) = cone.dim
dim(cone::PSDTriangleCone) = cone.dim

"""Packed stored length of a cone block."""
stored_length(cone::NonnegativeCone) = cone.dim
stored_length(cone::SOCone) = cone.dim
stored_length(cone::PSDTriangleCone) = cone.len

eltype(::NonnegativeCone) = Nothing
eltype(::SOCone) = Nothing
eltype(::PSDTriangleCone{T}) where {T} = T

Base.length(cone::NonnegativeCone) = stored_length(cone)
Base.length(cone::SOCone) = stored_length(cone)
Base.length(cone::PSDTriangleCone) = stored_length(cone)

# ---------------------------------------------------------------------------
# Packed lower-triangle <-> full symmetric matrix
# ---------------------------------------------------------------------------

"""Length of the packed lower triangle of an `n×n` symmetric matrix."""
@inline packed_len(n::Int) = div(n * (n + 1), 2)

"""
Copy a packed lower triangle into the full symmetric matrix `A`
(`A[j,i] = A[i,j] = packed[k]`), column-major packed order:
for `j = 1:n`, `i = j:n`. Returns `A`.
"""
@inline function _unpack!(A::AbstractMatrix{T}, packed::AbstractVector, n::Int) where {T}
    k = 1
    @inbounds for j in 1:n
        for i in j:n
            v = packed[k]
            A[i, j] = v
            A[j, i] = v
            k += 1
        end
    end
    return A
end

"""Pack the full symmetric matrix `A` (lower triangle) into packed vector `v`."""
@inline function _pack!(v::AbstractVector{T}, A::AbstractMatrix, n::Int) where {T}
    k = 1
    @inbounds for j in 1:n
        for i in j:n
            v[k] = A[i, j]
            k += 1
        end
    end
    return v
end

"""Fill the `n×n` matrix `V` with the identity."""
@inline function _identity!(V::AbstractMatrix{T}, n::Int) where {T}
    z = zero(T)
    o = one(T)
    @inbounds for j in 1:n
        for i in 1:n
            V[i, j] = i == j ? o : z
        end
    end
    return V
end

"""Normalise every column of `V` to unit Euclidean norm (defensive)."""
@inline function _orthonormalize!(V::AbstractMatrix{T}, n::Int) where {T}
    @inbounds for k in 1:n
        nrm = zero(T)
        for i in 1:n
            nrm += V[i, k] * V[i, k]
        end
        nrm = sqrt(nrm)
        if !iszero(nrm)
            for i in 1:n
                V[i, k] /= nrm
            end
        end
    end
    return V
end
