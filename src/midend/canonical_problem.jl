"""
    CanonicalConicProblem

Lossless semantic view of a compact `ConicProblem`.  This stage deliberately
does not formulate or lift a problem: an SOC remains a Lorentz cone, while the
linear and PSD families are represented by empty, typed collections until
their frontends are migrated.

All array fields are borrowed from the already-ingested `ConicProblem` and
must be treated as read-only.  In particular, `canonicalize` does not densify
sparse matrices, change their arithmetic type, or materialize a PSD arrow.
"""

abstract type AbstractCanonicalCone{T} end

abstract type AbstractCanonicalLinearCone{T} <: AbstractCanonicalCone{T} end
abstract type AbstractCanonicalLorentzCone{T} <: AbstractCanonicalCone{T} end
abstract type AbstractCanonicalPSDCone{T} <: AbstractCanonicalCone{T} end

"""Semantic boundary for a nonnegative/linear cone block."""
struct CanonicalLinearCone{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractCanonicalLinearCone{T}
    A::M
    b::V
end

"""Semantic boundary for a Lorentz (second-order) cone block."""
struct CanonicalLorentzCone{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractCanonicalLorentzCone{T}
    A::M
    b::V
end

"""Semantic boundary for a positive-semidefinite cone block."""
struct CanonicalPSDCone{
    T,
    C<:AbstractVector{<:AbstractMatrix{T}},
    M<:AbstractMatrix{T},
} <:
       AbstractCanonicalPSDCone{T}
    coefficients::C
    constant::M
end

"""Row-oriented affine equalities `A * x = b`."""
abstract type AbstractCanonicalEqualities{T} end

struct CanonicalEqualities{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractCanonicalEqualities{T}
    A::M
    b::V
end

"""
    CanonicalIdentityReconstructionMap

Minimal identity map for the stage-1 view.  Variables retain their original
order, and each Lorentz block retains its original local coordinate order.
The ranges are local to each block (the first coordinate is the Lorentz head).
"""
abstract type CanonicalReconstructionMap end

struct CanonicalIdentityReconstructionMap <: CanonicalReconstructionMap
    original_variable_indices::UnitRange{Int}
    lorentz_coordinate_order::Vector{UnitRange{Int}}
end

"""
    CanonicalConicProblem{T}

The semantic cone families are intentionally separate.  `ConicProblem`
contains only Lorentz blocks, so canonicalization leaves `linear_cones` and
`psd_cones` empty rather than representing either family as an implicit PSD
lift.
"""
struct CanonicalConicProblem{T}
    objective::Vector{T}
    equalities::AbstractCanonicalEqualities{T}
    linear_cones::Vector{AbstractCanonicalLinearCone{T}}
    lorentz_cones::Vector{AbstractCanonicalLorentzCone{T}}
    psd_cones::Vector{AbstractCanonicalPSDCone{T}}
    metadata::NamedTuple
    reconstruction::CanonicalReconstructionMap
end

Base.eltype(::CanonicalConicProblem{T}) where {T} = T

"""
    reconstruct_identity(map, x, lorentz_coordinates)

Validate and return original variables and Lorentz coordinates under the
stage-1 identity map.  The returned arrays are the caller's arrays (no scalar
conversion or copying), which keeps BigFloat ownership intact.
"""
function reconstruct_identity(
    map::CanonicalIdentityReconstructionMap,
    x::AbstractVector,
    lorentz_coordinates::AbstractVector,
)
    length(x) == length(map.original_variable_indices) || throw(DimensionMismatch(
        "identity reconstruction variable length does not match the map",
    ))
    length(lorentz_coordinates) == length(map.lorentz_coordinate_order) ||
        throw(DimensionMismatch(
            "identity reconstruction Lorentz block count does not match the map",
        ))
    @inbounds for index in eachindex(map.lorentz_coordinate_order)
        length(lorentz_coordinates[index]) ==
            length(map.lorentz_coordinate_order[index]) ||
            throw(DimensionMismatch(
                "identity reconstruction Lorentz block length does not match the map",
            ))
    end
    return (x=x, lorentz_coordinates=lorentz_coordinates)
end

"""Build the lossless semantic view of a compact conic problem."""
function canonicalize(problem::ConicProblem{T}) where {T}
    lorentz_cones = Vector{AbstractCanonicalLorentzCone{T}}(
        undef,
        length(problem.cones),
    )
    @inbounds for index in eachindex(problem.cones)
        cone = problem.cones[index]
        # The constructor only stores references to the owned input arrays.
        lorentz_cones[index] = CanonicalLorentzCone{
            T,
            typeof(cone.A),
            typeof(cone.b),
        }(cone.A, cone.b)
    end

    metadata = (
        source=:ConicProblem,
        formulation=:canonical_compact,
        objective_sense=:min,
        variables=problem.variables,
        equality_rows=size(problem.Aeq, 1),
        cone_order=fill(:lorentz, length(problem.cones)),
        arithmetic=T,
    )
    reconstruction = CanonicalIdentityReconstructionMap(
        1:problem.variables,
        [1:length(cone.b) for cone in problem.cones],
    )
    return CanonicalConicProblem{T}(
        problem.c,
        CanonicalEqualities{
            T,
            typeof(problem.Aeq),
            typeof(problem.beq),
        }(problem.Aeq, problem.beq),
        AbstractCanonicalLinearCone{T}[],
        lorentz_cones,
        AbstractCanonicalPSDCone{T}[],
        metadata,
        reconstruction,
    )
end
