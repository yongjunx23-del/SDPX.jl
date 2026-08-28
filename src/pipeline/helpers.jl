#=====================================================================
    Automatic solve pipeline

    This file owns cold, structural decisions only: classification,
    equality presolve, scaling/kernel selection, reconstruction, and
    diagnostics. Numeric Newton kernels remain in their specialized files.
=====================================================================#

"""
    _owned_array_copy(T, source) -> Array{T}

Convert an array while preserving independent scalar ownership. Ordinary
`Array{BigFloat}(source)` only copies MPFR object references when `source`
already contains `BigFloat`s; mutating a destination entry can then corrupt
the caller's problem data or warm start.
"""
_owned_array_copy(::Type{T}, source::AbstractArray) where {T} =
    Array{T}(source)

_owned_array_copy(::Type{T}, source::SparseMatrixCSC) where {T} =
    _ingest_owned_sparse(T, source)
_owned_array_copy(::Type{BigFloat}, source::SparseMatrixCSC) =
    _ingest_owned_sparse(BigFloat, source)

function _owned_array_copy(
    ::Type{BigFloat},
    source::AbstractArray,
)
    destination = alloc_zeros(BigFloat, size(source)...)
    if eltype(source) === BigFloat
        copy_owned!(destination, source)
    else
        @inbounds for index in eachindex(destination, source)
            converted = BigFloat(source[index])
            MA.operate_to!(destination[index], copy, converted)
        end
    end
    return destination
end
function _owned_equality_slice(
    ::Type{T},
    matrix::SparseMatrixCSC,
    rows,
    columns,
) where {T}
    return _ingest_owned_sparse(T, matrix[rows, columns])
end

function _owned_equality_slice(
    ::Type{T},
    matrix::AbstractMatrix,
    rows,
    columns,
) where {T}
    return _owned_array_copy(T, view(matrix, rows, columns))
end

"""Return a callback-safe scalar value that cannot mutate solver state."""
@inline _diagnostic_scalar_copy(value) = value
@inline _diagnostic_scalar_copy(value::BigFloat) = MA.mutable_copy(value)

struct EqualityPresolveMap{T}
    original_count::Int
    keep::Vector{Int}
    multiplier_map::Matrix{T}
    planning_evidence::EqualityPlanningEvidence
end


function EqualityPresolveMap{T}(
    original_count::Int,
    keep::Vector{Int},
    multiplier_map::Matrix{T},
) where {T}
    return EqualityPresolveMap{T}(
        original_count,
        keep,
        multiplier_map,
        EqualityPlanningEvidence(original_count; reason=:compatibility),
    )
end

@inline _presolve_enabled(opts::SolverOptions) =
    opts.presolve === true ||
    opts.presolve === :on ||
    opts.presolve === :auto

function EqualityPresolveMap(
    original_count::Int,
    keep::Vector{Int},
)
    multiplier_map = zeros(Float64, length(keep), original_count)
    @inbounds for (row, column) in pairs(keep)
        multiplier_map[row, column] = 1.0
    end
    return EqualityPresolveMap{Float64}(
        original_count,
        keep,
        multiplier_map,
        EqualityPlanningEvidence(original_count; reason=:compatibility),
    )
end
# ---------------------------------------------------------------------------
# Scaled identity construction with independent BigFloat scalar ownership.
# ---------------------------------------------------------------------------

import MutableArithmetics as MA

function _scaled_identity(
    ::Type{T},
    scale::T,
    dimension::Int,
) where {T}
    return Matrix{T}(scale * I, dimension, dimension)
end

function _scaled_identity(
    ::Type{BigFloat},
    scale::BigFloat,
    dimension::Int,
)
    matrix = alloc_zeros(BigFloat, dimension, dimension)
    @inbounds for index in 1:dimension
        MA.operate_to!(matrix[index, index], copy, scale)
    end
    return matrix
end
