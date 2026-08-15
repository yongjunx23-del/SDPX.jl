"""Exact storage facts for one canonical matrix."""
struct CanonicalMatrixFacts
    rows::Int
    columns::Int
    stored_entries::Int
    nonzero_values::Int
    storage::Symbol
end

"""Facts for an affine map, including columns with at least one nonzero."""
struct CanonicalAffineMapFacts
    matrix::CanonicalMatrixFacts
    active_columns::Int
end

"""
    DenseFormulationFeatures{T}

Cheap facts used to compare the two implemented dense SDP formulations.
They describe the mathematical input only: no provider, precision policy,
fallback, or selected formulation is stored here.
"""
struct DenseFormulationFeatures{T}
    variables::Int
    equalities::Int
    equality_density::Float64
    equality_scale_spread::T
    normal_dimension::Int
    augmented_dimension::Int
    augmented_square_ratio::Float64
end

"""
    EqualityPlanningEvidence

Post-presolve evidence produced by the equality basis analysis already needed
for correctness. It is kept separate from structural model facts: an RRQR
result is not an input fact, and the formulation planner must never run a
second RRQR to manufacture it.
"""
struct EqualityPlanningEvidence
    available::Bool
    basis_verified::Bool
    rank_before::Int
    rank_after::Int
    relative_rrqr_quality::Float64
    reason::Symbol
end

EqualityPlanningEvidence(equalities::Int; reason::Symbol=:not_computed) =
    EqualityPlanningEvidence(
        false,
        false,
        equalities,
        equalities,
        NaN,
        reason,
    )

@inline function _finite_abs(value, label::AbstractString)
    isfinite(value) || throw(ArgumentError("$label contains a non-finite value"))
    return abs(value)
end

function _equality_row_scale_spread(
    matrix::AbstractMatrix{T},
    label::AbstractString,
) where {T}
    minimum_nonzero = nothing
    maximum_scale = zero(T)
    @inbounds for row in axes(matrix, 1)
        scale = zero(T)
        for column in axes(matrix, 2)
            scale = max(scale, _finite_abs(matrix[row, column], label))
        end
        if !iszero(scale)
            minimum_nonzero = minimum_nonzero === nothing ? scale :
                              min(minimum_nonzero, scale)
            maximum_scale = max(maximum_scale, scale)
        end
    end
    minimum_nonzero === nothing && return one(T)
    return maximum_scale / minimum_nonzero
end

function _equality_row_scale_spread(
    matrix::SparseMatrixCSC{T,Int},
    label::AbstractString,
) where {T}
    row_scales = zeros(T, size(matrix, 1))
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    @inbounds for pointer in eachindex(values)
        row = rows[pointer]
        row_scales[row] = max(
            row_scales[row],
            _finite_abs(values[pointer], label),
        )
    end
    minimum_nonzero = nothing
    maximum_scale = zero(T)
    @inbounds for scale in row_scales
        if !iszero(scale)
            minimum_nonzero = minimum_nonzero === nothing ? scale :
                              min(minimum_nonzero, scale)
            maximum_scale = max(maximum_scale, scale)
        end
    end
    minimum_nonzero === nothing && return one(T)
    return maximum_scale / minimum_nonzero
end

function _equality_row_scale_spread(
    matrix::LinearAlgebra.Transpose{T,<:SparseMatrixCSC{T,Int}},
    label::AbstractString,
) where {T}
    parent_matrix = parent(matrix)
    values = nonzeros(parent_matrix)
    minimum_nonzero = nothing
    maximum_scale = zero(T)
    @inbounds for column in axes(parent_matrix, 2)
        scale = zero(T)
        for pointer in nzrange(parent_matrix, column)
            scale = max(scale, _finite_abs(values[pointer], label))
        end
        if !iszero(scale)
            minimum_nonzero = minimum_nonzero === nothing ? scale :
                              min(minimum_nonzero, scale)
            maximum_scale = max(maximum_scale, scale)
        end
    end
    minimum_nonzero === nothing && return one(T)
    return maximum_scale / minimum_nonzero
end

function _dense_formulation_features(
    ::Type{T},
    variables::Int,
    equalities::CanonicalAffineMapFacts,
    equality_matrix::AbstractMatrix{T},
) where {T}
    equality_count = equalities.matrix.rows
    denominator = equalities.matrix.rows * equalities.matrix.columns
    density = denominator == 0 ? 0.0 :
              equalities.matrix.nonzero_values / denominator
    augmented = variables + equality_count
    square_ratio = variables == 0 ? Inf :
                   Float64(augmented)^2 / Float64(variables)^2
    return DenseFormulationFeatures{T}(
        variables,
        equality_count,
        density,
        _equality_row_scale_spread(
            equality_matrix,
            "canonical equality matrix",
        ),
        variables,
        augmented,
        square_ratio,
    )
end

function dense_formulation_features(problem::SDPProblem{T}) where {T}
    equality_matrix = transpose(problem.B)
    equality_facts = CanonicalAffineMapFacts(
        _canonical_matrix_facts(
            equality_matrix,
            "canonical equality matrix",
        )...,
    )
    return _dense_formulation_features(
        T,
        problem.dims.m,
        equality_facts,
        equality_matrix,
    )
end

@inline _canonical_storage(::SparseMatrixCSC) = :sparse_csc
@inline _canonical_storage(::Matrix) = :dense_matrix
@inline _canonical_storage(::Base.ReshapedArray) = :dense_panel_matrix_view
@inline _canonical_storage(::LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC}) =
    :sparse_csc_transpose_view
@inline _canonical_storage(::LinearAlgebra.Transpose{<:Any,<:Matrix}) =
    :dense_transpose_view
@inline _canonical_storage(::AbstractMatrix) = :other

@inline function _checked_feature_add(left::Int, right::Int, label::AbstractString)
    try
        return Base.checked_add(left, right)
    catch exception
        exception isa OverflowError || rethrow()
        throw(OverflowError("$label exceeds Int capacity"))
    end
end

function _canonical_matrix_facts(
    matrix::AbstractMatrix{T},
    label::AbstractString,
) where {T}
    sparse_transpose = matrix isa LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC}
    sparse_parent = sparse_transpose ? parent(matrix) : nothing
    stored_entries = matrix isa SparseMatrixCSC || sparse_transpose ?
                     nnz(sparse_parent === nothing ? matrix : sparse_parent) :
                     length(matrix)
    nonzero_values = 0
    active_columns = 0
    if matrix isa SparseMatrixCSC
        sparse_matrix = matrix
        stored_values = nonzeros(sparse_matrix)
        @inbounds for column in axes(sparse_matrix, 2)
            column_active = false
            for pointer in nzrange(sparse_matrix, column)
                value = stored_values[pointer]
                isfinite(value) || throw(ArgumentError(
                    "$label contains a non-finite value",
                ))
                if !iszero(value)
                    nonzero_values = _checked_feature_add(
                        nonzero_values,
                        1,
                        "$label nonzero count",
                    )
                    column_active = true
                end
            end
            column_active && (active_columns += 1)
        end
    elseif sparse_transpose
        # A transpose view of CSC storage is scanned through the parent once.
        # Marking parent rows gives active logical columns without densifying.
        parent_matrix = sparse_parent
        active = Set{Int}()
        stored_values = nonzeros(parent_matrix)
        rows = rowvals(parent_matrix)
        @inbounds for pointer in eachindex(stored_values)
            value = stored_values[pointer]
            isfinite(value) || throw(ArgumentError(
                "$label contains a non-finite value",
            ))
            if !iszero(value)
                nonzero_values = _checked_feature_add(
                    nonzero_values,
                    1,
                    "$label nonzero count",
                )
                push!(active, rows[pointer])
            end
        end
        active_columns = length(active)
    else
        @inbounds for column in axes(matrix, 2)
            column_active = false
            for row in axes(matrix, 1)
                value = matrix[row, column]
                isfinite(value) || throw(ArgumentError(
                    "$label contains a non-finite value",
                ))
                if !iszero(value)
                    nonzero_values = _checked_feature_add(
                        nonzero_values,
                        1,
                        "$label nonzero count",
                    )
                    column_active = true
                end
            end
            column_active && (active_columns += 1)
        end
    end
    facts = CanonicalMatrixFacts(
        size(matrix, 1),
        size(matrix, 2),
        stored_entries,
        nonzero_values,
        _canonical_storage(matrix),
    )
    return facts, active_columns
end
