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

"""Facts for one linear or Lorentz affine-cone block."""
struct CanonicalAffineConeFacts
    dimension::Int
    map::CanonicalAffineMapFacts
end

"""Facts for one PSD affine-cone block."""
struct CanonicalPSDConeFacts
    dimension::Int
    coefficient_matrices::Int
    coefficient_stored_entries::Int
    coefficient_nonzero_values::Int
    active_variables::Int
    dense_coefficients::Int
    sparse_csc_coefficients::Int
    other_coefficients::Int
    offset::CanonicalMatrixFacts
end

"""
    ProblemFeatures{T}

Pure facts extracted from a `CanonicalConicProblem{T}`.  No field chooses a
formulation, algorithm, backend, precision policy, or size class.
"""
struct ProblemFeatures{T}
    variables::Int
    objective_nonzero_values::Int
    equalities::CanonicalAffineMapFacts
    linear_cones::Vector{CanonicalAffineConeFacts}
    lorentz_cones::Vector{CanonicalAffineConeFacts}
    psd_cones::Vector{CanonicalPSDConeFacts}
end

Base.eltype(::ProblemFeatures{T}) where {T} = T

@inline _canonical_storage(::SparseMatrixCSC) = :sparse_csc
@inline _canonical_storage(::Matrix) = :dense_matrix
@inline _canonical_storage(::Base.ReshapedArray) = :dense_panel_matrix_view
@inline _canonical_storage(::LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC}) =
    :sparse_csc_transpose_view
@inline _canonical_storage(::LinearAlgebra.Transpose{<:Any,<:Matrix}) =
    :dense_transpose_view
@inline _canonical_storage(view::CanonicalNegatedMatrixView{<:Any,<:SparseMatrixCSC}) =
    :negated_sparse_csc_view
@inline _canonical_storage(view::CanonicalNegatedMatrixView{<:Any,<:Matrix}) =
    :negated_dense_matrix_view
@inline _canonical_storage(::AbstractMatrix) = :other

@inline function _checked_feature_add(left::Int, right::Int, label::AbstractString)
    try
        return Base.checked_add(left, right)
    catch exception
        exception isa OverflowError || rethrow()
        throw(OverflowError("$label exceeds Int capacity"))
    end
end

function _canonical_matrix_facts(matrix::AbstractMatrix{T}, label::AbstractString) where {T}
    sparse_transpose = matrix isa LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC}
    negated_sparse = matrix isa CanonicalNegatedMatrixView{<:Any,<:SparseMatrixCSC}
    sparse_parent = sparse_transpose ? parent(matrix) :
                    negated_sparse ? parent(matrix) : nothing
    stored_entries = matrix isa SparseMatrixCSC || sparse_transpose || negated_sparse ?
                     nnz(sparse_parent === nothing ? matrix : sparse_parent) : length(matrix)
    nonzero_values = 0
    active_columns = 0
    if matrix isa SparseMatrixCSC || negated_sparse
        sparse_matrix = matrix isa SparseMatrixCSC ? matrix : sparse_parent
        stored_values = nonzeros(sparse_matrix)
        @inbounds for column in axes(sparse_matrix, 2)
            column_active = false
            for pointer in nzrange(sparse_matrix, column)
                value = negated_sparse ? -stored_values[pointer] : stored_values[pointer]
                isfinite(value) || throw(ArgumentError("$label contains a non-finite value"))
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
            isfinite(value) || throw(ArgumentError("$label contains a non-finite value"))
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
                isfinite(value) || throw(ArgumentError("$label contains a non-finite value"))
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

function _check_finite_vector(vector::AbstractVector, label::AbstractString)
    nonzero_values = 0
    for value in vector
        isfinite(value) || throw(ArgumentError("$label contains a non-finite value"))
        iszero(value) || (nonzero_values = _checked_feature_add(
            nonzero_values,
            1,
            "$label nonzero count",
        ))
    end
    return nonzero_values
end

function _canonical_affine_cone_facts(
    cone::Union{CanonicalLinearCone{T},CanonicalLorentzCone{T}},
    variables::Int,
    label::AbstractString,
) where {T}
    size(cone.A, 2) == variables || throw(DimensionMismatch(
        "$label has $(size(cone.A, 2)) columns; expected $variables",
    ))
    size(cone.A, 1) == length(cone.offset) || throw(DimensionMismatch(
        "$label offset length $(length(cone.offset)) does not match $(size(cone.A, 1)) rows",
    ))
    size(cone.A, 1) > 0 || throw(DimensionMismatch("$label must have positive dimension"))
    _check_finite_vector(cone.offset, "$label offset")
    matrix, active_columns = _canonical_matrix_facts(cone.A, "$label matrix")
    return CanonicalAffineConeFacts(
        size(cone.A, 1),
        CanonicalAffineMapFacts(matrix, active_columns),
    )
end

function _canonical_psd_cone_facts(
    cone::CanonicalPSDCone{T},
    variables::Int,
    label::AbstractString,
) where {T}
    size(cone.offset, 1) == size(cone.offset, 2) ||
        throw(DimensionMismatch("$label offset must be square"))
    dimension = size(cone.offset, 1)
    dimension > 0 || throw(DimensionMismatch("$label must have positive dimension"))
    length(cone.coefficients) == variables || throw(DimensionMismatch(
        "$label has $(length(cone.coefficients)) coefficient matrices; expected $variables",
    ))
    offset, _ = _canonical_matrix_facts(cone.offset, "$label offset")
    stored_entries = 0
    nonzero_values = 0
    active_variables = 0
    dense_coefficients = 0
    sparse_csc_coefficients = 0
    other_coefficients = 0
    coefficients = cone.coefficients
    if coefficients isa ActiveSparseCoefficientVector{T}
        coefficients.variables == variables || throw(DimensionMismatch(
            "$label active coefficient logical variable count $(coefficients.variables) does not match $variables",
        ))
        length(coefficients.active_variables) == length(coefficients.coefficients) ||
            throw(DimensionMismatch(
                "$label active variable and coefficient counts must match",
            ))
        size(coefficients.empty) == (dimension, dimension) ||
            throw(DimensionMismatch(
                "$label empty coefficient has size $(size(coefficients.empty)); expected ($dimension, $dimension)",
            ))
        empty_facts, _ = _canonical_matrix_facts(coefficients.empty, "$label empty coefficient")
        empty_facts.nonzero_values == 0 || throw(ArgumentError(
            "$label empty coefficient contains nonzero values",
        ))
        previous = 0
        for position in eachindex(coefficients.active_variables)
            variable = coefficients.active_variables[position]
            1 <= variable <= variables || throw(BoundsError(1:variables, variable))
            variable > previous || throw(ArgumentError(
                "$label active variables must be sorted and unique",
            ))
            previous = variable
            coefficient = coefficients.coefficients[position]
            size(coefficient) == (dimension, dimension) || throw(DimensionMismatch(
                "$label coefficient $variable has size $(size(coefficient)); expected ($dimension, $dimension)",
            ))
            facts, _ = _canonical_matrix_facts(
                coefficient,
                "$label coefficient $variable",
            )
            stored_entries = _checked_feature_add(
                stored_entries,
                facts.stored_entries,
                "$label coefficient stored count",
            )
            nonzero_values = _checked_feature_add(
                nonzero_values,
                facts.nonzero_values,
                "$label coefficient nonzero count",
            )
            facts.nonzero_values > 0 && (active_variables += 1)
        end
        sparse_csc_coefficients = variables
    elseif coefficients isa CompactScalarCoefficientVector{T}
        coefficients.variables == variables || throw(DimensionMismatch(
            "$label compact coefficient logical variable count $(coefficients.variables) does not match $variables",
        ))
        1 <= coefficients.active_variable <= variables ||
            throw(BoundsError(1:variables, coefficients.active_variable))
        for (index, coefficient) in ((:active, coefficients.coefficient), (:empty, coefficients.empty))
            size(coefficient) == (dimension, dimension) || throw(DimensionMismatch(
                "$label $index coefficient has size $(size(coefficient)); expected ($dimension, $dimension)",
            ))
            facts, _ = _canonical_matrix_facts(coefficient, "$label $index coefficient")
            if index === :empty
                facts.nonzero_values == 0 || throw(ArgumentError(
                    "$label empty coefficient contains nonzero values",
                ))
            else
                stored_entries = facts.stored_entries
                nonzero_values = facts.nonzero_values
                facts.nonzero_values > 0 && (active_variables = 1)
            end
        end
        sparse_csc_coefficients = variables
    else
        for (index, coefficient) in pairs(coefficients)
            size(coefficient) == (dimension, dimension) || throw(DimensionMismatch(
                "$label coefficient $index has size $(size(coefficient)); expected ($dimension, $dimension)",
            ))
            facts, active_columns = _canonical_matrix_facts(
                coefficient,
                "$label coefficient $index",
            )
            stored_entries = _checked_feature_add(
                stored_entries,
                facts.stored_entries,
                "$label coefficient stored count",
            )
            nonzero_values = _checked_feature_add(
                nonzero_values,
                facts.nonzero_values,
                "$label coefficient nonzero count",
            )
            active_columns > 0 && (active_variables += 1)
            coefficient_storage = facts.storage
            if coefficients isa CanonicalDensePanelCoefficients
                coefficient_storage = :dense_panel_matrix_view
            end
            if coefficient_storage === :dense_matrix ||
               coefficient_storage === :dense_panel_matrix_view
                dense_coefficients += 1
            elseif coefficient_storage === :sparse_csc
                sparse_csc_coefficients += 1
            else
                other_coefficients += 1
            end
        end
    end
    return CanonicalPSDConeFacts(
        dimension,
        variables,
        stored_entries,
        nonzero_values,
        active_variables,
        dense_coefficients,
        sparse_csc_coefficients,
        other_coefficients,
        offset,
    )
end

"""
    extract_problem_features(problem::CanonicalConicProblem)

Validate and summarize canonical dimensions and storage. Sparse CSC data are
visited in `O(ncols + nnz)`; no model array is copied, converted, or mutated.
"""
function extract_problem_features(problem::CanonicalConicProblem{T}) where {T}
    variables = length(problem.objective)
    objective_nonzero_values = _check_finite_vector(
        problem.objective,
        "canonical objective",
    )
    equalities = problem.equalities
    equalities isa CanonicalEqualities{T} || throw(ArgumentError(
        "unsupported canonical equality representation $(typeof(equalities))",
    ))
    size(equalities.A, 2) == variables || throw(DimensionMismatch(
        "canonical equality matrix has $(size(equalities.A, 2)) columns; expected $variables",
    ))
    size(equalities.A, 1) == length(equalities.b) || throw(DimensionMismatch(
        "canonical equality rhs length $(length(equalities.b)) does not match $(size(equalities.A, 1)) rows",
    ))
    _check_finite_vector(equalities.b, "canonical equality rhs")
    equality_matrix, equality_active_columns = _canonical_matrix_facts(
        equalities.A,
        "canonical equality matrix",
    )

    linear_cones = CanonicalAffineConeFacts[]
    for (index, cone) in pairs(problem.linear_cones)
        cone isa CanonicalLinearCone{T} || throw(ArgumentError(
            "unsupported canonical linear representation $(typeof(cone))",
        ))
        push!(linear_cones, _canonical_affine_cone_facts(
            cone,
            variables,
            "canonical linear block $index",
        ))
    end
    lorentz_cones = CanonicalAffineConeFacts[]
    for (index, cone) in pairs(problem.lorentz_cones)
        cone isa CanonicalLorentzCone{T} || throw(ArgumentError(
            "unsupported canonical Lorentz representation $(typeof(cone))",
        ))
        push!(lorentz_cones, _canonical_affine_cone_facts(
            cone,
            variables,
            "canonical Lorentz block $index",
        ))
    end
    psd_cones = CanonicalPSDConeFacts[]
    for (index, cone) in pairs(problem.psd_cones)
        cone isa CanonicalPSDCone{T} || throw(ArgumentError(
            "unsupported canonical PSD representation $(typeof(cone))",
        ))
        push!(psd_cones, _canonical_psd_cone_facts(
            cone,
            variables,
            "canonical PSD block $index",
        ))
    end
    return ProblemFeatures{T}(
        variables,
        objective_nonzero_values,
        CanonicalAffineMapFacts(equality_matrix, equality_active_columns),
        linear_cones,
        lorentz_cones,
        psd_cones,
    )
end
