using LinearAlgebra
using SparseArrays

function _soc_psd_reference_arrow(vector::AbstractVector{T}) where {T}
    dimension = length(vector)
    if dimension == 3
        return T[
            vector[1] + vector[2] vector[3]
            vector[3] vector[1] - vector[2]
        ]
    end
    matrix = zeros(T, dimension, dimension)
    @inbounds begin
        for index in 1:dimension
            matrix[index, index] = vector[1]
        end
        for index in 2:dimension
            matrix[1, index] = vector[index]
            matrix[index, 1] = vector[index]
        end
    end
    return matrix
end

function _soc_psd_reference_coefficient(
    ::Type{T},
    cone::SDPX.SOCConstraint{T},
    variable::Int,
) where {T}
    dimension = length(cone.b)
    if dimension == 3
        first = cone.A[1, variable]
        second = cone.A[2, variable]
        third = cone.A[3, variable]
        rows = Int[]
        columns = Int[]
        values = T[]
        first_diagonal = first + second
        second_diagonal = first - second
        if !iszero(first_diagonal)
            push!(rows, 1); push!(columns, 1); push!(values, first_diagonal)
        end
        if !iszero(second_diagonal)
            push!(rows, 2); push!(columns, 2); push!(values, second_diagonal)
        end
        if !iszero(third)
            push!(rows, 1); push!(columns, 2); push!(values, third)
            push!(rows, 2); push!(columns, 1); push!(values, third)
        end
        return sparse(rows, columns, values, 2, 2)
    end
    rows = Int[]
    columns = Int[]
    values = T[]
    head = cone.A[1, variable]
    if !iszero(head)
        for index in 1:dimension
            push!(rows, index); push!(columns, index); push!(values, head)
        end
    end
    @inbounds for index in 2:dimension
        value = cone.A[index, variable]
        iszero(value) && continue
        push!(rows, 1); push!(columns, index); push!(values, value)
        push!(rows, index); push!(columns, 1); push!(values, value)
    end
    return sparse(rows, columns, values, dimension, dimension)
end

"""Historical SOC-to-PSD lift, available only to tests and benchmarks."""
function soc_psd_reference_problem(
    problem::SDPX.ConicProblem{T};
    sparse=:auto,
    verbosity=0,
) where {T}
    blocks = SDPX.SparseCoefficientVector{T}[]
    constants = Matrix{T}[]
    sizehint!(blocks, length(problem.cones))
    sizehint!(constants, length(problem.cones))
    for cone in problem.cones
        dimension = length(cone.b)
        side = dimension == 3 ? 2 : dimension
        ids = Int[]
        coefficients = SparseMatrixCSC{T,Int}[]
        for variable in 1:problem.variables
            matrix = _soc_psd_reference_coefficient(T, cone, variable)
            nnz(matrix) == 0 && continue
            push!(ids, variable)
            push!(coefficients, matrix)
        end
        push!(blocks, SDPX.ActiveSparseCoefficientVector(
            T,
            problem.variables,
            ids,
            coefficients,
            side,
        ))
        push!(constants, -_soc_psd_reference_arrow(cone.b))
    end
    return SDPX.ingest(
        problem.c,
        blocks,
        constants,
        transpose(problem.Aeq),
        problem.beq;
        T,
        sparse,
        validate=true,
        symmetrize=false,
        verbosity,
    )
end

function _soc_psd_reference_result(
    problem::SDPX.ConicProblem{T},
    result::SDPX.SDPResult{T},
) where {T}
    slack = Vector{Vector{T}}(undef, length(problem.cones))
    dual = Vector{Vector{T}}(undef, length(problem.cones))
    for block in eachindex(problem.cones)
        dimension = length(problem.cones[block].b)
        primal_matrix = result.X[block]
        dual_matrix = result.Y[block]
        primal = Vector{T}(undef, dimension)
        dual_vector = Vector{T}(undef, dimension)
        if dimension == 3
            two = one(T) + one(T)
            primal[1] = (primal_matrix[1, 1] + primal_matrix[2, 2]) / two
            primal[2] = (primal_matrix[1, 1] - primal_matrix[2, 2]) / two
            primal[3] = (primal_matrix[1, 2] + primal_matrix[2, 1]) / two
            dual_vector[1] = dual_matrix[1, 1] + dual_matrix[2, 2]
            dual_vector[2] = dual_matrix[1, 1] - dual_matrix[2, 2]
            dual_vector[3] = dual_matrix[1, 2] + dual_matrix[2, 1]
        else
            primal[1] = primal_matrix[1, 1]
            dual_vector[1] = tr(dual_matrix)
            @inbounds for index in 2:dimension
                primal[index] = primal_matrix[index, 1]
                dual_vector[index] = T(2) * dual_matrix[1, index]
            end
        end
        slack[block] = primal
        dual[block] = dual_vector
    end
    return SDPX.ConicResult{T}(
        result.status,
        result.message,
        result.x,
        slack,
        dual,
        result.y,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.diagnostics,
        result,
    )
end

function solve_socp_psd_reference(problem::SDPX.ConicProblem{T}; kwargs...) where {T}
    lifted = soc_psd_reference_problem(problem; verbosity=0)
    return _soc_psd_reference_result(problem, SDPX.solve(lifted; kwargs...))
end
