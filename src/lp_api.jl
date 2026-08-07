"""
    linear_program(c, G, h; Aeq=nothing, beq=nothing, kwargs...)

Build an [`SDPProblem`](@ref) for the linear program

`min c'x` subject to `Gx >= h` and `Aeq*x = beq`.

The constructor stores each inequality row by its active variables and keeps a
sparse equality matrix when appropriate. It therefore avoids the historical
`number of inequalities × number of variables` grid of empty `1×1` matrices.
The returned problem uses the same dedicated LP solver selected for scalar-cone
models built through `ingest`, JuMP, or Convex.jl.

Keyword arguments are forwarded to [`ingest`](@ref). The legacy scalar-block
`ingest` interface remains supported.
"""
function linear_program(
    c::AbstractVector,
    G::AbstractMatrix,
    h::AbstractVector;
    Aeq=nothing,
    beq=nothing,
    T::Union{Nothing,Type}=nothing,
    sparse::Union{Bool,Symbol}=:auto,
    validate::Bool=true,
    verbosity::Int=1,
)
    inequalities, variables = size(G)
    length(c) == variables || throw(DimensionMismatch(
        "length(c) must equal the number of columns in G ($variables)",
    ))
    length(h) == inequalities || throw(DimensionMismatch(
        "length(h) must equal the number of rows in G ($inequalities)",
    ))
    inequalities > 0 || throw(ArgumentError(
        "G must contain at least one inequality row",
    ))

    equality_rows = Aeq === nothing ? 0 : size(Aeq, 1)
    Aeq !== nothing && size(Aeq, 2) != variables && throw(DimensionMismatch(
        "Aeq must have $variables columns",
    ))
    beq === nothing && equality_rows > 0 && throw(ArgumentError(
        "beq is required when Aeq is provided",
    ))
    beq !== nothing && length(beq) != equality_rows && throw(DimensionMismatch(
        "length(beq) must equal the number of rows in Aeq ($equality_rows)",
    ))

    equality_rhs_source = beq === nothing ? Float64[] : beq
    equality_matrix_source = Aeq === nothing ?
        spzeros(Float64, 0, variables) : Aeq
    ET = if T === nothing
        promoted = promote_type(
            eltype(c),
            eltype(G),
            eltype(h),
            eltype(equality_matrix_source),
            eltype(equality_rhs_source),
        )
        _require_supported_arithmetic_type(
            promoted <: AbstractFloat ? promoted : float(promoted),
        )
    else
        _require_supported_arithmetic_type(T)
    end

    function active_block(ids::Vector{Int}, values)
        if length(ids) == 1
            return CompactScalarCoefficientVector(
                ET,
                variables,
                only(ids),
                only(values),
            )
        end
        coefficients = Vector{SparseMatrixCSC{ET,Int}}(undef, length(ids))
        @inbounds for position in eachindex(ids)
            coefficients[position] =
                SparseArrays.sparse([1], [1], ET[values[position]], 1, 1)
        end
        return ActiveSparseCoefficientVector(
            ET,
            variables,
            ids,
            coefficients,
            1,
        )
    end

    blocks = SparseCoefficientVector{ET}[]
    sizehint!(blocks, inequalities)
    if G isa SparseMatrixCSC
        # CSC storage is column-oriented. Transposing once makes each original
        # inequality a contiguous column, so construction is O(nnz(G)).
        row_storage = SparseArrays.sparse(
            transpose(SparseArrays.sparse(ET.(G))),
        )
        rows = rowvals(row_storage)
        values = nonzeros(row_storage)
        @inbounds for inequality in 1:inequalities
            ids = Int[]
            coefficients = ET[]
            for position in nzrange(row_storage, inequality)
                value = values[position]
                iszero(value) && continue
                push!(ids, rows[position])
                push!(coefficients, _ingest_owned_scalar(ET, value))
            end
            push!(blocks, active_block(ids, coefficients))
        end
    else
        @inbounds for inequality in 1:inequalities
            ids = Int[]
            coefficients = ET[]
            for variable in 1:variables
                value = G[inequality, variable]
                iszero(value) && continue
                push!(ids, variable)
                push!(coefficients, _ingest_owned_scalar(ET, value))
            end
            push!(blocks, active_block(ids, coefficients))
        end
    end

    constants = Matrix{ET}[
        reshape(ET[_ingest_owned_scalar(ET, h[row])], 1, 1)
        for row in eachindex(h)
    ]
    equality_matrix = if Aeq === nothing
        spzeros(ET, variables, 0)
    elseif Aeq isa SparseMatrixCSC
        SparseArrays.sparse(
            transpose(SparseArrays.sparse(ET.(Aeq))),
        )
    else
        permutedims(ET.(Aeq))
    end
    equality_rhs = beq === nothing ? ET[] : ET.(beq)
    return ingest(
        c,
        blocks,
        constants,
        equality_matrix,
        equality_rhs;
        T=ET,
        sparse=sparse,
        validate=validate,
        symmetrize=false,
        verbosity=verbosity,
    )
end

"""
    solve_lp(c, G, h; Aeq=nothing, beq=nothing, solver_kwargs...)

Build and solve a linear program with the native active-row frontend. Use
[`linear_program`](@ref) when the model will be solved repeatedly.
"""
function solve_lp(
    c::AbstractVector,
    G::AbstractMatrix,
    h::AbstractVector;
    Aeq=nothing,
    beq=nothing,
    T::Union{Nothing,Type}=nothing,
    sparse::Union{Bool,Symbol}=:auto,
    validate::Bool=true,
    kwargs...,
)
    problem = linear_program(
        c,
        G,
        h;
        Aeq=Aeq,
        beq=beq,
        T=T,
        sparse=sparse,
        validate=validate,
        verbosity=get(kwargs, :verbosity, 1),
    )
    return solve(problem; kwargs...)
end
