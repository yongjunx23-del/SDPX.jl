module SDPXConvexExt

import Convex
import MathOptInterface as MOI
import LinearAlgebra
import SparseArrays
using SDPX

function SDPX.convex_semidefinite(
    side::Integer;
    representation::Symbol=:triangle,
    return_metadata::Bool=false,
)
    side > 0 || throw(ArgumentError("side must be positive"))
    representation in (:triangle, :square) || throw(ArgumentError(
        "representation must be :triangle or :square",
    ))
    if representation === :square
        matrix = Convex.Semidefinite(side)
        constraint = only(Convex.get_constraints(matrix))
        return return_metadata ?
               (
                   matrix=matrix,
                   packed=matrix,
                   constraint=constraint,
                   representation=:square,
               ) : matrix
    end

    packed = Convex.Variable(side * (side + 1) ÷ 2)
    coordinate(row, column) = begin
        r, c = row <= column ? (row, column) : (column, row)
        (c - 1) * c ÷ 2 + r
    end
    square_entries = side * side
    destination = collect(1:square_entries)
    source = Vector{Int}(undef, square_entries)
    @inbounds for column in 1:side, row in 1:side
        output = (column - 1) * side + row
        source[output] = coordinate(row, column)
    end
    # One sparse linear map is materially cheaper for Convex to canonicalize
    # than an n-row tree of scalar hcat/vcat atoms. Every output has exactly one
    # unit coefficient, and off-diagonal outputs deliberately share a source.
    unpack = SparseArrays.sparse(
        destination,
        source,
        ones(Int, square_entries),
        square_entries,
        length(packed),
    )
    matrix = reshape(unpack * packed, side, side)
    constraint = LinearAlgebra.isposdef(matrix)
    Convex.add_constraint!(packed, constraint)
    return return_metadata ?
           (
               matrix=matrix,
               packed=packed,
               constraint=constraint,
               representation=:triangle,
           ) : matrix
end

function SDPX.solve_convex!(
    problem;
    numeric_type::Type{T}=Float64,
    silent::Bool=true,
    warmstart::Bool=false,
    kwargs...,
) where {T<:AbstractFloat}
    optimizer = SDPX.convex_optimizer(T; kwargs...)
    return Convex.solve!(
        problem,
        optimizer;
        silent=silent,
        warmstart=warmstart,
    )
end

end
