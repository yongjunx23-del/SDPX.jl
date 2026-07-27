# A linear program, and how SDPX decides to solve it sparsely.
#
# LPs reach SDPX as 1x1 PSD blocks, and are routed to a dedicated scalar
# primal-dual path rather than through the general cone machinery. When the
# constraint matrix is sparse enough, that path factors a sparse Newton system
# instead of a dense one.
#
# The gate is measured, not assumed, and it is evaluated on the *assembled*
# system `GᵀDG` rather than on `G`: a single dense column of `G` fills the
# assembled matrix completely, so `G` being sparse does not imply the system
# is. This example builds one LP either side of that threshold and shows the
# selector reaching opposite conclusions.
#
# Run with:  julia --project=examples examples/03_sparse_lp.jl

using LinearAlgebra
using Printf
using SDPX
using SparseArrays

"""Build a strictly feasible LP with `entries_per_row` nonzeros per row.

Feasibility on both sides is by construction: an interior point is chosen
first and the right-hand side is set to put every slack at 1, then a strictly
positive multiplier vector defines the objective. That makes `Optimal` the
only correct answer, so a regression shows up as a status change rather than
as a slightly different number.

The data comes from an explicit linear congruential generator rather than
`MersenneTwister`. This example asserts a *routing decision* -- which
factorization the selector chooses -- and Julia's random stream is not stable
across versions, so seeded `randn` builds a slightly different problem on
each Julia release. The margin is wide today; an explicit generator makes the
example the same program everywhere, permanently.
"""
next_uniform(state::UInt64) =
    (0x5851f42d4c957f2d * state + 0x14057b7ef767814f,)[1]
draw(state) = (s = next_uniform(state); (s, 2.0 * (s >> 11) / (1 << 53) - 1.0))

function sparse_lp(; variables, base_rows, entries_per_row, seed=11)
    state = UInt64(seed)
    rows = spzeros(Float64, base_rows, variables)
    for row in 1:base_rows, _ in 1:entries_per_row
        state = next_uniform(state)
        column = 1 + Int(rem(state >> 11, UInt64(variables)))
        state, value = draw(state)
        rows[row, column] = value
    end
    # Box rows keep the program bounded.
    rows = [rows; sparse(1.0I, variables, variables); -sparse(1.0I, variables, variables)]
    total = size(rows, 1)

    interior = zeros(variables)
    for index in 1:variables
        state, interior[index] = draw(state)
    end
    righthand = rows * interior .- 1.0
    multipliers = zeros(total)
    for index in 1:total
        state, value = draw(state)
        multipliers[index] = abs(value) + 0.5
    end
    objective = vec(transpose(rows) * multipliers)

    coefficients = [
        [sparse([1], [1], [rows[row, column]], 1, 1) for column in 1:variables]
        for row in 1:total
    ]
    constants = [reshape([righthand[row]], 1, 1) for row in 1:total]
    return (objective=objective, coefficients=coefficients,
        constants=constants, rows=rows, variables=variables)
end

@printf("%18s %8s %10s %14s %10s %8s %16s\n",
    "entries per row", "rows", "variables", "nnz/row of K", "chosen", "status", "objective")

results = Dict{Int,Any}()
for entries_per_row in (2, 3)
    model = sparse_lp(; variables=220, base_rows=900, entries_per_row=entries_per_row)

    # What the selector sees: the assembled system, not the constraint matrix.
    assembled = transpose(model.rows) * model.rows
    formulation = SDPX.select_lp_formulation(;
        dimension=size(assembled, 1),
        nonzeros=nnz(assembled),
        equalities=0,
        arithmetic=Float64,
    )

    problem = SDPX.ingest(
        model.objective,
        model.coefficients,
        model.constants,
        zeros(model.variables, 0),
        Float64[];
        sparse=true,
        verbosity=0,
    )
    result = solve(problem; tolerance=1e-9, verbosity=0)
    results[entries_per_row] = (formulation=formulation, result=result)

    @printf("%18d %8d %10d %14.1f %10s %8s %16.8f\n",
        entries_per_row, size(model.rows, 1), model.variables,
        nnz(assembled) / size(assembled, 1), formulation,
        result.status, result.pObj)
end

println("""

Two entries per row assembles to about 9 nonzeros per row and is routed to a
sparse Cholesky; three entries assembles to about 24 and falls back to the
dense LU, because below roughly 13 the sparse factorization pays for itself
and above it the dense one is faster. Note how far the assembled density is
from the constraint density that produced it -- 2 and 3 entries per row of G
become 9 and 24 in GᵀG -- which is why the gate is evaluated after assembly.

The two rows solve different LPs, so their objectives differ; what is being
compared here is the routing decision, not the answer.""")

results[2].formulation === :sparse_normal ||
    error("2 entries/row should select :sparse_normal, got $(results[2].formulation)")
results[3].formulation === :dense_lu ||
    error("3 entries/row should select :dense_lu, got $(results[3].formulation)")
for (entries, outcome) in results
    outcome.result.status == SDPX.Optimal ||
        error("$(entries) entries/row: expected Optimal, got $(outcome.result.status)")
end
