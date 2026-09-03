#=====================================================================
    EXPERIMENTAL / OPT-IN — not reachable from `solve`.

    Tested building blocks, deliberately not wired into the automatic
    pipeline: no benchmark in this repository qualifies (see the Known
    limitations section of the README). Call the functions here directly.

    Chordal structure detection (plan §8.3, P2)

    A PSD constraint on a `k x k` block costs `O(k^3)` to factorize. When
    the block's *aggregate sparsity pattern* — the union of the nonzero
    patterns of `C` and every `A_i` acting on it — is chordal, the single
    constraint `X ⪰ 0` can be replaced by one smaller PSD constraint per
    maximal clique. If the cliques are much smaller than `k`, that turns
    one `k^3` factorization into several much cheaper ones.

    This module answers the question the decomposition depends on:
    *is the pattern chordal, what are its cliques, and would splitting
    actually be cheaper?* Detection is separated from the transformation
    deliberately — the answer is frequently "no benefit", and finding
    that out must not cost a rewrite of the problem.
=====================================================================#

"""
    aggregate_sparsity(prob, block) -> Vector{Vector{Int}}

Adjacency list of the aggregate sparsity graph for one PSD block.

The graph has one vertex per row of the block and an edge `(i, j)` whenever any
constraint matrix or the constant term has a structural nonzero there. It is the
union pattern that governs chordal decomposition: the decomposition is valid
only if *every* matrix in the constraint respects the clique structure, so a
single dense `A_i` makes the whole block dense regardless of the others.

Diagonal entries are ignored — every vertex is trivially adjacent to itself and
including it would only obscure the structure.
"""
function aggregate_sparsity(prob::SDPProblem{T}, block::Integer) where {T}
    dimension = prob.dims.k[block]
    neighbours = [Set{Int}() for _ in 1:dimension]

    function record!(row, column)
        row == column && return
        push!(neighbours[row], column)
        push!(neighbours[column], row)
        return
    end

    constant = prob.C[block]
    @inbounds for column in 1:dimension, row in 1:dimension
        iszero(constant[row, column]) || record!(row, column)
    end

    cons = prob.cons
    if cons isa SparseCons{T}
        for variable in cons.active[block]
            matrix = cons.Asp[block][variable]
            rows = rowvals(matrix)
            values = nonzeros(matrix)
            for column in 1:size(matrix, 2), index in nzrange(matrix, column)
                iszero(values[index]) || record!(rows[index], column)
            end
        end
    else
        coefficients = (cons::DenseCons{T}).Av[block]
        variables = prob.dims.m
        @inbounds for i in 1:variables
            slice = reshape(view(coefficients, :, i), dimension, dimension)
            for column in 1:dimension, row in 1:dimension
                iszero(slice[row, column]) || record!(row, column)
            end
        end
    end
    return [sort!(collect(set)) for set in neighbours]
end

"""
    aggregate_sparsity(canonical::CanonicalConicProgram, block::Integer) -> Vector{Vector{Int}}

Adjacency list of the aggregate sparsity graph for the `block`-th PSD cone
in a [`CanonicalConicProgram`](@ref).
"""
function aggregate_sparsity(canonical::CanonicalConicProgram{T}, block_idx::Integer) where {T}
    psd_blocks = [b for b in canonical.cone_layout.blocks if b.cone === :psd]
    1 <= block_idx <= length(psd_blocks) ||
        throw(BoundsError(psd_blocks, block_idx))
    block = psd_blocks[block_idx]
    dimension = block.dimension
    neighbours = [Set{Int}() for _ in 1:dimension]

    function record!(row, column)
        row == column && return
        push!(neighbours[row], column)
        push!(neighbours[column], row)
        return
    end

    pairs = psd_packed_pairs(dimension)
    offset = block.offset
    len = block.length

    # Check RHS b
    b = canonical.b
    @inbounds for i in 1:len
        r_idx = offset + i - 1
        if !iszero(b[r_idx])
            r, c = pairs[i]
            record!(r, c)
        end
    end

    # Check constraint matrix A (SparseMatrixCSC)
    A = canonical.A
    row_val = rowvals(A)
    nz_val = nonzeros(A)
    for col in 1:size(A, 2)
        for idx in nzrange(A, col)
            r_idx = row_val[idx]
            if offset <= r_idx < offset + len
                iszero(nz_val[idx]) && continue
                i = r_idx - offset + 1
                r, c = pairs[i]
                record!(r, c)
            end
        end
    end

    return [sort!(collect(set)) for set in neighbours]
end

"""
    maximum_cardinality_search(neighbours) -> (order, position)

Maximum cardinality search. Returns an elimination `order` (the reverse of the
visit order) and each vertex's `position` within it.

MCS is used because it is `O(V + E)` and produces a perfect elimination ordering
whenever the graph is chordal — so the same cheap pass both orders the graph and
supplies the certificate that `is_chordal` then checks.
"""
function maximum_cardinality_search(neighbours::Vector{Vector{Int}})
    dimension = length(neighbours)
    dimension == 0 && return (Int[], Int[])
    weight = zeros(Int, dimension)
    visited = falses(dimension)
    visit_order = Vector{Int}(undef, dimension)

    for step in 1:dimension
        best, best_weight = 0, -1
        @inbounds for vertex in 1:dimension
            visited[vertex] && continue
            if weight[vertex] > best_weight
                best, best_weight = vertex, weight[vertex]
            end
        end
        visited[best] = true
        visit_order[step] = best
        @inbounds for neighbour in neighbours[best]
            visited[neighbour] || (weight[neighbour] += 1)
        end
    end

    order = reverse(visit_order)          # perfect elimination order if chordal
    position = zeros(Int, dimension)
    @inbounds for (index, vertex) in pairs(order)
        position[vertex] = index
    end
    return (order, position)
end

"""
    is_chordal(neighbours, order, position) -> Bool

Whether `order` is a perfect elimination ordering, which holds exactly when the
graph is chordal.

The test: for each vertex, its neighbours that come later in the order must all
be adjacent to the earliest of them. A graph fails this precisely when it has an
induced cycle of length four or more — the chordless cycle that makes the
decomposition invalid.
"""
function is_chordal(neighbours::Vector{Vector{Int}}, order::Vector{Int},
                    position::Vector{Int})
    dimension = length(neighbours)
    dimension == 0 && return true
    adjacency = [Set(list) for list in neighbours]

    for vertex in order
        later = [n for n in neighbours[vertex] if position[n] > position[vertex]]
        isempty(later) && continue
        # The earliest later-neighbour must be adjacent to all the others.
        pivot = later[argmin([position[n] for n in later])]
        for other in later
            other == pivot && continue
            other in adjacency[pivot] || return false
        end
    end
    return true
end

"""
    maximal_cliques(neighbours, order, position) -> Vector{Vector{Int}}

Maximal cliques of a chordal graph, read off the perfect elimination ordering.

For a chordal graph each vertex together with its later neighbours forms a
clique, and the maximal ones are those not contained in another. This is why
chordality is worth testing first: on a chordal graph the cliques come out of a
single pass, whereas maximal-clique enumeration on a general graph is
exponential.
"""
function maximal_cliques(neighbours::Vector{Vector{Int}}, order::Vector{Int},
                         position::Vector{Int})
    candidates = Vector{Vector{Int}}()
    for vertex in order
        later = [n for n in neighbours[vertex] if position[n] > position[vertex]]
        push!(candidates, sort!(vcat(vertex, later)))
    end
    # Drop any clique contained in another.
    sort!(candidates; by=length, rev=true)
    cliques = Vector{Vector{Int}}()
    for candidate in candidates
        contained = any(clique -> issubset(candidate, clique), cliques)
        contained || push!(cliques, candidate)
    end
    return cliques
end

"""
    ChordalAnalysis

What was found for one PSD block, and whether splitting it is worth doing.

`beneficial` is the field that matters: chordality alone does not justify the
decomposition. A chordal pattern whose largest clique is nearly the whole block
splits one `k^3` factorization into several almost as large, while adding
coupling constraints between the overlapping cliques.
"""
struct ChordalAnalysis
    dimension::Int
    chordal::Bool
    cliques::Vector{Vector{Int}}
    largest_clique::Int
    density::Float64
    predicted_cost_ratio::Float64
    beneficial::Bool
end

"""Splitting is only worth it when the cliques cost this fraction of the dense
factorization or less; above it, the overlap bookkeeping outweighs the saving."""
const CHORDAL_MAXIMUM_COST_RATIO = 0.5

"""
    analyze_chordal_structure(prob, block) -> ChordalAnalysis

Detect chordal structure for one PSD block and predict whether decomposing it
would be cheaper.

The cost model compares `Σ_cliques |clique|³` against `k³`: factorization is
cubic in the block dimension, so the ratio of those sums is a direct estimate of
the arithmetic saved. A ratio near one means the cliques are nearly as large as
the block and nothing is gained.
"""
function analyze_chordal_structure(neighbours::Vector{Vector{Int}})
    dimension = length(neighbours)
    if dimension == 0
        return ChordalAnalysis(0, true, Vector{Int}[], 0, 0.0, 1.0, false)
    end

    edges = sum(length, neighbours) ÷ 2
    possible = dimension * (dimension - 1) ÷ 2
    density = possible == 0 ? 0.0 : edges / possible

    order, position = maximum_cardinality_search(neighbours)
    chordal = is_chordal(neighbours, order, position)
    cliques = chordal ? maximal_cliques(neighbours, order, position) : Vector{Int}[]
    largest = isempty(cliques) ? dimension : maximum(length, cliques)

    dense_cost = Float64(dimension)^3
    clique_cost = isempty(cliques) ? dense_cost :
                  sum(Float64(length(c))^3 for c in cliques)
    ratio = dense_cost == 0 ? 1.0 : clique_cost / dense_cost

    beneficial = chordal && length(cliques) > 1 &&
                 ratio <= CHORDAL_MAXIMUM_COST_RATIO
    return ChordalAnalysis(dimension, chordal, cliques, largest, density,
        ratio, beneficial)
end

analyze_chordal_structure(prob::SDPProblem, block::Integer) =
    analyze_chordal_structure(aggregate_sparsity(prob, block))

analyze_chordal_structure(canonical::CanonicalConicProgram, block::Integer) =
    analyze_chordal_structure(aggregate_sparsity(canonical, block))

"""
    chordal_summary(prob) -> Vector{ChordalAnalysis}
    chordal_summary(canonical) -> Vector{ChordalAnalysis}

Analysis for every PSD block, for diagnostics and for deciding whether the
decomposition is worth implementing on a given model family.
"""
chordal_summary(prob::SDPProblem) =
    [analyze_chordal_structure(prob, block) for block in 1:prob.dims.L]

chordal_summary(canonical::CanonicalConicProgram) =
    [analyze_chordal_structure(canonical, block) for block in 1:count(b -> b.cone === :psd, canonical.cone_layout.blocks)]
