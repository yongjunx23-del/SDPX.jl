# Phase 7 ChordalReduction validation (clique cover + chordality detection).
using SDPX
using Test

function _edge_covered(neighbours::Vector{Vector{Int}}, cliques::Vector{Vector{Int}})
    for i in 1:length(neighbours)
        for j in neighbours[i]
            j > i || continue
            any(c -> i in c && j in c, cliques) || return false
        end
    end
    return true
end

@testset "Phase 7 ChordalReduction" begin
    # A path graph 1-2-3-4-5 is a tree, hence chordal.
    path = [[2], [1, 3], [2, 4], [3, 5], [4]]
    order, position = SDPX.maximum_cardinality_search(path)
    @test SDPX.is_chordal(path, order, position)
    cliques = SDPX.maximal_cliques(path, order, position)
    # Every edge lies in some maximal clique (valid chordal cover).
    @test _edge_covered(path, cliques)
    # The path's maximal cliques are its 4 edges.
    @test length(cliques) == 4
    @test all(length(c) == 2 for c in cliques)

    # A 4-cycle WITHOUT a chord is NOT chordal.
    cycle = [[2, 4], [1, 3], [2, 4], [1, 3]]
    order2, position2 = SDPX.maximum_cardinality_search(cycle)
    @test !SDPX.is_chordal(cycle, order2, position2)
end