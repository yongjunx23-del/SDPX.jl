# Phase 7 SymmetryReduction foundation tests.
using SDPX
using Test

@testset "Phase 7 SymmetryReduction" begin
    # Swap of indices 1 and 2 on n = 3.
    g = SDPX.SymmetryGroup([[2, 1, 3]], 3)
    @test g.n == 3

    # Orbits: {1,2} and {3}.
    orbits = SDPX.group_orbits(g)
    @test orbits == [[1, 2], [3]]

    # A matrix invariant under the swap is constant on pair orbits.
    A = [1.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 3.0]
    @test SDPX.is_matrix_invariant(A, g)
    # Breaking the symmetry makes it non-invariant.
    B = [1.0 2.0 0.0; 3.0 1.0 0.0; 0.0 0.0 3.0]
    @test !SDPX.is_matrix_invariant(B, g)

    # Reduction permutation groups indices by orbit.
    perm = SDPX.reduction_permutation(g)
    @test perm == [1, 2, 3]  # orbits [1,2],[3] are already in order

    # Reconstruction round-trips a reduced vector to original coordinates.
    g2 = SDPX.SymmetryGroup([[2, 3, 1]], 3)  # 3-cycle
    perm2 = SDPX.reduction_permutation(g2)
    @test sort(SDPX.group_orbits(g2)) == [[1, 2, 3]]
    reduced = [10.0, 20.0, 30.0]
    @test SDPX.reconstruct(reduced, perm2) isa Vector{Float64}
end

@testset "Phase 7 SymmetryReduction commutant (pair orbits)" begin
    g = SDPX.SymmetryGroup([[2, 1, 3]], 3)
    orbits = SDPX.pair_orbits(g)
    # Commutant equivalence: invariance iff constant on pair orbits.
    A = [1.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 3.0]
    @test SDPX.is_matrix_invariant(A, g)
    @test SDPX.is_constant_on_pair_orbits(A, orbits)
    B = [1.0 2.0 0.0; 3.0 1.0 0.0; 0.0 0.0 3.0]
    @test !SDPX.is_matrix_invariant(B, g)
    @test !SDPX.is_constant_on_pair_orbits(B, orbits)
    # Pair orbits match the expected invariant structure.
    # (1,1) and (2,2) are in one orbit; (1,2) and (2,1) in another.
    has_key = any(orb -> (1,1) in orb && (2,2) in orb, orbits)
    @test has_key
    cross = any(orb -> (1,2) in orb && (2,1) in orb, orbits)
    @test cross
end
