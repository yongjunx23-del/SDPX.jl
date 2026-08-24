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