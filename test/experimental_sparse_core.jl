# Memory-admission checks only. Numerical research is explicitly separated
# into experimental_sparse_core_numerics/identity.jl and is NOT admitted.
using Test, SDPX, SparseArrays, LinearAlgebra

@testset "experimental sparse memory admission is unavailable" begin
    setprecision(BigFloat, 256) do
        A = sparse(BigFloat[1 0; 1 1; 0 1])
        cone = SDPX.ProductConeLinearization{BigFloat}(
            Matrix{BigFloat}(Diagonal(BigFloat[2, 3, 5])), zeros(BigFloat, 3),
            UnitRange{Int}[1:1, 2:2, 3:3])
        rhs = SDPX.HSDNewtonRHS(zeros(BigFloat, 3), zeros(BigFloat, 2),
            BigFloat(0), zeros(BigFloat, 3), BigFloat(0))
        system = SDPX.NewtonSystem(A, BigFloat[1, 2, 3], BigFloat[4, 5],
            cone, BigFloat(1), BigFloat(1), rhs)
        inventory = SDPX.experimental_sparse_core_memory_inventory(2, 3, nnz(A))
        @test inventory.proven === false
        @test inventory.reason === :owned_storage_bound_unavailable
        @test inventory.assumed_ordering === :natural
        q, d, a = nnz(A) + 5, 5, nnz(A)
        @test inventory.array_payload_bigfloat_slots == 8q + a + 18d + d * (d - 1) ÷ 2
        @test inventory.returned_pair_bigfloat_slots == 2 * 2 + 5 * 3 + 4
        @test inventory.acceptance_array_bigfloat_slots == 2 + 5 * 3
        @test !isempty(inventory.unresolved)
        before = lock(SDPX._SYMMETRIC_CORE_STRUCTURE_LOCK) do
            c = SDPX._SYMMETRIC_CORE_STRUCTURE_CACHE
            (c.hits, c.misses, length(c.patterns))
        end
        for (limit, rss) in ((typemax(Int), 0), (nothing, 0),
                              (1_000_000, nothing), (-1, 0), (1, -1), (1, 1))
            caught = try
                SDPX.prepare_experimental_sparse_core_state(system,
                    SDPX.IdentityRankBasis(BigFloat, 2), [:lp, :lp, :lp], [1, 2],
                    BigFloat("1e-30"), 256, limit, rss)
                nothing
            catch e
                e
            end
            @test caught isa ArgumentError
            msg = caught === nothing ? "" : sprint(showerror, caught)
            @test occursin("memory admission refused", msg)
            if limit === typemax(Int)
                @test occursin("owned_storage_bound_unavailable", msg)
            end
        end
        after = lock(SDPX._SYMMETRIC_CORE_STRUCTURE_LOCK) do
            c = SDPX._SYMMETRIC_CORE_STRUCTURE_CACHE
            (c.hits, c.misses, length(c.patterns))
        end
        @test before == after # no template/cache construction or lookup
        for (n, m, a) in ((-1, 1, 0), (1, -1, 0), (1, 1, -1), (1, 1, 2),
                          (typemax(Int), 1, 0), (32, 33, 0))
            @test_throws ArgumentError SDPX.experimental_sparse_core_memory_inventory(n, m, a)
        end
    end
end
