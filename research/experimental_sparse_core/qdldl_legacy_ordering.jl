# Run with a pinned legacy BFLA that has QDLDL but no ordering capability API.
# Its AMD route stays usable; natural requests must not retry through AMD.
include("experimental_sparse_core_identity.jl")

@testset "legacy BFLA ordering compatibility" begin
    @test SDPX.SparseQDLDLProviderOrderingAvailable(BigFloat, :amd)
    @test !SDPX.SparseQDLDLProviderOrderingAvailable(BigFloat, :natural)
    @test !SDPX.SparseQDLDLProviderOrderingAvailable(BigFloat, :unknown)
    U = sparse(BigFloat[2 1 0; 0 -2 0.25; 0 0 -1.5])
    @test_throws ArgumentError SDPX.SparseQDLDLCache{BigFloat}(U, [1, -1, -1]; ordering=:natural)
    @test_throws ArgumentError _identity_fixture(; ordering=:natural)
    cache = SDPX.SparseQDLDLCache{BigFloat}(U, [1, -1, -1])
    @test cache.ordering === :amd
    SDPX.factorize!(cache, U, 1)
    destination = zeros(BigFloat, 3)
    SDPX.solve!(cache, destination, BigFloat[1, 2, -1])
    @test all(isfinite, destination)
    @test SDPX.factor_diagnostics(cache).provider_ordering === :amd
    @test cache.numeric_count == 1 && cache.symbolic_count == 1
end
