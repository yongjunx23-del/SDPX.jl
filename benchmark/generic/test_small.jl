using Test
include(joinpath(@__DIR__, "GenericConicBenchmark.jl"))
using .GenericConicBenchmark
include(joinpath(@__DIR__, "test_parsers.jl"))

@testset "generic conic benchmark inventory and readers" begin
    small = inventory(tier=:small)
    @test length(small) == 16
    @test Set(spec.family for spec in small) == Set((:lp, :socp, :sdp, :exp, :power))
    @test length(inventory(tier=:medium)) == 5
    @test length(inventory(tier=:large)) == 5

    netlib = only(filter(spec -> spec.id === :netlib_afiro, external_inventory()))
    @test reference_matches(netlib, -4.6475314286e2)
    @test !reference_matches(netlib, -400.0)

    if isfile(joinpath(@__DIR__, "data", "netlib", "afiro.mps"))
        @test read_external(netlib) isa MPSData
        @test read_external(only(filter(spec -> spec.id === :netlib_adlittle,
            external_inventory()))) isa MPSData
        @test read_external(only(filter(spec -> spec.id === :sdplib_control1,
            external_inventory()))) isa SDPAData
        @test read_external(only(filter(spec -> spec.id === :sdplib_mcp100,
            external_inventory()))) isa SDPAData
        # The checksum-pinned CBLIB fixture contains INT declarations. The
        # continuous reader must reject it rather than silently relax the MIP.
        @test_throws ArgumentError read_external(only(filter(
            spec -> spec.id === :cblib_expdesign_reader, external_inventory())))
    end
end

@testset "generic conic small tier" begin
    run = run_tier(:small; assert_seconds=30.0)
    @test length(run.results) == 16
    @test all(result -> result.expectation_met, run.results)
    @test count(result -> result.certificate_valid, run.results) == 10
    @test count(result -> !result.certificate_valid, run.results) == 6
end
