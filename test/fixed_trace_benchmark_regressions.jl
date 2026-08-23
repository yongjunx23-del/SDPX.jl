module FixedTraceBenchmarkRegressions

using Test
using MultiFloats: Float64x2

include(joinpath(@__DIR__, "..", "benchmark", "soc_fixed_trace", "benchmark.jl"))

@testset "fixed-trace benchmark reporting and option gates" begin
    flattened = Dict{String,Any}()
    _flatten!(
        flattened,
        (arithmetic_type=Float64x2,),
        "executed",
    )
    @test flattened["executed_arithmetic_type"] isa String
    @test occursin(
        "MultiFloat",
        flattened["executed_arithmetic_type"],
    )

    @test_throws ErrorException parse_cli([
        "--synthetic",
        "--mode=sdp",
        "--q3-direction=nt",
    ])
    @test_throws ErrorException parse_cli([
        "--synthetic",
        "--mode=sdp",
        "--q3-gram-strategy=output_tiles",
    ])

    mktempdir() do root
        mkpath(joinpath(root, "src"))
        mkpath(joinpath(root, "ext"))
        mkpath(joinpath(root, "benchmark", "soc_fixed_trace"))
        write(joinpath(root, "Project.toml"), "name = \"HashFixture\"\n")
        write(joinpath(root, "src", "HashFixture.jl"), "module HashFixture end\n")
        extension = joinpath(root, "ext", "HashFixtureExt.jl")
        write(extension, "module HashFixtureExt end\n")
        first_hash = _source_tree_sha256(root)
        write(extension, "module HashFixtureExt\nconst changed = true\nend\n")
        @test _source_tree_sha256(root) != first_hash
    end
end

end
