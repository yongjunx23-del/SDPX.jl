using Test

isdefined(Main, :SDPXBenchmarkRegistry) || include(joinpath(
    @__DIR__, "..", "benchmark", "SDPXBenchmarkRegistry.jl",
))
using .SDPXBenchmarkRegistry

@testset "fixed LP/SOCP/SDP precision matrix" begin
    @test campaign_names() == (:core_matrix,)

    entries = suite_entries(:core_matrix)
    @test length(entries) == 9
    @test length(unique((
        entry.problem_id,
        entry.arithmetic,
        entry.provider,
    ) for entry in entries)) == 9

    problem_ids = (
        "synthetic/lp_box",
        "synthetic/soc_q3",
        "synthetic/sdp_dense",
    )
    expected_families = Dict(
        "synthetic/lp_box" => :lp,
        "synthetic/soc_q3" => :socp,
        "synthetic/sdp_dense" => :sdp,
    )
    expected_arithmetic = Set((:float64, :float64x4, :bigfloat256))
    expected_provider = Dict(
        :float64 => :auto,
        :float64x4 => :multifloat,
        :bigfloat256 => :bfla,
    )

    for problem_id in problem_ids
        selected = filter(entry -> entry.problem_id == problem_id, entries)
        @test length(selected) == 3
        @test Set(entry.arithmetic for entry in selected) == expected_arithmetic
        @test all(
            entry.provider === expected_provider[entry.arithmetic]
            for entry in selected
        )
        @test benchmark_spec(problem_id).family === expected_families[problem_id]
    end
end
