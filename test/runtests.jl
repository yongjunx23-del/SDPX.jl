# Sole SDPX regression suite: black-box modeling-to-certified-result E2E.

using Test
using SDPX

include(joinpath(
    @__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl",
))
using .GenericConicBenchmark

const E2E_CASE_IDS = (
    :lp_afiro_style,
    :lp_infeasible,
    :lp_unbounded,
    :socp_portfolio_small,
    :sdp_maxcut_k4,
    :exp_unit_small,
    :power_epigraph_small,
)
function e2e_spec(id::Symbol)
    matches = filter(
        spec -> spec.id === id,
        GenericConicBenchmark.inventory(; tier=:small),
    )
    length(matches) == 1 || error(
        "expected exactly one general E2E case $id, found $(length(matches))",
    )
    return only(matches)
end

@testset "SDPX public modeling-to-certified-result E2E" begin
    for id in E2E_CASE_IDS
        @testset "$id" begin
            spec = e2e_spec(id)
            result = GenericConicBenchmark.run_one(spec, Float64)
            @test result.status === spec.expected_status
            @test result.certificate_valid
            @test result.expectation_met
        end
    end
end
