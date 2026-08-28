# Sole black-box end-to-end suite.
#
# Contract:
#   benchmark/general case -> public SDPX.optimize! -> BenchmarkResult
#
# This suite intentionally checks only terminal solve results in original
# coordinates. Provider selection, KKT routes, factor receipts, allocations,
# precision matrices, threading, RSS, and physics/bootstrap campaigns belong to
# separate integration or benchmark suites.

using Test
using SDPX

if !isdefined(Main, :GenericConicBenchmark)
    include(joinpath(
        @__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl",
    ))
end
using .GenericConicBenchmark

const _E2E_CASE_IDS = (
    :lp_afiro_style,
    :lp_infeasible,
    :lp_unbounded,
    :socp_portfolio_small,
    :sdp_maxcut_k4,
    :exp_unit_small,
    :power_epigraph_small,
)
const _E2E_SECONDS_PER_SOLVE = 10.0

function _e2e_spec(id::Symbol)
    matches = filter(spec -> spec.id === id, GenericConicBenchmark.inventory(; tier=:small))
    length(matches) == 1 || error("expected exactly one general E2E case $id, found $(length(matches))")
    return only(matches)
end

@testset "public solve-to-result E2E" begin
    for id in _E2E_CASE_IDS
        @testset "$id" begin
            spec = _e2e_spec(id)
            result = GenericConicBenchmark.run_one(
                spec,
                Float64;
                time_limit=_E2E_SECONDS_PER_SOLVE,
            )

            @test result.status === spec.expected_status
            @test result.certificate_valid
            @test result.expectation_met
            @test isfinite(result.seconds)
            @test result.seconds <= _E2E_SECONDS_PER_SOLVE
        end
    end
end
