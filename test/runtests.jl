# Sole SDPX regression suite: black-box modeling-to-certified-result E2E.

using Test
using SDPX
using LinearAlgebra

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

@testset "LPLU factor cache lifecycle" begin
    # Keep this direct protocol fixture independent of the seven modeling E2Es:
    # Julia 1.10 does not provide LAPACK.getrf!(F, ipiv), so the production
    # cache must retain its own pivot storage while using the LAPACK kernel.
    cache = SDPX.LPLUCache{Float64}()
    @test SDPX.prepare!(cache, SDPX.FactorRequirements(2)) === cache
    @test SDPX.factor_status(cache) === SDPX.Prepared

    A = Float64[0 2; 1 3]  # requires a row pivot
    A2 = Float64[2 1; 1 3]
    @test SDPX.factorize!(cache, A, 1) === cache
    @test SDPX.factor_status(cache) === SDPX.Fresh
    @test SDPX.factor_epoch(cache) == 1

    x = zeros(2)
    rhs = A * [1.0, 2.0]
    @test SDPX.solve!(cache, x, rhs) === x
    @test x ≈ [1.0, 2.0]

    correction = zeros(2)
    residual = [1.0, -2.0]
    @test SDPX.refine_once!(cache, residual, correction) === correction
    @test A * correction ≈ residual

    # Numeric factorization and all solve variants remain allocation-free after
    # preparation/warm-up, including a fresh matrix epoch.
    SDPX.factorize!(cache, A2, 2)
    rhs2 = A2 * [2.0, -1.0]
    SDPX.solve!(cache, x, rhs2)
    @test x ≈ [2.0, -1.0]
    # Warm the new-epoch factorization call before measuring it so the check
    # observes cache behavior rather than first-call JIT compilation.
    SDPX.factorize!(cache, A, 3)
    @test @allocated(SDPX.factorize!(cache, A2, 4)) == 0
    @test @allocated(SDPX.solve!(cache, x, rhs)) == 0
    @test @allocated(SDPX.refine_once!(cache, residual, correction)) == 0
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
