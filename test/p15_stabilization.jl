using SDPX
using Test

function _p15_bounded_capped_lp()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :sum, x[1] + x[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :upper_one, x[1] - 1.0, SDPX.Nonpositive())
    SDPX.constraint!(model, :upper_two, x[2] - 1.0, SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Maximize(), x[1] + 2.0 * x[2])
    return model
end

function _p15_genuinely_unbounded_lp()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    return model
end

@testset "P1.5 corrected bounded and unbounded classification" begin
    settings = SDPX.Settings{Float64}(
        engine=:native_hsd,
        kkt_route=:expanded,
        limits=SDPX.Limits(iterations=400, time=60.0, threads=1),
    )

    capped = SDPX.optimize!(_p15_bounded_capped_lp(); settings=settings)
    @test SDPX.status(capped) === :optimal
    @test SDPX.certificate(capped).valid
    @test isapprox(SDPX.primal_objective(capped), 2.0; atol=2e-6)

    unbounded = SDPX.optimize!(_p15_genuinely_unbounded_lp(); settings=settings)
    @test SDPX.status(unbounded) === :dual_infeasible
    @test SDPX.certificate(unbounded).valid
end
