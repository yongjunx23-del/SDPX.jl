# V4-A1 regression: mixed free variables + PSD + ZeroCone equalities.
# The old native HSD path reached a valid near-boundary point, then reported
# direction_breakdown because it never verified the current accepted iterate.

using Test
using SDPX

@testset "A1 mixed Reals PSD ZeroCone terminal verification" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(
        model, :psd_bound,
        [1.0 x[1]; x[1] 1.0], SDPX.PSDCone(),
    )
    SDPX.constraint!(model, :link, x[2] - x[1], SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Maximize(), x[2])

    result = SDPX.optimize!(
        model; settings=SDPX.Settings{Float64}(engine=:native_hsd),
    )
    @test SDPX.status(result) === :optimal
    @test result.certificate.valid
    @test result.certificate.method === :original_coordinates
    @test isapprox(SDPX.primal_objective(result), 1.0; atol=1e-8)
    @test isapprox(SDPX.dual_objective(result), 1.0; atol=1e-8)
end
