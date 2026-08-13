using Test

@testset "BFLA optional provider core contract" begin
    LA = SDPX.Experimental

    # Without the optional package loaded, explicit BFLA requests fail at
    # planning time; no runtime try-and-switch fallback exists.
    descriptor = LA.la_provider_descriptor(BigFloat, 1)
    if !descriptor.available || descriptor.provider !== :bigfloat_linear_algebra
        @test_throws ArgumentError LA.plan_la_backend(
            BigFloat;
            requested=:bfla,
        )
    end
    @test_throws ArgumentError LA.plan_la_backend(
        Float64;
        requested=:bfla,
    )
    @test_throws ArgumentError LA.plan_la_backend(
        BigFloat;
        requested=:bfla,
        route=:block_arrow,
    )

    # Optional-provider presence may only affect BigFloat automatic planning;
    # explicit standard and legacy policies remain stable reference/rollback
    # routes.
    @test LA.plan_la_backend(
        BigFloat;
        requested=:standard,
    ).selected === :standard
    @test LA.plan_la_backend(
        BigFloat;
        requested=:legacy,
    ).selected === :legacy
end
