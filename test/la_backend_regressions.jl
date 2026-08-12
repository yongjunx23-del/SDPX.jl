using MultiFloats

@testset "linear algebra backend planning" begin
    f64 = SDPX.Experimental.plan_la_backend(Float64)
    @test f64.selected === :standard
    @test f64.provider === :none
    @test SDPX.Experimental.instantiate_la_backend(f64, Float64) isa
          SDPX.Experimental.StandardLABackend

    bf = SDPX.Experimental.plan_la_backend(BigFloat)
    @test bf.selected === :legacy
    @test bf.fallback_reason === :bigfloat_ownership
    @test SDPX.Experimental.instantiate_la_backend(bf, BigFloat) isa
          SDPX.Experimental.LegacyLABackend

    fixed = SDPX.Experimental.plan_la_backend(
        Float64x4;
        requested=:fixed_extended,
    )
    @test fixed.selected === :legacy
    @test fixed.fallback_reason === :missing_provider

    @test SDPX.Experimental.plan_la_backend(
        Float64;
        route=:block_arrow,
    ).fallback_reason === :route_not_migrated

    A = [4.0 1.0; 1.0 3.0]
    backend = SDPX.Experimental.StandardLABackend(:float64)
    @test SDPX.la_chol!(backend, A)
    rhs = [1.0, 2.0]
    SDPX.la_trsv_lower!(backend, A, rhs)
    @test all(isfinite, rhs)
end
