using MultiFloats

@testset "linear algebra backend planning" begin
    f64 = SDPX.Experimental.plan_la_backend(Float64)
    @test f64.selected === :standard
    @test f64.provider === :blas_lapack
    @test SDPX.Experimental.instantiate_la_backend(f64, Float64) isa
          SDPX.Experimental.StandardLABackend

    bf = SDPX.Experimental.plan_la_backend(BigFloat)
    @test bf.selected === :standard
    @test bf.provider === :generic_linear_algebra
    @test SDPX.Experimental.instantiate_la_backend(bf, BigFloat) isa
          SDPX.Experimental.StandardLABackend

    fixed = SDPX.Experimental.plan_la_backend(
        Float64x4;
        requested=:fixed_extended,
    )
    @test fixed.selected === :standard
    @test fixed.provider === :generic_linear_algebra

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

    factored = SDPX.la_cholesky_factor!(backend, [4.0 1.0; 1.0 3.0])
    @test factored isa SDPX.Experimental.StandardLACholeskyFactor
    solved = [1.0, 2.0]
    SDPX.la_cholesky_solve!(factored, solved)
    @test all(isfinite, solved)
end
