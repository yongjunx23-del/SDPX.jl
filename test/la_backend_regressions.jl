using MultiFloats: Float64x4

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
    auto_fixed = SDPX.Experimental.plan_la_backend(Float64x4)
    @test auto_fixed.selected === :standard
    @test auto_fixed.provider === :generic_linear_algebra

    # Non-dense routes stay on the historical backend for automatic/legacy
    # planning.  An explicit migrated backend must fail closed instead of
    # being silently rewritten to legacy.
    block_arrow = SDPX.Experimental.plan_la_backend(
        Float64;
        route=:block_arrow,
    )
    @test block_arrow.selected === :legacy
    @test block_arrow.fallback_reason === :route_not_migrated
    @test SDPX.Experimental.plan_la_backend(
        Float64;
        route=:block_arrow,
        requested=:legacy,
    ).selected === :legacy
    for request in (:standard, :multifloat, :fixed_extended)
        @test_throws ArgumentError SDPX.Experimental.plan_la_backend(
            Float64;
            route=:block_arrow,
            requested=request,
        )
    end

    # Positional plans from the pre-LA API carry the classification symbol
    # (`:fixed_extended`) rather than the concrete Float64x4 LA symbol.  The
    # Workspace compatibility seam normalizes only this explicitly marked
    # legacy descriptor before applying the modern exact arithmetic guard.
    classification = SDPX.ProblemClassification(
        :sdp,
        :dense,
        :fixed_extended,
        :small,
        2,
        1,
        1,
        2,
        0.5,
        0.5,
    )
    dense_config = SDPX.BackendConfiguration(
        :dense_cholesky,
        :auto,
        false,
        false,
        :off,
        (),
        false,
    )
    positional = SDPX.ExecutionPlan(
        classification,
        :sdp_primal_dual,
        :none,
        :dense_cholesky,
        dense_config,
        :pairwise_gram,
        :static,
        1,
        :general,
        0,
        (equality_solver=:auto,),
    )
    @test positional.la_config.arithmetic === :fixed_extended
    normalized = SDPX._normalize_compatibility_execution_plan(
        positional,
        Float64x4,
    )
    @test normalized.la_config.arithmetic ===
          SDPX._la_arithmetic_symbol(Float64x4)
    @test normalized.la_config.selected === :legacy
    @test normalized.la_config.fallback_reason === :compatibility

    A = [4.0 1.0; 1.0 3.0]
    backend = SDPX.Experimental.StandardLABackend(:float64)
    # Keep the dense-matrix signatures concrete enough that the generic
    # AbstractLABackend/AbstractArray fallback cannot become ambiguous for
    # Float64, BigFloat, or an optional MultiFloat provider.
    @test hasmethod(
        SDPX.la_cholesky_factor!,
        Tuple{SDPX.Experimental.StandardLABackend,Matrix{Float64}},
    )
    @test hasmethod(
        SDPX.la_cholesky_factor!,
        Tuple{SDPX.Experimental.StandardLABackend,Matrix{BigFloat}},
    )
    @test hasmethod(
        SDPX.la_cholesky_factor!,
        Tuple{SDPX.Experimental.MultiFloatLABackend{Any},Matrix{Float64x4}},
    )
    @test SDPX.la_chol!(backend, A)
    rhs = [1.0, 2.0]
    SDPX.la_trsv_lower!(backend, A, rhs)
    @test all(isfinite, rhs)

    factored = SDPX.la_cholesky_factor!(backend, [4.0 1.0; 1.0 3.0])
    @test factored isa SDPX.Experimental.StandardLACholeskyFactor
    solved = [1.0, 2.0]
    SDPX.la_cholesky_solve!(factored, solved)
    @test all(isfinite, solved)

    generic = SDPX.Experimental.StandardLABackend(
        :bigfloat,
        :generic_linear_algebra,
        :owned_mutable_scalars,
    )
    for value in (BigFloat(NaN), BigFloat(Inf))
        bad = BigFloat[4 1; 1 3]
        bad[1, 1] = value
        @test SDPX.la_cholesky_factor!(generic, bad) === nothing
        bad = BigFloat[4 1; 1 3]
        bad[1, 1] = value
        @test !SDPX.la_chol!(generic, bad)
    end
    @test SDPX.la_cholesky_factor!(generic, BigFloat[-1 0; 0 1]) === nothing
    @test !SDPX.la_chol!(generic, BigFloat[-1 0; 0 1])
end
