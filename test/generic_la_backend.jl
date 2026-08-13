using GenericLinearAlgebra
using LinearAlgebra
using MultiFloats: Float64x4
using Test

@testset "generic LA provider capabilities" begin
    LA = SDPX.Experimental

    for T in (Float64, BigFloat, Float64x4)
        config = LA.plan_la_backend(T; equality_solver=:auto)
        capabilities = config.capability_model
        @test capabilities isa LA.LAProviderCapabilities
        @test LA.la_provider_supports(capabilities, :cholesky)
        @test LA.la_provider_supports(capabilities, :lu)
        @test LA.la_provider_supports(capabilities, :qr)
        @test LA.la_provider_supports(capabilities, :rank_revealing_qr)
        @test LA.la_provider_supports(capabilities, :factor_solve)
        @test LA.la_provider_supports(capabilities, :multi_rhs)
        @test !LA.la_provider_supports(
            capabilities,
            :pivoted_symmetric_ldlt,
        )
        @test config.required_capabilities == SDPX._DENSE_CHOLESKY_REQUIRED
        @test LA.validate_la_backend_configuration(config, T) === config
    end

    f64 = LA.plan_la_backend(Float64)
    @test f64.provider_implementation === :julia_blas_lapack
    @test f64.capability_model.threading

    big = LA.plan_la_backend(BigFloat)
    @test big.provider === :generic_linear_algebra
    @test big.provider_implementation === :julia_generic_with_gla_loaded
    @test !big.capability_model.threading

    @test_throws ArgumentError LA.la_provider_supports(
        big.capability_model,
        :not_a_capability,
    )

    # Planner/execution capability validation fails before an operation is
    # attempted; there is no try-and-switch runtime fallback.
    incomplete = SDPX.LABackendConfiguration(
        :bigfloat,
        :standard,
        :standard,
        :generic_linear_algebra,
        (:cholesky,),
        SDPX.LAProviderCapabilities(cholesky=true),
        (:cholesky, :factor_solve, :multi_rhs),
        :julia_generic_with_gla_loaded,
        (),
        :none,
        :owned_mutable_scalars,
    )
    @test_throws ArgumentError LA.validate_la_backend_configuration(incomplete)
    @test_throws ArgumentError LA.instantiate_la_backend(incomplete, BigFloat)

    false_claim = SDPX.LABackendConfiguration(
        :bigfloat,
        :standard,
        :standard,
        :generic_linear_algebra,
        (:pivoted_symmetric_ldlt,),
        SDPX.LAProviderCapabilities(pivoted_symmetric_ldlt=true),
        (),
        :julia_generic_with_gla_loaded,
        (),
        :none,
        :owned_mutable_scalars,
    )
    @test_throws ArgumentError LA.instantiate_la_backend(false_claim, BigFloat)

    for T in (Float64, BigFloat, Float64x4)
        backend = LA.instantiate_la_backend(LA.plan_la_backend(T), T)
        source = T[4 1; 1 3]
        rhs = T[1 2; 3 5]

        chol_rhs = copy(rhs)
        chol_buffer = SDPX._owned_array_copy(T, source)
        chol = LA.la_cholesky_factor!(backend, chol_buffer)
        @test chol !== nothing
        LA.la_factor_solve!(chol, chol_rhs)
        @test source * chol_rhs ≈ rhs

        lu_rhs = copy(rhs)
        lu_buffer = SDPX._owned_array_copy(T, source)
        lu_factor = LA.la_lu_factor!(backend, lu_buffer)
        @test lu_factor !== nothing
        LA.la_factor_solve!(lu_factor, lu_rhs)
        @test source * lu_rhs ≈ rhs

        qr_rhs = copy(rhs)
        qr_buffer = SDPX._owned_array_copy(T, source)
        qr_factor = LA.la_qr_factor!(backend, qr_buffer; pivoted=true)
        LA.la_factor_solve!(qr_factor, qr_rhs)
        @test source * qr_rhs ≈ rhs
    end

    legacy = LA.instantiate_la_backend(
        LA.plan_la_backend(Float64; requested=:legacy),
        Float64,
    )
    @test_throws ArgumentError LA.la_lu_factor!(legacy, [1.0 0.0; 0.0 1.0])
    @test_throws ArgumentError LA.la_qr_factor!(legacy, [1.0 0.0; 0.0 1.0])
end
