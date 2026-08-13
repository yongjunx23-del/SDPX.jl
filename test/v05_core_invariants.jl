using SDPX
using Test

# Tiny deterministic fixtures for the certification handoff.  No solve is
# needed: the invariant is that a disabled policy never runs the independent
# certificate and never rewrites the raw core result.
function v05_scalar_certificate_fixture()
    coefficients = [reshape([1.0], 1, 1, 1)]
    problem = SDPX.ingest(
        [1.0],
        coefficients,
        [zeros(1, 1)],
        zeros(1, 0),
        Float64[];
        verbosity=0,
    )
    result = SDPX.SDPResult{Float64}(
        SDPX.Optimal,
        "candidate",
        [0.0],
        [reshape([-1.0e-3], 1, 1)],
        Float64[],
        [reshape([1.0], 1, 1)],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        nothing,
        NamedTuple[],
        nothing,
        (reason=:none,),
    )
    return problem, result
end

function v05_mismatched_la_config(provider::Symbol, arithmetic::Symbol)
    return SDPX.Experimental.LABackendConfiguration(
        arithmetic,
        :standard,
        :standard,
        provider,
        (:cholesky,),
        SDPX.Experimental.LAProviderCapabilities(cholesky=true),
        (:cholesky, :factor_solve, :multi_rhs),
        provider === :blas_lapack ? :julia_blas_lapack :
        :julia_generic_with_gla_loaded,
        (),
        :none,
        arithmetic === :float64 ? :immutable_scalars : :owned_mutable_scalars,
    )
end

@testset "v0.5 core invariants" begin
    @testset "standard auto equality QR plan authorization" begin
        config = SDPX.Experimental.plan_la_backend(
            Float64;
            requested=:auto,
            equality_solver=:auto,
        )
        @test config.selected === :standard
        @test config.provider === :blas_lapack
        @test config.fallback_chain === (:rank_revealing_qr,)
        @test SDPX.Experimental.la_provider_supports(
            config.capability_model,
            :rank_revealing_qr,
        )
        @test SDPX.Experimental.validate_la_backend_configuration(
            config,
            Float64,
        ) === config
        backend = SDPX.Experimental.instantiate_la_backend(config, Float64)
        @test backend isa SDPX.Experimental.StandardLABackend
        @test SDPX.la_backend_provider(backend) === :blas_lapack
    end

    @testset "explicit normal_equations disables QR" begin
        for requested in (:auto, :standard, :legacy)
            config = SDPX.Experimental.plan_la_backend(
                Float64;
                requested=requested,
                equality_solver=:normal_equations,
            )
            @test config.fallback_chain === ()
            @test :qr ∉ config.required_capabilities
            @test :rank_revealing_qr ∉ config.required_capabilities
        end
    end

    @testset "legacy and route provenance never silently change" begin
        dense = SDPX.Experimental.plan_la_backend(
            Float64;
            requested=:legacy,
        )
        @test dense.selected === :legacy
        @test dense.provider === :sdpx_legacy_la
        @test dense.fallback_reason === :requested_legacy

        route = SDPX.Experimental.plan_la_backend(
            Float64;
            route=:block_arrow,
        )
        @test route.selected === :legacy
        @test route.provider === :sdpx_legacy_la
        @test route.fallback_reason === :route_not_migrated

        for config in (dense, route)
            backend = SDPX.Experimental.instantiate_la_backend(
                config,
                Float64,
            )
            @test backend isa SDPX.Experimental.LegacyLABackend
            @test SDPX.la_backend_name(backend) === :legacy
            @test SDPX.la_backend_provider(backend) === :sdpx_legacy_la
            @test SDPX.la_backend_reason(backend) ===
                  config.fallback_reason
            @test SDPX.la_backend_ownership(backend) === config.ownership
        end

        # An explicit migrated request on an unmigrated route must fail
        # closed instead of being rewritten to the legacy backend.
        @test_throws ArgumentError SDPX.Experimental.plan_la_backend(
            Float64;
            route=:block_arrow,
            requested=:standard,
        )
    end

    @testset "QR validation follows provider capability model" begin
        no_qr = SDPX.Experimental.LABackendConfiguration(
            :float64,
            :standard,
            :standard,
            :generic_linear_algebra,
            (:cholesky,),
            SDPX.Experimental.LAProviderCapabilities(cholesky=true),
            (:cholesky,),
            :julia_blas_lapack,
            (:rank_revealing_qr,),
            :none,
            :immutable_scalars,
        )
        @test_throws ArgumentError (
            SDPX.Experimental.validate_la_backend_configuration(
                no_qr,
                Float64,
            )
        )
    end

    @testset "certification=false returns certification_disabled" begin
        problem, invalid = v05_scalar_certificate_fixture()
        options = SDPX.SolverOptions{Float64}(
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            certification=false,
            verbosity=0,
        )
        final, certificate, warning =
            SDPX.certify_final_result(problem, invalid, options)
        @test final === invalid
        @test final.status == SDPX.Optimal
        @test certificate == (available=false, reason=:certification_disabled)
        @test warning === nothing
        @test final.termination.reason == :none

        resolved = SDPX.Experimental.resolve_solve_options(
            Float64,
            SDPX.SolveOptions(certification=false),
        )
        @test resolved.certification === false
        @test resolved.core.certification === false
        @test resolved.summary.certification === false
    end

    @testset "modern capability mismatch fails closed" begin
        for provider in (:blas_lapack, :generic_linear_algebra)
            config = v05_mismatched_la_config(provider, :float64)
            @test_throws ArgumentError (
                SDPX.Experimental.validate_la_backend_configuration(config)
            )
            @test_throws ArgumentError (
                SDPX.Experimental.instantiate_la_backend(config, Float64)
            )
        end
    end
end
