using SDPX
using LinearAlgebra
using Test

if !isdefined(@__MODULE__, :soc_psd_reference_problem)
    include(joinpath(@__DIR__, "helpers", "soc_psd_reference.jl"))
end

# Tiny deterministic fixtures for the certification handoff. No solve is
# needed: disabling the detailed certificate payload must still preserve the
# minimal original-coordinate success semantics of `Optimal`.
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

function v05_raw_float32_sdp_problem()
    source = SDPX.ingest(
        [1.0],
        [reshape([1.0], 1, 1, 1)],
        [zeros(1, 1)],
        zeros(1, 0),
        Float64[];
        sparse=false,
        verbosity=0,
    )
    cons = SDPX.DenseCons{Float32}(
        [Float32.(matrix) for matrix in source.cons.Av],
    )
    return SDPX.SDPProblem{Float32}(
        Float32.(source.c),
        [Float32.(matrix) for matrix in source.C],
        Float32.(source.B),
        Float32.(source.b),
        cons,
        source.dims,
        source.structure,
    )
end

@testset "v0.5 core invariants" begin
    @testset "planning rejects manually constructed unsupported arithmetic" begin
        raw = v05_raw_float32_sdp_problem()
        @test raw isa SDPX.SDPProblem{Float32}
        @test_throws ArgumentError SDPX.build_execution_plan(
            raw,
            SDPX.SolverOptions{Float32}(
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
            ),
        )
        @test_throws ArgumentError SDPX.solve!(raw, SDPX.SolverOptions{Float32}(
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            verbosity=0,
        ))

        raw_cone = SDPX.SOCConstraint{Float32}(
            reshape(Float32[1], 1, 1),
            Float32[0],
        )
        raw_soc = SDPX.ConicProblem{Float32}(
            Float32[1],
            [raw_cone],
            zeros(Float32, 0, 1),
            Float32[],
            1,
        )
        @test_throws ArgumentError SDPX.plan_native_soc(
            raw_soc,
            SDPX.SolverOptions{Float32}(verbosity=0),
        )
    end

    @testset "typed formulation precedes backend planning" begin
        dense_problem, _ = v05_scalar_certificate_fixture()
        dense_plan = SDPX.Experimental.build_execution_plan(
            dense_problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
                threads=1,
            ),
        )
        @test dense_plan.formulation_plan isa
              SDPX.Experimental.FormulationPlan{
                  SDPX.Experimental.DenseNormalEquations,
              }
        @test dense_plan.formulation_plan.provenance ===
              :automatic_formulation_planner
        @test dense_plan.kkt_formulation === :dense_normal_equations
        @test dense_plan.parameters.formulation_decision.requested === :auto
        @test dense_plan.parameters.formulation_decision.selected ===
              :dense_normal_equations
        @test dense_plan.parameters.formulation_decision.reason in (
            :default_dense_normal_equations,
            :equality_quality_unavailable,
        )
        @test SDPX.kkt_backend_from_formulation(
            dense_plan.formulation_plan,
            dense_plan.algorithm,
            dense_plan.classification.equalities,
        ) === dense_plan.kkt_backend
        @test SDPX.kkt_backend_matches_formulation(
            :dense_cholesky_fallback,
            dense_plan.formulation_plan,
            dense_plan.algorithm,
            dense_plan.classification.equalities,
        )
        fallback_backend_config = SDPX.BackendConfiguration(
            :dense_cholesky_fallback,
            dense_plan.backend_config.equality_solver,
            dense_plan.backend_config.reduced_arrow,
            dense_plan.backend_config.mixed_reduced_arrow,
            dense_plan.backend_config.mixed_precision_mode,
            dense_plan.backend_config.fallback_chain,
            dense_plan.backend_config.deferred,
        )
        historical_fallback_plan = SDPX.ExecutionPlan(
            dense_plan.classification,
            dense_plan.algorithm,
            dense_plan.scaling,
            :dense_cholesky_fallback,
            fallback_backend_config,
            :dense_normal_equations,
            dense_plan.la_config,
            dense_plan.gram_kernel,
            dense_plan.schedule,
            dense_plan.threads,
            dense_plan.parameter_profile,
            dense_plan.memory_budget_bytes,
            dense_plan.parameters,
        )
        historical_workspace = SDPX.Workspace(
            dense_problem;
            execution_plan=historical_fallback_plan,
        )
        @test SDPX.select_backend(historical_workspace) isa
              SDPX.DenseCholeskyBackend
        @test_throws ArgumentError SDPX.kkt_backend_from_formulation(
            dense_plan.formulation_plan,
            :lp_primal_dual,
            dense_plan.classification.equalities,
        )

        lp_plan = SDPX.Experimental.build_execution_plan(
            dense_problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:lp,
                presolve=false,
                scaling=:none,
                threads=1,
            ),
        )
        @test lp_plan.formulation_plan.formulation isa
              SDPX.Experimental.NoKKTFormulation
        @test lp_plan.kkt_formulation === :not_applicable
        @test SDPX.kkt_backend_from_formulation(
            lp_plan.formulation_plan,
            lp_plan.algorithm,
            lp_plan.classification.equalities,
        ) === lp_plan.kkt_backend

        # Dualization has analysis metadata but is not a production solve
        # option; an explicit request must fail before backend planning.
        @test_throws ArgumentError SDPX.Experimental.build_execution_plan(
            dense_problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                formulation=:dual,
                presolve=false,
                scaling=:none,
                threads=1,
            ),
        )

        explicit_normal = SDPX.Experimental.build_execution_plan(
            dense_problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                formulation=:normal_equations,
                presolve=false,
                scaling=:none,
                threads=1,
            ),
        )
        @test explicit_normal.kkt_formulation === :dense_normal_equations
        @test explicit_normal.parameters.formulation_decision.reason ===
              :user_forced_normal

        explicit_primal = SDPX.Experimental.build_execution_plan(
            dense_problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                formulation=:primal,
                presolve=false,
                scaling=:none,
                threads=1,
            ),
        )
        @test explicit_primal.kkt_formulation === :dense_normal_equations
        @test explicit_primal.parameters.formulation_decision.reason ===
              :user_forced_normal

        # The historical :primal option fixes orientation only. It must not
        # erase an exact structural block-arrow choice introduced before the
        # dense formulation planner.
        arrow_coefficients = [
            reshape([1.0, 0.0], 2, 1, 1),
            reshape([0.0, 1.0], 2, 1, 1),
        ]
        arrow_problem = SDPX.ingest(
            [1.0, 1.0],
            arrow_coefficients,
            [zeros(1, 1), zeros(1, 1)],
            zeros(2, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        primal_arrow = SDPX.Experimental.build_execution_plan(
            arrow_problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                formulation=:primal,
                presolve=false,
                scaling=:none,
                threads=1,
            ),
        )
        @test primal_arrow.formulation_plan.formulation isa
              SDPX.Experimental.BlockArrowElimination
        @test isempty(primal_arrow.parameters.formulation_decision.candidates)

        explicit_qr = SDPX.Experimental.build_execution_plan(
            dense_problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                formulation=:auto,
                equality_solver=:qr,
                presolve=false,
                scaling=:none,
                threads=1,
            ),
        )
        @test explicit_qr.kkt_formulation === :dense_normal_equations
        @test explicit_qr.parameters.formulation_decision.candidates[2].reason ===
              :augmented_incompatible_equality_solver

        @test SDPX.dense_augmented_workspace_floor_bytes(
            Float64,
            6,
            5,
            1,
            1,
        ) > SDPX.dense_workspace_floor_bytes(Float64, 6, 5, 1, 1)
        @test SDPX.estimate_dense_augmented_workspace_bytes(
            dense_problem,
            1,
        ) > SDPX.estimate_dense_workspace_bytes(dense_problem, 1)
        @test SDPX.estimate_dense_workspace_bytes(dense_problem, 1) >=
              SDPX.estimate_sdp_workspace_bytes(dense_problem, 1)

        lp_problem = SDPX.ingest(
            [1.0],
            [reshape([1.0], 1, 1, 1)],
            [zeros(1, 1)],
            zeros(1, 0),
            Float64[];
            sparse=false,
            verbosity=0,
        )
        @test_throws ArgumentError SDPX.Experimental.build_execution_plan(
            lp_problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:lp,
                formulation=:normal_equations,
                presolve=false,
                scaling=:none,
                threads=1,
            ),
        )

        conic = SDPX.second_order_program(
            [1.0, 0.0, 0.0],
            [SDPX.SOCConstraint(
                Matrix{Float64}(LinearAlgebra.I, 3, 3),
                zeros(3);
                T=Float64,
            )];
            T=Float64,
        )
        normal_soc = SDPX.Experimental.build_execution_plan(
            soc_psd_reference_problem(conic; verbosity=0),
            SDPX.SolverOptions{Float64}(
                algorithm=:socp,
                formulation=:normal_equations,
                presolve=false,
                scaling=:none,
                threads=1,
            ),
        )
        @test normal_soc.kkt_formulation === :dense_normal_equations

    end

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

    @testset "certification=false keeps the minimal Optimal gate" begin
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
        @test final.status == SDPX.Stalled
        @test !certificate.available
        @test certificate.reason == :certification_disabled
        @test certificate.minimal_gate.available
        @test !certificate.minimal_gate.valid
        @test warning !== nothing
        @test final.termination.reason ==
              :minimal_original_coordinate_gate_failed

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

    @testset "dense LP routes use the migrated provider seam" begin
        LA = SDPX.Experimental
        pd = LA.plan_la_backend(
            Float64;
            route=:positive_definite_cholesky,
        )
        @test pd.selected === :standard
        @test pd.provider === :blas_lapack
        @test :cholesky in pd.required_capabilities
        @test :lu ∉ pd.required_capabilities
        @test pd.fallback_chain === ()
        @test pd.fallback_reason === :none

        lu = LA.plan_la_backend(Float64; route=:dense_lu)
        @test lu.selected === :standard
        @test lu.provider === :blas_lapack
        @test :lu in lu.required_capabilities
        @test :cholesky ∉ lu.required_capabilities
        @test lu.fallback_chain === ()
        @test lu.fallback_reason === :none

        for route in (:positive_definite_cholesky, :dense_lu)
            legacy = LA.plan_la_backend(
                Float64;
                route=route,
                requested=:legacy,
            )
            @test legacy.selected === :legacy
            @test legacy.provider === :sdpx_legacy_la
            @test legacy.fallback_reason === :requested_legacy
        end
    end

    @testset "specialized LP-adjacent routes remain non-migrated" begin
        LA = SDPX.Experimental
        for route in (:block_arrow, :q3_block_diagonal_equality)
            automatic = LA.plan_la_backend(Float64; route=route)
            @test automatic.selected === :legacy
            @test automatic.provider === :sdpx_legacy_la
            @test automatic.fallback_reason === :route_not_migrated
            legacy = LA.plan_la_backend(
                Float64;
                route=route,
                requested=:legacy,
            )
            @test legacy.selected === :legacy
            @test legacy.fallback_reason === :route_not_migrated
            @test_throws ArgumentError LA.plan_la_backend(
                Float64;
                route=route,
                requested=:standard,
            )
        end
    end

    @testset "dense LP provider requests fail closed" begin
        LA = SDPX.Experimental
        if Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) === nothing
            for route in (:positive_definite_cholesky, :dense_lu)
                @test_throws ArgumentError LA.plan_la_backend(
                    BigFloat;
                    requested=:bfla,
                    route=route,
                )
            end
        end
        if Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt) === nothing
            @test_throws ArgumentError LA.plan_la_backend(
                Float64;
                requested=:multifloat,
                route=:dense_lu,
            )
        end

        # A provider that claims the dense-LP route without the required LU
        # capability must be rejected during validation, never executed.
        incomplete = SDPX.Experimental.LABackendConfiguration(
            :float64,
            :standard,
            :standard,
            :blas_lapack,
            (:cholesky,),
            SDPX.Experimental.LAProviderCapabilities(cholesky=true),
            (:lu, :factor_solve),
            :julia_blas_lapack,
            (),
            :none,
            :immutable_scalars,
        )
        @test_throws ArgumentError (
            SDPX.Experimental.validate_la_backend_configuration(incomplete)
        )
    end
end
