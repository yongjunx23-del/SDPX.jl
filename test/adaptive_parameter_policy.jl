using LinearAlgebra
using MultiFloats: Float64x4
using SDPX
using Test

function policy_toy_sdp(::Type{T}) where {T}
    coefficients = zeros(T, 2, 2, 2)
    coefficients[1, 1, 1] = one(T)
    coefficients[2, 2, 2] = one(T)
    return SDPX.ingest(
        T[2, 3],
        [coefficients],
        [T[0 1; 1 0]],
        zeros(T, 2, 0),
        T[];
        verbosity=0,
    )
end

@testset "interior-point parameter policies" begin
    @testset "fixed policy reproduces configured values" begin
        options = SDPX.SolverOptions{Float64}(
            β=0.075,
            γ=0.8,
            refine_tol=1e-13,
            refine_max_steps=5,
            parameter_strategy=:fixed,
            verbosity=0,
        )
        policy = SDPX.FixedParameterPolicy(options)
        diagnostics = SDPX.IterationDiagnostics{Float64}()
        selected = SDPX.select_parameters(policy, diagnostics, NamedTuple[])
        @test selected.sigma == options.β
        @test selected.primal_fraction_to_boundary == options.γ
        @test selected.dual_fraction_to_boundary == options.γ
        @test selected.backtracking_factor == options.γ
        @test selected.refinement_tolerance == options.refine_tol
        @test selected.refinement_max_count == options.refine_max_steps
        @test !selected.fallback
    end

    @testset "adaptive Mehrotra selection is typed and bounded" begin
        for T in (Float64, Float64x4)
            options = SDPX.SolverOptions{T}(
                parameter_strategy=:adaptive,
                verbosity=0,
            )
            policy = SDPX.AdaptiveParameterPolicy(options)
            diagnostics = SDPX.IterationDiagnostics{T}(
                iteration=2,
                primal_residual=T(1) / T(1_000),
                dual_residual=T(1) / T(2_000),
                relative_gap=T(1) / T(100),
                mu=one(T),
                mu_aff=T(1) / T(2),
                affine_primal_step=T(9) / T(10),
                affine_dual_step=T(4) / T(5),
            )
            selected =
                SDPX.select_parameters(policy, diagnostics, NamedTuple[])
            @test selected.sigma isa T
            @test T(1) / T(50) <= selected.sigma <= T(1) / T(2)
            @test T(4) / T(5) <=
                  selected.primal_fraction_to_boundary <=
                  T(99) / T(100)
            @test T(4) / T(5) <=
                  selected.dual_fraction_to_boundary <=
                  T(99) / T(100)
            @test !selected.fallback
        end
    end

    @testset "short affine steps request more centering" begin
        options = SDPX.SolverOptions{Float64}(
            parameter_strategy=:adaptive,
            verbosity=0,
        )
        policy = SDPX.AdaptiveParameterPolicy(options)
        common = (
            iteration=3,
            primal_residual=1e-3,
            dual_residual=1e-3,
            relative_gap=1e-2,
            mu=1.0,
            mu_aff=1e-3,
        )
        long_step = SDPX.select_parameters(
            policy,
            SDPX.IterationDiagnostics{Float64}(;
                common...,
                affine_primal_step=0.95,
                affine_dual_step=0.95,
            ),
            NamedTuple[],
        )
        short_step = SDPX.select_parameters(
            policy,
            SDPX.IterationDiagnostics{Float64}(;
                common...,
                affine_primal_step=0.01,
                affine_dual_step=0.02,
            ),
            NamedTuple[],
        )
        @test short_step.sigma > long_step.sigma
        @test short_step.primal_fraction_to_boundary <
              long_step.primal_fraction_to_boundary
    end

    @testset "structural sigma cap prevents over-centering" begin
        options = SDPX.SolverOptions{Float64}(
            β=0.075,
            γ=0.8,
            adaptive_sigma_max=0.15,
            parameter_strategy=:adaptive,
            verbosity=0,
        )
        policy = SDPX.AdaptiveParameterPolicy(options)
        diagnostics = SDPX.IterationDiagnostics{Float64}(
            iteration=6,
            primal_residual=0.70,
            dual_residual=0.006,
            relative_gap=0.52,
            mu=0.00152,
            mu_aff=0.00142,
            affine_primal_step=0.019,
            affine_dual_step=0.135,
            previous_primal_step=0.512,
            previous_dual_step=0.640,
            backtracking_count=5,
        )
        history = [(sigma=0.272, unstable=false)]
        selected = SDPX.select_parameters(policy, diagnostics, history)
        @test selected.sigma == 0.15
        @test policy.sigma_max == 0.15

        @test SDPX.recommended_adaptive_sigma_max(
            :large_lattice_dense_schur,
            0.075,
            0.0,
        ) == 0.2
        @test SDPX.recommended_adaptive_sigma_max(
            :general_adaptive,
            0.1,
            0.0,
        ) == 0.5
        @test SDPX.recommended_adaptive_sigma_max(
            :large_lattice_dense_schur,
            0.075,
            0.2,
        ) == 0.2
        @test SDPX.AdaptiveParameterPolicy(
            SDPX.SolverOptions{Float64}(
                β=0.2,
                adaptive_sigma_max=0.1,
                verbosity=0,
            ),
        ).sigma_max == 0.2
        @test_throws ArgumentError SDPX._validate_solver_options(
            SDPX.SolverOptions{Float64}(
                adaptive_sigma_max=-0.1,
                verbosity=0,
            ),
        )
        plan = SDPX.build_execution_plan(
            policy_toy_sdp(Float64),
            SDPX.SolverOptions{Float64}(
                adaptive_sigma_max=0.2,
                verbosity=0,
            ),
        )
        @test plan.parameters.adaptive_sigma_max == 0.2
        @test SDPX.automatic_scaling_policy(
            :sdp_primal_dual,
            :large_lattice_dense_schur,
            :fixed,
        ) == :none
        @test SDPX.automatic_scaling_policy(
            :sdp_primal_dual,
            :large_lattice_dense_schur,
            :adaptive,
        ) == :sdp_ruiz
        @test SDPX.automatic_scaling_policy(
            :sdp_primal_dual,
            :general_adaptive,
            :fixed,
        ) == :sdp_ruiz
        @test SDPX.automatic_scaling_policy(
            :lp_primal_dual,
            :large_lattice_dense_schur,
            :fixed,
        ) == :lp_geometric
    end

    @testset "non-finite diagnostics fall back explicitly" begin
        options = SDPX.SolverOptions{Float64}(
            β=0.075,
            γ=0.8,
            parameter_strategy=:adaptive,
            verbosity=0,
        )
        selected = SDPX.select_parameters(
            SDPX.AdaptiveParameterPolicy(options),
            SDPX.IterationDiagnostics{Float64}(
                iteration=4,
                primal_residual=NaN,
                dual_residual=1.0,
                relative_gap=1.0,
                mu=1.0,
                mu_aff=0.5,
                affine_primal_step=0.5,
                affine_dual_step=0.5,
            ),
            NamedTuple[],
        )
        @test selected.fallback
        @test selected.fallback_reason == :nonfinite_diagnostics
        @test selected.sigma == options.β
        @test selected.backtracking_factor == options.γ
    end

    @testset "degraded factorization restores the complete fixed policy" begin
        options = SDPX.SolverOptions{Float64}(
            β=0.075,
            γ=0.8,
            parameter_strategy=:adaptive,
            verbosity=0,
        )
        controller = SDPX.AdaptiveIPMController(options)
        selected = SDPX.select_iteration_parameters!(
            controller,
            SDPX.IterationDiagnostics{Float64}(
                iteration=5,
                primal_residual=1e-5,
                dual_residual=1e-5,
                relative_gap=1e-4,
                mu=1e-3,
                mu_aff=1e-4,
                affine_primal_step=0.8,
                affine_dual_step=0.8,
                factorization_quality=0.0,
            ),
        )
        @test selected.fallback
        @test selected.fallback_reason == :degraded_factorization
        @test controller.fallback
        restored = SDPX.controller_options(options, controller)
        @test restored.parameter_strategy == :fixed
        @test restored.β == options.β
        @test restored.γ == options.γ
    end

    @testset "BigFloat selection stays at working precision" begin
        setprecision(BigFloat, 256) do
            options = SDPX.SolverOptions{BigFloat}(
                parameter_strategy=:adaptive,
                precision_bits=256,
                verbosity=0,
            )
            diagnostics = SDPX.IterationDiagnostics{BigFloat}(
                iteration=2,
                primal_residual=BigFloat(1) / BigFloat(1_000),
                dual_residual=BigFloat(1) / BigFloat(1_000),
                relative_gap=BigFloat(1) / BigFloat(100),
                mu=one(BigFloat),
                mu_aff=BigFloat(1) / BigFloat(2),
                affine_primal_step=BigFloat(9) / BigFloat(10),
                affine_dual_step=BigFloat(4) / BigFloat(5),
            )
            selected = SDPX.select_parameters(
                SDPX.AdaptiveParameterPolicy(options),
                diagnostics,
                NamedTuple[],
            )
            @test precision(selected.sigma) == 256
            @test selected.sigma == BigFloat(3) / BigFloat(25)
            @test precision(selected.refinement_tolerance) == 256
        end
    end

    @testset "adaptive SDP records canonical affine diagnostics" begin
        problem = policy_toy_sdp(Float64)
        result = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                parameter_policy=:fixed,
                parameter_strategy=:adaptive,
                ϵ_gap=1e-9,
                ϵ_primal=1e-9,
                ϵ_dual=1e-9,
                verbosity=0,
            ),
        )
        @test result.status == SDPX.Optimal
        @test result.pObj ≈ 2sqrt(6.0) atol=1e-8
        @test !isempty(result.parameter_history)
        @test all(
            row -> (
                isfinite(row.mu) &&
                isfinite(row.mu_aff) &&
                0 <= row.affine_primal_step <= 1 &&
                0 <= row.affine_dual_step <= 1 &&
                row.primal_psd_margin >= 0 &&
                row.dual_psd_margin >= 0 &&
                row.selected_refinement_max_count >= 0 &&
                row.selected_refinement_tolerance >= 0
            ),
            result.parameter_history,
        )
    end
end
