using LinearAlgebra
using Random
using SDPX
using SparseArrays
using Test

# Diagnostics must report the algorithms that ran, not the ones the plan
# chose before presolve. The LP path selects its sparse Newton system at
# runtime, after the plan is frozen, and the record previously copied the
# plan: a solve that executed sparse Cholesky with a sparse Gram product
# reported `:positive_definite_cholesky` and `:blas_syrk`. Every benchmark
# table built from diagnostics inherited that. (Maintainer review P2.4.)
@testset "diagnostics report executed algorithms" begin
    @testset "sparse LP reports its runtime backend" begin
        rng = MersenneTwister(11)
        variables, base_rows, per_row = 220, 900, 2
        G = spzeros(Float64, base_rows, variables)
        for row in 1:base_rows, column in randperm(rng, variables)[1:per_row]
            G[row, column] = randn(rng)
        end
        G = [G; sparse(1.0I, variables, variables); -sparse(1.0I, variables, variables)]
        rows = size(G, 1)
        interior = randn(rng, variables)
        h = G * interior .- 1.0
        multipliers = rand(rng, rows) .+ 0.5
        objective = vec(transpose(G) * multipliers)
        coefficients = [
            [sparse([1], [1], [G[row, column]], 1, 1) for column in 1:variables]
            for row in 1:rows
        ]
        constants = [reshape([h[row]], 1, 1) for row in 1:rows]
        problem = SDPX.ingest(objective, coefficients, constants,
            zeros(variables, 0), Float64[]; sparse=true, verbosity=0)

        result = SDPX.solve(problem; tolerance=1e-9, verbosity=0, diagnostics=true)
        selected = result.diagnostics.selected_algorithms
        @test result.status == SDPX.Optimal
        # The gate selects the sparse system for this problem; the record must
        # say so, and must not claim a Gram kernel that never assembled.
        @test selected.kkt === :sparse_normal
        @test selected.gram === :sparse_gram
        @test selected.planned_backend === :lp_deferred
        @test selected.executed_backend === :cholmod_sparse_cholesky
        @test selected.fallback_reason === :none
        @test selected.backend_resolution === :post_presolve
        @test selected.lp_formulation === :sparse_normal
        # The plan stays visible under its own name rather than silently
        # replaced, so a plan/executed divergence is observable, not hidden.
        @test selected.planned.kkt !== :sparse_normal
    end

    @testset "dense LP reports the dense backend it used" begin
        variables = 3
        rows = Matrix{Float64}([1.0I(variables); -1.0I(variables)])
        righthand = [-ones(variables); -3ones(variables)]
        blocks = [zeros(variables, 1, 1) for _ in 1:(2variables)]
        for row in 1:(2variables), column in 1:variables
            blocks[row][column, 1, 1] = rows[row, column]
        end
        constants = [reshape([righthand[row]], 1, 1) for row in 1:(2variables)]
        problem = SDPX.ingest(ones(variables), blocks, constants,
            zeros(variables, 0), Float64[]; sparse=false, verbosity=0)

        result = SDPX.solve(problem; tolerance=1e-9, verbosity=0, diagnostics=true)
        selected = result.diagnostics.selected_algorithms
        @test result.status == SDPX.Optimal
        @test selected.kkt === :positive_definite_cholesky
        @test selected.gram === :blas_syrk
        @test selected.planned_backend === :lp_deferred
        @test selected.executed_backend === :positive_definite_cholesky
        @test selected.fallback_reason === :none
        @test selected.backend_resolution === :post_presolve
        @test selected.lp_formulation === :positive_definite_cholesky
        @test selected.planned.kkt === selected.kkt
        @test selected.planned.gram === selected.gram

        no_iteration = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                iter_max=0,
                diagnostics=true,
                verbosity=0,
            ),
        )
        no_iteration_selected = no_iteration.diagnostics.selected_algorithms
        @test no_iteration.status == SDPX.IterLimit
        @test no_iteration_selected.planned_backend === :lp_deferred
        @test no_iteration_selected.executed_backend === :not_executed
        @test no_iteration_selected.kkt === :not_executed
        @test no_iteration_selected.backend_resolution ===
              :resolved_no_iteration
        @test no_iteration_selected.lp_formulation ===
              :positive_definite_cholesky
        @test no_iteration_selected.gram === :not_executed
        @test no_iteration_selected.la_backend === :not_executed
    end

    @testset "SDP core reports its executed KKT backend" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[2, 2, 2] = 1.0
        problem = SDPX.ingest([2.0, 3.0], [coefficients], [[0.0 1.0; 1.0 0.0]],
            Matrix{Float64}(undef, 2, 0), Float64[]; verbosity=0)
        result = SDPX.solve(
            problem;
            tolerance=1e-8,
            verbosity=0,
            diagnostics=true,
            algorithm=:sdp,
            scaling=:equilibrate,
        )
        @test result.status == SDPX.Optimal
        selected = result.diagnostics.selected_algorithms
        @test selected.kkt === :dense_cholesky
        @test selected.planned_backend === :dense_cholesky
        @test selected.executed_backend === :dense_cholesky
        @test selected.planned_la_backend === :standard
        @test selected.la_backend === :standard
        @test selected.la_provider === :blas_lapack
        @test selected.la_ownership === :immutable_scalars
        @test selected.la_executed_provider === :blas_lapack
        @test selected.la_executed_ownership === :immutable_scalars
        @test selected.la_fallback_reason === :none
        @test selected.fallback_reason === :none
        @test selected.backend_resolution === :planned
        @test selected.lp_formulation === :not_applicable
        @test selected.parameter_source === :post_equilibration
        @test selected.parameter_profile === :general_adaptive
        @test selected.planned_parameter_profile ===
              result.diagnostics.plan.parameter_profile
        @test result.termination.executed.parameter_source ===
              :post_equilibration
        @test result.termination.executed.parameter_profile ===
              selected.parameter_profile
        @test !isempty(result.parameter_history)
        # The history contains iteration-local adaptive choices, whereas
        # `executed_parameters` records the initial post-equilibration profile.
        @test selected.executed_parameters ==
              result.termination.executed.executed_parameters
        for name in keys(selected.executed_parameters)
            @test getproperty(selected.initial_parameters, name) ==
                  getproperty(selected.executed_parameters, name)
        end
        @test selected.planned_parameters == result.diagnostics.plan.parameters
        @test selected.effective_threads == selected.threads
        @test selected.fine_grained_block_tasks == 1
        @test selected.fine_grained_block_partition == :lpt
        @test selected.schur_threads == selected.threads
        @test selected.factor_threads === nothing
        @test hasproperty(
            result.diagnostics.timings,
            :schur_assembly,
        )

        untimed = SDPX.solve(
            problem;
            tolerance=1e-8,
            verbosity=0,
            diagnostics=true,
            timing=false,
        )
        @test untimed.timings === nothing
        @test !hasproperty(
            untimed.diagnostics.timings,
            :schur_assembly,
        )

        fixed = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                scaling=:none,
                parameter_policy=:fixed,
                parameter_strategy=:fixed,
                β=0.23,
                γ=0.73,
                Ωp=3.0,
                Ωd=4.0,
                predictor=:sdpb,
                adaptive_sigma_max=0.41,
                iter_max=0,
                diagnostics=true,
                verbosity=0,
            ),
        )
        @test fixed.status == SDPX.IterLimit
        fixed_selected = fixed.diagnostics.selected_algorithms
        @test fixed_selected.parameter_source === :options
        @test fixed_selected.parameter_profile === :fixed
        @test fixed.termination.executed.parameter_source === :options
        @test fixed.termination.executed.parameter_profile === :fixed
        @test fixed_selected.executed_parameters == (
            beta=0.23,
            gamma=0.73,
            omega_p=3.0,
            omega_d=4.0,
            predictor=:sdpb,
            strategy=:fixed,
            adaptive_sigma_max=0.41,
        )
        @test fixed_selected.initial_parameters.beta == 0.23
        @test fixed_selected.initial_parameters.gamma == 0.73
        @test fixed_selected.planned_parameters ==
              fixed.diagnostics.plan.parameters
    end
end
