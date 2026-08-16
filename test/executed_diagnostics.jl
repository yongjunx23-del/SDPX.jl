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
        # The finalized sparse payload is also the canonical planned LA
        # provider.  CHOLMOD owns the sparse factor/solve, so the executed
        # provider and ownership facts must not inherit the pre-row BLAS plan.
        @test selected.planned_la_backend === :sparse
        @test selected.la_provider === :cholmod
        @test selected.la_ownership === :provider_owned
        @test selected.planned_la_provider === :cholmod
        @test selected.planned_la_ownership === :provider_owned
        @test selected.la_executed_provider === :cholmod
        @test selected.la_executed_ownership === :provider_owned
        @test result.termination.executed.la_provider === :cholmod
        @test result.termination.executed.la_ownership === :provider_owned
        @test selected.parameter_profile === :post_scaling_mehrotra
        @test selected.parameter_source === :post_scaling_mehrotra
        @test selected.parameter_resolution_count == 1
        @test selected.stage === :post_scaling
        @test selected.planned_parameter_profile === :automatic_mehrotra
        @test selected.executed_parameters.adaptive_sigma_max == 0.5
        @test selected.initial_parameters.adaptive_sigma_max == 0.5
        # The automatic phase-2 KKT cold start ran through the sparse backend
        # and used exactly one accepted factor for its two RHS solves.
        initialization = result.termination.executed.initialization
        @test initialization.applied
        @test initialization.path === :phase2_kkt_cold_start
        @test initialization.kkt_formulation === :sparse_normal
        # The cold start factors the unregularized system first (one factor,
        # two RHS solves) and retries with the arithmetic floor only on
        # factor failure.
        @test initialization.factorization_attempts == 1
        @test initialization.rhs_solve_count == 2
        @test initialization.factorization_count == 1
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
        @test selected.parameter_profile === :post_scaling_mehrotra
        @test selected.parameter_resolution_count == 1
        @test selected.stage === :post_scaling
        @test selected.executed_parameters.adaptive_sigma_max == 0.5
        initialization = result.termination.executed.initialization
        @test initialization.applied
        @test initialization.path === :phase2_kkt_cold_start
        @test initialization.kkt_formulation === :positive_definite_cholesky
        @test initialization.rhs_solve_count == 2
        @test initialization.factorization_count == 1

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
        # The automatic phase-2 KKT cold start runs even at `iter_max=0`, so
        # the dense backend genuinely executed.
        @test no_iteration_selected.executed_backend ===
              :positive_definite_cholesky
        @test no_iteration_selected.kkt === :positive_definite_cholesky
        @test no_iteration_selected.backend_resolution ===
              :post_presolve
        @test no_iteration_selected.lp_formulation ===
              :positive_definite_cholesky
        @test no_iteration_selected.gram === :blas_syrk
        @test no_iteration_selected.la_backend !== :not_executed
        @test no_iteration.termination.executed.initialization.applied

        # The fixed-policy path preserves the historical no-execution record
        # at `iter_max=0`.
        fixed_no_iteration = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                iter_max=0,
                parameter_policy=:fixed,
                diagnostics=true,
                verbosity=0,
            ),
        )
        fixed_selected = fixed_no_iteration.diagnostics.selected_algorithms
        @test fixed_selected.executed_backend === :not_executed
        @test fixed_selected.kkt === :not_executed
        @test fixed_selected.backend_resolution === :resolved_no_iteration
        @test fixed_selected.lp_formulation === :positive_definite_cholesky
        @test fixed_selected.gram === :not_executed
        @test fixed_selected.la_backend === :not_executed
        @test !fixed_no_iteration.termination.executed.initialization.applied
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
        @test selected.parameter_source === :post_scaling_mehrotra
        @test selected.parameter_profile === :post_scaling_mehrotra
        @test selected.parameter_resolution_count == 1
        @test selected.stage === :post_scaling
        @test selected.planned_parameter_profile ===
              result.diagnostics.plan.parameter_profile
        @test result.diagnostics.plan.parameter_profile ===
              :automatic_mehrotra
        @test result.termination.executed.parameter_source ===
              :post_scaling_mehrotra
        @test result.termination.executed.parameter_profile ===
              selected.parameter_profile
        @test result.termination.executed.parameter_resolution_count == 1
        @test result.termination.executed.stage === :post_scaling
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

        # The plan stores only the user's request. The numeric resolver runs
        # after Ruiz scaling and therefore sees the scaled block constants,
        # exactly once. `iter_max=0` isolates this ordering contract from
        # convergence behavior.
        scaled_constant_problem = SDPX.ingest(
            [2.0, 3.0],
            [coefficients],
            [[0.0 10_000.0; 10_000.0 0.0]],
            Matrix{Float64}(undef, 2, 0),
            Float64[];
            verbosity=0,
        )
        scaled_options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            scaling=:equilibrate,
            Ωp=0.25,
            Ωd=0.25,
            iter_max=0,
            diagnostics=true,
            verbosity=0,
        )
        scaled_plan = SDPX.build_execution_plan(
            scaled_constant_problem,
            scaled_options,
        )
        equilibrated_problem, _ = SDPX.equilibrate(scaled_constant_problem)
        expected = SDPX.recommended_parameters(
            equilibrated_problem,
            scaled_options,
        )
        scaled_result = SDPX.solve!(scaled_constant_problem, scaled_options)
        scaled_selected = scaled_result.diagnostics.selected_algorithms
        @test scaled_plan.parameter_profile === :automatic_mehrotra
        @test scaled_plan.parameters.omega_p == 0.25
        @test scaled_plan.parameters.omega_d == 0.25
        # Phase 2: the automatic resolver is a compatibility hint only and
        # returns the raw requested Ω unchanged; the KKT cold start decides the
        # actual initial iterate.
        @test expected.Ωp == 0.25
        @test expected.Ωd == 0.25
        @test scaled_selected.executed_parameters.omega_p == expected.Ωp
        @test scaled_selected.executed_parameters.omega_d == expected.Ωd
        @test scaled_selected.parameter_resolution_count == 1
        @test scaled_selected.stage === :post_scaling
        # The cold-start initialization record is carried under
        # `termination.executed.initialization` and reports the identity-metric
        # KKT path: one factorization, two RHS solves, no Ω fallback.
        cold = scaled_result.termination.executed.initialization
        @test cold.ok === true
        @test cold.path === :kkt_cold_start
        @test cold.policy === :auto
        @test cold.factor_count == 1
        @test cold.rhs_solves == 2
        @test cold.kkt_formulation === :dense_normal_equations
        @test cold.factorization === :cholesky
        @test cold.fallback_reason === :none
        @test cold.regularization_attempts == 0
        @test isfinite(cold.normalized_primal_residual)
        @test isfinite(cold.normalized_dual_residual)
        @test cold.primal_margin > 0
        @test cold.dual_margin > 0
        @test cold.kappa_before > 0
        @test cold.kappa_after > 0
        # The identity-metric KKT point for this model is `x = 0`, independent
        # of the requested Ω hints; `iter_max=0` stops on that point.
        @test scaled_result.status == SDPX.IterLimit
        @test scaled_result.x == [0.0, 0.0]

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
        @test fixed_selected.parameter_source === :user_fixed
        @test fixed_selected.parameter_profile === :user_fixed
        @test fixed_selected.parameter_resolution_count == 0
        @test fixed_selected.stage === :not_applicable
        @test fixed.termination.executed.parameter_source === :user_fixed
        @test fixed.termination.executed.parameter_profile === :user_fixed
        @test fixed.termination.executed.parameter_resolution_count == 0
        @test fixed.termination.executed.stage === :not_applicable
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

    # A0 — first-class execution-attempt record. These tests pin the immutable
    # attempt schema, the single-attempt construction rule, truthful
    # planned/executed route facts, fail-closed fallback authorization, and
    # the diagnostics-disabled allocation/absence contract.
    @testset "A0 execution attempt record" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[2, 2, 2] = 1.0
        problem = SDPX.ingest(
            [2.0, 3.0],
            [coefficients],
            [[0.0 1.0; 1.0 0.0]],
            Matrix{Float64}(undef, 2, 0),
            Float64[];
            verbosity=0,
        )
        result = SDPX.solve(
            problem;
            tolerance=1e-8,
            verbosity=0,
            diagnostics=true,
            algorithm=:sdp,
            scaling=:equilibrate,
        )
        @test result.status == SDPX.Optimal
        @test result.diagnostics.attempts isa Tuple
        @test length(result.diagnostics.attempts) == 1
        record = only(result.diagnostics.attempts)
        @test record isa SDPX.ExecutionAttemptRecord
        @test record.attempt_id == 1
        @test record.plan_id == 1

        # Canonical solver family on both sides: planned `:sdp_primal_dual`
        # and executed `:sdp` map to the same `:sdp` family, so an ordinary
        # solve is not a false divergence.
        @test record.planned.family === :sdp
        @test record.executed.family === :sdp
        @test record.planned.formulation === :dense_normal_equations
        @test record.executed.formulation === :dense_normal_equations
        @test record.planned.storage === :dense
        @test record.executed.storage === :dense
        @test record.planned.provider === :blas_lapack
        @test record.executed.provider === :blas_lapack
        @test record.planned.threads == 1
        @test record.executed.threads == 1

        # Precision facts record the arithmetic tag, the mixed-precision mode,
        # and the explicit significand width.
        @test record.planned.precision.arithmetic === :float64
        @test record.executed.precision.arithmetic === :float64
        @test record.planned.precision.mode === :fixed
        @test record.executed.precision.mode === :fixed
        @test record.planned.precision.explicit_bits == 53
        @test record.executed.precision.explicit_bits == 53

        # The A0 plan adds no precision keys, so `planned_parameters` stays
        # byte-for-byte with the pre-A0 plan. The attempt width comes from
        # the arithmetic type (fixed-width) or the exact setprecision scope
        # (BigFloat), never from a request option.
        @test !hasproperty(result.diagnostics.plan.parameters, :precision_bits)
        @test !hasproperty(
            result.diagnostics.plan.parameters,
            :working_precision_policy,
        )
        @test SDPX._attempt_planned_precision_facts(
            SDPX.build_execution_plan(
                problem,
                SDPX.SolverOptions{Float64}(
                    algorithm=:sdp,
                    verbosity=0,
                ),
            ),
            Float64,
        ).explicit_bits == 53

        # No fallback events, no regularization, certified with no downgrade,
        # and A6 reuse facts are explicitly unavailable.
        @test record.fallback_events == ()
        @test record.regularization isa SDPX.RegularizationFacts
        @test record.regularization.count == 0
        @test record.regularization.events == ()
        @test record.certificate.available === true
        @test record.certificate.valid === true
        @test record.certificate.downgrade === false
        @test record.reuse.available === false
        @test record.reuse.reused === false
        @test record.status == SDPX.Optimal
        @test record.termination_reason === :none
    end

    @testset "A0 LP canonical family and planned/executed parity" begin
        # The sparse LP fixture from `executed_diagnostics.jl` exercises the
        # deferred runner whose executed solver is `:lp` while the plan
        # algorithm is `:lp_primal_dual`; the attempt record must not claim a
        # divergence for that.
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
        lp = SDPX.ingest(
            objective,
            coefficients,
            constants,
            zeros(variables, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        result = SDPX.solve(lp; tolerance=1e-9, verbosity=0, diagnostics=true)
        @test result.status == SDPX.Optimal
        record = only(result.diagnostics.attempts)
        @test record.planned.family === :lp
        @test record.executed.family === :lp
        @test record.planned.storage === :sparse
        @test record.executed.storage === :sparse
        @test record.planned.formulation === :not_applicable
        @test record.executed.formulation === :sparse_normal
        @test record.planned.provider === :cholmod
        @test record.executed.provider === :cholmod
        @test record.executed.threads == 1
        @test record.fallback_events == ()
        @test record.certificate.available === true
    end

    @testset "A0 early setup failure is an honest not-executed route" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[2, 2, 2] = 1.0
        problem = SDPX.ingest(
            [2.0, 3.0],
            [coefficients],
            [[0.0 1.0; 1.0 0.0]],
            Matrix{Float64}(undef, 2, 0),
            Float64[];
            verbosity=0,
        )
        plan = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{Float64}(algorithm=:sdp, verbosity=0),
        )
        report = SDPX.PresolveReport(
            0, 0, 0, 0, 0, false, Vector{Int}(), 0.0,
        )
        from_scratch = SDPX.SDPResult{Float64}(
            SDPX.NumericalBreakdown,
            "early setup failure",
            Float64[],
            Matrix{Float64}[],
            Float64[],
            Matrix{Float64}[],
            0.0,
            0.0,
            Inf,
            Inf,
            Inf,
            0,
            0,
            0,
            nothing,
            NamedTuple[],
            nothing,
            (reason=:time_limit, stage=:sdp_setup),
        )
        attached = SDPX._attach_diagnostics(
            from_scratch,
            plan,
            report,
            0.0,
            String[],
            0,
            true,
            (reason=:time_limit, stage=:sdp_setup),
            (available=false,),
            NamedTuple(),
        )
        record = only(attached.diagnostics.attempts)
        @test record.planned.family === :sdp
        @test record.executed.family === :not_executed
        @test record.executed.formulation === :not_executed
        @test record.executed.storage === :not_executed
        @test record.executed.provider === :not_executed
        @test record.executed.precision.mode === :not_executed
        @test record.executed.precision.explicit_bits === nothing
        @test record.executed.threads == 0
        @test record.fallback_events == ()
        @test record.termination_reason === :time_limit
    end

    @testset "A0 mixed-precision mode reflects the plan decision" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[2, 2, 2] = 1.0
        problem = SDPX.ingest(
            BigFloat[2, 3],
            [BigFloat.(coefficients)],
            [BigFloat[0 1; 1 0]],
            Matrix{BigFloat}(undef, 2, 0),
            BigFloat[];
            verbosity=0,
        )
        # `mixed_precision_kkt=:auto` on BigFloat can resolve to `:off`
        # (e.g. fixed refinement policy); the frozen decision must be
        # visible, not the request hint.
        auto_corrected = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{BigFloat}(
                algorithm=:sdp,
                mixed_precision_kkt=:auto,
                precision_bits=256,
                verbosity=0,
            ),
        )
        requested = get(auto_corrected.parameters, :mixed_precision_kkt, :off)
        @test requested === :auto
        @test auto_corrected.backend_config.mixed_precision_mode === :off

        enabled = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{BigFloat}(
                algorithm=:sdp,
                mixed_precision_kkt=:on,
                precision_bits=256,
                verbosity=0,
            ),
        )
        @test enabled.backend_config.mixed_precision_mode === :on
        @test SDPX._attempt_planned_precision_facts(
            enabled,
            BigFloat,
        ).mode === :mixed_precision
        @test SDPX._attempt_planned_precision_facts(
            auto_corrected,
            BigFloat,
        ).mode === :fixed
    end

    @testset "A0 BigFloat attempt precision facts" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[2, 2, 2] = 1.0
        problem = SDPX.ingest(
            BigFloat[2, 3],
            [BigFloat.(coefficients)],
            [BigFloat[0 1; 1 0]],
            Matrix{BigFloat}(undef, 2, 0),
            BigFloat[];
            verbosity=0,
        )
        # Fixed 256-bit request: both planned and executed carry 256.
        fixed = SDPX.solve(
            problem;
            tolerance=1e-16,
            verbosity=0,
            diagnostics=true,
            algorithm=:sdp,
            scaling=:equilibrate,
            precision=256,
            working_precision_policy=:fixed,
        )
        fixed_record = only(fixed.diagnostics.attempts)
        @test fixed_record.planned.precision.arithmetic === :bigfloat
        @test fixed_record.executed.precision.arithmetic === :bigfloat
        @test fixed_record.planned.precision.explicit_bits == 256
        @test fixed_record.executed.precision.explicit_bits == 256

        # Adaptive policy: the attempt runs at the selected rung inside its
        # exact setprecision scope, so both planned and executed facts carry
        # that factual width (here the successful 192-bit lower attempt).
        # A1 will add the requested-vs-selected rung distinction.
        adaptive = SDPX.solve(
            problem;
            tolerance=1e-16,
            verbosity=0,
            diagnostics=true,
            algorithm=:sdp,
            scaling=:equilibrate,
            precision=256,
            working_precision_policy=:auto,
            minimum_working_precision_bits=192,
        )
        adaptive_record = only(adaptive.diagnostics.attempts)
        @test adaptive_record.planned.precision.explicit_bits ==
              adaptive_record.executed.precision.explicit_bits
        @test 192 <= adaptive_record.planned.precision.explicit_bits <= 256
        @test adaptive_record.planned.precision.mode === :fixed
        @test adaptive_record.executed.precision.mode === :fixed

        # Early setup failure on BigFloat must not claim precision bits that
        # never executed.
        plan = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{BigFloat}(
                algorithm=:sdp,
                precision_bits=256,
                verbosity=0,
            ),
        )
        report = SDPX.PresolveReport(
            0, 0, 0, 0, 0, false, Vector{Int}(), 0.0,
        )
        from_scratch = SDPX.SDPResult{BigFloat}(
            SDPX.NumericalBreakdown,
            "early setup failure",
            BigFloat[],
            Matrix{BigFloat}[],
            BigFloat[],
            Matrix{BigFloat}[],
            big"0.0",
            big"0.0",
            big"Inf",
            big"Inf",
            big"Inf",
            0,
            0,
            0,
            nothing,
            NamedTuple[],
            nothing,
            (reason=:time_limit, stage=:sdp_setup),
        )
        attached = SDPX._attach_diagnostics(
            from_scratch,
            plan,
            report,
            0.0,
            String[],
            0,
            true,
            (reason=:time_limit, stage=:sdp_setup),
            (available=false,),
            NamedTuple(),
        )
        early = only(attached.diagnostics.attempts)
        @test early.executed.precision.mode === :not_executed
        @test early.executed.precision.explicit_bits === nothing
    end

    @testset "A0 no record and no construction when diagnostics are disabled" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[2, 2, 2] = 1.0
        problem = SDPX.ingest(
            [2.0, 3.0],
            [coefficients],
            [[0.0 1.0; 1.0 0.0]],
            Matrix{Float64}(undef, 2, 0),
            Float64[];
            verbosity=0,
        )
        disabled = SDPX.solve(
            problem;
            tolerance=1e-8,
            verbosity=0,
            diagnostics=false,
            algorithm=:sdp,
            scaling=:equilibrate,
        )
        @test disabled.diagnostics === nothing

        plan = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{Float64}(algorithm=:sdp, verbosity=0),
        )
        report = SDPX.PresolveReport(
            0, 0, 0, 0, 0, false, Vector{Int}(), 0.0,
        )
        warnings = String[]
        termination = (reason=:none,)
        certificate = (available=false,)
        pipeline_timings = NamedTuple()
        # Warm up method compilation, then the disabled path must be a plain
        # passthrough with zero allocations (no attempt record construction).
        SDPX._attach_diagnostics(
            disabled,
            plan,
            report,
            0.0,
            warnings,
            0,
            false,
            termination,
            certificate,
            pipeline_timings,
        )
        GC.gc()
        allocated = @allocated SDPX._attach_diagnostics(
            disabled,
            plan,
            report,
            0.0,
            warnings,
            0,
            false,
            termination,
            certificate,
            pipeline_timings,
        )
        @test allocated == 0
    end

    @testset "A0 compatibility constructor keeps attempts empty" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[2, 2, 2] = 1.0
        problem = SDPX.ingest(
            [2.0, 3.0],
            [coefficients],
            [[0.0 1.0; 1.0 0.0]],
            Matrix{Float64}(undef, 2, 0),
            Float64[];
            verbosity=0,
        )
        plan = SDPX.build_execution_plan(
            problem,
            SDPX.SolverOptions{Float64}(algorithm=:sdp, verbosity=0),
        )
        report = SDPX.PresolveReport(
            0, 0, 0, 0, 0, false, Vector{Int}(), 0.0,
        )
        diagnostics = SDPX.SolveDiagnostics(
            plan.classification,
            plan,
            report,
            (total=0.0,),
            (workspace_bytes=0,),
            (solver=:sdp,),
            NamedTuple[],
            String[],
            (reason=:none,),
        )
        @test diagnostics.attempts == ()
        @test diagnostics.precision_ladder === nothing
        # A1 appended the optional precision-ladder field to the diagnostics
        # schema; the compatibility constructor leaves it empty.
        @test length(fieldnames(SDPX.SolveDiagnostics)) == 11
        @test fieldnames(SDPX.SolveDiagnostics)[end] == :precision_ladder
    end
end

# A2 — LP final-route freeze: finalized LP diagnostics carry the resolved
# route as a typed `LPRoutePlan` on `ExecutionPlan.payload`.  Red until the
# A2 source lands.
@testset "A2 LP final-route payload (sparse accepted)" begin
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
    lp = SDPX.linear_program(objective, G, h; sparse=true, verbosity=0)

    result = SDPX.solve(lp; tolerance=1e-9, verbosity=0, diagnostics=true)
    payload = result.diagnostics.plan.payload
    @test payload isa SDPX.AbstractExecutionPlanPayload
    @test payload isa SDPX.LPRoutePlan
    @test payload.route in (
        :diagonal_reduced_cholesky,
        :sparse_normal,
        :positive_definite_cholesky,
        :dense_lu,
    )
    @test payload.route === :sparse_normal
    @test payload.storage === :sparse
    @test payload.provider === :cholmod
    @test result.diagnostics.plan.la_config.selected === :sparse
    @test result.diagnostics.plan.la_config.provider === payload.provider
    @test result.diagnostics.plan.la_config.ownership === :provider_owned
    record = only(result.diagnostics.attempts)
    @test payload.route === record.executed.formulation
    @test payload.storage === record.executed.storage
    @test payload.provider ===
          result.termination.sparse_schur_backend.provider
    @test payload.sparse_probe_count == 1
    # The outer pre-row plan stays deferred; the payload is the finalized
    # truth carried by the result/attempt, not an impossible pre-row plan.
    @test record.planned.formulation === :not_applicable
    @test payload.route !== :not_applicable
end
