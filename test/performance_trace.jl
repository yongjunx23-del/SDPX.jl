using LinearAlgebra
using SDPX
using Test

@testset "performance trace" begin
    @testset "projection from a recorded SDP solve" begin
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
            diagnostics=true,
            timing=true,
            verbosity=0,
        )
        @test result.status == SDPX.Optimal

        trace = SDPX.performance_trace(result)
        @test trace isa SDPX.PerformanceTrace
        @test propertynames(trace) == (:setup, :iteration, :final, :counters)

        # Terminal quality is projected directly from the solve record.
        @test trace.final.status === :Optimal
        @test trace.final.termination_reason isa Symbol
        @test isapprox(trace.final.primal_objective, 2 * sqrt(6); atol=1e-7)
        @test isapprox(trace.final.dual_objective, 2 * sqrt(6); atol=1e-7)
        @test trace.final.relative_gap <= 1e-8
        @test trace.final.primal_residual <= 1e-8
        @test trace.final.dual_residual <= 1e-8
        @test trace.counters.iterations > 0
        @test trace.counters.restarts >= 0
        @test trace.counters.regularizations >= 0
        @test trace.counters.parameter_history_length >= 0
        @test propertynames(trace.setup)[1:5] == (
            :frontend_seconds,
            :preprocess_seconds,
            :presolve_seconds,
            :equality_presolve_seconds,
            :structural_analysis_seconds,
        )
        @test :certification_seconds in propertynames(trace.final)
        @test :numeric_factorizations in propertynames(trace.counters)
        @test :symbolic_analyses in propertynames(trace.counters)
        @test :factor_nnz in propertynames(trace.counters)
        @test :factor_memory_estimate_bytes in propertynames(trace.counters)
        @test :la_factorization in propertynames(trace.setup)
        @test :kkt_schur_copy_seconds in propertynames(trace.iteration)
        @test :certificate_valid in propertynames(trace.final)
        @test :certificate_kind in propertynames(trace.final)
        @test :certificate_failures in propertynames(trace.final)
        @test :schur_assembly_count in propertynames(trace.counters)
        @test :sparse_factor_failures in propertynames(trace.counters)
        @test :symbolic_analysis_reuses in propertynames(trace.counters)
        @test :fixed_residual_blocks in propertynames(trace.counters)
        @test :fixed_rhs_contractions in propertynames(trace.counters)
        @test :fixed_direction_recoveries in propertynames(trace.counters)

        # Timing and routing were requested, so they are present, not markers.
        @test SDPX.isavailable(trace.final.total_seconds)
        @test trace.final.total_seconds >= 0.0
        @test SDPX.isavailable(trace.setup.solver)
        @test SDPX.isavailable(trace.setup.executed_backend)
        @test SDPX.isavailable(trace.final.certificate_available)
        @test trace.final.certificate_available isa Bool
        @test SDPX.isavailable(trace.iteration.accepted_update_seconds)
        @test trace.iteration.accepted_update_seconds >= 0.0
        @test SDPX.isavailable(trace.iteration.direction_recovery_seconds)
        @test trace.iteration.direction_recovery_seconds >= 0.0
        @test SDPX.isavailable(trace.setup.preprocess_seconds)
        @test trace.setup.preprocess_seconds >= 0.0
        @test SDPX.isavailable(trace.setup.frontend_seconds)
        @test trace.setup.frontend_seconds >= 0.0
        @test SDPX.isavailable(trace.setup.equality_presolve_seconds)
        @test trace.setup.equality_presolve_seconds >= 0.0
        @test SDPX.isavailable(trace.setup.structural_analysis_seconds)
        @test trace.setup.structural_analysis_seconds >= 0.0
        @test SDPX.isavailable(trace.setup.execution_planning_seconds)
        @test trace.setup.execution_planning_seconds >= 0.0
        @test SDPX.isavailable(trace.final.certification_seconds)
        @test trace.final.certification_seconds >= 0.0

        # Newly exposed routing/worker facts are lifted from the retained
        # execution record. Optional worker fields may record `nothing`
        # ("not applicable") and still count as available: the diagnostic
        # truthfully says the phase did not run, which is not a missing
        # measurement.
        @test SDPX.isavailable(trace.setup.la_factorization)
        @test trace.setup.la_factorization isa Symbol
        @test SDPX.isavailable(trace.setup.effective_threads)
        @test trace.setup.effective_threads >= 1
        @test SDPX.isavailable(trace.setup.schur_threads)
        @test trace.setup.schur_threads >= 1
        @test SDPX.isavailable(trace.setup.factor_threads)
        @test SDPX.isavailable(trace.setup.lp_pack_threads)
        @test SDPX.isavailable(trace.setup.arrow_linear_solve)
        @test SDPX.isavailable(trace.setup.fine_grained_block_tasks)
        @test trace.setup.fine_grained_block_tasks >= 1
        @test SDPX.isavailable(trace.setup.fine_grained_block_partition)
        @test trace.setup.fine_grained_block_partition isa Symbol
        @test SDPX.isavailable(trace.setup.parameter_resolution_count)
        @test trace.setup.parameter_resolution_count >= 1
        @test SDPX.isavailable(trace.setup.parameter_resolution_stage)
        @test trace.setup.parameter_resolution_stage isa Symbol

        # SDP KKT subphase and post-factor timings are recorded by the core
        # when timing is enabled; a measured zero-second phase is available.
        for name in (
            :kkt_schur_copy_seconds,
            :kkt_constraint_triangular_solve_seconds,
            :kkt_equality_factorization_seconds,
            :kkt_other_seconds,
            :predictor_direction_recovery_seconds,
            :corrector_direction_recovery_seconds,
            :best_iterate_seconds,
            :objective_and_targets_seconds,
            :other_seconds,
        )
            @test SDPX.isavailable(getproperty(trace.iteration, name))
            @test getproperty(trace.iteration, name) >= 0.0
        end

        # Certificate summary facts mirror the recorded original-coordinate
        # gate, including its kind and failure list.
        @test SDPX.isavailable(trace.final.certificate_valid)
        @test trace.final.certificate_valid isa Bool
        @test SDPX.isavailable(trace.final.certificate_kind)
        @test trace.final.certificate_kind === :optimality
        @test SDPX.isavailable(trace.final.certificate_failures)
        @test trace.final.certificate_failures isa Vector
        @test isempty(trace.final.certificate_failures)

        # Fixed-block counters are NativeSOC-only records; an SDP core solve
        # never produces them, so they must stay explicitly unavailable.
        @test !SDPX.isavailable(trace.counters.fixed_residual_blocks)
        @test !SDPX.isavailable(trace.counters.fixed_rhs_contractions)
        @test !SDPX.isavailable(trace.counters.fixed_direction_recoveries)
    end

    @testset "missing sources become unavailable, not defaults" begin
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
            diagnostics=false,
            timing=false,
            verbosity=0,
        )
        trace = SDPX.performance_trace(result)

        # Terminal status/objective/counters never depend on diagnostics.
        @test trace.final.status === :Optimal
        @test isapprox(trace.final.primal_objective, 2 * sqrt(6); atol=1e-7)
        @test trace.counters.iterations > 0

        # Timing and routing sources were disabled and are honestly missing.
        @test !SDPX.isavailable(trace.final.total_seconds)
        @test !SDPX.isavailable(trace.setup.solver)
        @test !SDPX.isavailable(trace.setup.executed_backend)
        @test !SDPX.isavailable(trace.final.certificate_available)
        @test !SDPX.isavailable(trace.final.workspace_bytes)
        @test !SDPX.isavailable(trace.final.process_peak_rss_bytes)
        @test !SDPX.isavailable(trace.final.certification_seconds)
        @test !SDPX.isavailable(trace.setup.preprocess_seconds)
        @test !SDPX.isavailable(trace.setup.equality_presolve_seconds)
        @test !SDPX.isavailable(trace.setup.structural_analysis_seconds)
        @test !SDPX.isavailable(trace.setup.execution_planning_seconds)
        @test !SDPX.isavailable(trace.setup.frontend_seconds)
        @test !SDPX.isavailable(trace.iteration.accepted_update_seconds)
        @test !SDPX.isavailable(trace.iteration.direction_recovery_seconds)
        @test isempty(trace.final.warnings)

        # Every newly exposed routing/worker field is missing too.
        for name in (
            :la_factorization, :effective_threads, :schur_threads,
            :factor_threads, :lp_pack_threads, :arrow_linear_solve,
            :fine_grained_block_tasks, :fine_grained_block_partition,
            :parameter_resolution_count, :parameter_resolution_stage,
        )
            @test !SDPX.isavailable(getproperty(trace.setup, name))
        end

        # Newly exposed phase timings are missing when timing was disabled.
        for name in (
            :kkt_schur_copy_seconds,
            :kkt_constraint_triangular_solve_seconds,
            :kkt_equality_factorization_seconds,
            :kkt_other_seconds,
            :predictor_direction_recovery_seconds,
            :corrector_direction_recovery_seconds,
            :best_iterate_seconds,
            :objective_and_targets_seconds,
            :other_seconds,
        )
            @test !SDPX.isavailable(getproperty(trace.iteration, name))
        end

        # Certificate detail fields follow the same marker contract.
        @test !SDPX.isavailable(trace.final.certificate_valid)
        @test !SDPX.isavailable(trace.final.certificate_kind)
        @test !SDPX.isavailable(trace.final.certificate_failures)

        # Sparse/fixed counters have no recorded source on this small dense
        # SDP solve; none may be defaulted to zero.
        for name in (
            :schur_assembly_count, :sparse_factor_failures,
            :symbolic_analysis_reuses, :fixed_residual_blocks,
            :fixed_rhs_contractions, :fixed_direction_recoveries,
        )
            @test !SDPX.isavailable(getproperty(trace.counters, name))
        end

        marker = trace.final.total_seconds
        @test marker === SDPX.unavailable
        @test string(marker) == "unavailable"
    end

    @testset "sparse structural and factor nonzeros stay distinct" begin
        coefficients = zeros(1, 1, 1)
        coefficients[1, 1, 1] = 1.0
        problem = SDPX.ingest(
            [1.0],
            [coefficients],
            [zeros(1, 1)],
            Matrix{Float64}(undef, 1, 0),
            Float64[];
            verbosity=0,
        )
        result = SDPX.solve(
            problem;
            diagnostics=false,
            timing=false,
            verbosity=0,
        )
        sparse_statistics = (
            analyses=1,
            factorizations=2,
            symbolic_reuse_ratio=0.5,
            schur_nnz=7,
            factor_nonzeros=11,
            assembly_count=3,
            failures=0,
            reused=1,
        )
        projected = SDPX.SDPResult{Float64}(
            result.status,
            result.message,
            result.x,
            result.X,
            result.y,
            result.Y,
            result.pObj,
            result.dObj,
            result.gap_rel,
            result.p_res,
            result.d_res,
            result.iterations,
            result.restarts,
            result.regularizations,
            result.timings,
            result.parameter_history,
            result.diagnostics,
            merge(
                result.termination,
                (sparse_schur_backend=sparse_statistics,),
            ),
        )
        trace = SDPX.performance_trace(projected)
        @test trace.counters.symbolic_analyses == 1
        @test trace.counters.numeric_factorizations == 2
        @test trace.counters.symbolic_analysis_reuse == 0.5
        @test trace.counters.schur_nnz == 7
        @test trace.counters.factor_nnz == 11
        @test trace.counters.schur_assembly_count == 3
        @test trace.counters.sparse_factor_failures == 0
        @test trace.counters.symbolic_analysis_reuses == 1
    end

    @testset "native SOC result exposes Lorentz phases" begin
        soc = SDPX.second_order_program(
            [1.0, 0.0, 0.0],
            Matrix{Float64}(I, 3, 3),
            zeros(3);
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        # This unit-data fixture converges to the boundary optimum (pObj = 5.0),
        # where the final status is sensitive to platform-level rounding among
        # the gap, residual, and strict-interior checks. Explicitly request
        # 1e-8, matching the projection testset above, so these assertions gate
        # the trace projection rather than a tighter numerical acceptance test.
        result = SDPX.solve_socp(
            soc; verbosity=0, diagnostics=true,
            ϵ_gap=1e-8, ϵ_primal=1e-8, ϵ_dual=1e-8,
        )
        @test result.status == SDPX.Optimal

        trace = SDPX.performance_trace(result)
        @test trace isa SDPX.PerformanceTrace
        @test trace.final.status === :Optimal
        @test isapprox(trace.final.primal_objective, 5.0; atol=1e-7)
        @test trace.counters.iterations > 0
        @test SDPX.isavailable(trace.setup.frontend_seconds)
        @test trace.setup.frontend_seconds >= 0.0
        if trace.setup.solver === :native_soc
            @test SDPX.isavailable(
                trace.iteration.cone_scaling_metric_seconds,
            )
            @test trace.iteration.cone_scaling_metric_seconds >= 0.0
            # Fixed-block counters are part of the NativeSOC termination
            # record and must be lifted, never defaulted.
            @test SDPX.isavailable(trace.counters.fixed_residual_blocks)
            @test trace.counters.fixed_residual_blocks >= 0
            @test SDPX.isavailable(trace.counters.fixed_rhs_contractions)
            @test trace.counters.fixed_rhs_contractions >= 0
            @test SDPX.isavailable(trace.counters.fixed_direction_recoveries)
            @test trace.counters.fixed_direction_recoveries >= 0
            # SDP-core KKT subphases are not relabeled onto the native path.
            @test !SDPX.isavailable(
                trace.iteration.kkt_schur_copy_seconds,
            )
            @test !SDPX.isavailable(
                trace.iteration.kkt_constraint_triangular_solve_seconds,
            )
            @test !SDPX.isavailable(
                trace.iteration.kkt_equality_factorization_seconds,
            )
            @test !SDPX.isavailable(trace.iteration.kkt_other_seconds)
        else
            # A reference/SDP-backed SOC solve has no separately measured
            # cone metric phase; do not relabel its block factor time.
            @test !SDPX.isavailable(
                trace.iteration.cone_scaling_metric_seconds,
            )
            @test !SDPX.isavailable(trace.counters.fixed_residual_blocks)
            @test !SDPX.isavailable(trace.counters.fixed_rhs_contractions)
            @test !SDPX.isavailable(trace.counters.fixed_direction_recoveries)
        end
    end

    @testset "new fields project when present and stay unavailable otherwise" begin
        rich_timings = (
            total=1.0,
            kkt_schur_copy=0.13,
            kkt_constraint_triangular_solve=0.15,
            kkt_equality_factorization=0.17,
            kkt_other=0.18,
            predictor_direction_recovery=0.21,
            corrector_direction_recovery=0.26,
            best_iterate=0.29,
            objective_and_targets=0.30,
            other=0.32,
        )
        rich_selected = (
            la_factorization=:pivoted_symmetric_ldlt,
            effective_threads=4,
            schur_threads=4,
            factor_threads=4,
            lp_pack_threads=nothing,
            arrow_linear_solve=nothing,
            fine_grained_block_tasks=4,
            fine_grained_block_partition=:lpt,
            parameter_resolution_count=1,
            stage=:post_scaling,
            certificate=(
                available=true,
                valid=true,
                kind=:optimality,
                failures=Symbol[],
                reason=:certified,
            ),
        )
        rich_diagnostics = (
            timings=rich_timings,
            memory=(
                workspace_bytes=1,
                process_peak_rss_bytes=2,
                memory_budget_bytes=0,
            ),
            selected_algorithms=rich_selected,
            warnings=String[],
            termination=(
                reason=:converged,
                stage=:native_soc,
                fixed_residual_blocks=2,
                fixed_rhs_contractions=1,
                fixed_direction_recoveries=0,
            ),
        )
        rich = SDPX.ConicResult{Float64}(
            SDPX.Optimal,
            "ok",
            [1.0],
            [[1.0]],
            [[1.0]],
            [0.0],
            5.0,
            5.0,
            0.0,
            0.0,
            0.0,
            8,
            rich_diagnostics,
        )
        trace = SDPX.performance_trace(rich)

        # Routing/worker facts, exact values.
        @test trace.setup.la_factorization === :pivoted_symmetric_ldlt
        @test trace.setup.effective_threads == 4
        @test trace.setup.schur_threads == 4
        @test trace.setup.factor_threads == 4
        @test SDPX.isavailable(trace.setup.lp_pack_threads)
        @test trace.setup.lp_pack_threads === nothing
        @test SDPX.isavailable(trace.setup.arrow_linear_solve)
        @test trace.setup.arrow_linear_solve === nothing
        @test trace.setup.fine_grained_block_tasks == 4
        @test trace.setup.fine_grained_block_partition === :lpt
        @test trace.setup.parameter_resolution_count == 1
        @test trace.setup.parameter_resolution_stage === :post_scaling

        # Phase timings, exact values.
        @test trace.iteration.kkt_schur_copy_seconds == 0.13
        @test trace.iteration.kkt_constraint_triangular_solve_seconds == 0.15
        @test trace.iteration.kkt_equality_factorization_seconds == 0.17
        @test trace.iteration.kkt_other_seconds == 0.18
        @test trace.iteration.predictor_direction_recovery_seconds == 0.21
        @test trace.iteration.corrector_direction_recovery_seconds == 0.26
        @test trace.iteration.best_iterate_seconds == 0.29
        @test trace.iteration.objective_and_targets_seconds == 0.30
        @test trace.iteration.other_seconds == 0.32

        # Certificate detail facts are lifted verbatim.
        @test trace.final.certificate_valid === true
        @test trace.final.certificate_kind === :optimality
        @test trace.final.certificate_failures == Symbol[]

        # NativeSOC fixed-block counters are present.
        @test trace.counters.fixed_residual_blocks == 2
        @test trace.counters.fixed_rhs_contractions == 1
        @test trace.counters.fixed_direction_recoveries == 0

        # Sparse-backend counters are not fabricated on the Conic path.
        @test !SDPX.isavailable(trace.counters.schur_assembly_count)
        @test !SDPX.isavailable(trace.counters.sparse_factor_failures)
        @test !SDPX.isavailable(trace.counters.symbolic_analysis_reuses)

        # Minimal diagnostics: every new field is explicitly unavailable.
        poor_diagnostics = (
            timings=NamedTuple(),
            memory=(
                workspace_bytes=1,
                process_peak_rss_bytes=2,
                memory_budget_bytes=0,
            ),
            selected_algorithms=(certificate=(available=false,),),
            warnings=String[],
            termination=(reason=:none,),
        )
        poor = SDPX.ConicResult{Float64}(
            SDPX.Optimal,
            "ok",
            [1.0],
            [[1.0]],
            [[1.0]],
            [0.0],
            5.0,
            5.0,
            0.0,
            0.0,
            0.0,
            8,
            poor_diagnostics,
        )
        trace_poor = SDPX.performance_trace(poor)
        for name in (
            :la_factorization, :effective_threads, :schur_threads,
            :factor_threads, :lp_pack_threads, :arrow_linear_solve,
            :fine_grained_block_tasks, :fine_grained_block_partition,
            :parameter_resolution_count, :parameter_resolution_stage,
        )
            @test !SDPX.isavailable(getproperty(trace_poor.setup, name))
        end
        for name in (
            :kkt_schur_copy_seconds,
            :kkt_constraint_triangular_solve_seconds,
            :kkt_equality_factorization_seconds,
            :kkt_other_seconds,
            :predictor_direction_recovery_seconds,
            :corrector_direction_recovery_seconds,
            :best_iterate_seconds,
            :objective_and_targets_seconds,
            :other_seconds,
        )
            @test !SDPX.isavailable(getproperty(trace_poor.iteration, name))
        end
        @test !SDPX.isavailable(trace_poor.final.certificate_valid)
        @test !SDPX.isavailable(trace_poor.final.certificate_kind)
        @test !SDPX.isavailable(trace_poor.final.certificate_failures)
        for name in (
            :schur_assembly_count, :sparse_factor_failures,
            :symbolic_analysis_reuses, :fixed_residual_blocks,
            :fixed_rhs_contractions, :fixed_direction_recoveries,
        )
            @test !SDPX.isavailable(getproperty(trace_poor.counters, name))
        end
    end

    @testset "experimental namespace mirrors the surface" begin
        @test SDPX.Experimental.PerformanceTrace === SDPX.PerformanceTrace
        @test SDPX.Experimental.performance_trace === SDPX.performance_trace
        @test SDPX.Experimental.unavailable === SDPX.unavailable
        @test SDPX.Experimental.isavailable === SDPX.isavailable
    end
end
