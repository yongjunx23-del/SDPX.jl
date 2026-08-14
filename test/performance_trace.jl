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

        # Terminal quality is lifted directly from the solve record.
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
    end

    @testset "native SOC result exposes Lorentz phases" begin
        soc = SDPX.second_order_program(
            [1.0, 0.0, 0.0],
            Matrix{Float64}(I, 3, 3),
            zeros(3);
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        result = SDPX.solve_socp(soc; verbosity=0, diagnostics=true)
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
        else
            # A lifted/reference SOC solve has no separately measured cone
            # metric phase; do not relabel its block factor time.
            @test !SDPX.isavailable(
                trace.iteration.cone_scaling_metric_seconds,
            )
        end
    end

    @testset "experimental namespace mirrors the surface" begin
        @test SDPX.Experimental.PerformanceTrace === SDPX.PerformanceTrace
        @test SDPX.Experimental.performance_trace === SDPX.performance_trace
        @test SDPX.Experimental.unavailable === SDPX.unavailable
        @test SDPX.Experimental.isavailable === SDPX.isavailable
    end
end
