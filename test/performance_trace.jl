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
        @test propertynames(trace.setup)[1:4] == (
            :frontend_seconds,
            :presolve_seconds,
            :equality_presolve_seconds,
            :structural_analysis_seconds,
        )
        @test :certification_seconds in propertynames(trace.final)
        @test :numeric_factorizations in propertynames(trace.counters)
        @test :symbolic_analyses in propertynames(trace.counters)
        @test :factor_memory_estimate_bytes in propertynames(trace.counters)

        # Timing and routing were requested, so they are present, not markers.
        @test SDPX.isavailable(trace.final.total_seconds)
        @test trace.final.total_seconds >= 0.0
        @test SDPX.isavailable(trace.setup.solver)
        @test SDPX.isavailable(trace.setup.executed_backend)
        @test SDPX.isavailable(trace.final.certificate_available)
        @test trace.final.certificate_available isa Bool
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
        @test isempty(trace.final.warnings)

        marker = trace.final.total_seconds
        @test marker === SDPX.unavailable
        @test string(marker) == "unavailable"
    end

    @testset "native SOC result projects its lifted solve" begin
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
    end

    @testset "experimental namespace mirrors the surface" begin
        @test SDPX.Experimental.PerformanceTrace === SDPX.PerformanceTrace
        @test SDPX.Experimental.performance_trace === SDPX.performance_trace
        @test SDPX.Experimental.unavailable === SDPX.unavailable
        @test SDPX.Experimental.isavailable === SDPX.isavailable
    end
end
