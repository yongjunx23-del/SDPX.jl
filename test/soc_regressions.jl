using LinearAlgebra
using SparseArrays

@testset "SOC compatibility regressions" begin
    @testset "compact frontend stays in Lorentz coordinates" begin
        problem = second_order_program(
            [1.0, 0.0, 0.0],
            Matrix{Float64}(I, 3, 3),
            zeros(3);
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        result = solve_socp(
            problem;
            tolerance=1e-8,
            maximum_iterations=120,
            verbosity=0,
        )
        @test result.status == SDPX.Optimal
        @test result.pObj ≈ 5.0 atol=1e-7
        @test !hasproperty(result, :lifted)
        @test result.diagnostics.selected_algorithms.solver === :native_soc
        @test result_certificate(
            problem,
            result,
            SolverOptions(Float64; tolerance=1e-8, verbosity=0),
        ).valid

        # The historical representation switch was removed; the compact API
        # exposes no public non-native path.
        @test_throws MethodError solve_socp(
            problem;
            soc_representation=:native,
            tolerance=1e-8,
            verbosity=0,
        )
    end

    @testset "fixed-trace SDP remains an SDP compatibility model" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, :, :] = [1.0 0.0; 0.0 -1.0]
        coefficients[2, :, :] = [0.0 1.0; 1.0 0.0]
        problem = ingest(
            [0.0, 0.0],
            [coefficients],
            [[0.0 0.0; 0.0 -2.0]],
            zeros(2, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        analysis = SDPX.analyze_fixed_trace(problem)
        @test analysis.fixed_blocks == 1
        @test analysis.soc_blocks == 1
        @test analysis.blocks[1].kind == :soc
        @test analysis.blocks[1].trace ≈ 2.0
        # The historical independent Q3 solver has been retired. An explicit
        # SDPProblem continues through the ordinary SDP compatibility route;
        # compact ConicProblem/MOI models use NativeSOC.
        plan = SDPX.build_execution_plan(problem)
        @test plan.algorithm == :socp_psd2
    end

    @testset "mixed SOC-shaped and general PSD blocks stay SDP" begin
        arrow = zeros(1, 3, 3)
        arrow[1, 1, 1] = 1.0
        arrow[1, 2, 2] = 1.0
        arrow[1, 3, 3] = 1.0
        large = zeros(1, 4, 4)
        large[1, 1, 1] = 1.0
        problem = ingest(
            [0.0],
            [arrow, large],
            [-Matrix{Float64}(I, 3, 3), -Matrix{Float64}(I, 4, 4)],
            zeros(1, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        @test SDPX.classify_problem(problem).cone == :sdp
        @test SDPX.build_execution_plan(problem).algorithm == :sdp_primal_dual
        @test_throws ArgumentError SDPX.build_execution_plan(
            problem,
            SolverOptions{Float64}(algorithm=:socp, verbosity=0),
        )
    end
end
