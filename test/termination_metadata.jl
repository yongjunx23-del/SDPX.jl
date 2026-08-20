using SDPX
using Test

function termination_metadata_fixture()
    T = Float64
    coefficients = zeros(T, 2, 2, 2)
    coefficients[1, 1, 1] = one(T)
    coefficients[2, 2, 2] = one(T)
    return SDPX.ingest(
        T[2, 3],
        [coefficients],
        [T[0 1; 1 0]],
        zeros(T, 2, 0),
        T[];
        sparse=false,
        verbosity=0,
    )
end

@testset "iterative termination metadata" begin
    @testset "Newton breakdown classification is stable" begin
        cases = (
            (
                "Cholesky factorization of X or Y failed (iterate left the interior)",
                (:cone_factorization_failed, :newton_factorization),
            ),
            (
                "Schur complement not positive definite after 2 regularization attempt(s)",
                (:kkt_factorization_failed, :newton_factorization),
            ),
            (
                "Native extended-precision fallback could not factor the Schur complement",
                (:direction_solve_failed, :newton_direction),
            ),
            (
                "final structured KKT direction residual 1.0 exceeded the accepted tolerance 1e-8",
                (:direction_residual_exceeded, :newton_refinement),
            ),
            (
                "future Newton diagnostic",
                (:newton_breakdown, :newton_step),
            ),
        )
        for (detail, expected) in cases
            @test SDPX._sdp_newton_termination_metadata(detail) == expected
        end
    end

    @testset "iterative exits carry explicit reason and stage" begin
        problem = termination_metadata_fixture()
        iteration_limited = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                parameter_policy=:auto,
                scaling=:none,
                iter_max=0,
                stall_iterations=0,
                diagnostics=true,
                verbosity=0,
            ),
        )
        @test iteration_limited.status === SDPX.IterLimit
        @test iteration_limited.termination.reason === :iteration_limit
        @test iteration_limited.termination.stage === :termination_check
        @test iteration_limited.diagnostics.termination.reason ===
              :iteration_limit

        timed_out = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                parameter_policy=:auto,
                scaling=:none,
                max_time=0.0,
                diagnostics=true,
                verbosity=0,
            ),
        )
        @test timed_out.status === SDPX.TimeLimit
        @test timed_out.termination.reason === :time_limit
        @test timed_out.termination.stage === :pipeline_setup

        optimal = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                parameter_policy=:auto,
                scaling=:none,
                iter_max=100,
                stall_iterations=0,
                diagnostics=true,
                verbosity=0,
            ),
        )
        @test optimal.status === SDPX.Optimal
        # Preserve the historical success metadata while non-success exits are
        # now explicit; this avoids changing any successful trajectory/API.
        @test optimal.termination.reason === :none
        @test optimal.termination.stage === :none
    end

    @testset "actual Newton residual breakdown is no longer :none" begin
        # A zero Schur operator with a nonzero objective has no valid Newton
        # direction.  The solve reaches the direction-residual guard and must
        # expose its stage instead of inheriting the detector's :none signal.
        coefficients = zeros(Float64, 1, 1, 1)
        singular = SDPX.ingest(
            [1.0],
            [coefficients],
            [zeros(1, 1)],
            zeros(1, 0),
            Float64[];
            sparse=false,
            verbosity=0,
        )
        result = SDPX.solve!(
            singular,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                parameter_policy=:fixed,
                presolve=false,
                scaling=:none,
                iter_max=3,
                stall_iterations=0,
                diagnostics=true,
                verbosity=0,
            ),
        )
        @test result.status === SDPX.NumericalBreakdown
        @test result.termination.reason === :direction_residual_exceeded
        @test result.termination.stage === :newton_refinement
        @test result.diagnostics.termination.reason ===
              :direction_residual_exceeded
        @test result.diagnostics.termination.stage === :newton_refinement
    end
end
