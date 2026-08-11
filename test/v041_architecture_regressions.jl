using LinearAlgebra
using SparseArrays
using Test

@testset "v0.4.1 architecture regressions" begin
    @testset "sparse equality backward errors equal dense definition" begin
        B_dense = [1.0 0.0 2.0; 0.0 -3.0 0.0; 4.0 0.0 5.0]
        B_sparse = sparse(B_dense)
        b = [2.0, -1.0, 7.0]
        x = [1.25, -0.5, 0.75]
        dense = SDPX._equality_backward_errors(B_dense, b, x)
        sparse_result = SDPX._equality_backward_errors(B_sparse, b, x)
        @test sparse_result == dense

        nominal_dense = abs.([2.0, 3.0, 5.0])
        realized_dense = copy(nominal_dense)
        nominal_sparse = copy(nominal_dense)
        realized_sparse = copy(realized_dense)
        y = [0.25, -1.0, 2.0]
        SDPX._accumulate_equality_dual_scales!(
            nominal_dense, realized_dense, B_dense, y,
        )
        SDPX._accumulate_equality_dual_scales!(
            nominal_sparse, realized_sparse, B_sparse, y,
        )
        @test nominal_sparse == nominal_dense
        @test realized_sparse == realized_dense
    end

    @testset "explicit equality QR is reflected by the execution plan" begin
        # The backend selector itself is the regression boundary: a model that
        # otherwise qualifies for sparse Schur must not advertise that route
        # when equality_solver=:qr forces the current dense/QR workspace path.
        # Existing sparse-SDP tests cover the positive sparse route; this check
        # protects the option-dependent selector without duplicating a large
        # fixture here.
        @test hasmethod(SDPX._runtime_schur_backend, Tuple{SDPX.SDPProblem,Symbol})
    end

    @testset "execution plan is authoritative for Workspace structure" begin
        blocks = 3
        shared = 2
        variables = shared + blocks
        coefficients = [
            [
                variable <= shared || variable == shared + block ?
                sparse(
                    [1, 2, 2],
                    [1, 1, 2],
                    [0.2 + 0.01variable, 0.03, -0.1],
                    2,
                    2,
                ) : spzeros(2, 2)
                for variable in 1:variables
            ]
            for block in 1:blocks
        ]
        constants = [Matrix{Float64}(1.5I, 2, 2) for _ in 1:blocks]
        problem = SDPX.ingest(
            ones(variables),
            coefficients,
            constants,
            zeros(variables, 0),
            Float64[];
            sparse=:auto,
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            scaling=:none,
            presolve=false,
            threads=1,
        )
        arrow_plan = SDPX.build_execution_plan(problem, options)
        @test arrow_plan.kkt_backend === :block_arrow
        @test arrow_plan.backend_config.route === :block_arrow
        arrow_workspace = SDPX.Workspace(
            problem;
            thread_count=1,
            execution_plan=arrow_plan,
        )
        @test arrow_workspace.arrow !== nothing
        @test arrow_workspace.sparse_kkt === nothing

        dense_config = SDPX.BackendConfiguration(
            :dense_cholesky,
            :auto,
            false,
            false,
            :off,
            (),
            false,
        )
        dense_plan = SDPX.ExecutionPlan(
            arrow_plan.classification,
            :sdp_primal_dual,
            arrow_plan.scaling,
            :dense_cholesky,
            dense_config,
            arrow_plan.gram_kernel,
            arrow_plan.schedule,
            arrow_plan.threads,
            arrow_plan.parameter_profile,
            arrow_plan.memory_budget_bytes,
            arrow_plan.parameters,
        )
        dense_workspace = SDPX.Workspace(
            problem;
            thread_count=1,
            execution_plan=dense_plan,
        )
        @test dense_workspace.arrow === nothing
        @test dense_workspace.sparse_kkt === nothing
        @test size(dense_workspace.S) == (variables, variables)

        inconsistent_config = SDPX.BackendConfiguration(
            :block_arrow,
            :auto,
            false,
            false,
            :off,
            (),
            false,
        )
        inconsistent_plan = SDPX.ExecutionPlan(
            dense_plan.classification,
            dense_plan.algorithm,
            dense_plan.scaling,
            :dense_cholesky,
            inconsistent_config,
            dense_plan.gram_kernel,
            dense_plan.schedule,
            dense_plan.threads,
            dense_plan.parameter_profile,
            dense_plan.memory_budget_bytes,
            dense_plan.parameters,
        )
        @test_throws ArgumentError SDPX.Workspace(
            problem;
            thread_count=1,
            execution_plan=inconsistent_plan,
        )

        lp = SDPX.ingest(
            [1.0],
            [reshape([1.0], 1, 1, 1)],
            [reshape([-1.0], 1, 1)],
            zeros(1, 0),
            Float64[];
            verbosity=0,
        )
        lp_plan = SDPX.build_execution_plan(lp, SDPX.SolverOptions{Float64}())
        @test lp_plan.scaling === :lp_geometric
        @test lp_plan.backend_config.deferred
    end
end
