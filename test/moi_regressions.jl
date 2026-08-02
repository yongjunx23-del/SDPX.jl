using SDPX
using SparseArrays
using Test
import MathOptInterface as MOI

function moi_regression_result(status::SDPX.SolveStatus)
    return SDPX.SDPResult{Float64}(
        status,
        string(status),
        [1.0],
        [fill(1.0, 1, 1)],
        Float64[],
        [fill(1.0, 1, 1)],
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1,
        0,
        0,
        nothing,
    )
end

function moi_sparse_conversion_model(variables::Int)
    model = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(model, variables)

    scalar_function = MOI.ScalarAffineFunction(
        [
            MOI.ScalarAffineTerm(2.0, x[1]),
            MOI.ScalarAffineTerm(-1.0, x[1]),
            MOI.ScalarAffineTerm(0.0, x[5]),
        ],
        0.0,
    )
    MOI.add_constraint(model, scalar_function, MOI.GreaterThan(0.0))

    soc_function = MOI.VectorAffineFunction(
        [
            MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
            MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[3])),
            MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(1.0, x[4])),
            MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(-1.0, x[4])),
        ],
        [1.0, 0.0, 0.0],
    )
    MOI.add_constraint(model, soc_function, MOI.SecondOrderCone(3))

    psd_function = MOI.VectorAffineFunction(
        [
            MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1])),
            MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(1.0, x[variables])),
        ],
        zeros(3),
    )
    MOI.add_constraint(
        model,
        psd_function,
        MOI.PositiveSemidefiniteConeTriangle(2),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    return model
end

@testset "MOI result and conversion regressions" begin
    @testset "result availability is conservative" begin
        optimizer = SDPX.Optimizer(verbosity=0)

        optimizer.result = moi_regression_result(SDPX.Optimal)
        @test MOI.get(optimizer, MOI.ResultCount()) == 1
        @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(optimizer, MOI.DualStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(optimizer, MOI.ObjectiveValue()) == 1.0

        optimizer.result = moi_regression_result(SDPX.FeasibleCert)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ResultCount()) == 1
        @test MOI.get(optimizer, MOI.PrimalStatus()) ==
              MOI.FEASIBLE_POINT
        @test MOI.get(optimizer, MOI.DualStatus()) ==
              MOI.UNKNOWN_RESULT_STATUS
        @test MOI.get(optimizer, MOI.ObjectiveValue()) == 1.0

        for status in (
            SDPX.IterLimit,
            SDPX.TimeLimit,
            SDPX.Stalled,
            SDPX.MaxRestartsExceeded,
            SDPX.UserStopped,
        )
            optimizer.result = moi_regression_result(status)
            @test MOI.get(optimizer, MOI.ResultCount()) == 1
            @test MOI.get(optimizer, MOI.PrimalStatus()) ==
                  MOI.UNKNOWN_RESULT_STATUS
            @test MOI.get(optimizer, MOI.DualStatus()) ==
                  MOI.UNKNOWN_RESULT_STATUS
            @test MOI.get(optimizer, MOI.ObjectiveValue()) == 1.0
        end

        optimizer.result = moi_regression_result(SDPX.InfeasibleCert)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.INFEASIBLE
        @test MOI.get(optimizer, MOI.ResultCount()) == 0
        @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        @test MOI.get(optimizer, MOI.DualStatus()) == MOI.NO_SOLUTION
        @test_throws MOI.ResultIndexBoundsError MOI.get(
            optimizer,
            MOI.ObjectiveValue(),
        )

        optimizer.result = moi_regression_result(SDPX.PrimalInfeasible)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.INFEASIBLE
        @test MOI.get(optimizer, MOI.ResultCount()) == 1
        @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        @test MOI.get(optimizer, MOI.DualStatus()) ==
              MOI.INFEASIBILITY_CERTIFICATE
        @test isnan(MOI.get(optimizer, MOI.ObjectiveValue()))

        optimizer.result = moi_regression_result(SDPX.DualInfeasible)
        @test MOI.get(optimizer, MOI.TerminationStatus()) ==
              MOI.DUAL_INFEASIBLE
        @test MOI.get(optimizer, MOI.ResultCount()) == 1
        @test MOI.get(optimizer, MOI.PrimalStatus()) ==
              MOI.INFEASIBILITY_CERTIFICATE
        @test MOI.get(optimizer, MOI.DualStatus()) == MOI.NO_SOLUTION
        @test isnan(MOI.get(optimizer, MOI.DualObjectiveValue()))

        optimizer.result = moi_regression_result(SDPX.NumericalBreakdown)
        @test MOI.get(optimizer, MOI.ResultCount()) == 0
        @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        @test MOI.get(optimizer, MOI.DualStatus()) == MOI.NO_SOLUTION
        @test_throws MOI.ResultIndexBoundsError MOI.get(
            optimizer,
            MOI.DualObjectiveValue(),
        )
    end

    @testset "raw aliases and thread attributes round-trip" begin
        optimizer = SDPX.Optimizer(verbosity=0)
        tolerance = MOI.RawOptimizerAttribute("tolerance")
        MOI.set(optimizer, tolerance, 2e-7)
        @test MOI.get(optimizer, tolerance) == 2e-7
        @test optimizer.options.ϵ_gap == 2e-7
        @test optimizer.options.ϵ_primal == 2e-7
        @test optimizer.options.ϵ_dual == 2e-7

        MOI.set(optimizer, MOI.RawOptimizerAttribute("tol_gap"), 1e-8)
        split_tolerance = MOI.get(optimizer, tolerance)
        @test split_tolerance == (
            gap=1e-8,
            primal=2e-7,
            dual=2e-7,
        )

        for (name, value) in (
            ("beta", 0.2),
            ("gamma", 0.8),
            ("max_iterations", 17),
            ("num_threads", 1),
        )
            attribute = MOI.RawOptimizerAttribute(name)
            MOI.set(optimizer, attribute, value)
            @test MOI.get(optimizer, attribute) == value
        end

        fresh = SDPX.Optimizer(verbosity=0)
        @test MOI.supports(fresh, MOI.NumberOfThreads())
        @test MOI.get(fresh, MOI.NumberOfThreads()) === nothing
        MOI.set(fresh, MOI.NumberOfThreads(), 1)
        @test MOI.get(fresh, MOI.NumberOfThreads()) == 1
        @test fresh.options.threads == 1
        MOI.set(fresh, MOI.NumberOfThreads(), nothing)
        @test MOI.get(fresh, MOI.NumberOfThreads()) === nothing
        @test fresh.options.threads == Base.Threads.nthreads()
        @test_throws ArgumentError MOI.set(
            fresh,
            MOI.NumberOfThreads(),
            0,
        )
    end

    @testset "copy-in allocates coefficients only for active variables" begin
        model = moi_sparse_conversion_model(64)
        optimizer = SDPX.Optimizer(sparse=true, verbosity=0)
        MOI.copy_to(optimizer, model)
        cons = optimizer.problem.cons
        @test cons isa SDPX.SparseCons{Float64}
        @test any(==([1]), cons.active)
        @test any(==([2, 3]), cons.active)
        @test any(==([1, 64]), cons.active)
        @test sum(length, cons.active) == 5

        for block in eachindex(cons.Asp)
            inactive = setdiff(1:optimizer.num_variables, cons.active[block])
            length(inactive) >= 2 || continue
            @test cons.Asp[block][inactive[1]] ===
                  cons.Asp[block][inactive[2]]
            @test nnz(cons.Asp[block][inactive[1]]) == 0
        end
    end

    @testset "pure-equality and unconstrained LP models are reachable" begin
        bounded_model = MOI.Utilities.Model{Float64}()
        bounded_variable = MOI.add_variable(bounded_model)
        equality = MOI.add_constraint(
            bounded_model,
            bounded_variable,
            MOI.EqualTo(2.0),
        )
        objective = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(3.0, bounded_variable)],
            0.5,
        )
        MOI.set(bounded_model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            bounded_model,
            MOI.ObjectiveFunction{typeof(objective)}(),
            objective,
        )

        bounded_optimizer = SDPX.Optimizer(sparse=true, verbosity=0)
        bounded_map = MOI.copy_to(bounded_optimizer, bounded_model)
        @test bounded_optimizer.problem.dims.L == 1
        @test bounded_optimizer.problem.dims.k == [1]
        @test bounded_optimizer.problem.C[1][1, 1] == -1.0
        @test isempty(bounded_optimizer.problem.cons.active[1])
        @test length(bounded_optimizer.constraint_info) == 1
        @test haskey(
            bounded_optimizer.constraint_info,
            bounded_map[equality],
        )

        MOI.optimize!(bounded_optimizer)
        @test MOI.get(
            bounded_optimizer,
            MOI.TerminationStatus(),
        ) == MOI.OPTIMAL
        @test MOI.get(bounded_optimizer, MOI.ResultCount()) == 1
        @test MOI.get(
            bounded_optimizer,
            MOI.VariablePrimal(),
            bounded_map[bounded_variable],
        ) ≈ 2.0 atol=1e-12
        @test MOI.get(
            bounded_optimizer,
            MOI.ConstraintPrimal(),
            bounded_map[equality],
        ) ≈ 2.0 atol=1e-12
        @test MOI.get(
            bounded_optimizer,
            MOI.ObjectiveValue(),
        ) ≈ 6.5 atol=1e-12

        unbounded_model = MOI.Utilities.Model{Float64}()
        unbounded_variable = MOI.add_variable(unbounded_model)
        MOI.set(unbounded_model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            unbounded_model,
            MOI.ObjectiveFunction{MOI.VariableIndex}(),
            unbounded_variable,
        )
        unbounded_optimizer = SDPX.Optimizer(sparse=true, verbosity=0)
        unbounded_map = MOI.copy_to(unbounded_optimizer, unbounded_model)
        @test isempty(unbounded_optimizer.constraint_info)
        @test unbounded_optimizer.problem.dims.L == 1
        @test unbounded_optimizer.problem.C[1][1, 1] == -1.0

        MOI.optimize!(unbounded_optimizer)
        @test MOI.get(
            unbounded_optimizer,
            MOI.TerminationStatus(),
        ) == MOI.DUAL_INFEASIBLE
        @test MOI.get(unbounded_optimizer, MOI.ResultCount()) == 1
        @test MOI.get(
            unbounded_optimizer,
            MOI.PrimalStatus(),
        ) == MOI.INFEASIBILITY_CERTIFICATE
        @test MOI.get(
            unbounded_optimizer,
            MOI.ObjectiveValue(),
        ) < 0.0
        @test bounded_map[bounded_variable] == MOI.VariableIndex(1)
        @test unbounded_map[unbounded_variable] == MOI.VariableIndex(1)
    end
end
