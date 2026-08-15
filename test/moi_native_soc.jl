import MathOptInterface as MOI
import MutableArithmetics as MA
using SparseArrays

@testset "MOI preserves native Lorentz constraints" begin
    model = MOI.Utilities.Model{Float64}()
    variables = MOI.add_variables(model, 3)
    terms = MOI.VectorAffineTerm{Float64}[
        MOI.VectorAffineTerm(
            coordinate,
            MOI.ScalarAffineTerm(1.0, variables[coordinate]),
        ) for coordinate in 1:3
    ]
    soc = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(terms, zeros(3)),
        MOI.SecondOrderCone(3),
    )
    for (variable, value) in zip(variables[2:3], (3.0, 4.0))
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, variable)], 0.0,
            ),
            MOI.EqualTo(value),
        )
    end
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, variables[1])], 0.0,
        ),
    )

    optimizer = SDPX.Optimizer(
        verbosity=0,
        tolerance=1e-8,
        max_iterations=120,
    )
    index_map = MOI.copy_to(optimizer, model)
    @test optimizer.problem isa SDPX.ConicProblem
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 5.0 atol=1e-6
    @test MOI.get(
        optimizer, MOI.ConstraintPrimal(), index_map[soc],
    ) ≈ [5.0, 3.0, 4.0] atol=2e-6
    @test MOI.get(
        optimizer, MOI.ConstraintDual(), index_map[soc],
    ) ≈ [1.0, -0.6, -0.8] atol=2e-6
    @test !hasproperty(MOI.get(optimizer, MOI.RawSolver()), :lifted)
end

@testset "MOI BigFloat metadata owns mutable scalars" begin
    setprecision(BigFloat, 128) do
        model = MOI.Utilities.Model{BigFloat}()
        variables = MOI.add_variables(model, 2)
        soc_terms = MOI.VectorAffineTerm{BigFloat}[
            MOI.VectorAffineTerm(
                coordinate,
                MOI.ScalarAffineTerm(BigFloat(1), variables[coordinate]),
            ) for coordinate in 1:2
        ]
        MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(soc_terms, zeros(BigFloat, 2)),
            MOI.SecondOrderCone(2),
        )
        bound = BigFloat(1)
        inequality = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(BigFloat(1), variables[1])],
                BigFloat(0),
            ),
            MOI.GreaterThan(bound),
        )
        objective_constant = BigFloat(2)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{BigFloat}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(BigFloat(1), variables[1])],
                objective_constant,
            ),
        )
        optimizer = SDPX.Optimizer{BigFloat}(verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        stored_bound = optimizer.constraint_info[index_map[inequality]].bound
        stored_constant = optimizer.objective_constant
        MA.operate_to!(bound, +, bound, BigFloat(10))
        MA.operate_to!(objective_constant, +, objective_constant, BigFloat(10))
        @test stored_bound == 1
        @test stored_constant == 2
        @test optimizer.constraint_info[index_map[inequality]].bound == 1
        @test optimizer.objective_constant == 2
    end
end

@testset "MOI keeps sparse native-SOC equalities sparse" begin
    model = MOI.Utilities.Model{Float64}()
    variables = MOI.add_variables(model, 50)
    terms = MOI.VectorAffineTerm{Float64}[
        MOI.VectorAffineTerm(
            coordinate,
            MOI.ScalarAffineTerm(1.0, variables[coordinate]),
        ) for coordinate in 1:3
    ]
    MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(terms, zeros(3)),
        MOI.SecondOrderCone(3),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, variables[2])], 0.0,
        ),
        MOI.EqualTo(1.0),
    )
    optimizer = SDPX.Optimizer(verbosity=0)
    MOI.copy_to(optimizer, model)
    @test optimizer.problem isa SDPX.ConicProblem
    @test optimizer.problem.Aeq isa SparseMatrixCSC
    @test nnz(optimizer.problem.Aeq) == 1
end
