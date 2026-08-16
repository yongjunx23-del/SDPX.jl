using SDPX
import MathOptInterface as MOI
using Test

function moi_native_soc_model()
    source = MOI.Utilities.Model{Float64}()
    variables = MOI.add_variables(source, 3)
    terms = MOI.VectorAffineTerm{Float64}[
        MOI.VectorAffineTerm(
            coordinate,
            MOI.ScalarAffineTerm(1.0, variables[coordinate]),
        ) for coordinate in 1:3
    ]
    soc = MOI.add_constraint(
        source,
        MOI.VectorAffineFunction(terms, zeros(3)),
        MOI.SecondOrderCone(3),
    )
    for (variable, value) in zip(variables[2:3], (3.0, 4.0))
        MOI.add_constraint(
            source,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, variable)],
                0.0,
            ),
            MOI.EqualTo(value),
        )
    end
    MOI.set(source, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        source,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, variables[1])],
            0.0,
        ),
    )
    return source, variables, soc
end

@testset "MOI Model-backed native SOC route" begin
    source, variables, soc = moi_native_soc_model()
    optimizer = SDPX.Optimizer(
        verbosity=0,
        tolerance=1e-8,
        max_iterations=120,
    )
    index_map = MOI.copy_to(optimizer, source)

    # The adapter owns exactly one public Model.  There is no alternate
    # problem object or second route-specific canonicalizer to inspect.
    @test optimizer.model isa SDPX.Model{Float64}
    @test SDPX.bridge_plan(optimizer).route == :soc_family
    @test MOI.get(optimizer, MOI.NumberOfVariables()) == 3
    @test index_map[variables[1]] == MOI.VariableIndex(1)

    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(optimizer, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 5.0 atol=1e-6
    @test MOI.get(
        optimizer,
        MOI.ConstraintPrimal(),
        index_map[soc],
    ) ≈ [5.0, 3.0, 4.0] atol=2e-6
    @test MOI.get(
        optimizer,
        MOI.ConstraintDual(),
        index_map[soc],
    ) ≈ [1.0, -0.6, -0.8] atol=2e-6

    result = MOI.get(optimizer, MOI.RawSolver())
    @test result isa SDPX.Result{Float64}
    @test result === optimizer.public_result
    @test SDPX.status(result) == :optimal
    @test SDPX.execution_plan(result) isa SDPX.ExecutionPlan
    diagnostics = SDPX.diagnostics(result)
    @test diagnostics isa SDPX.NativeSOCDiagnostics
    @test diagnostics.plan isa SDPX.ExecutionPlan
    @test diagnostics.plan.payload isa SDPX.NativeSOCPlan
    @test diagnostics.plan.algorithm === :native_soc
    @test diagnostics.plan.parameters.soc_specialization === :general_lorentz
    @test diagnostics.plan.classification.arithmetic === :float64
end

@testset "MOI BigFloat Model ownership and starts" begin
    # UniversalFallback is only a source-side MOI storage wrapper: it lets a
    # test retain start attributes before copy_to without adding any solver
    # behavior or an alternate SDPX solve path.
    expected_coefficient = setprecision(BigFloat, 256) do
        BigFloat(1) + BigFloat(2)^(-180)
    end
    source = setprecision(BigFloat, 256) do
        cached = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{BigFloat}())
        variables = MOI.add_variables(cached, 2)
        block = MOI.add_constraint(
            cached,
            MOI.VectorOfVariables(variables),
            MOI.Nonnegatives(2),
        )
        coefficient = expected_coefficient
        MOI.set(cached, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            cached,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{BigFloat}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(coefficient, variables[1])],
                BigFloat(0),
            ),
        )
        MOI.set(cached, MOI.VariablePrimalStart(), variables[1], BigFloat(1))
        MOI.set(cached, MOI.VariablePrimalStart(), variables[2], BigFloat(2))
        MOI.set(cached, MOI.ConstraintDualStart(), block, BigFloat[1, 1])
        cached
    end

    setprecision(BigFloat, 64) do
        optimizer = SDPX.Optimizer{BigFloat}(
            precision=256,
            verbosity=0,
            max_iterations=1,
        )
        index_map = MOI.copy_to(optimizer, source)
        model = optimizer.model::SDPX.Model{BigFloat}
        @test SDPX.precision_bits(model) == 256
        @test model.objective !== nothing
        objective = model.objective.expression
        @test precision(objective.coefficients[1]) == 256
        @test objective.coefficients[1] == expected_coefficient
        @test model.variable_blocks[1].primal_start !== nothing
        @test model.variable_blocks[1].dual_slack_start !== nothing
        @test all(value -> precision(value) == 256,
                  model.variable_blocks[1].primal_start)
        @test all(value -> precision(value) == 256,
                  model.variable_blocks[1].dual_slack_start)
        @test index_map[MOI.VariableIndex(1)] == MOI.VariableIndex(1)
    end
end
