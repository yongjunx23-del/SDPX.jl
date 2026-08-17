using SDPX
import MathOptInterface as MOI
using SparseArrays
using Test

@testset "MOI Model-backed adapter regressions" begin
    @testset "scaled PSD product variables fail closed" begin
        source = MOI.Utilities.Model{Float64}()
        variables = MOI.add_variables(source, 3)
        set = MOI.Scaled(MOI.PositiveSemidefiniteConeTriangle(2))
        MOI.add_constraint(source, MOI.VectorOfVariables(variables), set)
        optimizer = SDPX.Optimizer(verbosity=0)

        @test !MOI.supports_constraint(
            optimizer,
            MOI.VectorOfVariables,
            typeof(set),
        )
        @test MOI.supports_constraint(
            optimizer,
            MOI.VectorAffineFunction{Float64},
            typeof(set),
        )
        @test_throws MOI.UnsupportedConstraint MOI.copy_to(optimizer, source)
        @test optimizer.model === nothing
        @test optimizer.public_result === nothing
    end

    @testset "mixed families fail closed before Model allocation" begin
        source = MOI.Utilities.Model{Float64}()
        lp_variables = MOI.add_variables(source, 2)
        soc_variables = MOI.add_variables(source, 3)
        MOI.add_constraint(
            source,
            MOI.VectorOfVariables(lp_variables),
            MOI.Nonnegatives(2),
        )
        MOI.add_constraint(
            source,
            MOI.VectorOfVariables(soc_variables),
            MOI.SecondOrderCone(3),
        )
        optimizer = SDPX.Optimizer(verbosity=0)
        @test_throws SDPX.UnsupportedNativeConeRoute MOI.copy_to(optimizer, source)
        @test optimizer.model === nothing
        @test optimizer.public_result === nothing
    end

    @testset "sparse affine rows and typed maps are preserved" begin
        source = MOI.Utilities.Model{Float64}()
        variables = MOI.add_variables(source, 50)
        equality = MOI.add_constraint(
            source,
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(2.0, variables[2]),
                    MOI.ScalarAffineTerm(-1.0, variables[40]),
                ],
                0.0,
            ),
            MOI.EqualTo(3.0),
        )
        objective = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, variables[2])],
            0.75,
        )
        MOI.set(source, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            source,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            objective,
        )
        optimizer = SDPX.Optimizer(verbosity=0)
        index_map = MOI.copy_to(optimizer, source)
        model = optimizer.model::SDPX.Model{Float64}
        program = SDPX.compile_product_cone_model(model)
        @test program.equality_matrix isa SparseMatrixCSC
        @test size(program.equality_matrix) == (1, 50)
        @test nnz(program.equality_matrix) == 2
        @test program.rhs == [3.0]
        @test index_map[equality] isa MOI.ConstraintIndex
        @test index_map[variables[40]] == MOI.VariableIndex(40)
        @test MOI.get(optimizer, MOI.NumberOfVariables()) == 50
        @test MOI.get(optimizer, MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64},
        }()) == [index_map[equality]]
    end

    @testset "Max objective flips only in the LP lowerer" begin
        source = MOI.Utilities.Model{Float64}()
        variables = MOI.add_variables(source, 1)
        MOI.add_constraint(
            source,
            MOI.VectorOfVariables(variables),
            MOI.Nonnegatives(1),
        )
        MOI.set(source, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(
            source,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, variables[1])],
                2.0,
            ),
        )
        optimizer = SDPX.Optimizer(verbosity=0)
        MOI.copy_to(optimizer, source)
        program = SDPX.compile_product_cone_model(optimizer.model)
        @test program.objective_sense isa SDPX.Maximize
        @test program.objective_vector == [1.0]
        @test program.objective_constant == 2.0
        lowered = SDPX.lower_lp_native(program; sparse=true, verbosity=0)
        @test lowered.objective_sign == -1
        @test lowered.objective_constant == 2.0
        @test lowered.core.c == [-1.0]
    end

    @testset "unsupported warm-start seam fails with a typed reason" begin
        source = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
        variable = MOI.add_variable(source)
        interval = MOI.add_constraint(source, variable, MOI.Interval(0.0, 1.0))
        MOI.set(source, MOI.ConstraintDualStart(), interval, 0.0)
        optimizer = SDPX.Optimizer(verbosity=0)
        MOI.copy_to(optimizer, source)
        @test optimizer.start_error !== nothing
        @test optimizer.start_error[1] == :warm_start_core_gap
        @test_throws SDPX.MOIAdapterError MOI.optimize!(optimizer)
    end

    @testset "free vector duals stay concretely typed" begin
        source = MOI.Utilities.Model{Float64}()
        variables = MOI.add_variables(source, 2)
        cone = MOI.add_constraint(
            source,
            MOI.VectorOfVariables(variables[1:1]),
            MOI.Nonnegatives(1),
        )
        free = MOI.add_constraint(
            source,
            MOI.VectorAffineFunction(
                [MOI.VectorAffineTerm(
                    1,
                    MOI.ScalarAffineTerm(1.0, variables[2]),
                )],
                [0.0],
            ),
            MOI.Reals(1),
        )
        MOI.set(source, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            source,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, variables[1])],
                0.0,
            ),
        )
        optimizer = SDPX.Optimizer(verbosity=0, max_iterations=80)
        index_map = MOI.copy_to(optimizer, source)
        MOI.optimize!(optimizer)
        dual = MOI.get(optimizer, MOI.ConstraintDual(), index_map[free])
        @test dual isa Vector{Float64}
        @test dual == [0.0]
        @test MOI.get(optimizer, MOI.ConstraintPrimal(), index_map[cone]) ≈ [0.0] atol=1e-8
    end

    @testset "source function/set types disambiguate native block names" begin
        source = MOI.Utilities.Model{Float64}()
        variables = MOI.add_variables(source, 2)
        scalar = MOI.add_constraint(
            source,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, variables[1])],
                0.0,
            ),
            MOI.EqualTo(0.0),
        )
        variable = MOI.add_constraint(
            source,
            variables[2],
            MOI.EqualTo(1.0),
        )
        optimizer = SDPX.Optimizer(verbosity=0)
        index_map = MOI.copy_to(optimizer, source)
        model = optimizer.model::SDPX.Model{Float64}
        @test index_map[scalar].value == 1
        @test index_map[variable].value == 1
        @test length(model.constraint_names) == 2
        @test length(unique(collect(keys(model.constraint_names)))) == 2
        @test all(startswith(String(name), "moi_scalar_constraint_")
                  for name in keys(model.constraint_names))
    end

    @testset "scaled PSD dual starts use inverse off-diagonal scaling" begin
        source = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
        variables = MOI.add_variables(source, 1)
        psd = MOI.add_constraint(
            source,
            MOI.VectorAffineFunction(
                [MOI.VectorAffineTerm(
                    1,
                    MOI.ScalarAffineTerm(1.0, variables[1]),
                )],
                [0.0, 0.0, 0.0],
            ),
            MOI.Scaled(MOI.PositiveSemidefiniteConeTriangle(2)),
        )
        MOI.set(source, MOI.ConstraintDualStart(), psd, [1.0, sqrt(2.0), 3.0])
        optimizer = SDPX.Optimizer(verbosity=0)
        index_map = MOI.copy_to(optimizer, source)
        info = optimizer.model_constraint_records[
            (typeof(index_map[psd]), index_map[psd].value)
        ]
        record = optimizer.model.constraint_blocks[info.refs[1].block]
        @test record.dual_start !== nothing
        @test record.dual_start[1] == 1.0
        @test record.dual_start[2] ≈ 1.0 atol=1e-14
        @test record.dual_start[3] == 3.0
    end

    @testset "objective sense setter updates copied Model objective" begin
        fresh = SDPX.Optimizer(verbosity=0)
        @test MOI.get(fresh, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
        MOI.empty!(fresh)
        @test MOI.get(fresh, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE

        source = MOI.Utilities.Model{Float64}()
        variables = MOI.add_variables(source, 1)
        MOI.add_constraint(
            source,
            MOI.VectorOfVariables(variables),
            MOI.Nonnegatives(1),
        )
        MOI.set(source, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            source,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, variables[1])],
                0.0,
            ),
        )
        optimizer = SDPX.Optimizer(verbosity=0)
        MOI.copy_to(optimizer, source)
        @test optimizer.model.objective.sense isa SDPX.Minimize
        MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        @test MOI.get(optimizer, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        @test optimizer.model.objective.sense isa SDPX.Maximize
        @test optimizer.public_result === nothing
        @test SDPX.compile_product_cone_model(optimizer.model).objective_sense isa SDPX.Maximize

        MOI.set(optimizer, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
        @test MOI.get(optimizer, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
        @test optimizer.model.objective === nothing
        @test optimizer.objective_constant == 0.0
        feasibility_objective = MOI.get(
            optimizer,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        )
        @test isempty(feasibility_objective.terms)
        @test feasibility_objective.constant == 0.0
        @test MOI.get(optimizer, MOI.ResultCount()) == 0
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) == 0.0
    end

    @testset "BigFloat PSD getter scaling honors model precision" begin
        matrix = setprecision(BigFloat, 256) do
            BigFloat[1 1; 1 3]
        end
        values = setprecision(BigFloat, 64) do
            SDPX._moi_psd_vector_from_matrix(
                matrix,
                2,
                true;
                precision_bits=256,
            )
        end
        expected = setprecision(BigFloat, 256) do
            BigFloat(sqrt(BigFloat(2)))
        end
        @test precision(values[2]) == 256
        @test values[2] == expected
    end

    @testset "free/equality-only models fail before solver allocation" begin
        source = MOI.Utilities.Model{Float64}()
        variable = MOI.add_variable(source)
        MOI.add_constraint(
            source,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, variable)],
                0.0,
            ),
            MOI.EqualTo(0.0),
        )
        optimizer = SDPX.Optimizer(verbosity=0)
        MOI.copy_to(optimizer, source)
        @test_throws SDPX.MOIAdapterError MOI.optimize!(optimizer)
        @test optimizer.public_result === nothing
    end

    @testset "BigFloat MOI scalar getters honor requested precision" begin
        setprecision(BigFloat, 64) do
            optimizer = SDPX.Optimizer{BigFloat}(
                precision=256,
                verbosity=0,
            )
            empty_objective = MOI.get(
                optimizer,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{BigFloat}}(),
            )
            @test precision(empty_objective.constant) == 256

            source = MOI.Utilities.Model{BigFloat}()
            variable = MOI.add_variable(source)
            interval = MOI.add_constraint(
                source,
                variable,
                MOI.Interval(BigFloat(0), BigFloat(1)),
            )
            MOI.set(source, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(
                source,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{BigFloat}}(),
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm(BigFloat(1), variable)],
                    BigFloat(0),
                ),
            )
            optimizer = SDPX.Optimizer{BigFloat}(
                precision=256,
                verbosity=0,
                max_iterations=40,
            )
            index_map = MOI.copy_to(optimizer, source)
            copied_objective = MOI.get(
                optimizer,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{BigFloat}}(),
            )
            @test precision(copied_objective.constant) == 256
            MOI.optimize!(optimizer)
            interval_dual = MOI.get(
                optimizer,
                MOI.ConstraintDual(),
                index_map[interval],
            )
            @test precision(interval_dual) == 256

            unbounded = MOI.Utilities.Model{BigFloat}()
            variable = MOI.add_variable(unbounded)
            MOI.add_constraint(
                unbounded,
                MOI.VectorOfVariables([variable]),
                MOI.Nonnegatives(1),
            )
            MOI.set(unbounded, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(
                unbounded,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{BigFloat}}(),
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm(BigFloat(-1), variable)],
                    BigFloat(0),
                ),
            )
            optimizer = SDPX.Optimizer{BigFloat}(
                precision=256,
                verbosity=0,
                max_iterations=40,
            )
            MOI.copy_to(optimizer, unbounded)
            MOI.optimize!(optimizer)
            dual_objective = MOI.get(optimizer, MOI.DualObjectiveValue())
            @test dual_objective isa BigFloat
            @test isnan(dual_objective)
            @test precision(dual_objective) == 256
        end
    end

    @testset "MOI infeasibility status pairs are one-sided" begin
        unbounded = MOI.Utilities.Model{Float64}()
        variable = MOI.add_variable(unbounded)
        MOI.add_constraint(
            unbounded,
            MOI.VectorOfVariables([variable]),
            MOI.Nonnegatives(1),
        )
        MOI.set(unbounded, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            unbounded,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(-1.0, variable)],
                0.0,
            ),
        )
        optimizer = SDPX.Optimizer(verbosity=0, max_iterations=80)
        MOI.copy_to(optimizer, unbounded)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
        @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.INFEASIBILITY_CERTIFICATE
        @test MOI.get(optimizer, MOI.DualStatus()) == MOI.NO_SOLUTION

        infeasible = MOI.Utilities.Model{Float64}()
        variable = MOI.add_variable(infeasible)
        MOI.add_constraint(
            infeasible,
            MOI.VectorOfVariables([variable]),
            MOI.Nonnegatives(1),
        )
        MOI.add_constraint(
            infeasible,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, variable)],
                0.0,
            ),
            MOI.LessThan(-1.0),
        )
        MOI.set(infeasible, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            infeasible,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm{Float64}[],
                0.0,
            ),
        )
        optimizer = SDPX.Optimizer(verbosity=0, max_iterations=80)
        MOI.copy_to(optimizer, infeasible)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.INFEASIBLE
        @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        @test MOI.get(optimizer, MOI.DualStatus()) == MOI.NO_SOLUTION

        # Optimize-mode primal infeasibility carries a dual certificate; the
        # branch is asserted directly because the tiny structural LP above
        # intentionally terminates as the historical InfeasibleCert.
        @test SDPX._moi_primal_status(
            SDPX.PrimalInfeasible,
            1,
            nothing,
        ) == MOI.NO_SOLUTION
        @test SDPX._moi_dual_status(
            SDPX.PrimalInfeasible,
            1,
            nothing,
        ) == MOI.INFEASIBILITY_CERTIFICATE
        @test SDPX._moi_primal_status(
            SDPX.InfeasibleCert,
            1,
            nothing,
        ) == MOI.NO_SOLUTION
        @test SDPX._moi_dual_status(
            SDPX.InfeasibleCert,
            1,
            nothing,
        ) == MOI.NO_SOLUTION
    end
end
