using SDPX
import MathOptInterface as MOI
using SparseArrays
using Test

@testset "MOI Model-backed adapter regressions" begin
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
end
