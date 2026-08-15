import MathOptInterface as MOI
import MutableArithmetics as MA
using SDPX
using JuMP
using MultiFloats
using SparseArrays
using Test

function _vector_affine_identity(variables)
    return MOI.VectorAffineFunction(
        MOI.VectorAffineTerm{Float64}[
            MOI.VectorAffineTerm(
                coordinate,
                MOI.ScalarAffineTerm(1.0, variables[coordinate]),
            ) for coordinate in eachindex(variables)
        ],
        zeros(length(variables)),
    )
end

@testset "MOI vector cone and RSOC frontend" begin
    @testset "RSOC VAF known optimum, primal inverse, dual adjoint" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 3)
        rsoc = MOI.add_constraint(
            model,
            _vector_affine_identity(x),
            MOI.RotatedSecondOrderCone(3),
        )
        first_equality = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(1.0, x[1]),
                    MOI.ScalarAffineTerm(-1.0, x[2]),
                ],
                0.0,
            ),
            MOI.EqualTo(3.0),
        )
        second_equality = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[3])],
                0.0,
            ),
            MOI.EqualTo(2.0 * sqrt(2.0)),
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(1.0, x[1]),
                    MOI.ScalarAffineTerm(1.0, x[2]),
                ],
                0.0,
            ),
        )

        optimizer = SDPX.Optimizer(verbosity=0, tolerance=1e-8)
        index_map = MOI.copy_to(optimizer, model)
        @test optimizer.problem isa SDPX.ConicProblem
        @test optimizer.problem.cones[1].A isa SparseMatrixCSC
        MOI.optimize!(optimizer)

        sqrt2 = sqrt(2.0)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 5.0 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[1]]) ≈
              4.0 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[2]]) ≈
              1.0 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[3]]) ≈
              2.0 * sqrt2 atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[rsoc],
        ) ≈ [4.0, 1.0, 2.0 * sqrt2] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[rsoc],
        ) ≈ [0.4, 1.6, -0.8 * sqrt2] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[first_equality],
        ) ≈ 0.6 atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[second_equality],
        ) ≈ 0.8 * sqrt2 atol=1e-6
        @test optimizer.result isa SDPX.ConicResult
        @test !hasproperty(optimizer.result, :lifted)
    end

    @testset "RSOC VectorOfVariables dual stationarity" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 3)
        rsoc = MOI.add_constraint(
            model,
            MOI.VectorOfVariables(x),
            MOI.RotatedSecondOrderCone(3),
        )
        first_equality = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[1])],
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
        second_equality = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[3])],
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[2])],
                0.0,
            ),
        )

        optimizer = SDPX.Optimizer(verbosity=0, tolerance=1e-8)
        index_map = MOI.copy_to(optimizer, model)
        MOI.optimize!(optimizer)

        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 0.5 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[1]]) ≈
              1.0 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[2]]) ≈
              0.5 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[3]]) ≈
              1.0 atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[rsoc],
        ) ≈ [1.0, 0.5, 1.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[rsoc],
        ) ≈ [0.5, 1.0, -1.0] atol=3e-6
    end

    @testset "RSOC dimension two maps without tail coordinates" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)
        rsoc = MOI.add_constraint(
            model,
            MOI.VectorOfVariables(x),
            MOI.RotatedSecondOrderCone(2),
        )
        equality = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(1.0, x[1]),
                    MOI.ScalarAffineTerm(-1.0, x[2]),
                ],
                0.0,
            ),
            MOI.EqualTo(3.0),
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(1.0, x[1]),
                    MOI.ScalarAffineTerm(1.0, x[2]),
                ],
                0.0,
            ),
        )

        optimizer = SDPX.Optimizer(verbosity=0, tolerance=1e-8)
        index_map = MOI.copy_to(optimizer, model)
        @test optimizer.problem.cones[1].A isa SparseMatrixCSC
        @test nnz(optimizer.problem.cones[1].A) == 4
        MOI.optimize!(optimizer)

        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 3.0 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[1]]) ≈
              3.0 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[2]]) ≈
              0.0 atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[rsoc],
        ) ≈ [3.0, 0.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[rsoc],
        ) ≈ [0.0, 2.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[equality],
        ) ≈ 1.0 atol=1e-6
    end

    @testset "JuMP RSOC smoke with dual" begin
        model = Model(() -> SDPX.Optimizer(verbosity=0, tolerance=1e-8))
        @variable(model, a)
        @variable(model, b)
        @variable(model, w)
        rsoc = @constraint(model, [a, b, w] in RotatedSecondOrderCone())
        @constraint(model, a == 1)
        @constraint(model, w == 1)
        @objective(model, Min, b)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test objective_value(model) ≈ 0.5 atol=1e-6
        @test value(a) ≈ 1.0 atol=1e-6
        @test value(b) ≈ 0.5 atol=1e-6
        @test value(w) ≈ 1.0 atol=1e-6
        @test dual(rsoc) ≈ [0.5, 1.0, -1.0] atol=3e-6
    end

    @testset "Nonnegatives LP route signs and constants" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)
        cone = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                [1.0, -3.0],
            ),
            MOI.Nonnegatives(2),
        )
        equality = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[1])],
                0.0,
            ),
            MOI.EqualTo(2.0),
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[2])],
                0.0,
            ),
        )

        optimizer = SDPX.Optimizer(sparse=:auto, verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        @test optimizer.problem isa SDPX.SDPProblem
        MOI.optimize!(optimizer)

        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 3.0 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[2]]) ≈
              3.0 atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[cone],
        ) ≈ [3.0, 0.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[cone],
        ) ≈ [0.0, 1.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[equality],
        ) ≈ 2.0 atol=1e-6
    end

    @testset "Nonpositives LP route signs and constants" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)
        cone = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                [-5.0, 1.0],
            ),
            MOI.Nonpositives(2),
        )
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[1])],
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(-1.0, x[1]),
                    MOI.ScalarAffineTerm(-1.0, x[2]),
                ],
                0.0,
            ),
        )

        optimizer = SDPX.Optimizer(sparse=:auto, verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        MOI.optimize!(optimizer)

        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 0.0 atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[cone],
        ) ≈ [-4.0, 0.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[cone],
        ) ≈ [0.0, -1.0] atol=1e-6
    end

    @testset "Zeros LP route becomes equality rows" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)
        cone = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                [2.0, -1.0],
            ),
            MOI.Zeros(2),
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[1])],
                0.0,
            ),
        )

        optimizer = SDPX.Optimizer(sparse=:auto, verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        info = optimizer.constraint_info[index_map[cone]]
        @test info isa SDPX.MOIVectorLinearConstraintInfo
        @test info.kind === :zeros
        @test length(info.columns) == 2
        @test optimizer.problem.dims.n == 2
        MOI.optimize!(optimizer)

        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ -2.0 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[1]]) ≈
              -2.0 atol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[2]]) ≈
              1.0 atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[cone],
        ) ≈ [0.0, 0.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[cone],
        ) ≈ [1.0, 0.0] atol=1e-6
    end

    @testset "vector cones ride the native SOC route" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 3)
        soc = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                zeros(2),
            ),
            MOI.SecondOrderCone(2),
        )
        nonneg = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[3]),
                    ),
                ],
                [1.0],
            ),
            MOI.Nonnegatives(1),
        )
        equality = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[2])],
                0.0,
            ),
            MOI.EqualTo(2.0),
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [
                    MOI.ScalarAffineTerm(1.0, x[1]),
                    MOI.ScalarAffineTerm(1.0, x[3]),
                ],
                0.0,
            ),
        )

        optimizer = SDPX.Optimizer(verbosity=0, tolerance=1e-8)
        index_map = MOI.copy_to(optimizer, model)
        @test optimizer.problem isa SDPX.ConicProblem
        @test optimizer.problem.cones[1].A isa SparseMatrixCSC
        MOI.optimize!(optimizer)

        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[soc],
        ) ≈ [2.0, 2.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[soc],
        ) ≈ [1.0, -1.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[nonneg],
        ) ≈ [0.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[nonneg],
        ) ≈ [1.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[equality],
        ) ≈ 1.0 atol=1e-6
    end

    @testset "Zeros ride the native SOC route" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 2)
        soc = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                zeros(2),
            ),
            MOI.SecondOrderCone(2),
        )
        zeros_cone = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                [-2.0],
            ),
            MOI.Zeros(1),
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[1])],
                0.0,
            ),
        )

        optimizer = SDPX.Optimizer(verbosity=0, tolerance=1e-8)
        index_map = MOI.copy_to(optimizer, model)
        MOI.optimize!(optimizer)

        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 2.0 atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[soc],
        ) ≈ [2.0, 2.0] atol=1e-6
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[zeros_cone],
        ) ≈ [0.0] atol=1e-6
    end

    @testset "vector cones preserve sparse storage" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 200)
        cone = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                    MOI.VectorAffineTerm(
                        100,
                        MOI.ScalarAffineTerm(1.0, x[100]),
                    ),
                ],
                ones(200),
            ),
            MOI.Nonnegatives(200),
        )

        optimizer = SDPX.Optimizer(sparse=true, verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        info = optimizer.constraint_info[index_map[cone]]
        @test info isa SDPX.MOIVectorLinearConstraintInfo
        @test info.kind === :nonnegative
        @test length(info.blocks) == 200
        @test optimizer.problem.cons isa SDPX.SparseCons
        sparse_cons = optimizer.problem.cons::SDPX.SparseCons
        total_nnz = 0
        for block in info.blocks
            coefficient = sparse_cons.Asp[block]
            if coefficient isa SDPX.CompactScalarCoefficientVector
                total_nnz += nnz(coefficient.coefficient)
            elseif coefficient isa SDPX.ActiveSparseCoefficientVector
                total_nnz += sum(nnz, coefficient.coefficients; init=0)
            else
                total_nnz += sum(nnz, coefficient; init=0)
            end
        end
        @test total_nnz == 3
        @test nnz(optimizer.problem.B) == 0
    end

    @testset "cancelled vector rows preserve variable dimension" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 3)
        cone = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(-1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                [1.0, 0.0],
            ),
            MOI.Nonnegatives(2),
        )

        optimizer = SDPX.Optimizer(sparse=true, verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        info = optimizer.constraint_info[index_map[cone]]
        sparse_cons = optimizer.problem.cons::SDPX.SparseCons
        cancelled = sparse_cons.Asp[info.blocks[1]]
        @test cancelled isa SDPX.ActiveSparseCoefficientVector
        @test length(cancelled) == length(x)
        @test isempty(cancelled.active_variables)
        @test all(iszero, cancelled)
    end

    @testset "RSOC preserves sparse triplet construction" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 100)
        rsoc = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                    MOI.VectorAffineTerm(
                        3,
                        MOI.ScalarAffineTerm(1.0, x[3]),
                    ),
                ],
                zeros(3),
            ),
            MOI.RotatedSecondOrderCone(3),
        )

        optimizer = SDPX.Optimizer(verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        matrix = optimizer.problem.cones[1].A
        @test matrix isa SparseMatrixCSC
        @test nnz(matrix) == 5
        @test nnz(matrix) < 3 * 100
        @test optimizer.problem.Aeq isa SparseMatrixCSC
        info = optimizer.constraint_info[index_map[rsoc]]
        @test info isa SDPX.MOISOCConstraintInfo
        @test info.representation === :rotated_lorentz
    end

    @testset "standard SOC preserves sparse triplet construction" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 100)
        soc = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                    MOI.VectorAffineTerm(
                        3,
                        MOI.ScalarAffineTerm(1.0, x[3]),
                    ),
                ],
                zeros(3),
            ),
            MOI.SecondOrderCone(3),
        )

        optimizer = SDPX.Optimizer(verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        matrix = optimizer.problem.cones[1].A
        @test matrix isa SparseMatrixCSC
        @test nnz(matrix) == 3
        @test nnz(matrix) < 3 * 100
        @test optimizer.problem.Aeq isa SparseMatrixCSC
        info = optimizer.constraint_info[index_map[soc]]
        @test info isa SDPX.MOISOCConstraintInfo
        @test info.representation === :native_lorentz
    end

    @testset "scalar rows stay sparse beside native SOC" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 200)
        MOI.add_constraint(
            model,
            MOI.VectorOfVariables(x[1:2]),
            MOI.SecondOrderCone(2),
        )
        scalar = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(2.0, x[end])],
                0.0,
            ),
            MOI.GreaterThan(1.0),
        )

        optimizer = SDPX.Optimizer(verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        info = optimizer.constraint_info[index_map[scalar]]
        row = optimizer.problem.cones[info.block].A
        @test row isa SparseMatrixCSC
        @test size(row) == (1, length(x))
        @test nnz(row) == 1
        @test row[1, end] == 2.0
    end

    @testset "bridge provenance is inspection-accessible" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(model, 3)
        rsoc = MOI.add_constraint(
            model,
            MOI.VectorOfVariables(x),
            MOI.RotatedSecondOrderCone(3),
        )
        nonneg = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                ],
                [0.0],
            ),
            MOI.Nonnegatives(1),
        )
        zeros_cone = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                [0.0],
            ),
            MOI.Zeros(1),
        )

        optimizer = SDPX.Optimizer(verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        metadata = SDPX.bridge_plan(optimizer)
        raw = MOI.get(
            optimizer,
            MOI.RawOptimizerAttribute("bridge_plan"),
        )
        @test metadata.route === :native_soc
        @test raw.route === :native_soc
        by_index = Dict(
            index => entry
            for (index, entry) in zip(
                keys(optimizer.constraint_info),
                metadata.constraints,
            )
        )
        @test by_index[index_map[rsoc]] ==
              (kind=:lorentz, representation=:rotated_linear_map)
        @test by_index[index_map[nonneg]] ==
              (kind=:nonnegative, representation=:batch_linear_rows)
        @test by_index[index_map[zeros_cone]] ==
              (kind=:zeros, representation=:batch_linear_rows)
    end

    @testset "BigFloat RSOC owns coefficients and sqrt(2)" begin
        setprecision(BigFloat, 128) do
            model = MOI.Utilities.Model{BigFloat}()
            x = MOI.add_variables(model, 3)
            coefficient = BigFloat(2)
            constants = [BigFloat(1), BigFloat(2), BigFloat(3)]
            terms = MOI.VectorAffineTerm{BigFloat}[
                MOI.VectorAffineTerm(
                    1,
                    MOI.ScalarAffineTerm(coefficient, x[1]),
                ),
                MOI.VectorAffineTerm(
                    3,
                    MOI.ScalarAffineTerm(BigFloat(4), x[3]),
                ),
            ]
            MOI.add_constraint(
                model,
                MOI.VectorAffineFunction(terms, constants),
                MOI.RotatedSecondOrderCone(3),
            )

            optimizer = SDPX.Optimizer{BigFloat}(verbosity=0)
            MOI.copy_to(optimizer, model)
            stored = optimizer.problem.cones[1]
            sqrt2 = sqrt(BigFloat(2))
            values = nonzeros(stored.A)
            @test stored.A isa SparseMatrixCSC{BigFloat,Int}
            @test length(values) == 3
            @test values[1] == 2
            @test values[2] == 2
            @test values[3] ≈ 4 * sqrt2
            @test stored.b[1] == 3
            @test stored.b[2] == -1
            @test stored.b[3] ≈ 3 * sqrt2
            @test all(precision(value) >= 128 for value in values)
            @test all(precision(value) >= 128 for value in stored.b)

            MA.operate_to!(coefficient, +, coefficient, BigFloat(10))
            MA.operate_to!(constants[1], +, constants[1], BigFloat(10))
            @test all(nonzeros(stored.A) .== [2, 2, 4 * sqrt2])
            @test stored.b[1] == 3
        end
    end

    @testset "Float64x2 RSOC map preserves extended arithmetic" begin
        T = MultiFloats.Float64x2
        model = MOI.Utilities.Model{T}()
        x = MOI.add_variables(model, 3)
        MOI.add_constraint(
            model,
            MOI.VectorOfVariables(x),
            MOI.RotatedSecondOrderCone(3),
        )

        optimizer = SDPX.Optimizer{T}(verbosity=0)
        MOI.copy_to(optimizer, model)
        matrix = optimizer.problem.cones[1].A
        @test eltype(matrix) === T
        @test matrix[1, 1] == one(T)
        @test matrix[1, 2] == one(T)
        @test matrix[2, 1] == one(T)
        @test matrix[2, 2] == -one(T)
        @test matrix[3, 3] == sqrt(T(2))
    end

    @testset "BigFloat vector cone metadata owns scalars" begin
        setprecision(BigFloat, 128) do
            model = MOI.Utilities.Model{BigFloat}()
            x = MOI.add_variable(model)
            coefficient = BigFloat(1)
            constant = BigFloat(5)
            cone = MOI.add_constraint(
                model,
                MOI.VectorAffineFunction(
                    MOI.VectorAffineTerm{BigFloat}[
                        MOI.VectorAffineTerm(
                            1,
                            MOI.ScalarAffineTerm(coefficient, x),
                        ),
                    ],
                    [constant],
                ),
                MOI.Nonnegatives(1),
            )

            optimizer = SDPX.Optimizer{BigFloat}(sparse=true, verbosity=0)
            index_map = MOI.copy_to(optimizer, model)
            info = optimizer.constraint_info[index_map[cone]]
            @test info isa SDPX.MOIVectorLinearConstraintInfo
            block = optimizer.problem.cons.Asp[info.blocks[1]]
            @test block isa SDPX.CompactScalarCoefficientVector
            @test nonzeros(block.coefficient) == [1]
            @test optimizer.problem.C[info.blocks[1]][1, 1] == -5

            MA.operate_to!(coefficient, +, coefficient, BigFloat(10))
            MA.operate_to!(constant, +, constant, BigFloat(10))
            @test nonzeros(block.coefficient) == [1]
            @test optimizer.problem.C[info.blocks[1]][1, 1] == -5
        end
    end

    @testset "unsupported dimensions fail early" begin
        @test_throws DimensionMismatch MOI.RotatedSecondOrderCone(1)

        mismatched = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(mismatched, 3)
        MOI.add_constraint(
            mismatched,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                zeros(2),
            ),
            MOI.RotatedSecondOrderCone(3),
        )
        @test_throws DimensionMismatch MOI.copy_to(
            SDPX.Optimizer(),
            mismatched,
        )

        invalid_output = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(invalid_output)
        MOI.add_constraint(
            invalid_output,
            MOI.VectorAffineFunction(
                [
                    MOI.VectorAffineTerm(
                        4,
                        MOI.ScalarAffineTerm(1.0, x),
                    ),
                ],
                zeros(3),
            ),
            MOI.RotatedSecondOrderCone(3),
        )
        @test_throws DimensionMismatch MOI.copy_to(
            SDPX.Optimizer(),
            invalid_output,
        )

        for set in (MOI.Nonnegatives(0), MOI.Nonpositives(0), MOI.Zeros(0))
            empty_model = MOI.Utilities.Model{Float64}()
            MOI.add_constraint(
                empty_model,
                MOI.VectorAffineFunction(
                    MOI.VectorAffineTerm{Float64}[],
                    Float64[],
                ),
                set,
            )
            @test_throws ArgumentError MOI.copy_to(SDPX.Optimizer(), empty_model)
        end
    end

    @testset "mixed PSD and Lorentz rejection remains explicit" begin
        psd_and_soc = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(psd_and_soc, 2)
        MOI.add_constraint(
            psd_and_soc,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        3,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                zeros(3),
            ),
            MOI.PositiveSemidefiniteConeTriangle(2),
        )
        MOI.add_constraint(
            psd_and_soc,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                zeros(2),
            ),
            MOI.SecondOrderCone(2),
        )
        @test_throws ArgumentError MOI.copy_to(
            SDPX.Optimizer(verbosity=0),
            psd_and_soc,
        )

        psd_and_rsoc = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(psd_and_rsoc, 3)
        MOI.add_constraint(
            psd_and_rsoc,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        3,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                zeros(3),
            ),
            MOI.PositiveSemidefiniteConeTriangle(2),
        )
        MOI.add_constraint(
            psd_and_rsoc,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        coordinate,
                        MOI.ScalarAffineTerm(1.0, x[coordinate]),
                    ) for coordinate in 1:3
                ],
                zeros(3),
            ),
            MOI.RotatedSecondOrderCone(3),
        )
        @test_throws ArgumentError MOI.copy_to(
            SDPX.Optimizer(verbosity=0),
            psd_and_rsoc,
        )

        psd_and_nonneg = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(psd_and_nonneg, 2)
        MOI.add_constraint(
            psd_and_nonneg,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, x[1]),
                    ),
                    MOI.VectorAffineTerm(
                        3,
                        MOI.ScalarAffineTerm(1.0, x[2]),
                    ),
                ],
                zeros(3),
            ),
            MOI.PositiveSemidefiniteConeTriangle(2),
        )
        MOI.add_constraint(
            psd_and_nonneg,
            MOI.VectorOfVariables(x),
            MOI.Nonnegatives(2),
        )
        optimizer = SDPX.Optimizer(verbosity=0)
        MOI.copy_to(optimizer, psd_and_nonneg)
        @test optimizer.problem isa SDPX.SDPProblem
    end
end
