using SDPX
import MathOptInterface as MOI
using SparseArrays
using Test

function moi_contract_objective!(source, variables, coefficients; constant=0.0,
                                 sense=MOI.MIN_SENSE)
    T = eltype(coefficients)
    MOI.set(source, MOI.ObjectiveSense(), sense)
    terms = MOI.ScalarAffineTerm{T}[
        MOI.ScalarAffineTerm(coefficient, variable)
        for (coefficient, variable) in zip(coefficients, variables)
    ]
    MOI.set(
        source,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction(terms, T(constant)),
    )
    return nothing
end

function moi_contract_copy(source; kwargs...)
    # These tiny contract models should always finish to an optimal point;
    # the prior three-iteration/0.5-second cap made the result depend on the
    # surrounding test load while still asserting exact objective values.
    # Do not impose a wall-clock cap here: package-wide `--compiled-modules=no`
    # runs legitimately spend several seconds compiling the first LP kernels,
    # and this test is about the model/result contract rather than TimeLimit.
    optimizer = SDPX.Optimizer(;
        verbosity=0,
        max_iterations=200,
        kwargs...,
    )
    index_map = MOI.copy_to(optimizer, source)
    @test optimizer.model isa SDPX.Model
    @test optimizer.public_result === nothing
    return optimizer, index_map
end

@testset "MOI product-cone Model contract" begin
    @testset "LP orthant block and public result" begin
        source = MOI.Utilities.Model{Float64}()
        variables = MOI.add_variables(source, 2)
        cone = MOI.add_constraint(
            source,
            MOI.VectorOfVariables(variables),
            MOI.Nonnegatives(2),
        )
        moi_contract_objective!(source, variables, [1.0, 2.0]; constant=0.25)
        optimizer, index_map = moi_contract_copy(source)
        @test SDPX.bridge_plan(optimizer).route == :lp_family
        program = SDPX.compile_product_cone_model(optimizer.model)
        @test program.blocks[1].cone == :nonnegative
        @test program.objective_sense isa SDPX.Minimize
        @test program.objective_constant == 0.25
        @test program.equality_matrix isa SparseMatrixCSC
        @test size(program.equality_matrix) == (0, 2)

        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.ResultCount()) == 1
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        result = MOI.get(optimizer, MOI.RawSolver())
        @test result isa SDPX.Result{Float64}
        @test result === optimizer.public_result
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 0.25 atol=1e-8
        @test MOI.get(optimizer, MOI.ConstraintPrimal(), index_map[cone]) ≈
              [0.0, 0.0] atol=1e-8
    end

    @testset "SOC and RSOC blocks stay native" begin
        for (set, route) in (
            (MOI.SecondOrderCone(3), :soc_family),
            (MOI.RotatedSecondOrderCone(3), :soc_family),
        )
            source = MOI.Utilities.Model{Float64}()
            variables = MOI.add_variables(source, 3)
            cone = MOI.add_constraint(
                source,
                MOI.VectorOfVariables(variables),
                set,
            )
            # Pin two coordinates so both product cones have a bounded
            # scalar objective while retaining one native block.
            MOI.add_constraint(
                source,
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm(1.0, variables[2])],
                    0.0,
                ),
                MOI.EqualTo(1.0),
            )
            MOI.add_constraint(
                source,
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm(1.0, variables[3])],
                    0.0,
                ),
                MOI.EqualTo(0.0),
            )
            moi_contract_objective!(source, variables, [1.0, 0.0, 0.0])
            optimizer, index_map = moi_contract_copy(source)
            @test SDPX.bridge_plan(optimizer).route == route
            expected_domain = set isa MOI.SecondOrderCone ?
                              SDPX.LorentzCone : SDPX.RotatedLorentzCone
            @test optimizer.model.variable_blocks[1].domain isa
                  expected_domain
            program = SDPX.compile_product_cone_model(optimizer.model)
            @test program.blocks[1].cone ==
                  (set isa MOI.SecondOrderCone ? :soc : :rsoc)

            MOI.optimize!(optimizer)
            @test MOI.get(optimizer, MOI.ResultCount()) == 1
            @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
            result = MOI.get(optimizer, MOI.RawSolver())
            @test result isa SDPX.Result{Float64}
            @test length(MOI.get(
                optimizer,
                MOI.ConstraintPrimal(),
                index_map[cone],
            )) == 3
            @test length(MOI.get(
                optimizer,
                MOI.ConstraintDual(),
                index_map[cone],
            )) == 3
        end
    end

    @testset "PSD triangle and free/zero vectors" begin
        source = MOI.Utilities.Model{Float64}()
        psd_variables = MOI.add_variables(source, 3)
        psd = MOI.add_constraint(
            source,
            MOI.VectorOfVariables(psd_variables),
            MOI.PositiveSemidefiniteConeTriangle(2),
        )
        free_variables = MOI.add_variables(source, 2)
        free = MOI.add_constraint(source, MOI.VectorOfVariables(free_variables), MOI.Reals(2))
        zero = MOI.add_constraint(
            source,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, free_variables[1])],
                0.0,
            ),
            MOI.EqualTo(0.0),
        )
        moi_contract_objective!(source, psd_variables, [1.0, 0.0, 1.0])
        optimizer, index_map = moi_contract_copy(source)
        # Reals and Zeros have no independent numerical route; the PSD block
        # determines the only native family and remains one packed block.
        @test SDPX.bridge_plan(optimizer).route == :sdp_family
        program = SDPX.compile_product_cone_model(optimizer.model)
        psd_position = findfirst(
            block -> block.cone == :psd,
            program.blocks,
        )
        @test psd_position !== nothing
        psd_block = program.blocks[psd_position]
        @test psd_block.psd.side == :lower
        @test psd_block.psd.order == :column_major
        @test psd_block.psd.storage == :packed
        @test psd_block.psd.packed_length == 3
        @test index_map[psd] isa MOI.ConstraintIndex
        @test index_map[free] isa MOI.ConstraintIndex
        @test index_map[zero] isa MOI.ConstraintIndex
        @test MOI.get(optimizer, MOI.NumberOfVariables()) == 5
    end

    @testset "vector Reals primal and interval dual signs" begin
        # VectorAffineFunction-in-Reals(d): the constraint is vacuous (the
        # reals are the whole space) but its primal is still the full
        # d-vector of evaluated row expressions (regression: the getter
        # truncated it to the first coordinate).
        source = MOI.Utilities.Model{Float64}()
        variables = MOI.add_variables(source, 2)
        orthant = MOI.add_constraint(
            source,
            MOI.VectorOfVariables(variables),
            MOI.Nonnegatives(2),
        )
        moi_contract_objective!(source, variables, [1.0, 1.0])
        free = MOI.add_constraint(
            source,
            MOI.VectorAffineFunction(
                [
                    MOI.VectorAffineTerm(
                        1, MOI.ScalarAffineTerm(1.0, variables[1]),
                    ),
                    MOI.VectorAffineTerm(
                        2, MOI.ScalarAffineTerm(1.0, variables[2]),
                    ),
                ],
                [-1.0, -2.0],
            ),
            MOI.Reals(2),
        )
        optimizer, index_map = moi_contract_copy(source)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        primal = MOI.get(optimizer, MOI.ConstraintPrimal(), index_map[free])
        @test primal isa Vector{Float64}
        @test length(primal) == 2
        @test primal ≈ [-1.0, -2.0] atol = 1e-6
        @test MOI.get(
            optimizer, MOI.ConstraintPrimal(), index_map[orthant],
        ) ≈ [0.0, 0.0] atol = 1e-6

        # Interval dual at an active upper bound is negative (regression:
        # the getter subtracted the Nonpositive upper part — itself <= 0 —
        # so a negative interval dual was unrepresentable).
        interval_source = MOI.Utilities.Model{Float64}()
        bounded = MOI.add_variable(interval_source)
        moi_contract_objective!(interval_source, [bounded], [-1.0])
        interval = MOI.add_constraint(
            interval_source,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, bounded)], 0.0,
            ),
            MOI.Interval(0.0, 1.0),
        )
        interval_optimizer, interval_map = moi_contract_copy(interval_source)
        MOI.optimize!(interval_optimizer)
        @test MOI.get(interval_optimizer, MOI.TerminationStatus()) ==
              MOI.OPTIMAL
        @test MOI.get(
            interval_optimizer, MOI.VariablePrimal(), interval_map[bounded],
        ) ≈ 1.0 atol = 1e-6
        interval_dual = MOI.get(
            interval_optimizer, MOI.ConstraintDual(), interval_map[interval],
        )
        @test interval_dual ≈ -1.0 atol = 1e-6
    end
end

@testset "zero iteration request fails closed at the MOI seam" begin
    optimizer = SDPX.Optimizer(; verbosity=0)
    error_value = try
        MOI.set(
            optimizer,
            MOI.RawOptimizerAttribute("max_iterations"),
            0,
        )
        nothing
    catch caught
        caught
    end
    @test error_value isa ArgumentError
    @test occursin(
        "automatic sentinel", sprint(showerror, error_value),
    )
end
