# R6-B MOI mapping qualification (standalone; not wired into runtests.jl).
#
# Drives SDPX.Optimizer through raw MathOptInterface with no JuMP
# dependency. The wrapper is intentionally non-incremental
# (MOI.supports_incremental_interface == false), so every solve goes
# through a MOI.Utilities.Model source + MOI.copy_to + MOI.optimize!.
# Direct MOI.add_variable on the optimizer is asserted fail-closed.

using Test
using SDPX

const MOI_R6B = SDPX.MOI

@testset "R6-B MOI mapping (raw MOI, no JuMP)" begin
    @testset "capability discovery is fail-closed and honest" begin
        optimizer = SDPX.Optimizer()
        @test MOI_R6B.supports_incremental_interface(optimizer) == false
        @test MOI_R6B.supports_constraint(
            optimizer,
            MOI_R6B.VectorAffineFunction{Float64},
            MOI_R6B.Nonnegatives,
        ) == true
        @test MOI_R6B.supports_constraint(
            optimizer,
            MOI_R6B.VectorAffineFunction{Float64},
            MOI_R6B.Zeros,
        ) == true
        @test MOI_R6B.supports_constraint(
            optimizer,
            MOI_R6B.ScalarAffineFunction{Float64},
            MOI_R6B.GreaterThan{Float64},
        ) == true
        # Exponential cone stays fail-closed on the MOI surface.
        @test MOI_R6B.supports_constraint(
            optimizer,
            MOI_R6B.VectorAffineFunction{Float64},
            MOI_R6B.ExponentialCone,
        ) == false
        @test MOI_R6B.supports(optimizer, MOI_R6B.Silent()) == true
        @test MOI_R6B.supports(optimizer, MOI_R6B.TimeLimitSec()) == true
        @test MOI_R6B.supports(optimizer, MOI_R6B.NumberOfThreads()) == true
        @test MOI_R6B.is_empty(optimizer) == true
    end

    @testset "LP optimum through copy_to (scalar-affine surface)" begin
        # min -x - y  s.t.  x >= 0, y >= 0, x + y <= 2.
        # Analytic optimum: -2 at (1, 1).
        source = MOI_R6B.Utilities.Model{Float64}()
        x = MOI_R6B.add_variable(source)
        y = MOI_R6B.add_variable(source)
        MOI_R6B.add_constraint(source, x, MOI_R6B.GreaterThan{Float64}(0.0))
        MOI_R6B.add_constraint(source, y, MOI_R6B.GreaterThan{Float64}(0.0))
        MOI_R6B.add_constraint(
            source,
            MOI_R6B.ScalarAffineFunction(
                MOI_R6B.ScalarAffineTerm{Float64}[
                    MOI_R6B.ScalarAffineTerm(1.0, x),
                    MOI_R6B.ScalarAffineTerm(1.0, y),
                ],
                0.0,
            ),
            MOI_R6B.LessThan{Float64}(2.0),
        )
        MOI_R6B.set(source, MOI_R6B.ObjectiveSense(), MOI_R6B.MIN_SENSE)
        MOI_R6B.set(
            source,
            MOI_R6B.ObjectiveFunction{MOI_R6B.ScalarAffineFunction{Float64}}(),
            MOI_R6B.ScalarAffineFunction(
                MOI_R6B.ScalarAffineTerm{Float64}[
                    MOI_R6B.ScalarAffineTerm(-1.0, x),
                    MOI_R6B.ScalarAffineTerm(-1.0, y),
                ],
                0.0,
            ),
        )

        optimizer = SDPX.Optimizer()
        MOI_R6B.set(optimizer, MOI_R6B.Silent(), true)
        MOI_R6B.copy_to(optimizer, source)
        @test MOI_R6B.is_empty(optimizer) == false
        @test MOI_R6B.get(optimizer, MOI_R6B.NumberOfVariables()) == 2
        MOI_R6B.optimize!(optimizer)
        @test MOI_R6B.get(optimizer, MOI_R6B.TerminationStatus()) ==
              MOI_R6B.OPTIMAL
        @test MOI_R6B.get(optimizer, MOI_R6B.ObjectiveValue()) ≈ -2.0 atol =
            1e-6
        @test MOI_R6B.get(optimizer, MOI_R6B.PrimalStatus()) ==
              MOI_R6B.FEASIBLE_POINT
        @test MOI_R6B.get(optimizer, MOI_R6B.DualStatus()) ==
              MOI_R6B.FEASIBLE_POINT
        @test MOI_R6B.get(
            optimizer,
            MOI_R6B.VariablePrimal(),
            MOI_R6B.VariableIndex(1),
        ) ≈ 1.0 atol = 1e-6
        @test MOI_R6B.get(
            optimizer,
            MOI_R6B.VariablePrimal(),
            MOI_R6B.VariableIndex(2),
        ) ≈ 1.0 atol = 1e-6
    end

    @testset "vector-affine Nonnegatives mapping" begin
        # min x  s.t.  [x - 1, y - 1] in Nonnegatives. Optimum x = 1.
        source = MOI_R6B.Utilities.Model{Float64}()
        x = MOI_R6B.add_variable(source)
        y = MOI_R6B.add_variable(source)
        MOI_R6B.add_constraint(
            source,
            MOI_R6B.VectorAffineFunction{Float64}(
                MOI_R6B.VectorAffineTerm{Float64}[
                    MOI_R6B.VectorAffineTerm(1, MOI_R6B.ScalarAffineTerm(1.0, x)),
                    MOI_R6B.VectorAffineTerm(2, MOI_R6B.ScalarAffineTerm(1.0, y)),
                ],
                [-1.0, -1.0],
            ),
            MOI_R6B.Nonnegatives(2),
        )
        MOI_R6B.set(source, MOI_R6B.ObjectiveSense(), MOI_R6B.MIN_SENSE)
        MOI_R6B.set(
            source,
            MOI_R6B.ObjectiveFunction{MOI_R6B.VariableIndex}(),
            x,
        )
        optimizer = SDPX.Optimizer()
        MOI_R6B.set(optimizer, MOI_R6B.Silent(), true)
        MOI_R6B.copy_to(optimizer, source)
        MOI_R6B.optimize!(optimizer)
        @test MOI_R6B.get(optimizer, MOI_R6B.TerminationStatus()) ==
              MOI_R6B.OPTIMAL
        @test MOI_R6B.get(optimizer, MOI_R6B.ObjectiveValue()) ≈ 1.0 atol =
            1e-6
        @test MOI_R6B.get(optimizer, MOI_R6B.PrimalStatus()) ==
              MOI_R6B.FEASIBLE_POINT
        @test MOI_R6B.get(optimizer, MOI_R6B.DualStatus()) ==
              MOI_R6B.FEASIBLE_POINT
    end

    @testset "option round-trips follow wrapper semantics" begin
        optimizer = SDPX.Optimizer()
        # Silent defaults to false (verbosity != 0); set/get round-trips.
        @test MOI_R6B.get(optimizer, MOI_R6B.Silent()) == false
        MOI_R6B.set(optimizer, MOI_R6B.Silent(), true)
        @test MOI_R6B.get(optimizer, MOI_R6B.Silent()) == true
        MOI_R6B.set(optimizer, MOI_R6B.Silent(), false)
        @test MOI_R6B.get(optimizer, MOI_R6B.Silent()) == false
        # TimeLimitSec defaults to Inf; `nothing` resets to Inf.
        @test MOI_R6B.get(optimizer, MOI_R6B.TimeLimitSec()) == Inf
        MOI_R6B.set(optimizer, MOI_R6B.TimeLimitSec(), 30.0)
        @test MOI_R6B.get(optimizer, MOI_R6B.TimeLimitSec()) == 30.0
        MOI_R6B.set(optimizer, MOI_R6B.TimeLimitSec(), nothing)
        @test MOI_R6B.get(optimizer, MOI_R6B.TimeLimitSec()) == Inf
        # NumberOfThreads defaults to nothing (unset); positive values stick.
        @test MOI_R6B.get(optimizer, MOI_R6B.NumberOfThreads()) === nothing
        MOI_R6B.set(optimizer, MOI_R6B.NumberOfThreads(), 2)
        @test MOI_R6B.get(optimizer, MOI_R6B.NumberOfThreads()) == 2
        @test_throws ArgumentError MOI_R6B.set(
            optimizer,
            MOI_R6B.NumberOfThreads(),
            0,
        )
    end

    @testset "empty! resets; incremental use fails closed" begin
        source = MOI_R6B.Utilities.Model{Float64}()
        x = MOI_R6B.add_variable(source)
        MOI_R6B.add_constraint(source, x, MOI_R6B.GreaterThan{Float64}(0.0))
        MOI_R6B.set(source, MOI_R6B.ObjectiveSense(), MOI_R6B.MIN_SENSE)
        MOI_R6B.set(
            source,
            MOI_R6B.ObjectiveFunction{MOI_R6B.VariableIndex}(),
            x,
        )
        optimizer = SDPX.Optimizer()
        MOI_R6B.set(optimizer, MOI_R6B.Silent(), true)
        MOI_R6B.copy_to(optimizer, source)
        @test MOI_R6B.is_empty(optimizer) == false
        MOI_R6B.empty!(optimizer)
        @test MOI_R6B.is_empty(optimizer) == true
        @test MOI_R6B.get(optimizer, MOI_R6B.NumberOfVariables()) == 0
        # Non-incremental: direct variable addition is refused.
        @test_throws MOI_R6B.AddVariableNotAllowed MOI_R6B.add_variable(
            optimizer,
        )
        # Exponential-cone sources are rejected at copy time.
        bad = MOI_R6B.Utilities.Model{Float64}()
        bx = MOI_R6B.add_variable(bad)
        MOI_R6B.add_constraint(
            bad,
            MOI_R6B.VectorAffineFunction{Float64}(
                MOI_R6B.VectorAffineTerm{Float64}[
                    MOI_R6B.VectorAffineTerm(
                        1,
                        MOI_R6B.ScalarAffineTerm(1.0, bx),
                    ),
                ],
                [0.0],
            ),
            MOI_R6B.ExponentialCone(),
        )
        @test_throws MOI_R6B.UnsupportedConstraint MOI_R6B.copy_to(
            optimizer,
            bad,
        )
    end
end
