using SDPX
using JuMP
using MultiFloats: Float64x4
using Test
import MathOptInterface as MOI

function moi_t1_model(::Type{T}; scaled::Bool=false, equality::Bool=false) where {T}
    model = MOI.Utilities.Model{T}()
    x = MOI.add_variables(model, 2)
    scale = scaled ? sqrt(T(2)) : one(T)
    function_value = MOI.VectorAffineFunction(
        [
            MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(one(T), x[1])),
            MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(one(T), x[2])),
        ],
        T[0, -scale, 0],
    )
    set = MOI.PositiveSemidefiniteConeTriangle(2)
    actual_set = scaled ? MOI.Scaled(set) : set
    psd = MOI.add_constraint(model, function_value, actual_set)
    eq = if equality
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(one(T), x[1])],
                zero(T),
            ),
            MOI.EqualTo(T(2)),
        )
    else
        nothing
    end
    objective = MOI.ScalarAffineFunction(
        [
            MOI.ScalarAffineTerm(T(2), x[1]),
            MOI.ScalarAffineTerm(T(3), x[2]),
        ],
        T(0.25),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        objective,
    )
    return model, x, psd, eq
end

@testset "MathOptInterface wrapper" begin
    @testset "copy_to, PSD primal/dual, and equality — $T" for T in (Float64, Float64x4)
        model, x, psd, equality = moi_t1_model(T; equality=true)
        optimizer = SDPX.Optimizer{T}(
            sparse=true,
            verbosity=0,
            tol_gap=T(1e-8),
            tol_primal=T(1e-8),
            tol_dual=T(1e-8),
        )
        index_map = MOI.copy_to(optimizer, model)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(optimizer, MOI.DualStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ T(5.75) rtol=T(1e-6)
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[1]]) ≈ T(2) rtol=T(1e-6)
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x[2]]) ≈ T(0.5) rtol=T(1e-5)
        primal = MOI.get(optimizer, MOI.ConstraintPrimal(), index_map[psd])
        @test primal ≈ T[2, -1, 0.5] rtol=T(1e-5)
        dual = MOI.get(optimizer, MOI.ConstraintDual(), index_map[psd])
        @test dual ≈ T[0.75, 1.5, 3] rtol=T(1e-5)
        @test MOI.get(
            optimizer,
            MOI.ConstraintPrimal(),
            index_map[equality],
        ) ≈ T(2) rtol=T(1e-7)
        @test MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[equality],
        ) ≈ T(1.25) rtol=T(1e-5)
        @test MOI.get(optimizer, MOI.BarrierIterations()) > 0
        @test MOI.get(optimizer, MOI.SolveTimeSec()) >= 0
    end

    @testset "scaled PSD triangle conversion" begin
        model, _, psd, _ = moi_t1_model(Float64; scaled=true)
        optimizer = SDPX.Optimizer(sparse=true, verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        primal = MOI.get(optimizer, MOI.ConstraintPrimal(), index_map[psd])
        @test primal[2] ≈ -sqrt(2.0) rtol=1e-8
        dual = MOI.get(optimizer, MOI.ConstraintDual(), index_map[psd])
        @test dual ≈ [2.0, sqrt(12.0), 3.0] rtol=1e-7
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 2sqrt(6.0) + 0.25 rtol=1e-7
    end

    @testset "maximum objective and VariableIndex objective" begin
        model = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(model)
        psd_function = MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x))],
            [1.0, 0.0, 1.0],
        )
        MOI.add_constraint(
            model,
            psd_function,
            MOI.PositiveSemidefiniteConeTriangle(2),
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
        optimizer = SDPX.Optimizer(sparse=true, verbosity=0)
        index_map = MOI.copy_to(optimizer, model)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 1.0 rtol=1e-6
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x]) ≈ 1.0 rtol=1e-6
    end

    @testset "JuMP smoke test" begin
        model = Model(() -> SDPX.Optimizer(sparse=true, verbosity=0))
        @variable(model, x[1:2])
        @constraint(
            model,
            Symmetric([x[1] -1.0; -1.0 x[2]]) in PSDCone(),
        )
        @objective(model, Min, 2x[1] + 3x[2])
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test objective_value(model) ≈ 2sqrt(6.0) rtol=1e-7
        @test value(x[1]) ≈ sqrt(3 / 2) rtol=1e-6
        @test value(x[2]) ≈ sqrt(2 / 3) rtol=1e-6
    end

    @testset "JuMP Float64x4 smoke test" begin
        T = Float64x4
        model = GenericModel{T}(
            () -> SDPX.Optimizer{T}(sparse=true, verbosity=0),
        )
        @variable(model, x[1:2])
        @constraint(
            model,
            Symmetric([x[1] T(-1); T(-1) x[2]]) in PSDCone(),
        )
        @objective(model, Min, T(2) * x[1] + T(3) * x[2])
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test objective_value(model) ≈ T(2sqrt(6.0)) rtol=T(1e-7)
        @test value(x[1]) ≈ T(sqrt(3 / 2)) rtol=T(1e-6)
        @test value(x[2]) ≈ T(sqrt(2 / 3)) rtol=T(1e-6)
    end
end
