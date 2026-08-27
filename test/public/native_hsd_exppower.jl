using SDPX, Test

function _a3_settings(engine::Symbol)
    return SDPX.Settings{Float64}(
        engine=engine,
        tolerances=SDPX.Tolerances{Float64}(
            primal=1e-8, dual=1e-8, gap=1e-8,
        ),
        limits=SDPX.Limits(iterations=400, time=60.0, threads=1),
        verbosity=0,
    )
end

function _a3_model(kind::Symbol)
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    if kind === :exp
        SDPX.constraint!(
            model, :exp_row, (0.0, 1.0, x[1]), SDPX.ExponentialCone(),
        )
    else
        SDPX.constraint!(
            model, :power_row, (1.0, 1.0, x[1]), SDPX.PowerCone(0.5),
        )
    end
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    return model
end

@testset "public native HSD executes primal Exp/Power blocks" begin
    for kind in (:exp, :power)
        for engine in (:native_hsd, :auto)
            result = SDPX.optimize!(
                _a3_model(kind); settings=_a3_settings(engine),
            )
            @test SDPX.status(result) === :optimal
            @test SDPX.certificate(result).valid
            if kind === :exp
                @test SDPX.certificate(result).primal_objective ≈ 1.0 atol=2e-6
            else
                @test SDPX.certificate(result).primal_objective ≈ -1.0 atol=2e-6
            end
        end
    end
end

@testset "MOI asymmetric capability remains fail-closed" begin
    optimizer = SDPX.Optimizer{Float64}()
    @test !SDPX.MOI.supports_constraint(
        optimizer, SDPX.MOI.VectorOfVariables, SDPX.MOI.ExponentialCone,
    )
    @test !SDPX.MOI.supports_constraint(
        optimizer, SDPX.MOI.VectorAffineFunction{Float64},
        SDPX.MOI.PowerCone{Float64},
    )
end
