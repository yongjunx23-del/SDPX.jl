using Test
using SDPX

function _route_guard_boundary_model(T=Float64; eps=1e-8)
    model = SDPX.Model(T; name="route_guard_boundary")
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    e = T(eps)
    SDPX.constraint!(model, :fix_x, x[1] - (one(T) - e), SDPX.ZeroCone())
    SDPX.constraint!(model, :sum, x[1] + x[2] - one(T), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _route_guard_duplicate_model(T=Float64; eps=0.0)
    model = SDPX.Model(T; name="route_guard_duplicate")
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    a = one(T)
    b = one(T) + T(eps)
    SDPX.constraint!(model, :eq1, x[1] + x[2] - a, SDPX.ZeroCone())
    SDPX.constraint!(model, :eq2, x[1] + b * x[2] - b, SDPX.ZeroCone())
    SDPX.constraint!(model, :eq3, x[1] + x[2] - a, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Maximize(), x[1] + x[2])
    return model
end

function _route_guard_healthy_model(T=Float64)
    model = SDPX.Model(T; name="route_guard_healthy")
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :soc, [one(T), x[1], x[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _route_guard_outputs()
    SDPX.Outputs(:all, :all, :all; objectives=true, certificate=:summary,
        diagnostics=:full, history=false, trace=false)
end

function _route_guard_settings()
    SDPX.Settings{Float64}(kkt_route=:bordered, verbosity=0,
        limits=SDPX.Limits(iterations=500, time=30.0))
end

@testset "standard-v1 route outcomes and strict restart eligibility" begin
    # Legacy tests required these feasible problems to break down or restart.
    # Preserve their data, but qualify correct solutions under standard v1.
    for (model,objective) in ((_route_guard_boundary_model(),-(1-1e-8)),
                              (_route_guard_duplicate_model(),1.0),
                              (_route_guard_healthy_model(),-1.0))
        result=SDPX.optimize!(model;settings=_route_guard_settings(),outputs=_route_guard_outputs())
        cert=SDPX.certificate(result)
        @test SDPX.status(result)===:optimal
        @test cert.valid
        @test isapprox(cert.primal_objective,objective;atol=1e-8,rtol=1e-8)
        selected=SDPX.diagnostics(result).selected_algorithms
        @test selected.hsd_formulation_version===:standard_hsd_v1
        @test selected.requested_kkt_route===:bordered
        @test selected.planned_kkt_route===:bordered
        @test selected.attempted_kkt_routes in ((:bordered,),(:bordered,:expanded))
        @test selected.executed_kkt_route===last(selected.attempted_kkt_routes)
        @test selected.executed_fallback_chain===selected.attempted_kkt_routes
        if selected.executed_kkt_route===:expanded
            @test selected.route_restart_reason in (:symmetric_core_predictor_residual_failed,
                :disjoint_fixed_head_q3_predictor_residual_failed)
            @test selected.route_restart_iteration<=1
            @test selected.fallback_reason===:bordered_predictor_residual_fallback
            @test selected.executed_kkt_formulation===:dense_expanded_quasidefinite
        else
            @test !hasproperty(selected,:route_restart_reason)
            @test !hasproperty(selected,:route_restart_iteration)
            @test selected.fallback_reason===:none
            @test selected.executed_kkt_formulation===:symmetric_augmented_hsd_core
        end
    end
    # Eligibility is independent of whether a particular platform happens to
    # fail a factorization. The exact prior trigger is retained, not broadened.
    for reason in (:symmetric_core_predictor_residual_failed,
                   :disjoint_fixed_head_q3_predictor_residual_failed)
        for iteration in (0,1)
            @test SDPX._native_hsd_should_restart_bordered(:bordered,SDPX.NumericalBreakdown,reason,iteration)
        end
        @test !SDPX._native_hsd_should_restart_bordered(:bordered,SDPX.NumericalBreakdown,reason,2)
        @test !SDPX._native_hsd_should_restart_bordered(:expanded,SDPX.NumericalBreakdown,reason,0)
        @test !SDPX._native_hsd_should_restart_bordered(:bordered,SDPX.Optimal,reason,0)
    end
    @test !SDPX._native_hsd_should_restart_bordered(:bordered,SDPX.NumericalBreakdown,:unrelated_failure,0)
end
