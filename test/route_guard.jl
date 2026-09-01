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

@testset "bordered route fallback" begin
    boundary = SDPX.optimize!(_route_guard_boundary_model();
        settings=_route_guard_settings(), outputs=_route_guard_outputs())
    @test SDPX.status(boundary) === :optimal
    @test SDPX.certificate(boundary).valid
    selected = SDPX.diagnostics(boundary).selected_algorithms
    @test selected.requested_kkt_route === :bordered
    @test selected.planned_kkt_route === :bordered
    @test selected.executed_kkt_route === :expanded
    @test selected.attempted_kkt_routes === (:bordered, :expanded)
    @test selected.route_restart_reason === :fixed_trace_predictor_residual_failed
    @test selected.route_restart_iteration <= 1
    @test selected.fallback_reason === :bordered_predictor_residual_fallback

    duplicate = SDPX.optimize!(_route_guard_duplicate_model();
        settings=_route_guard_settings(), outputs=_route_guard_outputs())
    @test SDPX.status(duplicate) !== :optimal
    @test !SDPX.certificate(duplicate).valid
    duplicate_selected = SDPX.diagnostics(duplicate).selected_algorithms
    # This is an execution-level induced failure, not a synthetic status
    # mutation: the exact duplicate reaches the bordered early residual gate,
    # then runs the strict expanded terminal restart and exhausts tau recovery.
    @test duplicate_selected.requested_kkt_formulation === :auto
    @test duplicate_selected.planned_kkt_formulation === :symmetric_augmented_hsd_core
    @test duplicate_selected.executed_kkt_formulation === :dense_expanded_quasidefinite
    @test duplicate_selected.requested_kkt_route === :bordered
    @test duplicate_selected.planned_kkt_route === :bordered
    @test duplicate_selected.executed_kkt_route === :expanded
    @test duplicate_selected.planned_kkt_storage === :sparse
    @test duplicate_selected.executed_kkt_storage === :dense
    @test duplicate_selected.planned_factorization === :symmetric_ldl
    @test duplicate_selected.executed_factorization === :quasidefinite_ldlt
    @test duplicate_selected.factorization_kernel === :cholmod_symmetric_ldl
    @test duplicate_selected.planned_factorization_kernel === :cholmod_symmetric_ldl
    @test duplicate_selected.executed_factorization_kernel === :native_expanded_ldlt
    @test duplicate_selected.formulation === duplicate_selected.planned_kkt_formulation
    @test duplicate_selected.route === duplicate_selected.executed_kkt_route
    @test duplicate_selected.backend === duplicate_selected.executed_backend
    @test duplicate_selected.fallback_chain === ()
    @test duplicate_selected.attempted_kkt_routes === (:bordered, :expanded)
    @test duplicate_selected.executed_fallback_chain ===
        duplicate_selected.attempted_kkt_routes
    @test duplicate_selected.planned_la_provider === :cholmod
    @test duplicate_selected.la_executed_provider === :native_serial
    @test duplicate_selected.factorization_reuse ===
        :factor_once_homogeneous_predictor_corrector
    @test duplicate.termination.reason === :tau_collapse_recovery_exhausted
    @test duplicate_selected.route_restart_reason ===
        :fixed_trace_predictor_residual_failed
    @test duplicate_selected.route_restart_iteration <= 1

    healthy = SDPX.optimize!(_route_guard_healthy_model();
        settings=_route_guard_settings(), outputs=_route_guard_outputs())
    @test SDPX.status(healthy) === :optimal
    @test SDPX.certificate(healthy).valid
    healthy_selected = SDPX.diagnostics(healthy).selected_algorithms
    @test healthy_selected.requested_kkt_route === :bordered
    @test healthy_selected.planned_kkt_route === :bordered
    @test healthy_selected.executed_kkt_route === :bordered
    @test healthy_selected.attempted_kkt_routes === (:bordered,)
    @test !hasproperty(healthy_selected, :route_restart_reason)

end
