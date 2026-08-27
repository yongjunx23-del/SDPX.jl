using SDPX
using Test
using LinearAlgebra

function _p15_bounded_capped_lp()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :sum, x[1] + x[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :upper_one, x[1] - 1.0, SDPX.Nonpositive())
    SDPX.constraint!(model, :upper_two, x[2] - 1.0, SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Maximize(), x[1] + 2.0 * x[2])
    return model
end

function _p15_genuinely_unbounded_lp()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    return model
end

function _p15_exp_state()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :exp_row, (0.0, 1.0, x[1]), SDPX.ExponentialCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    state = SDPX.ProductConeHSDState(SDPX.hsd_equality_reduce(canonical).reduced)
    SDPX.product_hsd_cold_start!(state)
    return state
end

@testset "P1.5 corrected bounded and unbounded classification" begin
    settings = SDPX.Settings{Float64}(
        engine=:native_hsd,
        kkt_route=:expanded,
        limits=SDPX.Limits(iterations=400, time=60.0, threads=1),
    )

    capped = SDPX.optimize!(_p15_bounded_capped_lp(); settings=settings)
    @test SDPX.status(capped) === :optimal
    @test SDPX.certificate(capped).valid
    @test isapprox(SDPX.primal_objective(capped), 2.0; atol=2e-6)

    unbounded = SDPX.optimize!(_p15_genuinely_unbounded_lp(); settings=settings)
    @test SDPX.status(unbounded) === :dual_infeasible
    @test SDPX.certificate(unbounded).valid
end

@testset "P1.5 nonsymmetric trial uses trial complementarity" begin
    state = _p15_exp_state()
    base = state.base
    base.st .= 1.25 .* base.s
    base.yt .= 0.75 .* base.y
    base.tau_t = 0.6 * base.tau
    base.kappa_t = 0.5 * base.kappa
    mu_t = (dot(base.st, base.yt) + base.tau_t * base.kappa_t) /
           (base.nu + 1)
    @test abs(mu_t / base.mu - 1.0) > 0.05
    @test SDPX._product_hsd_trial_scaling!(state)
    @test state.runtime.last_mu == mu_t
    @test all(block.scaling.mu == mu_t for block in state.runtime.exp)
end

@testset "P1.5 verified terminal result owns its copied data" begin
    state = _p15_exp_state()
    marker = (:immutable_verified_result, [1.0])
    returned = SDPX._product_hsd_finish_terminal_restore!(state, marker, false)
    @test returned === marker
    @test !state.runtime.valid
    @test state.diagnostic === :post_result_state_restore_failed
end
