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

@testset "P1.5 direct nonsymmetric optimal verification precedes refinement" begin
    state = _p15_exp_state()
    solved = SDPX.product_hsd_solve!(state; max_iterations=400, tol=1e-8)
    @test solved.status === SDPX.ProductHSDOptimal

    replay = SDPX.ProductConeHSDState(state.base.canonical)
    copyto!(replay.base.x, solved.hsd_x)
    copyto!(replay.base.s, solved.hsd_s)
    copyto!(replay.base.y, solved.hsd_y)
    replay.base.tau = solved.tau
    replay.base.kappa = solved.kappa
    SDPX.hsd_residual!(replay.base)
    direct = SDPX._product_hsd_verified_result(
        replay, similar(solved.x), similar(solved.s), similar(solved.y), 1e-8,
        SDPX.ProductHSDVerifiedAcceptedStep, SDPX.HSDStepOK,
    )
    @test direct !== nothing
    @test direct.status === SDPX.ProductHSDOptimal
end

@testset "P1.5 ray unit normalization uses certificate tolerance" begin
    value = -1.0 + 1e-8
    @test SDPX._certificate_unit_normalization_ok(value, 2e-8)
    @test !SDPX._certificate_unit_normalization_ok(value, 5e-9)
    @test !SDPX._certificate_unit_normalization_ok(Inf, 1.0)
end

@testset "P1.5 RSOC transform is compact, precise, and alias-safe" begin
    transform = SDPX.RotatedSOCToSOC(4, Float32)
    @test transform.precision_bits == precision(Float32)
    @test transform.pairing_scale === one(Float32)
    @test !any(fieldtype(typeof(transform), index) <: AbstractMatrix for
              index in 1:fieldcount(typeof(transform)))

    source = Float32[2, 8, 3, -4]
    expected = similar(source)
    SDPX.forward_primal!(transform, expected, source)
    in_place = copy(source)
    @test SDPX.forward_primal!(transform, in_place, in_place) === in_place
    @test in_place == expected
    SDPX.backward_primal!(transform, in_place, in_place)
    @test isapprox(in_place, source; atol=8eps(Float32), rtol=8eps(Float32))
end

@testset "P1.5 tiny-tau dkappa recovery is residual-selected" begin
    consistent = _p15_exp_state().base
    fill!(consistent.dx, 0.0)
    fill!(consistent.dy, 0.0)
    consistent.rG = -2.0
    consistent.tau = 1e-20
    consistent.kappa = 1.0
    consistent.dtau = 0.5
    scalar_rhs = consistent.kappa * consistent.dtau +
                 consistent.tau * 2.0
    @test SDPX._product_hsd_recover_dkappa!(consistent, scalar_rhs)
    @test consistent.dkappa == 2.0

    inconsistent = _p15_exp_state().base
    fill!(inconsistent.dx, 0.0)
    fill!(inconsistent.dy, 0.0)
    inconsistent.rG = -2.0
    inconsistent.tau = 1e-20
    inconsistent.kappa = 1.0
    inconsistent.dtau = 0.5
    @test !SDPX._product_hsd_recover_dkappa!(inconsistent, 0.501)
end

@testset "P1.5 verified terminal result owns its copied data" begin
    state = _p15_exp_state()
    marker = (:immutable_verified_result, [1.0])
    returned = SDPX._product_hsd_finish_terminal_restore!(state, marker, false)
    @test returned === marker
    @test !state.runtime.valid
    @test state.diagnostic === :post_result_state_restore_failed
end
