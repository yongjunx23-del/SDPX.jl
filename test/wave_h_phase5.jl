using SDPX
using Test
using SparseArrays
using LinearAlgebra

function _wave_h_nonnegative_program()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :lower, x[1], SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    return SDPX.canonicalize(SDPX.compile_product_cone_model(model))
end

@testset "Wave H route storage ownership" begin
    state = SDPX.HSDState(_wave_h_nonnegative_program())
    mathematical_fields = fieldnames(typeof(state))
    for route_field in (
        :Ad, :Ar, :Atr, :rank_basis, :rank_null_objective,
        :rank_ambiguous, :rank_incompatible, :rank_ray, :H, :rhs,
        :qr, :rvec, :u, :w, :dxr, :driver,
    )
        @test route_field ∉ mathematical_fields
        @test hasfield(typeof(state.workspace), route_field)
    end
    @test state.Ar === state.workspace.Ar
    @test state.rank_basis === state.workspace.rank_basis
    @test state.H === state.workspace.H
end

function _wave_h_mixed_model()
    model = SDPX.Model(Float64)
    t = SDPX.variable!(model, :t, 1; domain=SDPX.Reals())
    M = SDPX.variable!(model, :M, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :link, M[1, 1] - t[1], SDPX.ZeroCone())
    SDPX.constraint!(model, :upper,
        [1.0 - M[1, 1] -M[1, 2]; -M[1, 2] 1.0 - M[2, 2]],
        SDPX.PSDCone())
    SDPX.objective!(model, SDPX.Maximize(), t[1])
    return model
end

@testset "Wave H frozen Ruiz opt-in" begin
    @test SDPX.Settings{Float64}().equilibration === :off
    @test SDPX.Settings{Float64}(equilibration=:ruiz).equilibration === :ruiz
    @test_throws ArgumentError SDPX.Settings{Float64}(
        scaling=:none, equilibration=:ruiz,
    )

    model = _wave_h_mixed_model()
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    reduced = SDPX.hsd_equality_reduce(canonical).reduced
    map = SDPX.equilibrate(reduced)
    Ahat, _, _ = SDPX.apply_equilibration(map, reduced)
    @test cond(Matrix(Ahat)) < cond(Matrix(reduced.A))

    for equilibration in (:off, :ruiz)
        result = SDPX.optimize!(model; settings=SDPX.Settings{Float64}(
            engine=:native_hsd, equilibration=equilibration, verbosity=0,
        ))
        @test SDPX.status(result) === :optimal
        @test result.certificate.valid
        @test isapprox(SDPX.primal_objective(result), 1.0; atol=2e-6)
    end
end

@testset "Wave H expanded factor fallback preserves iterate" begin
    model = _wave_h_mixed_model()
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    reduced = SDPX.hsd_equality_reduce(canonical).reduced
    state = SDPX.ProductConeHSDState(reduced; kkt_route=:expanded)
    SDPX.kkt_derived_start!(state)
    SDPX.hsd_residual!(state.base)
    @test SDPX.try_update_scaling!(
        state.runtime, state.base.s, state.base.y, state.base.mu,
    )
    state.base.epoch += 1
    before = (
        x=copy(state.base.x), y=copy(state.base.y), s=copy(state.base.s),
        tau=state.base.tau, kappa=state.base.kappa,
    )
    state.expanded.status = SDPX.EXPANDED_KKT_FACTOR_FAILED
    @test SDPX._product_hsd_expanded_fallback_allowed(state)
    code = SDPX._product_hsd_retry_bordered_same_iterate!(state, false)
    @test code === SDPX.HSDStepOK
    @test state.kkt_route === :bordered
    @test state.diagnostic === :expanded_to_bordered_same_iterate_fallback
    @test state.base.x == before.x
    @test state.base.y == before.y
    @test state.base.s == before.s
    @test state.base.tau == before.tau
    @test state.base.kappa == before.kappa
end

@testset "Wave H typed public exhaustion statuses" begin
    model = _wave_h_mixed_model()
    iteration_limited = SDPX.optimize!(model; settings=SDPX.Settings{Float64}(
        engine=:native_hsd,
        limits=SDPX.Limits(iterations=1, time=Inf, threads=1),
        verbosity=0,
    ))
    @test iteration_limited.status === SDPX.IterLimit
    @test SDPX.status(iteration_limited) === :iteration_limit
    @test !iteration_limited.certificate.valid

    time_limited = SDPX.optimize!(model; settings=SDPX.Settings{Float64}(
        engine=:native_hsd,
        limits=SDPX.Limits(iterations=400, time=0.0, threads=1),
        verbosity=0,
    ))
    @test time_limited.status === SDPX.TimeLimit
    @test SDPX.status(time_limited) === :time_limit
    @test !time_limited.certificate.valid
end
