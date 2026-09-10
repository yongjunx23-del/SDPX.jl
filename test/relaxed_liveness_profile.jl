# Opt-in relaxing liveness profile: default contract unchanged, relaxation
# recorded, and no acceptance gate altered.
using Test
using SDPX

function _relaxed_profile_model()
    model = SDPX.Model(Float64; name="relaxed_profile")
    x = SDPX.variable!(model, :x, 3; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :sum, sum(x) - 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1] - 0.5 * x[2])
    return model
end

function _solve_profile(relaxed::Bool)
    settings = SDPX.Settings{Float64}(
        verbosity=0, diagnostics=:full, certification=true,
        relaxed_liveness=relaxed,
    )
    result = SDPX.optimize!(_relaxed_profile_model(); settings=settings)
    return result, SDPX.diagnostics(result)
end

@testset "relaxing liveness profile" begin
    # Off by default, and the record reports it as off.
    base = SDPX.Settings{Float64}(verbosity=0)
    @test base.relaxed_liveness === false
    default_result, default_diag = _solve_profile(false)
    @test default_diag.timings.relaxed_liveness == 0

    # Opt-in, recorded, and it does not change this problem's answer or
    # certificate: the relaxation is a liveness gate, never an acceptance gate.
    relaxed_settings = SDPX.Settings{Float64}(verbosity=0, relaxed_liveness=true)
    @test relaxed_settings.relaxed_liveness === true
    relaxed_result, relaxed_diag = _solve_profile(true)
    @test relaxed_diag.timings.relaxed_liveness == 1
    @test SDPX.status(relaxed_result) === SDPX.status(default_result)
    @test SDPX.status(relaxed_result) === :optimal
    @test SDPX.certificate(relaxed_result).valid ===
          SDPX.certificate(default_result).valid
    @test SDPX.certificate(relaxed_result).valid
    @test SDPX.certificate(relaxed_result).primal_objective ==
          SDPX.certificate(default_result).primal_objective
    @test SDPX.iterations(relaxed_result) == SDPX.iterations(default_result)

    # The flag is a plain Bool field: no other setting changes with it.
    for field in (:tolerances, :limits, :kkt_route, :formulation, :provider,
                  :scaling, :certification, :diagnostics)
        @test getfield(base, field) == getfield(relaxed_settings, field)
    end
end
