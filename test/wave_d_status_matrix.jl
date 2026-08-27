using SDPX
using Test

const _WDD_TOL = 1.0e-7

function _wdd_settings(route::Symbol)
    return SDPX.Settings{Float64}(
        engine=:native_hsd,
        kkt_route=route,
        tolerances=SDPX.Tolerances{Float64}(
            primal=_WDD_TOL,
            dual=_WDD_TOL,
            gap=_WDD_TOL,
        ),
        limits=SDPX.Limits(iterations=400, time=60.0, threads=1),
        verbosity=0,
    )
end

function _wdd_outputs()
    return SDPX.Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
    )
end

"""A one-dimensional recession direction represented in each native cone."""
function _wdd_dual_infeasible_model(kind::Symbol)
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    t = x[1]
    if kind === :nonnegative
        SDPX.constraint!(model, :recession, [2t, 2t], SDPX.Nonnegative())
    elseif kind === :soc
        SDPX.constraint!(model, :recession, [2t, 0t, 0t], SDPX.LorentzCone())
    elseif kind === :psd
        SDPX.constraint!(
            model, :recession, [2t 0t; 0t 2t], SDPX.PSDCone(),
        )
    elseif kind === :exp
        # (0, 2t, 4t) is in K_exp for every t >= 0.
        SDPX.constraint!(
            model, :recession, (0t, 2t, 4t), SDPX.ExponentialCone(),
        )
    elseif kind === :power
        # (2t, 2t, 0) is in K_power(1/2) for every t >= 0.
        SDPX.constraint!(
            model, :recession, (2t, 2t, 0t), SDPX.PowerCone(0.5),
        )
    elseif kind !== :free
        throw(ArgumentError("unknown Wave-D cone $kind"))
    end
    SDPX.objective!(model, SDPX.Minimize(), -t)
    return model
end

@testset "Wave D cone × DualInfeasible public status matrix" begin
    # :free is the genuinely unbounded P1.5 probe. The remaining rows certify
    # the same recession semantics through each native numerical cone family.
    for kind in (:free, :nonnegative, :soc, :psd, :exp, :power)
        @testset "$kind" begin
            for route in (:bordered, :expanded)
                @testset "$route" begin
                    result = SDPX.optimize!(
                        _wdd_dual_infeasible_model(kind);
                        settings=_wdd_settings(route),
                        outputs=_wdd_outputs(),
                    )
                    certificate = SDPX.certificate(result)
                    ray = SDPX.value(result)

                    @test SDPX.status(result) === :dual_infeasible
                    @test result.status === SDPX.DualInfeasible
                    @test certificate.available
                    @test certificate.valid
                    @test certificate.method ===
                          :original_coordinate_dual_infeasibility_ray
                    @test certificate.reason === :valid
                    @test certificate.primal_residual_scaled <=
                          certificate.primal_limit
                    @test ray[1] > 0.0
                    @test -ray[1] < -certificate.primal_limit
                    @test all(iszero, SDPX.dual(result))
                    @test all(iszero, SDPX.dual_slack(result))
                    @test result.termination.stage ===
                          :original_coordinate_certification
                end
            end
        end
    end
end

@testset "Wave D terminal failures verify a finite dual ray first" begin
    model = _wdd_dual_infeasible_model(:nonnegative)
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    reduced = SDPX.hsd_equality_reduce(canonical).reduced
    @test reduced !== nothing

    state = SDPX.ProductConeHSDState(reduced; kkt_route=:bordered)
    state.base.x[1] = 1.0
    state.base.tau = 0.0
    state.base.kappa = 1.0
    @test SDPX._product_hsd_tau_collapsed(state.base, _WDD_TOL)

    promoted = SDPX._product_hsd_termination_or_dual_ray!(
        state,
        zeros(Float64, state.base.n),
        zeros(Float64, state.base.m),
        zeros(Float64, state.base.m),
        _WDD_TOL,
        SDPX.ProductHSDSingular,
        SDPX.ProductHSDSingularKKTReason,
        SDPX.HSDStepSingularKKT,
    )
    @test promoted.status === SDPX.ProductHSDDualInfeasible
    @test promoted.reason === SDPX.ProductHSDVerifiedTerminationRay
    @test promoted.x == [1.0]
    @test all(isfinite, promoted.s)

    rejected_state = SDPX.ProductConeHSDState(reduced; kkt_route=:bordered)
    rejected_state.base.x[1] = -1.0
    rejected = SDPX._product_hsd_termination_or_dual_ray!(
        rejected_state,
        zeros(Float64, rejected_state.base.n),
        zeros(Float64, rejected_state.base.m),
        zeros(Float64, rejected_state.base.m),
        _WDD_TOL,
        SDPX.ProductHSDSingular,
        SDPX.ProductHSDSingularKKTReason,
        SDPX.HSDStepSingularKKT,
    )
    @test rejected.status === SDPX.ProductHSDSingular
    @test rejected.reason === SDPX.ProductHSDSingularKKTReason
end
