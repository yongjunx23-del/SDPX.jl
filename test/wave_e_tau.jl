using SDPX
using Test
using LinearAlgebra

function _we_mixed_free_psd()
    model = SDPX.Model(Float64)
    t = SDPX.variable!(model, :t, 1; domain=SDPX.Reals())
    M = SDPX.variable!(model, :M, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :link, M[1, 1] - t[1], SDPX.ZeroCone())
    SDPX.constraint!(
        model, :upper_psd,
        [1.0 - M[1, 1] -M[1, 2]; -M[1, 2] 1.0 - M[2, 2]],
        SDPX.PSDCone(),
    )
    SDPX.objective!(model, SDPX.Maximize(), t[1])
    return model
end

function _we_bounded_nonpositive()
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :sum, y[1] + y[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :lower_one, y[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :lower_two, y[2], SDPX.Nonnegative())
    SDPX.constraint!(model, :upper_one, y[1] - 1.0, SDPX.Nonpositive())
    SDPX.constraint!(model, :upper_two, y[2] - 1.0, SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Maximize(), y[1] + 2.0 * y[2])
    return model
end

function _we_reduced(model)
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    reduction = SDPX.hsd_equality_reduce(canonical)
    return reduction.reduced::SDPX.CanonicalConicProgram{Float64}
end

function _we_settings()
    return SDPX.Settings{Float64}(
        engine=:native_hsd,
        kkt_route=:expanded,
        tolerances=SDPX.Tolerances{Float64}(
            primal=1e-8, dual=1e-8, gap=1e-8,
        ),
        limits=SDPX.Limits(iterations=100, time=60.0, threads=1),
        verbosity=0,
    )
end

@testset "Wave E tau-collapse recovery" begin
    probes = (
        ("mixed free/equality/PSD", _we_mixed_free_psd, 1.0),
        ("bounded Nonpositive", _we_bounded_nonpositive, 2.0),
    )

    @testset "gap-equation audit and scalar recenter" begin
        for (_, build, _) in probes
            state = SDPX.ProductConeHSDState(
                _we_reduced(build()); kkt_route=:expanded,
            )
            report = SDPX.kkt_derived_start!(state)
            @test report.ok
            base = state.base
            SDPX.hsd_residual!(base)
            # Frozen exact terms:
            # s'y - tau*kappa = -x'rD + y'rP - tau*rG.
            lhs = dot(base.s, base.y) - base.tau * base.kappa
            rhs = -dot(base.x, base.rD) + dot(base.y, base.rP) -
                  base.tau * base.rG
            @test lhs ≈ rhs atol=256eps(Float64) * max(1.0, abs(lhs), abs(rhs))
            observed_mu = base.mu
            @test SDPX._product_hsd_tau_collapse_recenter!(state)
            @test state.tau_collapse_recoveries == 1
            @test state.diagnostic === :tau_collapse_recentered
            @test state.base.tau * state.base.kappa ≈ observed_mu rtol=64eps(Float64)
        end
    end

    @testset "expanded probes close with authoritative certificates" begin
        for (name, build, expected) in probes
            @testset "$name" begin
                reduced = _we_reduced(build())
                state = SDPX.ProductConeHSDState(reduced; kkt_route=:expanded)
                product = SDPX.product_hsd_solve!(
                    state; max_iterations=100, tol=1e-8,
                )
                @test product.status === SDPX.ProductHSDOptimal
                @test product.reason === SDPX.ProductHSDVerifiedAcceptedStep
                @test product.tau_collapse_recoveries == 1
                @test isfinite(product.normalized_residual)
                @test product.normalized_residual <= 1e-8

                public_result = SDPX.optimize!(build(); settings=_we_settings())
                @test SDPX.status(public_result) === :optimal
                @test SDPX.certificate(public_result).valid
                @test isapprox(
                    SDPX.primal_objective(public_result), expected; atol=2e-6,
                )
            end
        end
    end

    @testset "stable recovered residual and arithmetic floor" begin
        state = SDPX.ProductConeHSDState(
            _we_reduced(_we_bounded_nonpositive()); kkt_route=:expanded,
        )
        @test SDPX.kkt_derived_start!(state).ok
        base = state.base
        SDPX.hsd_residual!(base)
        healthy = SDPX.hsd_normalized_residual(base) / base.tau
        @test SDPX._product_hsd_recovered_residual(base) == healthy
        base.tau = eps(Float64)
        @test SDPX._product_hsd_recovered_residual(base) == Inf
    end

    @testset "exhausted recovery is typed insufficient precision" begin
        for (_, build, _) in probes
            state = SDPX.ProductConeHSDState(
                _we_reduced(build()); kkt_route=:expanded,
            )
            product = SDPX.product_hsd_solve!(
                state;
                max_iterations=100,
                tol=1e-8,
                max_tau_collapse_recoveries=0,
            )
            @test product.status === SDPX.ProductHSDInsufficientPrecision
            @test product.reason === SDPX.ProductHSDTauCollapseRecoveryExhausted
            @test product.iterations < 100
        end
        @test SDPX._native_hsd_product_status(
            SDPX.ProductHSDInsufficientPrecision,
        ) === SDPX.InsufficientPrecision
        @test SDPX._native_hsd_product_reason(
            SDPX.ProductHSDTauCollapseRecoveryExhausted,
        ) === :tau_collapse_recovery_exhausted
    end
end
