# R0-P4 internal production adapter on the canonical power_epigraph_small data.
#
# The canonical problem is the real 12-row form used by
# validation/scientific_core/power_runtime/run_benchmark_power_loop.jl
# (orthant rows 1:3 + three power blocks; A = sparse([1,4,2,7,3,10],[1,1,2,2,3,3],-1,12,3),
# b[3i+2]=1, b[3i+3]=a_i, layout=Layout(3,(0.5,0.5,0.5))).  The expected
# terminal quantities are the recorded certified-loop results (27 accepted
# steps).  This test asserts the adapter reproduces them through `NP.epoch`
# admission, typed refusals, next-epoch readiness and the ordinary terminal
# audit -- no dense metric, no fallback, no tolerance change.

@testset "R0-P4 factor-pair HSD adapter: canonical power terminal" begin
    FPH = SDPX.FactorPairHSD
    a = (0.626678964309454, 0.3230223181314613, -0.7919401216799509)
    A = sparse([1, 4, 2, 7, 3, 10], [1, 1, 2, 2, 3, 3],
        fill(-1.0, 6), 12, 3)
    b = zeros(12)
    for (i, v) in enumerate(a)
        b[3i + 2] = 1.0
        b[3i + 3] = v
    end
    c = ones(3)
    layout = SDPX.NativeHalfPair.Layout(3, (0.5, 0.5, 0.5))

    st = FPH.cold_start(A, b, c, layout;
        settings = SDPX.NativeHalfPair.RootSettings(), target = 1.0e-8)
    @test st.iterations == 0
    @test SDPX.NativeHalfPair.verify(st.pair)
    @test st.owner.anchor === st.pair
    @test !st.pair.production_admitted

    terminal = FPH.solve!(st; max_iterations = 80)
    @test terminal.status === :certified_terminal
    @test terminal.iterations == 27
    aud = terminal.audit
    @test aud.cert_ok
    @test aud.membership
    @test terminal.merit <= 1.0e-8
    @test aud.norm_resid <= st.cert_tol
    @test aud.obj_gap <= st.cert_tol * aud.gap_scale
    @test aud.mu_norm <= st.cert_tol * (1.0 + 12.0)
    @test aud.kappa_tau <= st.cert_tol * aud.gap_scale
    @test aud.obj ≈ 1.1242390972345995 rtol = 0.0 atol = 1.0e-12
    exact_obj = sum(big(0.0) + big(v)^2 for v in a)
    @test abs(aud.obj - Float64(exact_obj)) <= 1.0e-8
    # Recorded certified-loop terminal quantities (R0-P3 closure-5).  The
    # reported merit 2.8017973855476926e-9 is the audit residual on recovered
    # original-scale coordinates (`aud.m`), which is the invariant quantity;
    # the raw homogeneous trial merit is <= the 1e-8 arithmetic target.
    @test aud.m ≈ 2.8017973855476926e-9 rtol = 1.0e-6
    @test aud.homo_gap ≈ 2.8017973855476926e-9 rtol = 1.0e-6
    @test aud.obj_gap ≈ 2.243317531736011e-9 rtol = 1.0e-4
    @test aud.mu_norm ≈ 5.588893961926076e-10 rtol = 1.0e-4
    @test aud.norm_resid ≈ 7.004493463869232e-10 rtol = 1.0e-4
    @test aud.kappa_tau ≈ 5.584798538116816e-10 rtol = 1.0e-4
    @test aud.sNy ≈ 6.707082406966896e-9 rtol = 1.0e-4
    # Every committed step carries the unchanged progress gate.
    @test length(terminal.history) == 27
    @test all(h -> isfinite(h.alpha) && h.alpha > 0.0, terminal.history)
    @test any(h -> h.alpha >= FPH.PROG_FLOOR, terminal.history)
    @test last(terminal.history).generation == 27

    # Typed refusal, not a silent fallback: an out-of-domain epoch input.
    bad = FPH.FactorPairState(st.A, st.b, st.c, st.layout, st.settings,
        st.owner, st.pair, st.x, 0.0, st.kappa, st.rP, st.rD, st.rG, 0,
        SDPX.FactorPairHSD.AcceptedFactorPairStep[], st.target, st.cert_tol, 0)
    err = try
        FPH.step!(bad)
        nothing
    catch caught
        caught
    end
    @test err isa FPH.FactorPairNumericalRefusal
    @test err.stage === :epoch
end
