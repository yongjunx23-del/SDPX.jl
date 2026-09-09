# R0-P4 internal production adapter on the canonical power_epigraph_small data.
#
# The canonical problem is the real 12-row form used by
# validation/scientific_core/power_runtime/run_benchmark_power_loop.jl
# (orthant rows 1:3 + three power blocks; A = sparse([1,4,2,7,3,10],[1,1,2,2,3,3],-1,12,3),
# b[3i+2]=1, b[3i+3]=a_i, layout=Layout(3,(0.5,0.5,0.5))).  The expected
# terminal quantities are recorded from this adapter (23 accepted steps).  The
# predictor boundary is a deliberately stabilized (algebraically equivalent,
# numerically different) implementation of the reviewed policy: the source
# quadratic formula loses its small root under cancellation, so the internal
# adapter adds a cancellation-safe positive root and explicit primal/dual
# coordinate positivity.  The validation modules keep the source-faithful
# formula; the namespace differential test still proves the ported arithmetic is
# bit-identical.

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

    if !SDPX.FactorPreservingAffine.RG.Phi._runtime_ok()
        # Unsupported arithmetic context: the reviewed kernels must refuse the
        # cold point (truthful behavior), never silently substitute arithmetic.
        @test_throws FPH.FactorPairNumericalRefusal FPH.cold_start(A, b, c,
            layout; settings = SDPX.NativeHalfPair.RootSettings(), target = 1.0e-8)
        return
    end
    st = FPH.cold_start(A, b, c, layout;
        settings = SDPX.NativeHalfPair.RootSettings(), target = 1.0e-8)
    @test st.iterations == 0
    @test SDPX.NativeHalfPair.verify(st.pair)
    @test st.owner.anchor === st.pair
    @test !st.pair.production_admitted

    terminal = FPH.solve!(st; max_iterations = 80)
    @test terminal.status === :certified_terminal
    @test terminal.iterations == 23
    aud = terminal.audit
    @test aud.cert_ok
    @test aud.membership
    @test terminal.merit <= 1.0e-8
    @test aud.norm_resid <= st.cert_tol
    @test aud.obj_gap <= st.cert_tol * aud.gap_scale
    @test aud.mu_norm <= st.cert_tol * (1.0 + 12.0)
    @test aud.kappa_tau <= st.cert_tol * aud.gap_scale
    @test aud.obj ≈ 1.124239097167378 rtol = 0.0 atol = 1.0e-12
    # Exact dyadic-rational reference of the stored Float64 inputs (not an
    # ambient-precision big() square).
    exact_obj = sum(Rational{BigInt}(v)^2 for v in a)
    @test abs(aud.obj - Float64(exact_obj)) <= 1.0e-8
    # Recorded certified-loop terminal quantities (R0-P3 closure-5).  The
    # reported merit 2.8017973855476926e-9 is the audit residual on recovered
    # original-scale coordinates (`aud.m`), which is the invariant quantity;
    # the raw homogeneous trial merit is <= the 1e-8 arithmetic target.
    @test aud.m ≈ 2.9174020235019777e-9 rtol = 1.0e-6
    @test aud.homo_gap ≈ 2.9174020235019777e-9 rtol = 1.0e-6
    @test aud.obj_gap ≈ 2.3358788237004546e-9 rtol = 1.0e-4
    @test aud.mu_norm ≈ 5.81949619632288e-10 rtol = 1.0e-4
    @test aud.norm_resid ≈ 7.293505058754944e-10 rtol = 1.0e-4
    @test aud.kappa_tau ≈ 5.815231998015231e-10 rtol = 1.0e-4
    @test aud.sNy ≈ 6.98382190908772e-9 rtol = 1.0e-4
    # Every committed step carries the unchanged progress gate.
    @test length(terminal.history) == 23
    @test all(h -> isfinite(h.alpha) && h.alpha > 0.0, terminal.history)
    @test any(h -> h.alpha >= FPH.PROG_FLOOR, terminal.history)
    @test last(terminal.history).generation == 23

    # Simultaneous-live memory admission: an undersized declared budget must
    # refuse before any epoch/factor allocation, and the estimate must cover
    # two live epochs (accepted + prepared next).
    estimate = FPH.estimate_live_bytes(A, b, c, layout, 2)
    @test estimate > 0
    @test FPH.estimate_live_bytes(A, b, c, layout, 2) >
          FPH.estimate_live_bytes(A, b, c, layout, 1)
    mem_err = try
        FPH.cold_start(A, b, c, layout; target = 1.0e-8,
            memory_limit_bytes = estimate - 1)
        nothing
    catch caught
        caught
    end
    @test mem_err isa FPH.FactorPairNumericalRefusal
    @test mem_err.stage === :memory && mem_err.reason === :insufficient_budget
    @test st.memory_estimate_bytes == estimate
    @test st.memory_limit_bytes === nothing

    # Time limit is honoured between accepted steps and never publishes a
    # public Optimal status.
    timed = FPH.cold_start(A, b, c, layout; target = 1.0e-8,
        max_time_seconds = 0.0) # deterministic; a 1 ns budget depends on clock resolution
    timed_terminal = FPH.solve!(timed; max_iterations = 5)
    @test timed_terminal.status === :time_limit
    @test timed_terminal.iterations == 0

    # Transaction atomicity: a refused step leaves anchor, generation, point
    # and iteration count unchanged.
    anchor_before = st.owner.anchor
    gen_before = st.owner.generation
    iters_before = st.iterations
    pair_before = st.pair
    x_before = copy(st.x)
    refused = try
        FPH.step!(st; sigma_override = floatmax(Float64))
        nothing
    catch caught
        caught
    end
    @test refused isa FPH.FactorPairNumericalRefusal
    @test st.owner.anchor === anchor_before
    @test st.owner.generation == gen_before
    @test st.iterations == iters_before
    @test st.pair === pair_before
    @test st.x == x_before

    # Post-reduction admission adapter: the real canonical/equality-reduction
    # pipeline for an admitted orthant+half-Power model must yield the exact
    # factor-pair layout, and every excluded shape must return `nothing`
    # (typed refusal by the caller), never a guessed permutation.
    model = SDPX.Model(Float64)
    sig = SDPX.variable!(model, :signal, 3; domain=SDPX.Reals())
    epi = SDPX.variable!(model, :epigraph, 3; domain=SDPX.Nonnegative())
    for i in 1:3
        SDPX.constraint!(model, Symbol(:fix_, i), sig[i] - a[i], SDPX.ZeroCone())
        SDPX.constraint!(model, Symbol(:term_, i), (epi[i], 1.0, sig[i]),
            SDPX.PowerCone(0.5))
    end
    SDPX.objective!(model, SDPX.Minimize(), epi[1] + epi[2] + epi[3])
    program = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(program)
    red = SDPX.hsd_equality_reduce(canon)
    layout_adm = FPH.reduced_layout(red.reduced)
    @test layout_adm isa SDPX.NativeHalfPair.Layout
    @test layout_adm.orthant == 3
    @test layout_adm.alphas == (0.5, 0.5, 0.5)
    Ared, bred, cred = FPH.canonical_problem(red.reduced)
    @test size(Ared, 1) == 12 && size(Ared, 2) == 3
    @test cred == ones(3)
    # Excluded shapes refuse instead of guessing: an SOC model and a model
    # without a Power block.
    soc_model = SDPX.Model(Float64)
    z = SDPX.variable!(soc_model, :z, 3; domain=SDPX.Reals())
    SDPX.constraint!(soc_model, :soc, Any[1.0, z[1], z[2]], SDPX.LorentzCone())
    SDPX.objective!(soc_model, SDPX.Minimize(), z[1] + z[3])
    soc_program = SDPX.compile_product_cone_model(soc_model)
    soc_red = SDPX.hsd_equality_reduce(SDPX.canonicalize(soc_program))
    @test FPH.reduced_layout(soc_red.reduced) === nothing
    lp_model = SDPX.Model(Float64)
    w = SDPX.variable!(lp_model, :w, 2; domain=SDPX.Reals())
    SDPX.constraint!(lp_model, :lo, w[1], SDPX.Nonnegative())
    SDPX.constraint!(lp_model, :hi, 1.0 - w[2], SDPX.Nonnegative())
    SDPX.objective!(lp_model, SDPX.Minimize(), w[1] + w[2])
    lp_program = SDPX.compile_product_cone_model(lp_model)
    lp_red = SDPX.hsd_equality_reduce(SDPX.canonicalize(lp_program))
    @test FPH.reduced_layout(lp_red.reduced) === nothing
    @test FPH.reduced_layout(nothing) === nothing

    # Boundary qualification (independent review counterexample): the source
    # quadratic formula returned Inf for a finite exit; the stabilized
    # positive root plus explicit coordinate positivity must not.
    @test FPH._min_positive_root(1.0, -1.0e16, 1.0) <= 1.0e-16
    @test FPH._power_boundary(1.0, 1.0, 0.0, -1.0e16, -1.0e-16, 0.0) <= 1.0e-16
    @test isfinite(FPH._dual_power_boundary(1.0, 1.0, 0.0, -1.0e16, -1.0e-16, 0.0))
    @test FPH._power_boundary(1.0, 1.0, 0.0, -1.0, -1.0, 0.0) ≈ 1.0 rtol = 1.0e-12  # (1-alpha)^2

    # Typed refusal, not a silent fallback: an out-of-domain epoch input.
    bad = FPH.FactorPairState(st.A, st.b, st.c, st.layout, st.settings,
        st.owner, st.pair, st.x, 0.0, st.kappa, st.rP, st.rD, st.rG, 0,
        SDPX.FactorPairHSD.AcceptedFactorPairStep[], st.target, st.cert_tol, 0,
        nothing, 0, nothing, nothing, 0, Inf, SDPX.FactorPreservingAffine.FactorizationLedger())
    err = try
        FPH.step!(bad)
        nothing
    catch caught
        caught
    end
    @test err isa FPH.FactorPairNumericalRefusal
    @test err.stage === :epoch

    # Numerical-failure translation: an overflowing combined RHS must surface
    # as the adapter's typed refusal, never as a raw FactorSeamNumericalFailure
    # or a generic ErrorException.
    overflow_err = try
        FPH.step!(st; sigma_override = floatmax(Float64))
        nothing
    catch caught
        caught
    end
    @test overflow_err isa FPH.FactorPairNumericalRefusal
    @test overflow_err.stage in (:combined, :combined_solve, :combined_certificate)

    # Direct FA typed refusal (declared delta from the validation reference):
    # a nonfinite RHS is a typed stage refusal, not a generic error.
    FA = SDPX.FactorPreservingAffine
    e = SDPX.NativeHalfPair.epoch(st.pair, st.A, st.b, st.c, st.x, st.tau,
        st.kappa).epoch
    badrhs = SDPX.HSDNewtonRHS(fill(NaN, 12), zeros(3), 0.0, -copy(e.s), -1.0)
    rhs_err = try
        FA.solve(e, badrhs)
        nothing
    catch caught
        caught
    end
    @test rhs_err isa FA.FactorPairStageRefusal
    @test rhs_err.reason === :nonfinite_rhs
end
