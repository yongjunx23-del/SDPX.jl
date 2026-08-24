using Test

@testset "typed public Result and pure LP optimize" begin
    model = SDPX.Model(Float64; name="typed-lp")
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
    lower = SDPX.constraint!(model, :lower, x[1] - 1, SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), x[1])

    outputs = SDPX.Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
        history=true,
        trace=true,
    )
    settings = SDPX.Settings{Float64}(
        limits=SDPX.Limits(iterations=80, time=30.0, threads=1),
        diagnostics=:summary,
        certification=false, # public optimize still runs the original cert gate
        verbosity=0,
    )
    result = SDPX.optimize!(model; settings=settings, outputs=outputs)

    @test result isa SDPX.Result{Float64}
    @test SDPX.status(result) === :optimal
    @test SDPX.execution_plan(result) === SDPX.diagnostics(result).plan
    @test SDPX.value(result, x)[1] ≈ 1.0 atol=1e-6
    @test SDPX.value(result, x[1]) ≈ 1.0 atol=1e-6
    @test SDPX.dual(result, lower)[1] ≈ 1.0 atol=1e-6
    @test SDPX.dual(result, lower[1]) ≈ 1.0 atol=1e-6
    @test SDPX.dual_slack(result, x)[1] ≥ -1e-8
    @test SDPX.primal_objective(result) ≈ 1.0 atol=1e-6
    @test SDPX.dual_objective(result) ≈ 1.0 atol=1e-6
    @test SDPX.certificate(result).available
    @test SDPX.certificate(result).valid
    @test SDPX.iteration_history(result) isa Vector{NamedTuple}
    @test SDPX.performance_trace(result) isa SDPX.PerformanceTrace

    # The plan, terminal facts, and certificate summary remain available even
    # when every optional payload is intentionally hidden.
    hidden = SDPX.Outputs(
        :none,
        :none,
        :none;
        objectives=false,
        certificate=:none,
        diagnostics=:none,
        history=false,
        trace=false,
    )
    hidden_result = SDPX.optimize!(model; settings=settings, outputs=hidden)
    @test hidden_result.execution_plan isa SDPX.ExecutionPlan
    @test hidden_result.status == SDPX.Optimal
    @test hidden_result.termination.status == SDPX.Optimal
    @test hidden_result.certificate.available
    @test_throws SDPX.ResultFieldNotRetained SDPX.value(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.dual(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.dual_slack(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.primal_objective(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.dual_objective(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.diagnostics(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.iteration_history(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.performance_trace(hidden_result)

    for name in fieldnames(SDPX.Result{Float64})
        @test fieldtype(SDPX.Result{Float64}, name) !== Any
    end
end

@testset "native family optimize routes" begin
    model = SDPX.Model(Float64)
    q = SDPX.variable!(model, :q, 2; domain=SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), q[1])
    result = SDPX.optimize!(
        model;
        settings=SDPX.Settings{Float64}(
            limits=SDPX.Limits(iterations=80, time=30.0, threads=1),
            verbosity=0,
        ),
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test result.status == SDPX.Optimal
    @test SDPX.value(result, q)[1] ≥ -1e-8
    @test SDPX.certificate(result).valid

    psd_model = SDPX.Model(Float64)
    X = SDPX.variable!(psd_model, :X, 1, 1; domain=SDPX.PSDCone())
    SDPX.objective!(psd_model, SDPX.Minimize(), X[1, 1])
    result = SDPX.optimize!(
        psd_model;
        settings=SDPX.Settings{Float64}(
            limits=SDPX.Limits(iterations=80, time=30.0, threads=1),
            verbosity=0,
        ),
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test result.status == SDPX.Optimal
    @test SDPX.value(result, X)[1, 1] ≥ -1e-8
    @test SDPX.certificate(result).valid
end

@testset "original-sense objective and free/zero dual certificate" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
    upper = SDPX.constraint!(model, :upper, 1 - x[1], SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Maximize(), 2 * x[1] + 3)
    outputs = SDPX.Outputs(:all, :all, :all; diagnostics=:summary)
    settings = SDPX.Settings{Float64}(
        limits=SDPX.Limits(iterations=80, time=30.0, threads=1),
        verbosity=0,
    )
    result = SDPX.optimize!(model; settings=settings, outputs=outputs)
    @test result.status == SDPX.Optimal
    @test SDPX.value(result, x)[1] ≈ 1.0 atol=1e-6
    @test SDPX.primal_objective(result) ≈ 5.0 atol=1e-6
    @test SDPX.dual_objective(result) ≈ 5.0 atol=1e-6
    @test SDPX.dual(result, upper)[1] ≈ 2.0 atol=1e-6
    @test SDPX.certificate(result).valid

    # A result keeps a solved layout snapshot, not the mutable builder.
    SDPX.set_start!(x, [1.0])
    @test SDPX.value(result, x)[1] ≈ 1.0 atol=1e-6

    free_zero = SDPX.Model(Float64)
    u = SDPX.variable!(free_zero, :u, 1; domain=SDPX.Reals())
    z = SDPX.variable!(free_zero, :z, 1; domain=SDPX.ZeroCone())
    n = SDPX.variable!(free_zero, :n, 1; domain=SDPX.Nonnegative())
    equality = SDPX.constraint!(free_zero, :equality, u[1] - 2, SDPX.ZeroCone())
    SDPX.objective!(free_zero, SDPX.Minimize(), 0 * u[1] + z[1] + n[1])
    free_result = SDPX.optimize!(
        free_zero;
        settings=settings,
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test free_result.status == SDPX.Optimal
    @test SDPX.certificate(free_result).valid
    @test SDPX.dual(free_result, equality)[1] isa Float64
end

@testset "status-aware certification and mapped warm starts" begin
    # A non-optimal terminal status never receives an `Optimal` certificate,
    # even when the current iterate happens to satisfy the original residuals.
    limited = SDPX.Model(Float64)
    lx = SDPX.variable!(limited, :x, 1; domain=SDPX.Nonnegative())
    SDPX.constraint!(limited, :bound, lx[1] - 1, SDPX.Nonnegative())
    SDPX.objective!(limited, SDPX.Minimize(), lx[1])
    limited_result = SDPX.optimize!(
        limited;
        settings=SDPX.Settings{Float64}(
            limits=SDPX.Limits(iterations=1, time=30.0, threads=1),
            verbosity=0,
        ),
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test limited_result.status == SDPX.IterLimit
    @test SDPX.status(limited_result) === :iteration_limit
    @test !SDPX.certificate(limited_result).valid
    @test SDPX.certificate(limited_result).reason == :nonoptimal_status

    # LP x0 and equality-dual starts map to the dedicated core's x0/y0 entry;
    # no start is silently dropped. A loose explicit tolerance preserves the
    # established fixed/warm trajectory while making the expected endpoint
    # status deterministic.
    warmed = SDPX.Model(Float64)
    wx = SDPX.variable!(warmed, :x, 1; domain=SDPX.Nonnegative())
    SDPX.set_start!(wx, [1.0])
    equality = SDPX.constraint!(warmed, :equality, wx[1] - 1, SDPX.ZeroCone())
    SDPX.set_dual_start!(equality, [0.0])
    SDPX.constraint!(warmed, :nonnegative, wx[1], SDPX.Nonnegative())
    SDPX.objective!(warmed, SDPX.Minimize(), wx[1])
    warmed_result = SDPX.optimize!(
        warmed;
        settings=SDPX.Settings{Float64}(
            tolerances=SDPX.Tolerances{Float64}(
                primal=1e-6,
                dual=1e-6,
                gap=1e-6,
            ),
            limits=SDPX.Limits(iterations=30, time=30.0, threads=1),
            verbosity=0,
        ),
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test warmed_result.status == SDPX.Optimal
    @test SDPX.value(warmed_result, wx)[1] ≈ 1.0 atol=1e-6
    @test SDPX.certificate(warmed_result).valid

    # A complete cone-dual start maps through scalar 1×1 Y0 blocks.  The
    # preprocessing stage may remove/reorder those blocks, so this regression
    # deliberately leaves default presolve enabled.
    mapped = SDPX.Model(Float64)
    mx = SDPX.variable!(mapped, :x, 1; domain=SDPX.Nonnegative())
    lower = SDPX.constraint!(mapped, :lower, mx[1] - 1, SDPX.Nonnegative())
    SDPX.set_start!(mx, [1.5])
    SDPX.set_dual_slack_start!(mx, [1.0])
    SDPX.set_dual_start!(lower, [1.0])
    SDPX.objective!(mapped, SDPX.Minimize(), mx[1])
    mapped_result = SDPX.optimize!(
        mapped;
        settings=SDPX.Settings{Float64}(
            tolerances=SDPX.Tolerances{Float64}(
                primal=1e-6,
                dual=1e-6,
                gap=1e-6,
            ),
            limits=SDPX.Limits(iterations=60, time=30.0, threads=1),
            verbosity=0,
        ),
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test mapped_result.status == SDPX.Optimal
    @test SDPX.value(mapped_result, mx)[1] ≈ 1.0 atol=1e-6
    @test SDPX.dual(mapped_result, lower)[1] ≈ 1.0 atol=1e-6
    @test SDPX.certificate(mapped_result).valid

    # Nonpositive origins carry the lowering sign into both the affine-row
    # dual and the product variable dual-slack start.
    nonpositive = SDPX.Model(Float64)
    nx = SDPX.variable!(nonpositive, :x, 1; domain=SDPX.Nonpositive())
    upper = SDPX.constraint!(nonpositive, :upper, nx[1] + 1, SDPX.Nonpositive())
    SDPX.set_start!(nx, [-1.5])
    SDPX.set_dual_slack_start!(nx, [-1.0])
    SDPX.set_dual_start!(upper, [-1.0])
    SDPX.objective!(nonpositive, SDPX.Minimize(), -nx[1])
    nonpositive_result = SDPX.optimize!(
        nonpositive;
        settings=SDPX.Settings{Float64}(
            tolerances=SDPX.Tolerances{Float64}(
                primal=1e-6,
                dual=1e-6,
                gap=1e-6,
            ),
            limits=SDPX.Limits(iterations=60, time=30.0, threads=1),
            verbosity=0,
        ),
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test nonpositive_result.status == SDPX.Optimal
    @test SDPX.value(nonpositive_result, nx)[1] ≈ -1.0 atol=1e-6
    @test SDPX.dual(nonpositive_result, upper)[1] ≈ -1.0 atol=1e-6
    @test SDPX.dual_slack(nonpositive_result, nx)[1] ≤ 1e-8
    @test SDPX.certificate(nonpositive_result).valid

    # Partial cone-dual coverage is rejected instead of silently filling the
    # missing affine-row multiplier.
    incomplete = SDPX.Model(Float64)
    ix = SDPX.variable!(incomplete, :x, 1; domain=SDPX.Nonnegative())
    SDPX.set_dual_slack_start!(ix, [1.0])
    SDPX.constraint!(incomplete, :bound, ix[1] - 1, SDPX.Nonnegative())
    SDPX.objective!(incomplete, SDPX.Minimize(), ix[1])
    @test_throws SDPX.PublicOptimizeError SDPX.optimize!(
        incomplete;
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
end

@testset "native SDP Result continuation" begin
    # Keep the native layout fixed while changing one numerical objective
    # coefficient.  This is the public analogue of a nearby lambda step.
    function continuation_model(lambda; dimension=2)
        model = SDPX.Model(Float64)
        X = SDPX.variable!(
            model,
            :X,
            dimension,
            dimension;
            domain=SDPX.PSDCone(),
        )
        upper = if dimension == 2
            [1 - X[1, 1] -X[1, 2]; -X[1, 2] 1 - X[2, 2]]
        else
            reshape([1 - X[1, 1]], 1, 1)
        end
        SDPX.constraint!(model, :upper, upper, SDPX.PSDCone())
        objective = dimension == 2 ? lambda * X[1, 2] + 3 : lambda * X[1, 1] + 3
        SDPX.objective!(model, SDPX.Maximize(), objective)
        return model, X
    end

    settings = SDPX.Settings{Float64}(
        limits=SDPX.Limits(iterations=80, time=30.0, threads=1),
        verbosity=0,
    )
    all_outputs = SDPX.Outputs(:all, :all, :all; diagnostics=:summary)

    base, _ = continuation_model(2.0)
    source = SDPX.optimize!(base; settings=settings, outputs=all_outputs)
    @test SDPX.status(source) === :optimal
    @test SDPX.certificate(source).valid
    source_primal = SDPX.value(source)

    target, target_X = continuation_model(2.01)
    continued = SDPX.optimize!(
        target;
        settings=settings,
        outputs=all_outputs,
        warm_start=source,
    )
    @test SDPX.status(continued) === :optimal
    @test SDPX.certificate(continued).valid
    @test isfinite(SDPX.primal_objective(continued))
    @test SDPX.primal_objective(continued) ≈ 4.005 atol=1e-5
    initialization = SDPX.diagnostics(continued).selected_algorithms.initialization
    @test initialization.method === :continuation
    @test initialization.path === :warm_start
    @test initialization.applied
    @test initialization.continuation.accepted
    @test !initialization.continuation.fallback
    # The source Result is immutable from the caller's perspective; solving
    # the target must not overwrite retained source coordinates.
    @test SDPX.value(source) == source_primal
    @test size(SDPX.value(continued, target_X)) == (2, 2)

    positional_target, _ = continuation_model(2.005)
    positional = SDPX.optimize!(
        positional_target,
        settings,
        all_outputs;
        warm_start=source,
    )
    @test SDPX.status(positional) === :optimal
    @test SDPX.diagnostics(positional).selected_algorithms.initialization.method ===
          :continuation

    # Continuation centering consumes residuals in fresh equilibrated working
    # coordinates. An explicit no-scaling target therefore fails closed to
    # the normal KKT cold path instead of applying a scale-dependent seed.
    unscaled_target, _ = continuation_model(2.01)
    unscaled = SDPX.optimize!(
        unscaled_target;
        settings=SDPX.Settings{Float64}(
            limits=SDPX.Limits(iterations=80, time=30.0, threads=1),
            scaling=:none,
            verbosity=0,
        ),
        outputs=all_outputs,
        warm_start=source,
    )
    @test SDPX.status(unscaled) === :optimal
    unscaled_init = SDPX.diagnostics(unscaled).selected_algorithms.initialization
    @test unscaled_init.path === :kkt_cold_start
    @test unscaled_init.continuation.fallback
    @test !unscaled_init.continuation.accepted
    @test unscaled_init.continuation.reason ===
          :continuation_requires_equilibration
    @test any(
        warning -> occursin("continuation_requires_equilibration", warning),
        SDPX.diagnostics(unscaled).warnings,
    )

    # The low-level fixed parameter policy uses its ordinary identity seed
    # after a rejected continuation and must not report that a warm start was
    # applied. Public Settings currently selects the automatic policy, so
    # exercise this expert core boundary directly.
    fixed_program = SDPX.compile_product_cone_model(unscaled_target)
    fixed_lowering = SDPX.lower_sdp_native(
        fixed_program;
        sparse=false,
        verbosity=0,
    )
    fixed_fallback = SDPX.solve!(
        fixed_lowering.core,
        SDPX.SolverOptions{Float64}(
            parameter_policy=:fixed,
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            iter_max=80,
            max_time=30.0,
            verbosity=0,
        );
        _continuation_reason=:layout_mismatch,
    )
    fixed_init = fixed_fallback.termination.executed.initialization
    @test fixed_init.method === :identity_cold_start
    @test fixed_init.path === :cold_start
    @test !fixed_init.applied
    @test fixed_init.continuation.fallback
    @test fixed_init.continuation.reason === :layout_mismatch

    # The compiler normalizes an omitted objective to Minimize().  Continuation
    # performs the same normalization when comparing layouts.
    function feasibility_model()
        model = SDPX.Model(Float64)
        X = SDPX.variable!(model, :X, 1, 1; domain=SDPX.PSDCone())
        SDPX.constraint!(
            model,
            :bounded,
            reshape([1 - X[1, 1]], 1, 1),
            SDPX.PSDCone(),
        )
        return model
    end
    feasibility_source = SDPX.optimize!(
        feasibility_model();
        settings=settings,
        outputs=all_outputs,
    )
    feasibility_continued = SDPX.optimize!(
        feasibility_model();
        settings=settings,
        outputs=all_outputs,
        warm_start=feasibility_source,
    )
    @test SDPX.status(feasibility_continued) === :optimal
    feasibility_diagnostics = SDPX.diagnostics(feasibility_continued)
    feasibility_init = feasibility_diagnostics.selected_algorithms.initialization
    @test feasibility_init.path === :warm_start
    @test feasibility_init.applied

    # A source that did not retain all three continuation vectors is rejected
    # before the core and safely falls back to ordinary cold initialization.
    sparse_outputs = SDPX.Outputs(
        :none,
        :none,
        :none;
        objectives=false,
        certificate=:summary,
        diagnostics=:summary,
    )
    sparse_source_model, _ = continuation_model(2.0)
    sparse_source = SDPX.optimize!(
        sparse_source_model;
        settings=settings,
        outputs=sparse_outputs,
    )
    @test SDPX.status(sparse_source) === :optimal
    missing_state_model, _ = continuation_model(2.01)
    missing_state = SDPX.optimize!(
        missing_state_model;
        settings=settings,
        outputs=all_outputs,
        warm_start=sparse_source,
    )
    @test SDPX.status(missing_state) === :optimal
    missing_init = SDPX.diagnostics(missing_state).selected_algorithms.initialization
    @test missing_init.continuation.requested
    @test !missing_init.continuation.accepted
    @test missing_init.continuation.reason === :state_not_retained
    @test any(
        warning -> occursin("state_not_retained", warning),
        SDPX.diagnostics(missing_state).warnings,
    )

    # A non-optimal source is never used as a continuation point.
    limited_model, _ = continuation_model(2.0)
    limited_source = SDPX.optimize!(
        limited_model;
        settings=SDPX.Settings{Float64}(
            limits=SDPX.Limits(iterations=1, time=30.0, threads=1),
            verbosity=0,
        ),
        outputs=all_outputs,
    )
    @test SDPX.status(limited_source) !== :optimal
    nonoptimal_target, _ = continuation_model(2.01)
    nonoptimal_fallback = SDPX.optimize!(
        nonoptimal_target;
        settings=settings,
        outputs=all_outputs,
        warm_start=limited_source,
    )
    @test SDPX.status(nonoptimal_fallback) === :optimal
    nonoptimal_init = SDPX.diagnostics(nonoptimal_fallback).selected_algorithms.initialization
    @test !nonoptimal_init.continuation.accepted
    @test nonoptimal_init.continuation.reason === :source_not_optimal

    # If a source Result is corrupted through internal fields after its
    # certificate was created, the adapter still sees finite values but the
    # shifted-Cholesky preparation rejects them.  The core must then restore
    # the ordinary auto-policy KKT cold start, not the legacy identity seed.
    corrupted_model, _ = continuation_model(2.0)
    corrupted_source = SDPX.optimize!(
        corrupted_model;
        settings=settings,
        outputs=all_outputs,
    )
    fill!(corrupted_source.dual_slack_data.values, -floatmax(Float64))
    corrupted_target, _ = continuation_model(2.01)
    recovered = SDPX.optimize!(
        corrupted_target;
        settings=settings,
        outputs=all_outputs,
        warm_start=corrupted_source,
    )
    @test SDPX.status(recovered) === :optimal
    recovered_init = SDPX.diagnostics(recovered).selected_algorithms.initialization
    @test recovered_init.path === :kkt_cold_start
    @test recovered_init.continuation.fallback
    @test !recovered_init.continuation.accepted
    @test any(
        warning -> occursin("Continuation warm start was rejected", warning),
        SDPX.diagnostics(recovered).warnings,
    )

    # Retained storage is result-owned but mutable internally. If it is
    # corrupted to an impossible length, reject it before any origin slice so
    # continuation remains a safe fallback rather than a BoundsError.
    short_model, _ = continuation_model(2.0)
    short_source = SDPX.optimize!(
        short_model;
        settings=settings,
        outputs=all_outputs,
    )
    resize!(short_source.dual_slack_data.values, 1)
    short_target, _ = continuation_model(2.01)
    short_fallback = SDPX.optimize!(
        short_target;
        settings=settings,
        outputs=all_outputs,
        warm_start=short_source,
    )
    @test SDPX.status(short_fallback) === :optimal
    short_init =
        SDPX.diagnostics(short_fallback).selected_algorithms.initialization
    @test short_init.continuation.fallback
    @test !short_init.continuation.accepted
    @test short_init.continuation.reason === :dimension_mismatch

    # A different block shape is a safe layout mismatch, even when the
    # objective sense and cone family agree.
    mismatch_model, _ = continuation_model(2.0; dimension=1)
    mismatch_source = SDPX.optimize!(mismatch_model; settings=settings, outputs=all_outputs)
    @test SDPX.status(mismatch_source) === :optimal
    mismatch_target, _ = continuation_model(2.01)
    mismatch = SDPX.optimize!(
        mismatch_target;
        settings=settings,
        outputs=all_outputs,
        warm_start=mismatch_source,
    )
    @test SDPX.status(mismatch) === :optimal
    mismatch_init = SDPX.diagnostics(mismatch).selected_algorithms.initialization
    @test !mismatch_init.continuation.accepted
    @test mismatch_init.continuation.reason === :layout_mismatch

    # Explicit model starts and Result continuation are mutually exclusive;
    # combining them would make the source of the iterate ambiguous.
    conflicting_model, conflicting_X = continuation_model(2.01)
    SDPX.set_start!(conflicting_X, [1.0 0.0; 0.0 1.0])
    @test_throws ArgumentError SDPX.optimize!(
        conflicting_model;
        settings=settings,
        outputs=all_outputs,
        warm_start=source,
    )
end

@testset "NativeSOC warm starts and RSOC adjoint mapping" begin
    settings = SDPX.Settings{Float64}(
        tolerances=SDPX.Tolerances{Float64}(
            primal=1e-6,
            dual=1e-6,
            gap=1e-6,
        ),
        limits=SDPX.Limits(iterations=100, time=30.0, threads=1),
        verbosity=0,
    )

    # Product and affine SOC starts are both required when a cone-dual start
    # is supplied.  The primal start induces strict interior slacks for both
    # native blocks and is forwarded through the one NativeSOC plan.
    soc_model = SDPX.Model(Float64)
    q = SDPX.variable!(soc_model, :q, 2; domain=SDPX.LorentzCone())
    upper = SDPX.constraint!(
        soc_model,
        :upper,
        [1 - q[1], -q[2]],
        SDPX.LorentzCone(),
    )
    SDPX.set_start!(q, [0.5, 0.0])
    SDPX.set_dual_slack_start!(q, [1.0, 0.0])
    SDPX.set_dual_start!(upper, [1.0, 0.0])
    SDPX.objective!(soc_model, SDPX.Maximize(), q[1] + 2)
    soc_result = SDPX.optimize!(
        soc_model;
        settings=settings,
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test soc_result.status == SDPX.Optimal
    @test SDPX.value(soc_result, q)[1] ≈ 1.0 atol=1e-6
    @test SDPX.certificate(soc_result).valid
    @test SDPX.execution_plan(soc_result) === SDPX.diagnostics(soc_result).plan

    # A partial SOC cone-dual vector is a typed fail-closed error, not a
    # silent cold-start fallback.
    incomplete = SDPX.Model(Float64)
    iq = SDPX.variable!(incomplete, :q, 2; domain=SDPX.LorentzCone())
    SDPX.constraint!(
        incomplete,
        :upper,
        [1 - iq[1], -iq[2]],
        SDPX.LorentzCone(),
    )
    SDPX.set_dual_slack_start!(iq, [1.0, 0.0])
    SDPX.objective!(incomplete, SDPX.Maximize(), iq[1])
    @test_throws SDPX.PublicOptimizeError SDPX.optimize!(
        incomplete;
        settings=settings,
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )

    # Complete but boundary cone duals are rejected by the NativeSOC core;
    # this confirms the adapter does not shift a non-interior start.
    noninterior = SDPX.Model(Float64)
    bq = SDPX.variable!(noninterior, :q, 2; domain=SDPX.LorentzCone())
    brow = SDPX.constraint!(
        noninterior,
        :upper,
        [1 - bq[1], -bq[2]],
        SDPX.LorentzCone(),
    )
    SDPX.set_dual_slack_start!(bq, [1.0, 1.0])
    SDPX.set_dual_start!(brow, [1.0, 1.0])
    SDPX.objective!(noninterior, SDPX.Maximize(), bq[1])
    @test_throws ArgumentError SDPX.optimize!(
        noninterior;
        settings=settings,
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )

    nonfinite = SDPX.Model(Float64)
    fq = SDPX.variable!(nonfinite, :q, 2; domain=SDPX.LorentzCone())
    @test_throws ArgumentError SDPX.set_dual_slack_start!(fq, [NaN, 0.0])

    # No-start models still return no warm coordinates; the normal cold-start
    # route remains the same public path.
    cold = SDPX.Model(Float64)
    cq = SDPX.variable!(cold, :q, 2; domain=SDPX.LorentzCone())
    SDPX.objective!(cold, SDPX.Minimize(), cq[1])
    cold_program = SDPX.compile_product_cone_model(cold)
    cold_lowering = SDPX.lower_soc_native(cold_program)
    cold_starts = SDPX._public_soc_starts(cold, cold_lowering)
    @test cold_starts.x0 === nothing
    @test cold_starts.z0 === nothing
    @test cold_starts.y0 === nothing

    # RSOC dual starts use the exact inverse-adjoint map.  Round-tripping a
    # native vector through each lowering record must recover it exactly.
    rsoc = SDPX.Model(Float64)
    rq = SDPX.variable!(rsoc, :q, 3; domain=SDPX.RotatedLorentzCone())
    rupper = SDPX.constraint!(
        rsoc,
        :upper,
        [2 - rq[1], 2 - rq[2], -rq[3]],
        SDPX.RotatedLorentzCone(),
    )
    SDPX.set_start!(rq, [1.0, 1.0, 0.0])
    SDPX.set_dual_slack_start!(rq, [1.0, 1.0, 0.0])
    SDPX.set_dual_start!(rupper, [1.0, 1.0, 0.0])
    SDPX.objective!(rsoc, SDPX.Maximize(), rq[1])
    rsoc_program = SDPX.compile_product_cone_model(rsoc)
    rsoc_lowering = SDPX.lower_soc_native(rsoc_program)
    native_dual = [1.0, 1.0, 0.0]
    for record in rsoc_lowering.dual_records
        core_dual = SDPX._public_soc_core_dual_start(
            rsoc_lowering,
            record,
            native_dual,
        )
        @test record.map * core_dual ≈ native_dual atol=1e-12
    end
    rsoc_result = SDPX.optimize!(
        rsoc;
        settings=settings,
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test rsoc_result.status == SDPX.Optimal
    @test SDPX.value(rsoc_result, rq)[1] ≈ 2.0 atol=1e-6
    @test SDPX.certificate(rsoc_result).valid
end

@testset "Max native SOC/SDP objective and packed dual maps" begin
    # RSOC feasibility is measured after the exact M map into Lorentz
    # coordinates, so its residual has linear (margin) units.
    @test SDPX._public_primal_cone_residual(
        [1.0, 1.0, 0.0],
        SDPX.RotatedLorentzCone(),
    ) == 0.0
    @test SDPX._public_primal_cone_residual(
        [1.0, 1.0, 2.0],
        SDPX.RotatedLorentzCone(),
    ) ≈ sqrt(8.0) - 2.0 atol=1e-12

    # Native SOC: q1 is bounded by the Lorentz row 1-q1 >= |q2|.
    soc_model = SDPX.Model(Float64)
    q = SDPX.variable!(soc_model, :q, 2; domain=SDPX.LorentzCone())
    SDPX.constraint!(soc_model, :upper, [1 - q[1], -q[2]], SDPX.LorentzCone())
    SDPX.objective!(soc_model, SDPX.Maximize(), q[1] + 2)
    soc_result = SDPX.optimize!(
        soc_model;
        settings=SDPX.Settings{Float64}(
            limits=SDPX.Limits(iterations=100, time=30.0, threads=1),
            verbosity=0,
        ),
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test soc_result.status == SDPX.Optimal
    @test SDPX.value(soc_result, q)[1] ≈ 1.0 atol=1e-6
    @test SDPX.primal_objective(soc_result) ≈ 3.0 atol=1e-6
    @test SDPX.dual_objective(soc_result) ≈ 3.0 atol=1e-6
    @test SDPX.certificate(soc_result).valid

    # Native SDP: maximize an off-diagonal objective over 0 <= X <= I.
    # The public dual matrix must undo the packed off-diagonal factor of two.
    sdp_model = SDPX.Model(Float64)
    X = SDPX.variable!(sdp_model, :X, 2, 2; domain=SDPX.PSDCone())
    upper = SDPX.constraint!(
        sdp_model,
        :upper,
        [1 - X[1, 1] -X[1, 2]; -X[1, 2] 1 - X[2, 2]],
        SDPX.PSDCone(),
    )
    SDPX.objective!(sdp_model, SDPX.Maximize(), 2 * X[1, 2] + 3)
    sdp_result = SDPX.optimize!(
        sdp_model;
        settings=SDPX.Settings{Float64}(
            limits=SDPX.Limits(iterations=100, time=30.0, threads=1),
            verbosity=0,
        ),
        outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
    )
    @test sdp_result.status == SDPX.Optimal
    @test SDPX.primal_objective(sdp_result) ≈ 4.0 atol=1e-6
    @test SDPX.dual_objective(sdp_result) ≈ 4.0 atol=1e-6
    @test SDPX.dual(sdp_result, upper)[1, 2] ≈ 0.5 atol=1e-6
    @test SDPX.certificate(sdp_result).valid
end

@testset "model-owned BigFloat precision crosses optimize!" begin
    setprecision(BigFloat, 64) do
        model = SDPX.Model(BigFloat; precision_bits=256)
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :bound, x[1] - BigFloat(1), SDPX.Nonnegative())
        SDPX.objective!(model, SDPX.Minimize(), x[1])
        result = SDPX.optimize!(
            model;
            settings=SDPX.Settings{BigFloat}(
                limits=SDPX.Limits(iterations=50, time=60.0, threads=1),
                verbosity=0,
            ),
            outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
        )
        @test result.status == SDPX.Optimal
        @test precision(result.certificate.primal_objective) == 256
        @test result.certificate.valid
        @test SDPX.value(result, x)[1] ≈ BigFloat(1) atol=BigFloat("1e-20")
    end
end

@testset "BigFloat Result PSD expansion keeps stored precision" begin
    # Mimic a retained 256-bit Result payload being read after the caller has
    # changed the ambient BigFloat scope to 64 bits.  The packed dual map
    # divides off-diagonal entries by two and must perform that operation at
    # the stored precision, not at the getter's ambient precision.
    packed = setprecision(BigFloat, 256) do
        [
            BigFloat("1.0"; precision=256),
            BigFloat("1.23456789012345678901234567890123456789"; precision=256),
            BigFloat("2.0"; precision=256),
        ]
    end
    setprecision(BigFloat, 64) do
        matrix = SDPX._result_packed_matrix(packed, 2, BigFloat, true)
        @test all(precision(value) == 256 for value in matrix)
        @test matrix[1, 2] == setprecision(BigFloat, 256) do
            packed[2] / BigFloat(2; precision=256)
        end
        @test matrix[2, 1] == matrix[1, 2]
    end
end
