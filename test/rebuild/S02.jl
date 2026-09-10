# S02 required tests — standalone:
#
#     julia --project=<repo> test/rebuild/S02.jl
#
# A. per-step trajectory identity for unchanged-arithmetic cases
# B. residual-freshness invariant, bitwise, with the negative control that
#    justifies the `residual_canonical` field
# C. typed outcomes for stagnation, line-search failure and rollback
# D. ONE HSD loop — STATIC call-graph argument (labelled STATIC, not a dynamic
#    proof)
# E. hot entry points infer; no cone instance in a session type parameter
using Test
using SDPX
using SDPX: alloc_zeros, copy_owned!, HSDStepCode, HSDStepOK,
    HSDStepAlreadyOptimal, HSDStepBreakdown, HSDStepSingularKKT,
    HSDStepDirectionFailed, ProductConeHSDState, ProductHSDSolveResult,
    product_hsd_step!, product_hsd_cold_start!, kkt_derived_start!,
    _product_hsd_residual!, _product_hsd_residual_is_fresh, _cert_residual!,
    _hsd_residual_is_fresh, _product_hsd_bump_point_epoch!,
    _product_hsd_tau_collapse_ready, _product_hsd_tau_collapse_recenter!,
    _product_hsd_termination_or_dual_ray!, _product_hsd_candidate_result!,
    _product_hsd_verified_result, _product_hsd_terminal_verified_result!,
    _product_hsd_line_search!, _product_hsd_make_result,
    _product_hsd_fixed_trace_hkm_neighborhood!, _reset_q3_phase_timings!,
    reset_phase_timings!, default_certificate_tol, verify_dual_infeasibility!,
    try_update_scaling!, FixedTraceQ3CoreWorkspace,
    ProductHSDVerifiedAcceptedStep, ProductHSDVerifiedInitialPoint,
    ProductHSDVerifiedTerminationRay, ProductHSDMaxIterations,
    ProductHSDIterationLimitReached, ProductHSDBreakdown,
    ProductHSDDirectionBreakdown, ProductHSDLineSearchBreakdown,
    ProductHSDSingular, ProductHSDSingularKKTReason,
    ProductHSDUnverifiedZeroComplementarity, ProductHSDTimeLimit,
    ProductHSDTimeLimitReached, ProductHSDInsufficientPrecision,
    ProductHSDTauCollapseRecoveryExhausted, ProductHSDRankAmbiguous,
    ProductHSDRankAmbiguousSetup, ProductHSDRankRayVerificationFailed,
    ProductHSDKKTInitializationFailed, ProductHSDDualInfeasible,
    ProductHSDPrimalInfeasible, ProductHSDOptimal

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SOLVER_FILES = [
    "iterate.jl", "session.jl", "residuals.jl",
    "globalization.jl", "recovery.jl", "loop.jl",
]
include(joinpath(REPO_ROOT, "src", "solver", "loop.jl"))

# ---------------------------------------------------------------------------
# Fixtures. Small on purpose: the identity claim is bitwise *per step*, so it
# needs many steps rather than a large problem.
# ---------------------------------------------------------------------------

function _lp_canonical(n::Int, m::Int)
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, n; domain=SDPX.Nonnegative())
    for i in 1:m
        expr = sum(
            (sin(Float64(i * 3 + j * 7)) * (1.0 + 0.1 * j)) * x[j] -
            Float64(i) * 0.1 for j in 1:n
        )
        SDPX.constraint!(model, Symbol(:eq, i), expr, SDPX.ZeroCone())
    end
    SDPX.objective!(model, SDPX.Minimize(),
        sum((1.0 + 0.3 * j) * x[j] for j in 1:n))
    return SDPX.canonicalize(SDPX.compile_product_cone_model(model))
end

function _soc_canonical()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return SDPX.canonicalize(SDPX.compile_product_cone_model(model))
end

_bitwise(a::AbstractVector, b::AbstractVector) = solver_bitwise_equal(a, b)

_snapshot(state::ProductConeHSDState) = (
    x=copy(state.base.x), y=copy(state.base.y), s=copy(state.base.s),
    tau=state.base.tau, kappa=state.base.kappa, mu=state.base.mu,
    point_epoch=state.base.point_epoch,
)

"""
Test-local reference replay of the production driver's epoch body
(`product_hsd_solve!` lines 796-910, common path): initialization -> entry
residual gate -> `product_hsd_step!` -> accepted iterate.  This is an oracle for
the trajectory-identity test; it is *not* production code and is not reachable
from the package.
"""
function _reference_steps(canonical; limit::Integer, initialization::Symbol)
    state = ProductConeHSDState(canonical)
    base = state.base
    tol = Float64(default_certificate_tol(Float64))
    x_original = alloc_zeros(Float64, base.n)
    s_original = alloc_zeros(Float64, base.m)
    y_original = alloc_zeros(Float64, base.m)
    if initialization === :kkt
        report = kkt_derived_start!(state)
        report.ok || error("reference KKT initialization failed")
    else
        product_hsd_cold_start!(state)
    end
    _product_hsd_residual!(state)
    steps = NamedTuple[]
    for _ in 1:Int(limit)
        if !_product_hsd_residual_is_fresh(state)
            _product_hsd_residual!(state)
        end
        before = base.point_epoch
        code = product_hsd_step!(state)
        base.point_epoch == before || push!(steps, _snapshot(state))
        verified = _product_hsd_candidate_result!(
            state, x_original, s_original, y_original, tol,
            ProductHSDVerifiedAcceptedStep, code,
        )
        verified === nothing || break
        code === HSDStepOK || break
    end
    return steps
end

# ---------------------------------------------------------------------------
# A. Trajectory identity.
# ---------------------------------------------------------------------------

@testset "S02-A trajectory identity (unchanged arithmetic)" begin
    cases = (
        (
            name="LP n=30 m=0 orthant (no equalities)",
            canonical=_lp_canonical(30, 0), limit=80, initialization=:identity,
            optimal=true,
        ),
        (
            name="LP n=40 m=8 equality rows",
            canonical=_lp_canonical(40, 8), limit=80, initialization=:identity,
            optimal=false,
        ),
        (
            name="LP n=60 m=12 equality rows",
            canonical=_lp_canonical(60, 12), limit=80, initialization=:identity,
            optimal=false,
        ),
        (
            name="SOC k=3 KKT start",
            canonical=_soc_canonical(), limit=80, initialization=:kkt,
            optimal=true,
        ),
    )
    for case in cases
        @testset "$(case.name)" begin
            # Production driver, black box.
            reference_state = ProductConeHSDState(case.canonical)
            reference_result = SDPX.product_hsd_solve!(
                reference_state; max_iterations=case.limit,
                initialization=case.initialization,
            )

            # Session loop, traced.
            session = SessionState(ProductConeHSDState(case.canonical))
            trace = SessionTrace{Float64}()
            outcome = solver_run_session!(
                session; max_iterations=case.limit,
                initialization=case.initialization, trace=trace,
            )

            # Test-local per-step oracle.
            reference_steps = _reference_steps(
                case.canonical; limit=case.limit,
                initialization=case.initialization,
            )

            @info "S02-A case" case.name reference_status=reference_result.status session_status=outcome.result.status outcome=outcome.status iterations=reference_result.iterations accepted_steps=outcome.accepted_steps rejected_steps=outcome.rejected_steps rollbacks=outcome.rollbacks rebinds=session.rebinds

            # Same typed result as the production driver.
            @test outcome.result.status === reference_result.status
            @test outcome.result.reason === reference_result.reason
            @test outcome.result.iterations == reference_result.iterations
            @test outcome.result.last_step === reference_result.last_step
            @test outcome.iterations == reference_result.iterations
            @test (reference_result.status === ProductHSDOptimal) ==
                  case.optimal

            # Terminal live iterate, bitwise, against the black-box driver.
            reference_live = _snapshot(reference_state)
            session_live = _snapshot(session.hsd)
            @test _bitwise(reference_live.x, session_live.x)
            @test _bitwise(reference_live.y, session_live.y)
            @test _bitwise(reference_live.s, session_live.s)
            @test reference_live.tau === session_live.tau
            @test reference_live.kappa === session_live.kappa
            @test reference_live.mu === session_live.mu

            # Per-step accepted trajectory: session trace vs oracle replay.
            accepted = solver_accepted_iterates(trace)
            selected = findall(trace.accepted)
            accepted_mu = trace.mu[selected]
            accepted_tau = trace.tau[selected]
            accepted_kappa = trace.kappa[selected]
            accepted_epoch = trace.point_epoch[selected]
            @test length(accepted) == outcome.accepted_steps
            @test length(accepted) == reference_result.iterations
            @test length(accepted) == length(reference_steps)
            @test length(accepted) > 0
            for k in eachindex(reference_steps)
                step = reference_steps[k]
                @test _bitwise(step.x, accepted[k])
                @test step.mu === accepted_mu[k]
                @test step.tau === accepted_tau[k]
                @test step.kappa === accepted_kappa[k]
                @test step.point_epoch == accepted_epoch[k]
            end

            # Every accepted point binds a complete state, and the live iterate
            # is bitwise the bound point at every accepted record.
            @test solver_binding_is_complete(session)
            @test solver_live_matches_accepted(session)
            @test session.bindings ==
                  outcome.accepted_steps + 1 + session.rebinds
            @test session.rebinds <= 1
            @test all(
                k -> trace.bound_epoch[k] == trace.point_epoch[k] &&
                     _bitwise(trace.bound_x[k], trace.x[k]),
                eachindex(trace.iteration),
            )

            # Determinism: an independent identical run traces identically.
            session2 = SessionState(ProductConeHSDState(case.canonical))
            trace2 = SessionTrace{Float64}()
            outcome2 = solver_run_session!(
                session2; max_iterations=case.limit,
                initialization=case.initialization, trace=trace2,
            )
            @test outcome2.result.status === outcome.result.status
            @test solver_first_divergent_step(trace, trace2) == 0
        end
    end
end

# ---------------------------------------------------------------------------
# B. Residual freshness.
# ---------------------------------------------------------------------------

@testset "S02-B residual freshness invariant" begin
    @testset "session predicate == production predicate at every epoch" begin
        canonical = _lp_canonical(60, 12)
        session = SessionState(ProductConeHSDState(canonical))
        checked = Ref(0)
        violations = Ref(0)
        agree = Ref(0)
        instrumented = function (state)
            code = product_hsd_step!(state)
            if solver_residual_is_fresh(session)
                checked[] += 1
                solver_residual_is_bitwise_canonical(session) ||
                    (violations[] += 1)
            end
            _hsd_residual_is_fresh(state.base) ==
                solver_residual_is_fresh(session) && (agree[] += 1)
            return code
        end
        trace = SessionTrace{Float64}()
        outcome = solver_run_session!(
            session; max_iterations=80, trace=trace, trial_fn=instrumented,
        )
        # The instrumented run (which recomputes the canonical residual between
        # epochs) must still reproduce the uninstrumented trajectory exactly.
        plain = SessionState(ProductConeHSDState(canonical))
        plain_trace = SessionTrace{Float64}()
        plain_outcome = solver_run_session!(
            plain; max_iterations=80, trace=plain_trace,
        )
        @info "S02-B freshness checks" claimed_fresh=checked[] violations=violations[] predicate_agreement=agree[] status=outcome.result.status
        @test plain_outcome.result.status === outcome.result.status
        @test solver_first_divergent_step(trace, plain_trace, :step) == 0
        @test checked[] > 0          # not vacuous: freshness was claimed
        @test violations[] == 0      # and the claim was true, bitwise
        @test agree[] > 0            # session and production predicates agree
    end

    @testset "certificate kernel: the negative control that justifies the field" begin
        # LP n=200 is the case where the two kernels really disagree bitwise
        # (ADR-001 §4 measured 2.2e-14 here).
        canonical = _lp_canonical(200, 40)
        session = SessionState(ProductConeHSDState(canonical))
        kkt_derived_start!(session.hsd)
        solver_refresh_residual!(session)
        @test solver_residual_is_fresh(session)
        @test solver_residual_is_bitwise_canonical(session)

        canonical_rP = copy(session.hsd.base.rP)
        canonical_rD = copy(session.hsd.base.rD)
        kernel = solver_certificate_residual!(session)
        @test kernel === RESIDUAL_CERTIFICATE
        @test !solver_residual_is_fresh(session)
        @test !_hsd_residual_is_fresh(session.hsd.base)
        @test !_product_hsd_residual_is_fresh(session.hsd)
        # The kernels really do differ bitwise; the disagreement is roundoff
        # scale but non-zero, and this records its magnitude.
        @test canonical_rP != session.hsd.base.rP
        @test canonical_rD != session.hsd.base.rD
        disagreement = max(
            maximum(abs, canonical_rP - session.hsd.base.rP),
            maximum(abs, canonical_rD - session.hsd.base.rD),
        )
        @test disagreement > 0.0
        @info "S02-B certificate-vs-canonical residual disagreement (LP n=200, m=40)" disagreement

        # The session's entry gate restores the canonical residual, so the
        # next direction build consumes canonical values again.
        @test solver_ensure_residual!(session)
        @test solver_residual_is_fresh(session)
        @test solver_residual_is_bitwise_canonical(session)
    end

    @testset "certificate-flavoured residual never reaches the direction build" begin
        # The production interleaving: a certificate check runs between two
        # epochs. The migrated token must force the canonical recomputation.
        canonical = _lp_canonical(200, 40)
        reference = _reference_steps(
            canonical; limit=40, initialization=:identity,
        )
        session = SessionState(ProductConeHSDState(canonical))
        differences = Float64[]
        interleaved = function (state)
            code = product_hsd_step!(state)
            canonical_rP = copy(state.base.rP)
            solver_certificate_residual!(session)
            push!(differences, maximum(abs, state.base.rP .- canonical_rP))
            return code
        end
        trace = SessionTrace{Float64}()
        outcome = solver_run_session!(
            session; max_iterations=40, trace=trace, trial_fn=interleaved,
        )
        accepted = solver_accepted_iterates(trace)
        accepted_mu = trace.mu[findall(trace.accepted)]
        @test length(accepted) == length(reference)
        for k in eachindex(reference)
            @test _bitwise(reference[k].x, accepted[k])
            @test reference[k].mu === accepted_mu[k]
        end
        # Non-vacuous: the interleaved certificate kernel really did overwrite
        # the cached residual with bitwise-different values at every epoch, and
        # the session's gate recomputed the canonical one before each direction.
        # The last epoch may execute without committing, so the executor can run
        # once more than the number of accepted steps.
        @test length(accepted) <= length(differences) <= length(accepted) + 1
        @test maximum(differences) > 0.0
        @info "S02-B interleaved-certificate epochs" epochs=length(accepted) status=outcome.result.status max_rP_overwrite=maximum(differences)

        # Sensitivity control: the trajectory comparison in this file CAN
        # detect a residual-driven divergence. Perturbing the cached canonical
        # rP by 1e-10 (about three orders above the kernel disagreement) changes
        # the next accepted iterate bitwise.
        control = ProductConeHSDState(canonical)
        product_hsd_cold_start!(control)
        _product_hsd_residual!(control)
        perturbed = ProductConeHSDState(canonical)
        product_hsd_cold_start!(perturbed)
        _product_hsd_residual!(perturbed)
        for _ in 1:3
            for state in (control, perturbed)
                _product_hsd_residual_is_fresh(state) ||
                    _product_hsd_residual!(state)
                product_hsd_step!(state)
            end
        end
        _product_hsd_residual!(control)
        _product_hsd_residual!(perturbed)
        perturbed.base.rP[1] += 1.0e-10
        perturbed.base.residual_canonical = true
        perturbed.base.residual_epoch = perturbed.base.point_epoch
        control_code = product_hsd_step!(control)
        perturbed_code = product_hsd_step!(perturbed)
        @test !_bitwise(control.base.x, perturbed.base.x)
        @info "S02-B residual sensitivity control" control_code perturbed_code max_dx=maximum(abs, control.base.x .- perturbed.base.x)

        # MEASURED NEGATIVE RESULT, recorded rather than asserted.
        #
        # Forging the pre-audit token ("the point did not move, so the cached
        # residual is canonical") does NOT change this fixture's trajectory,
        # even though the cached rP differs bitwise by up to 2.6e-13 at every
        # epoch. The direction build is fed `state.h + rP` through a refined
        # KKT solve and this fixture's rounding absorbs that difference. So the
        # `residual_canonical` field is justified by the *contract* (a
        # certificate-flavoured residual is not the canonical one, and the
        # sensitive control above shows the trajectory can see such a
        # difference when it is not absorbed), not by an observable trajectory
        # divergence on this problem.
        honest = ProductConeHSDState(canonical)
        product_hsd_cold_start!(honest)
        _product_hsd_residual!(honest)
        forged = ProductConeHSDState(canonical)
        product_hsd_cold_start!(forged)
        _product_hsd_residual!(forged)
        divergence = 0
        compared = 0
        max_association_difference = 0.0
        for k in 1:24
            _product_hsd_residual_is_fresh(honest) ||
                _product_hsd_residual!(honest)
            # The forged run is refreshed canonically first, so this measurement
            # isolates the *accumulation association* difference at one and the
            # same iterate: canonical values vs the certificate kernel's values.
            _product_hsd_residual!(forged)
            canonical_rP = copy(forged.base.rP)
            canonical_rD = copy(forged.base.rD)
            _cert_residual!(forged.base)
            max_association_difference = max(
                max_association_difference,
                maximum(abs, canonical_rP .- forged.base.rP),
                maximum(abs, canonical_rD .- forged.base.rD),
            )
            # Pre-audit metadata, forged: "the point did not move, so the cached
            # residual is canonical".
            forged.base.residual_canonical = true
            forged.base.residual_epoch = forged.base.point_epoch
            honest_code = product_hsd_step!(honest)
            forged_code = product_hsd_step!(forged)
            compared += 1
            if divergence == 0 &&
               (honest_code !== forged_code ||
                !_bitwise(honest.base.x, forged.base.x))
                divergence = k
            end
        end
        @info "S02-B forged-token negative control" steps_compared=compared first_divergent_epoch=divergence max_association_difference=max_association_difference
        @test compared == 24
        @test max_association_difference > 0.0
    end
end

# ---------------------------------------------------------------------------
# C. Typed outcomes: stagnation, failure, rollback.
# ---------------------------------------------------------------------------

@testset "S02-C typed stagnation / failure / rollback" begin
    @testset "stagnation reaches a typed outcome" begin
        canonical = _lp_canonical(20, 4)
        session = SessionState(ProductConeHSDState(canonical))
        stalled = _ -> HSDStepOK          # an epoch that commits nothing
        trace = SessionTrace{Float64}()
        outcome = solver_run_session!(
            session; max_iterations=10, stagnation_limit=3, trace=trace,
            trial_fn=stalled,
        )
        @test outcome.status === SessionStagnated
        @test outcome.stagnated
        @test outcome.rejected_steps == 3
        @test outcome.accepted_steps == 0
        @test outcome.rollbacks == 3
        @test outcome.last_trial === TRIAL_REJECTED
        @test outcome.result.status === ProductHSDMaxIterations
        @test outcome.result.reason === ProductHSDIterationLimitReached
        # Disabled by default: the same stub with stagnation_limit=0 runs the
        # full budget instead of stopping early.
        session_off = SessionState(ProductConeHSDState(canonical))
        outcome_off = solver_run_session!(
            session_off; max_iterations=10, trial_fn=stalled,
        )
        @test outcome_off.status === SessionExhausted
        @test !outcome_off.stagnated
        @test outcome_off.rejected_steps == 10
        @test outcome_off.rollbacks == 10
    end

    @testset "direction failure reaches a typed outcome" begin
        canonical = _lp_canonical(20, 4)
        session = SessionState(ProductConeHSDState(canonical))
        failing = _ -> HSDStepDirectionFailed
        outcome = solver_run_session!(
            session; max_iterations=10, trial_fn=failing,
        )
        @test outcome.status in (SessionStepFailure, SessionCertified)
        @test outcome.last_step === HSDStepDirectionFailed
        @test outcome.last_trial === TRIAL_REJECTED
        @test outcome.rejected_steps == 1
        # The accepted point survives a failed direction: the session is still
        # bound to a complete state.
        @test solver_binding_is_complete(session)
        @test solver_live_matches_accepted(session)
    end

    @testset "line-search breakdown reaches a typed outcome" begin
        canonical = _lp_canonical(20, 4)
        session = SessionState(ProductConeHSDState(canonical))
        breakdown = _ -> HSDStepBreakdown
        outcome = solver_run_session!(
            session; max_iterations=10, trial_fn=breakdown,
        )
        @test outcome.status in (SessionStepFailure, SessionCertified)
        @test outcome.last_step === HSDStepBreakdown
        @test outcome.status === SessionStepFailure ||
              outcome.result.status !== ProductHSDOptimal
        @test solver_binding_is_complete(session)
        # The genuine line-search-breakdown terminal path is exercised with the
        # real executor in test A (the equality-row LPs end in
        # `ProductHSDLineSearchBreakdown` after ~20 epochs, bitwise identical to
        # the production driver).
    end

    @testset "rollback repairs a leaked iterate, and is a verified no-op otherwise" begin
        canonical = _lp_canonical(20, 4)
        session = SessionState(ProductConeHSDState(canonical))
        kkt_derived_start!(session.hsd)
        solver_refresh_residual!(session)
        solver_bind_accepted!(session)
        before = _snapshot(session.hsd)
        @test solver_live_matches_accepted(session)

        # An epoch that writes the iterate without committing it: exactly the
        # leak the rollback must detect and repair. Driven through the lifecycle
        # phase directly so the terminal certificate phase cannot mask it.
        leaky = function (state)
            base = state.base
            @inbounds for j in eachindex(base.x)
                base.x[j] = base.x[j] + 0.5
            end
            return HSDStepOK
        end
        outcome = solver_run_trial!(session; trial_fn=leaky)
        @test outcome.status === TRIAL_REJECTED
        @test !outcome.committed
        @test outcome.rollback_repaired
        @test session.rollbacks == 1
        @test session.repairs == 1
        after = _snapshot(session.hsd)
        @test _bitwise(before.x, after.x)
        @test _bitwise(before.y, after.y)
        @test _bitwise(before.s, after.s)
        @test before.tau === after.tau
        @test before.kappa === after.kappa
        @test solver_live_matches_accepted(session)
        @test solver_binding_is_complete(session)
        # The repair is a new point, so the lifecycle token advanced, and the
        # residual was recomputed for it.
        @test after.point_epoch > before.point_epoch
        @test solver_residual_is_fresh(session)
        @test solver_residual_is_bitwise_canonical(session)
        @info "S02-C rollback repair" repairs=session.repairs rollbacks=session.rollbacks epochs_advanced=after.point_epoch - before.point_epoch

        # Production case: nothing leaked, so the rollback is a verified no-op
        # and the bound point is untouched.
        clean_session = SessionState(ProductConeHSDState(canonical))
        kkt_derived_start!(clean_session.hsd)
        solver_refresh_residual!(clean_session)
        solver_bind_accepted!(clean_session)
        bound_before = copy(clean_session.accepted.x)
        epoch_before = solver_point_epoch(clean_session)
        outcome_clean = solver_run_trial!(
            clean_session; trial_fn=_ -> HSDStepOK,
        )
        @test outcome_clean.status === TRIAL_REJECTED
        @test !outcome_clean.committed
        @test !outcome_clean.rollback_repaired
        @test clean_session.repairs == 0
        @test clean_session.rollbacks == 1
        @test _bitwise(bound_before, clean_session.accepted.x)
        @test solver_point_matches_accepted(clean_session)
        @test solver_live_matches_accepted(clean_session)
        @test solver_point_epoch(clean_session) == epoch_before

        # The same leak driven through the whole loop: the typed outcome is
        # stagnation and the session must have repaired, not accepted, the leak.
        loop_session = SessionState(ProductConeHSDState(canonical))
        loop_outcome = solver_run_session!(
            loop_session; max_iterations=5, stagnation_limit=1, trial_fn=leaky,
        )
        @test loop_outcome.status === SessionStagnated
        @test loop_session.repairs == 1
        @test loop_session.rollbacks == 1
        @test loop_outcome.accepted_steps == 0
        @test loop_outcome.rejected_steps == 1
        @test loop_outcome.last_trial === TRIAL_REJECTED
        @test solver_binding_is_complete(loop_session)
        @test solver_point_matches_accepted(loop_session)
    end

    @testset "conditioned-SOC recovery is a typed decision" begin
        session = SessionState(ProductConeHSDState(_soc_canonical()))
        kkt_derived_start!(session.hsd)
        solver_bind_accepted!(session)
        epoch_before = solver_point_epoch(session)
        recovery = solver_conditioned_soc_rescue!(session)
        @test recovery.status in (RECOVERY_CONDITIONED_SOC, RECOVERY_NONE)
        if recovery.status === RECOVERY_CONDITIONED_SOC
            @test solver_point_epoch(session) > epoch_before
            @test solver_live_matches_accepted(session)
            @test solver_binding_is_complete(session)
        else
            @test solver_point_epoch(session) == epoch_before
            @test solver_live_matches_accepted(session)
        end
        # An ineligible state (no SOC block) must decline cleanly.
        lp_session = SessionState(ProductConeHSDState(_lp_canonical(10, 2)))
        @test solver_conditioned_soc_rescue!(lp_session).status ===
              RECOVERY_NONE
    end
end

# ---------------------------------------------------------------------------
# D. ONE HSD loop — STATIC argument.
# ---------------------------------------------------------------------------

@testset "S02-D one HSD loop (STATIC)" begin
    # STATIC: a source scan, not a dynamic call-graph proof. It is labelled
    # STATIC deliberately and does not claim runtime reachability.
    loop_sites = Dict{String,Int}()
    for name in SOLVER_FILES
        lines = readlines(joinpath(REPO_ROOT, "src", "solver", name))
        current = "<top level>"
        for line in lines
            m = match(
                r"^\s*(?:@inline\s+|@noinline\s+)*function\s+([A-Za-z0-9_!]+)",
                line,
            )
            m === nothing || (current = String(m.captures[1]))
            # Loop *syntax*, not the English word (docstrings say "for the
            # current iterate"); the scan is heuristic and is labelled STATIC.
            if occursin(
                r"^\s*(?:@inbounds\s+|@simd\s+)*(?:for\s+\w+\s+in\b|while\s+[!(\w])",
                line,
            )
                key = string(name, ":", current)
                loop_sites[key] = get(loop_sites, key, 0) + 1
            end
        end
    end
    @info "S02-D loop sites in the extraction" loop_sites
    @test sort(collect(keys(loop_sites))) == [
        "loop.jl:solver_run_session!", "session.jl:solver_bitwise_equal",
    ]
    @test loop_sites["loop.jl:solver_run_session!"] == 1

    # One epoch executor call site: the loop calls `solver_run_trial!` once and
    # never calls the step executor directly.
    loop_source = read(joinpath(REPO_ROOT, "src", "solver", "loop.jl"), String)
    @test count("solver_run_trial!(session", loop_source) == 1
    @test count("product_hsd_step!(", loop_source) == 0

    # Production still has exactly one HSD loop, and it does not call this one:
    # src/SDPX.jl does not include src/solver/loop.jl at all, so the extraction
    # is inert until the integration task switches the include.
    root_source = read(joinpath(REPO_ROOT, "src", "SDPX.jl"), String)
    @test !occursin("solver/loop.jl", root_source)
    production = read(
        joinpath(REPO_ROOT, "src", "hsd", "product_cone_solve.jl"), String,
    )
    @test count("for _ in 1:Int(max_iterations)", production) == 1
    @test length(methods(solver_run_session!)) == 1
    @test length(methods(solver_run_trial!)) == 1
end

# ---------------------------------------------------------------------------
# E. Inference and type-parameter discipline.
# ---------------------------------------------------------------------------

@testset "S02-E hot entry points infer stably" begin
    session = SessionState(ProductConeHSDState(_lp_canonical(10, 2)))
    soc_session = SessionState(ProductConeHSDState(_soc_canonical()))

    @test (@inferred solver_residual_is_fresh(session)) isa Bool
    @test (@inferred solver_residual_is_fresh(soc_session)) isa Bool
    @test (@inferred solver_point_epoch(session)) isa Int
    @test (@inferred solver_mu(session)) isa Float64
    @test (@inferred solver_ensure_residual!(session)) isa Bool
    @test (@inferred solver_bind_accepted!(session)) isa Bool
    @test (@inferred solver_binding_is_complete(session)) isa Bool
    @test (@inferred solver_live_matches_accepted(session)) isa Bool
    @test (@inferred solver_workspace_x(session.workspace)) isa Vector{Float64}
    @test (@inferred solver_iterate_workspace(session.hsd)) isa IterateWorkspace

    # No cone instance is a type parameter of the session or the workspace.
    @test length(typeof(session).parameters) == 2
    @test length(typeof(session.workspace).parameters) == 1
    @test typeof(session).parameters[1] === Float64
    @test typeof(session.workspace).parameters[1] === Float64

    # The workspace is a lease on the single physical iterate, not a copy.
    @test solver_workspace_is_single_owner(session.workspace, session.hsd)
    @test session.workspace.x === session.hsd.base.x
    @test session.workspace.ds === session.hsd.base.ds
    @test session.workspace.xt === session.hsd.base.xt
end
