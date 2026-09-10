# ===========================================================================
# S02-c — the ONE HSD loop.
#
# Include order (Julia needs the data types before the session struct):
#   1. iterate.jl      the single iterate/direction workspace   (S02-a)
#   2. session.jl      session ownership + bindings             (S02-a)
#   3. residuals.jl    residual / scaling freshness             (S02-a)
#   4. globalization.jl accepted/trial/rollback lifecycle       (S02-b)
#   5. recovery.jl     typed recovery strategies                (S02-b)
#   6. this file       the single loop + traces                 (S02-c)
#
# THE ONE-LOOP CLAIM (STATIC).  `solver_run_session!` below is the only
# iterate-advancing loop in the extraction: it is the only place a step
# executor (`solver_run_trial!`) is called, and the only loop construct in
# these files other than the elementwise comparison helper in `session.jl`.
# The claim is a *static* call-graph argument executed by `test/rebuild/S02.jl`
# (source scan); it is not a dynamic proof.
#
# WHY THIS IS NOT A SECOND PRODUCTION LOOP.  As delivered, nothing in
# `src/SDPX.jl` includes this file, so production still calls exactly one HSD
# loop: `product_hsd_solve!`.  This loop is the *same* loop body with its phases
# named, which is why it reproduces that driver's per-step trajectory bit for
# bit (test A in `test/rebuild/S02.jl`).  I01 switches the include and deletes
# the old carrier; at that point exactly one loop remains.
#
# WHAT IS NOT CHANGED (card step 3): sigma, beta, initialization, recovery
# mathematics, the default provider, every tolerance, and the status-promotion
# rule that only a certificate verifier may publish optimal/infeasible.
# ===========================================================================

include(joinpath(@__DIR__, "iterate.jl"))
include(joinpath(@__DIR__, "session.jl"))
include(joinpath(@__DIR__, "residuals.jl"))
include(joinpath(@__DIR__, "globalization.jl"))
include(joinpath(@__DIR__, "recovery.jl"))

"""
    SessionStatus

Why the session stopped.  `SessionCertified` is assigned only when the carried
result's status is one the unchanged certificate verifiers can publish
(`Optimal`, `PrimalInfeasible`, `DualInfeasible`); every other value means the
session stopped for an operational reason and the result records the honest
typed status.
"""
@enum SessionStatus::UInt8 begin
    SessionCertified
    SessionExhausted
    SessionStagnated
    SessionTimeLimit
    SessionStepFailure
    SessionPrecisionLimit
    SessionSetupRejected
end

"""
    SessionOutcome{T}

Typed session result: the production `ProductHSDSolveResult` plus the session's
own accounting (accepted/rejected epochs, rollbacks, last trial code).
"""
struct SessionOutcome{T}
    status::SessionStatus
    result::ProductHSDSolveResult{T}
    iterations::Int
    accepted_steps::Int
    rejected_steps::Int
    rollbacks::Int
    last_trial::TrialStatus
    last_step::HSDStepCode
    stagnated::Bool
end

"""
    SessionTrace{T}

Opt-in per-epoch trace.  One record per executed epoch (accepted or rejected)
plus one per recovery event and a final record.  Snapshots are copies, so a
trace never aliases live storage.
"""
mutable struct SessionTrace{T}
    iteration::Vector{Int}
    stage::Vector{Symbol}
    accepted::Vector{Bool}
    step_code::Vector{HSDStepCode}
    point_epoch::Vector{Int}
    residual_epoch::Vector{Int}
    kernel::Vector{ResidualKernel}
    alpha::Vector{T}
    backtracking::Vector{Int}
    mu::Vector{T}
    tau::Vector{T}
    kappa::Vector{T}
    x::Vector{Vector{T}}
    y::Vector{Vector{T}}
    s::Vector{Vector{T}}
    bound_x::Vector{Vector{T}}
    bound_epoch::Vector{Int}
end

function SessionTrace{T}() where {T}
    return SessionTrace{T}(
        Int[], Symbol[], Bool[], HSDStepCode[], Int[], Int[], ResidualKernel[],
        T[], Int[], T[], T[], T[],
        Vector{T}[], Vector{T}[], Vector{T}[], Vector{T}[], Int[],
    )
end

function Base.empty!(trace::SessionTrace)
    empty!(trace.iteration); empty!(trace.stage); empty!(trace.accepted)
    empty!(trace.step_code); empty!(trace.point_epoch)
    empty!(trace.residual_epoch); empty!(trace.kernel); empty!(trace.alpha)
    empty!(trace.backtracking); empty!(trace.mu); empty!(trace.tau)
    empty!(trace.kappa); empty!(trace.x); empty!(trace.y); empty!(trace.s)
    empty!(trace.bound_x); empty!(trace.bound_epoch)
    return trace
end

Base.length(trace::SessionTrace) = length(trace.iteration)

"""
    _session_record!(trace, session, outcome, iteration, stage)

Append one trace record.  Read-only with respect to the carrier.
"""
function _session_record!(
    trace::Union{Nothing,SessionTrace{T}},
    session::SessionState{T},
    outcome::TrialOutcome{T},
    iteration::Int,
    stage::Symbol,
) where {T}
    trace === nothing && return nothing
    base = session.hsd.base
    accepted = session.accepted
    push!(trace.iteration, iteration)
    push!(trace.stage, stage)
    push!(trace.accepted, outcome.committed)
    push!(trace.step_code, outcome.step_code)
    push!(trace.point_epoch, base.point_epoch)
    push!(trace.residual_epoch, base.residual_epoch)
    push!(trace.kernel, session.kernel)
    push!(trace.alpha, outcome.alpha)
    push!(trace.backtracking, outcome.backtracking)
    push!(trace.mu, base.mu)
    push!(trace.tau, base.tau)
    push!(trace.kappa, base.kappa)
    push!(trace.x, copy(base.x))
    push!(trace.y, copy(base.y))
    push!(trace.s, copy(base.s))
    push!(trace.bound_x, copy(accepted.x))
    push!(trace.bound_epoch, accepted.point_epoch)
    return nothing
end

@inline function _session_records_differ(
    a::SessionTrace, i::Int, b::SessionTrace, j::Int,
)::Bool
    return a.accepted[i] != b.accepted[j] ||
           a.step_code[i] !== b.step_code[j] ||
           a.point_epoch[i] != b.point_epoch[j] ||
           a.residual_epoch[i] != b.residual_epoch[j] ||
           a.kernel[i] !== b.kernel[j] ||
           a.mu[i] != b.mu[j] ||
           a.tau[i] != b.tau[j] ||
           a.kappa[i] != b.kappa[j] ||
           !solver_bitwise_equal(a.x[i], b.x[j]) ||
           !solver_bitwise_equal(a.y[i], b.y[j]) ||
           !solver_bitwise_equal(a.s[i], b.s[j])
end

"""
    solver_first_divergent_step(a, b, stage=:step) -> Int

Index of the first record of `stage` at which two traces differ, `0` when they
agree, and `n + 1` when one trace is a strict prefix of the other.
"""
function solver_first_divergent_step(
    a::SessionTrace, b::SessionTrace, stage::Symbol=:step,
)::Int
    ia = findall(==(stage), a.stage)
    ib = findall(==(stage), b.stage)
    n = min(length(ia), length(ib))
    index = findfirst(
        k -> _session_records_differ(a, ia[k], b, ib[k]), 1:n,
    )
    index === nothing || return index
    return length(ia) == length(ib) ? 0 : n + 1
end

"""
    solver_accepted_iterates(trace) -> Vector{Vector{T}}

The accepted-point trajectory, in order.  This is the per-step sequence the
trajectory-identity test compares.
"""
solver_accepted_iterates(trace::SessionTrace) = trace.x[findall(trace.accepted)]
solver_accepted_epochs(trace::SessionTrace) =
    trace.point_epoch[findall(trace.accepted)]

"""
    _session_finish!(session, status, result, accepted, rejected, last_trial)

Terminal bookkeeping, then [`_session_outcome`](@ref).

A terminal phase may have restored or re-epoch'd the accepted iterate
(`_product_hsd_terminal_verified_result!` does exactly that), so the binding is
reconciled with the live point before the outcome is assembled.  This changes no
iterate values: the restored point *is* the accepted point.
"""
function _session_finish!(
    session::SessionState{T},
    status::SessionStatus,
    result::ProductHSDSolveResult{T},
    accepted_steps::Int,
    rejected_steps::Int,
    last_trial::TrialOutcome{T},
) where {T}
    solver_rebind_if_moved!(session)
    return _session_outcome(
        session, status, result, accepted_steps, rejected_steps, last_trial,
    )
end

"""
    _session_outcome(session, status, result, accepted, rejected, last_trial)

Assemble the typed outcome.  This is a labelling step: it never changes the
carried result, and it upgrades `status` to `SessionCertified` only when the
result's own status is one a certificate verifier published.
"""
function _session_outcome(
    session::SessionState{T},
    status::SessionStatus,
    result::ProductHSDSolveResult{T},
    accepted_steps::Int,
    rejected_steps::Int,
    last_trial::TrialOutcome{T},
) where {T}
    certified = result.status === ProductHSDOptimal ||
                result.status === ProductHSDPrimalInfeasible ||
                result.status === ProductHSDDualInfeasible
    return SessionOutcome{T}(
        certified ? SessionCertified : status,
        result,
        session.hsd.base.record.iterations,
        accepted_steps,
        rejected_steps,
        session.rollbacks,
        last_trial.status,
        last_trial.step_code,
        status === SessionStagnated,
    )
end

"""
    solver_run_session!(session; kwargs...) -> SessionOutcome

The single HSD loop.

The phase order and every gate are those of `product_hsd_solve!`
(`src/hsd/product_cone_solve.jl` lines 742-929), factored into the session
phases: setup gates -> initialization -> initial candidate -> per-epoch
{ time limit, tau-collapse recovery, trial, typed dispatch, certificate
candidate } -> terminal.

Keyword arguments:

* `max_iterations`, `max_time`, `tol`, `initialization`,
  `max_tau_collapse_recoveries` — identical meaning and identical validation to
  the production driver.
* `stagnation_limit` — opt-in, default `0` (disabled).  When positive, the loop
  stops after that many consecutive rejected epochs and reports
  `SessionStagnated`.  Disabled by default so the default control flow is the
  production one.
* `trace` — opt-in `SessionTrace`; recording never touches the carrier.
* `trial_fn` — the epoch executor, production's `product_hsd_step!` by default.
"""
function solver_run_session!(
    session::SessionState{T};
    max_iterations::Integer=300,
    max_time::Real=Inf,
    tol::Union{Nothing,T}=nothing,
    initialization::Symbol=:auto,
    max_tau_collapse_recoveries::Integer=1,
    stagnation_limit::Integer=0,
    trace::Union{Nothing,SessionTrace{T}}=nothing,
    trial_fn::F=product_hsd_step!,
) where {T,F}
    state = session.hsd
    base = state.base
    initialization in (:auto, :identity, :kkt) || throw(ArgumentError(
        "initialization must be :auto, :identity, or :kkt",
    ))
    max_iterations >= 0 || throw(ArgumentError(
        "max_iterations must be nonnegative, got $max_iterations",
    ))
    max_tau_collapse_recoveries >= 0 || throw(ArgumentError(
        "max_tau_collapse_recoveries must be nonnegative",
    ))
    stagnation_limit >= 0 || throw(ArgumentError(
        "stagnation_limit must be nonnegative, got $stagnation_limit",
    ))
    time_limit = Float64(max_time)
    (isfinite(time_limit) || isinf(time_limit)) && time_limit >= 0.0 ||
        throw(ArgumentError("max_time must be nonnegative and finite, or Inf"))
    started_ns = time_ns()
    certificate_tol = tol === nothing ? T(default_certificate_tol(T)) : tol
    (isfinite(certificate_tol) && certificate_tol > zero(T)) ||
        throw(ArgumentError("tol must be finite and positive"))

    state.tau_collapse_recoveries = 0
    reset_phase_timings!(state.phase_timings)
    state.symmetric_core isa FixedTraceQ3CoreWorkspace &&
        _reset_q3_phase_timings!(state.symmetric_core)
    empty!(state.kkt_route_attempts)
    push!(state.kkt_route_attempts, state.kkt_route)
    x_original = alloc_zeros(T, base.n)
    s_original = alloc_zeros(T, base.m)
    y_original = alloc_zeros(T, base.m)
    accepted_steps = 0
    rejected_steps = 0
    consecutive_rejections = 0
    last_trial = TrialOutcome{T}()
    trace === nothing || empty!(trace)

    if base.workspace.rank_ambiguous
        result = _product_hsd_make_result(
            state, ProductHSDRankAmbiguous, ProductHSDRankAmbiguousSetup,
            HSDStepDirectionFailed, zero(T), x_original, s_original,
            y_original,
        )
        return _session_finish!(
            session, SessionSetupRejected, result, accepted_steps,
            rejected_steps, last_trial,
        )
    end
    if base.workspace.rank_incompatible
        copy_owned!(base.x, base.workspace.rank_ray)
        _product_hsd_bump_point_epoch!(state)
        if verify_dual_infeasibility!(
            base.canonical, base, x_original, s_original; tol=certificate_tol,
        )
            result = _product_hsd_make_result(
                state, ProductHSDDualInfeasible,
                ProductHSDVerifiedInitialPoint, HSDStepDirectionFailed,
                zero(T), x_original, s_original, y_original,
            )
            return _session_finish!(
                session, SessionSetupRejected, result, accepted_steps,
                rejected_steps, last_trial,
            )
        end
        result = _product_hsd_make_result(
            state, ProductHSDBreakdown,
            ProductHSDRankRayVerificationFailed, HSDStepDirectionFailed,
            zero(T), x_original, s_original, y_original,
        )
        return _session_finish!(
            session, SessionSetupRejected, result, accepted_steps,
            rejected_steps, last_trial,
        )
    end

    selected_initialization = initialization === :auto ?
        (state.kkt_route in (:expanded, :sparse_schur) ? :kkt : :identity) :
        initialization
    if selected_initialization === :kkt
        start_report = kkt_derived_start!(state)
        if !start_report.ok
            result = _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDBreakdown, ProductHSDKKTInitializationFailed,
                HSDStepDirectionFailed,
            )
            return _session_finish!(
                session, SessionSetupRejected, result, accepted_steps,
                rejected_steps, last_trial,
            )
        end
    else
        product_hsd_cold_start!(state)
    end
    # The initialization wrote the accepted iterate: bind it before the first
    # epoch so the session always owns a complete accepted point.
    solver_sync_kernel!(session)
    solver_bind_accepted!(session)
    initial = _product_hsd_candidate_result!(
        state, x_original, s_original, y_original, certificate_tol,
        ProductHSDVerifiedInitialPoint, HSDStepOK,
    )
    if initial !== nothing
        _session_record!(trace, session, last_trial, 0, :initial)
        return _session_finish!(
            session, SessionCertified, initial, accepted_steps,
            rejected_steps, last_trial,
        )
    end

    for iteration in 1:Int(max_iterations)
        elapsed_seconds = Float64(time_ns() - started_ns) * 1.0e-9
        if elapsed_seconds >= time_limit
            result = _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDTimeLimit, ProductHSDTimeLimitReached, HSDStepOK,
            )
            return _session_finish!(
                session, SessionTimeLimit, result, accepted_steps,
                rejected_steps, last_trial,
            )
        end
        if _product_hsd_tau_collapse_ready(state, certificate_tol)
            # The preceding accepted-point candidate gate already checked all
            # three certificate classes. Re-run the ray-only gates explicitly
            # before numerical recovery so a genuine infeasibility face is
            # never redirected into an optimal-face restoration.
            recovery = solver_tau_collapse_recovery!(
                session, certificate_tol, x_original, s_original, y_original,
                max_tau_collapse_recoveries - state.tau_collapse_recoveries,
            )
            if recovery.result !== nothing
                _session_record!(trace, session, last_trial, iteration, :recovery)
                return _session_finish!(
                    session, SessionPrecisionLimit, recovery.result,
                    accepted_steps, rejected_steps, last_trial,
                )
            end
            if recovery.status === RECOVERY_TAU_RECENTERED
                _session_record!(trace, session, last_trial, iteration, :recenter)
                continue
            end
            result = _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDInsufficientPrecision,
                ProductHSDTauCollapseRecoveryExhausted, HSDStepOK,
            )
            _session_record!(trace, session, last_trial, iteration, :recovery)
            return _session_finish!(
                session, SessionPrecisionLimit, result, accepted_steps,
                rejected_steps, last_trial,
            )
        end
        outcome = solver_run_trial!(session; trial_fn=trial_fn)
        last_trial = outcome
        if outcome.committed
            accepted_steps += 1
            consecutive_rejections = 0
        else
            rejected_steps += 1
            consecutive_rejections += 1
        end
        _session_record!(trace, session, outcome, iteration, :step)
        code = outcome.step_code
        if code === HSDStepSingularKKT
            result = _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDSingular, ProductHSDSingularKKTReason, code,
            )
            return _session_finish!(
                session, SessionStepFailure, result, accepted_steps,
                rejected_steps, last_trial,
            )
        elseif code === HSDStepBreakdown
            terminal = _product_hsd_terminal_verified_result!(
                state, x_original, s_original, y_original, certificate_tol,
                code,
            )
            if terminal !== nothing
                return _session_finish!(
                    session, SessionStepFailure, terminal, accepted_steps,
                    rejected_steps, last_trial,
                )
            end
            # Preserve the ordinary condition-aware trajectory and terminal
            # certificate opportunity first. Only a still-unverified generic
            # symmetric SOC iterate may replay the same Newton direction with
            # actual map checks as the acceptance authority.
            rescue = solver_conditioned_soc_rescue!(session)
            if rescue.status === RECOVERY_CONDITIONED_SOC
                accepted_steps += 1
                consecutive_rejections = 0
                _session_record!(trace, session, last_trial, iteration, :rescue)
                continue
            end
            result = _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDBreakdown, ProductHSDLineSearchBreakdown, code,
            )
            return _session_finish!(
                session, SessionStepFailure, result, accepted_steps,
                rejected_steps, last_trial,
            )
        elseif code === HSDStepDirectionFailed
            # A failed *next* Newton direction does not invalidate the current
            # accepted iterate. First run the authoritative original-coordinate
            # gates, then the existing finite terminal-trial verifier; only an
            # unverified pair may be reported as direction breakdown.
            current = _product_hsd_candidate_result!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDVerifiedAcceptedStep, code,
            )
            if current !== nothing
                return _session_finish!(
                    session, SessionStepFailure, current, accepted_steps,
                    rejected_steps, last_trial,
                )
            end
            terminal = _product_hsd_terminal_verified_result!(
                state, x_original, s_original, y_original, certificate_tol,
                code,
            )
            if terminal !== nothing
                return _session_finish!(
                    session, SessionStepFailure, terminal, accepted_steps,
                    rejected_steps, last_trial,
                )
            end
            result = _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDBreakdown, ProductHSDDirectionBreakdown, code,
            )
            return _session_finish!(
                session, SessionStepFailure, result, accepted_steps,
                rejected_steps, last_trial,
            )
        elseif code === HSDStepAlreadyOptimal
            verified = _product_hsd_candidate_result!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDVerifiedAcceptedStep, code,
            )
            if verified !== nothing
                return _session_finish!(
                    session, SessionStepFailure, verified, accepted_steps,
                    rejected_steps, last_trial,
                )
            end
            result = _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDBreakdown,
                ProductHSDUnverifiedZeroComplementarity, code,
            )
            return _session_finish!(
                session, SessionStepFailure, result, accepted_steps,
                rejected_steps, last_trial,
            )
        end

        if stagnation_limit > 0 && consecutive_rejections >= stagnation_limit
            result = _product_hsd_termination_or_dual_ray!(
                state, x_original, s_original, y_original, certificate_tol,
                ProductHSDMaxIterations, ProductHSDIterationLimitReached, code,
            )
            return _session_finish!(
                session, SessionStagnated, result, accepted_steps,
                rejected_steps, last_trial,
            )
        end

        verified = _product_hsd_candidate_result!(
            state, x_original, s_original, y_original, certificate_tol,
            ProductHSDVerifiedAcceptedStep, code,
        )
        verified === nothing || return _session_finish!(
            session, SessionCertified, verified, accepted_steps,
            rejected_steps, last_trial,
        )
    end

    if _product_hsd_tau_collapse_ready(state, certificate_tol)
        ray = _product_hsd_verified_result(
            state, x_original, s_original, y_original, certificate_tol,
            ProductHSDVerifiedTerminationRay, HSDStepOK;
            check_optimal=false,
        )
        solver_sync_kernel!(session)
        ray === nothing || return _session_finish!(
            session, SessionExhausted, ray, accepted_steps, rejected_steps,
            last_trial,
        )
        result = _product_hsd_termination_or_dual_ray!(
            state, x_original, s_original, y_original, certificate_tol,
            ProductHSDInsufficientPrecision,
            ProductHSDTauCollapseRecoveryExhausted, HSDStepOK,
        )
        return _session_finish!(
            session, SessionPrecisionLimit, result, accepted_steps,
            rejected_steps, last_trial,
        )
    end
    result = _product_hsd_termination_or_dual_ray!(
        state, x_original, s_original, y_original, certificate_tol,
        ProductHSDMaxIterations, ProductHSDIterationLimitReached, HSDStepOK,
    )
    return _session_finish!(
        session, SessionExhausted, result, accepted_steps, rejected_steps,
        last_trial,
    )
end
