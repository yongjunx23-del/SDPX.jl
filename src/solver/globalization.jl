# ===========================================================================
# S02-b (1/2) — the unified accepted / trial / rollback lifecycle.
#
# Before S02 the lifecycle was spread over the driver's if/elseif ladder
# (src/hsd/product_cone_solve.jl lines 844-909) and over the per-step
# bookkeeping inside `product_hsd_step!`.  This file gives it one typed
# identity: one phase function per epoch, one typed outcome, one place that
# decides "accepted" versus "rejected" and that owns the rollback receipt.
#
# WHAT IS NOT CHANGED (card step 3): sigma, beta, initialization, recovery
# mathematics and the default provider.  The acceptance arithmetic itself stays
# in `_product_hsd_line_search!` / `product_hsd_step!`; this file only
# *classifies* their result.
#
# The classification is derived from the lifecycle token, not from a convention:
# the line search commits an accepted step by writing the accepted iterate and
# bumping `point_epoch`, so
#
#     committed == (point_epoch_after != point_epoch_before)
#
# is the authoritative "did the trial become the accepted point" test.
# ===========================================================================

"""
    TrialStatus

* `TRIAL_NONE` — no epoch has been executed yet.
* `TRIAL_ACCEPTED` — the line search committed a new accepted iterate and the
  session bound it.  The accompanying `step_code` is `HSDStepOK` on the normal
  path; a failure code with `committed == true` means the iterate moved and a
  later gate rejected the epoch (production keeps that iterate, and so does the
  session).
* `TRIAL_REJECTED` — nothing was committed; the last accepted point is intact
  (verified bitwise) and a rollback receipt was recorded.
"""
@enum TrialStatus::UInt8 begin
    TRIAL_NONE
    TRIAL_ACCEPTED
    TRIAL_REJECTED
end

"""
    TrialOutcome{T}

Typed outcome of one predictor/corrector epoch.
"""
struct TrialOutcome{T}
    status::TrialStatus
    step_code::HSDStepCode
    alpha::T
    backtracking::Int
    point_epoch_before::Int
    point_epoch_after::Int
    committed::Bool
    residual_refreshed::Bool
    rollback_repaired::Bool
    diagnostic::Symbol
end

TrialOutcome{T}() where {T} = TrialOutcome{T}(
    TRIAL_NONE, HSDStepOK, T(NaN), 0, -1, -1, false, false, false, :none,
)

"""
    solver_run_trial!(session; trial_fn = product_hsd_step!) -> TrialOutcome

Execute one epoch through the unified lifecycle:

1. entry residual gate (identical predicate to production's, so it recomputes on
   exactly the same occasions);
2. the direction/line-search executor (`trial_fn`, production's
   `product_hsd_step!` by default);
3. classify by the lifecycle token;
4. on commit, bind the complete accepted state; on rejection, record the trial
   and run the (verified) rollback.

`trial_fn` is injectable so the rejected, exhausted and rollback paths are
testable without a second production loop and without perturbing a real solve.
"""
function solver_run_trial!(
    session::SessionState{T};
    trial_fn::F=product_hsd_step!,
) where {T,F}
    state = session.hsd
    base = state.base
    before = base.point_epoch
    residual_refreshed = solver_ensure_residual!(session)
    code = trial_fn(state)::HSDStepCode
    after = base.point_epoch
    committed = after != before
    # The step's trailing `_product_hsd_residual!` (or a certificate-flavoured
    # writer inside the step) changed the token; reconcile before anyone reads
    # freshness again.
    solver_sync_kernel!(session)
    solver_capture_trial!(session, code)
    if committed
        session.trial.committed = true
        solver_bind_accepted!(session)
        return TrialOutcome{T}(
            TRIAL_ACCEPTED, code, base.record.alpha_combined,
            base.record.backtracking, before, after, true,
            residual_refreshed, false, state.diagnostic,
        )
    end
    session.rejected_trials += 1
    repaired = solver_rollback!(session)
    return TrialOutcome{T}(
        TRIAL_REJECTED, code, base.record.alpha_combined,
        base.record.backtracking, before, after, false,
        residual_refreshed, repaired, state.diagnostic,
    )
end

"""
    solver_trial_accepted(outcome) -> Bool
"""
@inline solver_trial_accepted(outcome::TrialOutcome)::Bool =
    outcome.status === TRIAL_ACCEPTED
