# ===========================================================================
# S02-b (2/2) — recovery as typed, dispatchable strategies.
#
# Recovery MATHEMATICS IS UNCHANGED (card step 3).  Each function here is a
# one-to-one transcription of a recovery branch that already exists in the
# production driver, with the control flow made explicit and the outcome typed:
#
#   * tau-collapse recovery   — driver lines 823-843
#       `_product_hsd_tau_collapse_ready` -> verifier-only ray gate
#       -> `_product_hsd_tau_collapse_recenter!` (bounded) -> insufficient
#       precision terminal.
#   * conditioned-SOC rescue  — driver lines 860-868
#       `_product_hsd_line_search!(state; allow_conditioned_soc=true)` under the
#       exact same eligibility guard.
#
# Only the branch conditions and the status promotion rules are reproduced; no
# tolerance, damping or centering constant is touched, and no recovery path can
# publish a certificate status by itself.
# ===========================================================================

"""
    RecoveryStatus

* `RECOVERY_NONE` — no recovery applies to this epoch.
* `RECOVERY_TERMINAL_VERIFIED` — a verified terminal result was produced; the
  loop must stop with the carried result.
* `RECOVERY_TAU_RECENTERED` — one bounded tau-collapse recentering was applied;
  the loop must `continue` without consuming a step.
* `RECOVERY_CONDITIONED_SOC` — the conditioned-SOC replay accepted a step; the
  loop must `continue`.
* `RECOVERY_EXHAUSTED` — the recovery budget for this condition is spent.
"""
@enum RecoveryStatus::UInt8 begin
    RECOVERY_NONE
    RECOVERY_TERMINAL_VERIFIED
    RECOVERY_TAU_RECENTERED
    RECOVERY_CONDITIONED_SOC
    RECOVERY_EXHAUSTED
end

"""
    RecoveryOutcome{T}

Typed recovery decision.  `result` is non-`nothing` only for
`RECOVERY_TERMINAL_VERIFIED`, and it is always a result produced by the
unchanged certificate verifiers.
"""
struct RecoveryOutcome{T}
    status::RecoveryStatus
    result::Union{Nothing,ProductHSDSolveResult{T}}
    diagnostic::Symbol
    recentered::Bool
end

RecoveryOutcome{T}(status::RecoveryStatus, diagnostic::Symbol) where {T} =
    RecoveryOutcome{T}(status, nothing, diagnostic, false)

"""
    solver_tau_collapse_recovery!(session, tol, x_original, s_original, y_original,
                                  remaining) -> RecoveryOutcome

Transcription of the driver's tau-collapse branch.  The ray gate runs *before*
recovery so a genuine infeasibility face is never redirected into an optimal-face
restoration; the number of recenterings is bounded by `remaining`.
"""
function solver_tau_collapse_recovery!(
    session::SessionState{T},
    tol::T,
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    remaining::Integer,
) where {T}
    state = session.hsd
    base = state.base
    _product_hsd_tau_collapse_ready(state, tol) ||
        return RecoveryOutcome{T}(RECOVERY_NONE, state.diagnostic)
    ray = _product_hsd_verified_result(
        state, x_original, s_original, y_original, tol,
        ProductHSDVerifiedTerminationRay, HSDStepOK;
        check_optimal=false,
    )
    solver_sync_kernel!(session)
    ray === nothing ||
        return RecoveryOutcome{T}(
            RECOVERY_TERMINAL_VERIFIED, ray, state.diagnostic, false,
        )
    if remaining > 0 && _product_hsd_tau_collapse_recenter!(state)
        solver_sync_kernel!(session)
        solver_bind_accepted!(session)
        return RecoveryOutcome{T}(
            RECOVERY_TAU_RECENTERED, nothing, state.diagnostic, true,
        )
    end
    solver_sync_kernel!(session)
    return RecoveryOutcome{T}(
        RECOVERY_EXHAUSTED, nothing, state.diagnostic, false,
    )
end

"""
    solver_conditioned_soc_rescue!(session) -> RecoveryOutcome

Transcription of the driver's conditioned-SOC replay branch, under the same
eligibility guard: no prepared fixed-trace core, at least one SOC block, and no
Exp/Power block.  The replay reuses the *same* Newton direction with actual map
checks as the acceptance authority; it is not a new direction computation.
"""
function solver_conditioned_soc_rescue!(
    session::SessionState{T},
) where {T}
    state = session.hsd
    eligible = !(state.symmetric_core isa FixedTraceQ3CoreWorkspace) &&
               !isempty(state.runtime.soc) &&
               isempty(state.runtime.exp) && isempty(state.runtime.power)
    eligible ||
        return RecoveryOutcome{T}(RECOVERY_NONE, state.diagnostic)
    # Deliberately NOT wrapped in a try/catch: the production driver calls this
    # replay bare (src/hsd/product_cone_solve.jl line 864), and swallowing an
    # exception here would silently convert a programmer error into "no
    # recovery", which is a behaviour change rather than an extraction.
    accepted = _product_hsd_line_search!(state; allow_conditioned_soc=true)
    if accepted
        solver_sync_kernel!(session)
        solver_bind_accepted!(session)
        return RecoveryOutcome{T}(
            RECOVERY_CONDITIONED_SOC, nothing, state.diagnostic, false,
        )
    end
    solver_sync_kernel!(session)
    return RecoveryOutcome{T}(RECOVERY_NONE, state.diagnostic)
end
