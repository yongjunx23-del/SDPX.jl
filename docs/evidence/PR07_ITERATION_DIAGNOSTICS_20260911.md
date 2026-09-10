# PR-07 evidence: per-step iteration diagnostics

Date: 2026-09-11. Baseline HEAD `0148f1e`. **No algorithm or default changed.**

## What the plan asked for

PR-07's first requirement is instrumentation, not tuning: *"先记录每步真实
`mu_aff/mu, sigma_used, alpha_aff, alpha_combined, correction norm, backtracks,
retry reason`，不要只记录 requested setting"* — record what the step actually
did, not only which knobs were requested.

Before this change, `mu_aff` and `backtracking` were recorded, but `sigma_used`,
`alpha_aff`, `alpha_combined`, `correction_norm` and `retry_reason` were computed
in the predictor and discarded. A receipt could therefore report the requested
`iteration_knobs` but not the centering actually used.

## What was added

`HSDStepRecord` gains five fields (`sigma_used`, `alpha_aff`, `alpha_combined`,
`correction_norm`, `retry_reason`), written in place at the three predictor sites
and at line-search acceptance. All are scalars or a `Symbol`, so recording costs
no allocation and cannot change a trajectory. `sigma_used` and `alpha_aff` are
written where `_product_hsd_sigma` is evaluated; `alpha_combined` at the accepted
step; `correction_norm` is the trial/current merit ratio (scale-free, `NaN` when
either merit is unresolvable); `retry_reason` distinguishes
`:line_search_rejected` from `:runtime_invalidated` when an epoch accepts nothing.

All six new values are projected into the public
`diagnostics(result).termination`, so they are observable without touching
internals.

## A mistake made and caught here

The first attempt referenced `trial_merit` after the line-search loop. That
variable is assigned inside the accepted branch, so on any path that skipped it
the reference threw `UndefVarError: trial_merit`. The exception was swallowed by
`product_hsd_step!`'s existing catch block, so the symptom was not an error but a
**silent regression**: `soc_disk` went from `optimal`/24 iterations to
`numerical_breakdown`/0 iterations.

It was found by driving `product_hsd_step!` by hand and by diffing against a
stashed build, not by reading the code. The fix is an explicit initial value
before the loop. Recorded here because "a diagnostic field silently broke the
solver" is exactly the class of change the plan's fail-closed discipline exists
to catch, and because the catching mechanism (a bare `catch` that turns
exceptions into `accepted = false`) is worth knowing about.

## Semantics, stated honestly

The step record is a **per-step overwritten** structure, so the fields reflect
the most recent write per field rather than one atomic snapshot. On a terminal
epoch that rejects, `retry_reason` and `backtracking` describe that final
attempt while `alpha_combined` may still carry the last accepted step's value. A
true per-step trace would need the iteration-history channel the native engine
does not currently publish. This is a real limitation of the record, not of the
individual fields.

## Verification

`test/iteration_diagnostics.jl`, 34 assertions: every field present and typed;
values plausible (`mu > 0` finite, `0 < alpha_combined <= 1`, `backtracking >= 0`,
`retry_reason` in a known set); **diagnostics are not constants** (at least one
field must differ between a 3-dimensional and a 64-dimensional cone, so the
fields cannot pass by returning a placeholder); the solve is unchanged
(`optimal` + certified + `-1.0` objective); and `retry_reason` stays a real
condition even under a forced iteration budget.

Full suite re-run with the patch: the only failure is the pre-existing
dirty-worktree guard.
