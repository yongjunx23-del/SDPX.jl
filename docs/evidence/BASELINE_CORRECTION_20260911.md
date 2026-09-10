# Correction: the previous round's "plan errata" was wrong

Date: 2026-09-11. This document retracts five findings published earlier the
same day in `docs/design/clarabel_borrowing.md`, `docs/design/frozen_math_contract.md`
and the errata section of `SDPX_Clarabel_engineering_plan_2ab596f.md`.

## What happened

The earlier round worked from a local `main` at `db42fd2` plus 30 uncommitted
changes. That base was **283 commits behind `origin/main`**, and the review that
produced the Clarabel-borrowing plan had been performed against `origin/main`
at `4b46cda`. Every "the plan is wrong" finding was therefore a comparison
between the plan's text (correct for `4b46cda`) and code from `db42fd2`.

| # | Earlier claim | Truth |
|---|---|---|
| 1 | The plan's baseline `2ab596fe360fc394698582e7d9cc4a4f548b5386` does not exist | **It exists**, on `origin/main`. `git log` on the stale clone could not see it. |
| 2 | `src/factor_cache/routes/experimental_sparse_core.jl` and `ExperimentalSparseCoreCache` do not exist | **They exist** on `origin/main`. There is also a `test/` family for them (`experimental_sparse_core*.jl`, 5 files). |
| 3 | `src/factor_cache/session_symbolic_lease.jl` does not exist | **It exists** on `origin/main`, and `test/runtests.jl` already includes `session_symbolic_lease.jl`. |
| 4 | `docs/evidence/P3_01_BETA_EXPERIMENT.md` does not exist, so F06's beta numbers are unsourced | **It exists**, and contains exactly the cited results: `0.98 \| optimal \| 118 \| valid`. |
| 5 | Section 3.1's sign claim (`c'dx + b'dy + dκ`) contradicts the source | **The plan is right.** The source used `-c'dx - b'dy + dκ` at `db42fd2`, and the sign was changed to `+` upstream by `3392e24 kkt: define semantic five-equation Newton system`, which landed after `db42fd2` and before the plan's baseline. |

Five out of five were wrong, and none was a defect in the plan.

## How the error was made

The first round treated `git cat-file` failing in the local clone as proof that
an object does not exist. It is not: it proves only that the object is absent
from *this* clone. The correct check is `git fetch` followed by a search across
remote refs. That step was skipped, and a stale checkout was silently treated as
authoritative.

The second contributing factor: the 30 "uncommitted changes" that the first
round classified into six commits were **byte-identical to files already
committed upstream** (verified by SHA-256 on `test/gap_normalization.jl`,
`test/certificate_scratch_ownership.jl`, `test/multifloat_trial_tail.jl`,
`test/certificate_layout_storage.jl` — all identical). They were not local work
at all; they were upstream commits the stale clone had never pulled. That should
have been the signal that the base was wrong.

## What survives

The findings that were derived from *measurement on the frozen tree* rather than
from comparing against remembered plan text:

- The SOC rank-2 expansion algebra and the **measured storage crossover k = 6**
  (`validation/clarabel_borrowing/soc_rank2_gate.jl`, 375 assertions). This is a
  quantified refinement of the plan's unstated threshold, not a contradiction.
- The derived SDPX SOC rank-2 mapping (`src/cones/symmetric/soc_rank2.jl`),
  verified to 2.2e-16, and the finding that **Clarabel's published `(D,u,v)`
  formulas do not fit SDPX's `theta_apply!`**. This remains a genuine trap and is
  the single most valuable result of the round. It is re-verified against the
  rebased tree.
- The PR-01 soundness argument (the certificate path writes `rP`/`rD` with a
  different accumulation association, so a naive residual cache is unsound) —
  re-verified as still true upstream.
- The PR-04B measurement (no consistent winner between the two starts).

## Action taken

The 17 genuinely novel commits were rebased from `db42fd2` onto `4b46cda`; the
six duplicate classification commits were dropped. The rebase is verified
separately in the accompanying suite run and smoke test. The retracted claims
are corrected in place in the three documents named above.
