# Clarabel borrowing: provenance and claim boundaries

**Established by PR-00. Last updated 2026-09-11 at HEAD `5d233c8`.**

This document is the per-item ledger required by Section 8 of
`SDPX_Clarabel_engineering_plan_2ab596f.md`. Every borrowed item records the
upstream revision, the original path, the local path, the category
(verbatim / adapted / independent), the key change, and the covering test.

The machine-readable, hash-pinned form of the source index is
`source_manifest.json` at the repository root.

## Pinned revisions

| Repository | Revision | Version | License |
|---|---|---|---|
| SDPX.jl | working tree (`source_manifest.json:repositories`) | 0.6.1 | MIT |
| Clarabel.jl (fork `yongjunx23-del/Clarabel.jl`) | `fbc5dd01576d47fda53861a992a02e93c8dbd03e` | 0.11.1 | Apache-2.0 |

The plan's stated SDPX baseline `2ab596fe360fc394698582e7d9cc4a4f548b5386`
**does not exist in this repository** (`git cat-file` fails; `git log --all`
does not contain it). The real starting point was `db42fd2` plus 30
uncommitted changes, which were classified into seven commits
(`b587b1b`, `30ed01a`, `8f7a5ee`, `195e39b`, `5c911c8`, `61e3336`, `e49f88b`)
before any plan work began.

## Ledger

| # | Clarabel source (revision `fbc5dd0`) | Local artifact | Category | Key change | Covering test |
|---|---|---|---|---|---|
| 1 | `src/cones/coneops_socone.jl` — `update_scaling!` rank-2 block; `src/kktsolvers/direct-ldl/directldl_datamaps.jl` — `SOCExpansionMap` | `validation/clarabel_borrowing/soc_rank2_gate.jl` | **Independent reimplementation** | Formula re-derived from the definition `Theta = 2ww' - J` in a self-contained gate that calls no SDPX production code | `soc_rank2_gate.jl`, 375 assertions |
| 2 | `src/kktsolvers/kktsolvers_directldl.jl` — single factor authority | `src/hsd/initialize.jl`, `src/kkt/expanded_quasidefinite.jl` | **Independent** | SDPX reuses its *own* `GenericPivotedLDL` for the RHS solves; no Clarabel code copied | `test/initialize_counting.jl` |
| 3 | `src/kktsystem.jl` — KKT structure | SDPX `src/kkt/system.jl` | **Independent** (design comparison only) | SDPX five-equation structure predates this work and was **not** changed | `validation/newton_system_reference.jl` |

**No Clarabel source file has been copied verbatim into SDPX.** If item 1 is
next turned into a production adapter, that adapter is **adapted** material and
must carry the in-file Apache-2.0 attribution note described in
`THIRD_PARTY_NOTICES.md`, plus its own gate.

## RETRACTED: the "corrections to the plan" that were published here

**Everything in the original version of this section has been retracted.** It
claimed that three artifacts cited by the plan did not exist and that the plan's
baseline SHA was invalid. All of those claims were artifacts of a stale local
checkout and were wrong.

See `docs/evidence/BASELINE_CORRECTION_20260911.md` for the full account. In
summary, the earlier round worked from `db42fd2` + 30 uncommitted changes, which
is **283 commits behind `origin/main`**; the plan had been written against
`origin/main` at `4b46cda`. Verified since:

| Retracted claim | Truth |
|---|---|
| baseline `2ab596f` absent | present on `origin/main` |
| `experimental_sparse_core.jl` absent | present on `origin/main`, with a 5-file test family |
| `session_symbolic_lease.jl` absent | present on `origin/main`, already included by `test/runtests.jl` |
| `P3_01_BETA_EXPERIMENT.md` absent | present, containing the cited `0.98 -> 118` result |
| Section 3.1 sign contradicts source | the plan is correct; the sign was changed upstream by `3392e24` |

The 30 "uncommitted changes" were likewise not local work: they are
**byte-identical** to files already committed upstream.

**PR-03 and PR-08 do NOT need re-scoping.** Their stated landing points exist.
The earlier instruction to that effect is withdrawn.

## Claim boundaries

* SDPX remains MIT for its own code. Imported Apache-2.0 material is not
  relicensed by the top-level `LICENSE`; see `THIRD_PARTY_NOTICES.md`.
* Nothing in this ledger asserts a performance result. Item 1 establishes an
  algebraic identity only; item 2 reduces startup factor count from two to one
  on the KKT-start path and does not by itself claim a wall-clock win.
* The plan's Section 3.1 sign convention was corrected in
  `docs/design/frozen_math_contract.md`; the executable source uses
  `-c'*dx - b'*dy + dκ = r_g`.
