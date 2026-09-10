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

## Corrections to the plan established during PR-00

Three plan claims were checked against the frozen revision and did not hold.
They are recorded here because a plan step that assumes an absent artifact is
unexecutable, and "the document says so" is not evidence.

1. **`src/factor_cache/routes/experimental_sparse_core.jl` does not exist**
   (plan F04, PR-03 落点, index S07). No file of that name at any tracked
   revision, and the identifier `ExperimentalSparseCoreCache` occurs nowhere in
   the source tree — only in the plan itself. The concrete type the plan
   describes is `SparseQDLDLCache` in
   `src/factor_cache/routes/qdldl_sparse.jl` (line 92).
2. **`src/factor_cache/session_symbolic_lease.jl` does not exist** (plan PR-08
   落点). Same verification. It exists only on unmerged branches.
3. **`docs/evidence/P3_01_BETA_EXPERIMENT.md` does not exist** (plan F06, index
   S10). No tracked file, at any revision, records the beta numbers the plan
   cites (105 / 107 / 118 steps). The plan's own F06 caveat — that these are
   "existing repository records" — is therefore unsupported here. F06's
   *conclusion* (do not raise beta by default) is separately defensible, but
   its stated evidence is not present in this repository.

Consequence for the plan sequence: PR-03 cannot "advance the existing
experimental wrapper" because the wrapper is not in the frozen tree. PR-03 must
either revive the unmerged branch deliberately (with its own review) or target
`SparseQDLDLCache` directly. PR-08 must build the lease mechanism rather than
extend it.

## Claim boundaries

* SDPX remains MIT for its own code. Imported Apache-2.0 material is not
  relicensed by the top-level `LICENSE`; see `THIRD_PARTY_NOTICES.md`.
* Nothing in this ledger asserts a performance result. Item 1 establishes an
  algebraic identity only; item 2 reduces startup factor count from two to one
  on the KKT-start path and does not by itself claim a wall-clock win.
* The plan's Section 3.1 sign convention was corrected in
  `docs/design/frozen_math_contract.md`; the executable source uses
  `-c'*dx - b'*dy + dκ = r_g`.
