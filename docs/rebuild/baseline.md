# A00 baseline freeze

**Frozen: 2026-09-11. Task A00 of the infrastructure rebuild packet
(`sdpx_infrastructure_review`).**

This file is A00's first acceptance item: "三仓完整SHA、provider版本、旧性能基线及
未验证能力有明确清单". It records what actually exists, what was measured, and —
just as importantly — what was NOT verified.

---

## 1. Repository freeze

| Alias | Repository | Frozen SHA | How verified |
|---|---|---|---|
| SDPX | `yongjunx23-del/SDPX.jl` | `5dd7a713078949380bffabd3c3a80b454716d16f` | `git rev-parse HEAD`, clean tree, `Pkg.test()` green |
| MFLA | `yongjunx23-del/MultiFloatLinearAlgebra.jl` | `50e6e0b` | local HEAD |
| BFLA | `yongjunx23-del/BigFloatLinearAlgebra.jl` | `f95d3e6` | local HEAD |
| Clarabel (reference only) | `yongjunx23-del/Clarabel.jl` | `fbc5dd0` | local HEAD |

### 1.1 SDPX: the packet's SHA is stale and the baseline moved

The packet (`sources.json`, `tasks.json:repo_aliases`) names SDPX
`4b46cda1aa752aced3cbd802c2f75a8a850d0e58`. **That SHA is real and is
`origin/main`** — unlike the earlier round's finding, which was retracted in
`docs/evidence/BASELINE_CORRECTION_20260911.md`.

The local `main` was nevertheless 283 commits behind it. A00's instruction is
"以语义重定位本文锚点，不强制回退到本次审查SHA", so the anchors were relocated
rather than the tree rolled back:

- The packet's SDPX anchors were re-read **at `4b46cda`** and confirmed accurate.
- Seventeen genuinely novel commits from the previous round were **rebased from
  `db42fd2` onto `4b46cda`**; six commits that duplicated upstream work were
  dropped (their content was byte-identical to files already upstream — SHA-256
  verified on four test files).
- The frozen SDPX baseline for this packet is therefore **`5dd7a713078949380bffabd3c3a80b454716d16f`**
  (`4b46cda` + 17 rebased commits), not `4b46cda` alone and not `db42fd2`.

Rationale for rebasing rather than rolling back, per the packet's own rule: the
rebased work is not duplicated upstream. Verified at `4b46cda`: the KKT start
still double-factorizes (`factorize_pivoted_ldl!` *and* `factorize_pivoted_lu!`
in `src/hsd/initialize.jl`), there is no residual-freshness token in
`src/hsd/hsd.jl`, no `sigma_used`/`correction_norm`/`retry_reason`, and
`_block_shape_code` still accepts only `:dense_lower`.

**MFLA drift — resolved as far as A00 can resolve it.**

The packet freezes MFLA at `b38dea1` (v0.4.0, 2026-08-30); the local checkout is
at `50e6e0b` (2026-09-01). The drift is exactly three commits:

| Commit | Subject |
|---|---|
| `920f810` | `bench(phase5): add Float64x4 kernel microbenchmark suite` |
| `5885060` | `perf(ldlt): thread independent weighted-panel rows` |
| `50e6e0b` | `test(phase5): document triangular solve parallelism ceiling` |

Diffstat: 6 files, +207/−18. `Project.toml` version is **`0.4.0` at both
revisions** — the threading behaviour changed without a version bump, which is
itself worth noting for any capability claim keyed on version.

Scope of the numeric change: `src/factorizations/ldlt.jl` has a **single hunk**
at `_factor_ldlt_panel!` (around lines 592–667). Checked explicitly: the diff
contains **zero** occurrences of `_bk_`, `pivot`, `2x2` or `normalize`. So the
BK pivot grammar — the subject of anchor **S12**, which task A01 uses to build
its 2×2 pivot oracle — is **unchanged** by the drift. A01's oracle therefore
remains valid against the local checkout.

BFLA: **no drift at all**. Local `f95d3e6` equals the frozen SHA exactly.

Consequence, binding on M01–M03, P02 and B03:
- The drift is threading-only. A task claiming a *numeric* result may measure at
  either revision, because the pivot grammar is identical.
- A task claiming a *threading or performance* result **must** state which
  revision it measured. `5885060` changes threaded weighted-panel rows, so a
  parallelism claim at `b38dea1` would be a claim about code that lacks the
  change.
- No result may be attributed to `b38dea1` if it was measured at `50e6e0b`.

## 2. Provider versions

`Manifest.toml` at the frozen SDPX revision does not resolve
`MultiFloats`, `MultiFloatLinearAlgebra`, `BigFloatLinearAlgebra` or `QDLDL`:
they are `[weakdeps]` and are absent from the default environment. This is a
first-class fact for the packet, because it means:

- The default `Pkg.test()` path exercises **Float64 only**.
- Every MF/BF capability claim is unverified in the default environment.
- `scripts/provider_smoke.sh` builds a **separate** environment via
  `Pkg.develop` on sibling checkouts plus `Pkg.add(["MultiFloats",
  "GenericLinearAlgebra"])`. The packet's `scripts/bootstrap_env.jl` would be a
  third such environment and has **not been executed**.
- `.github/workflows/provider-matrix.yml` is **manual-only by policy**
  (comment dated 2026-09-01): "the MFLA legs of the matrix fail on every CI".
  So there is currently **no automated MF/BF coverage at all**.

Consequence for the packet: ADR-002 must not assume a provider environment
exists, and any task whose acceptance depends on MF/BF must state how that
environment is produced and record its Manifest.

## 3. Inherited gates

These existed before this packet and are inherited, not invented here.

| Gate | Location | Status |
|---|---|---|
| Public E2E correctness + original-coordinate certificate | `test/runtests.jl` (`Pkg.test()`) | **green** at the frozen revision |
| Platform/thread matrix | `.github/workflows/test.yml` (1 and 4 threads) | automated |
| Sparse-provider smoke | `.github/workflows/provider-matrix.yml` | **manual-only**, MFLA legs known failing |
| Provider smoke script | `scripts/provider_smoke.sh` | manual |
| Single-factor KKT start counting | `test/initialize_counting.jl` | green (rebased work) |
| Accepted-point residual lifecycle invariant | `test/accepted_point_reuse.jl` | green (rebased work) |
| SOC rank-2 expansion algebra | `validation/clarabel_borrowing/soc_rank2_gate.jl` | 375 pass; **not wired into `Pkg.test`** |
| Start-point comparison | `benchmark/clarabel_borrowing/start_point_comparison.jl` | measured; **not a gate** |
| Allocation ceiling (Float64) | `benchmark/clarabel_borrowing/allocation_gate.jl` | 16 B/step; **not wired into CI** |

A00 rule carried forward: an inherited gate is only inherited if it is actually
run. The three rows marked "not wired" are evidence artefacts, not gates, until
a task wires them.

## 4. Unverified and partially verified capabilities

Recorded because the packet forbids presenting a prototype as production.

| Capability | Status | Evidence |
|---|---|---|
| SDPX Float64 public path (LP/SOC/SDP/Pow/Exp) | **verified** | `Pkg.test()` green |
| SDPX multi-precision public path | **partially verified** | default env is Float64-only; MF/BF need the provider env, which is not in CI |
| Sparse route beyond Float64 | **unverified** | `:sparse_augmented` is fail-closed for every non-Float64 `T` |
| `SparseQDLDLCache` | **unverified in CI** | exercised only when MFLA+QDLDL are installed; skips otherwise |
| Session symbolic lease | **exists but UNWIRED** | `src/factor_cache/session_symbolic_lease.jl` header states "Not wired into Prepared execution yet" |
| MFLA `b38dea1` vs `50e6e0b` | **resolved (threading-only)** | 3 commits, single hunk in `_factor_ldlt_panel!`; BK pivot grammar unchanged; version not bumped |
| BFLA local SHA | **matches** | `f95d3e6` = frozen |
| PSD large-scale storage | **not characterised** | see PR-09 evidence gap; no measurement recorded |
| Thread scaling 16/64 | **not measurable here** | `Sys.CPU_THREADS` reports 4 on this host |

## 5. What A00 did NOT do

- Did not run the packet's `scripts/bootstrap_env.jl`; no joint environment exists.
- Did not resolve the MFLA SHA drift (§1.1).
- Did not run any MF/BF test; the provider environment is not built.
- Did not re-measure any performance number. The start-point comparison and
  allocation figures cited are from the rebased work at the same revision.
- Did not delete, retire, or rename any source file. That is I01/I03's authority.
