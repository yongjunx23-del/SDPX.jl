# I03 — final release verdict: SDPX / MFLA / BFLA

**Verdict: BLOCKED.** The release decision is *not* frozen as released. Two rows of
`rebuild-reports/I03_RELEASE_DECISION.md` §4 are unconditional blockers and both are
unmet: **R1** (ADR-003 §5.1 — every public route terminates with a legal
original-coordinate certificate) and **R2** (S01-P2 — the public layer applies no sign
patch of its own). **R8** and **R9** are also unmet and are recorded below as
*not accepted* rather than waved through. A BLOCKED verdict is the outcome the packet's
own rule requires (必要能力未完成则保持发布阻塞), and the goal this task serves says so in
as many words: if R1 is unmet or R2 is still blocked at I03, the decision must stay
blocked rather than be released.

This document is the *verdict*. `rebuild-reports/I03_RELEASE_DECISION.md` is the
**criteria** document and is not overwritten by this file. Nothing here re-approves what
V01 audited; V01 is cited (`SDPX.jl/docs/rebuild/independent_audit.md`,
`rebuild-reports/V01/report.json`, 16 commands / 12 findings).

Order actually followed: **(a) delete the carriers → (b) re-measure → (c) decide.**
Step (b) is *partly* mine and *partly* handed back: I03 cannot commit, and SDPX's suite
refuses to run on a dirty tree (`test/runtests.jl:29` includes
`benchmark/optimization/test_v2_fresh_process_profile.jl`, which asserts
`_require_clean_source("test_clean")` at `:215`). So the carrier deletions are applied
and their targeted drivers re-run by me, and the three clean-tree suites plus the 28-leg
matrix are the parent's step at the commit that contains them. The one row that needs
those numbers (**R3**) is marked `not_measured`, not `satisfied`.

## 1. Revisions, and what is measured at which one

| object | revision | state |
| --- | --- | --- |
| SDPX, the revision this verdict's *measurements* were taken against | `c8fb65a0d5e15b6d03d34c4368cba7ad5f697094` (clean) | the deletion tree is `c8fb65a` + the 8 paths in §4 |
| SDPX, the release revision I03 proposes | the commit the parent makes from §4 (B1+B2+B3); **not yet existing** | named as a placeholder, never as a number |
| MFLA | `e3805c9607295f0e173567b060d4035e826310ca` | clean; **untouched by I03** |
| BFLA | `f087a72f2001088ea520b67588cf18c0d3fce2e8` | clean; **untouched by I03** |
| MFLA/BFLA as measured by the parent | `e3805c9` / `f087a72` | `Pkg.test()` logs in `rebuild-reports/PARENT_VERIFICATION/` |

Two counting rules are applied throughout, because the packet has been burned by both:
**(1)** every number names the revision it was measured at; **(2)** any thread-count claim
names its denominator — this host reports `Sys.CPU_THREADS == 4` on 10 OS cores, so tiers
16 and 64 are `unsupported`, never `pass` and never `0`.

I03's own raw logs are under `rebuild-reports/I03/logs/`:
`pre_delete/` (at `c8fb65a`, clean), `post_delete/` (after the deletions, before the
comment corrections), `final_tree/` (the exact tree handed back for commit).
`deletions.diff` and `deletions_paths.txt` are the change record; `i03_R2_sign_patch_sweep.log`
is the R2 sweep.

## 2. The release decision, R1–R9, each with the command that established it

Every row carries a command. `not_measured` is used where the honest answer is "not
established here"; it is never used to mean "0".

| # | status | command(s) that established it | what the command shows |
| --- | --- | --- | --- |
| **R1** | **not_satisfied** | `JULIA_DEPOT_PATH=$REBUILD_DEPOT:$HOME/.julia julia --project=$REBUILD_ENV -t1 /tmp/v01_t7_certificate.jl` → `rebuild-reports/I03/logs/final_tree/R1_certificate_routes.log` (exit 0); `grep -rn 'ResultCertificate{' SDPX.jl/src/`; `grep -rn 'SDPXCertification\.' SDPX.jl/src SDPX.jl/test \| grep -v certification/original.jl`; `sed -n '2576,2586p' SDPX.jl/src/hsd/native_hsd_public.jl` | Four routes (LP, SOC, SDP, infeasibility ray) terminate `cert_valid=true method=original_coordinates` — that is V01's F12/T7 positive half, independently re-run here. But the criterion is *every* route: the downgrade branch `native_hsd_public.jl:2580-2583` (status → `NumericalFailure`, reason `:original_coordinate_certificate_failed`) **has never been driven**, and four `ResultCertificate{T}` construction sites survive (`src/hsd/native_hsd_public.jl:2403`, `:2462`, `:2487`, `src/public/optimize.jl:341`). `grep` for `SDPXCertification.` outside its own file returns **0**, so the module is inert on the public path. The blocker is a **missing instrument**, not missing effort: I02 declined move 6 deliberately because a behaviour change to existing code needs its own before/after, and the plan marks it "deliberately not drafted". My tight-gap control (`gap_limit=1e-14`) did *not* exercise the invalid direction — the certificate was still valid — so the negative direction remains `not_run` |
| **R2** | **not_satisfied** | `rebuild-reports/I03/logs/i03_R2_sign_patch_sweep.log` (commands reproduced verbatim in the log): 8 patterns `'\.= *-' '= *-x' '= *-1 *\.\*' '\*= *-1' sign_flip flip_sign negate public_sign_patch` over `src/public/` (4 files) + `src/moi_wrapper.jl` + `src/frontend/` (2 files); `sed -n '470,476p' SDPX.jl/src/core/compiled_problem.jl`; `sed -n '472,475p' SDPX.jl/test/rebuild/S01.jl`; `grep -rn 'replay_public_signs' SDPX.jl/src SDPX.jl/test` | **8 patterns → 8 zeros.** There is no sign patch to delete, so the prescribed before/after MOI solve would compare a state to itself and can never unblock this row (V01 T8a, independently re-run here). `public_sign_patches=0` is still a **hardcoded literal** at `src/core/compiled_problem.jl:475` asserted against itself at `test/rebuild/S01.jl:474` — an assertion that cannot fail, and it is **not** cited as evidence here. `replay_public_signs` is defined at `compiled_problem.jl:449` and used at exactly one site, `S01.jl:473`, from a constructed `CompiledProblem`; it never inspects the public boundary. The closest thing to a sign correction in the boundary — the MOI interval dual at `src/moi_wrapper.jl:1845` — is a *sum* of two bridge duals, i.e. MOI conforming, not a patch. What is missing is the runtime observation: a count of sign applications at the public boundary during a real MOI solve |
| **R3** | **not_measured** | not run by I03 — the tree is dirty by construction and `_require_clean_source` is asserted inside the suite (`benchmark/optimization/test_v2_fresh_process_profile.jl:210-215`). Command for the parent, at the post-commit SHA: `cd SDPX.jl && JULIA_DEPOT_PATH=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env-depot:$HOME/.julia julia --project=. -e 'using Pkg; Pkg.test()'`; and for MFLA/BFLA the same from their directories; then `bash SDPX.jl/scripts/rebuild/run_driver_matrix.sh rebuild-reports/PARENT_VERIFICATION/driver_matrix_post_i03` | Pre-deletion, at `c4b109a`: SDPX `tests passed`, 170 testsets, `failcols=0`, **Broken=7 / Pass=9392 / Total=9399**; MFLA `e3805c9` **Pass=4164 / Total=4164**; BFLA `f087a72` **10864/10864**; matrix `legs_run=28 legs_failed=0 failed_legs=none`, every leg with a `Test Summary` (`rebuild-reports/PARENT_VERIFICATION/`). My **post-deletion targeted** legs are all exit 0: load, R1 probe, S01 189, S03 193, S04 313, S06 214, S07 289, S05_none 1466, S05_mfla/S05_bfla 1522, A01_default 1673 (`rebuild-reports/I03/logs/final_tree/`). That is targeted evidence, not the row: **R3 stays `not_measured` until the suites run on a clean tree at the new revision** |
| **R4** | **not_satisfied** | `python3 SDPX.jl/scripts/rebuild/check_reconstruction.py --record SDPX.jl/docs/rebuild/RELEASE_REVISIONS.txt --target /tmp/q02recon --depot /tmp/q02recon-depot` → `rebuild-reports/Q02/logs/recon_positive.log`, controls `recon_controls.log` (`arms correct: 10 arms wrong: 0`) | The record+check pair exists and works: three SHAs + `Manifest.toml` sha256 ×3 + the Julia version, reconstructed into a depot built from empty, 60 vs 60 entries, dependency set identical, with 10 two-sided control arms. But the record **pins SDPX `c4b109a`**, and the release triple I03 proposes moves SDPX. A record that names a revision the release does not ship does not satisfy the row. Retirement is one command pair: re-pin, then re-check (`pin_revisions_env.sh <new SHA> e3805c9 f087a72 --clean-check`, then `check_reconstruction.py`). Not a limitation — a two-command step |
| **R5** | **satisfied** | `python3 SDPX.jl/scripts/rebuild/validate_reports.py .` → before I03's report: **`reports=26 errors=0 warnings=147`**; after: `reports=27 errors=0` (re-run recorded in `rebuild-reports/I03/report.json`). Acceptance re-count over the 25 packet reports, reproduced by me: **122 verified / 29 partially_verified / 4 not_verified = 155** (A01b contributes 9 verified as an ad-hoc report). Capability table: `rebuild-reports/Q02/capability_table.tsv` — 56 capabilities, 53 with passing evidence, 26 `verified` / 25 `partially_verified` / 5 `not_verified`, **`certified 0`** | The two entries §4 held open were resolved in the honest direction and are re-verified here: the A01 composite is split into `pass` (20) + `unsupported` (3, explicitly not counted as passes), and S02's interrupted run is `pass` with its non-zero exit stated plus a limitation recording that the clean run supersedes it. `not_run` remains distinguishable from `pass` throughout |
| **R6** | **satisfied (dense path; R8 carries the residual)** | `cd MultiFloatLinearAlgebra.jl && JULIA_DEPOT_PATH=… julia --project=$REBUILD_ENV -t1 test/rebuild/M01.jl` → `rebuild-reports/I03/logs/pre_delete/M01_lease_token.log` (204/204 pass); `sed -n '350,372p' SDPX.jl/src/la/factor_lease.jl`; V01 `independent_audit.md` §M01 IP-2 and `V01/report.json` F6 | Re-measured by me at this revision: `cache_leases = (first = 0x…01, second = 0x…02)` — the token **differs** across two same-size dense refactors, with the generation advancing `0→1→2`. The `refactor_numeric!` admission refusal now revokes before returning (`factor_lease.jl:364` `_revoke!(h.lease, EvRefactorPreflightRejected, …)`), so all four exits revoke rather than three of four. **Scope, stated so the row is not read wider than it is:** this holds on the four dense `factorize!` methods; the sparse path does not advance the generation at all, which is **R8** |
| **R7** | **satisfied (bounded, as its own status cell states)** | `rebuild-reports/PARENT_VERIFICATION/driver_matrix_frozen/M02.log` (leg exit 0, `failcols=0`) + `rebuild-reports/M02/report.json` numeric tests `S05-F1 multi-RHS, dense caches (ldlt, cholesky, lu)` = pass, `S05-F1 multi-RHS, sparse QDLDL cache` = pass, `multi-RHS capability claim` = pass | `capabilities(MF)` no longer rests on a timing claim: it rests on method identity, bitwise agreement with the per-column loop, and the residual — explicitly *not* on timing, because M02's timing instrument **failed its own control** (a known per-column loop measured a sub-1 ratio). MFLA's `S05-F1` "is the dense matrix path genuinely batched" sub-question is **bounded, not settled**, and that bound is the honest form. I did not re-run M02 post-deletion: its subject is MFLA's own crate, and MFLA is untouched at `e3805c9` |
| **R8** | **not_satisfied — reproduced, and NOT accepted** | `JULIA_DEPOT_PATH=… julia --project=$REBUILD_ENV -t1 /tmp/v01_ip2_sparse.jl` → `rebuild-reports/I03/logs/pre_delete/R8_ip2_sparse_generation.log` (exit 0) | Re-measured by me at this revision, in one process with a positive control: `DIRECT_SPARSE generations=(0,0,0) delta_first=0 delta_second=0` while `DENSE_LDLT generations=(0,1,2) delta_first=1 delta_second=1`, and **SDPX's own seam** (`SDPX.SparseQDLDLProviderCache` + `SDPX._qdldl_provider_factorize!`) also `delta=0` with `issuccess=true`. So on the sparse path a lease taken before a refactor still validates after it — the `M01-F4`/`P02-F1` defect, unchanged, reachable through the real package extension (`ext/MultiFloatQDLDLExt.jl:160`) that loads for any user, not only under a test harness (V01 F6). **Decision: I03 declines to accept this as a release risk.** It is a default-reachable correctness gap, the fix is prepared and behaviourally verified (proposal `I02-P2`: instrument `MFSparseLDLCache.factorize!` so the generation advances, plus the evidence that its commit points exist), and accepting a correctness gap that has a one-line prepared fix would be exactly the "hazard with a false sense of coverage" ADR-002 §4 names. Retirement: apply `I02-P2` and re-run this same command until the sparse deltas read `1,1` with the dense control still `1,1` |
| **R9** | **not_satisfied — demotion declined** | `cd BigFloatLinearAlgebra.jl && P03_OUT=<dir> JULIA_DEPOT_PATH=… julia --project=$REBUILD_ENV -t1 test/rebuild/P03.jl` → `rebuild-reports/I03/logs/pre_delete/P03_release_gate.log` + `…/p03_out/P03_run_main.txt:112-117` (exit 0); primary log preserved unmodified as `rebuild-reports/I03/logs/P03_run_main.PRE_I03.txt` | Re-run by me at `f087a72`: the driver is **`365/365 Pass`** at the same revision whose own computed gate reads `release_gate_verdict = FAIL`, `required_rows = 10`, `verified_required_rows = 9`, `failing_rows = ["concurrent_sessions_different_precision_in_process_parallel"]`, with four unsupported rows of which exactly one is `required=true`. **`unsupported` means the capability is not offered — it does not mean broken**, and "the gate is red" must never be reported as "the tests fail". **Decision: I03 does not demote the row.** Editing `required=true` to green at the freeze point is the same failure as counting a SKIP as a PASS (the card's hard prohibition), and the correct fixed form is an ADR-level statement that in-process, concurrent, different-precision sessions are architecturally unavailable because BigFloat precision is process-global and `_ambient_guard` throws `PrecisionMismatch` — a contract change with its own evidence burden, which belongs to the ADR owner, not to the last task. Retirement: implement process-isolated precision sessions, or have the ADR owner change P03's contract row *with the measured reason* and re-run the gate |

### The rule that overrides everything else

R1 and R2 are unconditional blockers and both are unmet ⇒ **BLOCKED**. R8 and R9 were
eligible for explicit acceptance, and I03 considered and **declined** both, with the
reproduction and the retirement condition recorded above. That is a decision, not an
omission: even if R1 and R2 were retired tomorrow, R8 and R9 would still hold the release
until they are fixed or accepted on a stated basis. R4 is also unmet today and is a
two-command step, not a limitation. R3 is `not_measured` by construction of the
commit protocol. R5, R6 and R7 are satisfied within their stated scopes.

**Answer to the card's third acceptance criterion — "decide whether the release is
permitted": NO. Release is not permitted.** The card's completion is *not* approval of the
related capabilities for production; see §9.

## 3. Carrier dispositions — one action per row

From `I03_RELEASE_DECISION.md` §2, §2.5, §2.6 and §3. "Action" is what I03 *did*.

| carrier | repo / path | action | evidence and reason |
| --- | --- | --- | --- |
| `ext/rebuild/mfla_live.jl`, `bfla_live.jl` | SDPX | **kept, load-on-demand only** | They name their provider at parse time. They are reached only through `load_mfla()` / `load_bfla()` (`ext/rebuild/mfla_adapter.jl:465`, `bfla_live.jl:34`) from `test/rebuild/S05.jl`; nothing under `src/` includes `ext/rebuild` (`grep -rn 'ext/rebuild' src/` = **0**). Wiring them would put MFLA/BFLA in SDPX's dependency graph, which ADR-002 §1 and `baseline.md` §2 forbid |
| `ext/rebuild` in `Project.toml [extensions]` | SDPX | **verified absent — nothing to delete** | `grep -n 'rebuild' SDPX.jl/Project.toml` = **0 hits**. The `[extensions]` block lists exactly `SDPXAppleAccelerateExt`, `SDPXBigFloatLinearAlgebraExt`, `SDPXGenericLinearAlgebraExt`, `SDPXJLD2Ext`, `SDPXMultiFloatLinearAlgebraExt`, `SDPXMultiFloatsExt` |
| the two mock adapters | SDPX `ext/rebuild/{mfla,bfla}_adapter.jl` | **kept; reachability re-measured** | `MockMFLA`/`MockBFLA` are referenced only by their own files plus `test/rebuild/S05.jl` and `test/rebuild/S07.jl`; zero references under `src/` or under `ext/` outside `ext/rebuild/`. The file header's claim holds at this revision, re-measured after I02 — a comment is not a check, so this is the check |
| `src/sparse_la.jl`'s `GenericSparseCholeskyFactor` family | SDPX | **KEPT — and the "orphan, delete-or-wire" framing in §2 is unsafe; see finding F-I03-1** | The *type name* does have zero external references (12 hits, all in `src/sparse_la.jl`), which is what §2 and the manifest measured. But reachability of a type is not reachability of a capability: the family is the only implementation behind the **live predicate** `supports_sparse_generic` / `supports_sparse_execution`, which `src/pipeline/plan.jl:291` and `_use_sparse_schur_sdp` (`sparse_la.jl:35-40`) consume to route extended-precision sparse-classified problems. The entry points that instantiate it (`sparse_factor`, `sparse_factor_solve`, `freeze_schur_pattern`, `GenericSparseCholeskyBackend(…)`) have **zero callers repo-wide**, and `src/pipeline/plan.jl:508-520` records the descriptor symbol `:generic_sparse_cholesky` while explicitly not instantiating it. Deleting the family while the predicate stays true would convert a dormant implementation into a latent failure. **Action: keep; deletion is handed forward as an inert proposal that must change predicate + routing + family together, with a before/after.** This is the one place where I03's mandate is to decide *and* to record that the criteria row under-evidenced its own instruction |
| `src/caches.jl` | **BFLA** (not SDPX) | **KEPT — the withdrawn delete instruction stays withdrawn** | 339 lines, `struct BFLARRQRCache` at `:38`, one definition; 15 files under BFLA reference it. Six cache files define six disjoint type sets; the shared names are methods of the same generic functions, one per cache type — dispatch, not duplication. I03 deletes nothing here (also: the row's path is ambiguous between repos — the SDPX tree has no `src/caches.jl`; `git rev-parse HEAD:src/caches.jl` fails in SDPX) |
| `_product_hsd_soc_condition_budget` | SDPX `src/hsd/product_cone_hsd.jl` | **DELETED** | `grep -rn '_product_hsd_soc_condition_budget'` over the whole repo = **1 hit, its own definition**; already recorded dead at `docs/evidence/OPEN_SOC_ROUNDTRIP_ON4.md:12` ("DEAD: zero references repo-wide"). Post-deletion grep = 0. The five driver legs that exercise this file (S03, S04, S06, S07, A01_default) pass at the new tree |
| the `nzrange` defect | SDPX `src/hsd/product_cone_solve.jl` | **FIXED — verified, no I03 change needed** | The three sparse-API sites now carry guards: `if A isa SparseMatrixCSC` at `:208`, `:265`, `:298`, each with a dense fallback, plus the dispatch site at `:479`. I02's before/after is `rebuild-reports/I02/logs/nzrange_before_after.log` (command `I02_NZRANGE_ARM={before,after} julia --project=/tmp/i02nzr-env -t1 /tmp/i02_nzrange_target.jl`, exit 0). Recorded as **fixed**, not accepted-as-broken |
| `src/certification/{status,direction}.jl` | SDPX | **kept — and the submodule is on the load path** | `src/certification/original.jl:1408-1409` includes both **inside `module SDPXCertification`**, and `src/SDPX.jl:195` includes `certification/original.jl`. Deleting them would break the submodule. The module is loaded but **inert on the public path** (0 external `SDPXCertification.` references) — which is R1's shape, not a reason to delete files |
| **A01b-F1** — square `BFLARRQRCache` keeps `:success` across a preflight rejection | BFLA | **accepted as a known limitation of this candidate, with the reproduction cited; not fixed** | Live and reproduced: `control_square_cache_keeps_stale_success=true` in both modes (B02, preserved by B04), raw record `rebuild-reports/B04/B04_driver_SCRATCH_split.log`, quoted in `rebuild-reports/B04/report.json` acceptance note. The prescribed fix location does not exist yet — `_require_cache_matrix` still lives at `src/caches.jl:140`; it moves to `src/caches/common.jl:178` only when B04's split is wired, and that split is deliberately not wired. **Retirement:** wire B04's split, then move `_require_cache_matrix` and revoke on preflight rejection (the dense caches already revoke there — this is the one asymmetric path). I did not re-run B04's driver: it writes its own evidence bundle and its subject is BFLA, which I03 leaves untouched |
| **`refactor_numeric!` lease hazard** (admission refusal left a bound lease) | SDPX `src/la/factor_lease.jl` | **FIXED — verified** | `factor_lease.jl:350-372`: the admission-refusal branch now calls `_revoke!(h.lease, EvRefactorPreflightRejected, …)` before returning, so "on ANY failure — thrown, returned non-success, or returned with a digest/shape mismatch — `_revoke!` runs BEFORE `raw_status` is read" holds for **all four** exits. I02's rollback arm captures the pre-fix behaviour in a scratch tree. R6 depends on this and is satisfied on the dense path |
| O(n⁴) SOC roundtrip (`src/hsd/product_cone_hsd.jl:887-915` at `c8fb65a`) | SDPX | **documented performance limitation; deliberately not "fixed"** | 92.5× kernel redundancy in Float64 with bit-identical values, and Q01's scaling data (281×/313× for n 32→128) confirms the class — but the harness is in **no repository** (V01 F11), so the ratio is a claim, not a reproducible measurement, and **no end-to-end measurement exists**. V01 verified the half that needs no timing (the hoisted quantities are invariant in the loop variable; accumulation order unchanged ⇒ the hoist is structurally exact). Releasing it as a documented limitation is the honest disposition; a fix made without an end-to-end instrument would not be |
| `QDLDL.solve(factor, ::Matrix)` out-of-bounds write | third party (via MFLA's extension) | **known third-party limitation — stated, not fixable in this tree** | 8 of 8 isolated probes raised a catchable `ReadOnlyMemoryError`; 1 of 1 real driver runs **segfaulted** inside `ipermute!` (`QDLDL.jl:619`) under concurrent load, killing the process with no `Test Summary`; a 0..20M-iteration allocation-churn sweep did **not** reproduce it, so the mechanism is **not understood**. A01b-F2 records only the common case. Disposition: treat `QDLDL.solve(Q, ::Matrix)` as an unsupported call that may terminate the process; what SDPX can do is not call it and not probe it in-process, and P02/P03's dispatch blocks now say so |
| S07's dependency on `ext/rebuild` (outside its write allowlist) | SDPX | **stated as a dependency, not discovered later** | `test/rebuild/S07.jl` references `MockMFLA`/`MockBFLA`, i.e. it loads `ext/rebuild/mfla_adapter.jl` from S05. S07's evidence is therefore **not self-contained**: an edit to S05's adapters can change S07's result without touching an S07 file. Any verification of S07 must name the revision of `ext/rebuild/` it ran against |

### Step 2 of the card — temporary dual paths, debug globals, stale phase comments

| item | action | evidence |
| --- | --- | --- |
| `SDPX_DEBUG_EQUALITY_RECOVERY` (1 site) | **DELETED** | The guard wraps only a `println(stderr, …)`; the decision `valid \|\| return false` is outside it. Zero setters in any `.jl` under `ext/ test/ benchmark/ scripts/ validation/`; **zero** references in `docs/` or `rebuild-reports/`, so it is not a provenance instrument |
| `SDPX_DEBUG_SYMMETRIC_CORE` (4 sites) | **DELETED** | Same shape at every site: `state.diagnostic`, the `false`, and `return HSDStepDirectionFailed` all sit **outside** the guards; only `showerror`/`println` were gated. Zero setters repo-wide, zero documents. No correctness check removed |
| `SDPX_DEBUG_DIRECTION`, `SDPX_DEBUG_DIRECTION_ITER`, `SDPX_DEBUG_LINE_SEARCH`, `SDPX_DEBUG_ITER` | **kept, deliberately** | These are the reproduction instruments named in `docs/evidence/P3_01_BETA_EXPERIMENT.md`, `docs/evidence/P0_03_PLATFORM_DIRECTION_BREAKDOWN.md` and `docs/performance/EXECUTION_STATUS.md`. Deleting them would destroy the provenance of recorded experiments, which the card requires be retained. Their sites are all debug-only (verified site by site) |
| `SDPX_CORE_ROUTE_PLANNER` (legacy vs model core-route planner) | **kept, default unchanged** | `test/core_route_planner.jl:193-212` asserts the `"legacy"` default explicitly ("planner is shadow-mode by default (no silent default change)"). Deleting the switch would delete a test and silently promote an uncalibrated cost model — forbidden twice over (no default promotion; no deleting a test to reach green) |
| `iteration_knobs=(sigma,beta,gamma)` | **kept** | The archetypal temporary dual path *and* the archetypal legitimate one: default `nothing` reproduces the historical literals bit-identically, it is exercised by `test/runtests.jl:406-444` and `test/factor_pair_backend_selector.jl:90`, and it is the reproducible-experiment control P3_01 used |
| `SDPX_HKM_VEC4` (vec4 vs scalar HKM kernel) | **kept, recorded** | A real route switch whose default (`"1"`) is the new path, referenced by no test/benchmark/doc. It is the only A/B control for the two kernels; deleting it would be a behaviour change needing its own before/after. Recorded as a named remaining dual path |
| `linear_algebra_backend ∈ (…, :legacy)`, `prepare_symmetric_core`, `allow_expanded_bordered_fallback` | **kept, recorded** | Public/documented options and route selectors with documented defaults, not temporary switches |
| `iteration_predictor` field (`product_cone_hsd.jl:280`, assigned `:504`) | **recorded as a finding, not deleted (F-I03-4)** | It is **write-only** in the HSD state (`grep -rn 'iteration_predictor' src/` has no read site); its only real effect is through `factor_pair_admission.jl`. Deleting a struct field of the solver state at the freeze point is not a retirement, it is a refactor |
| stale "not yet wired / I01 will include" comments in `src/hsd/product_cone_solve.jl`, `src/hsd/product_cone_hsd.jl:577`, `src/hsd/hsd.jl:466`, `test/rebuild/S01.jl`, `S03.jl`, `S06.jl`, `S07.jl` | **corrected (comment-only, zero behaviour)** | Each was contradicted by code: `src/SDPX.jl:185-198` and `:203-207` are the include list; `native_hsd_public.jl:2204` calls `product_hsd_solve!`; `product_cone_hsd.jl:3833` dispatches on the prepared symmetric core; `_hsd_column_reduction` has no callers. The corrections name the line numbers that contradict the old text, so the next reader can re-check |
| `test/rebuild/S04.jl:48-50` (same class, lower confidence) | **not corrected — recorded** | The sentence is about intent ("written to be included by `src/SDPX.jl` … *before* integration") and the driver has no `isdefined` probe, so the contradiction is weaker than the seven above. Recorded rather than edited, and handed on as a proposal |
| phase-labelled comments (`GPTPro Phase 8 planning layer`, `Phase 4.1`, `P0-03`, `P3-00`, `TASK-P0-SPARSE-AUGMENTED`) | **kept** | They are phase-labelled but *accurate* — e.g. the formulation-planner scaffolding comment carries the load-bearing claim that it "never changes the default route", which the code bears out. A phase label is provenance, and removing labels is not a correctness improvement |

## 4. What I03 changed — deletions, with test and reachability evidence, and rollback

Base for every entry: `c8fb65a` (the blob SHAs are the pre-change objects, so any single
file is restorable with `git -C SDPX.jl checkout c8fb65a -- <path>`).

| # | path | change | reachability evidence | test evidence | pre-change blob |
| --- | --- | --- | --- | --- | --- |
| B1a | `SDPX.jl/src/hsd/product_cone_hsd.jl` | delete `_product_hsd_soc_condition_budget` (23 lines) + 4 `SDPX_DEBUG_SYMMETRIC_CORE` print blocks; correct the false docstring at `:577` | repo-wide grep = 1 hit (own definition) before, **0** after; the debug knobs have **0** setters repo-wide and **0** doc references | `final_tree/`: load exit 0; S01 189 pass, S03 193, S04 313, S06 214, S07 289, A01_default 1673, all exit 0 | `a08ada68e809ca19b1304448f1630ad2dcc2657b` |
| B1b | `SDPX.jl/src/hsd/equality_reduction.jl` | delete the `SDPX_DEBUG_EQUALITY_RECOVERY` print block | as above: 0 setters, 0 docs; the guarded body is a single `println` and the decision sits outside it | same legs; the infeasibility route (case C of the R1 probe) exercises this function and still returns a valid ray certificate | `19f1d987968e560abfd1a7cb304a0e2bfd125144` |
| B2a | `SDPX.jl/src/hsd/product_cone_solve.jl` | correct the false header claim ("deliberately not wired to the public/MOI route") | `native_hsd_public.jl:2204` calls `product_hsd_solve!`; that file's own header names the chain | comment-only; load + all legs | `533547075c813e7f4369dfa1ed4d5bb3e065c078` |
| B2b | `SDPX.jl/src/hsd/hsd.jl` | correct "retained for callers" (zero callers measured) | `grep -rn '_hsd_column_reduction'` = 2 definitions + 1 docstring cross-reference | comment-only | `388dd47b818a78c904321b47df7bdd8a61ed73a9` |
| B2c | `SDPX.jl/test/rebuild/S01.jl` | correct two false "not yet part of the package entry point … I01 will include" notes | `src/SDPX.jl:185-186` includes `core/compiled_problem.jl`, `core/transforms.jl` | S01 driver re-run: 9 testsets, 189 pass, exit 0 | `7c3c2cb96db3a2fa39889f0920a403b324f6e8fa` |
| B2d | `SDPX.jl/test/rebuild/S03.jl` | correct the false "NOT yet wired into `src/SDPX.jl`" loading note | `src/SDPX.jl:187-190` includes the four `kkt/` files | S03 driver re-run: 2 testsets, 193 pass (1 Broken row = baseline), exit 0 | `978a3447bc4159cb4d30cde825b75e4ab48818b9` |
| B2e | `SDPX.jl/test/rebuild/S06.jl` | correct the false "I01 will add to `src/SDPX.jl`; until then" note while preserving the standalone fallback | `src/SDPX.jl:196-198`; `plan_setup` at `src/planning/setup.jl:598` | S06 driver re-run: 1 testset, 214 pass, exit 0 | `73afb021112547b78b702e6461de9d716706d4ba` |
| B2f | `SDPX.jl/test/rebuild/S07.jl` | correct the false "NOT in the package include graph (wiring is I02's job)" note | `src/SDPX.jl:203-207` includes `session/{update,replay,cancellation}.jl` | S07 driver re-run: 3 testsets, 289 pass, exit 0 | `1248fbbfd9cb22abb9f0f20a5cc1b1c91d3dfbc7` |
| B3 | `SDPX.jl/docs/rebuild/final_verdict.md` | this document (new file) | — | — | — |

**Diffstat of B1+B2: 8 files, +7 / −53** (`rebuild-reports/I03/logs/deletions_paths.txt`,
full diff `rebuild-reports/I03/logs/deletions.diff`). **No behaviour-bearing code was
added, and no correctness check was removed** — every deleted line was either a dead
function with zero references or a `println`/`showerror` guarded by an env var that no
file in any of the three repositories sets.

**Intended commit boundaries** (explicit paths only; never `git add -A` — `rebuild-reports/`
is unversioned and was shared with two other workers):

    # B1 behaviour-neutral retirement
    git -C SDPX.jl add src/hsd/product_cone_hsd.jl src/hsd/equality_reduction.jl
    # B2 stale-comment corrections
    git -C SDPX.jl add src/hsd/product_cone_solve.jl src/hsd/hsd.jl \
        test/rebuild/S01.jl test/rebuild/S03.jl test/rebuild/S06.jl test/rebuild/S07.jl
    # B3 the verdict
    git -C SDPX.jl add docs/rebuild/final_verdict.md

**MFLA and BFLA are not modified by I03** (`e3805c9` / `f087a72`, both clean). That is a
decision with a reason: the two stale comments in those trees (`MFLA
src/contracts/workspace.jl:6`, `BFLA test/rebuild/B03.jl:18-22`) are comment-only, and
applying them at the freeze point would move two more repositories off the revisions at
which their suites and the frozen matrix were measured, for zero behavioural gain. They
are handed on as **inert patch proposals** in `rebuild-reports/I03/report.json`
(`integration_patch_proposals`), never applied.

**`rollback_commit`**: `c8fb65a0d5e15b6d03d34c4368cba7ad5f697094` for every deletion
above (the revision that still contains all of them), plus the per-file blob SHAs.
`rebuild-reports/` is in **no git repository** — this report, its logs and every finding
here have no history to roll back to; that is stated rather than implied.

## 5. What would unblock the release — in commands

Each line is a command, not a wish. `REBUILD_ENV` =
`/Users/xuyongjun/Desktop/project/SDPX/rebuild-env`, depot
`/Users/xuyongjun/Desktop/project/SDPX/rebuild-env-depot:$HOME/.julia`, always `-t1`.

**Unconditional blockers**

1. **R1** — build the missing instrument, then drive the negative direction.
   `julia --project=$REBUILD_ENV -t1 test/rebuild/S04.jl` must gain a case that makes
   `certificate_summary.valid == false` on a *terminal* status, and
   `sed -n '2576,2586p' SDPX.jl/src/hsd/native_hsd_public.jl` must be covered by a log
   showing `status = NumericalFailure`, `termination_reason =
   :original_coordinate_certificate_failed`. Then apply move 6 so the four
   `ResultCertificate{T}` sites (`native_hsd_public.jl:2403`, `:2462`, `:2487`,
   `public/optimize.jl:341`) are served by `SDPXCertification`, with a before/after in
   both directions.
2. **R2** — replace the static claim with a runtime observation.
   Drive `replay_public_signs` from a real MOI solve (MOI resolves here — the pinned
   Newton-gate child runs), or count sign applications at the public boundary during one,
   and show the count is zero **and** the result is sign-correct; then either compute
   `public_sign_patches` from that observation or delete the literal at
   `src/core/compiled_problem.jl:475` and its self-assertion at `test/rebuild/S01.jl:474`
   from the evidence chain. The command that can never unblock this row is a
   before/after of a patch deletion: there is no patch.

**Conditionally blocking (accept or fix; I03 declined to accept)**

3. **R8** — apply proposal `I02-P2` (instrument `MFSparseLDLCache.factorize!` at
   `ext/MultiFloatQDLDLExt.jl:160` so `record_factor_summary!` runs at its commit points),
   then re-run `julia --project=$REBUILD_ENV -t1 /tmp/v01_ip2_sparse.jl` and require
   `DIRECT_SPARSE delta_first=1 delta_second=1` **and** `SDPX_SEAM` deltas `1,1`, with the
   dense control still `1,1`.
4. **R9** — either implement process-isolated concurrent precision sessions and re-run
   `cd BigFloatLinearAlgebra.jl && P03_OUT=<dir> julia --project=$REBUILD_ENV -t1 test/rebuild/P03.jl`
   until `release_gate_verdict = PASS` with `verified_required_rows = 10`; or have the ADR
   owner change the `concurrent_sessions_different_precision_in_process_parallel` row's
   `required` value **with the measured reason** (BigFloat precision is process-global;
   `_ambient_guard` throws `PrecisionMismatch`) and re-run the gate. Demoting the row to
   green without that record is prohibited.

**Housekeeping that is also a release criterion**

5. **R4** — after the B1+B2+B3 commit:
   `bash SDPX.jl/scripts/rebuild/pin_revisions_env.sh <new SDPX SHA> e3805c9607295f0e173567b060d4035e826310ca f087a72f2001088ea520b67588cf18c0d3fce2e8 --clean-check`
   then
   `python3 SDPX.jl/scripts/rebuild/check_reconstruction.py --record SDPX.jl/docs/rebuild/RELEASE_REVISIONS.txt --target /tmp/i03recon --depot /tmp/i03recon-depot`
   and copy the new record over `docs/rebuild/RELEASE_REVISIONS.txt`.
6. **R3** — the three `Pkg.test()` runs and the 28-leg matrix on the clean post-commit
   trees (`bash SDPX.jl/scripts/rebuild/run_driver_matrix.sh <outdir>`; B03's leg needs
   `B03_WIRED_INJECT_TEST=1`, and read **per-leg** lines, never the aggregate, for any run
   made with a pre-`d6ca577` script).

## 6. Capability split: production / experimental / unimplemented

Source of truth: `rebuild-reports/Q02/capability_table.tsv` (56 rows) and
`SDPX.jl/docs/rebuild/support_matrix.md`. States are Q02's, not re-derived here; the
grouping is mine and each group names its membership rule.

**PRODUCTION — offered, default-reachable, with passing numeric or structural evidence
(26 rows `verified`).** `SDPX-PRECISION-UPGRADE`, `SDPX-STEP-STRATEGY`,
`SDPX-CERT-REJECTION`, `SDPX-NOT-RUN-NEVER-ZERO`, `SDPX-PRECOMPILE-GATE`,
`SDPX-MULTI-RHS-CAPABILITY`, `SDPX-SETUP-PLANNING`, `SDPX-REPLAY-CANCELLATION`,
`SDPX-LEASE-STRUCTURE`, `SDPX-ADAPTER-LEASE-REQUIRED`, `SDPX-GATE-CALIBRATION`,
`MFLA-PIVOT-GRAMMAR`, `MFLA-KERNEL-ORACLE`, `MFLA-SPARSE-QDLDL`, `BFLA-RECTANGULAR-QR`,
`PROVIDER-CONTRACT-JOINT`, `PROVIDER-SPARSE-DENSITY-POLICY`, `QA-MUTATION-DETECTION`,
`QA-ORACLE-PROVENANCE`, `QA-MULTI-PRECISION-ORACLE`, `QA-PROVIDER-LEGS`,
`INFRA-REPRODUCIBLE-BENCHMARK`, `INFRA-SINGLE-DEFAULT-PIPELINE`, `INFRA-INCLUDE-GRAPH`,
`INFRA-TOLERANCE-LEDGER`, `INFRA-RESEARCH-GATES`.
**Nothing in this group is `certified`** (`certified 0` across the table), and four
default-reachable capabilities in the wider table carry open defects that a release would
have to state: `SDPX-CERT-ORIGINAL-COORDS` (R1), `SDPX-PUBLIC-NO-SIGN-PATCH` (R2),
`SDPX-FACTOR-LEASE` on the sparse path (R8), `PROVIDER-CONCURRENT-PRECISION` (R9),
`BFLA-RRQR-STALE-SUCCESS` (A01b-F1).

**EXPERIMENTAL / OPT-IN — in the tree, tested, deliberately not on the default route
(comment-verified at this revision).** `src/nullspace.jl:2` and `src/chordal.jl:2`
("EXPERIMENTAL / OPT-IN — not reachable from `solve`"), `src/cone_algebra.jl:5`,
`src/factor_cache/session_symbolic_lease.jl:1`, `src/factor_cache/routes/qdldl_sparse.jl:9`,
`src/kkt/symmetric_core.jl`'s simultaneous-live bounds (the code itself refuses
promotion), `src/midend/formulation_planner.jl:295` (calibrated planner scaffolding that
"never changes the default route"), `SDPX_CORE_ROUTE_PLANNER="model"` (shadow only),
`SDPX_HKM_VEC4` (default on, A/B control), `iteration_knobs` sigma/beta/gamma,
`linear_algebra_backend=:legacy`, `equality_solver=:sparse_qr`
(`equality_reduction.jl:1066-1110`: "prepared, not executable").

**UNIMPLEMENTED / NOT OFFERED — explicit, and never counted as a pass.** Thread tiers 16
and 64 on this host (`Sys.CPU_THREADS == 4`; Q01, M02, B03, B04 all record `unsupported`
with that measured host fact); native MPFR-limb / malloc allocator accounting (no hook in
Julia: `B03`'s `mfr_native_allocation_attribution` = `not_run`, `P03`'s
`native_allocator_accounting` = `not_run`); provider in-place numeric refactorization
(`isdefined(BigFloatLinearAlgebra, :refactorize!) == false` at this revision);
in-process concurrent sessions at **different** precisions (R9); `QDLRL` (not resolvable
in either project; infrastructure, not numeric); the O(n⁴) SOC-roundtrip benchmark
harness (not in any repository); `QDLDL.solve(Q, ::Matrix)` (third-party, unsupported and
process-threatening).

**Interpretation rule this document follows:** `unsupported` means *not offered*. It does
not mean broken, and it is not a `0`. P03's driver is `365/365 Pass` at the very revision
whose gate reads `FAIL`.

## 7. Performance conclusions, with their raw records

No timing claim in this section is made by I03, and none is invented; each figure names
the record that contains it, and each *absent* figure is named as absent. The host was
shared for much of the packet (another worker's Julia process was live; `load1` 1.9-2.8 on
10 OS cores), so wall-clock comparisons across tasks are not available. All thread counts
below are `Sys.CPU_THREADS == 4` unless stated.

| conclusion | figure | raw record | status |
| --- | --- | --- | --- |
| Thread tiers 16/64 are unavailable on this host | partitioning measured and correct; execution at those tiers not available | `rebuild-reports/Q01/report.json`, `M02`, `B03`, `B04` numeric tests (`unsupported`, reason recorded) | **unsupported**, not a failure |
| Dense/sparse scaling class | 281×/313× from n 32→128 | Q01's scaling data (`rebuild-reports/Q01/`), reproduced-class only via V01's fingerprint work | measured at Q01's revision; not re-measured here |
| SOC roundtrip kernel redundancy | 92.5× redundancy, bit-identical values | `SDPX.jl/docs/evidence/OPEN_SOC_ROUNDTRIP_ON4.md` | **claim without an archived instrument** (V01 F11) — must be carried as a limitation, not a measurement |
| **No end-to-end wall-clock measurement exists** | `not_run` | `rebuild-reports/B04/report.json` (`wall_clock_timing = not_run`: contended host, `Sys.CPU_THREADS == 4`, two other workers live); `M03`'s E2E baseline `not_run` | **not_run**, stated as such |
| Multi-RHS claim | not a timing claim: method identity + bitwise agreement with the per-column loop + residual | `rebuild-reports/PARENT_VERIFICATION/driver_matrix_frozen/M02.log`; M02's timing instrument **failed its own control** | bounded, not settled |
| Factor-summary cost | `summary_read_bytes` = 0 for `kind/status/state/inertia/grammar/pivots/block_counts`; `summary_record_bytes` = 32; `same_size_factorize_bytes` = 144; `factor_matrix_bytes` = 4096 | `rebuild-reports/I03/logs/pre_delete/M01_lease_token.log` (my re-run) + `rebuild-reports/M01/` | every sample is in the log, printed field by field |
| RSS / allocator | `child_rss0 = 211238912`, `child_rss1 = 505085952`, `delta = 293847040`, `bytes_per_unit = 1.09466552734375`; native-limb accounting `not_run` | `rebuild-reports/I03/logs/pre_delete/p03_out/P03_run_main.txt` (my re-run) | RSS is the only native-level signal available |
| MFLA kernel performance | per-limb timings for x2/x3/x4 on 128³, variant grid, register-pressure proxy | `rebuild-reports/M02/report.json` numeric tests | measured on a shared host; no cross-machine claim |
| Compilation | precompile gate passes; `LOAD_OK public_export_count=69`; `using SDPX` precompiled in ~8 s | `rebuild-reports/I03/logs/final_tree/LOAD.log` | measured here |
| The scorecard is **incomplete** for a production release | setup/iteration/recovery/certification phases are instrumented, **failures and RSS partly, compilation partly, no end-to-end wall clock, no native allocator accounting** | §11 Q7 below | **partial** |

## 8. Answers to §11 of `ENGINEERING_REBUILD.md`

Each answer is 是 / 否 / 部分 with the evidence path. Unmet items are answered as unmet.

**1. 一个问题、一个主循环、一个当前KKT策略，是否已成为可验证事实？—— 部分（是，在默认路径上）。**
One loop body exists: `src/solver/loop.jl`, reached from `native_hsd_public.jl:2204`
through `product_hsd_solve!`, whose old body was deleted at the S02 cutover
(`src/hsd/product_cone_solve.jl:752+`; `S02/report.json`, I02's re-measurement).
Evidence: capability rows `INFRA-SINGLE-DEFAULT-PIPELINE` = `verified`,
`SDPX-HSD-LOOP` = `partially_verified` (11/12), `SDPX-NO-SECOND-HSD-LOOP` =
`partially_verified` (0/1, S02's own log row is `not_run`); I02 acceptance is
`partially_verified`. **What keeps it from a clean 是:** the second row's passing
acceptance count is 0 of 1 and S02's log is `not_run` in the table — the "no second loop"
claim rests on structure plus I02's re-check (`grep -rn plan_setup src/hsd/ src/public/`
was still empty), not on a green log row.

**2. public/MOI/compatibility是否不再实现数值证书、factor策略和算术kernel？—— 否。**
No numeric *certificate* is produced by the public layer — but that is because the public
layer does not use the certification module at all: `grep -rn 'SDPXCertification\.'
SDPX.jl/src SDPX.jl/test` outside its own file = **0**, while four `ResultCertificate{T}`
construction sites remain (`native_hsd_public.jl:2403`, `:2462`, `:2487`,
`public/optimize.jl:341`). That is **R1**, unmet. The sign-patch half is the same shape:
no patch exists (8 patterns × 0), but the evidence chain for "no sign patch" is a
hardcoded literal — **R2**, unmet. Factor strategy: the public layer consults the
planner's descriptor (`src/pipeline/plan.jl`), not a factor implementation; the
`:generic_sparse_cholesky` descriptor is recorded while explicitly not instantiated
(`plan.jl:508-520`) — **F-I03-1**. Kernel selection is likewise descriptor-based.

**3. provider是否只拥有数值事实和内存，不决定HSD/原始可行域/终止？—— 是（有测试），边界由ADR-002固定。**
ADR-002 §1/§3 fix the provider's ownership; `PROVIDER-CONTRACT-JOINT` and
`PROVIDER-SPARSE-DENSITY-POLICY` are `verified`; P01's sparse contract legs
(`test/provider_contracts/sparse_contract.jl`, three provider legs) and P02's adapter
tests pass. Two qualifications: `PROVIDER-QDLDL-RAW-MATRIX-ENTRY` is
`partially_verified` because the *third-party* raw matrix entry point is unsound
(§3 above), and `PROVIDER-THIRD-PARTY-GATE` is `partially_verified` because the private
field gate is `table_only` (P01-PATCH-1 not wired to enforce).

**4. factor失效、precision/rounding、pattern/value更新、并发session的所有权是否有测试？—— 部分。**
- factor invalidation/lease: **是 on the dense path** — M01 204/204 with
  `cache_leases=(1,2)` (`rebuild-reports/I03/logs/pre_delete/M01_lease_token.log`), and
  `refactor_numeric!`'s admission refusal now revokes (`factor_lease.jl:350-372`);
  **否 on the sparse path** — generation delta 0 (R8).
- precision/rounding: **部分** — B01's explicit rounding code and BFLA's
  `mpfr_context` tests pass; `PROVIDER-CONCURRENT-PRECISION` is `not_verified` (R9) and
  `BFLA-RRQR-STALE-SUCCESS` is `not_verified` (A01b-F1).
- pattern/value updates: **是** — `SDPX-SESSION-UPDATES` (`partially_verified`), M03, and
  `SDPX-REPLAY-CANCELLATION` = `verified`.
- concurrent session ownership: **部分** — same-precision concurrency is covered
  (S06 §5, P03 `365/365`); different-precision in-process concurrency is `unsupported`
  and is R9.

**5. 两库factor摘要是否便宜且统一语义；读取诊断是否不触发大对象copy/深扫描？—— 是，按测量。**
`M01`'s measured summary reads allocate **nothing**: `summary_read_bytes = (kind=0,
status=0, state=0, inertia=0, grammar=0, pivots=0, block_counts=0)`, with
`summary_record_bytes = 32` (`rebuild-reports/I03/logs/pre_delete/M01_lease_token.log`).
Semantics are unified by the contract (`M01 factor/cache/summary contract`, 204/204) and
`MFLA-PIVOT-GRAMMAR` is `verified`. The qualification the packet already recorded: MFLA's
summary semantics are the reviewed ones at `e3805c9`, and the "read does not deep-scan"
property is demonstrated by the zero-allocation counters above, not by an asymptotic proof.

**6. 所有宣称支持的算术、形状和稀疏路线是否真实运行过；prototype/unsupported是否显式？—— 部分，且显式性良好。**
Ran: Float64 dense/sparse, BigFloat, MultiFloat (S01 MultiFloat leg 8/8 under
`--project=$REBUILD_ENV`), rectangular QR, QDLDL sparse, the five provider legs, LP/SOC/SDP
routes. Not run, explicitly: tiers 16/64 (host), MF/BF live-kernel comparisons in the
default project (providers unresolvable there — recorded `unsupported`, ADR-003 §3),
`QDLRL`, native allocator accounting, in-process mixed-precision concurrency,
`GenericSparseCholeskyFactor`'s entry points (zero callers — F-I03-1). Prototype/unsupported
labelling is explicit and machine-checkable: `capability_table.tsv` has no row that counts
an `unsupported` item as passing, `certified 0`, and 5 rows are `not_verified` by name.

**7. 性能记分表是否包括setup/迭代/恢复/认证、失败、RSS和编译？—— 部分（否 as a complete scorecard）。**
Present: setup/iteration/recovery phase instrumentation (`src/performance_trace.jl`,
`SDPX-SETUP-PLANNING`), certificate timings (S04/S01), failure counts and statuses,
RSS (`P03`'s calibration numbers, §7), compilation (`INFRA-REPRODUCIBLE-BENCHMARK`,
precompile gate). **Absent: end-to-end wall clock** (`B04` `wall_clock_timing = not_run`;
`M03`'s E2E baseline `not_run`) and **native allocation accounting**
(`B03`/`P03` `not_run`, no MPFR hook). No paper in this packet claims a speed-up on the
strength of a kernel ratio alone — the 92.5× figure is explicitly a kernel redundancy, not
an end-to-end effect.

**8. 默认策略是否以跨形状证据决定，而非某一个CSDR案例？—— 部分。**
The default *pipeline* is single and verified (`INFRA-SINGLE-DEFAULT-PIPELINE`), the
storage/route decision is made by the planner from a structural classification
(`src/pipeline/classify.jl`, `plan.jl`), and Q01 supplies cross-shape scaling evidence.
But the *newest* policy layer is deliberately **not** default: `SDPX_CORE_ROUTE_PLANNER`
defaults to `"legacy"` and its `"model"` branch is shadow-only
(`test/core_route_planner.jl:193-212` asserts exactly that no silent default change
happens). So the honest answer is: the default was **preserved**, not **re-decided**, on
cross-shape evidence; promotion is still gated on calibration evidence that the shadow run
has not produced. `SDPX-DEFAULT-ROUTE-PLANNER` is `partially_verified` (1/2).

**9. 新路线是否真正替代旧控制逻辑，而不是又包一层？—— 部分（主循环是替代；若干子路线仍是并列开关）。**
Replacement, verified: the S02 cutover deleted the old loop body and left
`product_hsd_solve!` a thin wrapper over `solver_run_session!`
(`SDPX-HSD-LOOP`, `SDPX-NO-SECOND-HSD-LOOP`); the nzrange fix is a relocation of guards,
not a wrapper; M01 IP-3 was a **relocation** with 12-fixture equivalence evidence and IP-4
a 14-insertion/0-deletion additive alias. Not replaced: `SDPX_CORE_ROUTE_PLANNER`
(`legacy` default), `SDPX_HKM_VEC4`, `prepare_symmetric_core`,
`allow_expanded_bordered_fallback`, `linear_algebra_backend=:legacy` — all listed with
their defaults in §3. The criterion is therefore answered **部分**, and the enumeration
above is the evidence.

**10. 三库能否按兼容契约独立发布升级，且测试环境能复原？—— 部分。**
Independence: MFLA/BFLA are reached only through package extensions
(`Project.toml [extensions]`, ADR-002 §1, `baseline.md` §2); `INFRA-DEPENDENCY-BOUNDARIES`
is `partially_verified` (1/2, `I01.numeric_test = not_run`: the rollback rehearsal was not
run). Environment reconstruction: the record+check pair works and is control-tested
(three SHAs + three `Manifest.toml` sha256 + the Julia version; 10 of 10 control arms
behaved; `rebuild-reports/Q02/logs/recon_{positive,controls}.log`) — but the record pins
SDPX `c4b109a`, so **R4 is not satisfied for the release triple**, and `Manifest.toml` is
gitignored and therefore pinned by hash rather than by commit (`Q02` limitation). The
command pair that closes it is §5 item 5.

## 9. Limitations carried forward — and what this verdict is not

1. **This document is not a release approval.** Completing I03 does not approve any
   capability for production. The decision is BLOCKED, and every capability in §6 remains
   exactly as Q02's table states it: `certified 0`.
2. **R3 is `not_measured` by me.** The suites and the 28-leg matrix at the post-commit
   revision are the parent's step; my evidence is targeted (load + seven drivers, all
   exit 0). Until that run exists, nothing here should be quoted as "the suite is green at
   the release revision".
3. **R4's record names a superseded SDPX revision** until it is re-pinned.
4. **The two production hazards of §2.5 have opposite dispositions**: the
   `refactor_numeric!` node is fixed and verified; A01b-F1 is a live, reproduced defect
   **accepted as a known limitation** with the fix location blocked on B04's unwired split.
5. **Third-party hazards are stated, not solved**: `QDLDL.solve(Q, ::Matrix)` can raise
   `ReadOnlyMemoryError` or segfault the process, and the mechanism is not understood.
   Nothing in SDPX can fix it; SDPX's part is not to call it.
6. **The O(n⁴) SOC roundtrip is a documented limitation with no reproducible instrument**
   and no end-to-end measurement; the 92.5× figure must not be quoted as a measured
   speed-up.
7. **`RUN_HISTORY.md` is stale** (V01-F1) and is not cited here as a time series.
   **`REPORT_SCHEMA_AUDIT.md` is stale** (V01(c)) and is likewise not cited.
8. **Two published documents still carry a false mechanism** for R1 (F-I03-2):
   `support_matrix.md`/`.json`, `capability_table.tsv` and `release_checklist.md:196-197`
   repeat "src/SDPX.jl does not include src/certification/, so NO public route is wired to
   it", which is false (`src/SDPX.jl:195`). The *conclusion* is unchanged; the *reason* is
   wrong (the module is inert, not absent). `S02/report.json` acceptance[2] carries the
   same class of stale claim about `solver/loop.jl`. These are generated/copied texts; I03
   corrects the record here rather than editing another task's report.
9. **This task's reach does not extend to the packet's own unversioned material.**
   `rebuild-reports/` is in no repository; 588 cited artifact paths no longer resolve
   (mostly `/tmp`); a `pass` whose primary log cannot be re-read is a report's claim, not
   a re-read log.
10. **Not re-run by I03** (each with its reason): the three `Pkg.test()` runs and the
    28-leg matrix (dirty tree, no commit authority; handed back); B04's driver
    (writes its own evidence bundle; BFLA untouched); M02's driver (subject is MFLA's
    crate; MFLA untouched — the frozen leg is cited); Q02's reconstruction check (a depot
    built from empty; the frozen positive+control logs are cited); the `GET`-style
    `I01_reachability.jl` transcript (asserts nothing — V01 F44 — and is not used as a
    verdict anywhere in this document).
11. **No timing claim is made by this task**, and no benchmark answer is hardcoded. The
    host was shared and the packet's own performance rows are `not_run` or bounded.

## 10. Findings raised by I03

| id | severity | finding | evidence |
| --- | --- | --- | --- |
| **F-I03-1** | high | **§2's "orphaned, delete-or-wire" instruction for `sparse_la.jl`'s `GenericSparseCholeskyFactor` is under-evidenced and deleting on it would be unsafe.** A type-name grep measures the type, not the capability: `supports_sparse_generic`/`supports_sparse_execution` are **live predicates** consumed by `src/pipeline/plan.jl:291` and `_use_sparse_schur_sdp` (`sparse_la.jl:35-40`), and this family is the only implementation behind them; `plan.jl:508-520` records the descriptor `:generic_sparse_cholesky` while explicitly not instantiating it. Deleting the family while the predicates stay true converts dormant code into a latent dispatch failure. Disposition: keep, and hand forward a proposal that changes predicate + routing + family together with a before/after | `sparse_la.jl:22-28, 54-58, 1402-1504`; `plan.jl:291, 508-520`; zero callers for `sparse_factor`/`sparse_factor_solve`/`freeze_schur_pattern`/`GenericSparseCholeskyBackend(...)` repo-wide |
| **F-I03-2** | medium | **Two release documents carry a false mechanism for R1 and S02.** "src/SDPX.jl does not include src/certification/" is false (`src/SDPX.jl:195`); "src/SDPX.jl does not include src/solver/loop.jl" is false (`:194`). Conclusions survive for different reasons (inert module; inverted in-tree test) | `support_matrix.md/json`, `capability_table.tsv` (`SDPX-CERT-ORIGINAL-COORDS.blocked_by`), `release_checklist.md:196-197`, `S02/report.json` acceptance[2], `test/rebuild/S02.jl:641-648` |
| **F-I03-3** | medium | **`_sdpx_direction_trace` is called unconditionally on the per-iteration hot path** (`product_cone_hsd.jl:4193`), so every HSD step performs an `ENV` lookup and a `parse(Int, …)` before returning `nothing`. It is debug-only code with a hot-path cost. Not fixed at the freeze point (it is a behaviour-neutral but real change); retirement is to hoist the env check into a `const` read at load time or to delete the documented instrument | `product_cone_hsd.jl:4188-4194`, `:3995-4000`; survey call-site map |
| **F-I03-4** | low | **`iteration_predictor` is a write-only solver-state field** (`product_cone_hsd.jl:280`, assigned `:504`, validated in `public/settings.jl:396-398`) with no read site in `src/`; its only effect is via `factor_pair_admission.jl`. A public knob that appears to steer the HSD predictor but does not | `grep -rn 'iteration_predictor' src/`; the survey's Part 2 |
| **F-I03-5** | low | **`release_checklist.md:167` refers to an M02 planner flag `default_path = false` that does not exist in `src/`** — it appears only in docs and in the unapplied `docs/evidence/proposed/M02_wire_gemm_candidate.patch`. A checklist row asserting a property of a non-existent symbol cannot be verified | `grep -rn 'default_path' SDPX.jl/src` = 0 |
| **F-I03-6** | low | **`docs/evidence/P0_03_PLATFORM_DIRECTION_BREAKDOWN.md` uses `SDPX_DEBUG_DIRECTION` as its reproduction instrument**, which is the reason those knobs were retained. Retention is a *decision* with a cost (F-I03-3); if the instrument is ever deleted, that document must be marked as not reproducible | `docs/evidence/P0_03_…md`, `P3_01_BETA_EXPERIMENT.md`, `docs/performance/EXECUTION_STATUS.md` |

## 11. One-paragraph summary for a reader who reads only this

The rebuild produced a real, measured, self-audited system: three suites green at pinned
revisions with logs, a 28-leg driver matrix with a falsifiable verdict, 56 capabilities
mapped to evidence with `certified 0`, and an adversarial audit that refuted five of the
packet's own claims. It is **not releasable**, for reasons that are precise rather than
vague: the original-coordinate certificate criterion holds only on the routes that
succeed, because the downgrade branch has never been driven and no instrument exists to
drive it (R1); the "public layer applies no sign patch" claim has no patch to delete and no
runtime observation to replace it (R2); the sparse `factorize!` path still lets a lease
survive a refactor (R8); and one capability P03's own contract marks required is
`unsupported`, with the gate computing `FAIL` from the rows rather than asserting it (R9).
Deleting the old carriers did not change that, and no deletion was allowed to: what was
removed was one dead function, two undocumented debug knobs, and seven false comments.
