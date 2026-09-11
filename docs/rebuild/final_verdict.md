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
those numbers (**R3**) was marked `not_measured` by me and is now **satisfied** by the
parent's pinned run, which reproduces the baseline exactly (170 testsets,
Broken=7/Pass=9392/Total=9399 at `69c6c09`) — the citable half is the parent's, the
independent half is mine.

## 1. Revisions, and what is measured at which one

| object | revision | state |
| --- | --- | --- |
| SDPX, the revision this verdict's *measurements* were taken against | `c8fb65a0d5e15b6d03d34c4368cba7ad5f697094` (clean) | the deletion tree is `c8fb65a` + the 8 paths in §4 |
| SDPX, **the release revision** | **`69c6c0929d8027f29604ece68292b3c78f7d301e`** (short `69c6c09`), tree clean: B1 `2aef2ef` (retirement), B2 `08ad430` (stale comments), B3 `69c6c09` (this document) | the parent has run all three suites and the 28-leg matrix at a pin of `69c6c09` / `e3805c9` / `f087a72` and every number reproduces its baseline exactly: **R3 is satisfied** (`rebuild-reports/PARENT_VERIFICATION/release_revision/`) |
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
| **R1** | **not_satisfied** | `JULIA_DEPOT_PATH=$REBUILD_DEPOT:$HOME/.julia julia --project=$REBUILD_ENV -t1 /tmp/v01_t7_certificate.jl` → `rebuild-reports/I03/logs/final_tree/R1_certificate_routes.log` (exit 0); `grep -rn 'ResultCertificate{' SDPX.jl/src/`; `grep -rn 'SDPXCertification\.' SDPX.jl/src SDPX.jl/test \| grep -v certification/original.jl`; `sed -n '2576,2586p' SDPX.jl/src/hsd/native_hsd_public.jl` | Four routes (LP, SOC, SDP, infeasibility ray) terminate `cert_valid=true method=original_coordinates` — that is V01's F12/T7 positive half, independently re-run here. But the criterion is *every* route: the downgrade branch `native_hsd_public.jl:2581-2586` (status → `NumericalFailure` at `:2583`, reason `:original_coordinate_certificate_failed`) **has never been driven**, and four `ResultCertificate{T}` construction sites survive (`src/hsd/native_hsd_public.jl:2403`, `:2462`, `:2487`, `src/public/optimize.jl:341`). `grep` for `SDPXCertification.` outside its own file returns **0**, so the module is inert on the public path. The blocker is a **missing instrument**, not missing effort: I02 declined move 6 deliberately because a behaviour change to existing code needs its own before/after, and the plan marks it "deliberately not drafted". My tight-gap control (`gap_limit=1e-14`) did *not* exercise the invalid direction — the certificate was still valid — so the negative direction remains `not_run` |
| **R2** | **not_satisfied** | `rebuild-reports/I03/logs/i03_R2_sign_patch_sweep.log` (commands reproduced verbatim in the log): 8 patterns `'\.= *-' '= *-x' '= *-1 *\.\*' '\*= *-1' sign_flip flip_sign negate public_sign_patch` over `src/public/` (4 files) + `src/moi_wrapper.jl` + `src/frontend/` (2 files); `sed -n '470,476p' SDPX.jl/src/core/compiled_problem.jl`; `sed -n '472,475p' SDPX.jl/test/rebuild/S01.jl`; `grep -rn 'replay_public_signs' SDPX.jl/src SDPX.jl/test` | **8 patterns → 8 zeros.** There is no sign patch to delete, so the prescribed before/after MOI solve would compare a state to itself and can never unblock this row (V01 T8a, independently re-run here). `public_sign_patches=0` is still a **hardcoded literal** at `src/core/compiled_problem.jl:475` asserted against itself at `test/rebuild/S01.jl:474` — an assertion that cannot fail, and it is **not** cited as evidence here. `replay_public_signs` is defined at `compiled_problem.jl:449` and used at exactly one site, `S01.jl:473`, from a constructed `CompiledProblem`; it never inspects the public boundary. The closest thing to a sign correction in the boundary — the MOI interval dual at `src/moi_wrapper.jl:1845` — is a *sum* of two bridge duals, i.e. MOI conforming, not a patch. What is missing is the runtime observation: a count of sign applications at the public boundary during a real MOI solve |
| **R3** | **satisfied — all three suites and the matrix are green on clean trees at the release triple (the parent's measurement, cited)** | parent's pinned run at `/tmp/pinrel` with `SDPX 69c6c09 / MFLA e3805c9 / BFLA f087a72`, all three `dirty_paths=0`: `rebuild-reports/PARENT_VERIFICATION/release_revision/sdpx_pkgtest.log` (`Testing SDPX tests passed`, **170 testsets**, zero Fail/Error columns, and the column sums re-derived here as **Broken=7 / Pass=9392 / Total=9399 — the baseline exactly**) and `…/driver_matrix.summary` (**28 of 28 legs `exit=0`, `legs_run=28 legs_failed=0 failed_legs=none`, every leg with a `Test Summary`**; B03's leg ran with `B03_WIRED_INJECT_TEST=1`). MFLA `…/MFLA_pkgtest.log`: `tests passed`, 23 `Test Summary:` blocks, **Pass=4164 / Total=4164**; BFLA `…/BFLA_pkgtest.log`: `tests passed`, **10864/10864** — the frozen baselines exactly, at the byte-unchanged revisions `e3805c9` / `f087a72`. Not run by I03 — the tree is dirty by construction and `_require_clean_source` is asserted inside the suite (`benchmark/optimization/test_v2_fresh_process_profile.jl:210-215`). Command for the parent, at the post-commit SHA: `cd SDPX.jl && JULIA_DEPOT_PATH=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env-depot:$HOME/.julia julia --project=. -e 'using Pkg; Pkg.test()'`; and for MFLA/BFLA the same from their directories; then `bash SDPX.jl/scripts/rebuild/run_driver_matrix.sh rebuild-reports/PARENT_VERIFICATION/driver_matrix_post_i03` | Pre-deletion, at `c4b109a`: SDPX `tests passed`, 170 testsets, `failcols=0`, **Broken=7 / Pass=9392 / Total=9399**; MFLA `e3805c9` **Pass=4164 / Total=4164**; BFLA `f087a72` **10864/10864**; matrix `legs_run=28 legs_failed=0 failed_legs=none` with a `Test Summary` on every leg (`rebuild-reports/PARENT_VERIFICATION/driver_matrix_frozen.summary`, whose header records the script it used as the live, uncommitted copy with `sha256 4b1f872e…` — the per-leg lines are the evidence; the aggregate alone is not, per F42). My **post-deletion targeted** legs are all exit 0: load, R1 probe, S01 189, S03 193, S04 313, S06 214, S07 289, S05_none 1466, S05_mfla/S05_bfla 1522, A01_default 1673 (`rebuild-reports/I03/logs/final_tree/`). That targeted set is the independent half, not the citable half: the row is established by the parent's pinned run above, and the SDPX numbers reproduce the baseline exactly (170 testsets, Broken=7/Pass=9392/Total=9399) — which is the strongest single result in this verdict, because it says the retirement changed no measured behaviour. MFLA/BFLA suite numbers are corroboration only: those two revisions are byte-unchanged (`git rev-parse` equal), and their matrix legs were re-run rather than inherited |
| **R4** | **not_satisfied** | `python3 SDPX.jl/scripts/rebuild/check_reconstruction.py --record SDPX.jl/docs/rebuild/RELEASE_REVISIONS.txt --target /tmp/q02recon --depot /tmp/q02recon-depot` → `rebuild-reports/Q02/logs/recon_positive.log`, controls `recon_controls.log` (`arms correct: 10 arms wrong: 0`) | The record+check pair exists and works: three SHAs + `Manifest.toml` sha256 ×3 + the Julia version, reconstructed into a depot built from empty, 60 vs 60 entries, dependency set identical, with 10 two-sided control arms. But the record **pins SDPX `c4b109a`**, and the release triple I03 proposes moves SDPX. A record that names a revision the release does not ship does not satisfy the row. Retirement is one command pair: re-pin, then re-check (`pin_revisions_env.sh <new SHA> e3805c9 f087a72 --clean-check`, then `check_reconstruction.py`). Not a limitation — a two-command step |
| **R5** | **satisfied** | `python3 SDPX.jl/scripts/rebuild/validate_reports.py .` → before I03's report: **`reports=26 errors=0 warnings=147`**; after this report was added: **`reports=27 errors=0 warnings=147`**, with `--only I03` reading `I03 ok errors=0 warnings=0` (`rebuild-reports/I03/logs/i03_validator.log`). Acceptance re-count over the 25 packet reports, reproduced by me: **122 verified / 29 partially_verified / 4 not_verified = 155** (A01b contributes 9 verified as an ad-hoc report). Capability table: `rebuild-reports/Q02/capability_table.tsv` — 56 capabilities, 53 with passing evidence, 26 `verified` / 25 `partially_verified` / 5 `not_verified`, **`certified 0`** | The two entries §4 held open were resolved in the honest direction and are re-verified here: the A01 composite is split into `pass` (20) + `unsupported` (3, explicitly not counted as passes), and S02's interrupted run is `pass` with its non-zero exit stated plus a limitation recording that the clean run supersedes it. `not_run` remains distinguishable from `pass` throughout |
| **R6** | **satisfied (dense path; R8 carries the residual)** | `cd MultiFloatLinearAlgebra.jl && JULIA_DEPOT_PATH=… julia --project=$REBUILD_ENV -t1 test/rebuild/M01.jl` → `rebuild-reports/I03/logs/pre_delete/M01_lease_token.log` (204/204 pass); `sed -n '350,372p' SDPX.jl/src/la/factor_lease.jl`; V01 `independent_audit.md` §M01 IP-2 and `V01/report.json` F6 | Re-measured by me at this revision: `cache_leases = (first = 0x…01, second = 0x…02)` — the token **differs** across two same-size dense refactors, with the generation advancing `0→1→2`. The `refactor_numeric!` admission refusal now revokes before returning (`factor_lease.jl:371` `_revoke!(h.lease, EvRefactorPreflightRejected, …)`; branch at `:360-372`), so all four exits revoke rather than three of four. **Scope, stated so the row is not read wider than it is:** this holds on the four dense `factorize!` methods; the sparse path does not advance the generation at all, which is **R8** |
| **R7** | **satisfied (bounded, as its own status cell states)** | `rebuild-reports/PARENT_VERIFICATION/driver_matrix_frozen/M02.log` (leg exit 0, `failcols=0`) + `rebuild-reports/M02/report.json` numeric tests `S05-F1 multi-RHS, dense caches (ldlt, cholesky, lu)` = pass, `S05-F1 multi-RHS, sparse QDLDL cache` = pass, `multi-RHS capability claim` = pass | `capabilities(MF)` no longer rests on a timing claim: it rests on method identity, bitwise agreement with the per-column loop, and the residual — explicitly *not* on timing, because M02's timing instrument **failed its own control** (a known per-column loop measured a sub-1 ratio). MFLA's `S05-F1` "is the dense matrix path genuinely batched" sub-question is **bounded, not settled**, and that bound is the honest form. I did not re-run M02 post-deletion: its subject is MFLA's own crate, and MFLA is untouched at `e3805c9` |
| **R8** | **not_satisfied — reproduced, and NOT accepted** | `JULIA_DEPOT_PATH=… julia --project=$REBUILD_ENV -t1 /tmp/v01_ip2_sparse.jl` → `rebuild-reports/I03/logs/pre_delete/R8_ip2_sparse_generation.log` (exit 0) | Re-measured by me at this revision, in one process with a positive control: `DIRECT_SPARSE generations=(0,0,0) delta_first=0 delta_second=0` while `DENSE_LDLT generations=(0,1,2) delta_first=1 delta_second=1`, and **SDPX's own seam** (`SDPX.SparseQDLDLProviderCache` + `SDPX._qdldl_provider_factorize!`) also `delta=0` with `issuccess=true`. So on the sparse path a lease taken before a refactor still validates after it — the `M01-F4`/`P02-F1` defect, unchanged, reachable through the real package extension (`ext/MultiFloatQDLDLExt.jl:160`) that loads for any user, not only under a test harness (V01 F6). **Decision: I03 declines to accept this as a release risk.** It is a default-reachable correctness gap, the fix is prepared and behaviourally verified (proposal `I02-P2`: instrument `MFSparseLDLCache.factorize!` so the generation advances, plus the evidence that its commit points exist), and accepting a correctness gap that has a one-line prepared fix would be exactly the "hazard with a false sense of coverage" ADR-002 §4 names. Retirement: apply `I02-P2` and re-run this same command until the sparse deltas read `1,1` with the dense control still `1,1` |
| **R9** | **not_satisfied — demotion declined** | `cd BigFloatLinearAlgebra.jl && P03_OUT=<dir> JULIA_DEPOT_PATH=… julia --project=$REBUILD_ENV -t1 test/rebuild/P03.jl` → `rebuild-reports/I03/logs/pre_delete/P03_release_gate.log` + `…/p03_out/P03_run_main.txt:112-117` (exit 0); primary log preserved unmodified as `rebuild-reports/I03/logs/P03_run_main.PRE_I03.txt` | Re-run by me at `f087a72`: the driver is **`365/365 Pass`** at the same revision whose own computed gate reads `release_gate_verdict = FAIL`, `required_rows = 10`, `verified_required_rows = 9`, `failing_rows = ["concurrent_sessions_different_precision_in_process_parallel"]`, with four unsupported rows of which exactly one is `required=true`. **`unsupported` means the capability is not offered — it does not mean broken**, and "the gate is red" must never be reported as "the tests fail". **Decision: I03 does not demote the row.** Editing `required=true` to green at the freeze point is the same failure as counting a SKIP as a PASS (the card's hard prohibition), and the correct fixed form is an ADR-level statement that in-process, concurrent, different-precision sessions are architecturally unavailable because BigFloat precision is process-global and `_ambient_guard` throws `PrecisionMismatch` — a contract change with its own evidence burden, which belongs to the ADR owner, not to the last task. Retirement: implement process-isolated precision sessions, or have the ADR owner change P03's contract row *with the measured reason* and re-run the gate |

### The rule that overrides everything else

R1 and R2 are unconditional blockers and both are unmet ⇒ **BLOCKED**. R8 and R9 were
eligible for explicit acceptance, and I03 considered and **declined** both, with the
reproduction and the retirement condition recorded above. That is a decision, not an
omission: even if R1 and R2 were retired tomorrow, R8 and R9 would still hold the release
until they are fixed or accepted on a stated basis. R4 is also unmet today and is a
two-command step, not a limitation. R3 is satisfied by the parent's pinned run at the
release revision, not by mine (the commit protocol forbids my running the gated suite). R5, R6 and R7 are satisfied within their stated scopes.

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
file is restorable with `git -C SDPX.jl checkout c8fb65a -- <path>`). The parent committed
the three boundaries as **B1 `2aef2ef`, B2 `08ad430`, B3 `69c6c09`** (release revision
`69c6c0929d8027f29604ece68292b3c78f7d301e`), after re-verifying the deletion set itself:

* **B2 is comment-only mechanically**: `git diff -U0` filtered to non-comment changed lines
  gives **0** for each of the six files — the claim is checked, not asserted.
* **B1 removed no correctness check by inspection**: in `equality_reduction.jl` only the
  `println` inside the `SDPX_DEBUG_EQUALITY_RECOVERY` guard was removed and
  `valid || return false` is retained; in `product_cone_hsd.jl` each removed block held only
  `showerror`/`println`, with `state.diagnostic`, the `false`, the
  `return direction_ok ? HSDStepOK : HSDStepDirectionFailed` and the fail-closed comment
  outside and retained. That is the card's named risk discharged by inspection, by a party
  other than the author.
* **The deleted symbols have no code references**: `grep -rn '_product_hsd_soc_condition_budget'`
  post-deletion returns 3 hits, **all prose** (the pre-existing evidence note and two lines
  of this document), and both deleted switches have 0 `.jl` references.
* **The kept-set rationale is a positive control**: `SDPX_DEBUG_DIRECTION` 6 refs,
  `SDPX_DEBUG_LINE_SEARCH` 3, `SDPX_DEBUG_ITER` 2, `SDPX_CORE_ROUTE_PLANNER` 6 — a decision,
  not a half-finished sweep.

These four checks are the **parent's**, recorded here as parent verification, not as mine.

| # | path | change | reachability evidence | test evidence | pre-change blob |
| --- | --- | --- | --- | --- | --- |
| B1a | `SDPX.jl/src/hsd/product_cone_hsd.jl` | delete `_product_hsd_soc_condition_budget` (23 lines) + 4 `SDPX_DEBUG_SYMMETRIC_CORE` print blocks; correct the false docstring at `:577` | **code** references (`grep -rn --include='*.jl'`): 1 before (its own definition), **0** after. Prose references remain and are expected: the pre-existing `docs/evidence/OPEN_SOC_ROUNDTRIP_ON4.md:12` "DEAD" line and this document — which is why the claim is stated over code, not over a repo-wide grep (the parent caught the repo-wide form reading 3 post-deletion). The two deleted debug knobs have **0** references in any `.jl` and **0** in `docs/`/`rebuild-reports/` | `final_tree/`: load exit 0; S01 189 pass, S03 193, S04 313, S06 214, S07 289, A01_default 1673, all exit 0 | `a08ada68e809ca19b1304448f1630ad2dcc2657b` |
| B1b | `SDPX.jl/src/hsd/equality_reduction.jl` | delete the `SDPX_DEBUG_EQUALITY_RECOVERY` print block | as above: 0 setters, 0 docs; the guarded body is a single `println` and the decision sits outside it | same legs; the infeasibility route (case C of the R1 probe) exercises this function and still returns a valid ray certificate | `19f1d987968e560abfd1a7cb304a0e2bfd125144` |
| B2a | `SDPX.jl/src/hsd/product_cone_solve.jl` | correct the false header claim ("deliberately not wired to the public/MOI route") | `native_hsd_public.jl:2204` calls `product_hsd_solve!`; that file's own header names the chain | comment-only; load + all legs | `533547075c813e7f4369dfa1ed4d5bb3e065c078` |
| B2b | `SDPX.jl/src/hsd/hsd.jl` | correct "retained for callers" (zero callers measured) | `grep -rn '_hsd_column_reduction'` = 2 definitions + 1 docstring cross-reference | comment-only | `388dd47b818a78c904321b47df7bdd8a61ed73a9` |
| B2c | `SDPX.jl/test/rebuild/S01.jl` | correct two false "not yet part of the package entry point … I01 will include" notes | `src/SDPX.jl:185-186` includes `core/compiled_problem.jl`, `core/transforms.jl` | S01 driver re-run: 9 testsets, 189 pass, exit 0 | `7c3c2cb96db3a2fa39889f0920a403b324f6e8fa` |
| B2d | `SDPX.jl/test/rebuild/S03.jl` | correct the false "NOT yet wired into `src/SDPX.jl`" loading note | `src/SDPX.jl:187-190` includes the four `kkt/` files | S03 driver re-run: 2 testsets, 193 pass (1 Broken row = baseline), exit 0 | `978a3447bc4159cb4d30cde825b75e4ab48818b9` |
| B2e | `SDPX.jl/test/rebuild/S06.jl` | correct the false "I01 will add to `src/SDPX.jl`; until then" note while preserving the standalone fallback | `src/SDPX.jl:196-198`; `plan_setup` at `src/planning/setup.jl:598` | S06 driver re-run: 1 testset, 214 pass, exit 0 | `73afb021112547b78b702e6461de9d716706d4ba` |
| B2f | `SDPX.jl/test/rebuild/S07.jl` | correct the false "NOT in the package include graph (wiring is I02's job)" note | `src/SDPX.jl:203-207` includes `session/{update,replay,cancellation}.jl` | S07 driver re-run: 3 testsets, 289 pass, exit 0 | `1248fbbfd9cb22abb9f0f20a5cc1b1c91d3dfbc7` |
| B3 | `SDPX.jl/docs/rebuild/final_verdict.md` | this document (new file) | — | — | — |

**Third-party licences and provenance are untouched.** Every changed path is first-party
SDPX source, test or documentation; `git diff --name-only` matches no `licen|third|vendor`
path, and no vendored dependency, patch record or evidence document was modified. The
third-party surface this task *touched* is only what it cites: QDLDL's unsound raw matrix
entry point and the `MultiFloatLinearAlgebra`/`BigFloatLinearAlgebra` package extensions,
both left byte-unchanged.

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
6. **R3** — done by the parent at the pinned release triple: SDPX `tests passed` with the
   baseline totals, and the 28-leg matrix 28/28 with a `Test Summary` per leg
   (`rebuild-reports/PARENT_VERIFICATION/release_revision/`). Remaining corroboration: the
   MFLA/BFLA `Pkg.test()` re-runs at their byte-unchanged revisions, and the same matrix
   after the B4 docs-only commit (which cannot change a leg, since it touches one markdown
   file). For any future run, read **per-leg** lines, never the aggregate, when the script
   predates `d6ca577`.

## 6. Capability split: production / experimental / unimplemented

Source of truth: `rebuild-reports/Q02/capability_table.tsv` (56 rows) and
`SDPX.jl/docs/rebuild/support_matrix.md`. States are Q02's, not re-derived here; the
grouping is mine and each group names its membership rule.

**PRODUCTION — offered, default-reachable, with passing numeric or structural evidence
(26 rows `verified`).** "Production" here means *the default pipeline offers it and its
evidence row is `verified`*; it is **not** a release approval. The release is blocked (§0),
`certified 0` means nothing in this table is certified, and the four default-reachable
capabilities named at the end of this group carry open defects that the block records. `SDPX-PRECISION-UPGRADE`, `SDPX-STEP-STRATEGY`,
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

**Accepted tasks (the card asks for these explicitly).** Of the 26 packet tasks, **2 are
`accepted`: A00 and I01**; the other **24 are `needs_review`** — including Q02, V01 and I03
itself, because a worker does not accept its own work and I03 does not accept another
task's. "A report exists" is therefore not "the task is accepted", and the 19 tasks that
were already at `needs_review` before batch 5 are not silently promoted by this verdict.
The per-task acceptance counts are in `rebuild-reports/I03_DOSSIER.md`; the release-relevant
ones are restated in §2 and §6.

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
| Compilation | Q01 phases `first_compile` = 22.1 s (float64_default), 22.0 (float64_env), 50.9 (mf_x2), 48.9 (bf_256); S07 `7441.9 ms ✓ SDPX`; `using SDPX` here ~8 s | `SDPX.jl/benchmark/rebuild/measure_result.toml`, `rebuild-reports/S07/S07_pkgtest.log:90`, `rebuild-reports/I03/logs/final_tree/LOAD.log` | measured — except B03's `1116.4 ms`, which is **not** corroborated (below) |
| The scorecard is **incomplete** for a production release | setup/iteration/recovery/certification phases are instrumented, **failures and RSS partly, compilation partly, no end-to-end wall clock, no native allocator accounting** | §11 Q7 below | **partial** |

### Performance numbers that are NOT backed by a raw record — do not cite them

The card's second acceptance criterion is "performance claims supported by raw records".
These are the claims that fail it, found by re-reading the records they name:

| number | where it is claimed | what the record actually says |
| --- | --- | --- |
| B03 compilation `1116.4 ms` | `rebuild-reports/B03/report.json` (`performance.compilation_mode`, and acceptance `verified_by`) | the log says `1401.8 ms  ✓ BigFloatLinearAlgebra` (`rebuild-reports/B03/B03_driver_WIRED_static.log:24`); **no log contains 1116.4** |
| B03 `threads_requested = 4`, `threads_executed = 4` | `rebuild-reports/B03/report.json` `performance` | all **106** `julia_threads=` values in `B03_perf_samples.txt` are **1**; the 1/2/4 arms are three separate `-t1` processes (`B03_driver_perf.log:34-36`), and the budget phase records `observed_peak_workers=1` |
| B03 RSS samples | `B03/report.json`; `support_matrix.md` cites `rebuild-reports/B03/B03_rss_child_stderr.log` | **that file does not exist**; `B03_driver_rss.log` is **12 × `MEASURE rss_failed` with 0 successes**, and the driver discards failing children's stderr, so the cause is undetermined. The only successful line is a header |
| `lp_afiro_style`: 27.3 s first / 0.00085 s warm | `SDPX.jl/docs/rebuild/benchmark_protocol.md:23` — a *binding protocol document* | neither number appears in any artifact; the named baseline records `first_compile = 22.106584875000003` and `warm_fresh_setup = 0.000295917…`; the `0.000854` trace is another benchmark's `kkt_seconds` |
| M02 SoA penalty 1.30–1.37× | `rebuild-reports/I02_WORK_PLAN.md:896` | the declared artifact records `layout_soa_over_aos_ratio = 1.4486586044978993`, **outside the quoted range**; M02's own report records the correction |
| 92.5× SOC roundtrip | V01-F11 / `docs/evidence/OPEN_SOC_ROUNDTRIP_ON4.md` | no archived instrument; a kernel redundancy, not an end-to-end effect |

**Consequence:** the load-bearing performance statements in this verdict are the measured
ones in the table above; the six numbers in this subsection must not be cited as
measurements. Two of them (B03's compilation figure and its thread counts) contradict the
records they name, which is finding **F-I03-8**.

## 8. Answers to §11 of `ENGINEERING_REBUILD.md`

Verdicts at a glance: **1 partial · 2 partial · 3 yes · 4 partial · 5 partial · 6 partial ·
7 no · 8 no · 9 no · 10 partial.** Unmet items are answered as unmet.

**1. 一个问题、一个主循环、一个当前KKT策略，是否已成为可验证事实？—— 部分.**
One problem: `src/public/optimize.jl:375-402` compiles once, classifies once, dispatches
once; MOI is the sole adapter and `optimize!` invokes the public seam exactly once
(`src/moi_wrapper.jl:491-494`); `settings.algorithm` admits only `:auto`
(`src/public/optimize.jl:56-68`). One loop: `src/SDPX.jl:194` includes `solver/loop.jl`;
`test/rebuild/S02.jl:648` asserts the include, `:652` asserts
`count("for _ in 1:Int(max_iterations)", production) == 0`, `:653-655` asserts exactly one
`solver_run_session!` method — and those pre-cutover assertions were **inverted** rather
than deleted, so a silent rollback fails. **Not a clean 是:** three KKT strategies are
admitted (`src/kkt/strategy.jl:285-290`: `:augmented`, `:schur`, `:fixed_trace`),
`kkt_route` is a four-valued axis (`src/hsd/product_cone_hsd.jl:335-336`), and the loop
still carries two same-iterate internal re-route retries
(`:3893`, `:3912`, call site `:4144`) that no test exercises. The one-loop evidence is a
source scan and is self-labelled as one (`S02.jl:604`). So: one problem and one loop body;
**not** one strategy and not one route.

**2. public/MOI/compatibility是否不再实现数值证书、factor策略和算术kernel？—— 否.**
Kernels: yes — grepping `cholesky|CHOLMOD|qr(|lu(|ldlt|factorize` over `moi_wrapper.jl`,
`frontend/` and `public/` returns one hit, a string literal in an error message
(`src/public/settings.jl:274`). Compatibility: the old layer was **deleted, not shimmed**
(`git log --diff-filter=D -- src/compat*` → `10a8954`, 295 lines). Factor strategy:
partial — `settings.kkt_route` is still a public/expert field (`public/settings.jl:410`,
`moi_wrapper.jl:1272-1274`). Certificates: **no, and the responsibility was duplicated
rather than moved.** Three certificate implementations coexist: `certificates/certificates.jl`
(`src/SDPX.jl:106`), `certification/original.jl` inside `module SDPXCertification`
(`:195`, loaded and **never called** — `grep -rn 'SDPXCertification\.'` outside its own
file = 0), and the public one at `src/public/optimize.jl:223-357` whose construction site
`:341` is one of the four remaining. That is **R1**, and its correct statement is
"inert module, live duplicate", not "module absent".

**3. provider是否只拥有数值事实和内存，不决定HSD/原始可行域/终止？—— 是.**
The provider op set is a ten-value enum with **no** HSD/termination/convergence member
(`src/la/protocol.jl:30-41`); the minimum contract takes and returns no HSD state,
feasible region, tolerance or termination decision (`src/kkt/session.jl:143-160`);
`FactorSpec` "contains no policy" (`:163-171`); and the two libraries contain no
solver-policy vocabulary at all (`grep -rni 'hsd|kappa'` over both `src/` trees → none;
the single convergence-vocabulary hit, `MultiFloatLinearAlgebra.jl/src/residual.jl:299`,
is an explicit refusal: "performs no convergence test, iteration, fallback, or precision
escalation"). Negative tests exist: S03's `CapabilityLiar` is refused with
`liar.generation == 0` and no numeric work attempted (`test/rebuild/S03.jl:773-786`);
S05's stale-`:success` provider cannot buy a solve (`test/rebuild/S05.jl:263-290`).
Qualification: V01's only provider-boundary finding (F6) is a provider **failing to report**
a numeric fact — that is R8, not an authority leak.

**4. factor失效、precision/rounding、pattern/value更新、并发session的所有权是否有测试？—— 部分.**
pattern/value update: **yes** — a value update keeps `delta == 0`, a structural change
throws `PreparedStructureMismatch` and drops the symbolic slot
(`test/test_r2_full_qualification.jl:35-48`), with the same-size/same-nnz/pattern-changed
case in `test/session_symbolic_lease.jl:94-108`. factor invalidation: suite-level yes for
the caches (MFLA `test/factor_caches.jl:611` "fail-closed: no stale success"), but
`src/la/factor_lease.jl` and `src/la/admission.jl` have **zero in-suite coverage**
(`grep -rn 'FactorLease|FactorHandle|AdmissionRefused|SDPX.admit(' test/*.jl
test/provider_contracts/*.jl` → no output) and the ADR-003 §6 property runs only in
`test/rebuild/S03.jl`, which is outside `Pkg.test()`. precision/rounding: provider yes
(BFLA `test/precision.jl`), SDPX layer **no** — the exact property exists as an **orphan**
test (`test/test_r1b_owned_object_matrix.jl:106`, launched by nothing) and
`session_rounding`/`session_rounding_supported` (`src/session/update.jl:154-175`) have no
executable assertion anywhere. concurrency: **partial** — two-session isolation and the
sequential-collision refusal are tested (`test_r2_full_qualification.jl:166-241`), but that
file says itself it "does NOT establish multithread/task-level concurrency qualification",
and P03's required different-precision row is `unsupported` → **R9**. Residual: **R8**.

**5. 两库factor摘要是否便宜且统一语义；读取诊断是否不触发大对象copy/深扫描？—— 部分.**
Cheap: **measured**. `summary_read_bytes = 0` for all seven accessors and
`summary_record_bytes = 32` against a 4096-byte factor matrix
(`rebuild-reports/I03/logs/pre_delete/M01_lease_token.log`, my re-run; M01's own logs
agree). At the SDPX seam, `@allocated factor_summary(h) == 0` warm with `allocs == [0,0,0]`
at n = 8/32/128 and size-independent cold allocations
(`test/rebuild/S05.jl:354-419`; 761/761 in both live legs), and the hot path records
`deep_calls == 0`, `factor_copies == 0` (433/433). Structural, not just measured:
`FactorSummary` is forced concrete and immutable and **no field may be an
array/string/dict/set/tuple** (`src/la/protocol.jl:662-687`), so a summary that could hold
a matrix cannot be constructed. Unified semantics: **at the SDPX boundary yes** (one
`FactorSummary` for both providers), **between the two libraries no** — MFLA's 14-field
summary and BFLA's `FactorFacts` share no field names and there is no differential test;
the unification exists only because SDPX re-models it. Diagnostics reads: the clause is
**false for two queries** — B04-F3 records `factor_diagnostics(::BFLALUCache)` rescanning
`pivots` O(n) on every call and `factor_diagnostics(::BFLALDLTCache)` still copying 640 B
at n = 8 (`rebuild-reports/B04/report.json:351`).

**6. 所有宣称支持的算术、形状和稀疏路线是否真实运行过；prototype/unsupported是否显式？—— 部分.**
`unsupported` is explicit and machine-visible: across the 25 packet reports the numeric
tests are **267 pass / 12 not_run / 10 unsupported / 5 fail = 294** (adding the ad-hoc
A01b report: 284 / 12 / 10 / 5 = 311 — the two scopes are stated because the wider number
has been quoted without its extra report), and ADR-003 §3 forbids an `unsupported` from
satisfying a required capability. `prototype` is **not a label in any
vocabulary**: `grep -c prototype` over `support_matrix.json` and `capability_table.tsv`
returns 0, and `EVIDENCE_STATE_VOCAB = {verified, partially_verified, not_verified}`
(`scripts/rebuild/gen_support_matrix.py:62`). ADR-004's sparse provider is
`Status: PROPOSED (P01, needs_review). Not accepted.` with "keep the explicit prototype"
and "not wired into any public `optimize!` route" (`ADR-004:3`, `:133-140`) — a reader of
the capability table alone cannot see that. Claimed but never run: **MultiFloat x3/x4**
(the arm list is `(:float64, :multifloat_x2, :bigfloat_256)`,
`benchmark/rebuild/manifest.jl:85-95`, while `release_checklist.md:161` claims x2/x3/x4);
BigFloat-512 is not a declared arm; Double64 was **removed** (`CHANGELOG.md:290`). The
MF/BF product outcomes include **5 `fail` rows that were kept as failures** (MF
`mixed_soc_nonneg` `MethodError nzrange(::Matrix{MultiFloat})` — since fixed by I02's
guards — and BF `numerical_breakdown` / `memory_upper_bound_exceeded`), which is the
behaviour the packet asks for.

**7. 性能记分表是否包括setup/迭代/恢复/认证、失败、RSS和编译？—— 否.**
The `performance` block in all 26 reports is a fixed 12-key schema; setup, iterations,
recovery, certification and RSS have **no field**. What exists is single-task: setup
timed by Q01 only (`warm_fresh_setup = 0.000295917 s`); iterations **counted but never
timed** (zero hits for any per-iteration timing key across `rebuild-reports/` and
`benchmark/rebuild/`); recovery measured as **accuracy** (`7.8517e-78` against an
`8.8434e-75` tolerance) with replay timing `not_run` (`benchmark_protocol.md:44`);
certification a **boolean**, although the engine records the number
(`src/hsd/phase_timings.jl:32`) and the payload **drops** it
(`grep -c certification_seconds measure_result.toml` = 0); failures measured by Q01 only
(14 of 56 rows failed, kept); RSS by Q01 and P03, while B03's RSS phase is 12/12 failed
with no captured stderr; compilation by Q01 and S07. **Five of the seven dimensions the
question names are missing.** Thread denominators, named: `Sys.CPU_THREADS = 4` (Julia),
`length(Sys.cpu_info()) = 10` (OS); rejected tier requests are recorded as `unsupported`
with that host fact; **executed worker threads were 1 in every measured arm**.

**8. 默认策略是否以跨形状证据决定，而非某一个CSDR案例？—— 否.**
No default was *decided from* cross-shape evidence; the one behaviour-changing policy is
**deferred in shadow mode precisely because that evidence does not exist**
(`src/hsd/native_hsd_public.jl:2015-2027`: "It is NOT yet the default. The plan requires a
representative end-to-end improvement, with paired receipts, before the default policy may
change"). The planner's coefficients are "calibrated, not fitted" priors
(`src/hsd/core_route_planner.jl:1-60`); the pre-existing single-dimension rule is retained
as the documented fallback (`:257-264`); M02's shape-packing plan keeps
`default_path = false` at **both** construction sites
(`MultiFloatLinearAlgebra.jl/src/planning/gemm_plan.jl:306`, `:426`; no
`default_path = true` anywhere in either tree) — the checklist claim is verifiable in
**MFLA**, not in SDPX, which is where my first reading of it went wrong; and
`settings.kkt_route`/`provider`/`linear_algebra_backend` defaults are inherited unchanged.
The evidence base is real but thin: **8 deterministic cases** (2 LP, 4 SOC, 1 PSD, 1 mixed)
× 3 arithmetic arms, with **no sparse case and no large-scale case**
(`benchmark/rebuild/manifest.jl:409-427`). The single-case risk the question names is not
present — but neither is the cross-shape licence to change a default.

**9. 新路线是否真正替代旧控制逻辑，而不是又包一层？—— 否.**
The shipped default still executes the **old** rule:
`planner_authoritative = get(ENV, "SDPX_CORE_ROUTE_PLANNER", "legacy") == "model"`
(`:2045`), with `use_compact_schur` falling back to `legacy_dimension_rule`
(`:2043`, `:2046-2049`) — the new planner is computed and *reported* on every solve
(`:2029`) but cannot change the executed route, and **no test anywhere sets the variable
to `"model"`** (`test/core_route_planner.jl:193-196, 212` pins the shadow default and
restores the environment). A second `:legacy` path is a legal public value
(`src/pipeline/options.jl:32`, reachable from `settings.provider` and the MOI attribute)
with **zero tests** (`grep -rn linear_algebra_backend SDPX.jl/test` → nothing). Where
replacement *is* real: the loop extraction (S02 inverted its assertions rather than
deleting them) and the nzrange fix (guards relocated, not wrapped). So the honest answer
is 否 as stated, with the enumeration above as the evidence.

**10. 三库能否按兼容契约独立发布升级，且测试环境能复原？—— 部分.**
Independence: each library lists only its own dependencies, and the string `SDPX` appears
in the other two only in four comments (`I01_integration_record.md:277-283`; I01's
acceptance is `partially_verified` — "a text measurement over `src/hsd/`, not a call
graph"). Environment: the record+check pair works and is two-sidedly controlled — three
40-hex SHAs + three gitignored `Manifest.toml` sha256 + the Julia version, positive arm
PASS, **10 of 10 control arms correct** (`rebuild-reports/Q02/logs/recon_{positive,controls}.log`).
What it does **not** do, and this is the material limitation: it **re-runs nothing**, so it
proves the *environment* is reconstructible and never that a reconstructed tree reproduces
a measured number (Q02-F12); the Manifest is pinned by hash and the hashed copy came from a
dirty tree; and the in-repo half carries **tautological assertions** —
`test/rebuild/release_matrix.jl` carried `@test agree >= 0`, `@test checked >= 0` and
`@test true`, so with live SDPX ≠ the recorded `c4b109a` the agreement assertion still
passed (**F-I03-7** — raised by I03, confirmed and fixed by the parent in `2ef68fa`, where
the skip became visible as `42 Pass / 1 Broken` instead of `44/44`). Plus **R4**: the record
pins `c4b109a`.

## 9. Limitations carried forward — and what this verdict is not

1. **This document is not a release approval.** Completing I03 does not approve any
   capability for production. The decision is BLOCKED, and every capability in §6 remains
   exactly as Q02's table states it: `certified 0`.
2. **R3 is satisfied by the PARENT's run, not by mine.** At the pinned release triple
   (`69c6c09` / `e3805c9` / `f087a72`, all `dirty_paths=0`): SDPX `tests passed`, 170
   testsets, Broken=7 / Pass=9392 / Total=9399; MFLA `tests passed`, Pass=4164 / Total=4164;
   BFLA `tests passed`, 10864/10864; the 28-leg matrix 28/28 `exit=0` with a `Test Summary`
   on every leg. Every one of those reproduces its baseline exactly — the retirement changed
   no measured behaviour. My own evidence is targeted (load, the R1 certificate probe, and
   nine driver legs — S01, S03, S04, S06, S07, S05 in its three provider modes, A01_default —
   all exit 0 on the identical tree content), and it is the independent half rather than the
   citable half. Any amendment to this document after that run is docs-only and does not
   touch the code the suites measured.
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
12. **Six performance numbers circulate without a raw record** — B03's `1116.4 ms`
    compilation figure (the log says `1401.8 ms`), B03's `threads_requested/executed = 4`
    (all 106 samples say `julia_threads=1`), B03's RSS samples (12/12 failed, the cited
    stderr file does not exist), `benchmark_protocol.md`'s `27.3 s / 0.00085 s`,
    `I02_WORK_PLAN.md`'s `1.30–1.37×` SoA penalty, and the 92.5× SOC-roundtrip kernel
    ratio. §7 lists them; none may be cited as a measurement (F-I03-8).
13. **`support_matrix.md` attributes both B03 numeric-test rows to both capabilities** —
    `MFLA-THREAD-TIERS` and `MFLA-MPFR-ALLOCATION` each carry
    `bfa_tiers_16_and_64 = unsupported` *and* `mfr_native_allocation_attribution = not_run`
    (lines 1374-1377 and 1400-1403), so each row's non-passing list contains the other
    row's item (F-I03-9).
14. **The capability table cannot express `prototype`.** `EVIDENCE_STATE_VOCAB` has three
    members and `prototype` is not one, so ADR-004's explicitly-proposed sparse provider is
    absent from the table rather than labelled in it (F-I03-10). A release note must say
    "prototype" in prose; the table cannot.

## 10. Findings raised by I03

| id | severity | finding | evidence |
| --- | --- | --- | --- |
| **F-I03-1** | high | **§2's "orphaned, delete-or-wire" instruction for `sparse_la.jl`'s `GenericSparseCholeskyFactor` is under-evidenced and deleting on it would be unsafe.** A type-name grep measures the type, not the capability: `supports_sparse_generic`/`supports_sparse_execution` are **live predicates** consumed by `src/pipeline/plan.jl:291` and `_use_sparse_schur_sdp` (`sparse_la.jl:35-40`), and this family is the only implementation behind them; `plan.jl:508-520` records the descriptor `:generic_sparse_cholesky` while explicitly not instantiating it. Deleting the family while the predicates stay true converts dormant code into a latent dispatch failure. Disposition: keep, and hand forward a proposal that changes predicate + routing + family together with a before/after | `sparse_la.jl:22-28, 54-58, 1402-1504`; `plan.jl:291, 508-520`; zero callers for `sparse_factor`/`sparse_factor_solve`/`freeze_schur_pattern`/`GenericSparseCholeskyBackend(...)` repo-wide |
| **F-I03-2** | medium | **Two release documents carry a false mechanism for R1 and S02.** "src/SDPX.jl does not include src/certification/" is false (`src/SDPX.jl:195`); "src/SDPX.jl does not include src/solver/loop.jl" is false (`:194`). Conclusions survive for different reasons (inert module; inverted in-tree test) | `support_matrix.md/json`, `capability_table.tsv` (`SDPX-CERT-ORIGINAL-COORDS.blocked_by`), `release_checklist.md:196-197`, `S02/report.json` acceptance[2], `test/rebuild/S02.jl:641-648` |
| **F-I03-3** | medium | **`_sdpx_direction_trace` is called unconditionally on the per-iteration hot path** (`product_cone_hsd.jl:4193`), so every HSD step performs an `ENV` lookup and a `parse(Int, …)` before returning `nothing`. It is debug-only code with a hot-path cost. Not fixed at the freeze point (it is a behaviour-neutral but real change); retirement is to hoist the env check into a `const` read at load time or to delete the documented instrument | `product_cone_hsd.jl:4188-4194`, `:3995-4000`; survey call-site map |
| **F-I03-4** | low | **`iteration_predictor` is a write-only solver-state field** (`product_cone_hsd.jl:280`, assigned `:504`, validated in `public/settings.jl:396-398`) with no read site in `src/`; its only effect is via `factor_pair_admission.jl`. A public knob that appears to steer the HSD predictor but does not | `grep -rn 'iteration_predictor' src/`; the survey's Part 2 |
| **F-I03-5** | — | **WITHDRAWN by I03 after re-measurement.** An earlier reading of mine held that `release_checklist.md:167`'s `default_path = false` claim referred to a symbol that does not exist. It does exist — in **MFLA**, not SDPX: `MultiFloatLinearAlgebra.jl/src/planning/gemm_plan.jl:110` declares `default_path::Bool`, both construction sites (`:306`, `:426`) pass it `false`, and `grep -rn 'default_path *= *true'` over both trees finds nothing. The checklist row is verifiable; my first grep was scoped to `SDPX.jl/src` and I recorded the wrong conclusion. Withdrawn, and left visible rather than deleted | `gemm_plan.jl:110, 306, 426`; no `default_path = true` anywhere |
| **F-I03-7** | high | **`test/rebuild/release_matrix.jl` contained three assertions that could not fail on the property they named** — `@test agree >= 0`, `@test checked >= 0`, `@test true`, the last standing in for a capability-table check the same run records as `not_run` (a SKIP counted as a PASS, the failure the packet forbids by name). I03 raised it and did not apply a fix (Q02's committed instrument, mid-flight for step (b)); **the parent confirmed each and fixed it in `2ef68fa`**: the agreement count is now recorded with no assertion and a comment saying why equality would be wrong in the other direction (the record is a historical fact; a live tree ahead of it is legitimate — so my own suggested equality assertion was correctly rejected), `checked` is asserted `== length(FIRST_PARTY)` so a missing Manifest now fails, and the never-run check is a visible `@test_skip`. Measured before/after by the parent: `44/44` → `42 Pass, 1 Broken, 43 Total` — the Broken column did not exist before, which is exactly the invisibility. This changes no product code and no suite-wired test | `SDPX.jl/test/rebuild/release_matrix.jl` (post-fix), commit `2ef68fa` |
| **F-I03-8** | medium | **Performance claims with no raw record, and two that contradict theirs**: B03's `1116.4 ms` compilation (log: `1401.8 ms`), B03's `threads_requested/executed = 4` (106/106 samples `julia_threads=1`; the 1/2/4 arms are three `-t1` processes), B03's RSS samples (12/12 `rss_failed`; the cited `B03_rss_child_stderr.log` does not exist and failing children's stderr is discarded, so the cause is undetermined), `benchmark_protocol.md:23`'s `27.3 s / 0.00085 s`, `I02_WORK_PLAN.md:896`'s `1.30–1.37×` (logged `1.4487`) | `B03/report.json`, `B03/B03_driver_WIRED_static.log:24`, `B03/B03_perf_samples.txt`, `B03/B03_driver_rss.log`, `B03/B03_driver_perf.log:34-36`, `M02/M02_driver.log:289` |
| **F-I03-9** | low | **`support_matrix.md` duplicates both B03 numeric-test rows under both `MFLA-THREAD-TIERS` and `MFLA-MPFR-ALLOCATION`**, so each capability's non-passing list contains the other's item; the two `capability_table.tsv` `non_passing` strings are identical for that reason | `support_matrix.md:1374-1377, 1400-1403`; `capability_table.tsv` |
| **F-I03-10** | medium | **`prototype` is not expressible in the capability vocabulary.** `EVIDENCE_STATE_VOCAB = {verified, partially_verified, not_verified}` (`scripts/rebuild/gen_support_matrix.py:62`), and `grep -c prototype` over `support_matrix.json` and `capability_table.tsv` = 0 — so ADR-004's explicitly-proposed sparse provider (`Status: PROPOSED … Not accepted`) is *absent* from the table rather than labelled "prototype" in it. Also `unsupported` (a test status per ADR-003 §3) and the table's three states are two vocabularies with no crosswalk, which is why `release_checklist.md` and the table can say different words about thread tiers and the QDLDL raw entry point | `gen_support_matrix.py:62`; `ADR-004-sparse-provider.md:3, 133-140`; `release_checklist.md:158-163` |
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

---

# PARENT ADDENDUM (appended after I03 closed — I03's findings above are unchanged)

This addendum records two things the parent measured **after** the verdict above was
written. It does not revise any of I03's markings; it fills in the two rows that were
pending on the parent's own numbers.

## R3 — satisfied, and the numbers it was satisfied on

I03 marked R3 `satisfied` on the parent's pinned run. The parent's own logs:

| check | revision | result |
| --- | --- | --- |
| SDPX `Pkg.test()` | `69c6c09` | `tests passed`; 170 testsets; `failcols=0`; **Broken=7 / Pass=9392 / Total=9399** — the inherited baseline exactly |
| MFLA `Pkg.test()` | `e3805c9` | `tests passed`; outer **1875/1875**; aggregate **4164/4164**; `failcols=0` |
| BFLA `Pkg.test()` | `f087a72` | `tests passed`; **10864/10864**; `failcols=0` |
| 28-leg driver matrix | release pin | **`legs_run=28  legs_failed=0  failed_legs= none`**; `MATRIX_EXIT=0`; a `Test Summary` on every leg |

Logs: `rebuild-reports/PARENT_VERIFICATION/release_revision/`. All three pinned worktrees were
verified `dirty_paths=0` after the runs.

**Why these transfer to the shipped revision `16ef605`.** The runs were taken at a pin of
`69c6c09`. Between that pin and `16ef605` the only differing files are
`docs/rebuild/final_verdict.md`, `test/rebuild/release_matrix.jl` and the I03 retirement
commits already present at the pin — `git diff --stat 69c6c09..16ef605 -- src/` is the
justification for the suites, and `release_matrix.jl` occurs **0 times** in
`run_driver_matrix.sh`, so no matrix leg can observe it. **The retirement changed no measured
behaviour**: the SDPX suite reproduces the baseline to the count after deleting a dead
carrier, four null debug blocks and seven false comments.

## R4 — retired: the record now names the release revision, and the check passes

I03 marked R4 `not_satisfied` because `docs/rebuild/RELEASE_REVISIONS.txt` pinned `c4b109a`,
a pre-retirement revision. That is now fixed rather than recorded:

    pin_revisions_env.sh 16ef605 e3805c9 f087a72   -> all three verified, dirty_paths=0
    check_reconstruction.py --record <release record> --workspace <ws>
      -> RESULT: PASS — the release environment is reconstructible from the record
         60 recorded / 60 reconstructed entries, dependency set identical,
         all three Manifest sha256 match, 27 sha-less entries all matched stdlibs
         RECON EXIT=0

**R4 is therefore satisfied for the release triple.** The record's own caveat is unchanged and
still true: `Manifest.toml` is gitignored and cannot be pinned by a commit, so the check
verifies a sha256 *against the record*; if someone edited both the Manifest and the recorded
hash it would pass. Provenance here is not mechanical, and that is stated in the record.

## One instrument defect found by I03 and fixed by the parent

I03 found three assertions in `test/rebuild/release_matrix.jl` that cannot fail —
`@test agree >= 0`, `@test checked >= 0`, and `@test true` standing in for the
capability-table check that the same run records as `not_run`. The third counted a check that
never executed as a passing assertion. Fixed in `2ef68fa`; measured before and after, same
tier, same host:

    before   Q02 release matrix (fast) |  44        44   0.4s
    after    Q02 release matrix (fast) |  42    1   43   0.4s

The `Broken` column did not exist before, which is why the never-run check was invisible.
Q02's report citation was corrected from `44 44` to `42 1 43` with a limitation recording what
it had said. This is the same class as `public_sign_patches == 0`, and it was found by asking
what could make the value differ — not by anything failing.
