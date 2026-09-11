# Independent audit (V01) — SDPX infrastructure rebuild

**Reviewer role.** V01 is the independent adversarial reviewer. This document was
written by a worker that did not write any of the code under review, has no stake in
it, and whose only useful output is a check someone else can repeat. The implementer
does not approve its own correctness; nor does this document approve its own. Every
verdict below names the command that produced it and the raw result, and every item
that was not executed is `not_run` — never `0`, never `pass`.

**Revision under review (fixed, all three trees clean and committed):**

| repo | HEAD | `git status --porcelain` at audit start |
| --- | --- | --- |
| SDPX.jl | `c4b109a` | empty |
| MultiFloatLinearAlgebra.jl | `e3805c9` | empty |
| BigFloatLinearAlgebra.jl | `f087a72` | empty |

Julia 1.12.6; `Sys.CPU_THREADS = 4`, `hw.ncpu = 10`; all execution `-t1`. The host is
shared — a second Julia process from a pinned revision set (`--project=/tmp/pinfinal/SDPX.jl`)
and an unrelated worker's untracked files appeared in SDPX during this audit — so **no
timing claim is made anywhere in this document**.

**Method.** V01_PREP §1 names five error classes (attribution, revision,
self-comparison, instrument, text-vs-runtime); §2b adds two more (an explanation is a
claim; a patch inside a report is not the artifact that was tested). Targets T1–T14 plus
the two un-reviewed additions (M01 IP-2's sparse gap; the collision gates) are attacked
below. **Pure static checks and execution results are kept in separate subsections and
never merged: a grep is not a run, and a reachability claim gets a reachability check.**

The two clauses this audit is built to serve, from V01_PREP §4 and §3:

> A `not_checked` is an acceptable answer; a `confirmed` without a command is not.
> Do not report a claim as false without showing the control that settles it.

---

## T1 — `nzrange`: is the fix still needed, and is it still correct?

### Static

```
$ grep -c 'nzrange' SDPX.jl/src/hsd/product_cone_solve.jl
5
$ grep -n 'nzrange' SDPX.jl/src/hsd/product_cone_solve.jl
90:        for pointer in nzrange(A, column)
148:            for pointer in nzrange(A, j)
209:            for pointer in nzrange(A, column)
266:            for pointer in nzrange(A, column)
299:            for pointer in nzrange(A, column)
```

**5, confirmed.** Of the five, four are now guarded and one is typed:

| site | guard | evidence |
| --- | --- | --- |
| `:90` | receiver typed sparse in the signature | `:85-87` reads `function _product_hsd_owned_dense(A::SparseMatrixCSC{T,Int},) where {T<:AbstractFloat}` |
| `:148` | `:143 if A isa SparseMatrixCSC` | read at `:143`/`:148` — **the guard is at `:143`, not `:144`** |
| `:209` | `:208 if A isa SparseMatrixCSC` | with a dense `else` arm at `:212-215` |
| `:266` | `:265 if A isa SparseMatrixCSC` | dense `else` at `:270-273` |
| `:299` | `:298 if A isa SparseMatrixCSC` | dense `else` at `:302-305` |

So V01_PREP §T1's two sub-claims both hold at this revision: the count is 5, the guard is
at `:143`, and the `:86` signature is already `SparseMatrixCSC{T,Int}`. The
documented off-by-one that once put `:144` and `:460` in patch notes is **not** present in
this file today.

### Execution — the patch applies and reproduces the tree

```
$ git -C SDPX.jl cat-file -t 4cb8d60
commit
$ mkdir -p /tmp/v01_nzr_old && git -C SDPX.jl archive 4cb8d60 | tar -x -C /tmp/v01_nzr_old
$ shasum -a 256 /tmp/v01_nzr_old/src/hsd/product_cone_solve.jl
edfffdf734aa926eeed2edf3f6ec7788c2d063a8a5c798355996f0868d135e3d
$ cd /tmp/v01_nzr_old && git apply .../product_cone_solve_nzrange.patch ; echo "exit=$?"
exit=0
$ shasum -a 256 /tmp/v01_nzr_old/src/hsd/product_cone_solve.jl
0c91a39f738a2ad202a6efc80346cc8806cc4f7603d403a9439c3340d2bc6275
```

`edfffdf7…` is exactly the pristine hash I02's `nzrange_before_after.log` records for
`ce1c42f`, and `0c91a39f…` is exactly the hash the live SDPX file has today
(`shasum -a 256 SDPX.jl/src/hsd/product_cone_solve.jl`). `diff` of the patched base
against the live file is **empty**.

The corollary is a fact about the patch's own header, and it is a real (if small) defect:
the patch file says "NOT APPLIED", and applying it to the *current* HEAD fails with
`error: patch failed: src/hsd/product_cone_solve.jl:205`. That failure is **not** a
malformed patch and **not** a missing fix — it is that the fix is already in the tree.
The patch is against `4cb8d60`, and its own text names that base. A reader who checks it
at HEAD and reads `patch does not apply` as "the fix is absent" would invert the truth.

### Execution — the before/after, re-measured, not accepted

`not_run` by V01 in the *end-to-end* form, with the reason recorded rather than a number:
I02 itself could not reproduce the end-to-end leg post-cutover (its `limitations` says so,
and its probe's own header at `/tmp/i02_nzrange_target.jl:3-15` explains that the mixed
SOC+NonNeg solve returns `status=optimal` on the **unpatched** tree under both
`SDPX_CORE_ROUTE_PLANNER` settings, so the probe never reaches terminal refinement with a
dense `Ad`). The claim "a MultiFloat mixed SOC+NonNeg solve goes from throwing to
Optimal" therefore stands on a **pre-cutover** measurement that no revision in this tree
reproduces. What V01 verified at the current revision is the artifact identity chain
above plus the guard structure; what V01 did **not** verify is that the end-to-end route
reaches these functions at all. That is a gap in the *reachability* leg of the claim, and
it is I02-P3's subject.

## T2 — O(n⁴) SOC roundtrip

### Static — the complexity class and the hoistability both hold

Read at the frozen revision (`src/hsd/product_cone_hsd.jl`):

* `:826-842` `_product_hsd_soc_q_coefficient(w, n, i, j)` accumulates `ww` in
  `for k in 2:n` (`:831-833`) — **O(n) per call**, as the note says.
* `:887-924` is the nest: `for i in 1:n` → `for j in 1:n` → `for k in 1:n`, with
  `_product_hsd_soc_q_coefficient` called **inside the k loop** (`:909`) and again in the
  `k` loop above it (`:892`). O(n) × O(n³) = **O(n⁴)**, confirmed by reading, not by
  accepting the document.

**The bit-identity argument is a structural argument, and V01 checked the structure.**
The note's claim is that hoisting the three invariants cannot change a value because the
final accumulation stays in the same `i,j,k` order. That is checkable per site:

| invariant | line | invariant in | hoisting it is bit-identical because |
| --- | --- | --- | --- |
| `aik = …(w, n, i, k)` | `:892` | `j` | it is a pure function of `i,k`; the `j` loop body's first statement does not accumulate it |
| `aik = …(w, n, i, k)` | `:909` | `j` and `k` | same expression, same call — the `k` loop accumulates `term = aik*bkj` and `product_work`, neither of which touches `aik` |
| `bkj = …(winv, n, k, j)` | `:910` | `i` | pure function of `k,j`; the `i` loop resets `product`/`product_work` per `j` and never accumulates `bkj` itself |
| `ww` inside the callee | `:831-833` | both callers' `i` and `j` | it is the same sum, over the same bounds, in the same order, recomputed each call |

So the hoist is an **exact** transformation: no reassociation, no reordering, no change of
accumulation order. That is the load-bearing half of "no numeric recertification needed",
and it holds on the source.

### Execution — `not_run`, and the reason is a missing artifact rather than a missing fact

**The 92.5× benchmark is not reproducible from this workspace.** The note
(`docs/evidence/OPEN_SOC_ROUNDTRIP_ON4.md:32-47`) says the replication "was replicated
verbatim and compared against a hoisted form" in Float64 at n=128, and quotes
`current = 2.2793195189810083e6`, `hoisted = 2.2793195189810083e6`, `equal = true`,
`129.11 ms / 1.40 ms = 92.5×`. **The script that produced it is not in any of the three
repositories** (`glob` for the obvious names returns nothing under `SDPX.jl`), so V01
cannot re-run it and does **not** report its numbers as verified. This is the same defect
class as the report-embedded patches (V01_PREP §2b.7): a measurement whose instrument was
not kept is a claim, not an artifact.

Consistently with the note's own scope limit, **no end-to-end figure exists** — the note
says so at `:45-47` and I03_RELEASE_DECISION §3B repeats it. V01 agrees with the
disposition the release decision proposes: this is releasable as a **documented
performance limitation** with the kernel/end-to-end distinction stated, and it must not be
"fixed" into a green assertion.

### The scaling leg is a Q01 number, and its baseline is T14's

The 281×/312.8× ratios are read from Q01's already-collected data at `5f9e5d8`, whose
source fingerprint V01 reproduced exactly (§T14). They are therefore attributable — to
that revision, not to `c4b109a`, and V01 did not re-measure them here.

## T3 — the 28-leg driver matrix at the I02 revision

### Static

```
$ grep -cE '^run ' SDPX.jl/scripts/rebuild/run_driver_matrix.sh
28
$ grep -oE '^run +[A-Za-z0-9_]+' ... | awk '{print $2}' | tr '\n' ' '
S01 S02 S03 S04 S05_none S05_mfla S05_bfla S06 A01_default A01_all A01_mfla A01_bfla
A01_qdldl P01_none P01_mfla P01_bfla Q01_rules M01 B01_sandbox B01_wired M02 M03 P02
B02 B03 B04 P03 S07
```

**28 legs, confirmed** — V01_PREP §3b's correction (28, not 20) holds. Both recorded
invocation traps are still handled, read from the code:

1. **The A01 default leg runs under SDPX's own project.** `:94` is
   `run A01_default "$SDPX" "$SDPX" test/rebuild/A01.jl` — the third positional is the
   *project*, and it is `$SDPX`, not `$ENV`. `:95-98` use `$ENV`. So the leg that asserts
   the providers are ABSENT and the legs that require them present cannot collide.
2. **`env` no longer swallows `--provider`.** The helper at `:59` takes `envs` and `args`
   as separate positionals and executes
   `env $envs julia --project="$proj" -t1 "$script" $args`, so `--provider=all` reaches
   Julia as a program argument rather than as an environment assignment.

**Driver presence: 28/28.** Checked by hand against the workdir each leg names
(`test/rebuild/*.jl` in SDPX/MFLA/BFLA and `test/provider_contracts/sparse_contract.jl`);
`MISSING: 0`. **And the ledger's M03 caveat is now stale in the safe direction**:
`git -C MultiFloatLinearAlgebra.jl ls-files test/rebuild/M03.jl` prints the path, so M03 is
committed and a *pinned* matrix run would find it. The ledger was written when it was
untracked.

### Execution

Command (run once, sequentially, against the live committed trees):

```
$ bash SDPX.jl/scripts/rebuild/run_driver_matrix.sh \
      /Users/xuyongjun/Desktop/project/SDPX/rebuild-reports/V01/logs/driver_matrix
```

Header of the produced summary, which is where the revision identity comes from:

```
workspace   /Users/xuyongjun/Desktop/project/SDPX
env         /Users/xuyongjun/Desktop/project/SDPX/rebuild-env
outdir      .../rebuild-reports/V01/logs/driver_matrix
SDPX.jl                      c4b109a  dirty_paths=6
MultiFloatLinearAlgebra.jl   e3805c9  dirty_paths=0
BigFloatLinearAlgebra.jl     f087a72  dirty_paths=0
julia       julia version 1.12.6
```

**The `dirty_paths=6` is disclosed rather than smoothed over.** Five of the six are
another worker's untracked files (`docs/rebuild/release_checklist.md`,
`docs/rebuild/support_matrix.json`, `scripts/rebuild/{check_reconstruction,extract_capability_evidence,gen_support_matrix}.py`);
the sixth is this audit's own `docs/rebuild/independent_audit.md`. **None is under `src/`,
`ext/` or `test/`**, so the matrix measures the committed source. The two provider trees
are clean, which matters because `rebuild-env` resolves them by dev path (F6).

**Per-leg results: 27 legs `exit=0 failcols=0 fail_lines=0`; `B03` `exit=1
failcols=0 fail_lines=1`.** This run was produced by the **pre-fix**
`run_driver_matrix.sh` (`scripts/rebuild/run_driver_matrix.sh` as committed at
SDPX `c4b109a`; the file now carries the parent's uncommitted fix). **No leg reported
`SKIP`.**

> **CORRECTION (parent, same session — accepted after re-derivation).** An earlier
> revision of this section read the missing `B03_WIRED_INJECT_TEST` as selecting
> UNWIRED. That inference was wrong and the premise it rested on was sound, so the
> error was mine, not the artifact's. `rebuild-reports/V01/logs/driver_matrix/B03.log`
> is 1457 bytes, contains **zero** `Test Summary:` lines, and its first line is
> `ERROR: LoadError: WIRED mode needs the Test names injected into the package`.
> The ordering in the driver is `detect_mode()` (`B03.jl:356`) and only then
> `load_core`, whose refusal is at `:217`. So the detected mode was **WIRED** and the
> driver refused to proceed: **nothing ran.** A leg exercising the local battery
> module would have emitted a battery summary; this one emitted a stack trace.
> A leg that does not run is not a pass, and it is not an UNWIRED pass either.

The detected mode is WIRED because `detect_mode()` (`B03.jl:169-173`) asks whether
`plan_gemm!` and `plan_cholesky_trail!` are bound in `BigFloatLinearAlgebra`, and at the
frozen revision they are:

```
$ grep -rn 'function plan_gemm!\|function plan_cholesky_trail!' BFLA/src/kernels/*.jl
src/kernels/native_level3.jl:157:      function plan_gemm!(
src/kernels/native_triangular.jl:462:  function plan_cholesky_trail!(
$ grep -n 'native_level3\|native_triangular' BFLA/src/BigFloatLinearAlgebra.jl
91:include(joinpath("kernels", "native_level3.jl"))
92:include(joinpath("kernels", "native_triangular.jl"))
```

**Two distinct defects fall out of that, and they are not the same defect.**

1. **The matrix leg was misconfigured (harness defect, finding V01-F5).** The leg needed
   `B03_WIRED_INJECT_TEST=1` and did not pass it. The correct invocation runs to
   completion: D01 of `rebuild-reports/B03/parent_verify3/FROZEN_WIRED_*.log` and V01's
   own second run (below) both reach `inclusion_mode=WIRED` with `fail_lines=0`. **This is
   not a B03 defect and must not be reported as one.**
2. **`B03.jl`'s own header is stale, and its UNWIRED arm is now unreachable (finding
   V01-F9).** The header (`:18-26`) says the four B03 source files "are NOT in BFLA's
   include graph: this task's allowlist excludes the package entry point, so the real tree
   cannot load them and `Pkg.test()` cannot exercise them". That was true before I02. At
   `f087a72` it is false, and the mode is **detected**, not selectable: no flag makes the
   driver run UNWIRED on a wired tree. So the two-mode comparison B03's report rests on is
   no longer reproducible in this tree without editing BFLA's entry point back — and a
   reader who trusts the header would conclude the matrix exercised UNWIRED.

**`MATRIX_EXIT=0` in my summary is not a verdict and is not cited as one (finding
V01-F6).** The pre-fix script ended in `echo`, so its exit status was `echo`'s:

```
$ git -C SDPX.jl show HEAD:scripts/rebuild/run_driver_matrix.sh | tail -3
echo
echo "done. logs in $OUT"
$ git -C SDPX.jl show HEAD:scripts/rebuild/run_driver_matrix.sh | grep -c '^exit'
0
```

`echo` returns 0, so `MATRIX_EXIT=0` was printed in the same summary that reported
`B03 exit=1`. The correct pre-fix reading of that summary is **27 legs exit=0, 1 leg
exit=1, and no aggregate verdict existed**. The parent has since added aggregation
(`MATRIX_LEGS_RUN`/`MATRIX_LEGS_FAILED`, `legs_run=/legs_failed=/failed_legs=`, `exit 1`
on any failed leg) and controlled it in both directions; V01's re-run with the fixed
script is at §T3b.

**One further qualification, because a leg that runs is not a leg that passed:**
`failcols` and `fail_lines` are greps over the log, so a leg that printed **no Test
Summary at all** would still show `exit=0 failcols=0 fail_lines=0`. Every leg's log is
kept under `rebuild-reports/V01/logs/driver_matrix/`, and the testset counts were
re-derived from those logs mechanically (`rebuild-reports/V01/logs/matrix_testset_counts.txt`);
a leg with no summary line is called out there.

### T3b — the post-fix re-run, by the same reviewer, at the same revisions

The parent fixed both harness defects mid-audit (`run_driver_matrix.sh`: `run()` now
aggregates `rc`/`failcols`/`fail_lines` into `MATRIX_LEGS_RUN`/`MATRIX_LEGS_FAILED`, prints
`legs_run= legs_failed= failed_legs=`, and exits 1 on any failed leg; the B03 leg passes
`B03_WIRED_INJECT_TEST=1`). Because the fix is uncommitted, the two runs are attributable
to **different script revisions** and are kept apart.

```
$ bash SDPX.jl/scripts/rebuild/run_driver_matrix.sh rebuild-reports/V01/logs/driver_matrix_fixed
…
legs_run=28  legs_failed=0  failed_legs= none
MATRIX_EXIT=0
$ python3 /tmp/v01_matrix_counts.py rebuild-reports/V01/logs/driver_matrix_fixed
legs=28  legs with a Test Summary=28  legs with NONE=0 []
```

**28/28 legs now exit 0 with zero failure columns, and all 28 have a Test Summary** — the
two runs disagree on exactly the leg the fix targets (B03: `NONE` → `68 pass / 2 sets`,
`inclusion_mode=WIRED`) and on nothing else. Per-leg pass counts are identical between the
two runs for every other leg.

**Is the new aggregate verdict itself controlled?** The parent reports controlling it in
both directions (one deliberately failing driver → `legs_run=1 legs_failed=1 … MATRIX_EXIT=1`,
exit 1; a passing driver → `legs_run=1 legs_failed=0 MATRIX_EXIT=0`, exit 0). V01 read the
diff and confirms the mechanism — the counter is incremented on every `run()` call and the
script `exit 1`s when it is non-zero — but **did not re-run those two controls**; that is
the parent's measurement, cited as such. The failure predicate is strictly wider than the
old one (`rc != 0 || failcols != 0 || fail_lines != 0`), which is the right direction.

## T4 — was any test weakened to reach green?

This is the target the parent weighted most, and the honest summary is: **no removal was
found that turned a failing run green — but the strongest available instrument, the
packet's own run history, is stale, and one class of driver is outside version control
entirely.**

### 4a. `RUN_HISTORY.md` is not current (finding V01-F1)

```
$ python3 SDPX.jl/scripts/rebuild/record_run_history.py . -o /tmp/v01_run_history_regen.md
wrote /tmp/v01_run_history_regen.md
tasks=28 summary_rows=1666
$ shasum -a 256 /tmp/v01_run_history_regen.md rebuild-reports/RUN_HISTORY.md
1a561244…  /tmp/v01_run_history_regen.md
8db4c61e…  rebuild-reports/RUN_HISTORY.md
$ diff -q /tmp/v01_run_history_regen.md rebuild-reports/RUN_HISTORY.md
Files ... differ
```

The committed history is missing logs that exist on disk (e.g. the whole `B03` group,
`COMPOSED_B04_driver.log`, `PARENT_pkgtest_after_commit.log`) and records
`M02_driver.log (no Test Summary line …)` where the regenerated file records
`Pass=450 Total=450`. V01_PREP §T4 says to use this file as the discriminator; at this
revision it answers a different question than it claims to.

### 4b. The instrument is blind to the B03/B04 batteries (finding V01-F4)

`BigFloatLinearAlgebra.jl/test/rebuild/B03.jl:18-25` runs its battery by **reading and
evaluating `rebuild-reports/B03/B03_core.jl`** (43,527 B). That is where B03's assertions
live; the driver itself contains four `@test`-shaped strings and no test bodies. The same
pattern appears in `B04.jl`.

**`rebuild-reports/` is inside none of the three repositories.**
`git -C <repo> ls-files | grep -c rebuild-reports` is `0` for all three, and the three
toplevels are `SDPX.jl/`, `MultiFloatLinearAlgebra.jl/`, `BigFloatLinearAlgebra.jl/`.
So the file that contains B03's assertions has **no version history at all**: a git-based
"did the testset total drop?" check cannot see it, `git status` cannot see it in the
package trees, and `RUN_HISTORY.md` is the only time series that covers it. B03's totals
per phase are in fact stable across its recorded runs (119/119 static, 68/68 numeric,
46/46 budget, 12/12 alloc, 6/6 reach, 15/15 scratch, in both modes), so **nothing was
found** — but the review cannot establish the negative from git, and it says so.

### 4c. Every committed driver diff that removed an assertion, examined

`git log --numstat` over `test/rebuild/` in all three trees gives six commits with
deletions. Each was opened and the removed `@test` lines read:

| commit | removed `@test` lines | verdict |
| --- | --- | --- |
| SDPX `c788c10` (I02, S07 driver) | 8 | **replaced, not deleted.** The 8 were the *unwired* collision controls (`@test isempty(collisions)`, `@test r_bad.revoked == false`, …); 9 were added. The inverted, mode-conditional controls are the fix. Net `+9/-8`. |
| SDPX `5f9e5d8` (S06 fix) | 6 | **relocated out of a `try` block.** The removed lines sat under `capability = try …` where a throw meant they never ran; the added lines assert the same quantities in the success branch. Verified by reading the diff hunk's own context. |
| SDPX `37b57ea` (Q01) | 1 | **strengthened.** `@test occursin("prepared_solve_note", measure)` was replaced by four assertions (`phase_semantics`, `structure cache`, `prepared_update_replay_status`, `clear_structure_cache!`). |
| SDPX `da2e7a4` (A01b) | 8 | 4 were `@test_skip` moved into per-provider leg functions; 4 were `PROVIDER_STATUS` booleans replaced by the provider-gated legs. Net `+97/-8`. |
| BFLA `513894f` (P03 fix) | 2 | **the two already-known defects**: `@test gate_failure == true` (a pinned verdict) and `@test counted_as_pass == (row.state === :verified)` (a tautology). Their removal is the fix, and it is confirmed by the assertion arithmetic `387 → 365 = −15 removed, +18 added, ×2 modes`. |
| BFLA `f087a72` (I02) | 0 | additions only. |

**No `@test_broken` or `@test_skip` was added anywhere.** Swept across all 24 driver
files: the only occurrences are pre-existing and each carries a reason string
(`A01.jl:1908`, `S01.jl:763`, `S03.jl:1067`, `S05.jl:794`, `S06.jl:612,850,1221,1241,1249`,
`M01.jl:534`, `M02.jl:1324,1519`, `provider_checks.jl:346`, `sparse_contract.jl:550`).
`S06.jl:612,850` are `@test_broken integer("julia_threads") >= 2` and `@test_broken wide >= 2`
— both encode the 4-thread host limit, which is a genuine environment fact, and both were
present before this packet's fixes.

### 4d. An assertion that cannot fail is still in the tree (finding V01-F2, T8's second half)

```
$ grep -rn 'public_sign_patches' SDPX.jl/ rebuild-reports/
SDPX.jl/src/core/compiled_problem.jl:475:        public_sign_patches=0,
SDPX.jl/test/rebuild/S01.jl:474:    @test audit.public_sign_patches == 0
```

Read in place: `replay_public_signs` (`src/core/compiled_problem.jl:449-477`) computes
`owned_signs`, `stacked_signs`, `unowned_signs` and `objective_shifts` from
`compiled.problem.cone_layout.blocks` and `compiled.transforms.transforms`, and then
**returns the literal `0`** for `public_sign_patches` in the same named tuple. The
function never inspects the public layer. `test/rebuild/S01.jl:474` asserts that literal
equals zero.

This is the same defect class as `@test gate_failure == true`, which this packet deleted
from P03's gate after F10. It survives here, still cited by `S01/report.json` as evidence
for S01's third acceptance item, and it is **not** closed by I02 (`c788c10` touched
`src/SDPX.jl`, `src/hsd/product_cone_solve.jl`, `src/la/factor_lease.jl`,
`test/rebuild/S07.jl` and three scripts — not `compiled_problem.jl`).

## T5 — every `verified` acceptance item that cites only a report

This target was executed as a **delegated, independent evidence audit** over all 24
reports with acceptance entries (147 entries; **120 with `state: "verified"`**), each row
carrying the `file:line` actually read. Full table and reproducers:
`rebuild-reports/V01/logs/T5_acceptance_evidence_audit.md`. V01 re-verified the three
sharpest claims itself before citing them:

```
$ grep -rn '1\.85e-16' rebuild-reports/S01/ SDPX.jl/test/rebuild/S01.jl
rebuild-reports/S01/report.json:108      (a "note")
rebuild-reports/S01/report.json:157      (the "evidence" field)
$ grep -rn 'sparse_ratio_batched_over_loop' rebuild-reports/M02/*.log
M02_driver.log:259:        ... = 0.9929018083488254
M02_driver_rerun.log:259:    ... = 0.8013058055408505
PARENT_verify_driver.log:266: ... = 1.0339591836734694
$ grep -n 'P03 BFLA sparse adapter' rebuild-reports/P03/P03_driver.log
4:P03 BFLA sparse adapter |  365    365  16.5s
```

Three defects that matter for a release decision, each of the T5 class ("a `verified`
whose cited evidence does not say what the criterion claims"):

1. **`S01[2]` cites a number that exists nowhere and is wrong.** The entry says the
   `0.1+0.2` relative error is `1.85e-16`; the test it describes
   (`test/rebuild/S01.jl:528-596`) computes `|Fraction(0.1+0.2) - 3/10| / (3/10)`, which
   is **1.4802973661668753e-16**, and asserts only `1e-16 < err < 1e-15`
   (`S01.jl:535-536`). The measurement is real; the quoted value is not in any artifact.
2. **`M02[3]` cites a ratio that exists in no log, and the parent's own re-run
   contradicts it.** The entry says the sparse batched/loop ratio "was 0.822 — BELOW 1
   while being a loop, which is the control proving the timing ratio cannot settle
   batched-vs-loop". The three archived values are 0.9929, 0.8013 and **1.0340**. The
   *conclusion survives* — on the parent's own run the known-per-column loop measured
   **above** 1, which is an even cleaner demonstration that the ratio is not a
   discriminator — but the quoted number is not the one any artifact contains, and an
   overturn of the quoted evidence is not an overturn of the finding.
3. **`P03[1]` quotes a pre-fix count from the log it cites.** The evidence says
   `'P03 BFLA sparse adapter | 387 387'`; the cited `P03/P03_driver.log:4` says
   **`365 365`**. `P03[2]` in the same file says 365/365 correctly, and P03's own finding
   (`report.json:330`) explains the 387 → 365 move. So this is a stale quote inside a
   correct report.

Further classes found and worth carrying into I03's record: **`B01[2]`** asserts
cross-process determinism with five tokens (`ADMITTED_NEAREST`, `DOT2`,
`AMBIENT_PATH`, …) that appear **only in `B01/report.json`** — no artifact at all;
**`S04[8]`** quotes an artifact (`/tmp/s04/…`) that no longer contains the quote;
**`B04[0][2][5][7]`** quote per-testset counts (63/63, 23/23, 101/101, 26/26) that no
archived log contains, because that driver prints a single aggregate row;
**`A01b[8]`** has `"evidence": "This file."` — circular; **`B03[2]`'s** drift-control
ratios are the reciprocals of what `B03_perf_samples.txt` recomputes (the control came
back 1.0–2.8 % **slower**, not faster); and four entries (`A01b[6]`, `B01[9]`, `M01[5]`,
`S02[6]`) quote `git status` output that is no longer reproducible now that the trees are
committed (the claims survive; the quoted bytes do not).

**On `REPORT_SCHEMA_AUDIT.md`'s premise:** it is stale. The seven reports it audited were
rewritten at 08:31, after the 06:36 audit; `validate_reports.py .` now reports
`reports=25 errors=2` — `A01 numeric_tests[9].status='partial'` (a report the audit never
listed) and `S02 commands[3].result='pass_with_expected_dirty_tree_error'`. **No entry says
`verified` while resting on an undisclosed `not_run`/`unsupported` row**; the residual risk
is that `S03 commands[4]` and `S02 commands[3]` are exit-1 `Pkg.test()` rows with
`note=null`.

**No defect was found** in `B02`, `D01`, `I02`, `P02`, `S05`, `S06` or `S07`: every
`verified` entry there is backed by an existing path and a measurement the audit read
inside it.

## T6 — the two providers' wiring — is it genuinely additive?

### 6a. The name-surface claim, re-derived at the current revisions

The claim under test (WORKER_BRIEF §4): BFLA's exported set is unchanged (115), 0 real name
removals, 78 additions; MFLA wired with the `const MultiFloat` collision resolved by an
`isdefined` guard; both suites unchanged. I02's own gate ran this comparison at
`513894f`+wiring and reported `NAMES ADDED 120, REMOVED 0, SIGNATURES ADDED 20 on 19
names, REPLACED 0, SHAPE-CHANGED 1 (declared), UNINSPECTABLE 0`.

V01's independent re-derivation of the **collision mechanism** is in §B below (the same
instrument, on a planted mutant, in `/tmp`). Two things V01 did verify directly at the
current revisions:

```
$ git -C BigFloatLinearAlgebra.jl show --stat f087a72 | tail -3
 src/BigFloatLinearAlgebra.jl | …
$ grep -n 'include(joinpath("caches", "rectangular_qr.jl"))' BFLA/src/BigFloatLinearAlgebra.jl
104:include(joinpath("caches", "rectangular_qr.jl"))
$ ls BFLA/src/caches/
cholesky.jl  common.jl  ldlt.jl  lu.jl  rectangular_qr.jl
```

and the three B02 files are byte-identical to the material the B02 patch produces (§T12b).
So the BFLA side of "the wiring is additive" holds at `f087a72` on the file-content axis.

### 6b. `src/contracts/workspace.jl` in BOTH inclusion modes — run, both ways

This file is the third recorded instance of the "exercised in one mode only" defect, and
WORKER_BRIEF §4 requires both configurations. Its guard is at
`MultiFloatLinearAlgebra.jl/src/contracts/workspace.jl:39-41`:

```julia
import MultiFloatLinearAlgebra
import MultiFloats
if !isdefined(@__MODULE__, :MultiFloat)
    const MultiFloat = MultiFloats.MultiFloat
end
```

`@__MODULE__` resolves to the module the file is being included **into**, which is exactly
what the fix requires, and mode 2 is the case an unguarded `const` breaks ("cannot declare
MultiFloatLinearAlgebra.MultiFloat constant; it was already declared as an import" — the
error quoted in the file's own comment at `:28-34`).

```
$ julia --project=MultiFloatLinearAlgebra.jl -t1 /tmp/v01_t6_workspace_modes.jl
=== mode 2: the package's own include graph ===
MODE2_copy_operator_snapshot=true   MODE2_BlockGrammar=true
MODE2_MultiFloat_bound=true         MODE2_MultiFloat_is_import=true
MODE2_BlockGrammar_constructed=true MODE2_snapshot_type=OperatorSnapshot
=== mode 1: workspace.jl included standalone into a fresh module ===
MODE1_included=true                 MODE1_copy_operator_snapshot=true
MODE1_BlockGrammar=true             MODE1_MultiFloat_bound=true
MODE1_BlockGrammar_constructed=true
MODE1_is_MODE2_module=false
```

**Both modes work, and the file is live in the package** (`include("contracts/workspace.jl")`
at `src/MultiFloatLinearAlgebra.jl:81`). Log:
`rebuild-reports/V01/logs/t6_workspace_both_modes.log`. Target **confirmed**, including the
half the brief says was never tested.

**One stale comment, not a behaviour defect (finding V01-F10):** `workspace.jl`'s own
header still says *"Nothing here is `include`d by the package bootstrap yet."* That was
true when written and is false at `e3805c9`. Same class as I02-F7 and V01-F9: a comment
that records the pre-wiring state and is now read as current.

## T7 — ADR-003 §5.1 — is it satisfied yet?

The criterion: **every public route terminates with a valid original-coordinate
certificate.** I03_RELEASE_DECISION.md records it as **STILL UNMET** on the ground that
four `ResultCertificate{T}` construction sites remain and `src/public/` references
`SDPXCertification` nowhere. Each part of that was re-measured.

### Static

```
$ grep -rn 'return ResultCertificate{T}(' SDPX.jl/src/
SDPX.jl/src/hsd/native_hsd_public.jl:2403:    return ResultCertificate{T}(
SDPX.jl/src/hsd/native_hsd_public.jl:2462:    return ResultCertificate{T}(
SDPX.jl/src/hsd/native_hsd_public.jl:2487:    return ResultCertificate{T}(
SDPX.jl/src/public/optimize.jl:341:    return ResultCertificate{T}(
```

**Four, and I03's line numbers are exactly right** — not stale. The other two hits for
the string are the type definition (`src/public/result.jl:24`) and the `Result` field
(`:103`), so the count is not inflated.

```
$ grep -rn 'SDPXCertification\.' SDPX.jl/src/ SDPX.jl/test/ | grep -v certification/original.jl
(no output)
```

`SDPXCertification` is **defined and loaded but never referenced** anywhere in `src/` or
`test/` outside its own file. `src/SDPX.jl:195` includes `certification/original.jl`, so
the module exists at runtime and is inert — the I01 "loaded but not wired" pattern.

**But the criterion as written is not the same claim as "the module is used", and the
difference matters.** At `src/hsd/native_hsd_public.jl:2547-2586`, the one function that
builds every public `Result` (`_public_result_from_native_hsd`) does this:

```julia
certificate_summary = if core.status === Optimal
    _public_original_certificate(model, program, primal, constraint_dual, dual_slack,
                                 primal_objective, dual_objective, settings, Optimal)
elseif core.status === PrimalInfeasible
    _native_hsd_primal_infeasible_certificate(...)      # :2560
elseif core.status === DualInfeasible
    _native_hsd_dual_infeasible_certificate(...)        # :2568
else
    _native_hsd_unavailable_certificate(T, model, settings, core.reason)  # :2575
end
…
if core.status in (Optimal, PrimalInfeasible, DualInfeasible) && !certificate_summary.valid
    result_status = NumericalFailure
    termination_reason = :original_coordinate_certificate_failed
    termination_stage = :certification
end
```

and `_public_original_certificate` (`src/public/optimize.jl:223-357`) builds the residuals
**in original coordinates** — per-block primal and dual-slack cone residuals, `A*x - b`,
original stationarity `c - A'y - s`, a relative gap with a data scale, scaled against
`primal_limit`/`dual_limit`/`gap_limit` derived from the caller's tolerances.

So the code's own structure says: every public route constructs a certificate, and an
invalid certificate **downgrades the public status out of `Optimal`**. That is the
functional content of "terminates with a valid original-coordinate certificate", and it is
enforced in one place that all four public statuses flow through. The `SDPXCertification`
module is a second, unused implementation of L3 — it is *not* what the public route uses.

**Verdict — and it differs from the ledger's, on the mechanism rather than the outcome.**

The ledger and I03_RELEASE_DECISION.md call R1 unmet because `src/public/` never references
`SDPXCertification`. **That measurement is right and V01 reproduces it.** But it is not the
same claim as the criterion, and the execution below shows why the difference decides
whether this is a *functional* gap or a *structural* one. What the code actually does at
`c4b109a` is: build an original-coordinate certificate on every public route, and refuse
`Optimal` when it is not valid. The unused `SDPXCertification` module is a **second,
independent L3 implementation that nothing calls** — so the packet has two certification
paths and the public one does not consult the one the ADR names.

### Execution — what a caller actually gets

```
$ julia --project=$REBUILD_ENV -t1 /tmp/v01_t7_certificate.jl    # logs/t7_certificate_route.log
CASE lp_float64   status=Optimal  reason=verified_accepted_step
   cert_available=true cert_valid=true cert_method=original_coordinates cert_reason=valid
   primal_res=0.0 dual_res=7.1e-15 rel_gap=1.9e-9  limits=(1e-8,1e-8,1e-8)
   CONTROL optimal_implies_valid_certificate=true
CASE soc_float64  status=Optimal  cert_valid=true cert_method=original_coordinates
CASE sdp_float64  status=Optimal  cert_valid=true cert_method=original_coordinates
CASE infeasible_lp status=PrimalInfeasible
   cert_method=original_coordinate_primal_infeasibility_ray cert_valid=true
CASE lp_float64_tight_gap status=Optimal cert_valid=true cert_gap_limit=1e-14
SDPXCertification_defined_in_SDPX=true
```

Four routes (LP, SOC, SDP, an infeasibility ray) each terminate with
`certificate.available = true`, `valid = true`, `method = :original_coordinates`, and a
`termination_stage` of `original_coordinate_certification`. The control
`!(status === :optimal && !cert_valid)` holds in all four.

**What this does not establish, stated because the target deserves it:**

* **The downgrade branch was not exercised.** `src/hsd/native_hsd_public.jl:2581-2586`
  turns an invalid certificate into `NumericalFailure` /
  `:original_coordinate_certificate_failed`; V01 read it at the source but did not produce
  a case that reaches it. Case D tried (a `gap` limit of `1e-14`) and the solve simply met
  it — the certificate came back valid — so the control shows the limit is *carried*, not
  that it *bites*. **The negative direction of this gate is `not_run`.**
* The LP/SOC/SDP cases are tiny and well conditioned. No claim is made about a hard
  instance.
* `optimize!` is the only public entry point in this probe; `src/moi_wrapper.jl` and
  `src/frontend/` were **not** driven. The static reading says they lower to the same
  `_optimize_impl` (`src/public/optimize.jl:375-400`), but that is a reading.

**So:** R1's *mechanism* exists and its *named module* is unused. Whether that satisfies
ADR-003 §5.1 depends on whether the criterion names the module or the property — and the
ADR's text (`:65-67`) names the property: *"Every public route must terminate with a valid
original-coordinate certificate. A solver status alone is never acceptance."* On that text,
R1 is **met in the mechanism and unproven in the negative direction**, and the correct
action is not "unblock" but "measure the downgrade with a case that should fail it".
The ledger's framing — *"the blocker is a missing instrument"* — is right about the
instrument and wrong to attribute it to the module reference. **V01 does not recommend
unblocking R1 on this evidence.**

## T8 — S01-P2

### 8a. There is no patch to delete — verified, not inherited

```
$ for p in '\.= *-' '= *-x' '= *-1 *\.\*' '\*= *-1' 'sign_flip' 'flip_sign' 'negate' 'public_sign_patch'; do
      n=$(grep -rnE "$p" SDPX.jl/src/public/ SDPX.jl/src/moi_wrapper.jl SDPX.jl/src/frontend/ | wc -l)
      echo "pattern [$p] matches=$n"; done
pattern [\.= *-] matches=0
pattern [= *-x] matches=0
pattern [= *-1 *\.\*] matches=0
pattern [\*= *-1] matches=0
pattern [sign_flip] matches=0
pattern [flip_sign] matches=0
pattern [negate] matches=0
pattern [public_sign_patch] matches=0
```

The parent's measurement reproduces exactly: **eight patterns, eight zeros**, over
`src/public/` (4 files: `optimize.jl`, `outputs.jl`, `result.jl`, `settings.jl`),
`src/moi_wrapper.jl` and `src/frontend/` (2 files). **The prescribed before/after MOI
solve of "a deletion" therefore compares a state to itself and cannot unblock.** T8's
first half is mis-posed, as §3b says; the second half (below) is where the defect is.

### 8b. `public_sign_patches=0` is a constant, unaddressed

See §4d. Finding **V01-F2**. Directly relevant to T8 because `S01/report.json` cites that
field as one leg of its evidence and `test/rebuild/S01.jl:474` asserts it against itself.

### 8c. Did I02 delete a sign patch anyway?

`git -C SDPX.jl show --stat c788c10` is `src/SDPX.jl`, `src/hsd/product_cone_solve.jl`,
`src/la/factor_lease.jl`, `test/rebuild/S07.jl` and three new gate scripts. **No sign patch
was deleted, and nothing under `src/public/`, `src/moi_wrapper.jl` or `src/frontend/` was
touched** — which is consistent with there being nothing to delete. S01-P2 is therefore
still blocked, and blocked on an experiment that must be respecified (F21's restatement).

## T9 — QDLDL: is the corrected description the right one, and is the withdrawn claim gone?

```
$ grep -rniE 'heap history|heap_history|depends on the heap|heap churn' \
      <3 repos' src/ext/test> rebuild-reports/
```

Full hit list, each classified as an **assertion** (a defect) or a **record of what was
tested** (evidence):

| hit | text | class |
| --- | --- | --- |
| `MFLA/test/rebuild/M02.jl:424` | `# version of this comment asserted that the fatal mode "depends on the heap history". The experiment that tested it … came back NEGATIVE 4/4 … so that mechanism is WITHDRAWN.` | withdrawal itself — legitimate |
| `MFLA/test/rebuild/M02.jl:468` | `# Bounded heap churn before the call, to give the undefined behaviour a chance to show its other failure mode.` | describes the probe — legitimate |
| `MFLA/test/rebuild/M02.jl:1603` | `# The same call after a bounded heap churn in the child. If the child still raises a catchable exception, this driver did NOT reproduce the parent's signal death at this churn level…` | describes the probe — legitimate |
| `rebuild-reports/M02/report.json` (F7) | `THE MECHANISM IS NOT ESTABLISHED, and the 'depends on heap history' phrasing in the first version of this finding is WITHDRAWN` | the corrected record — legitimate |
| `rebuild-reports/M02/M02_qdldl_probe_sweep.log:2` | `# Probes the hypothesis that the signal death is a function of heap history` | purpose of the probe — legitimate |
| `rebuild-reports/{V01_PREP,PARENT_FINDINGS_BATCH5,PARENT_VERIFICATION_LOG,ORCHESTRATION}.md` | the parent's own accounts | legitimate |

**No surviving assertion of the withdrawn mechanism, including in comments.** The parent's
own claim reproduces. Two further observations the target did not ask for but which the
same rule requires:

* **The M02 child probe is correctly gated, which is not true of every driver in this
  packet.** `M02.jl:1598-1601` records and asserts
  `occursin("PROBE_CALLING_RAW_MATRIX_SOLVE", clean_probe.output)` — i.e. it asserts the
  child *ran* before asserting what it found. That is exactly the rule WORKER_BRIEF §7
  states and F1 shows B02 originally violated.
* **The number in the corrected text is arithmetically checkable and checks out**: 10
  catchable observations = 4/4 parent + 6/6 driver, and 1 fatal (n=1, the parent's
  in-process run). These are *counts of observations*, not probabilities, and the record
  does not inflate them into one.

## T10 — `refactor_numeric!` lease hazard at `src/la/factor_lease.jl:357-364`

### Static: it is fixed, and the fix is where the hazard was

`refactor_numeric!` (`:357`) now reads:

```julia
if !adm.allowed
    # ADR-002 §4 requires revocation on ANY failure, and an admission refusal
    # is a failure. … `h.request` has already changed, so a lease left bound here
    # authorises a request no factor was ever built for -- measured as
    # `is_valid(lease) == true` with `bound_request_digest != request_digest`,
    # and `commit_failure_observation` throwing because it refuses to run while
    # a lease is valid (S07-F1). Revoking here is what makes the §4 guarantee
    # hold for every exit from this function rather than all but one.
    _revoke!(h.lease, EvRefactorPreflightRejected,
             "request not admitted: $(adm.detail)")
    return (ok=false, status=StatusUnsupported, generation=h.provider_generation,
            revoked=true, provider_status=nothing, admission=adm, …)
end
```

The previously-hazardous return value is now `revoked=true` (was `revoked=false`) and the
lease is revoked before the return. **The four control values V01_PREP asks to re-measure
from one's own run are therefore the pre-fix values**, and I02's bidirectional gate is the
reason they are falsifiable: the fixed arm reports
`control_lease_still_bound == false` at `test/rebuild/S07.jl:760`, and the same driver
bytes on a tree with the fix rolled back report `135 passed, 6 failed of 141`.

### Execution — re-measured from V01's own run, and the values are the post-fix ones

```
$ grep 'control_' rebuild-reports/V01/logs/driver_matrix/S07.log
S07 MEASURE control_commit_observation_throws = false
S07 MEASURE control_lease_digest_matches_request = false
S07 MEASURE control_lease_still_bound = false
S07 MEASURE control_refactor_revoked_flag = true
$ grep -A2 'Test Summary' rebuild-reports/V01/logs/driver_matrix/S07.log
S07 mode A (explicit imports)                    |  141    141  31.4s
S07 mode B (SDPX names aliased, then included)   |  141    141   4.4s
S07 inclusion modes agree                        |    7      7   0.1s
```

| V01_PREP §T10's pre-fix value | V01's measured value at `c4b109a` |
| --- | --- |
| `control_refactor_revoked_flag = false` | **`true`** |
| `control_lease_still_bound = true` | **`false`** |
| `control_lease_digest_matches_request = false` | `false` (unchanged — the request still differs) |
| `control_commit_observation_throws = true` | **`false`** |

**The hazard is fixed**, and the fix is falsifiable in both directions: I02's rollback arm
(`rebuild-reports/I02/logs/S07_driver_rollback_unfixed_lease.log`, same driver bytes, fix
reverted) is `135 passed, 6 failed of 141`, and the parent verified that the six failures
are exactly the inverted controls. **V01 did not re-run the rollback arm** — the rollback
tree was a scratch tree I02 built in `/tmp/i02armA` — so what V01 contributes here is the
**fixed** arm measured independently at the committed revision, plus the static reading of
`src/la/factor_lease.jl:360-376` above. The before/after pair is I02's, and its bidirectionality
is the reason it is believable.

## T11 — `A01b-F1` is still live

**Reproduced from the record at the current revision, not from the report's prose:**

```
$ grep -rn 'control_square_cache_keeps_stale_success' <3 repos> rebuild-reports/
BigFloatLinearAlgebra.jl/test/rebuild/B02.jl:506:  measure("control_square_cache_keeps_stale_success", …)
BigFloatLinearAlgebra.jl/test/rebuild/B04.jl:461:  println("MEASURE control_square_cache_keeps_stale_success=$(issuccess(cq))")
rebuild-reports/B02/B02_driver_REBUILD_ENV.log:29:MEASURE control_square_cache_keeps_stale_success = true
rebuild-reports/B04/B04_driver_REAL_TREE_monolith.log:44:MEASURE control_square_cache_keeps_stale_success=true
rebuild-reports/B04/B04_driver_SCRATCH_split.log:44:MEASURE control_square_cache_keeps_stale_success=true
```

So the behaviour is **asserted in the drivers that ship today** and measured `true` in four
independent runs (B02's two-env driver, B04's monolith, split and pristine-checkout arms).
**It was not fixed by I02** — and I02's report says so deliberately, with its acceptance
item for move 12 at `partially_verified` and the reason: ADR-002 §4 places the obligation
on the consumer ("revoke the logical lease on ANY failed refactor"), so the cache-side
"fix" would remove a documented guarantee. The consumer-side audit §2.28 asks for was
`not_run` by I02 and is `not_run` by V01 as well: **no consumer of the four square caches
was checked for reading `status` without `is_valid(lease)`/`authorize`.** That is the
outstanding half of the finding, and it is the half that decides whether this is a live
hazard or a documented contract.

The matrix leg for `B02` and `B04` at this revision re-runs both drivers; the values above
are the ones the committed drivers print.

## T12 — the regenerated patches

### 12a. Structural validity

```
$ python3 SDPX.jl/scripts/rebuild/validate_patches.py SDPX.jl/docs/evidence/proposed/*.patch
ok   …/B02_wire_rectangular_rrqr.patch  hunks=1 +8/-0 files=1
ok   …/I02_M01_IP2_record_at_commit.patch  hunks=4 +13/-0 files=1
ok   …/I02_M01_IP4_sparse_solve_dense_order.patch  hunks=1 +14/-0 files=1
ok   …/I02_adapt_B02_driver.patch  hunks=2 +72/-4 files=1
ok   …/I02_adapt_S07_driver.patch  hunks=2 +26/-5 files=1
ok   …/I02_fix_refactor_numeric_lease.patch  hunks=1 +13/-1 files=1
ok   …/M02_wire_gemm_candidate.patch  hunks=1 +16/-0 files=1
ok   …/S07_wire_session_layer.patch  hunks=1 +9/-0 files=1
ok   …/product_cone_solve_nzrange.patch  hunks=4 +26/-8 files=1
9/9 patch file(s) structurally valid
```

### 12b. `git apply --check` at HEAD is the wrong test, and it fails for five of five

```
$ (in each patch's own repository) git apply --check <patch>
S07_wire_session_layer.patch        exit=1 :: error: patch failed: src/SDPX.jl:196
M02_wire_gemm_candidate.patch       exit=1 :: error: src/MultiFloatLinearAlgebra.jl: No such file or directory
B02_wire_rectangular_rrqr.patch     exit=1 :: error: src/BigFloatLinearAlgebra.jl: No such file or directory
I02_M01_IP2_record_at_commit.patch  exit=1 :: error: src/factor_caches.jl: No such file or directory
I02_adapt_B02_driver.patch          exit=1 :: error: src/B02.jl: No such file or directory
```

**This is not a defect in the patches.** Running `git apply --check` from `SDPX.jl` for a
patch that targets MFLA or BFLA reports `No such file or directory` — the exact
wrong-tree artefact `check_proposed_patches.sh` was written to prevent — and the ones that
*do* target SDPX fail because **they are already applied**. The correct test uses the
pre-image blob each patch's own `index` line records:

```
$ python3 /tmp/v01_t12_patches.py <patches>
S07_wire_session_layer.patch    src/SDPX.jl: applies; result 5d4e9546ef75 == post == LIVE
M02_wire_gemm_candidate.patch   src/MultiFloatLinearAlgebra.jl: applies; result 56414452f4b6 == post == LIVE
B02_wire_rectangular_rrqr.patch src/BigFloatLinearAlgebra.jl: applies; result 116e06b4bf01 == post != LIVE
I02_M01_IP2_record_at_commit.patch   src/factor_caches.jl: applies; result cc9868c5886a == post == LIVE
I02_fix_refactor_numeric_lease.patch src/la/factor_lease.jl: applies; result d9313084e5e1 == post == LIVE
I02_M01_IP4_sparse_solve_dense_order.patch ext/MultiFloatQDLDLExt.jl: applies; result e1b9e9c34c2a == post == LIVE
```

Each patch reconstructs its own recorded post-image, and five of six match the live file
**byte for byte**. The sixth is not a discrepancy but a **relocation**, and it was measured
rather than assumed:

```
$ git -C BigFloatLinearAlgebra.jl archive 76ae0e8 | tar -x -C /tmp/v01b02 && cd /tmp/v01b02
$ git apply --check .../B02_wire_rectangular_rrqr.patch ; echo "exit=$?"
exit=0
$ git hash-object src/BigFloatLinearAlgebra.jl
116e06b4bf0144cbe0805266f52c5f51dcc52164      # == the patch's own post-image
$ for f in src/caches/rectangular_qr.jl src/kernels/qr_panel.jl src/solves/least_squares.jl;
    do compare hash-object in /tmp/v01b02 with the live BFLA tree; done
SAME src/caches/rectangular_qr.jl
SAME src/kernels/qr_panel.jl
SAME src/solves/least_squares.jl
```

The B02 entry point differs from the patch's post-image only because I02's wiring placed
the same three `include`s in its own block (`src/BigFloatLinearAlgebra.jl:103-105`) rather
than inline at `:78`. All three included **files** are byte-identical.

### 12c. The B04 patches

```
$ mkdir -p /tmp/v01_b04 && git -C BigFloatLinearAlgebra.jl archive 5cabc00 | tar -x -C /tmp/v01_b04
$ cd /tmp/v01_b04 && wc -l src/caches.jl
1532
$ git apply --check .../B04-PATCH-1-entry-include.diff && git apply … ; echo $?
0
$ git apply --check .../B04-PATCH-2-caches-reduction.diff && git apply … ; echo $?
0
$ wc -l src/caches.jl
339
$ diff src/caches.jl /Users/xuyongjun/Desktop/project/SDPX/BigFloatLinearAlgebra.jl/src/caches.jl
(empty)
```

**Both apply at their own base `5cabc00` and reconstruct the live 339-line file exactly.**
This independently reproduces F18's composition check: the 1229-line reduction is anchored
to a byte-identical reference, not to a description.

### 12d. "A patch that applies is not a patch that is correct"

The intent checks that are checkable statically, each read at the destination:

* `S07_wire_session_layer.patch` — the ordering comment it inserts says update → replay →
  cancellation, and `src/SDPX.jl:203-207` has exactly that order.
* `I02_fix_refactor_numeric_lease.patch` — its post-image is the live file, and the live
  file's `_revoke!`-before-return is what T10 verified.
* `I02_M01_IP2_record_at_commit.patch` — 4 hunks, and the live tree has exactly the four
  call sites the patch prescribes, one per dense cache:

  ```
  $ grep -n 'record_factor_summary!' MultiFloatLinearAlgebra.jl/src/factor_caches.jl
  61:    record_factor_summary!(cache)      # MFCholeskyCache
  167:   record_factor_summary!(cache)      # MFLUCache
  284:   record_factor_summary!(cache)      # MFLDLTCache
  516:   record_factor_summary!(cache)      # MFRRQRCache
  ```

  and the function it calls does what the patch's comment claims —
  `src/contracts/summary.jl:382-386` records the summary **then** `bump_generation!(x)`,
  so the summary is taken before the bump that invalidates it. The patch's stated intent
  ("allocation-free; the summary is recorded before the bump") is exact.
* `B02_wire_rectangular_rrqr.patch` — the three files it names are present and loaded
  (T12b).
* The behaviour claims of the two driver-adaptation patches (`I02_adapt_*`) are the
  subject of T3's matrix run and F3's two-arm verification; the patches are plain `diff -u`
  with no `index` line, so their pre-image cannot be recovered from the file itself. That
  is a limitation of those two artifacts, recorded rather than hidden, and it is why the
  parent generated them by editing a real file and taking `diff -u` (WORKER_BRIEF §7).

## T13 — `B04-F2`: do NOT let this be "fixed" into a green assertion

**Verified as left alone.** `B04/report.json`'s acceptance item for the repeated-operation
control carries `partially_verified` and its own `evidence` field says why:

> … ordinary-factor isolation 24/24 (values, element objectids and `ldiv!` result
> unchanged; no aliasing with cache storage). **NOT verified: per-element identity of the
> LDLT factor matrix across a repeated `factorize!`, which is false in BOTH modes**
> (`factor_elements_kept_ldlt=false`) because `_ldlt!`'s `_swap_sym!` permutes MPFR objects
> one-for-one between slots. That is a pre-existing kernel property, unrelated to the
> split, and is recorded as B04-F2 rather than asserted away.

and the committed driver **prints** the flag rather than asserting it:

```
$ grep -n 'factor_elements_kept' BigFloatLinearAlgebra.jl/test/rebuild/B04.jl
498:                "factor_elements_kept=$elements_kept")
512:        println("MEASURE factor_elements_kept_$family=$elements_kept")
$ grep -n -B3 -A3 'factor_elements_kept_ldlt' BigFloatLinearAlgebra.jl/test/rebuild/B04.jl
(no @test on that name)
```

So there is nothing to weaken here and nothing was weakened: the property is measured,
reported as false, and written into the report as unverified. I02 left `B04/report.json`
untouched. **Target confirmed.** The physical reason is consistent with everything else I
read — a pivoting factorization that permutes storage slots cannot preserve per-element
object identity across repeats — and **no assertion anywhere claims per-element identity
for LDLT**.

## T14 — the promotion baseline was measured on a dirty worktree

**The parent's claim is confirmed about the flag and refuted about what it implies.**

Confirmed, read from the raw artifacts (not from prose):

```
$ grep -n 'worktree_dirty\|sdpx_head' rebuild-reports/Q01/measure_*.toml | head
measure_bigfloat_256_rebuild_env.toml:14:sdpx_head = "5f9e5d8d68dd5a963f28db5cabe2d99635e1aa0a"
measure_bigfloat_256_rebuild_env.toml:18:source_fingerprint_at_end = "d62abb18584239432e3355cb9bfd914df77783cb0c6f2c41c24d91be3c28ba1d"
measure_bigfloat_256_rebuild_env.toml:19:source_fingerprint_at_start = "d62abb18584239432e3355cb9bfd914df77783cb0c6f2c41c24d91be3c28ba1d"
measure_bigfloat_256_rebuild_env.toml:20:worktree_dirty = true
($same for measure_float64_ and measure_multifloat_x2_ and bigfloat_256)
$ awk 'NR>=2091 && NR<=2130' rebuild-reports/Q01/report.json
all four `source_integrity` arms: sdpx_head 5f9e5d8…, worktree_dirty true,
source_changed_during_run false, provider_revisions_changed_during_run []
```

All four arms carry `worktree_dirty = true` at `sdpx_head = 5f9e5d8`, as stated. (The
`false` values that also grep in each TOML are in a nested sub-table, not the run header;
the header is unambiguous.)

Refuted, by re-implementing `source_fingerprint()` in Python from its definition
(`benchmark/rebuild/measure.jl:94-128`: SHA-256 over `relpath\0sha256(file)\n` for every
`.jl` under `src/` and `ext/` plus `Project.toml`, sorted by path):

```
$ python3 /tmp/v01_fingerprint.py SDPX.jl
files=202
f64344519ef2db7338aaa6cc4458db7c5c8f677e0b017ff85fcc01e31780bc06
$ mkdir -p /tmp/v01_q01base && git -C SDPX.jl archive 5f9e5d8 | tar -x -C /tmp/v01_q01base
$ python3 /tmp/v01_fingerprint.py /tmp/v01_q01base
files=199
d62abb18584239432e3355cb9bfd914df77783cb0c6f2c41c24d91be3c28ba1d
```

**`d62abb18…` — the fingerprint Q01 recorded — is exactly what `5f9e5d8` alone
reproduces, over 199 files, from a clean archive of the commit.** The baseline is therefore
**reconstructible from its named revision**, which is the strongest form of the property
T14 asks about. `worktree_dirty = true` was true of the *workspace*; it was **not** true of
the measured sources. Two facts settle the reading:

1. `source_fingerprint` covers every `.jl` under `src/` and `ext/` plus `Project.toml` —
   the whole of what the measurement consumes.
2. `sdpx_head_changed_during_run = false` and
   `source_fingerprint_at_start == source_fingerprint_at_end` in all four arms, so the
   dirty content was stable across each run; and the content it hashed is the content at
   `5f9e5d8`.

What a hash cannot prove is the *identity of the dirty paths* — that is the parent's
correct general point — but here the fingerprint closes the question for every file that
can affect a number, and the remaining possibility (a dirty file that is not a source
file) cannot change a measurement.

**Does any promotion cite a baseline that `5f9e5d8` alone does not reproduce?** Not on this
evidence: the numbers Q01 promoted are attributed to `5f9e5d8` + `d62abb18…`, and that pair
is reproducible. What has changed since is stated separately and honestly — the sources
have moved on (`5f9e5d8 → c4b109a` touches 6 files under `src/`/`ext/`, 3 of them new:
`src/session/{update,replay,cancellation}.jl`), so the *baseline revision* is reproducible
while the *current revision* is not the baseline. V01 did **not** re-measure Q01's scaling
numbers at `c4b109a`; that is `not_run` here.

## A — M01 IP-2 and the sparse `factorize!` gap

### A.1 Static — the four dense sites, and the fifth method

```
$ grep -rn 'record_factor_summary!' MultiFloatLinearAlgebra.jl/src/ MultiFloatLinearAlgebra.jl/ext/
src/contracts/summary.jl:382:   function record_factor_summary!(x)          # the definition
src/factor_caches.jl:61:       record_factor_summary!(cache)              # MFCholeskyCache
src/factor_caches.jl:167:      record_factor_summary!(cache)              # MFLUCache
src/factor_caches.jl:284:      record_factor_summary!(cache)              # MFLDLTCache
src/factor_caches.jl:516:      record_factor_summary!(cache)              # MFRRQRCache
ext/rebuild/mfla_sparse_adapter.jl:664,676,701: MFLA.record_factor_summary!(a.cache)
```

`record_factor_summary!` (`:382-384`) calls `bump_generation!(x)`. **`MFSparseLDLCache.factorize!`
is `ext/MultiFloatQDLDLExt.jl:160-195` and contains no such call** — it sets
`cache.status = 0` at `:188` (success) and `cache.status = -3` at `:191` (failure). So
I02-F1's reading is **confirmed at the source**: four dense commit points are instrumented,
the sparse method is not.

### A.2 Execution — the gap is real, and it is reachable from SDPX's own provider seam

```
$ julia --project=$REBUILD_ENV -t1 /tmp/v01_ip2_sparse.jl     # rebuild-reports/V01/logs/ip2_sparse_rebuild_env.log
QDLDL_loadable=true
MFLA_sparse_ldlt_available_Float64x2=true
DIRECT_SPARSE generations=(0x0, 0x0, 0x0)  delta_first=0 delta_second=0
DIRECT_SPARSE status_after=0 issuccess=true
DENSE_LDLT    generations=(0x0, 0x1, 0x2)  delta_first=1 delta_second=1
SDPX_SEAM     generations=(0x0, 0x0, 0x0)  delta_first=0 delta_second=0
```

Three measurements, one process, same instrument for each:

* **Two same-size refactors of an `MFSparseLDLCache` advance the generation by 0** — both
  of them, with `issuccess=true` (the refactors succeeded). The defect M01-F4/P02-F1
  reported on the dense caches is **unchanged on the sparse path**, and it is not a
  failed-refactor artefact.
* The same two refactors on the **dense** `MFLDLTCache` advance it by **1 each** — so the
  instrument is sensitive and the contrast is a property of the method, not of the probe.
* **`SDPX_SEAM` is the reachability answer**, and it is the part I02's report did not
  measure: driving SDPX's own public seam —
  `SDPX.SparseQDLDLProviderCache(MF2, pattern, dsigns)` followed by
  `SDPX._qdldl_provider_factorize!` — also advances the generation by **0**.

**Reachability, read from the call graph and confirmed by that seam probe.** The chain is

```
src/factor_cache/routes/qdldl_sparse.jl:334   _qdldl_provider_factorize!(cache.provider, A)
ext/SDPXMultiFloatLinearAlgebraExt.jl:1399    SDPX._qdldl_provider_factorize!(provider, A)
                                              -> MultiFloatLinearAlgebra.factorize!(provider, A)
ext/MultiFloatQDLDLExt.jl:160                 the uninstrumented method
```

and `ext/MultiFloatQDLDLExt.jl` is a **real package extension** (`[extensions]
MultiFloatQDLDLExt = ["QDLDL", "SparseArrays"]` in MFLA's `Project.toml`), with QDLDL
present in `rebuild-env` (`Project.toml:6`). Unlike the `ext/rebuild/` adapter — which is
`include`d by a driver and is not part of the package — this path loads for any user who
has QDLDL. **So the gap is reachable from a public path**, and it is not only the
adapter's concern: adapter callers are covered (the adapter calls
`record_factor_summary!` itself at `:701` and advances by 1), while a caller who uses
`sparse_ldlt_cache` + `factorize!` directly, or SDPX's provider seam, is not.

**What V01 did not do:** run a *lease* end-to-end on the sparse path (take a lease, refactor
directly, show the lease still validates). The generation delta of 0 is the mechanism and
it is measured; the lease consequence is I02's inference and P02's driver covers it on the
adapter path.

**Disposition.** This is a live correctness gap on a release-relevant path (I03's **R8**),
and I02's own proposal I02-P2 is the right shape: one line of code, but three exits
(`:171` rejection, `:188` success, `:191` failure) and an extension-only load that needs
both-configuration evidence. It must be **fixed or explicitly recorded as accepted risk
with this reproduction** before the release decision; it is not a report-format issue.

## B — the collision gates

The two gates are `scripts/rebuild/name_surface_snapshot.jl` + `name_surface_diff.py`
(surface) and `scripts/rebuild/overwrite_warning_check.sh` (overwrite). I02 claims their
coverage is a **decomposition** and that the surface gate cannot see a same-signature
redefinition. The claim was checked in both directions on **one mutant at a time**, in a
scratch copy of BFLA under `/tmp` — nothing under `src/`, `ext/` or `test/` was touched.

### B.1 The overwrite gate is genuinely sensitive, and it can fail

```
$ bash SDPX.jl/scripts/rebuild/overwrite_warning_check.sh BigFloatLinearAlgebra <log> /tmp/v01_gatemut-env
pristine copy          rc=0 overwrite_warnings_total=0 mentioning_BigFloatLinearAlgebra=0
SAME-SIGNATURE mutant  rc=0 overwrite_warnings_total=1 mentioning_BigFloatLinearAlgebra=1
    WARNING: Method definition factor_Rdiag(…BFLARRQRCache…) in module BigFloatLinearAlgebra
             at /tmp/v01_gatemut/src/caches.jl:82 overwritten at /tmp/v01_gatemut/src/caches.jl:342.
DIFFERENT-SIGNATURE mutant  rc=0 overwrite_warnings_total=0
```

So the instrument is **not vacuous**: `0 → 1` on a planted same-signature redefinition,
built by appending a second `factor_Rdiag(cache::BFLARRQRCache)` to a scratch copy. Logs:
`rebuild-reports/V01/logs/gate_v01_overwrite_{pristine,samesig,diffsig}.log`.

**The SDPX arm of this gate is vacuous, exactly as I02 disclosed.** I02's report says
"0 warnings in both arms, and no SDPX mutant was built, so this arm has not been shown to
be able to fail". That disclosure is correct and V01 did not build an SDPX mutant either —
it is recorded here as a **limitation of the SDPX arm**, not as a pass. The BFLA mutant
demonstrates the mechanism; it does not demonstrate that any particular SDPX wiring move
was checked by a sensitive instrument.

### B.2 The surface gate really is blind to a same-signature redefinition

```
$ julia --project=/tmp/v01_gatemut-env -t1 --compiled-modules=no name_surface_snapshot.jl \
        BigFloatLinearAlgebra <pristine.json>     # names=1065 uninspectable=0
$ (append the same-signature mutant)
$ julia --project=/tmp/v01_gatemut-env -t1 --compiled-modules=no name_surface_snapshot.jl \
        BigFloatLinearAlgebra <samesig.json>      # names=1065 uninspectable=0
$ python3 name_surface_diff.py <pristine.json> <samesig.json>
NAMES ADDED: 0   NAMES REMOVED: 0   SIGNATURES ADDED: 0   SIGNATURES REMOVED: 0
SIGNATURES REPLACED: 0   SHAPE-CHANGED: 0   UNINSPECTABLE: 0
GATE: PASS (nothing removed or replaced; every pre-existing binding kept its kind,
           exportedness, fields and method set)
surface_gate_exit=0
```

**`GATE: PASS` on the mutant the overwrite gate catches with one warning.** The blindness
is confirmed empirically, not just by reading `name_surface_diff.py:111-122` (which
compares `set(b["methods"])` with `set(a["methods"])`, so an identical signature is the
same set element). This is the strongest single result about these controls in the packet,
and it is the reason a green surface gate must not be read as "nothing was redefined".

**One procedural trap found while doing it, worth recording (finding V01-F8).** The
snapshot's first run against the mutant printed
`ERROR: Method overwriting is not permitted during Module precompilation` and still wrote a
JSON — from the **cached** image, not from the edited source. The overwrite gate avoids
this by passing `--compiled-modules=no` (`overwrite_warning_check.sh:37`, and its own
comment at `:18-23` explains why). `name_surface_snapshot.jl` **does not** pass it, so a
surface snapshot taken against a warm cache measures the previous build. V01 re-ran both
snapshots with `--compiled-modules=no` to make the comparison above sound. For the gate's
normal use this is harmless (a wiring move invalidates the cache, so the source *is*
re-evaluated), but it means a calibration arm that edits a file **without** changing its
precompile trigger can silently measure the old image.

### B.3 What the gates are calibrated against, and the allowance mechanism

Both allowance flags (`--allow-shape-changed`, `--allow-removed`) are one-shot and
**fail when stale** (`name_surface_diff.py:124-137, 200-202`), and an allowance with no
`--allow-note` is a hard fail (`:197-199`). `UNINSPECTABLE` is a hard fail (`:211-215`),
so "we could not look" never compares equal to "nothing changed". Those are the right
properties, and V01 read them at the source rather than from the calibration log. The
residual risk is inherent and named: an allowance is supplied on the command line by the
same party running the gate, so a reviewer must read the gate log to see *what* was
allowed — which is why I02's claim "0 replaced, 0 removed, one declared shape change with
a falsifiable justification" is the sentence that matters, not "GATE: PASS".

## Summary of verdicts

Every row names the command or file the verdict rests on. `confirmed` means V01
reproduced the claim; `refuted` means V01 measured the opposite; `not_run` means the
experiment was not executed and no value is asserted.

| target | verdict | the measurement it rests on |
| --- | --- | --- |
| **T1** nzrange count, guards, patch identity | **confirmed** | count 5; guards at `:143/:208/:265/:298`; receiver typed at `:86`; patch applies at `4cb8d60` (`edfffdf7…` → `0c91a39f…`) and the result is byte-identical to HEAD |
| **T1** end-to-end "throws → Optimal" | **not_run** | the route to the patched functions is not reached post-cutover; I02 hit the same wall (I02-P3) |
| **T2** O(n⁴) class and structural hoistability | **confirmed** | read at `product_cone_hsd.jl:826-842, 887-924`; each hoisted quantity is invariant in the loop variable named, accumulation order unchanged |
| **T2** 92.5× kernel benchmark, bit-identity | **not_run** | the instrument is in no repository; the figure is not reproducible here |
| **T3** 28 legs, both invocation traps handled, no silent SKIP | **confirmed** | `grep -cE '^run '` = 28; `:94` runs the default leg under `$SDPX`; `env $envs julia … $args` at `:59`; 28/28 drivers present; `M03.jl` now tracked |
| **T3** the matrix is green | **pre-fix 27/28, post-fix 28/28** | `B03 exit=1`, zero Test Summary lines, pre-fix; `legs_run=28 legs_failed=0` and 28 legs with a Test Summary post-fix |
| **T3** `MATRIX_EXIT=0` as a verdict | **refuted** | the pre-fix script has no `exit` statement and ends in `echo`; the value cannot be anything but 0 |
| **T4** a test was weakened to reach green | **not found** | every committing diff with a deletion opened; no `@test_broken`/`@test_skip` added; the two removed P03 assertions are the known pin and tautology |
| **T4** the instrument is sound | **refuted** | `RUN_HISTORY.md` is stale; B03/B04's assertions live outside version control |
| **T5** all 120 `verified` entries are backed by their artifacts | **refuted** | three cite numbers in no artifact; further classes in the delegated audit |
| **T6** `contracts/workspace.jl` in both inclusion modes | **confirmed** | both modes run in one process; `MODE1_is_MODE2_module=false` |
| **T7** every public route terminates with a valid original-coordinate certificate | **confirmed, positive direction only** | LP/SOC/SDP/infeasibility ray all `cert_valid=true method=original_coordinates` |
| **T7** the gate can refuse | **not_run** | the `NumericalFailure` downgrade branch was never reached |
| **T7** "`src/public/` never references `SDPXCertification`" | **confirmed** | `grep -rn 'SDPXCertification\.' src/ test/` outside its own file is empty; the four `ResultCertificate{T}` sites are at the exact lines I03 names |
| **T8** there is no public sign patch to delete | **confirmed** | eight patterns, eight zeros |
| **T8** `public_sign_patches` is evidence | **refuted** | it is a literal at `compiled_problem.jl:475` asserted against itself at `S01.jl:474` |
| **T9** the withdrawn mechanism is gone from every assertion | **confirmed** | seven hits, each a record or the withdrawal; none asserts it |
| **T10** the lease hazard is fixed and falsifiable | **confirmed** | controls post-fix in my own run (`revoked=true`, lease bound `false`, no throw), 141/141 both modes; I02's rollback arm is the other direction |
| **T11** A01b-F1 is still live | **confirmed** | asserted in the shipped drivers and `true` in four archived runs; consumer-side audit `not_run` |
| **T12** the regenerated patches apply and reproduce their targets | **confirmed** | applied against each patch's own pre-image blob; five of six byte-identical to live, B02's three files identical; B04's patches reproduce `caches.jl` exactly |
| **T12** `git apply --check` at HEAD passes | **refuted** | it fails for all five SDPX-tree patches, for two different and benign reasons |
| **T13** B04-F2 was not asserted into green | **confirmed** | the LDLT flag is printed, not asserted; the report keeps `partially_verified` |
| **T14** the baseline is unreproducible from `5f9e5d8` | **refuted** | the recorded fingerprint `d62abb18…` is reproduced exactly from a clean archive |
| **A** the sparse `factorize!` gap and its reachability | **confirmed** | generation delta 0 direct and 0 through `SDPX.SparseQDLDLProviderCache`; dense control delta 1 |
| **B** the two gates are complementary | **confirmed** | one mutant: overwrite gate `0 → 1` warning, surface gate `GATE: PASS` exit 0 |
| **B** the SDPX overwrite arm works | **not_run** | vacuous by I02's own disclosure; no SDPX mutant exists |

### Findings, with owner and blocking level

| id | severity | one line | owner | blocks I03? |
| --- | --- | --- | --- | --- |
| **V01-F1** | medium | `RUN_HISTORY.md` is stale, so the packet's weakening instrument answers a different question | parent | no — but regenerate before the record is frozen |
| **V01-F2** | high | `public_sign_patches=0` is a literal asserted against itself (`compiled_problem.jl:475`, `S01.jl:474`) | S01 / I01–I02 authority | **YES** — it is a leg of R2's evidence and the exact defect F10 removed elsewhere |
| **V01-F3** | high | the 28-leg matrix was 27/28 pre-fix (B03 did not run) and `MATRIX_EXIT=0` was vacuous | parent (fixed) | **record** — post-fix run is green; I03 must cite the post-fix summary, not the pre-fix one |
| **V01-F4** | medium | B03/B04 assertions live in files inside no repository | I03 / parent | no, but state it: it bounds what "the driver is green" means |
| **V01-F5** | medium | `B03.jl`'s header is stale and its UNWIRED arm is unreachable in this tree | I03 or a follow-up | no |
| **V01-F6** | high | IP-2 does not cover the sparse `factorize!`, reachable from SDPX's provider seam | I02-P2 / MFLA | **YES** — I03's own R8 |
| **V01-F7** | medium | three `verified` entries cite numbers in no artifact; one quotes a pre-fix count | task owners / parent | **YES for S01[2]** (same chain as F2); the other two are record corrections |
| **V01-F8** | low | `name_surface_snapshot.jl` does not pass `--compiled-modules=no` | parent | no |
| **V01-F9** | low | the SDPX arm of the overwrite gate is vacuous | next SDPX wiring move | no |
| **V01-F10** | low | `contracts/workspace.jl` says it is not included; it is | MFLA | no |
| **V01-F11** | low | the 92.5× SOC benchmark has no archived instrument | parent | no |
| **V01-F12** | low | the certificate gate's negative direction is unexercised; the ADR's module is unused | I03 | no — but it changes the *reason* R1 is blocked |

### What must block the release decision (explicit)

**Yes, these block I03:**

1. **V01-F2 / V01-F7(a)** — R2's evidence chain contains an assertion that cannot fail and a
   number that appears in no artifact. This is the criterion the release decision itself
   says must block, and "blocked on an experiment that can never unblock" is the state F21
   identified. It stays blocked until the runtime observation F21 specifies is made and the
   literal is either computed from it or removed from the evidence chain. V01 found **no
   sign patch to delete** and confirms that half.
2. **V01-F6** — I03's own **R8**: a direct sparse `factorize!`, and SDPX's own
   `SparseQDLDLProviderCache` seam, leave the contract generation unchanged, so a lease
   taken before a refactor still validates after it. Reproduced with the dense arm as the
   sensitivity control. Fix (I02-P2's shape: one line, three exits, extension-only load) or
   record as accepted risk **with this reproduction**.
3. **T7's negative direction (V01-F12)** added to **R1** — the positive direction is now
   measured on four routes, but no case has ever driven the `NumericalFailure` downgrade, so
   "every public route terminates with a valid certificate" has only been shown for routes
   that succeed. R1 stays blocked. **V01 explicitly does not recommend unblocking it on the
   positive evidence**, and the reason to give I03 is the missing instrument, not the
   missing module reference.

**No, these do not block, and should be recorded rather than fixed at the end of a long
session:**

* **V01-F3** — the harness defects are fixed and the post-fix matrix is 28/28 with a real
  aggregate verdict. The pre-fix summary must be kept next to it so the record shows what
  was repaired, and `B03`'s red leg must **not** be written up as a B03 defect.
* **V01-F1, F4, F5, F8, F9, F10, F11** — instrument, comment and reproducibility defects.
  Each is cheap to fix and none changes a numerical result.
* **T2's disposition** — releasable as a documented performance limitation, with the
  kernel/end-to-end distinction and the missing instrument stated. It must not be fixed
  without an end-to-end measurement.
* **T14** — the parent's stated concern is answered by measurement: the Q01 baseline **is**
  reproducible from `5f9e5d8`. What must still be recorded is that `c4b109a` is a different
  tree (202 files vs 199), so Q01's numbers are attributable to the baseline revision and
  not to the frozen one.

**A note on what this audit could not do.** Four of the fourteen targets required an
instrument the packet does not have: T1's end-to-end route, T2's benchmark, T7's downgrade
case, and T5's full 120-row re-derivation (delegated, and preserved verbatim rather than
summarised). Each is recorded as `not_run` with its reason. A review that reported these as
passes would have been the most expensive artifact in the packet.
