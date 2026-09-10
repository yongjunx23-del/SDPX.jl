# I01 integration record — reachability, difference, rollback

Task card `I01`, acceptance item 3: 新旧模块的可达性、差异和回滚commit明确.

Written by the integration role (the only role permitted to edit includes and
delete old definitions). Every number below is produced by a command in this
document or by a log named in it. Where a claim is bounded, the bound is stated
rather than left for the reader to assume.

## 0. Revisions this record describes

| Repo | Pre-I01 | I01 integration commit | Version |
|---|---|---|---|
| SDPX | `37b57ea` (Q01) | `11fb0ee` (section A, 14 includes), `91f0c3f` (S02 cutover) | 0.6.1 |
| MFLA | `3ddf8ed` (M01) | `2294ada` (contract wiring) | 0.4.0 |
| BFLA | `9d9683c` (B01) | `db06034` (contract wiring) | 0.3.0 |

Environment: `REBUILD_ENV=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env`,
depot `rebuild-env-depot:~/.julia`, Julia 1.12.6, `Sys.CPU_THREADS == 4`, all
provider legs run in **separate processes with `-t1`** (the two-process rule from
Q01: MF and BF legs exhaust the Julia 1.12 inference compiler when co-resident).

ADR-002 §10 states that the provider load-code equivalence **expires** the moment
I01 wires these files. It has now expired for both providers. See §6.

## 1. Load reachability

Measured by `SDPX.jl/test/rebuild/I01_reachability.jl`, log
`rebuild-reports/I01_prework/I01_reachability.log`:

    files under src/                                  188
    files reached from src/SDPX.jl by include          188
    modules reachable from SDPX by nesting              17

The set of reached files is **equal** to the set of files under `src/`, not merely
equal in count — verified by set difference in both directions, which is the check
that catches an orphan file. There is no source file in `src/` that no `include`
reaches. Independently, no file in the closure is included more than once
(0 targets with >1 includer), so there is no double-execution of a file body.

### The 14 section-A entry points are loaded, and one of them is called

Load reachability is not call reachability. The section-A files add 338 distinct
top-level definitions (sum of the per-file counts in `I01_reachability.log`); a
call-syntax scan over the loaded non-new sources finds a call site for exactly
**one** of the fourteen modules:

| New module | Production call site | Verdict |
|---|---|---|
| `solver/loop.jl` | `src/hsd/product_cone_solve.jl:756` — `product_hsd_solve!` calls `solver_run_session!` | **call-reachable** |
| `core/compiled_problem.jl`, `core/transforms.jl`, `kkt/{operator,session,strategy,refinement_policy}.jl`, `la/{protocol,admission,factor_lease}.jl`, `certification/original.jl`, `planning/{costs,resources,setup}.jl` | none | **load-reachable only** |

This is the intended I01 state, and it is the reason I01 is safe: section A wires
*definitions*, and section C moves *responsibilities* one at a time with a
re-measurement after each. The single call edge is the S02 cutover, which was
taken as section C move 1 and has its own record in `ORCHESTRATION.md`.

`planning/setup.jl` is the next module S06 wants on the production path (its
`plan_setup` call-site swap). I01 deliberately did **not** take that: the manifest
lists it under section C, and it is deferred to I02. Until then `plan_setup` is
loaded and callable but not called by the default pipeline.

### The seven files section A did NOT list are loaded anyway

Section A names 14 entry points. Seven packet files are absent from that list. A
working note in this rebuild recorded them as "deliberately NOT listed" — which
was true — and then, in a later summary, as not being on the production path,
which was **false**. Measured:

| File | Loaded? | Into which module | Via |
|---|---|---|---|
| `solver/iterate.jl` | yes — 4/4 defs | `SDPX` | `solver/loop.jl:31` |
| `solver/session.jl` | yes — 13/13 defs | `SDPX` | `solver/loop.jl:32` |
| `solver/residuals.jl` | yes — 7/7 defs | `SDPX` | `solver/loop.jl:33` |
| `solver/globalization.jl` | yes — 3/3 defs | `SDPX` | `solver/loop.jl:34` |
| `solver/recovery.jl` | yes — 4/4 defs | `SDPX` | `solver/loop.jl:35` |
| `certification/status.jl` | yes — 12/12 defs | `SDPX.SDPXCertification` | `certification/original.jl:1408` |
| `certification/direction.jl` | yes — 6/6 defs | `SDPX.SDPXCertification` | `certification/original.jl:1409` |

They are reached **transitively**: `solver/loop.jl` and `certification/original.jl`
are themselves section-A entry points, and they `include` their siblings. So the
seven are load-reachable, and the five under `src/solver/` have their definitions
in the `SDPX` namespace. They are not orphans and they are not absent from the
build.

The two certification files land in the submodule `SDPXCertification`, which
`original.jl` opens at `:52` and closes at `:1411`. That placement is load-bearing:
`direction.jl` calls `min_cone_margin`, `stationarity_residual`, `psd_scale_of` and
`dual_map_consistent`, none of which exist in `SDPX` — they are defined by
`original.jl` inside the same submodule. An earlier draft of this record
proposed that these were dangling calls in dead files. Measured at runtime, all
four resolve inside `SDPXCertification`, and the files are live. The calls are
fine; the *reachability story* was what needed correcting.

### A dual-mode file, and why "run it in both modes" is now a rule

`src/kkt/{operator,session,strategy,refinement_policy}.jl` each end with:

    Core.eval(SDPXKKT_CONTAINER, :(include($(String(@__FILE__)))))

When included from `src/SDPX.jl` — where `SDPX` is defined and
`SDPXKKT_BOOTSTRAPPED` is not — the file splices its definitions directly into
`SDPX`. When a test script includes it, it creates a container module `SDPXKKT`
and re-includes itself there. Two inclusion modes, two namespaces, one file.

This is the correct design for the constraint, and it is the fourth appearance in
this rebuild of the same hazard class: *a file exercised in only one inclusion
mode*. The first three were defects (MFLA's `const MultiFloat` collision, BFLA's
`AdmittedContext` ordering, S06's missing `else` branch). The mitigation adopted in
`CHANGE_MANIFEST.md` §E step 3 — run every driver in **both** modes — is what
catches this class, and it is the reason §2 below reports two logs per driver.

### No name has two live homes

`I01_reachability.log` asks the loaded module tree, not the text, which names from
the section-A files resolve in more than one module. Three do:

| Name | Homes | Assessment |
|---|---|---|
| `OriginalOperator` | `SDPX` (from `kkt/operator.jl`), `SDPX.SDPXCertification` (from `certification/original.jl`) | two distinct `Type`s in two modules; no shadowing |
| `certify!` | `SDPX` (1 method, `kkt/refinement_policy.jl`), `SDPX.SDPXCertification` (1 method) | two distinct generics |
| `dual_slack` | `SDPX` (4 methods) + 7 factor-pair submodules | pre-existing in the submodules; `certification/original.jl` adds an eighth, separate, definition |

All three are cross-module, which is not a collision. A first version of this
check compared *files* instead of modules and reported 109 "duplicates"; every one
was a false positive caused by `src/` opening 17 submodules and by generic
functions legitimately collecting methods from several files. The check now asks
the loaded module tree and the method tables, and the false positive rate is zero.

**Same-signature method overwrite is a different failure and is not observable by
this script.** The evidence is the inherited suite, which `Pkg.test()` loads with
`--warn-overwrite=yes`. Exactly one overwrite warning appears in all three logs,
in a test-only module and outside the change surface:

    WARNING: Method definition _center_upper(CompensatedExpReference.CompensatedBound)
    ... validation/scientific_core/exp_runtime/compensated_exp_reference.jl:155
    overwritten at ...:156

Zero warnings originate in `src/`.

## 2. Difference

### SDPX — the inherited suite is byte-identical after normalisation

Three full `Pkg.test()` runs on three revisions:

    SDPX_pkgtest_CLEAN_TREE.log      37b57ea   pre-I01
    SDPX_pkgtest_WIRED.log           11fb0ee   section A
    SDPX_pkgtest_POST_CUTOVER.log    91f0c3f   after the S02 cutover

    testsets=170   Pass=9392   Broken=7   Total=9399   Fail/Error columns: 0

identical in all three. Normalised by
`SDPX.jl/scripts/rebuild/normalize_pkgtest_log.py`, which strips only wall-clock
times, `jl_*` temp directory names, and the recorded SHA, the three logs are
**diff-identical**:

    python3 scripts/rebuild/normalize_pkgtest_log.py \
        rebuild-reports/I01_prework/SDPX_pkgtest_{CLEAN_TREE,WIRED}.log
    # normalized diff: empty

**Correction to an earlier figure.** `ORCHESTRATION.md` reports "9395 pass" for
these runs. That number does not reconcile: the log carries `Pass` and `Broken`
columns per testset, and the only split that sums to the 9399 total is
`9392 + 7`. There is no outer `Test Summary` line to quote instead — the log has
170 per-testset summaries and no grand total. The 9395 figure appears to have been
formed by subtracting four `@test_skip`s from the total, which double-counts:
Julia reports skips inside `Broken`. The reconciled numbers are above, and because
the error was in the *reported split* and not in the total, it does not change any
conclusion — but the headline number of the rebuild should be one that a reader can
re-derive from the log.

### BFLA — additive, with a control

    10864/10864 before wiring (9d9683c-equivalent)   BFLA_pkgtest_baseline.log
    10864/10864 after wiring  (db06034)              BFLA_pkgtest_WIRED.log

Adding a wiring and finding the suite unchanged is weak evidence on its own,
because a suite can be blind to additions. So the name surface was diffed
directly, at both revisions:

    exported names            115 -> 115   (byte-identical set)
    real (non-gensym) names   272 -> 350   0 removals, 78 additions

The 108 apparent removals are `#`-prefixed closure and gensym names whose counter
shifted because the contracts add top-level expressions. No real binding was lost.

Both B01 driver modes pass, with **identical measured values**:

    BFLA_B01_driver_WIRED.log       12 testsets, 91/91, 33 MEASURE lines
    BFLA_B01_driver_WIREDMODE.log   12 testsets, 91/91, 33 MEASURE lines

The two logs' `MEASURE` values are identical except the module qualification of
the `FactorFacts` type (`Main.B01Contracts.FactorFacts` vs
`BigFloatLinearAlgebra.FactorFacts`). That single difference is the point: it
proves the wired run exercised the package's own definitions while the sandbox run
exercised a private copy. Without it, "both modes pass" would not distinguish the
two. The wired variant is generated by `make_B01_wired.jl`, which fails closed if
`B01.jl` changes shape, and it reproduces the checked-in `B01_wired.jl`
byte-for-byte.

### MFLA — identical to baseline

    23 testsets, 4164 assertions, same testset names and counts,
    `Testing MultiFloatLinearAlgebra tests passed`

### What is *not* claimed

The suite is a black-box regression reference for arithmetic, and it is the one
that matters here, but it is not a proof of arithmetic identity: it compares
observable results, not execution traces. The structural argument is separate and
stronger for this particular change — section A adds only top-level definitions to
the module, and the S02 cutover moves an unchanged loop body behind a wrapper whose
keyword arguments and return type match the function it replaces (verified field by
field in `ORCHESTRATION.md`). No tolerance, damping, centering constant, default, or
promotion rule was touched.

## 3. Rollback

The three I01 commits are independent and revert in reverse order.

    # SDPX — back to the pre-I01 tree
    git -C SDPX.jl revert --no-edit 91f0c3f      # the S02 cutover
    git -C SDPX.jl revert --no-edit 11fb0ee      # section A's 14 includes
    # now at 37b57ea

    # MFLA
    git -C MultiFloatLinearAlgebra.jl revert --no-edit 2294ada   # -> 3ddf8ed

    # BFLA
    git -C BigFloatLinearAlgebra.jl revert --no-edit db06034     # -> 9d9683c

Each revert is a pure include-graph change plus, for `91f0c3f`, restoring one
function body. Nothing else in the three repositories depends on the new includes.

**What rollback restores.** The pre-I01 include graph and the monolithic
`product_hsd_solve!`. After reverting `91f0c3f`, `solver/loop.jl` is loaded but
never called, which is exactly the state section A was verified in.

**What rollback does NOT restore, deliberately.**

  * The new files *stay in the tree*. `src/contracts/*.jl` (MFLA),
    `src/contracts/*.jl` + `src/mpfr_context.jl` (BFLA) and all 14 SDPX modules
    existed before I01 — verified inert for BFLA by extracting `9d9683c` and
    grepping its include list, which matched nothing. Reverting the includes
    returns them to inert, not to absent.
  * `S02-D` turns **red**. It was inverted from "section A has not wired
    `solver/loop.jl`" to "the loop body is gone and `product_hsd_solve!` is a
    wrapper". A silent rollback therefore fails a test rather than passing
    quietly, which is the property the inversion was made for.
  * The provider load-code equivalence in ADR-002 §10 does **not** come back. See
    §6.

**Rollback was not exercised.** Reverting three commits and re-running three
suites is a bounded, mechanical operation, but it is not free, and it was not
performed here. This is recorded as `not_run` rather than asserted as safe. The
claim that each commit is independently revertible rests on the commits touching
disjoint files, which is checkable and is the reason they were made separately:

    git -C SDPX.jl show --stat 11fb0ee 91f0c3f

## 4. The default pipeline is unique

I01 acceptance also requires 一个生产路径 — no new and old definitions of the same
operation both live. Three checks:

  * **No file is included twice** anywhere in the `src/` closure, so no file body
    executes twice and overwrites its own methods.
  * **Zero method-overwrite warnings from `src/`** under `--warn-overwrite=yes`.
  * **`product_hsd_solve!` is the only HSD loop.** `src/solver/loop.jl` is the
    single loop implementation; `src/hsd/product_cone_solve.jl` retains no loop
    body (`S02-D` asserts the loop-body count is 0 and that the wrapper is present,
    9/9).

Also verified, and bounded to what was actually measured:

  * **MFLA and BFLA take no dependency on `SDPX`.** Both `Project.toml`s list only
    their own deps (`LinearAlgebra` + `MultiFloats`; `LinearAlgebra` +
    `MutableArithmetics`) plus declared weak deps, and the only occurrences of the
    string `SDPX` anywhere in either `src/` are four comments (`factor_caches.jl`
    :730, `MultiFloatLinearAlgebra.jl:41`, `native_backend.jl:1`, `types.jl:13`).
    No code reference.
  * **The HSD path does not call into the public layer.** `src/hsd/*.jl` mentions
    `public/` twice, both in comments. This is a text measurement over `src/hsd/`,
    not a call-graph proof for the whole solver path.

There is no `src/compatibility/` directory in this tree, so the task card's
"does not depend on compatibility" has no artifact to point at; `compatibility`
appears in `src/` only as a constraint name ("compatibility constraint
contractions"). **Not measured:** a full call graph from
`product_hsd_solve!` showing every function it can reach. The check above bounds
the question; it does not close it.

## 5. Residual state after I01

| Item | State |
|---|---|
| Section A (14 includes) | complete, verified |
| Section C move 1 (S02 cutover) | complete, verified; `S02-D` 9/9 |
| Section C remaining moves | **not started** — S05-P2, S01-P2, S04-I03, S06 `plan_setup`, M01 IP-2/3/4 |
| B01's `src/mpfr.jl` rounding change | **not applied** — behaviour change, belongs to I02 |
| S01-P2 (public sign patch removal) | **blocked** — needs a before/after MOI solve that nobody has run |
| Batch 5 (M02, M03, B02–B04, P02, P03, S07) | not started; all depend on I01 |

## 6. ADR-002 §10's equivalence has expired

§10 stated that its provider results hold only while the packet's new files stay
un-wired, and that I01/I02 must re-check and I03 must not freeze a release on a
stale equivalence. Both providers are now wired:

| Provider | Equivalence was to | Wired at | Status |
|---|---|---|---|
| MFLA | `50e6e0b` (via `3ddf8ed`) | `2294ada` | **void** |
| BFLA | `f95d3e6` (via `9d9683c`) | `db06034` | **void** |

The justification in §10 was "zero deletions + an unchanged module entry point +
no new file in the include graph ⇒ `using <Provider>` executes identical code at
either revision". I01 changed the entry point for both, so that inference no longer
holds at the new revisions and no result may cite it. Results measured at the
pre-wiring revisions remain valid **as results at those revisions**; they may not
be re-attributed. Every provider-dependent result from here on names the revision
it measured.

ADR-002 carries the revision of this record as §11.
