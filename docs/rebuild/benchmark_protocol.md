# Benchmark and measurement protocol

**Established by Q01, 2026-09-11. Revised the same day for the provider
environment (`$REBUILD_ENV`) and the arithmetic axis.** Binding on every task
that reports a number. Implements ADR-003 §7 (measurement discipline) and the
packet's Q01 card.

Baseline for every figure in this document: `docs/rebuild/baseline.md`, plus the
correction recorded in `scripts/bootstrap_env.jl` (the provider environment now
exists and is recorded here).

---

## 1. The three phases are never averaged

| Phase | What it is | May be quoted? |
|---|---|---|
| `first_compile` | the first solve of a case in a process, LLVM compilation included | **no** — diagnosis only |
| `warm_fresh_setup` | repeats with `clear_structure_cache!()` before each solve: compiled code warm, symbolic structure rebuilt | with the counts and the environment |
| `prepared_solve` | repeats with the cross-solve structure cache left warm: the frozen sparsity structure is reused | with the counts and the environment |

Measured on this baseline the gap between the first and the warm phases is four
orders of magnitude (`lp_afiro_style`: **27.3 s** first, **0.00085 s** warm). A
single averaged number describes neither and would move with the order the cases
happen to run in.

**The phase distinction is evidenced, not asserted.** Every sample records
`structure_cache_hits_delta` / `structure_cache_misses_delta`, so a reader can
check that `warm_fresh_setup` really rebuilt the structure (0 hits / N misses)
and that `prepared_solve` really reused it (N hits / 0 misses).

**Limit, stated because it is easy to overstate:** the cross-solve structure
cache is the *only* difference between the two warm phases. It does **not**
eliminate the provider symbolic analysis —
`symbolic_analyses_delta` counts real CHOLMOD/QDLDL factor-construction analyses
and is **1 per solve in both phases** on the bordered route. On these shapes the
timing difference between `warm_fresh_setup` and `prepared_solve` is within
run-to-run noise, and **no conclusion may be drawn from it**. The payload carries
this caveat as `phase_evidence.caveat` so it travels with the numbers.

**A true prepared-update replay is NOT measured.** Re-solving a new
objective/RHS into an existing prepared object needs the S04/S07 prepared-update
API, which does not exist at this baseline. The payload reports
`prepared_update_replay_status = "not_run"` with that reason. Per ADR-003 §3 an
unreachable phase is `not_run` with a reason — not estimated, and not `0`.

## 2. Timing medians are over successes, with failures reported beside them

```julia
success(row) = !row.threw && row.status == "optimal" && row.certificate_valid === true
```

Medians are taken over successful runs only, and every aggregate carries
`solved` and `failed` beside it, plus `seconds_min` / `seconds_max`, so a fast
failure cannot be presented as a speed win and a single lucky sample cannot be
presented as the number. The full raw rows are retained in the artifact under
`cases[].all_rows`. This rule was already implemented in
`benchmark/clarabel_borrowing/start_point_comparison.jl`.

A failure is data: it is recorded with its status, exception text and elapsed
time. Nothing is filtered, dropped or re-run into a success.

## 3. Unmeasured is `not_run`, never `0`

ADR-003 §3. A missing measurement and a measured zero are different facts, and
serialisation preserves the difference: `_tomlify(nothing) == "not_run"`.

Two further conventions keep `"not_run"` from having to explain itself:

1. Every nullable field that matters has a companion `*_status` or `*_reason`
   field. A field that is `nothing` for a boring reason carries
   `reason = "none"`; a field that is `nothing` because it could not be measured
   carries the specific reason (for example
   `native_allocator_reason = "no in-process counter for MPFR/GMP native malloc…"`).
2. Per-run unavailability is listed in `cases[].unavailable` as
   `"field — reason"` strings, produced by `null_reason` at the point where the
   value is found to be missing.

A case with no independent oracle records
`objective_oracle.status = "no_independent_oracle"` and `objective_error` is
`"not_run"`, never `0.0`.

### A per-case time budget is allowed; deleting a case is not

A pathological case can consume unbounded wall clock (measured: an unbounded
BigFloat-256 arm was still running after 23 min with a profile dominated by
MPFR heap allocation and GC). `--case-budget-seconds=N` bounds the measured wall
time one case may consume. A case that exceeds it is **not deleted and not
quietly given fewer repeats**:

- every sample already taken is kept;
- every phase that could not be reached is `not_run` with `runs = 0`;
- `cases[].budget` records `limit_seconds`, `measured_seconds`,
  `budget_tripped_in`, `phases_with_samples`, `phases_without_samples` and a
  reason naming the measured elapsed time and the limit.

The reason must be the measurement ("budget of 300 s exceeded after 315.4 s of
measured wall clock"), never "slow". A case reported `not_run` with a measured
cost and a reason is a *counted* result; a dropped case is a *missing* one.

The artifact is also **streamed after every case** (`partial = true`,
`cases_completed`), so a bounded, interrupted or killed arm keeps the evidence it
already paid for. The first version of this harness wrote only at the end and
lost 23 minutes of BigFloat work when the arm was killed — that is a defect this
protocol now forbids.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | measured; case failures are allowed and counted (they are data) |
| 2 | harness/usage error — e.g. not started with `-t1` |
| 3 | the requested arithmetic arm is not runnable in this environment |
| 4 | `SDPX_MEASURE_REQUIRE_IDLE=1` and the host is not idle |

Measured at this baseline:

```
$ julia --project=. -t2 benchmark/rebuild/measure.jl --cases=lp_degenerate --repeats=1
measure.jl: this harness must run with -t1 (got Threads.nthreads()=2). …          (exit 2)

$ SDPX_MEASURE_REQUIRE_IDLE=1 julia --project=. -t1 benchmark/rebuild/measure.jl --cases=lp_degenerate --repeats=1
measure.jl: refusing to measure: the host is not idle (other_julia_processes=1,
  loadavg_1m=3.7158203125). …                                                     (exit 4)
```

`3` **must not** be reported as a numeric failure, and a missing provider is an
infrastructure result (ADR-003 §3), not a solver defect.

## 4. Requested and executed threads are recorded separately

They genuinely differ on this host:

```
requested=1  julia=1  blas=4  cpu=4      (default project, -t1)
```

`requested_threads`, `julia_threads`, `blas_threads` and `cpu_threads` are all
recorded, together with the full Julia command line, so quoting one of them
cannot hide the others. Packet tiers 16 and 64 are **not measurable here**
(`Sys.CPU_THREADS == 4`) and are recorded as `unsupported` with the host fact.
Per `AGENTS.md`, `unsupported` may satisfy an explicit-refusal negative test but
**cannot** satisfy a required release capability.

## 5. The environment is part of the measurement identity

A figure measured in a **capability-enabled** environment is not comparable to
one measured in the **provider-free** default project, so every payload names its
environment:

| Field | Why it matters |
|---|---|
| `environment.label` | `sdpx_default_project` (no provider resolves) or `rebuild_env` (MFLA/BFLA/QDLDL resolve) |
| `environment.project_path`, `manifest_path` | which environment actually produced the number |
| `environment.project_sha256`, `manifest_sha256` | the exact resolved environment, hashed |
| `environment.providers[].version`, `.revision` | **the provider revision the number belongs to** |
| `environment.providers[].resolvable_in_load_path` | resolvability is not the same fact as being in the active Manifest |

Providers are read from the active `Manifest.toml` (version + dev path) and the
dev path's `git rev-parse HEAD` (revision). Reading the manifest does not load
the package, so a Float64 arm can record the provider configuration without
compiling provider specialisations.

**Route changes between environments are configuration differences, not
speed-ups.** Adding QDLDL makes some sparse routes executable that skip or fail
closed in the default project, so a route can change between the two
environments. Every single run therefore records
`requested_kkt_route`, `executed_kkt_route`, `executed_kkt_storage`,
`executed_provider` and `executed_factorization_kernel`, read from the engine
receipt. If the executed route differs between two environments, the difference
must be reported as a configuration change.

### Provider revisions actually measured

| Provider | Revision measured | Note |
|---|---|---|
| SDPX | read from `git rev-parse HEAD` **at measurement time** | every payload records `sdpx_head` and `worktree_dirty` |
| MFLA | `50e6e0b` = `v0.4.0` (`b38dea1`) + 3 commits | version string still `0.4.0` |
| BFLA | `9d9683c` = `v0.3.0` (`f95d3e6`) + 1 commit | the extra commit is packet work B01 |

The BFLA checkout is one commit past the freeze. It is **not** reported as
`f95d3e6`. The loaded code is nevertheless identical to `f95d3e6`, and that is
shown rather than asserted:

```
$ git -C BigFloatLinearAlgebra.jl diff --name-status f95d3e6 HEAD
A  src/contracts/context.jl
A  src/contracts/factor_summary.jl
A  src/contracts/ownership.jl
A  src/mpfr_context.jl
A  test/rebuild/B01.jl
$ git -C BigFloatLinearAlgebra.jl diff f95d3e6 HEAD -- src/BigFloatLinearAlgebra.jl
(empty — the module entry point is byte-identical)
```

Five added files, zero deletions, and an unchanged entry point: no existing file
can reference the new ones, so the include graph and therefore the loaded code
are the same. That is an equivalence of **loaded code**, established from the
diff — not a measurement taken at `f95d3e6`. Full output:
`rebuild-reports/Q01/logs/bfla_revision_equivalence.txt`.

## 6. The arithmetic axis, and the two-process rule

Arithmetic is an **environment axis**, never a property of a case name. The same
eight inputs are measured once per arm:

| Arm | Arithmetic | Bits | Provider |
|---|---|---|---|
| `float64` | `Float64` | 53 | stdlib |
| `multifloat_x2` | `Float64x2` | 106 | MFLA |
| `bigfloat_256` | `BigFloat` | 256 | BFLA |

`measure.jl --arithmetic=<arm>` measures exactly one arm. Two hard rules:

1. **One arithmetic per process.** `scripts/provider_smoke.sh` documents that
   Julia 1.12 can exhaust its inference compiler when the MFLA fixed-width and
   the BFLA/MPFR specialisations are compiled in the same process; its `all`
   target therefore runs as two fresh processes. This harness does the same.
2. **`-t1` is enforced.** The harness refuses to run otherwise (exit 2), because
   the thread payload must be unambiguous and because the two-process rule above
   assumes a single-threaded process.

A dynamically resolved arithmetic type (`Float64x2` does not exist until
`MultiFloats` is loaded) makes the entry two-staged, both stages through
`Base.invokelatest`: stage 1 resolves the arm and loads its provider packages,
stage 2 runs the measurement in the world that includes their methods. Without
this, `MultiFloat{Float64,2}(::Float64)` fails with a world-age error.

If a package the arm names cannot be loaded, or the SDPX extension for that
provider is inactive, the arm is **not runnable**: the payload is still written,
with `arithmetic.arm_status = "not_run"` and the resolution facts as its reason,
and the process exits 3. No number is invented.

## 7. BigFloat allocation is disaggregated into four axes

ADR-003 §7 forbids reporting one allocation axis as another. Each axis is
recorded separately, with its own provenance:

| Axis | How it is obtained | What it does NOT show |
|---|---|---|
| `julia_heap_bytes` | `@allocated` around the whole public call | it covers the GC heap only, and includes every non-BigFloat allocation |
| `cell_identity` | `objectid` counts over published primal+dual BigFloat values: elements, distinct cells, aliased pairs | identity, not bytes; `@allocated == 0` would not have shown MPFR traffic |
| `native_allocator_bytes` | **`not_run`** — no in-process MPFR/GMP `malloc` counter exists here | it is not RSS, and RSS is not reported as it |
| `rss` | `Sys.maxrss()` peak counter, before/after | it is a **peak** counter: a `0` delta means "no new peak", **not** "no memory touched" |

For non-BigFloat arithmetic the block is `not_applicable` with the reason
"arithmetic is $(T): no BigFloat cells and no MPFR allocations exist" — which is
a different statement from `not_run`, and is kept different.

## 8. No name-based dispatch

A benchmark **name** may be recorded in a result and used to look up a fixture.
It may **not** choose a route, provider, formulation or settings value.
`case_settings(case, T; …)` takes an arithmetic type, tolerances and limits only;
`test/rebuild/dependency_rules.jl` asserts that structurally, and separately
asserts that no name comparison anywhere under `src/` or `benchmark/` branches to
a route or provider identifier.

Two earlier formulations of the rule were tried and rejected, and the rejections
are recorded in the test file because they are the reason the final rule is
shaped as it is: *"does the tree compare a spec id?"* flagged nine legitimate
fixture lookups; adding *"is strategy vocabulary nearby?"* flagged a descriptive
`conic_formulation` field. The final rule asks the question the packet actually
poses — does a name comparison branch to a real route or provider identifier? —
with the vocabulary taken from the public route validator. Result at this
baseline: **0 violations**, 13 informational name comparisons that only label
records.

## 9. Static checks declare themselves static

A static scan can find a forbidden dependency that exists. It **cannot** prove
that an allowed path is exercised, that two methods are never both loaded, or
that some code is unreachable. Every check in `dependency_rules.jl` carries
`static_only = true` and `limits_note`, and the first testset asserts those
declarations exist.

Where dynamic evidence was cheap and real, it is recorded **as an observation of
one session**, printed with `@info` and explicitly labelled *not a reachability
proof*: which provider modules are loaded in the session running the test, and
which are resolvable from the load path. The absence of a loaded provider in one
session does not prove that no path can reach one.

## 10. Contention is recorded, not assumed away

A performance number taken while other processes are running is not comparable
to one taken on an idle host. Every case records
`other_julia_processes`, `loadavg_1m/5m/15m`, and a `contended` boolean under a
stated rule:

```
contended = (other_julia_processes >= 1) || (loadavg_1m > 0.75 * cpu_threads)
```

This is an **observation of the host**, not a measurement of another process's
effect on this one. `SDPX_MEASURE_REQUIRE_IDLE=1` turns it into a refusal
(exit 4).

## 11. Functional tests and performance measurement do not share a machine

Packet acceptance: "支持独占性能runner；并行功能测试与性能测量分开."

- `test/rebuild/dependency_rules.jl` is a **functional** check (234 assertions,
  ~2 s) and is safe to run in parallel with anything. It performs no timing and
  no allocation measurement.
- `benchmark/rebuild/measure.jl` is **not** part of `Pkg.test()` and must own the
  machine. It is never included from `test/runtests.jl`, so a test run cannot
  silently become a performance measurement.
- The measurement window used for this revision was **contended** (sibling packet
  tasks were running Julia processes; load average ≈ 7–9 on a 4-thread host).
  Every artifact says so. Wall-clock medians from that window are valid for
  *phase separation* and *environment identity*; they must not be quoted as
  clean speed figures, and no speed-up claim is made from them.

## 12. Reproducibility

`manifest.jl` produces two identities:

- `cases_fingerprint` — a stable hash over the **declared** input shape trace
  (dimensions, cone signature, id, family), unchanged from the first Q01
  revision so the manifest identity stays comparable; it deliberately excludes
  the `build` closure.
- `inputs_fingerprint` — SHA-256 over the solver's actual numeric input: the
  canonical `A`, `b`, `c`, the canonical block layout, the arithmetic, the
  precision bits, the tolerances and the limits. Route, provider and formulation
  are deliberately **not** hashed: they are outcomes, not inputs. Two runs with
  the same fingerprint at the same SDPX SHA describe the same problem at the same
  target.

`Serialization` is the byte encoding, which is deterministic within a Julia
version; the Julia version is recorded beside the hash.

### Shape trace: declared vs observed

`observed_shape_trace` compiles each case and reports what the canonical program
really contains. The declared trace was hand-written in the first Q01 revision
and **does not agree with it for all eight cases**; the disagreement is returned
as data (`cases[].shape.declared_vs_observed`) instead of being quietly
rewritten. Notable differences: the variable count agrees everywhere, but `m`
was declared as user constraint rows while the canonical slack dimension is
larger, and the declared cone signature lists a phantom leading `(:zero, n)`
block for the free variables and uses `:nonneg` where the canonical layout says
`:nonnegative`. See the Q01 report for the per-case list.

### SHA discipline

`measure.jl` reads `git rev-parse HEAD` **at measurement time** and records
`sdpx_head` and `worktree_dirty`. SHA values copied from a brief, a card or an
older report are not evidence: this baseline moved several times during the day,
and the SHA recorded in the artifact is the only one a number may be attributed
to.

## 13. What this protocol DOES establish at this baseline (2026-09-11)

Recorded so the next task inherits facts rather than re-deriving them. Every
number below is read from an artifact, not from memory.

| Arm | Environment | Rows optimal | Rows failed | Worst case measured |
|---|---|---|---|---|
| `float64` | default project (provider-free) | 56/56 | 0 | 22.11 s (`lp_afiro_style`, compile-inclusive) |
| `float64` | `$REBUILD_ENV` | 56/56 | 0 | 22.04 s |
| `multifloat_x2` | `$REBUILD_ENV` | 48/55 | 7 | 315.4 s (`soc_k128`, budget-stopped) |
| `bigfloat_256` | `$REBUILD_ENV` | 42/56 | 14 | 48.95 s (`lp_afiro_style`) |

- **Executed routes are identical between the two environments** for all 8
  cases, so the QDLDL capability difference did *not* move any of these cases
  onto a different route. This was checked from the receipts, not assumed.
- **Phase separation is real and evidenced**: `warm_fresh_setup` shows 0 hits /
  N misses, `prepared_solve` shows N hits / 0 misses, per case, in every
  artifact.
- **Compilation dominates the first solve**: 22-49 s first vs 0.0003-3 s warm,
  depending on case and arithmetic.
- **BigFloat at 256 bits is allocation/GC-bound on this host**, not
  arithmetic-bound: a profile of the in-flight arm attributes 597 of 692
  main-thread BigFloat samples to `jl_alloc_genericmemory_unchecked`, 432 of them
  inside `jl_gc_collect`.
- **soc_k128 is the slow case in every arithmetic** (3.2 s Float64, ~52 s
  Float64x2 per solve, memory-refused in BigFloat 256) — kept, never dropped.

## 14. What this protocol does NOT establish

- **No prepared-update replay**, per §1.
- **No native-allocator measurement** for BigFloat, per §7. RSS is recorded, but
  it is not that number.
- **No BigFloat cell-identity number for Float64**, because it does not exist —
  recorded `not_applicable`, not `0`.
- **No MFLA threading or performance claim at the frozen revision.** MFLA is
  measured at `50e6e0b` (v0.4.0 + 3 commits, one of which is a threading change);
  a threading or throughput claim must name that revision.
- **No thread tiers above 4** on this host, per §4.
- **No clean-machine timing.** The window was contended, per §11.
- **The allocation figure is a whole-public-call figure**, including result
  construction, certificate assembly and recovery. It is not an inner-loop
  allocation figure and must never be reported as one.
- **`dependency_rules.jl` is STATIC**, per §9. It cannot prove dynamic
  reachability.

## 15. Exact commands

```bash
# functional/structural check (no timing; safe to parallelise)
cd SDPX.jl
julia --startup-file=no --project=. test/rebuild/dependency_rules.jl

# Float64 arm, provider-free default project, one process, one thread
julia --startup-file=no --project=. -t1 benchmark/rebuild/measure.jl \
  --arithmetic=float64 --repeats=3 --env-label=sdpx_default_project \
  --out=benchmark/rebuild/measure_result.toml

# provider arms: capability-enabled environment, SEPARATE processes, -t1
export JULIA_DEPOT_PATH=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env-depot:$HOME/.julia
export REBUILD_ENV=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env
julia --startup-file=no --project="$REBUILD_ENV" -t1 benchmark/rebuild/measure.jl \
  --arithmetic=float64      --repeats=3 --env-label=rebuild_env --out=…/measure_float64_rebuild_env.toml
julia --startup-file=no --project="$REBUILD_ENV" -t1 benchmark/rebuild/measure.jl \
  --arithmetic=multifloat_x2 --repeats=3 --env-label=rebuild_env --out=…/measure_multifloat_x2_rebuild_env.toml
julia --startup-file=no --project="$REBUILD_ENV" -t1 benchmark/rebuild/measure.jl \
  --arithmetic=bigfloat_256  --repeats=3 --env-label=rebuild_env --out=…/measure_bigfloat_256_rebuild_env.toml
```

Selection and diagnostics: `--cases=id1,id2` runs a subset (used for smoke
runs); `--repeats=N` sets the warm repeat count per phase;
`SDPX_MEASURE_REQUIRE_IDLE=1` refuses a contended host.
