# SDPX parallel-execution design (diagnosis + proposal)

Date: 2026-09-09. Evidence-based design note. It does not change any default;
it records why thread scaling is poor and what a better execution model looks
like. Numbers are measured unless marked as an estimate.

## 1. Diagnosis: thread scaling is poor, and the cause is workload- and policy-dependent

**Framing correction (Astra review).** The `s` values below are *effective
scaling parameters fitted to two/three points under the current design*, not
proven intrinsic serial fractions, and the "cap" is conditional on that model.
A true serial fraction requires per-phase worker-count and wall-time
instrumentation (Section 4). Likewise a low whole-job CPU% does **not** by
itself prove node contention: serial phases or policy gating produce the same
signature. Treat every number here as an effective observation to be explained,
not as a structural constant.

### 1.1 Two workloads, two different observed signatures

| Workload | p=1 | p=4 | p=8 | p=16 | p=32 | fitted effective `s` | model cap at 32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LP `random_large` (m=400,n=1200), cluster | 6.737 s | — | 6.573 s | 6.730 s | 7.960 s | ~0.97 | ~1.03× |
| CSDR frozen α3 Float64x4, local | 41.313 s | 23.056 s | — | — | — | ~0.41 | ~2.33× |

- The LP does **not** lack a parallel path: `src/pipeline/plan.jl:318-322`
  selects `:parallel_blas_panels` for `:lp_primal_dual` Float64 when
  `selected_threads > 1 && cone_rows * variables^2 >= 2_000_000 &&
  blas_threads() == 1`; weighted-Gram kernels exist. The flat curve may
  therefore be a **policy/threshold/work-size** outcome (path not selected, or
  selected but dominated by serial work, or overhead), not an absence of
  parallel arithmetic. This must be resolved by instrumentation before
  redesigning.
- CSDR α3 does scale (1.79× at 4 threads) but far below ideal; the serial
  share needs attribution.

### 1.2 Where α3's time goes (1 thread, solver 41.3 s, phase trace)

| Phase | seconds | share | note |
| --- | --- | --- | --- |
| `setup` | 1.216 | 2.9% | serial |
| `residual_seconds` | 2.911 | 7.0% | mostly serial sweeps |
| `scaling_seconds` | 0.104 | 0.25% | serial |
| `direction_seconds` | 28.397 | 68.7% | **overlaps its children** |
| — `kkt_factorization_seconds` | 12.335 | 29.9% | charges metric prep + panel transform + SYRK + LU + homogeneous solve, not pure LU |
| unaccounted | ~8.7 | ~21% | the canonical phase projection omits buckets |

105 factorizations. **Label caveat:** the trace reports
`selected.executed_factorization_kernel=mfla_pivoted_ldlt`, but
`src/hsd/native_hsd_public.jl:1107-1117` labels every executed core that way
while fixed-trace actually constructs `ProviderLPLUCache`
(`fixed_trace_q3.jl:949`) and `factor_cache/routes/lp_lu.jl:291` calls
`la_lu_factor!`. The label is **mislabeled**; do not infer LDLT. The
"factorization" bucket is dominated by the **triangular Gram assembly**
(exactly 7,585,200 MACs per assembly), which *is* parallelizable; the 42×42
factor is negligible. The `direction_seconds` bucket overlaps its children, so
adding phase times double-counts; the canonical projection omits
residual/scaling buckets instead.

### 1.3 Design defects that make efficiency worse than the Amdahl bound

1. **Thread-budget mismatch.** Fixed-trace Q3 hot loops and
   `predictor_corrector.jl` schedule with `Threads.nthreads()` rather than the
   admitted `plan.selected_threads`. A requested `Limits(...threads=4)` does not
   establish four workers, and concurrent solves each expand to the full Julia
   pool.
2. **SIMD metric silently disabled by a layout assumption.** The vec4 HKM
   metric required SOC offsets `3*(b-1)+1` (SOC rows first), but the
   structured-A gate requires equality rows **before** SOC rows
   (`fixed_trace_q3.jl:983-994`). For α3 (42 equality rows, SOC offsets
   `43+3(b-1)`) the vec4 metric never activates and every block runs the scalar
   metric; SPD inverse and RHS also stay scalar. A parent candidate makes the
   kernel offset-aware (bit-identical; see Section 5).
3. **Atomic per-block claiming.** The HKM loops claim one block (scalar) or four
   blocks (vec4) per `atomic_add!`; with thousands of blocks this is a
   contention point, and the claim order is nondeterministic. Whether it
   dominates relative to metric inversion/RHS arithmetic is **unmeasured**.
4. **Barriers per residual pass.** The Q3 residual computation runs as several
   separate whole-vector passes with `@sync` barriers; each pass pays a
   full-vector memory sweep. The residual bucket is 7% of α3 T1, so eliminating
   it entirely bounds the gain at ~1.076×.
5. **Incomplete accounting.** ~21% of α3 wall time is not attributed to any
   phase, and `direction_seconds` overlaps children. We cannot optimize what
   we do not measure.
6. **Wrong axis for some LP-like cases.** Where no parallel path is selected or
   the parallel Gram share is small, inner threading adds overhead; but this is
   a policy/threshold question (see `plan.jl:318-322`) to be settled by
   instrumentation, not assumed.
7. **Static-partitioning composability hazard.** Coarse `@threads :static`
   partitioning (a proposed fix for atomic-claim contention) is unsuitable for
   nested/non-primary-thread calls: nested parallel regions, task migration,
   and blocking when the solve runs inside an outer task pool. Scratch must
   belong to a worker range, not a migrating task's transient `threadid()`.
8. **SIMD type assumption.** `_hkm_vec4_full_metric!` hardcodes
   `MultiFloatVec{4,Float64,4}` under a broader MultiFloat dispatch; do not
   broaden SIMD eligibility without explicit x4/type qualification.

## 2. Proposal: a two-level execution model

### Level A — throughput (the proven win)

Independent solves/scans run as **separate processes with one Julia thread and
BLAS=1**, scheduled against a shared core and memory budget. Measured on the
cluster for the same LP: 282 → 559 → 893 solves/node-hour at ppn 8/16/32, i.e.
~3.2× from 8 to 32 cores, versus a flat/negative inner-thread curve.

Requirements:
- a batch/scheduler API (queue of independent solves, per-process memory
  accounting, result collection) rather than one solve per invocation;
- construction amortization for scans (see the bulk affine builder, 7.6× on the
  CSDR build) so per-solve frontend cost does not dominate;
- no shared mutable solver state across processes; each process owns its
  provider workspace.

### Level B — latency (only where there is parallel work)

Inner-thread parallelism is enabled by a measured **parallel fraction**, not by
a user thread count alone:
1. enforce an explicit **effective worker budget** in every workspace
   (`plan.selected_threads` threaded into Q3/provider construction; execute
   exactly N tasks, not `nthreads()`);
2. use **coarse static partitioning aligned to the SIMD width** (multiples of 4)
   instead of per-block atomics, with a work-stealing fallback only when the
   partition is heterogeneous;
3. **fuse consecutive Q3 residual passes** to remove barriers and full-vector
   sweeps (same arithmetic order, fewer passes);
4. consider **batched small-block kernels** (SoA over blocks) to raise
   arithmetic intensity for the 4200 identical Q3 blocks;
5. fix phase accounting first: split the Q3 "factor" bucket into metric prep,
   panel transform, Gram/SYRK, numeric factor, homogeneous solve; remove the
   `direction_seconds` overlap; attribute the missing ~21%.

### Level C — cross-cutting

- **NUMA/affinity** on the cluster: first-touch placement, per-thread scratch,
  avoid false sharing on adjacent block writes.
- **Validity gate for scaling claims**: every timed run records
  `Percent of CPU this job got`. A low value is **not** by itself evidence of
  contention: under the fitted α3 model even an uncontended p=32 solve averages
  only ~2.33 CPU-equivalents because the serial share dominates. Compare CPU%
  against the model's expected CPU-equivalents and require independent
  node-load/allocation/affinity evidence before claiming contention; otherwise
  attribute the gap via per-phase instrumentation.

## 5. Implementation progress (2026-09-09)

- **Phase 1 — exclusive Q3 sub-phase timing (merged `0b7bc99`).**
  `Q3EpochTimings` records metric-preparation, numeric-factor and
  homogeneous-solve wall times plus the effective worker count per epoch;
  they accumulate into `ProductHSDPhaseTimings` as exclusive children of
  `kkt_factorization_seconds` (`q3_metric_seconds`, `q3_factor_seconds`,
  `q3_homogeneous_seconds`, `q3_epochs`, `q3_workers`) and into the timings
  snapshot. Zero-allocation additive writes; no numerical change. Partitioned
  regression 4197/196/4622+1 all pass.
- **Phase 2 — admitted Q3 worker budget for task-based loops (merged `f0325a4`).**
  `settings.limits.threads` sets a process-wide budget at state construction;
  `_q3_workers()` caps it by the Julia pool and now drives the task-based HKM
  scalar and vec4 loops, which previously expanded to `Threads.nthreads()`.
  `@threads :static` loops still use the pool (a budget below the pool needs
  `--threads` to match; one process per configuration). No numerical change.
  Verified: pool=4/budget=2 → 2 workers; regression unchanged.
- **Next:** run α3 with the new instrumentation to attribute the previously
  unaccounted ~21% and the Q3 factor bucket, then target the measured serial
  loops (Phase 3) and implement the persistent process-worker pool (Phase 4,
  Astra's first recommendation).

## 6. Astra design-review outcome (2026-09-09)

Verdict: **architecture sound, causal claims overstated**. The fitted `s`
values are effective scaling parameters, not intrinsic limits; several proposed
causes (atomic claims, bandwidth, false sharing, GC) remain unmeasured
hypotheses. Ranked changes from the review:

| Rank | Change | Expected effect | Risk / measurement |
| --- | --- | --- | --- |
| 1 | **Persistent warmed one-thread process workers** for queued independent solves (the campaign currently pays Julia startup + warmup per item: `shard_template.pbs:49-56`, `run_case.jl:19-36`) | Amortizes startup/JIT/warmup on the proven throughput axis; no isolated-solve latency gain | Retained memory, state leakage, certification failures; compare certified jobs per reserved node-hour including startup, monitor RSS |
| 2 | **Exclusive phase timing + truthful receipts** (`performance_trace.jl:158-205`, `phase_timings.jl:39-51`, Q3 factor body `:1385-1407`) | No direct speedup; stops optimizing the wrong component | Instrumentation overhead; keep inclusive `direction_seconds`, add exclusive children/remainder |
| 3 | **One budgeted coarse-range executor** for Q3 + provider (`product_cone_hsd.jl:234,258-267,546-548`; `fixed_trace_q3.jl:949-950`) | Enforces limits, removes atomic claiming, composable scheduling | Nested oversubscription, migrated-task scratch, cancellation; test admitted budgets inside larger pools |
| 4 | **Make the existing HKM SIMD usable for equality-prefixed layouts** (`ext/SDPXMultiFloatLinearAlgebraExt.jl:1425-1433,1475-1487`) | Accelerates a currently scalar fallback; benefits T1 as well as threaded runs | Gather actual offsets, preserve per-lane arithmetic, count real SIMD batches |

**Implemented (2026-09-09, merged `15c281f`):** the vec4 metric is now
offset-aware; bit-identical to the scalar kernel on equality-first offsets.
Measured kernel gain **2.75×** (0.0044 s → 0.0016 s per 4200-block pass). The
metric is only ~4% of the α3 factor bucket (4.4 ms vs 117 ms per iteration), so
the bounded solver-level impact is ~0.7% — correct and positive, but not the
main lever. The fix also exposed a latent bug in the never-executed path
(`all(isfinite, ::Vec4)` has no method), now fixed with a lane-wise check.
| 5 | **Fuse narrowly adjacent residual operations** (`fixed_trace_q3.jl:1070-1092,1106-1115,1150-1158`) | Removes barriers/dispatch/writes; residual is 7% of T1, so the bound is ~1.076× | Free variables, generic-layout fallback, aliases; verify all paths and fixed-worker parity |

**First change to implement:** persistent warmed one-thread process workers (not
a solver-wide task graph or public batching API). Accept gate: same reserved
cores/memory/problem stream/environment/certification; compare current
fresh-process launcher with persistent workers over repeated batches; count
only independently certified completions including startup/collection/failures;
accept a reproducible **≥2% throughput improvement** with bounded retained
memory and no state-isolation/certification regression; otherwise keep the
existing one-thread process scheduler. Keep execution management outside the
numerical core.

**Architecture comparison:** two-level process throughput + gated inner-thread
latency is the best fit; a persistent process pool is the recommended
throughput implementation; a persistent Julia task pool needs budget
enforcement/ownership qualification first; a task graph is a poor initial fit
due to numerical dependencies; multiple solves in one process exposes shared
GC/cache/ownership hazards (reconsider after R1/R2 concurrency qualification);
parallel sparse factorization ranks below the above until LP profiling shows a
large enough factor share (a 20% overall gain with a 4× faster factor needs the
factor to be ≥22.2% of time).

**Additional hazards:** `@threads :static` is unsuitable for arbitrary nested
calls; scratch must belong to a worker range, not a migrating task's
`threadid()`; `_hkm_vec4_full_metric!` hardcodes `MultiFloatVec{4,Float64,4}`
(do not broaden without x4 qualification); `fixed_trace_q3_core_preflight` is
diagnostic-only, not a memory-admission bound; whole-job CPU% includes
startup/warmup; the campaign's bare `wait` does not aggregate child failures
into FAIL, so throughput must count validated result records.
- **Digest caveat**: the measured cluster α3 runs gave the same digest at
  1/8/16/32 threads (`c354cf07`) while the local Mac gave `3a7833` at 1 and 4
  threads, so the machine/runtime — not the thread count — changed the
  trajectory here. Bit-identity/A-B gates must still compare within a fixed
  configuration; do not assume either outcome.

## 3. What NOT to do

- Do not add more threads to LP-like workloads (no parallel work; measured
  regression).
- Do not treat the Q3 "factorization" bucket as serial LU; it is mostly
  parallel Gram work.
- Do not relax tolerances or drop constraints to make parallel runs "match".
- Digest caveat: the cluster α3 campaign produced the SAME digest (`c354cf07`) at
  1/8/16/32 threads, and the local 1/4-thread runs both produced `3a7833`; the
  machine difference (not the thread count) changed the trajectory. Do not
  assume thread count changes the digest, and do not assume it cannot; compare
  per fixed configuration and record the digest.

## 4. Measurement plan for any candidate

1. Instrument per-phase **worker count and wall time** (not just totals).
2. Fresh process per configuration; warm-up outside timing; ≥3 repetitions.
3. Record CPU%, allocations, peak RSS, iterations, objective, residuals,
   certificate, digest; reject contended runs.
4. Accept an inner-thread change only if the **total certified time** improves
   at a thread count that actually has parallel work, with no 1-thread
   regression and no allocation/RSS increase.
5. Accept a process-level change only if throughput (solves/node-hour) improves
   at fixed reserved cores with unchanged per-solve certification.
