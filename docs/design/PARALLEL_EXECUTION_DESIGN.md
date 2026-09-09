# SDPX parallel-execution design (diagnosis + proposal)

Date: 2026-09-09. Evidence-based design note. It does not change any default;
it records why thread scaling is poor and what a better execution model looks
like. Numbers are measured unless marked as an estimate.

## 1. Diagnosis: the poor efficiency is structural, not a tuning gap

### 1.1 Two workloads, two different causes

| Workload | p=1 | p=4 | p=8 | p=16 | p=32 | implied serial fraction s | Amdahl cap at 32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LP `random_large` (m=400,n=1200), cluster | 6.737 s | — | 6.573 s | 6.730 s | 7.960 s | **~0.97** | **1.03×** |
| CSDR frozen α3 Float64x4, local | 41.313 s | 23.056 s | — | — | — | **~0.41** | **2.33×** |

- The LP has essentially **no parallelizable work**: the cone is the orthant
  (diagonal), so there is no Gram/metric assembly; the solve is sparse
  factorization plus vector sweeps. Threading is the wrong axis; at p=32 the
  overhead makes it 18% *slower*.
- CSDR α3 does scale (1.79× at 4 threads) but is capped near 2.3× by a ~41%
  serial fraction.

### 1.2 Where α3's time goes (1 thread, solver 41.3 s, phase trace)

| Phase | seconds | share | note |
| --- | --- | --- | --- |
| `setup` | 1.216 | 2.9% | serial |
| `residual_seconds` | 2.911 | 7.0% | mostly serial sweeps |
| `scaling_seconds` | 0.104 | 0.25% | serial |
| `direction_seconds` | 28.397 | 68.7% | **overlaps its children** |
| — `kkt_factorization_seconds` | 12.335 | 29.9% | charges metric prep + panel transform + SYRK + LU + homogeneous solve, not pure LU |
| unaccounted | ~8.7 | ~21% | the canonical phase projection omits buckets |

105 factorizations (`mfla_pivoted_ldlt`). The "factorization" bucket is
dominated by the **triangular Gram assembly** (exactly 7,585,200 MACs per
assembly), which *is* parallelizable; the 42×42 LU is negligible. The
`direction_seconds` bucket overlaps its children, so adding phase times
double-counts.

### 1.3 Design defects that make efficiency worse than the Amdahl bound

1. **Thread-budget mismatch.** Fixed-trace Q3 hot loops and
   `predictor_corrector.jl` schedule with `Threads.nthreads()` rather than the
   admitted `plan.selected_threads`. A requested `Limits(...threads=4)` does not
   establish four workers, and concurrent solves each expand to the full Julia
   pool.
2. **Atomic per-block claiming.** The HKM loops claim one block (scalar) or four
   blocks (vec4) per `atomic_add!`; with thousands of blocks this is a
   contention point, and the claim order is nondeterministic.
3. **Barriers per residual pass.** The Q3 residual computation runs as several
   separate whole-vector passes with `@sync` barriers; each pass pays a
   full-vector memory sweep.
4. **Incomplete accounting.** ~21% of α3 wall time is not attributed to any
   phase, and `direction_seconds` overlaps children. We cannot optimize what we
   do not measure.
5. **Wrong axis for LP-like problems.** For diagonal cones there is no
   parallelizable arithmetic; only process-level parallelism helps.

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
  `Percent of CPU this job got`; a run whose CPU% is far below the requested
  thread count is contended and must not be reported as scaling evidence (the
  first t32 cluster run used ~1.4 cores while requesting 32).
- **Digest caveat**: threading changes Gram reduction order, so trajectories
  differ across thread counts; bit-identity/A-B gates must compare within a
  fixed thread count.

## 3. What NOT to do

- Do not add more threads to LP-like workloads (no parallel work; measured
  regression).
- Do not treat the Q3 "factorization" bucket as serial LU; it is mostly
  parallel Gram work.
- Do not relax tolerances or drop constraints to make parallel runs "match".
- Do not compare digests across thread counts.

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
