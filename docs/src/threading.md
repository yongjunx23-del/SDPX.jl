# Threading Guide

## Starting and limiting threads

Start Julia with `-t N` or `JULIA_NUM_THREADS=N`. The process thread count is
an upper bound; `SolverOptions(threads=...)` or the public `threads=...`
keyword may select fewer threads for one solve:

```bash
JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 julia --project=. solve_problem.jl
```

```julia
result = solve(c, A, C, B, b; threads=4)
```

The execution plan can reduce that request when the arithmetic or estimated
work does not justify parallel scheduling.

| Arithmetic | Solver scheduling policy |
|---|---|
| `Float64` | Threaded for sufficiently large block and Schur work; small latency-bound cases remain serial. |
| `Float64xN` | Threaded when the type is an immutable fixed-width arithmetic; extended arithmetic crosses over earlier than Float64. |
| `BigFloat` | General native MPFR phases use one solver thread. Exact singleton-local `2x2` arrows and all-local 2x2 cell systems with explicit equalities may use ownership-safe block tasks and disjoint Schur/Gram-tile workers. |
| Unknown scalar type | Serial unless the kernel layer explicitly marks it safe. |

## Scheduling and synchronization

PSD blocks are normally assigned to workers by longest-processing-time (LPT)
greedy partitioning. The weights estimate factorization and Schur work, so
heterogeneous bootstrap blocks are balanced by cost rather than by block
count. One measured exception applies to uniform Float64x4 reduced-arrow
systems: with at least 256 identical `2x2` blocks and at most 32 active block
tasks, contiguous ownership improves cache and pointer locality without
creating load imbalance. Wider teams and every heterogeneous system retain
LPT. Executed diagnostics report both `fine_grained_block_tasks` and
`fine_grained_block_partition`.

The principal threaded paths avoid locks and atomics:

- dense blocks accumulate into one private Schur matrix per active LPT bin;
- sparse blocks transform only active variables and own their compact pair
  values;
- exact-arrow problems write disjoint local and coupling blocks, with a small
  private global-global accumulator per bin;
- residuals, predictor/corrector right-hand sides, direction recovery, local
  arrow factorization, and local arrow solves reuse the same cached schedule.

Reductions happen in a fixed bin order. At a fixed thread count the result is
deterministic. Changing the thread count may change the floating-point
reduction order and therefore the last few bits.

## Block-local crossovers

Residual construction, block Cholesky, predictor/corrector right-hand sides,
and direction recovery use the cached LPT schedule when either the historical
256-block threshold is met or estimated cubic block work is large enough:

```text
Float64-family sum(k[l]^3):          at least 1,000,000
fixed-width extended sum(k[l]^3):   at least   100,000
```

This dimension-aware route matters for dense lattice SDPs that have a small
number of moderately large blocks. A maintained dense workload has 32 blocks of dimensions
23--74 and `sum(k[l]^3) = 3,977,757`; the former count-only policy therefore
left all block-local phases serial. On one node125 run with 16 OpenBLAS
threads, the retained scheduler measured median solve times of 36.708 /
32.158 / 29.780 / 27.424 seconds at 4 / 8 / 16 / 32 Julia workers. All runs
used 28 iterations and 119 backtracking contractions and passed the same
original-coordinate certificate. A direct old/new same-node A/B at 32 workers
improved median solve time from 32.062 to 28.438 seconds (11.3%).

Task-local dense Schur accumulators remain memory-capped. The default share is
15% of scheduler-aware available memory. A deliberately narrow `Float64`
policy raises it to 25% only when all of these conditions hold:

```text
Schur dimension:       at least 4,096
PSD block count:       at least 16
requested workers:     at least 16
available memory:      at least 16 GiB
```

On that workload with a 28 GiB explicit ceiling this selects 25 of the 32
possible bins. A same-node controlled run reduced median Schur assembly from
8.140 to 6.701 seconds and the stable solve from 27.449 to 25.965 seconds.
Increasing the share to 35% produced only a noisy 1.7% median change, made the
stable second repetition 2.1% slower, and raised peak RSS by 0.44%; it was
rejected. Fixed-width extended and BigFloat arithmetic retain the 15% rule.

## Schur crossovers

Thread launch and reduction are measurable costs. Dense Schur assembly uses a
conservative arithmetic-work crossover:

```text
Float64 estimated Gram work:          at least 4,000,000
fixed-width extended estimated work:  at least   100,000
```

The automatic plan also keeps Float64 SDP models with only `1x1`/`2x2`
blocks and fewer than 1,000 variables serial. These are policy thresholds,
not universal hardware constants; benchmark changes on the target cluster
before recalibrating them.

When a block kernel produces only the lower triangle, the reducer also reads,
initializes, and accumulates only that triangle. Work is split by triangular
area across column-major ranges. For other fixed-width paths, the established
full-matrix reducer remains in use where it benchmarks faster.

The Float64x4 reduced-arrow path also has phase-specific worker and tile
crossovers. A narrow shared triangle may expose fewer independent tiles than
requested workers, while its thousands of local `2x2` blocks can still use a
wider team. SDPX therefore selects Schur/SYRK workers separately from short
block-local tasks and reports the executed value as `schur_threads`. On the
measured 1,700-block, 144-shared-variable reduced-arrow model, 48--95 Schur workers use
an eight-column tile only when the default twelve-column triangle has fewer
than two jobs per worker. Requests of 96 or more use at most 64 Schur workers
for this narrow geometry and at most 32 tasks for synchronization-sensitive
block phases. Wider shared systems retain the full request.

The same path uses a separate `factor_threads` crossover for the reduced
shared Cholesky. Orders 128--256 use a sixteen-column lower-only factor with
one cached reciprocal per pivot and four-lane SIMD across independent panel
rows. Trailing panel solves and triangular updates use at most eight workers;
other dimensions and arithmetic types keep their established factor. On the
order-144 EPYC benchmark, the previous factor took 15.891 ms, the new serial
kernel took 10.253 ms, and the eight-worker kernel took 6.890 ms. In the full
41-iteration medium solve, cumulative factor time fell from 0.671 to 0.140
seconds. Executed diagnostics report the selected factor width.

Dense non-arrow Float64x4 refinement repeatedly applies the symmetric Schur
matrix after a lower-only build. For dimensions of at least 1,024 and more
than one requested worker, SDPX partitions complete output-row ranges among
at most `ceil(dimension / 256)` workers. Each task reads the stored triangle
and owns every scalar it writes; no task-local matrix or reduction is needed.
The scalar hot loop allocates zero bytes. Aliased input/output vectors fall
back to the established multiplication path.

The opt-in dense `Float64x4` mixed KKT fallback uses `Float64x2` only after
the Float64 residual guard fails. Conversion and first-touch initialization
own disjoint linear ranges; blocked Cholesky panel solves own disjoint rows;
trailing and equality SYRK tasks own complete lower-triangular tiles; and
`L^-1 B` plus recovery products own disjoint right-hand-side columns or
output ranges. This path respects `SolverOptions.threads`, leaves BLAS at one
thread, and allocates its 0.596 GiB dense-workload workspace lazily. Native
`Float64x4` remains the final serial fallback.

## BigFloat policy

Native `BigFloat` uses the serial owned-storage path for general models even if
Julia was started with multiple threads. Exact singleton-local `2x2` arrow
models and block-diagonal 2x2 cell models whose Schur variables are all local
and whose only global coupling is through explicit equalities are the current
native exceptions.

The reason is solver-specific: a `BigFloat` is mutable, ordinary
`zeros(BigFloat, ...)`/`fill!` storage can alias the same object, and
arbitrary-precision task-local matrices grow quickly with worker count. SDPX
therefore uses independently owned workspace entries plus allocation-reusing
MPFR scalar kernels. For the reduced-arrow exception, each preparation task
owns a disjoint block workspace and two panel rows, and each SYRK task owns a
complete lower-triangular output tile. Inputs are read-only and no writable
BigFloat object crosses tasks. The equality specialization likewise assigns
complete local row blocks during forward/back substitution and complete lower
Gram tiles during `Btil' * Btil`; it never allocates one full Gram per worker.

The current fixed-trace release campaign intentionally stops at 32 workers
(J40: 1/2/4/8/16/32; J80: 8/16/32, BLAS=1). The wider measurements below are
archived scaling evidence from earlier campaigns, not current template
defaults or recommendations.

The two equality matrix-vector products in each KKT solve also partition
complete output ranges. Every dot product keeps the serial reduction order,
and a work crossover caps the task count when the panel cannot amortize
startup. Fine-grained all-local block, triangular, GEMV, predictor/corrector,
line-search, and update phases are additionally capped at 64 ownership tasks;
the disjoint equality Gram tiles may use the full requested width. On a
512-bit 16,400-by-230 synthetic reduced-arrow-shaped system, the equality
Gram scaled from 58.618 seconds at one worker to 1.339 seconds at 128 workers
on a dual-socket EPYC 7742 node while retaining a `5.64e-152` relative KKT
residual. Use `numactl --interleave=all` on that eight-NUMA-domain node and
keep BLAS/OMP at one thread for this MPFR path.

The cap comes from an end-to-end certificate run, not the synthetic Gram
alone. On the J40 BigFloat512 model, a uniform 128-worker schedule took
495.811 seconds, versus 368.704 seconds at 64 workers. The phase-aware cap
reduced the 128-worker time to 425.880 seconds and peak RSS by 6.6%, while
reproducing the 158-iteration objective, gap, residuals, PSD margin, off-grid
residual, and disk certificate bit-for-bit. Because 64 remains 15.5% faster,
a 96-worker check was also run; it took 398.303 seconds, still 8.0% slower than
64. Those results motivated measuring rather than assuming the widest
allocation; current fixed-trace templates do not request more than 32 workers.

The 64-worker path was separately validated at a fixed 1,024-bit precision.
It passed the J40 physical certificate in 157 iterations and took 553.959
seconds, with 4,268,480 KiB peak RSS. Equality Gram (115.491 seconds) and the
170-by-170 equality factorization (42.838 seconds) became the main
precision-scaling costs. The corresponding 512-bit solve remained faster at
368.704 seconds and had a tighter terminating relative gap on this once-rounded
`Float64x4` input, so 512 bits remains the recommended model-specific setting.

Use `Float64x4` or another fixed-width `MultiFloats` type when its precision
and Float64 exponent range are sufficient and broader solver phases need
multicore speedup. The experimental `mixed_precision_kkt=:float64x4` mode is
another path for exact singleton-local `2x2` arrows: it constructs and factors
the reduced shared system in Float64x4, checks residuals and refines in
BigFloat, and automatically falls back to native BigFloat if correction is
unsafe. Only the Float64x4 panel/factorization uses the requested workers; the
native MPFR phases remain serial. This mode did not deliver a clear
end-to-end improvement on the medium reduced-arrow benchmark and therefore remains off
by default.

For general non-arrow arbitrary precision, run independent BigFloat instances
as separate processes or scheduler-array elements.

On the medium exact-arrow model at 256 bits, the ownership-safe native
path measured 205.202 / 191.701 / 110.741 / 86.752 seconds with
1 / 2 / 4 / 8 Julia threads. The matched one-thread legacy path took
280.011 seconds. All thread counts produced the same 41-iteration certified
result. This scaling is specific to the exact reduced-arrow structure; it is
not evidence for enabling arbitrary threaded MPFR loops.

## Phase-aware BLAS threads

Block-parallel phases issue many small linear-algebra calls. SDPX temporarily
sets BLAS to one thread there so `Julia workers × BLAS workers` does not
oversubscribe the node. Dense KKT Cholesky is a single large operation, so the
solver restores a bounded BLAS width for that phase and restores the caller's
setting afterward, including on exceptions.

### Apple Accelerate

On macOS, AppleAccelerate.jl can replace OpenBLAS through
libblastrampoline. Load it explicitly before solving:

```julia
using SDPX
using AppleAccelerate

AppleAccelerate.set_num_threads(1)
@assert AppleAccelerate.get_num_threads() == 1
```

Use `AppleAccelerate.set_num_threads`, rather than
`LinearAlgebra.BLAS.set_num_threads`, when Apple Accelerate is loaded.
Accelerate uses `BLASSetThreading`; the ordinary libblastrampoline setter does
not switch it into strict single-threaded mode. AppleAccelerate remains an
optional dependency because loading it changes the process-wide BLAS/LAPACK
backend for every Julia package in the process.

### Linux BLAS backends

Linux deployments can benchmark alternative libblastrampoline backends in a
release-specific environment. Load the backend before SDPX, then use
libblastrampoline's standard thread controller:

```julia
using MKL                 # or: using BLISBLAS
using LinearAlgebra
using SDPX

LinearAlgebra.BLAS.set_num_threads(16)
```

Do not assume that a vendor backend is faster on every CPU. On the cluster's
dual-socket AMD EPYC 7742 node, equal eight-core, eight-iteration dense-workload
jobs took 10.816 seconds with OpenBLAS, 13.018 seconds with MKL, and 21.999
seconds with BLIS. OpenBLAS therefore remains the selected backend there.
MKL and BLIS are benchmark-only environment dependencies, not SDPX
dependencies.

Do not start by assigning both Julia and BLAS every core. Use one core per
Julia thread, set a realistic process limit, and measure complete iteration
time. Nested or concurrent solves share the process-global BLAS setting and
are not a supported way to obtain parallelism; use separate processes.

### Medium reduced-arrow affinity and NUMA result

The measured Float64x4 reduced-arrow model has 1,700 uniform `2x2`
blocks and a 144-column reduced shared system. On a dual-socket, eight-NUMA-
domain AMD EPYC 7742 node, keep BLAS at one thread and start Julia with an
exact compute pool (`--threads=N,0 --gcthreads=1,0`). Setting
`JULIA_EXCLUSIVE=1` is important: at 64 workers it reduced the median solve
from 5.981 to 4.791 seconds. Without it, 56 of 62 sampled workers migrated
between CPUs and visited 9.32 CPUs each on average; exclusive affinity pinned
every sampled worker to one CPU and reduced voluntary context switches from
2.45 million to 0.72 million.

Bind the selected CPUs and allocate memory locally. A 64-worker run bound to
CPUs 0--63 and NUMA nodes 0--3 took 5.981 seconds before the affinity gain;
forcing interleaving over the same nodes took 6.304 seconds and was rejected.
At 32 exclusive workers, contiguous block ownership improved the same-node
median from 5.010 to 4.555 seconds. At 64 workers, LPT was slightly faster
than contiguous ownership (4.789 versus 4.866 seconds), so the automatic
contiguous crossover stops at 32 tasks. The final pre-factor scaling sweep
measured 44.293 / 23.382 / 12.969 / 7.526 / 4.880 / 4.709 / 4.614 / 12.444
seconds at 1 / 2 / 4 / 8 / 16 / 32 / 64 / 128 workers. The 128-worker process
really reached all 128 workers, but averaged only 13.53 cores, incurred 24.89
million voluntary context switches, and was 2.70 times slower than 64. The
short phases and 144-column triangle do not contain enough work for that team
width.

The later factor A/B used 32-worker pools and reduced the two process medians
from 4.710/4.793 to 3.422/3.418 seconds. Those results do not imply that a
larger Julia pool is free: even an eight-task factor can be scheduled across
all NUMA domains of a 128-thread process. Launch the smallest pool that wins
the full scaling sweep; a large scheduler reservation is not evidence that
all reserved cores helped the solve.

The retained narrow-geometry solver cap keeps the scheduling decision based
on the original request but limits actual workspace and numerical work to 64
workers at requests of 96 or more. For a 128-thread Julia runtime this
preserves 32 contiguous fine-grained bins and 64 Schur tasks. An alternating
same-node A/B reduced the combined solve median from 12.216 to 11.839 seconds
and allocation from 216.41 to 166.54 MB per solve; a genuine 64-thread control
was neutral. Diagnostics distinguish `threads=128`, `effective_threads=64`,
`fine_grained_block_tasks=32`, `schur_threads=64`, and `factor_threads=8`.

The remaining 128-pool loss was predominantly Julia worker wake-up overhead.
With otherwise identical retained code, setting
`JULIA_THREAD_SLEEP_THRESHOLD=10000000` before Julia starts reduced a
20-sample full-solver median from 11.635 to 3.409 seconds. That was within
1.0% of permanently awake workers, but used about 12.5 CPU-core equivalents
over the complete process instead of 113. The 10 ms processes still reached
124--126 cores during parallel bursts and cut voluntary context switches from
about 27.5 million to 108--115 thousand. All 35 comparison rows were
`Optimal` in 41 iterations with bit-for-bit identical printed objectives,
residuals, gaps, and PSD certificates.

The threshold is expressed in nanoseconds and is consumed by the Julia
runtime before SDPX loads. It is therefore a launch recommendation for this
measured EPYC topology, not solver-global state or a portable default. Use it
only in an exclusive allocation, and repeat the exact-pool sweep on materially
different hardware or problem geometry. Permanent spinning is rejected for
routine use because it consumes the entire node during serial work without a
meaningful solve-time advantage over 10 ms.

These are topology-specific measurements, not a reason to hide resource
usage. Verify each new node with process RSS, per-thread CPU affinity, active
worker counts, BLAS width, and `numastat` samples. A large reservation can
improve queue placement on this site, but it does not make unused cores part
of the numerical speedup.

### Final exact-pool Float64x4 audit

The retained exact-pool campaign reserved all 128 physical cores of one
dual-socket EPYC 7742 node, but launched a separate Julia process for each
measured pool. Every process used `--threads=N,0 --gcthreads=1,0`,
`JULIA_EXCLUSIVE=1`, one BLAS/OMP worker, three complete warm-ups, and five
recorded solves. The finite 10-millisecond worker-sleep policy was used for
the table below. The 32-worker row combines two independently launched
five-solve controls; other rows contain five solves.

| Julia workers | Solve median (s) | Speedup vs 1 | Whole-process mean cores | Peak sampled cores | Maximum active workers | Peak RSS (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.112 | 1.00x | 1.00 | 1.00 | 1 | 2.332 |
| 2 | 22.954 | 1.92x | 1.50 | 1.99 | 2 | 2.307 |
| 4 | 12.450 | 3.54x | 2.05 | 3.98 | 4 | 2.358 |
| 8 | 6.946 | 6.35x | 2.57 | 7.95 | 8 | 2.582 |
| 16 | 4.275 | 10.32x | 3.09 | 15.89 | 16 | 2.955 |
| 32 | 3.286 | 13.42x | 4.15 | 31.79 | 32 | 3.041 |
| **48** | **2.967** | **14.87x** | **5.25** | **45.38** | **48** | **3.144** |
| 64 | 3.074 | 14.35x | 6.70 | 62.53 | 64 | 3.304 |
| 96 | 3.540 | 12.46x | 9.42 | 94.87 | 96 | 3.291 |
| 128 | 3.403 | 12.96x | 12.39 | 125.78 | 128 | 3.359 |

The mean CPU column covers package/model loading, compilation, serialization,
and serial regions as well as all warm-up and measured solves; it is not a
parallel-kernel utilization claim. The peak samples and per-thread affinity
records independently prove that every requested worker was created, pinned,
and active. At 48 workers, 48 Julia compute threads were pinned one per CPU
over CPUs 0--47. At 128, all 128 were pinned over CPUs 0--127. The solver's
diagnostics additionally report `effective_threads=64`,
`fine_grained_block_tasks=32`, `schur_threads=64`, and `factor_threads=8` for
the over-wide 96/128 narrow-arrow requests; runtime/model phases may still
activate the complete Julia pool. Use a 48-thread runtime when solving this
geometry rather than relying on the defensive wide-pool cap.

NUMA sampling confirms that placement requests affected real memory. The
48-worker socket-local run placed about 2,696 / 177 / 107 / 27 MiB on nodes
0--3 (3,009 MiB total at the sampled peak). The 128-worker interleaved run
placed 349--474 MiB on each of nodes 0--7 (3,231 MiB total). A prior 64-worker
same-domain interleave experiment was 5.4% slower than local first touch, so
the recommended 48-worker launch uses `--physcpubind=0-47 --membind=0-3`.

The one-worker phase split was dominated by Schur assembly (24.540 seconds,
55.6%), predictor (6.421 seconds), corrector (6.681 seconds), and residual
work (4.417 seconds). At the 48-worker optimum those phases were 0.741,
0.371, 0.513, and 0.222 seconds. Schur therefore scales well but the many
short predictor/corrector, residual, line-search, and update regions impose
the limiting barriers. Increasing from 48 to 64 makes Schur 80 milliseconds
faster but adds more synchronization and NUMA cost elsewhere; 96/128 lose
further despite genuinely activating those pools.

Three final structure-specific changes were retained after this sweep. Lazy
legacy Schur partials reduced the 48-worker solve from 2.982 to 2.940 seconds
and allocation from 148.46 to 118.08 MiB. Cache-hot singleton-local factor
preparation then removed 99.96% of the local-elimination phase and reduced a
14-solve median from 2.9268 to 2.9042 seconds. Finally, allocation-free
`MultiFloatVec` RHS/recovery kernels reduced predictor/corrector linear solves
by 6.76%/8.88% and a separate 14-solve median from 2.9134 to 2.8952 seconds.
Each retained A/B used reverse ordering, all 48 workers, BLAS=1, and exact
certificate comparison. Contiguous 48-bin ownership (0.52%), a 100-millisecond
sleep policy, and an eight-block recovery grouping were rejected because
their gains were absent or negative under repeat measurement.

Recommended PBS launch separation for this site:

```bash
# Reserve the complete node for predictable placement, but create only the
# measured useful Julia pool for this 1,700-block / 144-shared-column model.
export JULIA_EXCLUSIVE=1
export JULIA_THREAD_SLEEP_THRESHOLD=10000000
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

numactl --physcpubind=0-47 --membind=0-3 \
  julia --threads=48,0 --gcthreads=1,0 --project=. solve.jl
```

Executed solve diagnostics expose `arrow_linear_solve` in addition to the
effective/block/Schur/factor widths, so resource audits can distinguish the
Float64x4 SIMD singleton route from the scalar fallback.

## Measured scheduler behavior

The latest isolated benchmarks used Julia 1.12.6 on an Apple M4 process
exposing four hardware threads, with BLAS restricted to one thread. The
eight-worker rows intentionally oversubscribe that host and are not
eight-core results.

| Arithmetic and case | 1 worker | Best measured | Speedup |
|---|---:|---:|---:|
| Float64 dense, `L/m/k=8/96/6` | 0.199 ms | 0.196 ms at 4 | 1.02x |
| Float64 dense, `8/512/10` | 6.561 ms | 3.998 ms at 2 | 1.64x |
| Float64x4 dense, `8/128/8` | 137.318 ms | 33.669 ms at 8 | 4.08x |
| Float64 sparse arrow, `blocks/shared=512/24` | 0.254 ms | 0.230 ms at 8 | 1.11x |
| Float64x4 sparse arrow, `256/16` | 5.415 ms | 1.453 ms at 8 | 3.73x |

The small Float64 crossover held the build near its serial 0.20 ms value and
reduced calling-task allocations to 768 bytes. Maximum relative Schur errors
were `2.67e-16` for dense Float64, `3.57e-65` for dense Float64x4,
`1.67e-15` for sparse Float64, and `1.33e-64` for sparse Float64x4.

Revalidate these historical crossover values with a versioned catalog through
the current [benchmark protocol](benchmarks.md) before changing a default.

## Memory limits

Dense task-local partials still occupy a full `m × m` matrix per active bin.
The bin planner caps this storage from current available memory. Available
memory is the minimum usable value reported by:

1. host free memory;
2. Linux cgroup v2 (`memory.max`/`memory.current`) or v1 counters;
3. the optional `SDPX_MEMORY_LIMIT_BYTES` environment ceiling.

Accepted explicit units include `B`, `KB`, `KiB`, `MB`, `MiB`, `GB`, `GiB`,
`TB`, and `TiB`. SDPX keeps additional headroom and may choose fewer workers
or reject optional panel packing even when the requested fraction appears to
fit.

For PBS/Slurm jobs, set the explicit ceiling slightly below the scheduler
request when the cgroup does not expose a reliable limit. See the
[cluster workflow](cluster-workflow.md).

## Remaining multicore limits

- Float64x4 reduced-arrow factors of order 128--256 now use cached
  reciprocals, SIMD panel rows, and at most eight trailing-update workers.
  Larger reduced factors and generic native extended-precision dense
  Cholesky still need a scalable factorization backend.
- Generic non-arrow Schur/KKT factorization is dense and remains a dominant
  phase on large lattice problems. The Float64x4 refinement matvec is
  threaded, but native extended-precision dense Cholesky is still serial.
- Dense task-local Schur matrices are not packed; changing their workspace
  layout could nearly halve partial-storage memory.
- Worker teams are created per region; persistent teams could remove
  approximately 20–100 microseconds of repeated launch cost.
- Automatic topology discovery, portable NUMA first-touch policy, and
  distributed factorization have not been implemented. Explicit socket-local
  binding is validated only for the medium EPYC reduced-arrow profile above.
