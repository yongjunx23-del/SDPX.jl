# Schur Scheduler Benchmark

Date: 2026-07-25

## Environment

- Julia 1.12.6
- Apple M4
- `Sys.CPU_THREADS == 4`
- 16 GiB physical memory
- ILP64 OpenBLAS, restricted to one BLAS thread during measurement
- Julia launched with `JULIA_NUM_THREADS=8`

The 8-worker rows intentionally exercise the requested boundary, but they
oversubscribe the four hardware threads exposed to this process and should not
be interpreted as an eight-core result.

## Changes measured

1. Dense Float64 Schur builds below 4,000,000 estimated Gram multiply-adds now
   stay serial. This avoids task launch, partial-buffer clearing, and reduction
   when those costs exceed the useful BLAS work.
2. BLAS `syrk!` writes only the lower triangle. Thread-local Float64 partials
   now clear and reduce only that triangle, initialize from the first partial,
   use column-major ranges, and choose a task count from the reduction work.
3. Fixed-width extended arithmetic keeps block parallelism. Full-matrix
   Float64x4 reduction remains on the established path; opt-in triangular
   extended-precision storage uses the new triangular reducer.
4. BigFloat is explicitly guarded as serial and continues to use the
   allocation-free MPFR scalar kernels.

All timings below are medians after warm-up. Allocations are the median bytes
reported by Julia for the calling task; task-local allocations on worker
threads may not all be included by `@timed`.

## Dense Schur thread scaling

| Arithmetic | Case | 1 worker | 2 workers | 4 workers | 8 workers | Best measured speedup |
|---|---:|---:|---:|---:|---:|---:|
| Float64 | small, `L/m/k=8/96/6` | 0.199 ms | 0.197 ms | 0.196 ms | 0.201 ms | 1.02x |
| Float64 | medium, `8/512/10` | 6.561 ms | 3.998 ms | 4.839 ms | 4.606 ms | 1.64x |
| Float64x4 | small, `8/48/4` | 6.184 ms | 4.888 ms | 3.399 ms | 1.982 ms | 3.12x |
| Float64x4 | medium, `8/128/8` | 137.318 ms | 70.243 ms | 47.301 ms | 33.669 ms | 4.08x |

The small Float64 crossover holds the runtime near the 0.20 ms serial value.
Against the prior always-threaded implementation, it was 2.06x, 2.08x, and
2.72x faster for requested worker counts 2, 4, and 8, respectively, in the
same process. Calling-task allocations fell from 3,616/5,760/9,664 bytes to
768 bytes.

For the medium Float64 case, the isolated triangular partial reduction changed
as follows:

| Workers | Previous reduction | New reduction | Speedup | Build allocation, previous → new |
|---:|---:|---:|---:|---:|
| 2 | 450.9 µs | 136.1 µs | 3.31x | 3,616 → 1,920 B |
| 4 | 521.5 µs | 388.5 µs | 1.34x | 5,760 → 3,984 B |
| 8 | 774.0 µs | 568.4 µs | 1.36x | 9,664 → 5,904 B |

Full medium-case runtime is dominated by the block transforms and BLAS calls,
so the end-to-end difference from the reducer alone is small and sensitive to
system noise.

## Sparse exact-arrow thread scaling

| Arithmetic | Case | 1 worker | 2 workers | 4 workers | 8 workers | Best measured speedup |
|---|---:|---:|---:|---:|---:|---:|
| Float64 | small, `blocks/shared=64/8` | 0.019 ms | 0.144 ms | 0.150 ms | 0.156 ms | 1.00x |
| Float64 | medium, `512/24` | 0.254 ms | 0.318 ms | 0.268 ms | 0.230 ms | 1.11x |
| Float64x4 | small, `64/8` | 0.518 ms | 0.561 ms | 0.352 ms | 0.234 ms | 2.22x |
| Float64x4 | medium, `256/16` | 5.415 ms | 4.653 ms | 2.384 ms | 1.453 ms | 3.73x |

The automatic execution plan already keeps small Float64 1x1/2x2 SDP blocks
serial. This benchmark confirms why that policy is important. Float64x4 has
enough arithmetic per compact Schur entry to amortize the scheduler.

## Memory and synchronization

- Dense `m=512` Float64 partials require 2.097 MB per worker:
  4.194 MB at two workers, 8.389 MB at four, and 16.778 MB at eight.
- Dense `m=128` Float64x4 partials require 0.524 MB per worker:
  1.049 MB at two workers, 2.097 MB at four, and 4.195 MB at eight.
- Sparse Float64x4 exact-arrow reduction storage is compact:
  approximately 2.1 KB per worker in the small case and 8.3 KB per worker in
  the medium case.
- Warm empty scheduling regions measured approximately 13–138 µs. This is
  enough to dominate the smallest Float64 Schur builds.
- Absolute process peak RSS was 762–805 MB across the benchmark processes.
  This includes Julia, compiled code, input construction, all earlier cases in
  the process, and workspace memory; the CSV files retain the per-row readings.

## Numerical validation

- Dense Float64 maximum relative Schur error: `2.67e-16`
- Dense Float64x4 maximum relative Schur error: `3.57e-65`
- Sparse Float64 maximum relative Schur error: `1.67e-15`
- Sparse Float64x4 maximum relative Schur error: `1.33e-64`
- The small Float64 serial crossover is bitwise identical to direct serial
  assembly in the regression test.
- The opt-in threaded Float64x4 lower-triangle path agrees with the serial
  full-matrix path to better than `1e-55`.

## Reproduction

```bash
JULIA_NUM_THREADS=8 julia --project=. \
  bench/threading/benchmark_schur_scheduler.jl \
  bench/threading/results/schur-scheduler-2026-07-25.csv

JULIA_NUM_THREADS=8 julia --project=. \
  bench/threading/benchmark_sparse_schur_scheduler.jl \
  bench/threading/results/sparse-schur-scheduler-2026-07-25.csv
```

The raw results are:

- `results/schur-scheduler-2026-07-25.csv`
- `results/sparse-schur-scheduler-2026-07-25.csv`

## Remaining scheduler opportunities

- Dense task-local accumulators are still full square matrices. Packed
  triangular per-worker storage could halve their memory, but it requires a
  workspace-layout change and should be evaluated together with KKT storage.
- Persistent worker teams could remove repeated 20–100 µs task-region launch
  costs, but lifecycle and nested-solve behavior need a solver-wide design.
- Sparse scheduling could cap compute tasks independently of allocated bins for
  latency-bound direct-workspace use. The public automatic pipeline already
  avoids the important small-Float64 case.
- NUMA-aware first-touch placement and socket-local binning should be measured
  on the cluster; this laptop cannot validate those policies.
