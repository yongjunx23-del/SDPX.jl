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
| `Float64xN`, `Double64` | Threaded when the type is an immutable fixed-width `AbstractFloat`; extended arithmetic crosses over earlier than Float64. |
| `BigFloat` | General native MPFR phases use one solver thread. Exact singleton-local `2x2` arrows may use ownership-safe native block/panel tasks and disjoint Schur-tile workers. |
| Unknown scalar type | Serial unless the kernel layer explicitly marks it safe. |

## Scheduling and synchronization

PSD blocks are assigned to workers by longest-processing-time (LPT) greedy
partitioning. The weights estimate factorization and Schur work, so
heterogeneous bootstrap blocks are balanced by cost rather than by block
count.

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

## BigFloat policy

Native `BigFloat` uses the serial owned-storage path for general models even if
Julia was started with multiple threads. Exact singleton-local `2x2` arrow
models are the only current native exception.

The reason is solver-specific: a `BigFloat` is mutable, ordinary
`zeros(BigFloat, ...)`/`fill!` storage can alias the same object, and
arbitrary-precision task-local matrices grow quickly with worker count. SDPX
therefore uses independently owned workspace entries plus allocation-reusing
MPFR scalar kernels. For the reduced-arrow exception, each preparation task
owns a disjoint block workspace and two panel rows, and each SYRK task owns a
complete lower-triangular output tile. Inputs are read-only and no writable
BigFloat object crosses tasks.

Use `Float64x4` or another fixed-width `MultiFloats` type when its precision
and Float64 exponent range are sufficient and broader solver phases need
multicore speedup. The experimental `mixed_precision_kkt=:float64x4` mode is
another path for exact singleton-local `2x2` arrows: it constructs and factors
the reduced shared system in Float64x4, checks residuals and refines in
BigFloat, and automatically falls back to native BigFloat if correction is
unsafe. Only the Float64x4 panel/factorization uses the requested workers; the
native MPFR phases remain serial. This mode did not deliver a clear
end-to-end improvement on the medium CSDR benchmark and therefore remains off
by default.

For general non-arrow arbitrary precision, run independent BigFloat instances
as separate processes or scheduler-array elements.

On the medium exact-arrow CSDR model at 256 bits, the ownership-safe native
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

SDPX.set_blas_threads!(1)
@assert SDPX.blas_backend() == :apple_accelerate
@assert SDPX.blas_threads() == 1
```

Use `SDPX.set_blas_threads!`, rather than
`LinearAlgebra.BLAS.set_num_threads`, when Apple Accelerate is loaded.
Accelerate uses `BLASSetThreading`; the ordinary libblastrampoline setter does
not switch it into strict single-threaded mode. AppleAccelerate remains an
optional dependency because loading it changes the process-wide BLAS/LAPACK
backend for every Julia package in the process.

### Linux BLAS backends

Linux deployments can benchmark alternative libblastrampoline backends in a
release-specific environment. Load the backend before SDPX, then use the same
backend-aware thread controller:

```julia
using MKL                 # or: using BLISBLAS
using SDPX

SDPX.set_blas_threads!(16)
```

Do not assume that a vendor backend is faster on every CPU. On the cluster's
dual-socket AMD EPYC 7742 node, equal eight-core, eight-iteration Task_Low08
jobs took 10.816 seconds with OpenBLAS, 13.018 seconds with MKL, and 21.999
seconds with BLIS. OpenBLAS therefore remains the selected backend there.
MKL and BLIS are benchmark-only environment dependencies, not SDPX
dependencies.

Do not start by assigning both Julia and BLAS every core. Use one core per
Julia thread, set a realistic process limit, and measure complete iteration
time. Nested or concurrent solves share the process-global BLAS setting and
are not a supported way to obtain parallelism; use separate processes.

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

See [`bench/threading/RESULTS.md`](../bench/threading/RESULTS.md) for the full
protocol, all 1/2/4/8-worker rows, reduction timings, memory estimates, and
reproduction commands.

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

- The reduced global arrow factorization is serial.
- Generic non-arrow Schur/KKT factorization is dense and remains the dominant
  phase on large lattice problems.
- Dense task-local Schur matrices are not packed; changing their workspace
  layout could nearly halve partial-storage memory.
- Worker teams are created per region; persistent teams could remove
  approximately 20–100 microseconds of repeated launch cost.
- NUMA first-touch, socket-local scheduling, and distributed factorization
  have not been validated.
