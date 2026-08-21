# Extended-Precision BLAS and Sparse Schur Report

Date: 2026-07-25

## Current outcome

SDPX provides an extended-precision kernel module under
`src/kernels/extended_precision_blas/` with in-place triangular `syrk!`,
blocked `gemm!`, cache blocking, panel packing, fixed-width multithreading,
and serial ownership-safe BigFloat kernels. The established Float64 numerical
path is unchanged. Float64x4 and BigFloat use conservative automatic
selection by default:

```julia
extended_precision_blas = :auto
```

Use `:off` to force the established sparse outer-product path:

```julia
SolverOptions{T}(
    extended_precision_blas=:off,
    extended_precision_memory_fraction=0.10,
)
```

Current-path notice: exact-arrow `2x2` models dispatch to a stronger fused
compute-and-scatter specialization before any packed-panel decision. The
selector records `reason=:fused_arrow_specialized`, allocates no transformed
panel or pair buffer, and reports `gram_kernel=:fused_arrow_2x2`. The
historical 2,100-block packed BigFloat experiment below is retained only as
evidence about the generic packed kernel; it is not the current automatic
solver path.

## Implementation

The backend implements:

- lower-triangular in-place `syrk!`;
- blocked in-place `gemm!`;
- cache-sized row and column tiles;
- packed column-major panels;
- a fixed-width micro-kernel with disjoint output-tile threading;
- serial BigFloat kernels that reuse MutableArithmetics/MPFR accumulators;
- explicit independent BigFloat destination ownership;
- active-variable and shared-pattern packing for sparse inputs;
- packed-triangle and mapped lower-triangle scatter;
- a per-block, inspectable crossover decision;
- cgroup-, host-, and configured-memory-aware packing limits.

For exact-arrow `2x2` sparse SDPs, a separate fused kernel computes each
transformed coefficient and scatters directly into compact global, coupling,
and local Schur blocks. It bypasses both packed `syrk!` and the dense Schur
matrix.

## Automatic crossover

| Rule | Fixed-width extended | BigFloat |
|---|---:|---:|
| Default mode | `:auto` | `:auto` |
| Minimum active columns | 32 | 20 |
| Minimum pair-row work | 200,000 | 50,000 |
| Minimum expected Schur density | 0.20 | 0.05 |
| Minimum sparse coefficient fill | 0.42 | 0.62 |
| Minimum predicted speedup | 1.18 | 1.12 |
| Exact-arrow `2x2` | fused-arrow bypass | fused-arrow bypass |
| Threading | bounded by Julia workers | serial |
| Cache geometry | rows 48–64, columns 12–16, micro-tile 2 | rows 24, columns 8 |

The requested packing budget is
`extended_precision_memory_fraction × available memory`, capped at half of
the available amount. “Available” is the minimum usable signal from host free
memory, Linux cgroups, and `SDPX_MEMORY_LIMIT_BYTES`. If no reliable signal is
available, optional packing is disabled.

## Dense kernel benchmark

Host: Apple M4, Julia 1.12.6, one BLAS thread. The process exposed four
hardware workers; the requested eight-worker row is an oversubscription test,
not an eight-core claim. The Float64x4 panel is `64 × 512`.

| Arithmetic | Julia workers | Pairwise reference | Blocked triangular kernel | Speedup | Reference → new allocation | Relative error |
|---|---:|---:|---:|---:|---:|---:|
| Float64x4 | 1 | 0.212714 s | 0.180485 s | 1.18× | 0 → 0 B | 0 |
| Float64x4 | 2 | 0.214640 s | 0.092039 s | 2.33× | 0 → 1,008 B | 0 |
| Float64x4 | 4 | 0.219209 s | 0.047669 s | 4.60× | 0 → 1,952 B | 0 |
| Float64x4 | 8 requested | 0.216394 s | 0.037302 s | 5.80× | 0 → 3,840 B | 0 |
| BigFloat, 256 bit | 1 | 0.010582 s | 0.010320 s | 1.03× | 1,835,232 → 560 B | 0 |

The small threaded allocation is Julia task-launch bookkeeping; tile and
scalar hot loops allocate no heap storage.

## Requested sparse benchmark

Generated instance:

- `J/K/N_a/N_mu = 40/4/20/100`;
- 2,100 PSD blocks;
- 184 global and 2,284 total variables;
- coefficient density `0.053999`;
- structural Schur density `0.155403`;
- input SHA-256
  `8622330ad83989a3c183e82b7b0cedc5bbcd1572e96c35e8cc19750129dc4d4a`.

### Float64x4

Automatic mode retains the specialized exact-arrow path and selects zero
packed blocks.

| Julia workers | Reference | Automatic | Reference/automatic | Automatic allocation |
|---:|---:|---:|---:|---:|
| 1 | 3.188267 s | 3.286486 s | 0.97× | 0 B |
| 2 | 1.668524 s | 1.664335 s | 1.00× | 1,696 B |
| 4 | 0.842500 s | 0.836037 s | 1.01× | 3,104 B |
| 8 requested | 0.615540 s | 0.620717 s | 0.99× | 5,728 B |

All sampled Schur errors were zero. Forcing the generic packed route at four
workers took `1.523235 s`, 1.81× slower than the specialized `0.842500 s`
reference. This validates the bypass.

### BigFloat

Before the fused exact-arrow path existed, a forced packed experiment changed
one 256-bit Schur build from `281.416819 s` to `16.562989 s` (16.99×), with
zero sampled error. That historical result demonstrates the value of owned
MPFR packing over allocation-heavy pairwise arithmetic, but it materialized
panels and is superseded for this problem shape.

The current fused path has been rerun on the real 360-block CSDR `s15`
artifact at 256 bits and one Julia thread:

| Stage | Allocation-heavy reference | Current owned/fused path | Speedup | Reference → new allocation |
|---|---:|---:|---:|---:|
| `buildP!` | 1.839 ms | 0.609 ms | 3.02× | 3.06 MB → 0 |
| Sparse accumulation | 1.636 ms | 0.501 ms | 3.27× | 3.02 MB → 0.121 MB |
| Fused-arrow Schur | 16.604 ms | 4.869 ms | 3.41× | 30.44 MB → 0.282 MB |
| Arrow KKT factorization | 6.233 ms | 2.989 ms | 2.09× | 10.69 MB → 0.209 MB |
| Arrow KKT solve | 1.607 ms | 0.508 ms | 3.16× | 2.86 MB → 0.042 MB |

The relative Schur error was zero and the checked KKT residual was below
`1e-65`. BigFloat remains serial.

## Task_Low08 validation

The required Float64 gate passed before extended-precision experiments:

| Metric | Value |
|---|---:|
| Status | `Optimal` |
| Julia / BLAS threads | 16 / 16 |
| Iterations | 24 |
| Median solve time | 27.400998 s |
| Primal objective | 0.6532912858860604 |
| Dual objective | 0.6532908587043673 |
| Relative gap | `4.27181693e-7` |
| Recomputed primal residual | `2.73189471e-9` |
| Recomputed dual residual | `1.14351315e-10` |
| Maximum equality residual | `2.73155942e-11` |
| Minimum primal PSD eigenvalue | `-1.57405773e-9` |
| Minimum dual PSD eigenvalue | `4.06997424e-14` |
| Equality presolve | 482 → 394 |
| Post-solve certificate | passed |

Task_Low08 has sparse individual coefficient matrices but a dense aggregate
Schur structure. Automatic extended packing is rejected because densifying
the coefficient contractions costs more than the sparse outer-product path.
The retained lower-triangle Float64x4 Schur path is 1.12--1.20× faster over
1/2/4/8/16 threads with zero sampled error. The owned-row refinement product
reduces a same-node three-iteration solve from 128.506 to 72.993 seconds and
refinement from 45.957 to 9.422 seconds without changing any recorded iterate
metric.

When Float64 correction stops meeting the target-precision residual guard,
the opt-in dense KKT hierarchy can now promote its preconditioner to
`Float64x2` before native `Float64x4`. On the controlled 39-iteration
Task_Low08 trajectory, parallel first-touch conversion, blocked Cholesky,
disjoint `L^-1 B`, triangular equality SYRK, and disjoint recovery products
reduced ten promoted factor stages from 737.022 to 151.993 seconds. Total
solve time fell from 1,759.502 to 1,368.566 seconds, and both runs produced
the same objectives, residuals, gap, and PSD checks. The retained full
hierarchy returned `Optimal` in 55 iterations and 2,241.803 seconds, used
20.934 GiB peak RSS, did not enter native Float64x4 factorization, and passed
the independent objective, residual, gap, equality, and PSD checks.

A full native BigFloat solve is not claimed: the conservative
256-bit workspace estimate is 95.720 GiB, above the tested 64 GiB job limit.
The complete campaign, including the matched MOSEK reference and memory gate,
is documented in
[`TASK_LOW08_PRECISION_CLUSTER_REPORT_2026-07-27.md`](../opt2026/TASK_LOW08_PRECISION_CLUSTER_REPORT_2026-07-27.md).

## Remaining bottlenecks

1. Equality-constrained dense SDPs such as Task_Low08 remain dominated by
   dense Schur storage. The guarded Float64x2 promotion avoids most native
   Float64x4 Cholesky work, but a native fallback remains serial.
2. Per-task `m²` Schur accumulators limit useful lattice parallelism under a
   memory budget; tiled or NUMA-aware triangular reductions remain valuable.
3. Large non-arrow BigFloat SDPs still have quadratic dense KKT storage and
   serial factorization.
4. The requested 2,100-block full solve still needs stronger globalization
   and scaling; Schur correctness alone is not a convergence claim.
5. Cache tiles and memory fractions should be calibrated per cluster node
   type. The repository defaults remain conservative.

Raw CSV files and serialized inputs remain ignored. Portable English summaries
such as this file are committed.
