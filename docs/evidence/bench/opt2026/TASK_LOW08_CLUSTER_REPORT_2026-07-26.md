# Task_Low08 Cluster Optimization Report

## Scope

This campaign ran entirely through the PBS cluster.  It used the immutable
Task_Low08 input:

```text
path: /public/home/yongjunxu/projects/SDPX.jl/data/task-low08/5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3.bin
sha256: 5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3
```

All performance comparisons below ran on `node100` with eight Julia threads,
eight OpenBLAS threads, a 32 GiB PBS request, fixed `β=0.1`, fixed `γ=0.85`,
and tolerance `1e-6`.  Each scaling point used one warm-up solve and two
measured complete solves; the table reports the faster measured solve.

## Changes

1. Float32/Float64 `ksyrk!` now sends `Btil' * Btil` to one level-3 BLAS
   SYRK operation instead of issuing one BLAS dot product per output entry.
   The public kernel still materializes both triangles.
2. Dense-Cholesky Float32/Float64 workspaces may retain only the lower Schur
   triangle.  Sparse block assembly and packed-pair reduction no longer write
   the unused upper triangle.  Diagnostics can still request a full symmetric
   matrix through `materialize_schur!`, and refinement multiplies through a
   lower-triangle `Symmetric` view.
3. The lower-triangle sparse reduction uses exact, precomputed pair counts to
   partition contiguous output columns.  The schedule is built once with the
   workspace and introduces no per-iteration allocation.
4. Solver timing now separates setup, equilibration, Schur factor copy,
   Schur Cholesky, `L^-1 B`, equality Gram assembly, equality factorization,
   predictor/corrector right-hand sides, linear solves, refinement, direction
   recovery, line search, update, finalization, and unclassified overhead.
5. The Task_Low08 drivers export these detailed solver phases.

## End-to-end results

| Version | PBS job | Best solve (s) | Speedup | Schur (s) | KKT (s) | Allocated bytes | Peak RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline `dcbaa06` | `193930.node220` | 51.5876 | 1.000x | 19.2237 | 17.2883 | 2,975,657,480 | 6.057 |
| BLAS Q-Gram + lower triangle | `193934.node220` | 39.5055 | 1.306x | 11.7243 | 14.0087 | 2,975,731,776 | 5.621 |
| Pair-balanced lower reduction | `193936.node220` | 37.0343 | **1.393x** | 9.7898 | 13.0023 | 2,975,793,944 | 5.627 |

The final version is 28.2% faster than the latest unmodified baseline.  Schur
assembly is 1.96x faster, KKT factorization is 1.33x faster, allocation volume
is effectively unchanged, and measured peak RSS is 7.1% lower.

## Kernel and scheduling measurements

For the actual Task_Low08 equality shape, `Btil` is `6119 x 394`:

| Q-Gram implementation | Time (s) |
|---|---:|
| Pairwise dot products | 0.137945 |
| BLAS SYRK | 0.016404 |

The kernel speedup is 8.41x and the relative matrix error is
`3.13e-16`.

Task_Low08 contains 225,561,425 packed sparse Schur pairs per iteration.
Equal-width columns gave task loads from 1.83 million to 64.70 million pairs.
The precomputed schedule gives every task 28.16-28.22 million pairs:

| Sparse lower reduction | Time (s) | Allocated bytes |
|---|---:|---:|
| Equal-width columns | 0.221587 | 3,680 |
| Pair-balanced columns | 0.140869 | 3,680 |

This isolated reduction speedup is 1.57x with zero Schur-matrix difference.

## Detailed final timing

The best final 27-iteration solve (`193936.node220`) reported:

| Stage | Seconds | Share |
|---|---:|---:|
| Schur assembly | 9.790 | 26.4% |
| Schur Cholesky | 10.006 | 27.0% |
| `L^-1 B` | 1.899 | 5.1% |
| Schur copy | 0.738 | 2.0% |
| Equality Gram | 0.322 | 0.9% |
| Equality factorization | 0.036 | 0.1% |
| Predictor | 4.282 | 11.6% |
| Corrector | 4.454 | 12.0% |
| Residuals and PSD block factors | 3.262 | 8.8% |
| Line search and update | 0.081 | 0.2% |
| Setup, finalization, and other | 1.963 | 5.3% |

The one-time benchmark front end was small relative to the solve: input
construction `0.440 s`, equality presolve `0.157 s`, and ingest `0.456 s`.
Equilibration was disabled by the selected plan and therefore measured
approximately zero.

## Numerical validation

The final validation solve returned:

```text
status:                         Optimal
iterations:                     27
primal objective:               0.6532913938979223
dual objective:                 0.6532909328335563
relative gap:                   4.6106436601967005e-7
reported primal residual:       2.0622792362701148e-10
reported dual residual:         1.0442846587466192e-8
maximum equality residual:      2.0599077998895154e-12
minimum primal PSD eigenvalue: -3.256729705443651e-15
minimum dual PSD eigenvalue:    2.1208915014509984e-14
```

The primal objective differs from the baseline by `1.63e-13`.  All acceptance
gates pass comfortably at the requested `1e-6` tolerance.

Focused cluster validation passed 511 assertions across genericity,
correctness, sparse assembly, KKT, and Schur scheduling tests.  The complete
package suite passed all 1,948 tests in PBS job `193938.node220`; the test
body took 7 minutes 56.8 seconds.

## Remaining bottlenecks

The largest remaining operation is the dense `6119 x 6119` Cholesky
factorization.  It is already an in-place LAPACK POTRF using all eight reserved
cores.  The cluster exposes OpenBLAS but no alternative MKL, BLIS, or oneAPI
module.  Converting this 84.26%-dense Schur matrix to sparse Cholesky is not an
obvious improvement because symbolic conversion and numerical fill would
largely recover a dense factor.

The remaining Schur work is the structurally sparse block transform and pair
contraction.  Its coefficients are only 0.102% dense, so replacing it with a
dense packed SYRK would increase arithmetic and memory traffic substantially.
Further speedups here require a new sparse contraction kernel or a different
mathematical factorization, not another low-risk loop or scheduling change.

No additional low-risk bottleneck with a clear expected end-to-end gain was
identified in this campaign.
