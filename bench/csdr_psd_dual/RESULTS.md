# CSDR PSD-Dual Benchmark Summary

Date: 2026-07-25

This tracked summary preserves the important measurements from the generated
reports under `results/`. Raw CSV files, serialized models, and run-local
reports remain ignored because some inputs exceed 100 MiB.

## Matched problem protocol

Clarabel and SDPX read the same SHA-256-identified serialized PSD dual.
Ingestion and JIT warm-up are excluded from solve time, OpenBLAS/OMP use one
thread, and reported times are the minimum after a full warm-up unless noted.
These are shape-specific measurements, not general solver rankings.

## Optimization progression

Representative `N_mu=60` problem:

- 180 native `2x2` PSD blocks;
- 184 variables;
- four globally shared variables and one block-local variable per block;
- Float64x4 input SHA-256
  `688cfb73652dbc44ae4d75096e41ee3a69bd51febbc22c7f2f0cfcacf1406e81`.

| Implementation | Threads | Best solve | Iterations | Speedup vs original sparse |
|---|---:|---:|---:|---:|
| Original sparse | 1 | 5.650685 s | 52 | 1.00× |
| Active-variable sparse | 1 | 1.659685 s | 52 | 3.40× |
| Active variables + block arrow | 1 | 0.266444 s | 52 | 21.21× |
| Small-block kernels, historical profile | 1 | 0.192604 s | 52 | 29.34× |
| Small-block kernels, tuned profile | 2 | 0.042539 s | 13 | 132.84× |

The active-pair count falls from 3,063,600 to 2,700. Exact block-arrow
elimination replaces a dense `184 × 184` factorization with scalar local
factors and a reduced `4 × 4` global factor.

The tuned result used the then-calibrated
`β=0.1, γ=0.85, Ωp=Ωd=10, predictor=:sdpb` profile. Current automatic
initialization additionally scales Ω from the PSD-block norms.

Correctness against the matched Clarabel solution:

| Metric | Clarabel | SDPX tuned |
|---|---:|---:|
| Primal objective | 76.86220477768425 | 76.86220498885040 |
| Relative objective difference | — | `2.75e-9` |
| Relative gap | `6.84e-9` | `5.62e-9` |
| Minimum PSD slack | `-4.10e-11` | `+4.55e-11` |

## Historical cluster scaling

These 2026-07-24 rows predate the final large-arrow parameter profile and are
retained as scheduler evidence. The old ≥15-active profile `(0.4, 0.7)` has
since been replaced by `(0.01, 0.85)` after larger-model convergence sweeps.

| PSD blocks | Variables | Iterations | 1 thread | 2 threads | 4 threads | 8 requested | 1→8 speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 180 | 184 | 13 | 0.044840 s | 0.040644 s | 0.041621 s | 0.040414 s | 1.11× |
| 360 | 371 | 17 | 0.228176 s | 0.128413 s | 0.087249 s | 0.085198 s | 2.68× |
| 600 | 617 | 26 | 0.939237 s | 0.497861 s | 0.292186 s | 0.268696 s | 3.50× |
| 900 | 929 | 28 | 3.153491 s | 1.641945 s | 1.014964 s | 0.827828 s | 3.81× |

The smallest case is scheduling-latency bound. Parallel efficiency improves
as each barrier covers more block work.

## BigFloat accuracy

The 180-block Float64x4 artifact was converted to BigFloat, so arithmetic
above the 209-bit source precision does not add input information. BigFloat
uses one Julia solver thread.

At 256 bits with `β=0.1, γ=0.75`:

| Requested tolerance | Time | Iterations | Achieved relative gap | Minimum PSD slack |
|---:|---:|---:|---:|---:|
| `1e-12` | 0.496315 s | 29 | `6.60e-13` | `2.44e-15` |
| `1e-20` | 0.691929 s | 40 | `4.24e-21` | `1.58e-23` |
| `1e-30` | 0.864947 s | 51 | `2.84e-31` | `3.94e-35` |

## Current exact-arrow kernel

The final 256-bit, one-thread rerun on the 360-block `s15` artifact measured:

| Stage | Reference | Current | Speedup | Reference → current allocation |
|---|---:|---:|---:|---:|
| Sparse `buildP!` | 1.839 ms | 0.609 ms | 3.02× | 3.06 MB → 0 |
| Sparse accumulation | 1.636 ms | 0.501 ms | 3.27× | 3.02 MB → 0.121 MB |
| Fused-arrow Schur | 16.604 ms | 4.869 ms | 3.41× | 30.44 MB → 0.282 MB |

The current exact-arrow `2x2` path bypasses extended-precision panel packing
and pair buffers. Relative Schur error was zero.

See also:

- [`../extended_precision_blas/REPORT.md`](../extended_precision_blas/REPORT.md)
- [`../bigfloat_sparse_schur/RESULTS.md`](../bigfloat_sparse_schur/RESULTS.md)
- [`../threading/RESULTS.md`](../threading/RESULTS.md)
