# Task_Low08 cluster backend and scaling campaign

## Scope

The campaign used the immutable SDPX release:

```text
commit: affa8992f0da59b55a10dfe4ae697ef3b2e4a325
release: /public/home/yongjunxu/projects/SDPX.jl/releases/affa8992f0da59b55a10dfe4ae697ef3b2e4a325
input sha256: 5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3
```

All timed points ran as independent PBS jobs on nodes with two AMD EPYC 7742
sockets, 128 physical cores, eight NUMA domains, and Julia 1.12.6. Each
eight-iteration point ran one warm-up iteration followed by three timed solves;
the tables report the minimum complete `solve!` time. The arithmetic was
`Float64`.

Task_Low08's late Schur matrix is 84.2601% dense and its CHOLMOD factor is
100% dense. Sparse conversion plus refactorization was previously measured at
0.5914 seconds, versus 0.3229 seconds for dense copy plus Cholesky. This
campaign therefore retained dense Cholesky.

## Release validation and promotion

Validation job `193957.node220` passed all 1,955 package tests, the
Float64x4 and BigFloat smoke solves, sparse scheduler validation, and a complete
Task_Low08 solve. Every result artifact passed its recorded SHA-256 check.

```text
status: Optimal
iterations: 27
primal objective: 0.653291393897909
dual objective: 0.653290931953755
relative gap: 4.61944e-7
reported primal residual: 2.06455e-10
reported dual residual: 1.33121e-8
maximum original equality residual: 2.06013e-12
minimum primal PSD eigenvalue: -7.28632e-15
minimum dual PSD eigenvalue: 2.12095e-14
```

The release passed all gates and was atomically promoted through:

```text
/public/home/yongjunxu/projects/SDPX.jl/current
```

The previous release was preserved for rollback.

## Dense backend comparison

MKL 2025.2 and BLISBLAS 0.2 were installed into release-specific benchmark
environments. They were loaded through libblastrampoline and did not become
SDPX dependencies.

Equal single-core, eight-iteration results:

| Backend | Job | Runtime (s) | Relative to OpenBLAS | KKT (s) |
|---|---|---:|---:|---:|
| OpenBLAS 0.3.29 | `193987.node220` | 31.6783 | 1.000x | 17.9798 |
| MKL 2025.2 | `193992.node220` | 34.5162 | 1.090x slower | 20.7122 |
| BLISBLAS 0.2 | `193993.node220` | 42.4286 | 1.339x slower | 26.6250 |

Equal eight-core, eight-iteration results:

| Backend | Job | Runtime (s) | Relative to OpenBLAS | KKT (s) |
|---|---|---:|---:|---:|
| OpenBLAS 0.3.29 | `193961.node220` | 10.8161 | 1.000x | 3.8933 |
| MKL 2025.2 | `193994.node220` | 13.0183 | 1.204x slower | 4.7207 |
| BLISBLAS 0.2 | `193995.node220` | 21.9989 | 2.034x slower | 9.7062 |

OpenBLAS remains the selected backend for these AMD nodes. Installing a vendor
backend is useful as a measured alternative, but changing the backend merely
because it is available would reduce performance.

## Power-of-two scaling

The first sweep assigned the same ambient width to Julia and OpenBLAS. SDPX
serializes BLAS inside Julia-parallel phases and bounds the KKT BLAS width
internally, so this does not multiply the two widths in a hot phase.

| Julia | Ambient BLAS | Job | Runtime (s) | Speedup | Efficiency | Schur (s) | KKT (s) | Peak RSS (GiB) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | `193987.node220` | 31.6783 | 1.000x | 100.0% | 10.4413 | 17.9798 | 1.841 |
| 2 | 2 | `193988.node220` | 21.0953 | 1.502x | 75.1% | 6.4673 | 10.1673 | 3.778 |
| 4 | 4 | `193989.node220` | 15.0346 | 2.107x | 52.7% | 4.6197 | 6.2571 | 4.588 |
| 8 | 8 | `193961.node220` | 10.8161 | 2.929x | 36.6% | 3.0692 | 3.8933 | 5.581 |
| 16 | 16 | `193962.node220` | 10.4178 | 3.041x | 19.0% | 2.7954 | 3.1218 | 6.173 |
| 32 | 32 | `193963.node220` | 12.2361 | 2.589x | 8.1% | 2.3465 | 4.0581 | 5.695 |
| 64 | 64 | `193964.node220` | 10.6278 | 2.981x | 4.7% | 1.7892 | 3.8742 | 6.114 |
| 128 | 128 | `193965.node220` | 13.4585 | 2.354x | 1.8% | 1.9550 | 4.1275 | 11.206 |

The solver does use all requested Julia workers, but Task_Low08 has only 32 PSD
blocks. Beyond that point, synchronization, memory bandwidth, line search, and
the dense factorization dominate. Reserving 128 cores is slower and almost
doubles peak memory relative to the useful 16–64 core range.

## BLAS-width and NUMA tuning

The second sweep separated the Julia scheduler width from the dense KKT BLAS
width. `numactl --interleave=all` was tested because the solve allocates
approximately 6 GiB on an eight-domain NUMA node.

| Julia | BLAS | NUMA policy | Job | Runtime (s) | Speedup | Schur (s) | KKT (s) | Peak RSS (GiB) |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 16 | 16 | interleave | `193969.node220` | 10.2446 | 3.092x | 1.9752 | 3.3721 | 5.680 |
| 32 | 16 | interleave | `193971.node220` | 9.3201 | 3.399x | 1.8039 | 2.5262 | 5.892 |
| 64 | 16 | default | `193972.node220` | **9.1624** | **3.457x** | 1.7784 | 3.1567 | 6.086 |
| 128 | 16 | interleave | `193974.node220` | 10.9395 | 2.896x | 1.7574 | 2.4665 | 10.938 |

Sixteen BLAS threads were the stable optimum around the useful Julia range.
Twelve threads made the 64-worker point 9.7098 seconds and twenty threads made
it 10.8349 seconds. Forty-eight Julia workers did not beat 64 in the
eight-iteration kernel-focused comparison.

NUMA interleaving improves Schur assembly and dense KKT phases, but it can slow
the residual, predictor, and corrector block phases. It should therefore be a
launch profile selected from complete-solve evidence, not enabled globally
inside SDPX.

## Complete solve

The early eight-iteration optimum and the complete-solve optimum are not
identical. Two independent complete jobs were run with one warm-up iteration
and two timed solves:

| Configuration | Job | Status | Iterations | Best runtime (s) | Schur (s) | KKT (s) | Peak RSS (GiB) |
|---|---|---|---:|---:|---:|---:|---:|
| 32 Julia, 16 BLAS, interleave | `193980.node220` | Optimal | 27 | **34.7877** | 6.5744 | 11.9123 | 5.837 |
| 64 Julia, 16 BLAS, default | `193981.node220` | Optimal | 27 | 35.3490 | 6.2032 | 13.1271 | 6.065 |

The previous promoted release's best complete solve was 37.0343 seconds. The
new 32/16/interleave profile is 1.065x faster, a further 6.5% reduction, with
the same objective and residual quality.

## Recommended launch profile

For this exact problem on the measured dual-EPYC nodes:

```text
dense backend: OpenBLAS
reserved cores: 32
Julia threads: 32
BLAS threads: 16
NUMA policy: numactl --interleave=all
```

Use 8 Julia and 8 BLAS threads for conservative validation jobs. Do not use
128 cores for one Task_Low08 solve; independent solves make better use of
those resources.

These values are hardware- and problem-specific. They are documented as a
cluster profile and are not hard-coded as global SDPX defaults.

## Remaining bottlenecks

- Dense Schur Cholesky and `L^-1 B` remain the largest backend operations.
- Schur assembly scales, but only across 32 PSD blocks and then becomes
  bandwidth-limited.
- Predictor/corrector recovery and line search become synchronization-limited
  at very high Julia widths.
- The current PBS installation does not expose scheduler CPU affinity to the
  job in a portable way. Forced Julia/OpenMP binding was slower and was
  rejected.
- Sparse Cholesky, MKL, and BLIS were all measured and rejected for this
  workload.
