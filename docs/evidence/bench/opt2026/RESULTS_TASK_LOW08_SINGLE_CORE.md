# Task_Low08 single-core optimization

Date: 2026-07-26

Host: Apple M4, Julia 1.12.6, one Julia thread and one BLAS thread

Arithmetic: `Float64`

## Decision: keep dense Cholesky

The Schur matrix was inspected after iteration 8, when the structure had
become representative of the expensive part of the solve.

| Measurement | Value |
|---|---:|
| Schur dimension | 6,119 |
| Nonzeros in stored lower triangle | 15,776,975 |
| Lower-triangle density | 84.2601% |
| CHOLMOD factor nonzeros | 18,724,140 |
| CHOLMOD factor density | 100.0000% |
| Symbolic fill ratio | 1.1868x |
| Sparse conversion | 0.0656 s |
| CHOLMOD numeric refactorization | 0.5259 s |
| Dense lower-copy plus Cholesky | 0.3229 s |

The sparse representation becomes completely filled during factorization.
Conversion plus a reused CHOLMOD numeric factorization takes 0.5914 seconds,
which is 1.83x the dense factorization time. The existing automatic decision
therefore remains correct: Task_Low08 uses dense Cholesky. A sparse SDP KKT
backend is not enabled for this structure.

## Implemented optimizations

- The sparse-coefficient Schur scatter uses the ascending active-variable
  order to write directly to the lower triangle. This removes `min` and `max`
  from a loop that executes approximately 226 million times per iteration.
- `Float32` and `Float64` KKT factor buffers copy only the lower Schur
  triangle before `POTRF`. The untouched upper triangle is never read.
- Apple Accelerate is available as an optional package extension. SDPX exposes
  backend-aware thread control because Accelerate's strict single-thread mode
  is not controlled by `LinearAlgebra.BLAS.set_num_threads`.
- Suspicious equality Gram factorizations use a vendor-independent numerical
  rank guard. An explicit pivot tolerance is used only for exact duplicate
  columns, avoiding removal of legitimate ill-conditioned Task_Low08
  directions.

Manual unrolling of the common two- and four-nonzero sparse dot products was
also benchmarked. It increased an isolated late Schur build from approximately
0.62 seconds to 0.96 seconds, so it was rejected and no unrolled path remains.

## Runtime

The source-only comparison uses the same strict single-thread Apple Accelerate
backend, eight iterations, three repetitions, and reports the best repetition.

| Version | Runtime | Schur assembly | Schur copy | Allocations |
|---|---:|---:|---:|---:|
| Previous source | 9.8914 s | 4.9989 s | 0.0645 s | 1,176,040,464 B |
| Optimized source | 9.6596 s | 4.7226 s | 0.0541 s | 1,171,731,472 B |
| Improvement | 2.40% | 5.53% | 16.16% | 0.37% |

The complete 27-iteration solve demonstrates the optional dense backend gain:

| Configuration | Runtime | Speedup | Peak RSS |
|---|---:|---:|---:|
| Previous source, OpenBLAS | 73.0863 s | 1.00x | 2.276 GiB |
| Optimized source, Apple Accelerate | 34.1839 s | 2.14x | 2.153 GiB |

The second comparison combines the source optimizations with the backend
change. Apple Accelerate is optional and macOS-only; loading it changes the
process-wide BLAS/LAPACK backend.

## Numerical validation

The original 482 equality constraints were reduced to numerical rank 394
before solving. The complete optimized solve terminated with `Optimal` after
27 iterations.

| Quantity | Value |
|---|---:|
| Primal objective | 0.653291393897897 |
| Dual objective | 0.653290937294294 |
| Relative gap | 4.56604e-7 |
| Reported primal residual | 2.06228e-10 |
| Reported dual residual | 7.45197e-9 |
| Maximum residual over all original equalities | 2.05969e-12 |
| Minimum primal PSD eigenvalue | -1.90577e-14 |
| Minimum dual PSD eigenvalue | 2.12098e-14 |

The negative primal eigenvalue is at the scale of `Float64` roundoff. The
objective agrees with the previous complete baseline to approximately
6e-15.

## Remaining bottlenecks

The optimized complete solve spends approximately 16.8 seconds in Schur
assembly and 9.0 seconds in dense Cholesky. These are the next targets, but the
obvious alternatives tested in this study do not improve them:

- sparse Cholesky loses to full fill-in;
- materializing packed pair values would require roughly 1.8 GB for this
  problem;
- manual short-dot unrolling is slower than the existing streaming kernel.

Further gains likely require a more substantial change, such as a
problem-specific batched sparse Gram kernel or a factorization backend tuned
for Apple silicon. They should be gated by end-to-end measurements rather than
enabled from a microbenchmark alone.

## Validation commands

```sh
JULIA_DEPOT_PATH=/tmp/sdpx-local-depot:$HOME/.julia \
  OPENBLAS_NUM_THREADS=1 \
  julia --project=. --threads=1 -e \
  'using Pkg; Pkg.test()'

SCALE_ITERS=100 SCALE_REPS=1 \
  julia --project=bench --threads=1 \
  bench/opt2026/scaling_lattice_apple_accelerate.jl \
  /tmp/task-low08.bin /tmp/task-low08-single-core.csv 1
```
