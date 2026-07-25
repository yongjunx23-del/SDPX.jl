# Benchmark results

The first section preserves early rewrite measurements against upstream base
commit `51f363b`; it is a historical checkpoint, not the current release
baseline. Per Appendix D's measurement protocol: minimum of 3 timed runs after
one untimed warmup, `@timed`/`@elapsed`. Julia 1.12.6, Apple Silicon
(`aarch64-apple-darwin`), macOS. Current high-precision and pipeline evidence is
linked below.

## Old (commit `51f363b`) vs. new, matched instance

Instance: `m=40, k=20, L=3`, dense uniform-random symmetric blocks, no equality constraints, 15 iterations, identical data (same seed) fed to both solvers.

| T | wall-clock (old → new) | speedup | bytes allocated (old → new) | allocation reduction |
|---|---|---|---|---|
| `Float64` | 0.90s → 0.0052s | **174×** | 385.2 MB → 1.2 MB | **321×** |
| `BigFloat` (997-bit) | 9.7s → 7.2s | **1.34×** | 19454 MB → 2567 MB | **7.6×** |

**Why the historical BigFloat gain was much smaller than Float64's, despite a
large allocation reduction:** removing array-level allocation accounted for
nearly all of Float64's speedup, because Float64 scalar arithmetic does not
allocate. Profiling then found that generic Cholesky, triangular solve, and
matrix multiply created BigFloat temporaries internally. Rewriting those
kernels around the allocation-free dot product improved the historical
allocation reduction from 1.7× to 7.6× and runtime from about 1.0× to 1.34×.
The current owned-destination pass goes further by reusing independent MPFR
destinations in additional Schur, KKT, and LP kernels; use the current reports
linked below for present measurements.

## Against Clarabel.jl (via JuMP), Float64

Same-shaped instance (`m=40, k=20, L=3`), `C` built strongly negative-definite so `x=0` is a verified-feasible interior point (a naive uniform-random `C` has no reason to be negative-semidefinite, and produced a degenerate/infeasible comparison on the first attempt — both solvers correctly flagged it, just with different vocabulary).

| Solver | wall-clock (min of 3) | status | objective |
|---|---|---|---|
| Clarabel.jl 0.x via JuMP | 0.089s | `ALMOST_OPTIMAL` | -56.582886 |
| SDPX | 0.0068s | `Optimal` | -56.582887 |

**~13× faster, agreeing to 7 significant figures** — a useful independent cross-check that both the original bug-fixing work and the rewrite's KKT/Schur math are correct, not just self-consistent. Not a general claim: this is one instance shape (dense, no sparsity, `n=0`), and Clarabel is a general-purpose conic solver (LP/QP/SOCP/PSD/exponential/power cones) paying overhead SDPX doesn't have to; SDPX is purpose-built for exactly this SDP structure. Interesting incidental finding: Clarabel *also* tops out at `ALMOST_OPTIMAL` rather than `OPTIMAL` on this instance — both solvers hit the same practical precision ceiling from different implementations, which is reassuring rather than concerning.

## Genuine bugs found via this measurement discipline (Phase 0's actual purpose)

Measuring rather than assuming caught four real, independently-confirmed defects during this rewrite:

1. **The original's sparse-mode Schur formula was mathematically wrong.** `dot(Y·A_i, X⁻¹·A_j)` (two-panel form) does not equal the canonical `tr(Y·A_i·X⁻¹·A_j)`, and isn't even symmetric in `(i,j)` despite the code mirroring it as `S[j,i]=S[i,j]`. Verified numerically; both call sites in the original test suite are commented out, so this was never exercised. Fixed with a verified single-panel form.
2. **`CholeskyPivoted \` returns `NaN` on rank-deficient systems for `Float64`/LAPACK**, even though the generic `BigFloat` fallback happens to degrade gracefully — a real, type-dependent difference, not a hypothetical one. Fixed with a manual rank-aware solve using the pivot/rank directly.
3. **The original threaded `BigFloat` implementation violated mutable-scalar ownership.** The same LPT-scheduled, partial-buffer-accumulating code gave bit-identical results for `Float64` and silently wrong results for `BigFloat` because shallow copies and shared destinations were mutated concurrently. SDPX keeps the current BigFloat solver path serial until every parallel workspace has explicit ownership guarantees.
4. **`copy(::BigFloat)` is not a deep copy** (`copy(x) === x`) despite `BigFloat` being a mutable struct — caught by a hand-computable 2×2 `ktrmm!` test where every output entry came out equal to the last one computed. Fixed with `MutableArithmetics.mutable_copy` and explicitly owned workspace storage.

None of these were hypothetical or design-time guesses — all four were caught by running code and comparing against an independent reference, exactly the discipline Phase 0 exists to enforce.

## Current high-precision update

The allocation-heavy BigFloat baseline above predates the final owned MPFR
kernel pass. A matched 256-bit complete-solve benchmark changed from
`4.1976 s` and `894.3 MB` allocated to `3.6867 s` and `588.7 MB`, a 1.14×
runtime improvement and 34.2% allocation reduction. Kernel-level and sparse
arrow measurements are tracked in:

- [BigFloat hot kernels](bigfloat_kernels/RESULTS.md)
- [BigFloat sparse Schur and KKT](bigfloat_sparse_schur/RESULTS.md)
- [Extended-precision BLAS](extended_precision_blas/REPORT.md)

## Test suite

The final local release-candidate matrix passed 1,272/1,272 assertions with
four Julia threads. The one-thread run passed 1,263 assertions with one
expected broken multithread-only assertion and no failures. Coverage includes
correctness, Float64x4, BigFloat ownership, sparse/dense dispatch, LP/SDP
pipelines, presolve, MathOptInterface, JLD2 and Double64 extensions,
checkpoints, certificates, spectrum export, and threaded scheduling.

## Adaptive lattice-bootstrap path

The exact `Task_Low08` structure and Float64/Float64x4 kernel measurements are
reported in
[Adaptive Dense/Sparse Optimization for SDPX](../docs/adaptive-dense-sparse-optimization.md).
The reproducible drivers and raw JSON/CSV outputs are under
`bench/lattice_bootstrap` and `bench/adaptive_structure`.
