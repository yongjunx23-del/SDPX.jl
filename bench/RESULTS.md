# Benchmark results

Per Appendix D's measurement protocol: minimum of 3 timed runs after one untimed warmup, `@timed`/`@elapsed`. Commit hash omitted below (this rewrite predates the first commit in this working tree — see the session notes in the PR/commit description instead). Julia 1.12.6, Apple Silicon (aarch64-apple-darwin), macOS.

## Old (commit `51f363b`) vs. new, matched instance

Instance: `m=40, k=20, L=3`, dense uniform-random symmetric blocks, no equality constraints, 15 iterations, identical data (same seed) fed to both solvers.

| T | wall-clock (old → new) | speedup | bytes allocated (old → new) | allocation reduction |
|---|---|---|---|---|
| `Float64` | 0.90s → 0.0052s | **174×** | 385.2 MB → 1.2 MB | **321×** |
| `BigFloat` (997-bit) | 9.7s → 7.2s | **1.34×** | 19454 MB → 2567 MB | **7.6×** |

**Why `BigFloat`'s wall-clock gain is much smaller than `Float64`'s, despite a large allocation reduction:** eliminating array-level allocation (`SS` panels, per-`i` slice copies, redundant LU factorizations, `tr(A.*B)` full matrix products, forced `GC.gc()`) accounts for nearly all of `Float64`'s speedup, since `Float64` arithmetic itself never allocates. For `BigFloat`, that same set of fixes was not enough on its own — profiling this comparison directly is what caught it: `kchol!`/`ktrsm!`/`ktrmm!`/`kmul!` initially routed through Base's generic (allocating) `cholesky!`/`ldiv!`/`rmul!`/`mul!`, which allocate a fresh `BigFloat` on every internal scalar `+`/`-`/`*`/`/`. Rewriting those four kernels directly in terms of the already-allocation-free `kdot!` (Cholesky–Banachiewicz and triangular solves are themselves just sequences of dot products) took the allocation reduction from 1.7× to 7.6× and wall-clock from ~1.0× to 1.34×. What's left is believed to be dominated by genuine MPFR arithmetic cost at 997-bit precision (which scales with precision itself, independent of allocation) plus the residual per-entry allocation at the outer "combine and store" layer, which was deliberately kept non-mutating everywhere for BigFloat-aliasing safety (see `kernels/bigfloat.jl` — `copy(acc)` was tried and found to silently alias in one spot; see below).

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
3. **`BigFloat`/MPFR is not safe to use truly concurrently across OS threads.** The exact same LPT-scheduled, partial-buffer-accumulating threaded code gave bit-identical results for `Float64` and silently wrong results (off by orders of magnitude, no error) for `BigFloat` on the same problem. All threaded code paths now detect `BigFloat` and fall back to serial, regardless of `-t`.
4. **`copy(::BigFloat)` is not a deep copy** (`copy(x) === x`) despite `BigFloat` being a mutable struct — caught by a hand-computable 2×2 `ktrmm!` test where every output entry came out equal to the last one computed. Fixed by using non-mutating arithmetic (`acc + zero(BigFloat)`) to force a genuinely fresh object instead.

None of these were hypothetical or design-time guesses — all four were caught by running code and comparing against an independent reference, exactly the discipline Phase 0 exists to enforce.

## Test suite

188/188 assertions pass across correctness, generic arithmetic, sparse
dispatch, MathOptInterface, checkpoint, and threaded tests.

## Adaptive lattice-bootstrap path

The exact `Task_Low08` structure and Float64/Float64x4 kernel measurements are
reported in
[Adaptive Dense/Sparse Optimization for SDPX](../docs/adaptive-dense-sparse-optimization.md).
The reproducible drivers and raw JSON/CSV outputs are under
`bench/lattice_bootstrap` and `bench/adaptive_structure`.
