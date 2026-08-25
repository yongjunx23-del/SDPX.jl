# LinearSolve A/B decision (Wave C)

Branch refactor/zeroalloc-factorcache-hsd-v2. Evidence from
benchmark/linearsolve_ab.jl (Wave A-2): 50x50 dense SPD, native
LinearAlgebra.cholesky vs LinearSolveFactorCache(LUFactorization()).

## A/B numbers
| metric | native Cholesky | LinearSolve LU | ratio |
|---|---|---|---|
| first factorization | 0.000072 s | 0.001936 s | ~27x slower |
| same-A solve | 0.000021 s | 0.016637 s | ~800x slower (noisy first-call) |
| solve allocations | 480 B | 64 B | LinearSolve allocates less |
| relative residual | 3.8e-16 | 4.6e-16 | both accurate |

## Decision (per WORKPLAN decision rules)
- Dense route degradation > 2%: LinearSolve LU is ~27x slower on first
  factorization and ~800x slower on solve for dense SPD -> **DO NOT enter the
  default dense path**.
- Warm allocation: LinearSolve solve allocates 64 B vs native 480 B, but the
  native path is already in-place (la_chol! on ws.Sbuf); the LinearSolve
  advantage is on the solve wrapper, not the factorization.
- Sparse route: not yet benchmarked; only enable if it shows stable speedup on
  multiple real benchmarks.
- Conclusion: LinearSolve stays as an OPTIONAL user-facing interface
  (ext/SDPXLinearSolveExt.jl, weakdep), never the default ExecutionPlan route.

## Follow-up
- Re-benchmark on sparse fixed-pattern systems before any sparse-route
  consideration.
- The optional extension is validated by the CI v2-gates job
  (benchmark/linearsolve_ab.jl smoke).

