# Mixed-precision KKT benchmark

## Scope

This benchmark evaluates the opt-in dense extended-precision KKT path. It
does not change the ordinary `Float64` path. The Schur complement and equality
complement are factored in `Float64`; residuals, direction accumulation, and
acceptance tests remain in the requested arithmetic (`BigFloat` or
`Float64x4`). The native extended-precision factorization is the fallback.

Command:

```bash
julia --project=. docs/evidence/bench/mixed_precision_kkt/benchmark.jl
```

Environment: Apple M4, four hardware threads exposed, Julia 1.12.6, one Julia
thread, one OpenBLAS thread. Timings are the best of three warmed runs. The
direct KKT timing excludes workspace construction and includes factorization,
one solve, and target-precision iterative refinement.

## Direct dense KKT cycle

All systems contain `m/16` equality columns. The source is a deterministic
dense positive-definite Schur matrix.

| Arithmetic | `m` | Native | Mixed | Speedup | Native allocation | Mixed allocation | Refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| BigFloat, 256 bit | 64 | 0.003841 s | 0.001676 s | 2.29x | 293,456 B | 75,984 B | 5 |
| BigFloat, 256 bit | 128 | 0.030852 s | 0.008458 s | 3.65x | 1,100,016 B | 132,496 B | 5 |
| BigFloat, 256 bit | 256 | 0.248491 s | 0.041119 s | 6.04x | 4,276,016 B | 245,648 B | 5 |
| Float64x4 | 64 | 0.001486 s | 0.000367 s | 4.05x | 288 B | 6,000 B | 4 |
| Float64x4 | 128 | 0.011897 s | 0.001401 s | 8.49x | 288 B | 10,288 B | 4 |
| Float64x4 | 256 | 0.104025 s | 0.005598 s | 18.58x | 288 B | 18,992 B | 4 |

Numerical validation:

| Arithmetic | `m` | KKT residual | Primal equation error | Equality equation error |
|---|---:|---:|---:|---:|
| BigFloat | 64 | `3.37e-76` | `3.45e-76` | `1.73e-77` |
| BigFloat | 128 | `1.38e-75` | `1.38e-75` | `8.64e-77` |
| BigFloat | 256 | `2.24e-75` | `2.24e-75` | `1.38e-76` |
| Float64x4 | 64 | `6.50e-64` | `6.62e-64` | `5.70e-65` |
| Float64x4 | 128 | `3.94e-63` | `3.34e-63` | `1.42e-64` |
| Float64x4 | 256 | `3.08e-63` | `2.58e-63` | `1.90e-64` |

The conservative condition estimates were approximately `7.13e3`, `5.30e4`,
and `3.22e5` for `m=64`, `128`, and `256`. Every row remained on the mixed
path and agreed with the native solution at the target arithmetic's rounding
floor.

## Complete-solve correctness gate

The analytic two-variable SDP was solved to `1e-30` tolerance. This tiny model
is a correctness gate, not a performance target: its KKT matrix is too small
to amortize conversion and residual checks.

| Arithmetic | Native | Mixed | Status | Iterations | Objective | Mixed gap | Mixed primal/dual residual |
|---|---:|---:|---|---:|---:|---:|---:|
| BigFloat, 256 bit | 0.014781 s | 0.014792 s | Optimal / Optimal | 39 / 39 | 4.898979485566356 / same | `8.60e-31` | `0 / 3.45e-77` |
| Float64x4 | 0.015603 s | 0.015109 s | Optimal / Optimal | 39 / 39 | 4.898979485566356 / same | `8.60e-31` | `0 / 5.16e-64` |

The native and mixed primal and dual objectives agree beyond the requested
tolerance. Complete-solve allocations were 3,683,600 versus 3,834,928 bytes
for BigFloat and 2,897,360 versus 2,923,200 bytes for Float64x4.

## Condition-estimator overhead

The guard uses LAPACK `xTRCON` reciprocal-condition estimators for both the
one- and infinity-norm of the lower Cholesky factor. It does not materialize
an inverse and costs quadratic rather than cubic work.

For a warmed `4096x4096` Float64 system:

| Operation | Runtime | Allocation |
|---|---:|---:|
| Float64 Cholesky | 0.501942 s | not isolated |
| Two `xTRCON` estimates | 0.053090 s | 262,272 B |

The guard cost was 10.6% of the low-precision factorization in this
single-thread measurement and becomes increasingly small relative to native
extended-precision factorization.

## Selection and fallback policy

The feature is disabled by default:

```julia
SolverOptions{T}(mixed_precision_kkt=:off)
```

`:auto` is an explicit opt-in. It attempts the mixed path only when all of the
following hold:

- arithmetic is `BigFloat` or a fixed-width extended type such as
  `Float64x4`;
- the runtime Schur backend is dense Cholesky and no exact block-arrow KKT
  path applies;
- `m >= 256`;
- required Float64 scratch fits the configured memory budget, whose default is
  10% of reliably available memory;
- adaptive/automatic residual-driven refinement is enabled;
- Float64 conversion is finite;
- both Schur and equality-complement Cholesky factorizations succeed without
  low-precision regularization or pivoting;
- the conservative condition estimate is at most `1e8` by default;
- the predicted refinement count fits the default cap of 32.

`:on` removes only the `m >= 256` crossover. It cannot override memory,
conditioning, rank, finiteness, or accuracy guards.

After each low-precision solve, a target-precision residual check requires a
relative predictor residual no larger than `1e-8`. Refinement uses a
conservative contraction model with an 8x safety multiplier. Failure to reach
the target, a non-finite conversion, or a nondecreasing residual activates the
native factorization and recomputes the direction.

A dynamic fallback skips the next two outer factorizations before one recovery
attempt. Two dynamic fallbacks disable mixed precision for the rest of that
solve. This prevents repeated failed low-precision work while allowing a later
better-conditioned KKT system one bounded opportunity to recover acceleration.

A static rejection that already paid for a low Cholesky (condition limit,
Float64 loss of positive definiteness/rank, non-finite conversion, or an
excessive predicted refinement count) uses the same two-iteration cooldown.
Three repeated rejections for the same reason disable the optional path for the
rest of the solve. A successful recovery resets the static-rejection counter.
Configuration that cannot change during a solve, such as fixed-count
refinement, disables the path immediately.

The final state is exposed as
`result.termination.mixed_precision_kkt`, including the reason, factor-attempt
count, dynamic-fallback count, static-rejection count, cooldown, condition
estimate, and predicted refinement count. Pipeline diagnostics also emit a
warning when the native fallback was used.

Focused regression tests also cover a high-condition rejection and a
deliberately corrupted Float64 factor. Both cases fall back to the native
factorization and retain target-precision residuals.

## Memory note

Allocation totals above come from Julia's warmed `@timed` measurement. An
isolated per-case process peak RSS was not measured, so the table does not
mislabel allocation totals as peak memory. Persistent mixed scratch requires
approximately

```text
8 * (m^2 + m*n + n^2 + 3m + 2n) bytes
```

and is admitted only through the explicit memory-budget guard.
