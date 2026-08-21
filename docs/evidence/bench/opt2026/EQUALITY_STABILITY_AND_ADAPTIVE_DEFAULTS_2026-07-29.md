# Equality Stability and Adaptive Defaults

Date: 2026-07-29

## Scope

This change set addresses the failure described in *Technical Report: SDPX
Solver Failure in Large Primal Crossing-Symmetric SDPs*. The report correctly
identifies two separate facts:

1. the local 2×2 PSD block reduction is fast and should be retained;
2. the reduced equality normal equations can square the condition number and
   become unreliable as the equality basis approaches numerical rank
   deficiency.

The implementation therefore preserves the local block solver and changes the
equality stage, adaptive defaults, diagnostics, and failure reporting.

## Implemented algorithms

### Exactly block-diagonal Schur systems with equalities

When every primal variable belongs to exactly one sparse PSD block, SDPX now:

1. factors each local Schur block independently;
2. applies the local lower factors directly to the corresponding rows of
   `B`, producing `Btilde = L^-1 B`;
3. solves only the reduced equality system;
4. recovers the primal Newton direction with independent local transpose
   solves.

No full primal Schur matrix is allocated or factored. Equalities coupled to
shared arrow variables remain on the general backend because they require a
different joint reduction. Native BigFloat also remains on its ownership-safe
serial backend; the new equality-arrow specialization is restricted to
immutable arithmetic such as Float64 and Float64x4.

### Condition-aware equality hierarchy

The automatic hierarchy is:

1. normal-equation Gram plus Cholesky while the factor has a safe pivot ratio;
2. pivoted Cholesky for a suspicious or rank-deficient Gram factor;
3. rank-revealing column-pivoted Householder QR of `Btilde` when the QR
   dimension and memory crossover permits it;
4. a persistent QR choice on later IPM iterations once automatic mode has
   switched.

The QR rank threshold is arithmetic- and accuracy-aware:

```text
max(maximum(size(Btilde)) * eps(T),
    min(sqrt(eps(T)), requested_accuracy))
```

This avoids using the squared conditioning of `Btilde' * Btilde` for the rank
decision.

### Blocked extended-precision equality Gram

The previous Float64x4 equality Gram used one pairwise column contraction for
every lower-triangular entry. It now uses the existing cache-blocked
triangular SYRK when the automatic crossover predicts a benefit:

- only the lower triangle is computed;
- fixed-width extended arithmetic uses exclusive output-tile ownership;
- scalar hot loops allocate nothing;
- task creation is outside the arithmetic loops;
- ordinary Float32/Float64 bypass the selector and retain the existing vendor
  BLAS SYRK path exactly.

Automatic mode rejects equality panels below 32 columns or 250,000 scalar
contractions even if the general kernel selector would admit them. This
equality-specific floor prevents dispatch overhead on small CSDR models.

### Adaptive numerical defaults

The public defaults are now adaptive or automatically selected for:

| Area | Default behavior |
|---|---|
| `beta` / centering `sigma` | Bounded Mehrotra rule from `mu_aff / mu`, affine steps, centrality, and recent progress |
| primal and dual steps | Independent adaptive fraction-to-boundary safeguards |
| backtracking | Adaptive contraction from recent rejection and achieved reduction |
| KKT regularization | Failure-driven escalation with the measured magnitude fed back to the controller |
| refinement | Residual-driven tolerance and adaptive maximum count |
| restart threshold/scale | Precision-, feasibility-, and factor-quality-aware |
| initialization | Structure- and data-scale-aware automatic profile |
| SDP scaling | Adaptive-pass Ruiz by default |
| equality factorization | Normal equations, pivoting, or QR selected automatically |
| extended kernels | Arithmetic, dimensions, density, threads, and memory crossover |
| mixed precision | Conservative automatic attempt with native fallback |
| BigFloat working precision | Conservative staged precision with certification and full-precision retry |

Requested tolerances, maximum iterations, time limits, thread limits, and
memory limits remain user constraints rather than values that the solver may
silently relax. Expert fixed overrides remain available. The legacy
`sdp`/`findFeasible` compatibility wrappers intentionally retain their
historical trajectory.

## Automatic crossover limits

Rank-revealing equality QR is allowed only when all corresponding limits pass:

| Arithmetic | Maximum equality columns | Maximum `rows * columns` |
|---|---:|---:|
| Float64 | 2,048 | 50,000,000 |
| Fixed extended, including Float64x4 | 1,024 | 20,000,000 |
| BigFloat | 256 | 2,000,000 |

The estimated extra QR storage must also fit within the larger of 256 MiB and
one eighth of currently available memory. B3 has 9,708 retained equalities and
is therefore deliberately rejected by this dense QR crossover.

The blocked equality SYRK additionally uses the host-calibrated extended
kernel selector, the configured memory fraction, the current thread count,
and the equality-specific 32-column/250,000-contraction floor.

## Kernel benchmark

Apple M4, Julia 1.12.6, Float64x4 panel `4096 × 128`, three warmed repetitions.
Times are medians. The old pairwise implementation is serial; the speedup
compares the new configuration with that unchanged reference.

| Julia/kernel threads | Pairwise Gram (s) | Blocked triangular SYRK (s) | Speedup | Pairwise allocation | Blocked allocation |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.813849 | 0.363929 | 2.236× | 0 B | 0 B |
| 2 | 0.807976 | 0.185770 | 4.349× | 0 B | 1,312 B |
| 4 | 0.829291 | 0.102320 | 8.105× | 0 B | 2,336 B |
| 8 | 0.820238 | 0.069099 | 11.870× | 0 B | 4,192 B |

The lower-triangle relative error was exactly zero in every run. The small
parallel allocations are Julia task setup; the tile arithmetic loops allocate
zero bytes. At four threads the automatic selector enabled the kernel with
reason `predicted_speedup` and predicted 4.898×; the measured speedup was
8.112× in the confirmation run.

The isolated cluster candidate repeated the same benchmark on one AMD EPYC
7742 node (two 64-core sockets, eight NUMA domains). Each row is the median of
three warmed repetitions; BLAS threads were fixed to one:

| Julia/kernel threads | Pairwise Gram (s) | Blocked triangular SYRK (s) | Speedup | Blocked allocation |
|---:|---:|---:|---:|---:|
| 1 | 2.425759 | 0.624802 | 3.882× | 0 B |
| 2 | 2.370332 | 0.313283 | 7.566× | 1,312 B |
| 4 | 2.411381 | 0.166360 | 14.495× | 2,336 B |
| 8 | 2.956138 | 0.121917 | 24.247× | 4,192 B |

Every cluster row had exactly zero relative lower-triangle error. The same
candidate job passed 74/74 KKT regressions and 36/36 adaptive-controller
assertions before running the timing sweep.

## Report-derived CSDR validation

The small primal crossing-symmetric model was generated with BigFloat1024
precomputation and solved in Float64x4 using four Julia threads, mandatory
adaptive Ruiz scaling, adaptive IPM parameters, adaptive refinement, exact
low-energy elimination, orthogonal Chebyshev equality modes, and equality
whitening.

| Metric | Result |
|---|---:|
| Status | `Optimal` |
| Iterations | 77 |
| Restarts | 0 |
| Physical objective | `2.8851214247397112e-11` |
| Relative gap | `3.405128880157754e-11` |
| Primal residual | `5.901558114380114e-60` |
| Dual residual | `2.6567143188443284e-46` |
| Reduced equality Gram whitening error | `4.686678660012128e-58` |
| Executed KKT backend | `block_arrow` |
| Equality basis | `orthogonal_chebyshev_modes_cholesky_whitened` |
| Adaptive fallback count | 0 |

The one- and four-thread runs produced the same iterations, objective,
residuals, and gap.

## B3 implications

B3 and the crossing-symmetric primal model expose different equality
regimes:

- the crossing-symmetric model has an exactly block-diagonal primal Schur
  operator and a moderate equality dimension, so local elimination plus
  blocked Float64x4 equality SYRK and QR fallback are applicable;
- B3 retains a 9,708-column sparse equality system and a large sparse primal
  Schur operator, so dense QR is rejected and the existing sparse
  equality-aware Cholesky backend remains selected.

B3 still benefits from adaptive Ruiz passes, the default adaptive controller,
factor-quality diagnostics, original-coordinate certification, and the rule
that an unsuccessful iterate objective is never reported as a physical bound.
This change does not claim that dense QR will accelerate B3.

An isolated one-iteration B3 cluster smoke test exercised the complete
preprocessing and Newton path on the serialized production model. This is a
correctness and bottleneck check, not a converged physical result:

| Metric | Result |
|---|---:|
| Queue / allocation | `normal`, one node, 32 Julia threads |
| Status | `IterLimit` after the requested one iteration |
| Dependent equalities removed | 328 |
| Presolve | 149.745 s |
| Adaptive Ruiz | 0.340 s |
| Core iteration | 143.402 s |
| Schur assembly | 4.194 s |
| KKT total | 64.537 s |
| Sparse Schur factorization | 22.795 s |
| Equality Gram | 3.452 s |
| Equality factorization | 1.018 s |
| Constraint triangular solves | 36.344 s |
| Refinement | 56.373 s, 2 steps |
| Peak RSS | 40,298,776 KiB |
| Full process wall time | 484.35 s |

The path completed without an equality-rank crash, used the sparse
equality-aware backend, and automatically applied one
`1.4901161193847656e-8` relative regularization. The resulting iterate is
deliberately not a bound: its primal residual is 0.40951, dual residual is
62.1475, relative gap is 2, and minimum PSD eigenvalue is -0.52546. The timing
also confirms that B3's next targets are sparse factorization, triangular
solves, refinement, and reusable/cached equality presolve rather than dense
QR.

An isolated cluster candidate was uploaded to:

```text
/public/home/yongjunxu/projects/SDPX.jl/candidates/equality-adaptive-20260729-1
```

The production `current` symlink was not changed.

## Task_Low08 regression

The isolated candidate was also run to a complete Float64 solution of
`Task_Low08` on one AMD EPYC 7742 node. Both runs used:

- `parameter_policy=:auto`;
- `parameter_strategy=:adaptive`;
- `step_rule=:auto`;
- adaptive Ruiz scaling;
- the automatically selected `large_lattice_dense_schur` profile;
- requested primal, dual, and gap tolerances of `1e-6`.

| Metric | 1 thread | 16 threads |
|---|---:|---:|
| Status | `Optimal` | `Optimal` |
| Iterations | 29 | 29 |
| Solver time | 130.600 s | 41.560 s |
| End-to-end benchmark time | 132.239 s | 43.375 s |
| Primal objective | 0.653291423400531 | 0.653291423398558 |
| Dual objective | 0.653290438063478 | 0.653290438061729 |
| Relative gap | 9.853371e-7 | 9.853368e-7 |
| Primal residual | 5.422847e-10 | 5.424380e-10 |
| Dual residual | 2.467421e-11 | 2.479764e-11 |
| Minimum primal PSD eigenvalue | -6.563307e-11 | -6.576539e-11 |
| Minimum dual PSD eigenvalue | 5.164783e-15 | 5.164820e-15 |

The 16-thread solver speedup was 3.142×. Schur assembly improved 4.591×
(43.334 s to 9.439 s), and total KKT factorization improved 6.235×
(66.652 s to 10.690 s). Both independent certificates were valid.

Accuracy and robustness are unchanged, but this fully adaptive trajectory
needed 29 iterations, while the earlier Task_Low08-specific fixed profile
needed 24 iterations. Inspection of the recorded history shows why: after the
cold-start phase, the generic controller raised `sigma` from 0.075 to 0.5 for
several iterations and over-centered this lattice problem. A future retained
speed optimization should add a conservative structure-aware upper bound for
`sigma`; it should not disable adaptive control globally or weaken the
certificate. The current result is therefore a correctness pass, not evidence
that generic adaptation is faster than the best Task_Low08-specific fixed
trajectory.

## Regression tests

Retained local results:

| Suite | Result |
|---|---:|
| KKT regressions | 74 / 74 passed |
| Extended-precision BLAS | 67 / 67 passed |
| Solver certificate regressions | 52 / 52 passed |
| Mixed-precision KKT | 110 / 110 passed |
| Automatic pipeline | all test sets passed |
| Conservative preprocessing | 72 / 72 passed |
| Ingest, sparse ownership, and precision | all test sets passed |
| Earlier full suite before the final equality-kernel patch | 2,318 passed; the subsequently fixed semantic assertions were rerun in their focused suites |

The full-suite Aqua subprocess can remain alive after its assertions pass on
this Julia installation; focused numerical suites were therefore used for the
final patch verification rather than treating that persistent task as a
solver failure.

## Failure reporting

Final diagnostics now expose:

- equality factorization method;
- numerical rank and dimension;
- rank-deficiency verdict;
- factor quality;
- equality Gram kernel;
- adaptive parameter history and fallback reasons;
- original-coordinate certificate validity.

Any non-success status explicitly states that the primal/dual objectives are
best-iterate diagnostics, not certified physical bounds.

## Rejected or deferred approaches

- A generic augmented indefinite LDL factorization was not added. It would
  increase memory and requires a robust pivoting/refinement backend for all
  supported arithmetic types; the current QR route is safer for the eligible
  moderate equality dimensions.
- SVD was not placed in the iteration loop. It is substantially more
  expensive and does not provide an acceptable automatic path for the report
  dimensions. QR rank diagnostics plus equality-basis preprocessing cover the
  current failure mode.
- SDPX does not guess Chebyshev groups from an arbitrary equality matrix. The
  crossing-symmetric model generator already constructs the orthogonal
  transform in BigFloat1024, rounds once to Float64x4, stores the
  reconstruction mapping, and validates in the original nodal basis.
- Native BigFloat block-equality threading remains disabled until complete
  MPFR output-tile ownership and allocation benchmarks justify it.

## Remaining bottlenecks

1. At high `N_a`, generic Float64x4 pivoted QR can still be expensive even
   though it is more stable than normal equations. Orthogonal equality modes,
   exact low-energy elimination, and whitening should remain the primary path.
2. Very large sparse equality systems such as B3 remain dominated by sparse
   Schur construction/factorization and target-accuracy refinement.
3. The current controller reports regularization hints, but KKT factorization
   remains failure-driven because the current iteration must be factored
   before its quality is known.
4. A general shared-arrow-plus-equality reduction is not implemented.
5. Native BigFloat non-arrow factorization remains serial and memory-heavy.

## Modified implementation areas

- `src/workspace.jl`: equality-capable all-local arrow workspace and executed
  equality-kernel state.
- `src/kkt.jl`: local equality transforms, blocked equality SYRK crossover,
  QR hierarchy, rank diagnostics, and block-diagonal equality solve.
- `src/step.jl`: equality and mixed-factor quality in adaptive diagnostics.
- `src/adaptive_parameters.jl`: bounded adaptive centering, step,
  backtracking, refinement, minimum-step, and restart controls.
- `src/ingest.jl` and `src/pipeline.jl`: adaptive Ruiz scaling and
  arithmetic-aware equality presolve.
- `src/solve.jl`: adaptive defaults, precision/fallback behavior, equality
  diagnostics, and uncertified-objective warnings.
- `src/types.jl` and `src/compat.jl`: public automatic defaults with explicit
  legacy fixed compatibility.
- `test/kkt_regressions.jl` and related pipeline/solver tests: block equality,
  QR, extended arithmetic, Ruiz, and diagnostic coverage.
- `bench/opt2026/benchmark_equality_gram.jl`: reproducible equality-kernel
  benchmark.
