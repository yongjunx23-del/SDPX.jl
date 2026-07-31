# B3 without universal `[-1, 1]` bounds: MOSEK and SDPX

## Executive summary

This report concerns the Float64 B3 model at `g = 0.8`, without sending the
stored scalar lower and upper bounds to either solver.

The original SDPX benchmark was not a valid solver comparison. It used a dense
61,603-dimensional Schur path, disabled presolve and scaling, consumed about
89.5 GB for the first direction, and returned a stalled point rather than an
optimal certificate. The returned value must not be interpreted as a bound.

SDPX now has a guarded equality-aware sparse Schur route for this model. It:

- keeps the equality operator sparse;
- removes 328 dependent equalities with SuiteSparse SPQR;
- assembles only the lower Schur triangle;
- reuses CHOLMOD symbolic analysis;
- solves all equality right-hand sides together;
- equilibrates the equality normal matrix by exact diagonal congruence;
- uses residual-controlled refinement; and
- validates the returned slack in the original coordinates.

These changes reduce the controlled first-iteration core time from approximately
162--200 seconds on the historical dense route to 103.02 seconds, and the
first-iteration peak RSS to 41.1 GB. They do **not** yet make the current primal
formulation as robust or as fast as MOSEK on a complete solve. The latest
certificate status is reported explicitly below; an infeasible or stalled
iterate is never presented as an optimum.

## Common model

The deterministic Julia artifact is generated locally from the repository's
benchmark scripts. Machine-specific paths and scheduler identifiers are
intentionally omitted from this public report.

Both solvers receive the same mathematical model:

| Property | Value |
|---|---:|
| Scalar variables | 61,603 |
| Linear equalities | 10,036 |
| Equality nonzeros | 88,526 |
| Scalar inequalities | 1 |
| PSD blocks | 23 |
| PSD block dimensions | 53--136 |
| Affine PSD coefficient terms | 390,711 |
| Scalar bounds sent to either solver | 0 |

The bound metadata retained in the artifact is used only for post-solve
diagnostics. The MOSEK export record confirms
`scalar_bounds_sent_to_solver = 0`.

## Reproduction and source integrity

Generate the native Julia artifact, then run the minimum and maximum
directions in separate processes. A completed SuiteSparse factorization and
its dense multi-right-hand-side workspace can otherwise overlap the next
direction's high-water mark. Scheduler scripts and machine-specific launch
commands are maintained outside the public repository.

The SDPX launcher records SHA256 hashes of the principal KKT, step, and
pipeline source files. It also asserts that `pathof(SDPX)` resolves to the
selected release. The numerical process currently uses
`--compiled-modules=no`: the shared cluster depot was observed to reuse a
package image compiled through an older `current` symlink target even though
`pathof(SDPX)` pointed at the new source. Disabling compiled modules prevents a
silent mixture of new files and old compiled methods. This adds JIT work to the
first iteration but makes the benchmark auditable.

## MOSEK Float64 reference

Environment:

- CPU: AMD EPYC 7742;
- requested cores: 32;
- CVXPY 1.9.2;
- MOSEK 11.2.2;
- requested conic tolerance: `1e-8`.

| Quantity | Minimize | Maximize |
|---|---:|---:|
| Reconstructed feasible objective | 0.644898213735908 | 0.808305921734623 |
| MOSEK internal opposite objective | 0.64489501166 | 0.80830772974 |
| Interior-point iterations | 18 | 21 |
| MOSEK numerical solve | 150.01 s | 179.78 s |
| CVXPY + MOSEK wall time | 247.49 s | 275.80 s |
| Maximum original equality residual | 4.07e-9 | 1.98e-9 |
| Minimum original PSD eigenvalue | -2.08e-9 | -2.30e-10 |
| Minimum omitted-bound slack | -30.85 | -5.73 |

The conservative intervals from MOSEK's final primal and dual objectives are:

- minimum optimum approximately
  `[0.64489501166, 0.64489821374]`;
- maximum optimum approximately
  `[0.80830592173, 0.80830772974]`.

The complete job took 9 minutes 36 seconds:

- Julia manifest export: 44.72 s;
- manifest parse: 0.60 s;
- CVXPY compilation: 95.41 s and 94.58 s;
- MOSEK numerical solve: 150.01 s and 179.78 s.

Peak process RSS was approximately 34.6 GB.

## Historical SDPX result: invalid dense baseline

The original minimum run stopped after 15 iterations:

| Quantity | Value |
|---|---:|
| Status | `Stalled` |
| Returned primal candidate | 0.6652793532 |
| Returned dual candidate | -4.3998829e6 |
| Relative gap | 2 |
| Solver primal / dual residual | 3.29e5 / 1.67e7 |
| Original equality residual | 7.77e-4 |
| Minimum original PSD eigenvalue | -9.96e-5 |
| Time | 3,154.16 s |
| Peak RSS | approximately 89.5 GB |

The historical maximum direction also stalled after 16 iterations and
3,350.63 seconds. Its primal and dual candidates were approximately 0.6764041
and 6.6621765e6, with relative gap 2, equality residual 1.04e-4, and minimum
PSD eigenvalue -4.87e-6.

Neither historical result is an optimum or a valid bound. The two-direction
process reached approximately 159.8 GB RSS because the two large workspaces
overlapped in one Julia process.

## Retained SDPX changes

### Sparse input and equality presolve

Sparse equality matrices now remain sparse during ingest, structural cleanup,
equality slicing, Ruiz equilibration, and BigFloat rerounding. Float64
dependent-equality presolve uses SuiteSparse SPQR.

For B3, SPQR takes 44.17 seconds and consistently reports rank 9,708, removing
328 equations. Varying the user tolerance from zero through `1e-8` does not
change that rank:

| User tolerance range | Detected rank | Removed |
|---|---:|---:|
| 0 to 1e-8 | 9,708 | 328 |

Increasing the threshold therefore cannot safely remove the later
iteration-dependent numerical rank loss.

### Equality-aware sparse Schur/KKT backend

The guarded route is selected only when all of the following hold:

- arithmetic is Float64;
- coefficient and equality storage are sparse;
- the primal Schur dimension is at least 10,000;
- predicted triangular Schur density is at most 10%; and
- the predicted factor is safe for the SuiteSparse integer index width.

For B3:

- triangular Schur pattern entries: 139,444,163;
- predicted triangular density: 7.35%;
- CHOLMOD factor nonzeros: 314,532,162.

Each Schur column has a single owner during parallel assembly. Only the lower
triangle is stored. CHOLMOD symbolic analysis is cached across iterations.
Equality elimination uses one dense multi-right-hand-side solve rather than
9,708 independent calls.

### Equality-normal equilibration

The equality system

```text
Q = B' * S^-1 * B
```

is equilibrated by an exact diagonal congruence before Cholesky, with the
multiplier transformed back afterward. On the controlled one-iteration runs,
this reduced equality-factor time from 5.93 to 1.09 seconds and core iteration
time from 111.91 to 103.02 seconds.

### Adaptive-control correctness

The sparse KKT result now distinguishes:

- a pivoted but full-rank equality-normal factor; and
- an actually rank-deficient equality-normal factor.

Only actual rank loss forces factorization quality to zero. Sparse Schur
quality is measured from the regularized diagonal that was actually
factorized, not from a structurally zero unregularized diagonal.

The source-verified first iteration produced:

| Diagnostic | Value |
|---|---:|
| Factorization quality | 7.8390875e-12 |
| Adaptive fallback | `false` |
| Selected beta / sigma | 0.3289707548 / 0.3289707548 |
| Selected gamma | 0.95 |
| Primal / dual step | 0.512 / 0.640 |
| Refinement steps | 2 |

This confirms that the previous “fallback from iteration 1” record came from a
stale package image rather than the corrected controller.

### Certificate and memory handling

Final primal PSD slacks are reconstructed from the returned `x` and the
original affine data before certification. This prevents an internally scaled
slack from being mistaken for the original-coordinate certificate.

`force_gc=true` performs a full collection after accepted iterations and calls
`malloc_trim(0)` on glibc Linux. It can release allocator pages between large
operations or directions, but it is not guaranteed to reduce an
iteration-local high-water mark. The default remains `false`.

## Controlled sparse performance

All entries below use one minimum-direction iteration on 32 requested cores.
They are diagnostic comparisons, not converged solves.

| Configuration | Presolve | Core | Schur assembly | KKT total | Equality factor | Equality solve | Refinement | Solve total | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial sparse route | 145.94 s | 133.04 s | 4.18 s | 55.96 s | 6.26 s | 25.40 s | 56.86 s | 312.37 s | 37.7 GB |
| Without Q congruence | 145.33 s | 111.91 s | 4.60 s | 55.16 s | 5.93 s | 25.15 s | 36.69 s | 291.46 s | 41.0 GB |
| With Q congruence | 138.28 s | 103.02 s | 3.91 s | 48.94 s | 1.09 s | 23.99 s | 34.91 s | 275.94 s | 41.1 GB |
| Source-verified adaptive/JIT | 144.26 s | 149.64 s | 3.87 s | 73.26 s | 1.21 s | 40.90 s | 56.06 s | 305.07 s | 40.5 GB |

The source-verified run includes fresh JIT work because compiled modules were
disabled. Its numerical trajectory also differs from the fixed one-iteration
diagnostics, so it should be used to validate code identity and controller
behavior, not as a clean microbenchmark against the congruence-enabled run.

The sparse route is about 1.6--1.9 times faster per controlled iteration than
the historical dense route's approximately 162--200 seconds, while avoiding
the approximately 89.5 GB first-direction dense workspace.

## Complete sparse trajectories and numerical validation

Earlier 15/16-iteration records predate the hard source check. They are useful
for profiling the sparse backend, but not for evaluating the corrected
adaptive controller:

| Strategy | Status | Iterations | Primal candidate | Dual candidate | Relative gap | Equality residual | Minimum PSD eigenvalue | Solve time | Peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Earlier adaptive | `Stalled` | 15 | 0.6505624709 | 0.5478210296 | 1.027e-1 | 9.88e-7 | -3.50e-8 | 1,693.64 s | 136.9 GB |
| Fixed | `Stalled` | 15 | 0.6539552882 | 0.4618268141 | 1.921e-1 | 2.08e-6 | -3.30e-8 | 1,996.81 s | 136.8 GB |
| Stale-image adaptive diagnostic | `Stalled` | 16 | 0.6505624343 | -4.725e6 | 2 | 9.88e-7 | -1.03e-7 | 2,615.36 s | 184.0 GB |

The first sparse attempt also is not a successful result. Its
minimum direction stopped with an equality residual of 3.12e-5 and a minimum
PSD eigenvalue of -2.15e-7; the process then failed before the maximum
direction with an OpenBLAS `dsyrk` allocation error at an approximately
236 GB high-water mark. The separate-direction launcher prevents that specific
workspace overlap.

The earlier adaptive result is numerically closer to the MOSEK interval than
the fixed result, but neither has an acceptable gap or dual residual. Neither
candidate is a certified bound.

The source-verified adaptive run used `force_gc=true` as a memory-pressure
diagnostic and stopped naturally at iteration 15:

| Quantity | Source-verified adaptive result |
|---|---:|
| Status | `Stalled` |
| Iterations | 15 |
| Primal candidate | 0.648757865551533 |
| Dual candidate | 0.559431256111679 |
| Relative gap | 8.93266e-2 |
| Solver primal residual | 2.24730e-6 |
| Solver dual residual | 2.71989e-3 |
| Original equality residual | 2.24730e-6 |
| Minimum original PSD eigenvalue | -1.10404e-8 |
| Refinement steps | 49 |
| Solve time | 2,384.08 s |
| Core time | 2,232.08 s |
| Presolve time | 141.17 s |
| Peak RSS | 136.92 GB |

The controller remained adaptive for iterations 1--12. At iteration 13 the
equality-normal factor genuinely lost numerical rank, factorization quality
became zero, and the controller safely fell back with
`degraded_factorization`. It retained the fixed fallback thereafter because of
the recorded instability. This is the intended behavior: a merely pivoted
full-rank factor no longer causes a false fallback, while actual rank loss
still does.

Relative gap improved from 0.1027 in the earlier adaptive record to 0.0893,
and the PSD margin improved from approximately -3.50e-8 to -1.10e-8. However,
the original equality residual was 2.25e-6 and the final dual residual was
2.72e-3. The run therefore remains uncertified. It also took longer because it
combined fresh JIT, forced full collections, and 49 refinement steps. This is
not a stable speed or accuracy win, so `parameter_strategy=:fixed` remains the
default.

## Current bottleneck

The optimized sparse backend has moved the bottleneck away from Schur
assembly. In the fixed 15-iteration profiling run:

| Phase | Time | Share of core time |
|---|---:|---:|
| Schur assembly | 44.99 s | 2.4% |
| Total KKT factorization path | 1,024.35 s | 55.6% |
| Constraint triangular solves | 614.35 s | 33.4% |
| Equality factorization | 56.80 s | 3.1% |
| Iterative refinement | 722.16 s | 39.2% |
| Total core | 1,841.43 s | 100% |

Some timers overlap by design: refinement calls the KKT solve path, so phase
percentages must not be summed.

The source-verified adaptive run tells the same story: 44.44 seconds in Schur
assembly, 923.42 seconds in the KKT path, 540.50 seconds in constraint
triangular solves, 61.85 seconds in equality factorization, and 1,201.94
seconds in refinement. Refinement is larger than in the fixed profile because
the late ill-conditioned iterations required up to eight corrections.

The dominant costs are now numeric sparse factorization, the dense block of
9,708 triangular right-hand sides, and repeated refinement. Further Schur
assembly micro-optimization alone cannot close the remaining gap to MOSEK.

## Rejected or inconclusive approaches

- An augmented sparse LDL route was unstable or slower on this model.
- Independent sparse equality right-hand sides and small RHS panels were
  approximately 2--5 times slower than the dense multi-RHS solve.
- Removing the Schur diagonal shift increased runtime by about 9.8 seconds.
- Reusing SPQR's `R11 \\ R12` during presolve produced 140.14 seconds versus
  143.70--145.63 seconds in comparable runs, within observed run-to-run noise;
  the added complexity was reverted.
- Raising the SPQR threshold through `1e-8` removed no additional equalities.
- Forced GC can release retained allocator pages, but did not eliminate the
  factorization/solve high-water mark within a long direction. Separate
  processes are the more reliable boundary between directions.
- Merely adding threads cannot repair the formulation: Schur assembly is
  already a small fraction of total core time, while SuiteSparse solve and
  refinement have limited scaling and high memory traffic.

## Remaining work

The largest remaining structural improvement is executable primal/dual
formulation selection with a typed reconstruction map. The current estimator
can recognize a cheaper dual form, but B3 still executes the primal form.
MOSEK's transformed problem is dramatically smaller in its scalar Newton
dimension, which is the main reason it remains faster and more robust.

Other high-value work:

1. Implement and validate executable dualization for affine PSD blocks and
   equality reconstruction.
2. Replace normal-equation equality elimination when its condition estimate
   deteriorates, using a stable rank-revealing block factorization or a
   null-space basis.
3. Reduce the cost of the 9,708-RHS triangular solve through structure-aware
   blocking and NUMA placement, without duplicating the factor.
4. Make refinement forcing terms depend on achieved residual decrease so late
   iterations do not repeat ineffective solves.
5. Reuse a safe warm-start or factor preconditioner between minimum and maximum
   processes without sharing mutable factor storage.

## Regression tests

The final local source passed:

- the dedicated sparse Schur/KKT test: 17 of 17 checks;
- the complete offline Julia package suite: 2,311 passed, one pre-existing
  known broken test, 2,312 total.

`git diff --check` also passed. The source-verified cluster job records the
exact source hashes used for the B3 result.

## Conclusion

The report's original dense SDPX diagnosis has been fixed in the
implementation: B3 now uses sparse equality-aware algebra, verified presolve,
scaling, symbolic reuse, and original-coordinate certification. First-iteration
time and memory are substantially better.

MOSEK remains the valid Float64 reference for this model. SDPX must report a
certificate with acceptable primal residual, dual residual, gap, and PSD
margin before any returned objective can be compared as a bound. The
source-verified run confirms that the corrected adaptive controller no longer
falls back spuriously, but also confirms a genuine equality-normal rank
failure in the late trajectory. The next meaningful step is executable
dualization rather than additional pairwise Schur tuning or a longer
memory-intensive blind run.
