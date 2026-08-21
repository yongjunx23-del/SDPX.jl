# CSDR J200 Gap Certification and Tiny-Block Pass Optimization

## Scope

This report records the 2026-07-28 optimization and validation campaign for:

```text
J/K/Na/Nmu = 200/2/10/400
model = /public/home/yongjunxu/projects/sdpx-benchmarks/csdr-j200-k2-na10-nmu400/model
```

The model has 40,400 PSD blocks, 40,453 variables, 53 shared variables, and
one local variable per block. The retained configuration used:

- `MultiFloats.Float64x4` (209-bit significand);
- adaptive predictor-corrector parameters;
- SDP Ruiz equilibration;
- the reduced block-arrow KKT path;
- the combined two-row Float64x4 reduced-panel SYRK;
- 64 Julia threads and one BLAS thread;
- `numactl --interleave=all`;
- requested primal, dual, and relative-gap tolerances of `1e-10`.

The performance reference and final run both executed on cluster node
`node91`, an AMD EPYC 7742 system.

## Problems found

### Objective-scale acceptance

Objective equilibration divides both objectives by `objective_scale`. Near a
zero optimum, the relative-gap denominator remains one, so the gap is
multiplied by that scale when the solution is returned to original
coordinates. The old core accepted an internal `1e-10` gap that reconstructed
to `7.70e-9`, and the independent final certificate correctly downgraded the
result to `Stalled`.

The first correction replaced the controller's global `epsilon_gap` with the
scaled threshold. That was rejected: the artificially strict value changed
the stagnation merit and stopped this model at iteration 18, long before its
known recovery.

The retained implementation keeps the user tolerance for adaptive control,
stagnation, refinement, and precision diagnostics. A separate conservative
acceptance threshold,

```text
requested_gap / max(1, objective_scale)
```

is used only when deciding whether the scaled iterate can receive an optimal
certificate.

### Repeated full residual certification

The Newton residual recurrence carries exact affine residual norms between
accepted steps. The solver nevertheless rebuilt the complete original
residual whenever feasibility alone was inside tolerance, even when the gap
was still too large for an optimal certificate. On this model that meant
repeatedly traversing all 40,400 blocks during the late iterations.

The full residual scan now runs only when feasibility and the corresponding
gap or feasibility-certificate sign condition can all pass. The prospective
success check and the independent final certificate remain unchanged.

### Serial tiny-block bookkeeping

Several post-step operations were serial and repeatedly traversed the same
40,400 two-by-two blocks:

- accepted `X` and `Y` updates;
- finite-value checks;
- complementarity evaluation, including a duplicate evaluation for `mu`;
- block target updates;
- dual-objective contractions;
- best-iterate copies.

The retained path assigns complete blocks to workers. Each worker exclusively
owns every matrix it writes. Per-block complementarity and objective scalars
reuse preallocated workspace arrays and are reduced in original block order.
This is enabled only for immutable arithmetic with at least 256 blocks.
Mutable `BigFloat` objects remain serial, but their serial path fuses updates,
finite checks, and complementarity so it also avoids redundant passes.

## Benchmark results

The earlier reference is useful for performance, but it did **not** satisfy
the requested original-coordinate gap. The final solver performs a stronger
calculation and runs nine more iterations.

| Metric | Previous equilibrated path | Retained path |
| --- | ---: | ---: |
| Commit | `c837fed` | `f339923` |
| Status after final certificate | `Stalled` | `Optimal` |
| Iterations | 61 | 70 |
| Solve time | 142.878 s | 109.923 s |
| Core time | not separately recorded | 101.285 s |
| Solve allocations | 5.546 GB | 4.109 GB |
| Solve GC time | 3.641 s | 3.329 s |
| Peak RSS | 6.157 GB | 6.116 GB |
| Primal objective | `5.16025e-9` | `1.70383e-11` |
| Dual objective | `-2.53630e-9` | `-2.29594e-11` |
| Original relative gap | `7.69654e-9` | `3.99977e-11` |
| Primal residual | `1.38e-58` | `1.47e-58` |
| Dual residual | `2.15e-15` | `1.28e-43` |
| Minimum PSD slack proxy | `1.12e-15` | `3.72e-18` |

The retained solve is 1.30x faster (23.1% less time) and allocates 25.9% fewer
bytes while satisfying a substantially stronger final certificate.

### Core phase comparison

| Phase | Previous | Retained | Comment |
| --- | ---: | ---: | --- |
| Equilibration | 8.004 s | 7.759 s | Same algorithm |
| Residual and block factors | 8.428 s | 9.334 s | 70 versus 61 iterations |
| Schur assembly | 20.118 s | 22.137 s | 70 versus 61 iterations |
| KKT | 0.755 s | 0.814 s | Reduced arrow |
| Predictor | 10.183 s | 11.839 s | 70 versus 61 iterations |
| Corrector | 18.405 s | 21.181 s | 70 versus 61 iterations |
| Line search | 1.294 s | 1.259 s | Fraction-to-boundary |
| Accepted-step update | 8.953 s | 0.815 s | 10.99x faster |
| Best-iterate copies | not recorded | 0.557 s | Now explicitly timed |
| Objective and targets | not recorded | 1.892 s | Now explicitly timed |
| Finalization | not recorded | 1.239 s | Original coordinates |
| Other core time | not recorded | 17.966 s | Remaining control/runtime overhead |

Schur, predictor, corrector, and residual totals rose because the correct solve
required 14.8% more Newton iterations. Their per-iteration costs remained
stable. The material improvement comes from removing unnecessary
full-residual scans and parallelizing/fusing the post-step block passes.

## Rejected approaches

- Tightening the controller's global gap tolerance by `objective_scale` was
  rejected because it changed stagnation behavior and caused a false
  iteration-18 stop.
- Two direct Clarabel auxiliary cone-feasibility probes were inconclusive.
  Clarabel returned a numerical error on the converging control model, so no
  conclusion about the target model was drawn.
- A block-norm-ratio-only automatic equilibration threshold was rejected.
  The converging and stalling CSDR models had overlapping ratios, so this
  statistic cannot safely classify them.

## Regression validation

The final commit passed:

- local Julia 1.12.6 suite: 2,277 passed, one expected broken, zero failed;
- cluster Julia 1.12.6 suite: 2,332 passed, zero failed;
- Task_Low08 Float64 regression:
  - status `Optimal`;
  - objective `0.653291393898`;
  - equality residual `2.060e-12`;
  - minimum primal PSD eigenvalue `-5.744e-15`;
  - minimum dual PSD eigenvalue `2.121e-14`;
  - solve time 81.009 seconds on the validation node.

## Remaining bottlenecks

For the final 70-iteration run, the dominant measured core phases are Schur
assembly (22.137 s), corrector work (21.181 s), unclassified core/runtime
overhead (17.966 s), predictor work (11.839 s), and residual/block
factorization (9.334 s). The public-pipeline certificate adds approximately
8.64 seconds beyond the 101.285-second core.

Further work should be measurement-driven. The most plausible next targets
are allocation-free/thread-safe final certificate contractions and fewer
Newton iterations on highly degenerate scaled objectives. Neither should be
enabled without an equivalent-certificate benchmark on this model and a
Task_Low08 regression.
