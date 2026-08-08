# Task_Low08 Float64 and extended-precision optimization

Date: 2026-07-27

SDPX version: 0.2.0 release candidate

Primary model: lattice bootstrap `Task_Low08`

Input SHA-256:
`5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3`

## Technical summary

Three groups of changes were retained.

1. A narrowly gated Task_Low08-like parameter profile selects
   `beta=0.075`, `gamma=0.8`, `OmegaP=100`, `OmegaD=0.001`, and the SDPB
   predictor. It reduces the Float64 solve from 27 to 24 iterations and the
   matched one-thread SDPX median from 97.643 to 88.123 seconds. The final
   isolated one-thread sweep measured 86.981 seconds.
2. Repeated dense lower-triangular Float64x4 Schur products now assign
   complete output-row ranges to workers. On the same node, the median
   three-iteration Task_Low08 pilot fell from 128.506 to 72.993 seconds
   (1.76x), refinement from 45.957 to 9.422 seconds (4.88x), and corrector
   recovery from 61.188 to 17.119 seconds (3.57x). Every recorded iterate
   metric was bit-for-bit identical.
3. A guarded dense-KKT hierarchy now uses Float64, promotes to a blocked
   `Float64x2` preconditioner when measured correction fails, and retains
   native `Float64x4` as the final fallback. Parallel first-touch conversion,
   panel solves, equality SYRK, and recovery products reduced ten promoted
   factor stages from 737.022 to 151.993 seconds. The controlled
   39-iteration solve fell from 1,759.502 to 1,368.566 seconds while producing
   the same iterate and validation metrics. The retained full solve reached
   `Optimal` in 55 iterations and 2,241.803 seconds without native fallback.

The fair same-node one-thread Float64 comparison remains in MOSEK's favor:
75.783 seconds versus 88.123 seconds for tuned SDPX. At an equal 16-thread
limit, SDPX takes 32.411 seconds versus 40.369 seconds for MOSEK: 19.7%
faster in solver time and 21.0% faster from an already constructed model
through validation. At 32 threads the solver-only medians are effectively
tied, while SDPX remains faster through validation. These are narrow instance
results, not a general solver-to-solver claim.

A native 256-bit BigFloat Task_Low08 run was not submitted. The conservative
workspace estimate is 95.720 GiB before operating-system headroom, above the
64 GiB job limit. The current native BigFloat implementation was instead
validated on the medium exact-arrow CSDR model, where its retained
eight-thread solve is 87.168 seconds versus a 280.011-second legacy
one-thread baseline, with an exact high-precision certificate.

The report uses tables rather than charts because the evidence consists of a
small number of discrete hardware configurations and exact certificate
values; an audit table preserves the timing boundaries and validation fields
more clearly than an interpolated scaling curve.

## Scope and comparison definitions

The Task_Low08 input has 6,119 variables, 482 supplied equalities, 394
independent equality columns, and 32 PSD blocks. Individual coefficients are
sparse (`0.10204%` aggregate coefficient density), but the structural Schur
density is `84.2601%`. CHOLMOD's factor was previously measured as 100% dense,
so sparse Cholesky was excluded from the retained path.

All cluster runs used:

- AMD EPYC 7742 nodes with two 64-core sockets and eight NUMA domains;
- Julia 1.12.6 and OpenBLAS 0.3.29 for SDPX;
- MOSEK 11.1.10 through CVXPY 1.6.5 for the Float64 reference;
- tolerance `1e-6` for Float64 and `1e-12` for Float64x4;
- one complete warm-up for matched Float64 solver comparisons;
- repeated timed solves with medians reported;
- the same compact binary input and the same 394-column equality basis; and
- independent original-coordinate equality and PSD validation.

The matched drivers report input reading, equality-basis selection, model
construction or ingestion, canonicalization, solver time, validation time,
and end-to-end time separately. “End-to-end from constructed model” means
MOSEK task construction/optimization, solution inversion, and validation for
MOSEK, and `solve!` plus validation for SDPX. It excludes the one-time input
read and CVXPY canonicalization or SDPX ingestion.

## Float64: the automatic profile removes three iterations

The coarse and refined parameter sweeps showed that performance is not smooth
around the optimum. For example, `(0.075,0.8)` converged in 24 iterations,
while `(0.075,0.82)`, `(0.08,0.8)`, and `(0.09,0.8)` stalled. The optimized
pair is therefore selected only when all Task_Low08-like structural gates
match; it is not a new global default.

| SDPX strategy | Threads | Iterations | Median solve (s) | Change | Relative gap |
|---|---:|---:|---:|---:|---:|
| Previous fixed `0.1/0.85` | 1 | 27 | 97.643 | baseline | `4.52e-7` |
| Tuned fixed `0.075/0.8` | 1 | 24 | 88.123 | 9.75% faster | `4.27e-7` |
| Automatic structural profile | 32/16 BLAS | 24 | 29.496 | profile validation | `4.27e-7` |

The automatic run recorded
`parameter_profile=:large_lattice_dense_schur` and the intended initial
parameters in diagnostics. The final result also exposes the total number of
accepted refinement corrections across all iterations.

## Float64: fair MOSEK reference

MOSEK and SDPX were run sequentially on the same node for each matched point.
The solver thread limit, tolerance, equality basis, model, warm-up, repetitions,
NUMA policy, and timing boundaries were held constant.

| Threads | Solver | Median solver (s) | End-to-end from constructed model (s) | Iterations | Peak RSS |
|---:|---|---:|---:|---:|---:|
| 1 | MOSEK | 75.783 | 76.398 | 19 | 1.52 GB |
| 1 | SDPX | 88.123 | 88.464 | 24 | 2.01 GB |
| 16 | MOSEK | 40.369 | 41.307 | 20 | 12.66 GB |
| 16 Julia / 16 BLAS | SDPX | **32.411** | **32.651** | 24 | 6.19 GB |
| 32 | MOSEK | 28.166 | 28.954 | 19 | 24.65 GB |
| 32 Julia / 16 BLAS | SDPX | 28.292 | 28.532 | 24 | 6.39 GB |

At one thread, SDPX is 16.3% slower than MOSEK after tuning. At 32 threads,
SDPX is 0.45% slower in solver time, 1.46% faster through validation, and uses
74% less peak resident memory. The later isolated sweep reached 27.672 seconds
at 32/16, but that point ran on another node and is not used as the matched
MOSEK claim. The final equal-16-thread pair ran sequentially on node186 and
shows SDPX 19.7% faster in solver time, 21.0% faster through validation, and
using 51% less peak resident memory.

MOSEK reports 58.215 billion factorization flops and 19--20 interior-point
iterations. Its internal presolve is approximately 0.002 seconds because the
same independent equality basis is supplied to both solvers.

### Matched 16-thread timing boundaries

| Boundary | MOSEK (s) | SDPX (s) |
|---|---:|---:|
| Binary input read | 0.669 | 0.487 |
| Shared equality-basis selection / verification | 0.960 | 0.303 |
| Model construction or ingestion | 21.229 including canonicalization | 0.535 |
| Internal solver presolve | 0.002 | 0, basis supplied |
| Median solver | 40.369 | 32.411 |
| Solution inversion | 0.270 | included in `solve!` |
| Independent validation | 0.011 | 0.087 |
| Median from constructed model through validation | 41.307 | 32.651 |

The construction row is reported rather than folded into solver time. MOSEK's
CVXPY canonicalization takes 21.222 seconds; its direct model-building work is
0.008 seconds. SDPX ingestion takes 0.535 seconds. These frontend differences
are real for the supplied drivers but are not used to claim a faster
interior-point kernel.

The matched SDPX 16-thread median contains:

| SDPX phase | Seconds |
|---|---:|
| Residual and block factors | 3.641 |
| Schur assembly | 5.976 |
| KKT total | 10.773 |
| Schur copy | 0.788 |
| Dense Schur Cholesky | 7.905 |
| Constraint triangular solve (`L^-1 B`) | 1.739 |
| Equality Gram / factorization | 0.308 / 0.051 |
| Predictor total | 4.585 |
| Predictor RHS / solve / recovery | 1.832 / 0.932 / 1.848 |
| Corrector total | 5.192 |
| Corrector RHS / solve / recovery | 1.863 / 0.925 / 1.860 |
| Iterative refinement | 0.546, two accepted corrections |
| Line search / update | 0.145 / 0.024 |

## Float64: complete-solve scaling

The final sweep used automatic parameters, one full warm-up, two timed solves,
and a 128-core reservation. Julia and BLAS widths were varied independently;
NUMA interleaving was used only for the configurations previously shown to
benefit from it.

| Julia / BLAS | NUMA | Median solve (s) | Speedup vs. 1 core | Schur (s) | KKT (s) | Factor (s) | `L^-1 B` (s) | Allocated (GB) | Peak RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / 1 | default | 86.981 | 1.00x | 30.158 | 48.202 | 39.433 | 7.682 | 0.646 | 1.875 |
| 2 / 2 | default | 54.464 | 1.60x | 17.913 | 28.180 | 23.034 | 4.131 | 1.245 | 3.425 |
| 4 / 4 | default | 38.643 | 2.25x | 11.249 | 17.829 | 14.436 | 2.440 | 1.845 | 4.539 |
| 8 / 8 | default | 33.601 | 2.59x | 8.522 | 11.930 | 8.980 | 1.865 | 2.452 | 5.692 |
| 16 / 16 | interleave | **27.401** | **3.17x** | 5.656 | 7.674 | 5.399 | 1.168 | 2.453 | 5.769 |
| 32 / 16 | interleave | 27.672 | 3.14x | 5.466 | 7.718 | 5.433 | 1.154 | 2.455 | 5.912 |
| 64 / 16 | default | 28.364 | 3.07x | 5.542 | 10.964 | 7.358 | 2.555 | 2.455 | 6.205 |
| 128 / 16 | interleave | 30.152 | 2.88x | 5.675 | 7.746 | 5.519 | 1.191 | 2.456 | 8.719 |

Every completed point took 24 iterations and returned the same certificate.
Sixteen Julia and 16 BLAS threads are the current complete-solve optimum.
Beyond 32 Julia workers, Task_Low08's 32 PSD blocks cannot provide more
independent block work, while synchronization and dense-factorization
variability increase.

## Float64x4: Schur construction

Task_Low08 does not satisfy the packed-panel crossover. Automatic selection
rejects all 32 blocks with `reason=:sparse_outer_product_cheaper`, retains the
sparse outer-product kernel, and stores only the lower Schur triangle. The
lower-only automatic path is nevertheless consistently faster than the old
full-symmetric path.

| Julia threads | Previous Schur (s) | Automatic lower-only (s) | Speedup | Hot allocation | Sampled relative error |
|---:|---:|---:|---:|---:|---:|
| 1 | 73.754 | 65.801 | 1.12x | 0 B | 0 |
| 2 | 39.476 | 34.749 | 1.14x | 2,896 B | 0 |
| 4 | 21.931 | 18.272 | 1.20x | 5,152 B | 0 |
| 8 | 12.541 | 10.515 | 1.19x | 9,536 B | 0 |
| 16 | 7.419 | 6.245 | 1.19x | 18,720 B | 0 |

The small nonzero values are task-launch bookkeeping; the arithmetic loop
allocates zero bytes. The sampled validator now reads the workspace
representation directly instead of allocating a second 6,119 by 6,119
Float64x4 matrix.

## Float64x4: repeated refinement matvec

The mixed Float64 factorization reduced KKT time, but the first complete
profile revealed approximately 46--52 seconds of refinement work over only
three iterations, independent of thread count. The dominant operation was a
serial symmetric matrix-vector product over the 6,119-square Float64x4 Schur
matrix.

The retained kernel:

- reads only the stored lower triangle;
- assigns complete output-row ranges to workers;
- caps workers at `ceil(dimension/256)`;
- requires immutable fixed-width extended arithmetic;
- allocates zero bytes inside each row-range kernel; and
- is selected only for non-arrow lower-only systems of dimension at least
  1,024 with distinct input and output vectors.

| Same-node 16-thread, 3 iterations | Previous | Retained | Speedup |
|---|---:|---:|---:|
| Solve | 128.506 s | 72.993 s | 1.76x |
| Schur construction | 39.456 s | 36.974 s | 1.07x |
| KKT setup/factorization | 7.428 s | 6.925 s | 1.07x |
| Predictor | 13.341 s | 6.007 s | 2.22x |
| Corrector | 61.188 s | 17.119 s | 3.57x |
| Refinement | 45.957 s | 9.422 s | 4.88x |
| Cumulative allocation | 10.086 GB | 10.086 GB | unchanged |
| Peak RSS | 21.686 GB | 21.733 GB | +0.2% |

Both versions produced exactly the same primal and dual objectives, relative
gap, primal and dual residuals, equality violation, and minimum primal and
dual PSD eigenvalues after the third iteration. A 1,536-square local
microbenchmark measured 2.92x for the same kernel at four threads with zero
relative error.

## Float64x4: guarded mixed KKT trajectory

The original mixed path used `64*eps(Float64x4)` as its implicit refinement
target. That target asks a Float64 preconditioner to reproduce essentially all
209 bits even when the requested solve certificate is `1e-12`. On
Task_Low08, the conservative condition and predicted-step estimates eventually
rejected a factor whose measured target-precision corrections were still
decreasing. The resulting native Float64x4 Cholesky consumed more than 2,400
seconds in one iteration.

Four changes are retained:

- an inaccurate mixed predictor receives bounded target-precision corrections
  up to the existing `1e-8` predictor residual guard before native fallback;
- the implicit corrector target is
  `max(64*eps(T), min(tolerances)^2)`, while an explicit `refine_tol` still
  wins; and
- explicit `mixed_precision_kkt=:on` with fixed-width extended arithmetic
  treats condition and predicted-step estimates as diagnostics and accepts
  only measured monotone target-precision residual reduction. Automatic mode
  and BigFloat retain the conservative static cutoffs; and
- a failed Float64 correction promotes to a lazily allocated `Float64x2`
  preconditioner before native `Float64x4`. The promoted factor uses blocked
  lower Cholesky, disjoint trailing tiles, parallel first-touch conversion
  and `L^-1 B`, one-triangle equality SYRK, and disjoint recovery outputs.
  Every result is accepted only by the original `Float64x4` residual guards.

| 16-thread controlled run | Requested iterations | Completed | Solve (s) | Schur (s) | KKT (s) | Refinement (s) | Native fallback | Final relative gap |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Previous target and static guards | 100 | 4 | 3,092.428 | 52.237 | 2,954.727 | 44.462 | yes | `2.00` |
| Threaded matvec only | 100 | 4 | 3,198.687 | 52.4 | 3,101.337 | 44.4 | yes | `2.00` |
| Predictor guard correction | 5 | 5 | 145.501 | 77.339 | 20.170 | 13.547 | no | `1.84` |
| Tolerance-aware corrector target | 8 | 8 | 219.360 | 130.222 | 30.082 | 8.347 | no | `8.18e-1` |
| Measured fixed-width trial | 12 | 12 | 281.412 | 173.980 | 27.131 | 13.519 | no | `1.18e-1` |
| Wide-condition trajectory | 16 | 16 | 441.848 | 267.681 | 57.882 | 17.155 | no | `3.80e-3` |
| Blocked `Float64x2` fallback | 40 | 39 | 1,759.502 | 583.738 | 111.127 | 301.033 | no | `9.66e-10` |
| Parallelized `Float64x2` fallback | 40 | 39 | 1,368.566 | 687.028 | 167.197 | 132.345 | no | `9.66e-10` |
| Retained full hierarchy | 100 | 55, `Optimal` | 2,241.803 | 963.781 | 334.530 | 170.973 | no | `7.09e-13` |

The iteration-gated rows are not final solution certificates. They demonstrate
that the retained path remains parallel and continues the same barrier
trajectory instead of entering a multi-thousand-second native factorization.
The two 39-iteration rows are numerically identical:
primal/dual objectives `0.6532912207919219` /
`0.6532912198259296`, primal/dual residuals `1.32e-15` /
`7.41e-14`, equality violation `2.43e-17`, and minimum primal/dual
PSD eigenvalues `-1.99e-13` / `9.62e-19`. The first used ten promoted
factor stages totaling 737.022 seconds; parallel conversion, panel solves,
and equality SYRK reduced those stages to 151.993 seconds (4.85x) and total
solve time by 22.2%, despite the second node spending 103 seconds more in
Schur assembly.

The 15-iteration diagnostic stop was
`reason=:too_slow`: at 0.501 nats/iteration it projected 45.9 more iterations,
but the artificial job had only one iteration left. With a normal iteration
budget that projection does not trigger the stagnation rule. The full run
used 26 Float64x2 promotions totaling 386.243 seconds, 27.366 seconds of
intermediate solves, 11.115 GB of cumulative allocation, and 20.934 GiB peak
RSS. It returned primal/dual objectives `0.6532912207868252` /
`0.6532912207861165`, primal/dual residuals `8.95e-20` / `5.48e-17`,
equality violation `2.43e-17`, and minimum primal/dual PSD eigenvalues
`-1.10e-11` / `-3.98e-14`.

## BigFloat: validated fallback instead of an unsafe full Task_Low08 run

The native BigFloat path was not modified in this Task_Low08 campaign. It
already uses independent MPFR storage, allocation-reusing scalar operations,
native Cholesky and triangular solves, exact singleton-local `2x2` reduction,
ownership-safe block/panel tasks, disjoint lower-triangular Schur tiles, and a
staged working-precision option.

The retained medium CSDR results remain the applicable validation:

| BigFloat configuration | Precision | Solve (s) | Schur (s) | KKT (s) | Allocated | Peak RSS | Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy native, 1 thread | 256 bit | 280.011 | 160.235 | 73.571 | 3.696 GB | 2.280 GB | 41 |
| Reduced arrow, 1 thread | 256 bit | 205.202 | 154.763 | 3.467 | 3.817 GB | 2.422 GB | 41 |
| Reduced arrow, 2 threads | 256 bit | 191.701 | 139.957 | 5.266 | 3.818 GB | 2.240 GB | 41 |
| Reduced arrow, 4 threads | 256 bit | 110.741 | 60.690 | 4.137 | 3.818 GB | 2.169 GB | 41 |
| Reduced arrow, 8 threads | 256 bit | 86.752 | 38.770 | 3.581 | 3.819 GB | 2.164 GB | 41 |
| Reduced arrow, 8 threads | 192 bit | 80.703 | 33.227 | 3.683 | 3.091 GB | 2.051 GB | 41 |

The final instrumented 256-bit run was 87.168 seconds at eight threads, with
an exact certificate. The optional Float64x4 factorization was rejected when
high-precision refinement stalled and correctly fell back to the native
BigFloat reduced panel. It remains off by default. See
[`BIGFLOAT_NATIVE_OPTIMIZATION_2026-07-26.md`](BIGFLOAT_NATIVE_OPTIMIZATION_2026-07-26.md)
for the full MPFR profile, exact objectives, allocation data, and fallback
tests.

## Extended-precision memory gate

The planner includes dense solver storage, task-local Schur accumulators,
conversion buffers, object headers, allocator overhead, and a conservative
operating margin.

| Arithmetic | Threads | Storage floor (GiB) | Conservative estimate (GiB) |
|---|---:|---:|---:|
| Float64x4 | 1 | 3.449 | 18.232 |
| Float64x4 | 8 | 11.260 | 29.949 |
| Float64x4 | 16 | 20.187 | 43.339 |
| Float64x4 | 32+ | 38.041 | 70.120 |
| BigFloat, 256 bit | 1 | 18.109 | 95.720 |
| BigFloat, 256 bit | 8 | 18.109 | 95.720 |

The mixed Float64 factorization adds approximately 0.303 GiB. A full native
BigFloat run should request at least 128 GiB, and preferably more operating
headroom, before it is attempted. The public interface still accepts
`precision=:bigfloat` or a BigFloat-typed `SDPProblem`; the memory gate reports
the estimated requirement rather than silently changing arithmetic.

The dense mixed BigFloat recovery loop now converts Float64 solutions directly
into already-owned MPFR destinations with `mpfr_set_d`. A 256-bit local
microbenchmark over 1,223,800 entries measured 0.0218 versus 0.0662 seconds
(3.03x) and zero versus 137,065,600 allocated bytes. This removes conversion
overhead without changing rounding or enabling unsafe shared MPFR writes.

## Numerical validation

The matched Float64 results use the same tolerance but do not have identical
barrier trajectories. Both satisfy the requested certificate.

| Quantity | SDPX, tuned | MOSEK, 1 thread | MOSEK, 16 threads | MOSEK, 32 threads |
|---|---:|---:|---:|---:|
| Primal objective | `0.653291285886` | `0.653291498427` | `0.653291340513` | `0.653291542620` |
| Dual objective | `0.653290858704` | `0.653290664787` | `0.653291314635` | `0.653290581709` |
| Relative gap | `4.27e-7` | `8.34e-7` | `2.59e-8` | `9.61e-7` |
| Primal residual | `2.73e-9` | `8.58e-9` internal | `5.39e-9` internal | `6.78e-9` internal |
| Dual residual | `1.14e-10` | `5.37e-7` maximum | `4.15e-7` maximum | `1.19e-6` maximum |
| Maximum equality violation | `2.73e-11` | `1.07e-10` | `3.30e-12` | `1.23e-10` |
| Minimum primal PSD eigenvalue | `-1.57e-9` | `2.24e-11` | `7.22e-13` | `2.57e-11` |
| Minimum dual PSD eigenvalue | `4.07e-14` | `3.99e-13` | `1.24e-14` | `4.66e-13` |

The SDPX minimum primal eigenvalue is negative but well inside the
problem-scaled `1e-6` certificate allowance. Float64x4 Schur samples had zero
relative error, the retained matvec produced a bit-for-bit identical
three-iteration state, and the complete Float64x4 hierarchy returned
`Optimal` with a `7.09e-13` gap and `2.43e-17` equality violation. The medium
BigFloat optimized and legacy Schur matrices differ by `9.38e-78` relative at
256 bits.

The independent medium CSDR high-precision audit reports:

| Quantity | Float64x4, 8 threads | BigFloat, 256 bit |
|---|---:|---:|
| Physical primal objective | `1.2733696122998658e-11` | `1.2733696122998658e-11` |
| Physical dual objective | `-1.7146679678993970e-10` | `-1.7146679678993970e-10` |
| Physical relative gap | `1.8420049291293836e-10` | `1.8420049291293836e-10` |
| Equality residual | `1.52e-68` | `1.56e-68` |
| Dual/stationarity residual | `3.2571e-14` | `3.2571e-14` |
| Minimum primal PSD eigenvalue | `2.5085e-18` | `2.5085e-18` |
| Minimum dual PSD eigenvalue | `6.8099e-19` | `6.8099e-19` |

## Automatic crossover rules

### Task_Low08 parameter profile

All conditions must hold:

- at least 4,000 variables;
- at least 100 equality columns after ingestion;
- at least 16 PSD blocks;
- coefficient density at most 0.5%;
- expected Schur density at least 75%; and
- automatic parameter policy enabled.

Otherwise the general large equality-constrained profile retains
`beta=0.1`, `gamma=0.85`.

### Dense Float64x4 refinement product

All conditions must hold:

- no block-arrow workspace;
- lower-only Schur storage;
- immutable fixed-width extended arithmetic;
- Schur dimension at least 1,024;
- at least two requested solver threads; and
- distinct input and output vectors.

Worker count is at most the requested solver threads, Julia process threads,
and `ceil(dimension/256)`.

### Mixed KKT

`mixed_precision_kkt=:auto` requires a Schur dimension of at least 256,
adequate memory, successful Float64 Schur/equality factorizations, condition
estimate at most `1e14` for Float64x4 (`1e8` otherwise), and a predicted
correction count within the configured cap. BigFloat retains these cutoffs in
explicit mode. Explicit `:on` for fixed-width extended arithmetic uses the
same memory, finite-conversion, positive-definiteness, rank, predictor, and
monotone-refinement guards, but treats the conservative condition and
predicted-step estimates as diagnostic.

For `Float64x4`, a failed measured correction may allocate a
`Float64x2` workspace only when its 0.596 GiB Task_Low08 footprint fits the
configured fraction of currently available host/cgroup memory. Dimensions of
at least 512 with more than one solver thread use blocked Cholesky; smaller or
serial cases retain the generic factor. Parallel copy work requires at least
262,144 elements, panel solves use at least 64 rows per worker, `L^-1 B` uses
at least eight columns per worker, and recovery products use at least 16
columns or 128 rows per worker. The promoted factor is rebuilt from the
current Schur matrix; stale-factor reuse was measured and rejected.

### Packed extended-precision Schur

The existing general selector requires at least 32 active columns, 200,000
pair-row operations, 20% expected Schur density, 42% sparse coefficient fill,
and a predicted 1.18x speedup for fixed-width extended arithmetic. BigFloat
uses thresholds of 20 columns, 50,000 operations, 5% Schur density, 62% fill,
and 1.12x predicted speedup. Packing must also fit within the configured
fraction and half of currently available host/cgroup memory.

Task_Low08 fails the packing-benefit test and therefore keeps sparse outer
products. The lower-only storage and refinement matvec are independent
optimizations and remain active.

## Retained and rejected approaches

Retained:

- narrowly gated automatic `beta/gamma` selection;
- lower-triangle-only Float64x4 Schur construction;
- direct sampled Schur validation without a second full matrix;
- disjoint-row Float64x4 symmetric refinement products; and
- total refinement-step diagnostics;
- tolerance-aware mixed corrector refinement and bounded predictor correction;
- measured explicit Float64x4 mixed-KKT acceptance with safe native fallback;
- guarded `Float64x2` promotion with blocked/disjoint-output kernels and
  target-precision validation; and
- allocation-free Float64-to-BigFloat MPFR writes.

Rejected after measurement:

- symmetry-compressed sparse COO: Schur 14.76 versus 11.36 seconds and total
  38.02 versus 33.57 seconds over the controlled partial solve;
- work-weighted block-loop fanout for 32 large lattice blocks: 29.94 versus
  28.92 seconds, 3.5% slower;
- generic packed Float64x4 SYRK for Task_Low08: packing cost exceeds the
  sparse outer-product benefit;
- Float64 limb splitting and Ozaki GEMM on the medium CSDR panel: short
  splits were fast but lost Float64x4 accuracy (`3.76e-16` to `6.77e-21`
  relative error), while the exact Ozaki construction took 0.739 seconds
  versus 0.266 seconds for scalar blocking and 0.080 seconds for the retained
  eight-worker MultiFloatVec kernel;
- sparse Cholesky: 100% filled factor and slower conversion/factorization;
- MKL and BLIS on the EPYC nodes: both slower than OpenBLAS;
- forcing 64--128 workers for one solve: factorization and synchronization
  erase the Schur gain;
- reusing a recent Float64x2 factor: nine retries produced zero accepted
  solves, still required all ten current-matrix promotions, and added 5.31
  seconds of intermediate solves. The apparent total-time reduction in that
  run came from a 99.77-second faster Schur phase on a different node; and
- BigFloat through Float64x4 factorization on the medium CSDR model:
  refinement guard failure followed by the intended native fallback.

The original full Float64x4 attempt entered native factorization and did not
finish within its one-hour wall limit. Two early intermediate experiments
reused a precompiled cache from the previous workspace layout and ended with
`SIGBUS`; those runs are invalid setup data and are excluded from every
performance table. All retained intermediate results used
`--compiled-modules=no`; the final installed release is rebuilt and
precompiled against its own source layout.

Matrix-free Schur plus PCG was not retained. For this dense, ill-conditioned
equality-constrained system, a robust preconditioner would still require most
of the dense factorization storage and convergence evidence was insufficient
to replace the direct method.

## Recommended cluster launch profiles

### Task_Low08 Float64

```bash
OPENBLAS_NUM_THREADS=16 \
numactl --interleave=all \
julia --threads=16 --project=/public/home/yongjunxu/projects/SDPX.jl/current \
    solve_task_low08.jl
```

Reserve 16 physical cores and at least 16 GiB. Use the default automatic
parameter policy. Thirty-two Julia threads are a stable alternative but were
slightly slower in the final complete-solve sweep.

### Task_Low08 Float64x4

```bash
OPENBLAS_NUM_THREADS=1 \
SDPX_MEMORY_LIMIT_BYTES=60GiB \
julia --threads=16 --project=/public/home/yongjunxu/projects/SDPX.jl/current \
    solve_task_low08_float64x4.jl
```

Use `extended_precision_blas=:auto`, `mixed_precision_kkt=:on`, and a 64 GiB
request. Do not use 32 task-local Schur bins under 64 GiB; the conservative
estimate exceeds the job limit.

### BigFloat

Use eight Julia threads and one BLAS thread for exact singleton-local `2x2`
arrow models such as medium CSDR. Keep `mixed_precision_kkt=:off`. The
opt-in staged policy may start a 256-bit request at 192 bits and must retain
certificate-gated fallback.

For full native BigFloat Task_Low08, request at least 128 GiB and expect the
dense native Cholesky to remain serial. The current 64 GiB queue profile is
not safe for that run.

## Modified files

Solver:

- `src/solve.jl`
- `src/kkt.jl`
- `src/kernels/bigfloat.jl`
- `src/kernels/mixed_precision_kkt.jl`
- `src/types.jl`
- `ext/SDPXMultiFloatsExt.jl`

Tests:

- `test/pipeline.jl`
- `test/extended_precision_blas.jl`
- `test/mixed_precision_kkt_regressions.jl`

Benchmarks:

- `bench/extended_precision_blas/small_solve_validation.jl`
- `bench/lattice_bootstrap/benchmark_task_low08_matched_mosek.py`
- `bench/lattice_bootstrap/benchmark_task_low08_matched_sdpx.jl`
- `bench/opt2026/benchmark_task_low08_extended_solve.jl`
- `bench/opt2026/estimate_task_low08_extended_memory.jl`
- `bench/opt2026/sweep_task_low08_parameters.jl`
- `bench/extended_precision_blas/benchmark_schur.jl`

Documentation:

- `README.md`
- `CHANGELOG.md`
- `docs/automatic-optimization-pipeline.md`
- `docs/cluster-workflow.md`
- `docs/parameters.md`
- `docs/precision.md`
- `docs/threading.md`
- `bench/extended_precision_blas/REPORT.md`
- this report

## Regression validation

- Local package suite: 2,107 / 2,107 passed.
- Isolated cluster package suite: 2,069 / 2,069 passed, job `194226`.
- Earlier isolated release-candidate suite: 2,065 / 2,065 passed, job
  `194219`.
- Float64 automatic-profile full solve: `Optimal`, 24 iterations.
- Float64x4 Schur samples: zero relative error at 1/2/4/8/16 threads.
- Float64x4 matvec unit test: target-precision agreement and explicit alias
  rejection.
- Float64x4 full hierarchy: `Optimal`, 55 iterations, no native fallback, and
  all independent objective/residual/PSD gates passed.
- BigFloat medium CSDR: `Optimal`, 41 iterations, exact objective/residual/PSD
  certificate at 256 and 192 bits.

## Remaining bottlenecks and next questions

1. Float64 single-core dense Cholesky and `L^-1 B` consume approximately 47
   of 87 seconds. Matching MOSEK further requires a faster dense
   equality-constrained factorization, not weaker convergence checks.
2. Float64x4 Schur construction remains expensive even at 16 threads.
   Task-local full matrices also make peak memory, rather than CPU count, the
   scaling limit.
3. Native Float64x4 dense Cholesky is still serial. The mixed Float64 backend
   is much faster but requires repeated high-precision refinement products.
4. General non-arrow BigFloat dense Cholesky and triangular solves are serial,
   and Task_Low08 exceeds a 64 GiB memory budget before a safe full solve.
5. NUMA first-touch and socket-local persistent worker teams remain
   unimplemented. They are plausible only after the current 16-thread optimum
   is reproducible across more nodes.
6. A matrix-free method needs a demonstrably robust preconditioner and
   certificate-equivalent convergence before it can replace the direct dense
   KKT path.

## Reproducibility artifacts

The cluster campaign root is:

```text
/public/home/yongjunxu/projects/sdpx-benchmarks/task-low08-20260727
```

Principal jobs:

- `194204`: previous SDPX and one-thread MOSEK baseline;
- `194216`: tuned same-node one-thread MOSEK/SDPX comparison;
- `194217`: tuned same-node 32-thread MOSEK/SDPX comparison;
- `194210`: extended-precision memory preflight;
- `194212`: Float64x4 Schur sweep;
- `194214`, `194215`: coarse and refined parameter sweeps;
- `194222`: automatic-profile validation;
- `194223`: rejected block-loop A/B;
- `194225`: final Float64 1--128-thread sweep;
- `194226`: final cluster package test suite;
- `194227`: Float64x4 refinement-matvec same-node A/B;
- `194229`: final equal-16-thread same-node MOSEK/SDPX comparison;
- `194241`: rejected machine-epsilon mixed refinement target;
- `194242`, `194244`, `194246`, `194248`, `194249`: incremental guarded
  mixed-KKT trajectory and termination diagnostics;
- `194257`, `194258`: controlled Float64x2 blocking and parallelization; and
- `194260`: rejected stale Float64x2-factor reuse experiment; and
- `194261`: retained full Float64x4 certificate run.

Each completed result directory contains environment metadata, raw JSON or
CSV, `/usr/bin/time -v` output, logs, and SHA-256 hashes.
