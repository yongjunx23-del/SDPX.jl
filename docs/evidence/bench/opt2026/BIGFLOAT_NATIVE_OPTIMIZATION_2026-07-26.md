# Native BigFloat optimization report

Date: 2026-07-26
SDPX version: 0.2.0 release candidate
Julia: 1.12.6
Primary benchmark: medium CSDR `J/K/Na/Nmu = 32/4/16/100`
Canonical model SHA-256:
`df62be289368abb162e43cddba72cd13efe79cbf441d1596454a658b4175592b`

## Outcome

The retained native BigFloat path reduces the 256-bit, eight-thread medium
CSDR solve from 280.011 seconds to 87.168 seconds, a 3.21x speedup. The
one-thread solve falls to 205.202 seconds, a 1.36x speedup. All runs terminate
`Optimal` in 41 iterations and reproduce the reference objective, residuals,
and PSD certificate.

The main improvement is structural rather than a faster scalar contraction.
Each `2x2` block eliminates its singleton local variable in the three-entry
symmetric coefficient space. Two rank-factor rows are packed per block, and
one ownership-safe lower-triangular BigFloat SYRK constructs the already
reduced `144 x 144` shared Schur matrix. This removes approximately 18 million
pairwise contractions per iteration and 1,700 later local KKT eliminations.

BigFloat now defaults to `extended_precision_blas=:auto`, but the new path is
selected only for a conservatively profitable exact singleton-local `2x2`
arrow. General sparse and dense BigFloat models retain their previous path.

## Baseline profile

The reproducible 256-bit, one-thread baseline took 289.681 seconds:

| Phase | Seconds | Share of solve |
|---|---:|---:|
| Schur construction | 165.634 | 57.2% |
| KKT factorization and local elimination | 76.337 | 26.4% |
| Predictor | 11.466 | 4.0% |
| Corrector | 12.063 | 4.2% |
| Residual and block factors | 7.767 | 2.7% |
| Refinement | 0.002 | <0.1% |
| Other | 16.413 | 5.7% |

The run allocated 3.797 GB cumulatively and reached 2.311 GB peak resident
memory. Allocation sampling identified residual validation, Newton-step
temporaries, trial combinations, Cholesky solves, vector combinations, and
workspace construction as the largest remaining allocation sites. Sampling
was used for ranking only; the exact byte totals in the benchmark tables come
from Julia's complete allocation counter.

## Retained implementation

### Exact reduced-arrow Schur

For every eligible block, SDPX:

1. constructs the exact symmetric `3 x 3` coefficient metric in BigFloat;
2. eliminates the singleton local coefficient in that metric;
3. computes a pivoted rank-two factor with allocation-reusing MPFR
   square-root and division operations;
4. packs the dominant `(a12,a22)` shared pattern without multiplying by the
   absent `a11` entry;
5. handles the 0.7% exceptional patterns separately;
6. writes two private panel rows per block; and
7. constructs only the lower triangle of the shared Schur matrix with blocked
   SYRK.

The local diagonal and coupling remain available for exact direction recovery
and diagnostic Schur actions. A failed rank test, regularized local factor, or
failed factorization reconstructs the legacy native Schur representation and
disables the optimized path for the rest of that solve.

### Ownership-safe threading

Parallelism is limited to regions with provable exclusive ownership:

- contiguous block ranges own their `2x2` workspaces, local metric, coupling,
  and two output panel rows;
- a SYRK job owns one complete lower-triangular output tile;
- every task has private MPFR accumulation, multiplication, and storage
  scalars; and
- panel inputs and scalar coefficients are read-only.

No writable `BigFloat` object crosses tasks. Residual evaluation, direction
recovery, refinement, and all non-arrow native BigFloat paths remain serial.

### MPFR and solve kernels

- Native Cholesky reuses the destination MPFR objects for `sqrt` and division.
- Owned right-hand sides use allocation-free forward/back substitution with
  caller-provided scratch.
- Trial and accepted iterate combinations reuse one owned multiplication
  buffer.
- Singleton local KKT factors cache their inverse instead of repeating two
  divisions for every right-hand side.
- Uninitialized BigFloat panel storage is constructed with independent
  `BigFloat(0)` objects. Regression tests cover this because `BigFloat()`
  represents NaN and `fill!` aliases mutable MPFR objects.

Public alias-safe kernels remain available. The owned variants are used only
where the solver can prove distinct storage.

### Mixed-precision fallback hierarchy

The optional Float64x4 reduced-arrow factorization now falls back in two
stages:

1. exact native BigFloat reduced panel; then
2. legacy native pairwise Schur construction.

The medium model rejected the Float64x4 correction after refinement stalled,
then solved correctly through the native fallback. Mixed precision remains
off by default.

### Staged working precision

`working_precision_policy=:auto` is an opt-in staged policy. It selects

```text
ceil_to_32_bits(
    -log2(minimum_requested_tolerance)
    + 96
    + ceil(log2(problem_dimension))
)
```

clamped to `[minimum_working_precision_bits, precision_bits]`. The default
floor is 192 bits. A lower-precision result is accepted only after normal
original-coordinate certification. `AlmostOptimal`, precision exhaustion,
stagnation, or numerical failure triggers a retry at the requested precision
when time remains. The lower-precision solution is released before allocating
the fallback workspace. Checkpoint resumes always use the requested precision.

This policy is not enabled by default: the medium result is positive, but one
high-precision benchmark is not enough evidence for an unconditional
precision reduction.

## Medium CSDR benchmarks

All jobs used one BLAS thread, the normal cluster queue, the same model and
solver parameters, and 256-bit BigFloat. The 1/2/4/8 table is from one
candidate containing the same retained reduced-arrow kernel.

| Configuration | Threads | Solve (s) | Speedup vs. 280.011 s | Schur (s) | KKT (s) | Allocated | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy native | 1 | 280.011 | 1.00x | 160.235 | 73.571 | 3.696 GB | 2.280 GB |
| Reduced arrow | 1 | 205.202 | 1.36x | 154.763 | 3.467 | 3.817 GB | 2.422 GB |
| Reduced arrow | 2 | 191.701 | 1.46x | 139.957 | 5.266 | 3.818 GB | 2.240 GB |
| Reduced arrow | 4 | 110.741 | 2.53x | 60.690 | 4.137 | 3.818 GB | 2.169 GB |
| Reduced arrow | 8 | 86.752 | 3.23x | 38.770 | 3.581 | 3.819 GB | 2.164 GB |

The final instrumented candidate confirmed 209.208 seconds at one thread and
87.168 seconds at eight threads. Its eight-thread detailed timing was:

| Phase | Seconds |
|---|---:|
| Schur construction | 35.232 |
| KKT total | 4.131 |
| Shared Cholesky | 2.875 |
| Local elimination | 1.243 |
| Residual and block factors | 7.556 |
| Predictor total | 11.305 |
| Corrector total | 11.797 |
| Line search | 1.410 |
| Update | 0.295 |

The final run allocated 3.500 GB cumulatively and reached 2.207 GB peak RSS.
The extra panel makes the one-thread peak slightly higher than the matched
legacy run; at useful multicore settings, exclusive tiles avoid per-worker
Schur replicas and peak memory is lower.

### Working-precision A/B

| Precision | Threads | Solve (s) | Schur (s) | KKT (s) | Allocated | Peak RSS | Iterations |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 bits | 8 | 87.168 | 35.232 | 4.131 | 3.500 GB | 2.207 GB | 41 |
| 192 bits | 8 | 80.703 | 33.227 | 3.683 | 3.091 GB | 2.051 GB | 41 |
| 256-bit request, automatic 192-bit first attempt | 8 | 83.933 | 33.649 | 3.264 | 3.091 GB | 2.175 GB | 41 |

The 192-bit run is 1.08x faster, allocates 11.7% fewer bytes, and reduces peak
RSS by 7.1%. The end-to-end staged run, which retains the original 256-bit
input objects, is 1.04x faster than fixed 256-bit execution and records the
selected 192-bit width in diagnostics. This supports the 192-bit automatic
floor but is not a broad enough result to change the fixed-precision default.

### SYRK tile calibration

The exact `3400 x 144` BigFloat panel was measured three times per tile. The
one-thread best times were 4.006, 3.855, 3.809, 3.801, 4.033, and 4.722
seconds for tile widths 4, 6, 8, 12, 16, and 24. Tile 12 improved on tile 8 by
only 0.2%, while the concurrent eight-thread calibration was noisy and favored
tile 6 over 8 by only 1.2%. The existing width 8 is retained because it is
stable across the solver runs and avoids overfitting one node measurement.
Every timed call allocated only 560 caller-visible bytes.

## Numerical validation

All 256-bit thread counts and the 192-bit run terminated `Optimal` in 41
iterations. The 256-bit result is:

| Quantity | Value |
|---|---:|
| Primal objective | `6.36684806149932875943740286152801590045582859002868e-12` |
| Dual objective | `-8.57333983949698491090488098482511479589010192063630e-11` |
| Relative gap | `9.21002464564691778684862127097791638593568477963917e-11` |
| Primal residual | `1.14326174155719285037098791675331677517e-68` |
| Dual residual | `3.25711804328909479299019725759885764096e-14` |
| Minimum primal PSD eigenvalue | `2.50847355620877625276577639759957051191e-18` |

The 192-bit objective agrees through the precision needed by the requested
`1e-10` tolerance, with primal residual `2.70e-49`, dual residual
`3.25711804328909479298918e-14`, and minimum PSD eigenvalue
`2.50847355620877625276574e-18`.

A small exact-arrow regression compares the optimized and legacy Schur
matrices at relative error `9.38e-78` at 256 bits. Tests also compare the full
Schur action and KKT solution, exercise materialization and fallback, verify
independent MPFR object ownership, and require zero hot-loop allocation when
scratch is supplied.

## Crossover rules

The native reduced-arrow path requires all of the following:

- sparse SDP with no explicit equalities;
- all PSD blocks are `2x2`;
- exact arrow incidence with one singleton local variable per block;
- at least 20 shared columns and at least `5e4` weighted panel-pair work;
- expected shared-Schur density at least 0.05;
- shared active density at least 0.10;
- predicted speedup at least 1.12x; and
- panel memory within both the configured fraction and half of currently
  available host/cgroup memory.

The default panel budget is 10% of reported available memory. Automatic
threading is further limited by Julia's process thread count, the requested
solver thread count, available disjoint tiles, and at least 18,000 weighted
operations per selected worker. Unsupported structures keep the legacy sparse
outer-product path.

## Rejected or deferred approaches

- Float64x4 mixed factorization took 323.897 seconds, allocated 15.42 GB, and
  fell back after refinement stalled. It remains opt-in.
- A packed triangular factor would save only the unused half of two
  `144 x 144` reference arrays on this model while requiring a new packed
  Cholesky/solve interface. The risk and complexity exceed the memory saving;
  computation is lower-only, but the shared factor retains strided square
  backing storage.
- Matrix-free reduced Schur plus PCG is inappropriate for a `144 x 144`
  shared system: it replaces one robust small Cholesky with repeated passes
  over 1,700 blocks and adds convergence/preconditioner failure modes.
- General `Threads.@threads` over BigFloat loops was rejected. Only explicitly
  owned tiles and task-local scratch are parallel.
- Native BigFloat Task_Low08 was not attempted. Its `6119 x 6119` dense Schur
  workspace requires several multi-gigabyte MPFR matrices before input and
  factor storage, making it impractical on the benchmark allocation. The
  required Float64 regression remains the appropriate guard.

## Task_Low08 regression

Task_Low08 is a final Float64 regression, not the optimization target. The
validated eight-thread result remains:

| Quantity | Result |
|---|---:|
| Status / iterations | `Optimal` / 27 |
| Solve / end-to-end | 57.116 / 59.483 s |
| Primal / dual objective | `0.6532913938979086` / `0.6532909319537545` |
| Relative gap | `4.6194e-7` |
| Primal / dual residual | `2.0646e-10` / `1.3312e-8` |
| Minimum primal / dual PSD eigenvalue | `-7.2863e-15` / `2.1209e-14` |
| Peak RSS | 3.88 GiB |

The BigFloat-only kernel selection does not alter this Float64 route. The
final run is slower than the earlier 47.604-second measurement on a different
node allocation, but it retains the same 27 iterations, objective, residuals,
and PSD certificate; this is treated as a correctness regression rather than
a cross-node performance claim.

## Final validation gate

The release candidate passed the complete cluster package test suite:

| Validation | Result |
|---|---:|
| Full package tests | 2,062 / 2,062 passed |
| PBS job | `194196.node220` |
| Exit status | 0 |
| Wall time | 10m08s |
| Peak RSS | 2,201,108 KiB |

This full-suite result is in addition to the medium CSDR solve, the
lower-level exact-arrow and ownership regressions, and the final Task_Low08
certificate reported above.

## Modified files

Core implementation:

- `src/kernels/bigfloat.jl`
- `src/kernels/generic.jl`
- `src/kernels/extended_precision_blas/packing.jl`
- `src/kernels/extended_precision_blas/syrk.jl`
- `src/kernels/mixed_precision_kkt.jl`
- `src/kernels/threaded.jl`
- `src/schur.jl`
- `src/kkt.jl`
- `src/workspace.jl`
- `src/pipeline.jl`
- `src/step.jl`
- `src/solve.jl`
- `src/types.jl`
- `src/lp_solver.jl`

Validation and documentation:

- `test/bigfloat_kernel_regressions.jl`
- `test/bigfloat_ownership_regressions.jl`
- `test/bigfloat_sparse_schur_regressions.jl`
- `test/extended_blas_regressions.jl`
- `test/extended_precision_blas.jl`
- `test/pipeline.jl`
- `test/sparse.jl`
- `bench/extended_precision_blas/benchmark_bigfloat_syrk_tiles.jl`
- `bench/opt2026/BIGFLOAT_NATIVE_OPTIMIZATION_2026-07-26.md`
- `bench/opt2026/CSDR_MEDIUM_J32_K4_CLUSTER_2026-07-26.md`
- `README.md`, `CHANGELOG.md`, and precision/threading/interface/cluster
  documentation.

## Remaining bottlenecks

At eight threads, Schur construction is still 40.4% of total time.
Predictor/corrector work is 26.5%, residual/block factor work is 8.7%, and
KKT is only 4.7%. The next useful work is therefore:

1. persistent worker teams or coarser iteration-level task reuse;
2. allocation-free residual and direction-recovery contractions;
3. NUMA-aware first touch and contiguous block placement on larger nodes;
4. a broader adaptive-precision benchmark corpus before promotion; and
5. a different dense high-precision backend for non-arrow models, where the
   current structural specialization does not apply.

No remaining change has an obvious low-risk benefit comparable to the retained
reduced-arrow implementation.

## 2026-08-01 all-local equality addendum

SDPX 0.3.0 now recognizes a second exact BigFloat structure: block-diagonal
`2x2` cells whose Schur variables are all local and are coupled by explicit
equalities. It factors each local cell once, applies the equality panel through
ownership-safe triangular solves, forms only the lower equality Gram with
disjoint tiles, and solves the resulting 170-by-170 normal system. Sparse CSC
residual GEMV, block residual/factor work, predictor/corrector recovery,
fraction-to-boundary diagnostics, and accepted updates use disjoint mutable
MPFR destinations. Global reductions retain block order.

End-to-end measurements on the certified J40 BigFloat512 model showed that
uniform 128-way scheduling was counterproductive. A retained phase-aware
controller therefore limits fine-grained MPFR work to 64 task streams while
allowing the larger equality Gram to use the requested width:

| Requested workers | Fine-phase cap | Solver (s) | Internal (s) | Equality Gram (s) | Peak RSS (KiB) |
|---:|---:|---:|---:|---:|---:|
| 64 | 64 | 368.704 | 229.729 | 60.580 | 3,907,204 |
| 96 | 64 | 398.303 | 265.944 | 56.632 | 4,147,008 |
| 128, old uniform schedule | 128 | 495.811 | 356.574 | 43.446 | 4,346,976 |
| 128, retained phase cap | 64 | 425.880 | 286.429 | 41.672 | 4,058,792 |

The capped 128-worker route is 14.1% faster and uses 6.6% less peak RSS than
the uniform 128-worker route, with a bit-for-bit identical 158-iteration
certificate. The 64-worker configuration is still fastest and is recommended
for this geometry. Wider allocations should be tested only on materially
larger equality panels.

A fixed 1,024-bit, 64-worker support run passed in 157 iterations with no
restart, regularization, refinement, or fallback. Solver time was 553.959
seconds, internal time 360.413 seconds, and peak RSS 4,268,480 KiB. Equality
Gram, equality factorization, and constraint triangular work took 115.491,
42.838, and 49.298 seconds. The physical certificate had original linear
residual `3.900e-63`, off-grid relative residual `1.750e-11`, minimum PSD
eigenvalue `1.760e-34`, and zero disk violation. Its relative gap
`1.890e-13` was not tighter than the 512-bit result `3.451e-14`; both solve the
same coefficient set rounded once to Float64x4. This confirms BigFloat1024
support but does not justify changing the model-specific 512-bit default.

The complete local release-candidate suite passed 5,752 tests after a Julia
1.10 compatibility fix made automatic rank-loss handling go directly to the
existing rank-revealing QR backend instead of probing unavailable generic
BigFloat pivoted Cholesky. The final Task_Low08 Float64 guard also passed with
a valid original-coordinate
certificate in 28 iterations and 33.846 seconds solver time. The remaining
high-precision bottleneck is the exact equality normal system, especially
Gram construction and native BigFloat Cholesky at 1,024 bits. Reciprocal
caching improved isolated kernels but changed the adaptive trajectory, made
the relative gap 5.48 times weaker, and did not improve end-to-end time, so it
was reverted. No further untested high-risk kernel is enabled by default.
