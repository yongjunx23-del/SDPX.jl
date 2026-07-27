# CPU/HPC and conservative preprocessing report

Date: 2026-07-27

SDPX version: 0.2.0 release candidate

Julia: 1.12.6

Cluster policy: normal queue, one multicore node per job

## Outcome

This pass retained one new numerical-data optimization and one frontend
factorization reuse:

- sparse equilibration copies and scales only active coefficient matrices and
  shares one read-only empty CSC matrix per block;
- equality presolve reuses its already verified normalized dependency
  coefficients when constructing the multiplier reconstruction map.

On the canonical medium CSDR model, the sparse equilibration change reduced
the median call time from 1.241 to 0.633 seconds (1.96x) and cumulative
allocation from 685.8 to 269.7 MB (60.7% less). The scaled-problem checksums
were identical.

The new conservative preprocessing pipeline does not alter the numerical hot
path for either maintained benchmark. Task_Low08 has no scalar bounds and the
medium CSDR model has neither scalar bounds nor explicit equalities. The
Task_Low08 and medium solve comparisons therefore serve as no-change
regressions, not as claims of an additional Schur or KKT speedup.

Previously tested kernel, backend, and thread-count alternatives were not
repeated. The existing choices remain:

- dense Cholesky for Task_Low08 because sparse Cholesky fills completely;
- the exact reduced-arrow `Float64x4`/BigFloat kernels for medium CSDR;
- OpenBLAS on the validated AMD EPYC platform;
- 16 Julia plus 16 BLAS threads with NUMA interleaving for Float64
  Task_Low08;
- at most eight Julia workers and one BLAS thread for the medium
  extended-precision arrow;
- no `Float64x8` selection because MultiFloats 3.2.6 lacks the required
  scalar multiplication path and the validated nodes expose AVX2, not
  AVX-512.

## Problem classification

### Task_Low08

| Property | Value |
|---|---:|
| Variables | 6,119 |
| Supplied equalities | 482 |
| Equality rank | 394 |
| PSD blocks | 32 |
| Stored coefficient nonzeros | 228,325 |
| Aggregate PSD pattern density | 99.84% |
| Final Schur lower-triangle density | 84.26% |
| Scalar bound blocks | 0 |

The preprocessing classifier rejects bound elimination and chordal
decomposition. Equality presolve removes 88 dependent columns after verifying
the normalized column relation and right-hand side. The resulting Schur system
is dense enough that CHOLMOD becomes 100% filled; dense Cholesky remains 1.83x
faster than conversion plus sparse numeric refactorization.

### Medium CSDR `J/K/Na/Nmu = 32/4/16/100`

| Property | Value |
|---|---:|
| Variables | 1,844 |
| Shared / local variables | 144 / 1,700 |
| Equalities | 0 |
| PSD blocks | 1,700 of dimension 2 |
| Active variables per block | 145 |
| Active coefficient incidences | 246,500 |
| Dominant coefficient pattern | `(a12,a22)`, 99.3% |

This is the exact singleton-local block-arrow structure selected by the
existing reduced coefficient-space elimination and combined triangular SYRK.
The preprocessing stages correctly report no structural change.

## New implementation

### Typed structural pipeline

`preprocess(problem, options)` now executes typed stages for:

1. exact scalar-bound extraction and merging;
2. exact fixed-variable elimination;
3. exact zero, duplicate, and proportional equality cleanup;
4. arithmetic-aware numerical equality presolve;
5. scaling after structural reduction;
6. analysis-only primal/dual and chordal cost estimation; and
7. original-coordinate warm-start and result reconstruction.

The plan, map, and report retain arithmetic type `T`. BigFloat copies use
independently owned MPFR objects. Approximate fixed-variable removal and
near-duplicate equality removal are deliberately disabled.

MOI `GreaterThan`, `LessThan`, and `Interval` bounds use
`CompactScalarCoefficientVector`, which preserves
`Asp[block][variable]` compatibility without allocating a length-`m`
reference vector for every scalar block.

### Active-only sparse equilibration

The previous equilibration copy constructed a new CSC object for every
`L * m` slot, including structurally empty coefficients. The retained path:

- allocates one empty CSC matrix per block;
- installs that read-only object in every inactive slot;
- copies only matrices listed in `cons.active[block]`;
- computes variable maxima from active incidences;
- scales only active nonzero arrays; and
- rebuilds packed `2x2` and COO caches from the unchanged active lists.

No shared matrix is written. This is safe for BigFloat because the shared
empty matrices contain no mutable scalar entries.

### Equality dependency reuse

The normalized least-squares coefficients computed to validate each dropped
equality are now passed directly to the reconstruction-map builder. The
retained equality matrix is no longer QR-factorized a second time. Ambiguous
rank decisions still retain the original equalities, and only an exact
verified relation can certify an inconsistent right-hand side.

## New benchmark results

All A/B rows within a table ran in the same PBS allocation, with identical
model data, arithmetic, thread settings, and timing boundaries.

### Sparse equilibration

Canonical medium CSDR, Float64x4, eight Julia threads, one BLAS thread, ten
measurements per implementation after warm-up:

| Implementation | Median time | Allocation | Relative speed |
|---|---:|---:|---:|
| Previous `L * m` empty-matrix copies | 1.241448 s | 685,817,600 B | 1.00x |
| Active-only copies | **0.633317 s** | **269,675,920 B** | **1.96x** |

Both paths produced checksum `-20997.267101906484`.

### Bound-heavy frontend

Synthetic MOI model with 5,000 variables and one interval per variable:

| Quantity | Result |
|---|---:|
| Scalar PSD blocks | 10,000 |
| Compact blocks | 10,000 |
| MOI copy | 1.820879 s / 58,973,536 B |
| Structural preprocessing | 0.001489 s / 1,504,184 B |
| Coefficient nonzeros | 10,000 |
| Historical reference-grid floor | 400,000,000 B |

The historical floor counts references only; it excludes the sparse matrix
objects that the old representation also required.

### Task_Low08 frontend

| Stage | Time | Allocation | Result |
|---|---:|---:|---|
| Binary input | 0.461322 s | 369,314,248 B | checksum-verified model |
| Ingest | 0.394430 s | 372,297,008 B | sparse coefficients |
| Structural preprocessing | 0.019865 s | 4,397,480 B | no structural change |
| Equality presolve | 0.576270 s | 1,023,482,120 B | 482 to 394 |

Input and ingest allocation are dominated by deserialization and construction
of the benchmark representation, not by the solver iteration.

### Task_Low08 equality presolve A/B

Ten warm measurements per source, one Julia thread and one BLAS thread in the
same PBS allocation:

| Source | Median time | Median allocation | Equalities | Feasibility diagnosis |
|---|---:|---:|---:|---|
| Previous release | 0.996255 s | 1,049,379,120 B | 482 to 482 | incorrect `inconsistent=true` |
| Candidate | **0.694331 s** | **1,023,468,440 B** | **482 to 394** | correct `inconsistent=false` |

The candidate is 1.43x faster and allocates 2.5% less while fixing the
important diagnosis error. The root cause was a normalized QR cancellation
residual of about `3.3e-16` being compared with a vanishing local right-hand
side scale. The new check uses a global normalized RHS scale, retains
ambiguous relations, and reuses the verified dependency coefficients when
building the reconstruction map.

### Task_Low08 no-change solve regression

Float64, 16 Julia threads, 16 OpenBLAS threads, NUMA interleaving, one complete
warm-up per source and two interleaved measurements:

| Source | Median solve | Iterations | Objective | Certificate |
|---|---:|---:|---:|---|
| Previous release | 45.710 s | 24 | 0.6532912858859853 | valid |
| Preprocessing candidate | 47.053 s | 24 | 0.6532912858859853 | valid |

The 2.9% timing difference is not a retained hot-kernel regression or
improvement: Schur and KKT code is identical, structural preprocessing takes
about 0.006 seconds, and the variation appears in unchanged predictor and
corrector phases. Both versions returned relative gap `4.27176e-7`, maximum
equality violation `2.73155e-11`, minimum primal PSD eigenvalue
`-1.57406e-9`, and minimum dual PSD eigenvalue `4.06998e-14`.

After synchronizing the exact final source, one additional warm-up plus
measured regression completed in 45.714 seconds with the same 24 iterations,
objective, residuals, PSD eigenvalues, and valid certificate. This confirms
that the earlier 2.9% A/B difference was node/runtime variation rather than a
stable slowdown.

### Medium no-change solve regression

Float64x4, eight Julia threads, one BLAS thread, three interleaved repetitions:

| Presolve | Median solve | Preprocess | Allocation | Iterations | Certificate |
|---|---:|---:|---:|---:|---|
| Off | 15.815 s | 0.000253 s | 143,116,680 B | 41 | valid |
| Auto | 16.508 s | 0.000926 s | 143,323,400 B | 41 | valid |

The three solve measurements varied by about two seconds because of shared
node load; preprocessing itself remains below one millisecond and reports
`changed=false`. Both routes returned the same objectives, residuals, gap,
and PSD certificate.

## Existing CPU/HPC results retained

### Task_Low08 Float64 scaling

The prior controlled sweep used one complete warm-up, two measured solves,
OpenBLAS, and a 128-core reservation:

| Julia / BLAS threads | NUMA | Median solve | Speedup | Peak RSS |
|---|---|---:|---:|---:|
| 1 / 1 | default | 86.981 s | 1.00x | 1.875 GiB |
| 2 / 2 | default | 54.464 s | 1.60x | 3.425 GiB |
| 4 / 4 | default | 38.643 s | 2.25x | 4.539 GiB |
| 8 / 8 | default | 33.601 s | 2.59x | 5.692 GiB |
| 16 / 16 | interleave | **27.401 s** | **3.17x** | 5.769 GiB |
| 32 / 16 | interleave | 27.672 s | 3.14x | 5.912 GiB |
| 64 / 16 | default | 28.364 s | 3.07x | 6.205 GiB |
| 128 / 16 | interleave | 30.152 s | 2.88x | 8.719 GiB |

Task_Low08 has only 32 PSD blocks. Beyond 32 Julia workers, synchronization
and memory traffic outweigh additional block parallelism.

### BLAS backend selection

Equal eight-core, eight-iteration Task_Low08 measurements on the dual AMD EPYC
node:

| Backend | Time | Decision |
|---|---:|---|
| OpenBLAS | **10.816 s** | retained |
| MKL | 13.018 s | benchmark-only |
| BLIS | 21.999 s | rejected |

AOCL was unavailable on the cluster and was not installed into the release
environment. OpenBLAS remains the platform default; backend selection is
documented as hardware-specific.

### Extended precision

The retained medium Float64x4 reduced-arrow solver scales from 51.479 seconds
at one thread to 11.728 seconds at eight threads. Its exact `3400 x 144`
four-lane triangular SYRK scales from 0.6151 to 0.0802 seconds from one to
eight workers, with zero relative error.

Native BigFloat256 medium solves scale from 205.202 seconds at one worker to
86.752 seconds at eight ownership-safe tile workers. General BigFloat
problems remain serial; only exact disjoint reduced-arrow panels and Schur
tiles are parallel. The 192-bit opt-in working-precision run took 80.703
seconds and retained the 41-iteration certificate, but automatic precision
reduction remains conservative.

Task_Low08 Float64x4 uses lower-triangle sparse outer products because all
packed-panel crossover decisions report `sparse_outer_product_cheaper`.
Native BigFloat Task_Low08 is not launched in a 64 GiB allocation: the
planner estimates 95.72 GiB before adequate safety headroom.

## Automatic decisions and crossover rules

- Scalar bounds are extracted only from one-active-variable `1x1` PSD blocks.
- Fixed variables require exactly equal typed lower and upper bounds.
- Near-duplicate equalities are diagnostic only.
- Chordal analysis stops when aggregate PSD density predicts no profitable
  clique decomposition.
- Float64 never enters extended-precision BLAS kernels.
- Float64x4/BigFloat panel paths require compatible structure, sufficient
  dimensions and work, shared-Schur density, predicted speedup, available
  workers, and packing storage within the memory budget.
- Exact singleton-local arrows require at least 20 shared columns, at least
  `5e4` weighted work, shared density at least 0.10, Schur density at least
  0.05, and predicted speedup at least 1.12x.
- Dense Task_Low08 uses 16 Julia and 16 BLAS threads on the validated dual EPYC
  platform; medium Float64x4 and BigFloat use eight Julia workers and one BLAS
  thread.

## Numerical validation

The final local package suite passed 2,124 tests with one pre-existing broken
test marker and no failures. Focused preprocessing and solver regression
suites passed 69/69 and 44/44 assertions. They cover Float64, Float64x4, and
BigFloat reconstruction, MOI signed interval duals, exact equality cleanup,
ambiguous rank fallback, MPFR ownership, sparse active-only copies, and
original-coordinate certificates.

The cleaned cluster candidate passed 2,179/2,179 package tests in job
`194312.node220`. The difference in test count comes from platform-dependent
extensions and gates. The exact final-source Task_Low08 regression passed in
job `194311.node220`.

Task_Low08 and medium validation values are listed in the benchmark tables
above. All retained solver runs terminated `Optimal`; no stopping tolerance
or certificate threshold was weakened.

## Rejected or deferred approaches

- Sparse Task_Low08 Cholesky: rejected because symbolic fill reaches 100% and
  conversion plus numeric factorization is 1.83x slower than dense Cholesky.
- Per-block GEMM for Task_Low08: rejected because active patterns are too
  fragmented and packing is not amortized.
- Generic packed Float64x4 SYRK on Task_Low08: rejected by measured crossover;
  sparse outer products remain faster.
- MKL and BLIS on the validated EPYC node: slower than OpenBLAS.
- More than 32 Julia Task_Low08 workers: slower and more memory intensive.
- Float64x8: unsupported by the current MultiFloats arithmetic and not
  appropriate for the AVX2-only validated nodes.
- BigFloat `Threads.@threads` over existing loops: rejected because mutable
  MPFR scalar ownership cannot be guaranteed. Only disjoint owned tiles are
  used.
- Mixed Float64x4 factorization for BigFloat medium CSDR: refinement stalled
  and safely fell back; it remains opt-in.
- Matrix-free PCG for the 144-square medium shared system: rejected because
  robust dense Cholesky is already small and faster.

## Remaining bottlenecks

1. Task_Low08 remains dominated by Schur assembly and dense Schur Cholesky.
   The useful scale ceiling is memory bandwidth plus 32 independent PSD
   blocks, not the node's nominal core count.
2. Task_Low08 equality presolve is fast in wall time but still allocates about
   1.06 GB cumulatively because generic pivoted QR and verification materialize
   normalized dense work arrays.
3. Medium Float64x4 is now dominated by the reduced-panel SYRK and residual
   work; the 144-square factorization is no longer the bottleneck.
4. Medium BigFloat remains dominated by MPFR Schur construction. More
   parallelism requires the same exclusive-tile ownership discipline and
   should not be generalized to arbitrary BigFloat loops.
5. Full BigFloat Task_Low08 requires a packed or out-of-core factorization
   design before it is practical under a 64 GiB limit.

## Files changed in this pass

- `src/preprocessing.jl`
- `src/types.jl`
- `src/solve.jl`
- `src/pipeline.jl`
- `src/ingest.jl`
- `src/moi_wrapper.jl`
- `src/lp_solver.jl`
- `src/chordal.jl`
- `src/nullspace.jl`
- `src/SDPX.jl`
- `test/preprocessing_regressions.jl`
- `test/solver_regressions.jl`
- `test/runtests.jl`
- `bench/preprocessing/benchmark_frontend.jl`
- `bench/preprocessing/benchmark_equality_presolve.jl`
- `bench/preprocessing/benchmark_medium_preprocess.jl`
- `bench/preprocessing/benchmark_sparse_equilibration.jl`
- `bench/lattice_bootstrap/benchmark_sdpx_float64_solve.jl`
- `docs/preprocessing.md`
- `docs/cluster-workflow.md`
- `README.md`
- `CHANGELOG.md`
