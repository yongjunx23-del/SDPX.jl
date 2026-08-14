# TASK — Eliminate the NativeSOC FixedTrace performance regression on Full-unitraity-EFT

Continue from the actual current SDPX.jl worktree. Do not reset, overwrite, or
revert unrelated dirty changes. First inspect `git status`, current HEAD, and
the files below.

## Repository and external references

- SDPX worktree:
  `/Users/xuyongjun/Desktop/project/SDPX/SDPX-v041-legacy-la`
- Benchmark:
  `benchmark/bootstrap/Full-unitraity-EFT/`
- Historical study:
  `/Users/xuyongjun/Desktop/project/massless eft scalar/benchmarks/c00_nx3_primal_full_study/mac_bf256_zero_c00_j40_na15_nmu200_nx2_nalpha2_v040_worktree`
- Historical result:
  `<study>/RESULTS.md`
- Current MFLA:
  `/Users/xuyongjun/Desktop/project/MFLA`

MFLA is an external provider and may continue changing. Do not modify its
source. If an MFLA interface or performance improvement is genuinely needed,
add only a concise Markdown feedback file in the MFLA directory.

## Problem definition

Preserve the archived BigFloat256-generated Float64x4 coefficients and the
existing low-energy/equality reduction. Convert each PSD block exactly:

```text
[q r; r 2-q] >= 0    iff    (1, q-1, r) in Q3.
```

The reduced NativeSOCP has:

- 4,200 independent fixed-trace Q3 blocks;
- 8,400 variables, two local variables per block;
- 84 equalities after the archived 90-to-84 reduction;
- Float64x4 solve arithmetic;
- physical reference objective
  `30.4732058529286002611344264526887472`;
- required original-coordinate Lorentz certificate.

The benchmark must select `NativeSOC`, `fixed_trace_q3`, and the MFLA provider.
Production solving must not construct PSD matrices.

## Historical performance baseline

The one-thread v0.4 fixed-trace HKM-Q3 result was:

```text
solve!                         80.9359 s
timed core                    61.3077 s
iterations                          121
Schur assembly                48.2472 s
equality Gram                 37.6191 s
KKT factorization              0.2876 s
certificate valid                  true
```

The old equality Gram was formed once per iteration and its factor was reused
for predictor and corrector.

## Confirmed Round 5 regression

The first one-thread NativeSOC/MFLA run planned correctly but was stopped after
more than fourteen minutes. It had already accumulated 205,332,219 allocations
and 225 garbage collections. The interrupted stack was in MFLA transpose GEMV
called by `_native_soc_residuals!`.

Read `NEGATIVE_OPTIMIZATION.md` before changing code.

### Root cause 1 — accidental O(L*m) cone operations

Every fixed-trace cone is stored as a `3 x 8400` sparse matrix with two
nonzeros. Generic NativeSOC repeatedly calls provider GEMV on the complete
logical matrix:

```julia
cone.A * x
transpose(cone.A) * z
```

MFLA's dense GEMV scans every logical zero. Across 4,200 blocks this replaces
an O(L) local kernel with O(L*m) Float64x4 work. It affects residuals, RHS
contraction, predictor/corrector, and direction recovery.

### Root cause 2 — duplicate equality Gram/factor work

The fixed-trace `_native_soc_solve_kkt!` currently transforms the equality
panel, forms its SYRK Gram, and factors it separately for predictor and
corrector. The metric/factor is identical within an iteration. Prepare these
objects once per iteration and reuse them for both RHS.

## Required architecture

1. Extend the fixed-trace reduction/plan with an immutable compiled layout:
   active variable indices, the 2x2 tail coefficient map, fixed head/offset,
   block ownership, and any equality scatter facts.
2. Add fixed-trace-specific structure-of-arrays kernels for:
   - primal cone residual;
   - dual residual scatter;
   - predictor/corrector RHS contraction;
   - direction recovery;
   - fraction-to-boundary/update when beneficial.
3. These kernels must touch only two variables and three Lorentz coordinates
   per block. They must not invoke dense GEMV on `cone.A`.
4. Retain the general NativeSOC path unchanged for genuinely general cones.
5. Split KKT preparation from RHS solve:
   - local metric/factor once per iteration;
   - transformed equality panel once;
   - equality Gram SYRK once;
   - equality Cholesky/RRQR factor once;
   - predictor and corrector use the cached objects with separate RHS solves.
6. Keep MFLA on dense provider-owned work: equality panel SYRK, triangular
   operations, equality factorization, and solves. Do not pass sparse
   per-cone matrices into dense MFLA kernels.
7. Preserve explicit plan authority and fail-closed fallback semantics. No
   hidden provider switch, normal/augmented switch, PSD lift, or generic retry.
8. Preserve Float64x4 and BigFloat owned-storage rules. Do not introduce shared
   mutable BigFloat entries or ambient-precision dependence.

## Instrumentation

Add measured NativeSOC timing fields for at least:

- fixed local scaling/metric;
- local factorization;
- equality panel transform;
- equality Gram SYRK;
- equality factorization;
- predictor RHS solve;
- corrector RHS solve;
- block residual/contraction/recovery;
- allocations/workspace bytes.

Do not estimate or infer unavailable timings. `performance_trace` should expose
the measured values additively without changing existing fields.

## Test and profiling sequence

Do not immediately launch another 200-iteration solve.

1. Add tiny algebra tests comparing optimized fixed kernels against the
   existing generic Lorentz equations for Float64 and Float64x4.
2. Add a structural counter test proving no fixed hot-path kernel performs
   O(number_of_blocks * number_of_variables) work.
3. Add a factor-once/two-RHS test: one IPM iteration must form exactly one
   equality Gram and one equality factor while executing two RHS solves.
4. Run the real benchmark with `SDPX_BENCH_MAX_ITERATIONS=1`, then 5, then 20.
   Record time/allocation per iteration and verify the objective/residual trend.
5. Microbenchmark the real `8400 x 84` transformed panel:
   - current MFLA `syrk!`, 1 thread;
   - MFLA `syrk!`, 2 and 4 threads;
   - historical output-tiled kernel if still callable as a benchmark reference.
6. Only after the per-iteration gates show no regression, run the complete
   tolerance `1e-12` solve.

## Numerical acceptance gates

The complete result must satisfy:

- status `Optimal`;
- original-coordinate Lorentz certificate valid;
- physical objective agrees with the archived certificate interval;
- relative gap at most `1e-12`;
- no PSD representation in the production path;
- specialization exactly `fixed_trace_q3`;
- executed provider exactly MFLA for dense equality LA;
- no unplanned fallback;
- deterministic 1-thread result within the declared arithmetic tolerance.

## Performance gates

Use the historical one-thread run as the first hard comparison:

- first restore `solve! <= 80.94 s` and core `<= 61.31 s`;
- then target a meaningful improvement, preferably `solve! <= 65 s`;
- equality Gram must be formed once, not twice, per iteration;
- report iteration count separately from per-iteration time;
- report 1/2/4-thread results without mixing thread-count comparisons.

If the new NT controller uses more iterations, do not hide that by reporting
only kernel time. Separate:

```text
total time = setup + iterations * per-iteration work + certification.
```

Tune NT/controller parameters only after the hot-path complexity and duplicate
Gram bugs are fixed.

## Scope and safety

- Keep the benchmark inside
  `benchmark/bootstrap/Full-unitraity-EFT/`.
- Preserve unrelated Round 6 sparse/LP work already present in the worktree.
- Use lightweight Mac tests; do not submit cluster jobs in this task.
- Do not commit generated model binaries or benchmark output.
- Make small, reversible commits only after focused tests pass.

## Final report

Return:

1. exact root causes with source locations;
2. before/after algorithmic complexity;
3. changed files and preserved contracts;
4. focused test results;
5. 1/5/20-iteration profiling table;
6. full-solve correctness and performance table if the short gates pass;
7. MFLA versus historical Gram A/B;
8. remaining numerical/performance risks;
9. explicit verdict: regression fixed or not fixed.
