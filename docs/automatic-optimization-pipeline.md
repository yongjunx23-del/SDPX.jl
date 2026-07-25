# Automatic Optimization Pipeline

Date: 2026-07-25

## Overview

Every `solve` and `solve!` call now builds an inspectable `ExecutionPlan`.
The pipeline performs:

1. cone, storage, arithmetic, and size classification;
2. rank-revealing equality presolve with a consistency check and dual
   reconstruction map;
3. redundant scalar-cone row elimination for LPs;
4. automatic scaling and equilibration selection;
5. solver, Gram/Schur kernel, KKT backend, and thread schedule selection;
6. optional adaptive interior-point parameter control;
7. phase timing, workspace estimation, warnings, and result reconstruction.

Pure `1x1` cone models are solved by a dedicated scalar Mehrotra
predictor-corrector LP engine. Standard scalar inequalities supplied through
JuMP/MOI are converted directly to that representation. SOC constraints are
recognized and currently use an exact PSD arrow lift. General PSD blocks use
the existing SDP engine, so its Float64 numerical path is unchanged.

## Presolve

Equality columns in `B` are reduced using column-pivoted QR on a Float64
structural projection. Discarded columns are checked against both the retained
basis and their right-hand sides. An inconsistent system returns
`InfeasibleCert` before factorization. Reduced dual multipliers are mapped back
to the original equality ordering, with zero assigned to non-unique discarded
multipliers.

The LP presolver additionally removes:

- zero rows that are always satisfied;
- infeasible zero rows;
- positive scalar multiples with the same left-hand side, retaining the
  strongest lower bound.

## Dedicated LP kernels and multicore scheduling

The LP normal matrix is

```text
H = G' * Diagonal(z ./ s) * G.
```

For Float64 and one thread, SDPX packs `sqrt(z ./ s) .* G` once and calls BLAS
`syrk`. For sufficiently large models with single-threaded BLAS and more than
one requested Julia thread, variables are split into coarse column panels.
Each worker performs one independent BLAS-3 GEMM into a disjoint output panel.
There are no partial-matrix reductions, atomics, or locks. BigFloat remains
serial.

The current crossover for parallel LP panels is:

```text
Float64
requested threads > 1
BLAS threads == 1
inequalities * variables^2 >= 2,000,000
```

Small Float64 SDPs containing only `1x1`/`2x2` blocks and fewer than 1,000
variables are kept serial because repeated task barriers cost more than the
block kernels. Fixed-width extended arithmetic retains multicore block
scheduling. BigFloat is always serial.

## Adaptive beta and gamma

`parameter_strategy=:adaptive` enables a guarded controller:

- beta uses the affine-predictor complementarity ratio cubed and the observed
  complementarity reduction squared;
- gamma responds to accepted primal/dual steps, backtracking count, and
  feasibility progress;
- beta is clamped to `[0.02, 0.50]`;
- gamma is clamped to `[0.65, 0.95]`;
- the first two infeasible-start iterations retain the configured defaults;
- repeated non-finite or explosive complementarity behavior disables
  adaptation and restores the configured defaults.

Every accepted iteration is stored in `result.parameter_history`, including
beta, gamma, predictor quality, complementarity reduction, step sizes,
backtracking count, residuals, and fallback state.

Adaptive parameters remain opt-in. They reduced iterations on analytic LP and
SDP tests, but did not improve the representative CSDR SDP benchmark.

## Benchmark summary

Host: Apple Silicon, Julia 1.12.6, eight Julia threads available, one BLAS
thread unless noted. Times are medians of five warmed solves.

### Dedicated LP path

Dense LP: 80 variables and 400 inequalities.

| Path | Iterations | Runtime | Speedup | Allocations |
|---|---:|---:|---:|---:|
| General SDP engine | 18 | 44.33 ms | 1.00x | 5.56 MB |
| Dedicated LP, fixed | 14 | 2.74 ms | 16.19x | 1.74 MB |
| Dedicated LP, adaptive | 14 | 2.90 ms | 15.28x | 1.74 MB |

All three returned an objective within `6.3e-9` of the constructed optimum,
with primal and dual residuals below `3e-11`.

### Large LP multicore scaling

Dense LP: 256 variables and 4,000 inequalities.

| Threads | Kernel | Runtime | Speedup | Allocations |
|---:|---|---:|---:|---:|
| 1 | BLAS SYRK | 178.09 ms | 1.00x | 44.85 MB |
| 2 | Parallel BLAS panels | 174.40 ms | 1.02x | 44.87 MB |
| 4 | Parallel BLAS panels | 126.74 ms | 1.41x | 44.89 MB |
| 8 | Parallel BLAS panels | 121.89 ms | 1.46x | 44.92 MB |

The one-thread phase profile spent about 68% of time in Gram assembly. At
eight threads, Gram time fell from approximately 125 ms to 84 ms in a warmed
profile. Dense KKT factorization and vector operations are the next scaling
limits.

### Fixed versus adaptive SDP

Sparse CSDR PSD dual: 371 variables and 360 `2x2` PSD blocks.

| Strategy | Iterations | Runtime | Objective | Relative gap |
|---|---:|---:|---:|---:|
| Fixed beta=0.1, gamma=0.8 | 17 | 7.22 ms | 15.5589580088 | 5.65e-8 |
| Adaptive | 18 | 7.80 ms | 15.5589571819 | 9.03e-8 |

The automatic scheduler selected one thread; forcing eight threads roughly
doubled runtime on this latency-bound Float64 instance. Fixed parameters
therefore remain the default.

## Initialization result

On the same CSDR model, `OmegaP=OmegaD=1` failed at the 100-iteration limit
with exploding residuals. `OmegaP=OmegaD=10` converged in 17 iterations.
The automatic sparse `2x2` profile now selects 10/10. The large
equality-constrained dense-Schur profile used by `Task_Low08` selects the
previously validated asymmetric initialization `OmegaP=100`,
`OmegaD=0.001`, `predictor=:sdpb`.

## Result and optional spectrum

`SDPResult` contains status, objectives, residuals, iteration counts, phase
timings, and parameter history. `result.diagnostics` adds the classification,
execution plan, presolve summary, analytical workspace estimate, process peak
RSS, selected algorithms, and warnings.

Spectrum reconstruction is intentionally post-solve:

```julia
records = reconstruct_spectrum(result; source=:primal)
export_spectrum("spectrum.csv", result)
export_spectrum("spectrum.json", result)

using JLD2
export_spectrum("spectrum.jld2", result)
```

Generic extended-precision matrices are projected to Float64 only for this
optional eigenvalue post-processing because Julia's standard library does not
provide a generic symmetric eigensolver for every scalar type.

## Remaining bottlenecks

1. The large dense SDP Schur matrix and its factorization still dominate
   `Task_Low08`; the Float64 path was intentionally left numerically unchanged.
2. The LP KKT system uses dense LU. A null-space/range-space selector and
   sparse LDL backend should improve models with many variables but few
   equalities.
3. LP panel GEMM computes full output panels. A lower-triangle-only blocked
   BLAS-3 kernel would reduce arithmetic further while retaining multicore
   scaling.
4. The native SOCP path currently uses an exact PSD lift. A dedicated
   Nesterov-Todd SOC scaling kernel would reduce memory and factorization work.
5. Presolve currently removes equality dependence and scalar-row redundancy;
   bound propagation, singleton substitution, coefficient strengthening, and
   chordal SDP decomposition remain future work.
