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
5. solver, Gram/Schur kernel, KKT backend, thread schedule, and memory-budget
   selection;
6. optional adaptive interior-point parameter control;
7. phase timing, workspace estimation, warnings, and result reconstruction.

Pure `1x1` cone models are solved by a dedicated scalar Mehrotra
predictor-corrector LP engine. Standard scalar inequalities supplied through
JuMP/MOI are converted directly to that representation. SOC constraints are
recognized and currently use an exact PSD arrow lift. General PSD blocks use
the existing SDP engine, so its Float64 numerical path is unchanged.

## Presolve

Equality columns in `B` are normalized independently and reduced using
column-pivoted QR in the problem's arithmetic type. This avoids narrowing away
a direction that is meaningful to Float64x4 or BigFloat. Discarded columns are
checked against both the retained basis and their right-hand sides in the same
arithmetic; an ambiguous rank decision keeps all equalities. An inconsistent
system returns `InfeasibleCert` before factorization. Reduced dual multipliers
are mapped back
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

For `Float64xN` and `BigFloat`, the default remains the serial weighted
outer-product path. Setting `extended_precision_blas=:auto` permits the
planner to choose the packed blocked triangular `syrk!` path only when its
predicted benefit exceeds packing cost and it fits the memory budget.
`extended_precision_blas=:on` is intended for diagnostics and still cannot
override the memory-safety check. Float64 is never redirected by this option.

The equality-free LP Newton matrix is positive definite after regularization,
so SDPX factors it with Cholesky and reuses the kernel triangular solves.
Equality-constrained LPs retain dense LU because their augmented KKT matrix is
indefinite.

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

## High-precision ownership and allocation policy

Julia `BigFloat` values are mutable. Solver workspaces are therefore created
with independent scalar objects instead of relying on
`zeros(BigFloat, ...)` or `fill!`, which can install the same object into
multiple slots. Internal `*_owned!` operations may mutate destinations only
after that ownership invariant has been established.

The dedicated BigFloat layer covers copying, zeroing, fused vector updates,
matrix products, triangular solves, Cholesky factorization/solve, Schur and KKT
right-hand sides, weighted LP Hessian/KKT assembly, and the optional serial
blocked triangular `syrk!`/`gemm!` path. This removes temporary MPFR products
and sums from dominant inner loops without reintroducing aliasing. Input
conversion, result construction, and operations that must create independent
output values may still allocate.

BigFloat remains one-threaded. Fixed-width `Float64xN` types use the multicore
scheduler when their precision and Float64 exponent range are sufficient.

## Extended-precision crossover and memory budget

The packed-kernel selector uses arithmetic family, dimensions, active density,
average nonzeros, expected Schur density, requested threads, and packing bytes.
The current conservative automatic gates are:

| Gate | Fixed-width extended | BigFloat |
|---|---:|---:|
| Minimum columns | 32 | 20 |
| Minimum pair-row work | 200,000 | 50,000 |
| Minimum expected Schur density | 0.20 | 0.05 |
| Minimum predicted speedup | 1.18x | 1.12x |
| Sparse average fill required | 0.42 | 0.62 |

Fixed-width sparse blocks of dimension at most two retain their specialized
small-block path. Every packed block must also fit its cumulative memory
budget.

Available memory is the minimum usable signal from host free memory, Linux
cgroup v1/v2 counters, and the optional `SDPX_MEMORY_LIMIT_BYTES` ceiling.
The requested packing budget is
`extended_precision_memory_fraction × available` (default `0.10 ×`), capped
at half of the available amount so at least half remains as general headroom.
If no reliable signal exists, optional packing is disabled instead of risking
an out-of-memory kill.

For the fused exact-arrow `2x2` SDP path, transformed panels and packed pair
buffers are not allocated because the compute-and-scatter kernel consumes
neither. On the target 4,100-block sparse model this avoids a predicted
9.08 GB pair buffer; that is a workspace estimate, not a full-solve memory
measurement. This specialization takes precedence over optional packed
extended BLAS for both Float64x4 and BigFloat. Execution diagnostics report
`gram_kernel=:fused_arrow_2x2` and
`gram_kernel_reason=:fused_arrow_specialized`.

## Automatic LP cold-start parameters

The dedicated LP engine uses a deterministic distance diagnostic before the
first iteration:

```text
max(max_i |h_i| / ||G_i||_inf, max_j |b_j| / ||B_j||_inf).
```

This quantity is invariant to positive rescaling of an individual constraint.
With `parameter_policy=:auto`, an indicator at most `1000` selects
`beta=1/50, gamma=99/100`; larger or non-finite values retain the configured
conservative parameters. The guard applies to BigFloat as well as
fixed-exponent arithmetic: extra precision prevents rounding loss, but does
not by itself globalize a very distant infeasible start.

On the measured Float64 LPs this reduced iterations from 14 to 10 at
80-by-400 and from 18 to 14 at 256-by-4,000. A 256-bit BigFloat 32-by-128 LP
fell from 25 to 15 iterations and was `1.73x` faster. The crossover and
distant-start sweeps are reported in
[`PARAMETER_SELECTION.md`](../bench/automatic_pipeline/PARAMETER_SELECTION.md).
`parameter_policy=:fixed` remains an exact override.

## Adaptive beta and gamma

`parameter_strategy=:adaptive` enables a guarded controller:

- beta uses the affine-predictor complementarity ratio cubed and the observed
  complementarity reduction squared;
- gamma responds to accepted primal/dual steps, backtracking count, and
  feasibility progress;
- beta uses nominal bounds `[0.02, 0.50]`, expanded only when needed to retain
  the configured fallback (for example, the large-arrow `β=0.01` profile);
- gamma uses nominal bounds `[0.65, 0.95]`, likewise expanded only to retain
  the configured fallback;
- the first two infeasible-start iterations retain the configured defaults;
- repeated non-finite or explosive complementarity behavior disables
  adaptation and restores the configured defaults.

Every accepted iteration is stored in `result.parameter_history`, including
beta, gamma, predictor quality, complementarity reduction, step sizes,
backtracking count, residuals, and fallback state.

Adaptive parameters remain opt-in. In the final benchmark they matched the
fixed iteration count and runtime within noise on the analytic LP, but needed
14 extra iterations on the representative CSDR SDP even after the guarded
fallback restored convergence.

## Benchmark summary

Host: Apple M4, Julia 1.12.6, a process started with eight Julia workers on a
four-hardware-thread test allocation, and one BLAS thread. Times are medians of
five warmed solves. The eight-worker rows are therefore oversubscribed and are
not eight-core scaling claims.

### Dedicated LP path

Dense LP: 80 variables and 400 inequalities.

| Path | Iterations | Runtime | Speedup | Allocations |
|---|---:|---:|---:|---:|
| General SDP engine | 16 | 57.23 ms | 1.00x | 18.42 MB |
| Dedicated LP, fixed | 14 | 15.06 ms | 3.80x | 14.25 MB |
| Dedicated LP, adaptive | 14 | 15.22 ms | 3.76x | 14.25 MB |

All three returned an objective within `6.5e-9` of the constructed optimum.
The fixed LP dual residual was below `2e-15`; the adaptive result remained
within tolerance at `5.0e-12`.

### High-precision LP Hessian kernel

This isolated weighted-Hessian benchmark compares the former row-wise
construction with the current owned/blocked path. Float64x4 rows force the
optional blocked kernel to measure its scaling; BigFloat compares the former
allocating scalar loop with the current serial owned-MPFR implementation.

| Arithmetic | Julia workers | Reference | Optimized | Speedup | Allocation, reference → optimized | Relative error |
|---|---:|---:|---:|---:|---:|---:|
| Float64x4 | 1 | 43.177 ms | 37.966 ms | 1.14× | 0 → 0 B | `3.31e-64` |
| Float64x4 | 2 | 43.478 ms | 20.198 ms | 2.15× | 0 → 1,312 B | `3.31e-64` |
| Float64x4 | 4 | 44.927 ms | 11.876 ms | 3.78× | 0 → 2,336 B | `3.31e-64` |
| Float64x4 | 8 requested | 44.984 ms | 9.557 ms | 4.71× | 0 → 4,192 B | `3.31e-64` |
| BigFloat, 256 bit | 1 | 70.929 ms | 23.331 ms | 3.04× | 114,573,424 → 336 B | 0 |

### Large LP multicore scaling

Dense LP: 256 variables and 4,000 inequalities.

| Threads | Kernel | Runtime | Speedup | Allocations |
|---:|---|---:|---:|---:|
| 1 | BLAS SYRK | 631.89 ms | 1.00x | 440.37 MB |
| 2 | Parallel BLAS panels | 601.79 ms | 1.05x | 440.40 MB |
| 4 | Parallel BLAS panels | 615.71 ms | 1.03x | 440.41 MB |
| 8 | Parallel BLAS panels | 605.42 ms | 1.04x | 440.44 MB |

All rows took 18 iterations and returned identical objectives and residuals.
The modest end-to-end gain shows that dense KKT factorization, certificate
construction, and vector work now dominate this test; adding workers to the
panel phase alone has limited value.

### Fixed versus adaptive SDP

Sparse CSDR PSD dual: 371 variables and 360 `2x2` PSD blocks.

| Strategy | Iterations | Runtime | Objective | Relative gap |
|---|---:|---:|---:|---:|
| Fixed beta=0.1, gamma=0.8 | 19 | 14.37 ms | 15.5589572374 | 8.07e-9 |
| Adaptive with fallback | 33 | 24.05 ms | 15.5589576192 | 5.55e-8 |

Both results passed the final certificate. The adaptive controller detected
one-sided cone-boundary stagnation and restored the fixed fallback; without
that guard this run stalled. Fixed parameters remain the default because the
guarded adaptive strategy was 67% slower and produced a larger, though still
valid, relative gap.

### Schur scheduling

The current isolated Schur benchmarks are summarized in
[`bench/threading/RESULTS.md`](../bench/threading/RESULTS.md). On the measured
four-hardware-thread Apple M4 process, the medium Float64 dense case reached
1.64x at two workers, the medium Float64x4 dense case reached 4.08x at eight
requested workers, and the medium Float64x4 sparse exact-arrow case reached
3.73x at eight requested workers. The eight-worker rows oversubscribed the
host and must not be presented as eight-core scaling.

Maximum relative Schur error was `2.67e-16` for dense Float64,
`3.57e-65` for dense Float64x4, `1.67e-15` for sparse Float64, and
`1.33e-64` for sparse Float64x4. These are kernel benchmarks, not complete
solve times.

## Initialization result

In a fixed-policy sweep on the same CSDR model, `OmegaP=OmegaD=1` returned
`Stalled` after 15 iterations with exploding residuals, while
`OmegaP=OmegaD=10` converged in 19 iterations. The automatic sparse `2x2`
profile now chooses the larger of 10 and the measured PSD-block norm. The large
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

The JLD2 file contains a top-level dataset named `spectrum`. Its versioned
payload has three fields:

- `format_version`: currently `1`;
- `metadata`: solve-wide status, objective, residual, precision, block-shape,
  and warning information;
- `records`: the per-block eigenvalue records.

Generic extended-precision matrices are projected to Float64 only for this
optional eigenvalue post-processing because Julia's standard library does not
provide a generic symmetric eigensolver for every scalar type.

## Remaining bottlenecks

1. The large dense SDP Schur matrix and its factorization still dominate
   `Task_Low08`; the Float64 path was intentionally left numerically unchanged.
2. Equality-constrained LPs use dense LU. A null-space/range-space selector
   and sparse LDL backend should improve models with many variables but few
   equalities; equality-free LPs already use Cholesky.
3. LP panel GEMM computes full output panels. A lower-triangle-only blocked
   BLAS-3 kernel would reduce arithmetic further while retaining multicore
   scaling.
4. The native SOCP path currently uses an exact PSD lift. A dedicated
   Nesterov-Todd SOC scaling kernel would reduce memory and factorization work.
5. Presolve currently removes equality dependence and scalar-row redundancy;
   bound propagation, singleton substitution, coefficient strengthening, and
   chordal SDP decomposition remain future work.
6. BigFloat is serial, and large non-arrow high-precision SDPs still require
   dense quadratic Schur/KKT storage. No full BigFloat `Task_Low08` solve is
   claimed by this document.
7. Nested solves in one process are not supported because BLAS thread count is
   process-global. Use separate processes for concurrent instances.
