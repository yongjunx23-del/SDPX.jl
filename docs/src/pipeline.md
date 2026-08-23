# Automatic pipeline

Every public `optimize!` call compiles one typed model and builds an
inspectable `ExecutionPlan`. The pipeline performs:

1. cone, storage, arithmetic, and size classification;
2. rank-revealing equality presolve with a consistency check and dual
   reconstruction map;
3. redundant scalar-cone row elimination for LPs;
4. automatic scaling and equilibration selection;
5. solver, Gram/Schur kernel, KKT backend, thread schedule, and memory-budget
   selection;
6. guarded adaptive interior-point parameter control with a fixed fallback;
7. phase timing, workspace estimation, warnings, and result reconstruction.

Pure scalar-cone models are lowered once to the dedicated Mehrotra LP engine.
Pure SOC/RSOC models use NativeSOC in original Lorentz coordinates; general
Lorentz blocks use the Nesterov--Todd path, while exactly certified fixed-trace
Q3 cells use the compact HKM specialization (see [socp.md](socp.md)). Pure PSD
models use the SDP engine. Mixed cone families fail during classification;
there is no production SOC-to-PSD lift or model retry.

## Presolve

Equality columns in `B` are normalized independently and reduced using
column-pivoted QR in the problem's arithmetic type. This avoids narrowing away
a direction that is meaningful to Float64x4 or BigFloat. Discarded columns are
checked against both the retained basis and their right-hand sides in the same
arithmetic; an ambiguous rank decision keeps all equalities. An inconsistent
system returns `InfeasibleCert` before factorization. Reduced dual multipliers
are mapped back to the original equality ordering, with zero assigned to
non-unique discarded multipliers.

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

For `Float64xN` and `BigFloat`, `extended_precision_blas=:auto` permits the
planner to choose the packed blocked triangular `syrk!` path only when its
predicted benefit exceeds packing cost and it fits the memory budget.
`extended_precision_blas=:on` is intended for diagnostics and still cannot
override the memory-safety check. Float64 is never redirected by this option.

The equality-free LP Newton matrix is positive definite after regularization,
so SDPX factors it with Cholesky and reuses the kernel triangular solves.
Equality-constrained LPs retain dense LU because their augmented KKT matrix is
indefinite. Small Float64 SDPs containing only `1×1`/`2×2` blocks and fewer
than 1,000 variables are kept serial because repeated task barriers cost more
than the block kernels. Fixed-width extended arithmetic retains multicore
block scheduling.

## High-precision ownership and allocation policy

Julia `BigFloat` values are mutable. Solver workspaces are therefore created
with independent scalar objects instead of relying on `zeros(BigFloat, ...)`
or `fill!`, which can install the same object into multiple slots. Internal
`*_owned!` operations may mutate destinations only after that ownership
invariant has been established.

The dedicated BigFloat layer covers copying, zeroing, fused vector updates,
matrix products, triangular solves, Cholesky factorization/solve, Schur and
KKT right-hand sides, weighted LP Hessian/KKT assembly, and blocked triangular
`syrk!`/`gemm!` paths. Input conversion, result construction, and operations
that must create independent output values may still allocate.

## Extended-precision crossover and memory budget

The packed-kernel selector uses arithmetic family, dimensions, active density,
average nonzeros, expected Schur density, requested threads, and packing
bytes. The current conservative automatic gates are:

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
`extended_precision_memory_fraction × available` (default `0.10 ×`), capped at
half of the available amount so at least half remains as general headroom. If
no reliable signal exists, optional packing is disabled instead of risking an
out-of-memory kill.

For the fused exact-arrow `2×2` SDP path, transformed panels and packed pair
buffers are not allocated because the compute-and-scatter kernel consumes
neither. This specialization takes precedence over optional packed extended
BLAS for both Float64x4 and BigFloat. Execution diagnostics report
`gram_kernel=:fused_arrow_2x2` and
`gram_kernel_reason=:fused_arrow_specialized`.

## Automatic cold-start parameters

`parameter_policy=:auto` runs one generic automatic Mehrotra controller before
the first iteration. It selects no benchmark-, size-, cone-, or
precision-specific parameter profile: `beta`, `gamma`, `predictor`, and
`parameter_strategy` come from the `SolverOptions` defaults or user choices.
After presolve and scaling, the selected formulation/provider factors an
identity-metric initialization system once and solves its primal and dual
right-hand sides. Cone-native strict-interior shifts are followed by the
smallest aggregate identity shift that raises orthant/PSD starts, and Lorentz
sides still within the typed cone-vertex envelope, to unit identity mass, then
by deterministic complementarity cross-centering. This floor prevents a valid
affine point at the cone vertex from remaining at machine-epsilon scale without
renormalizing an already balanced Lorentz point; it is separate from barrier
degree. An
accepted regularized or mixed-precision SDP factor is reused for bounded
structured residual corrections when either original-KKT right-hand side is
above the existing cold-start gate. `Omega_p` and
`Omega_d` are not read by this automatic path. The public resolver reports
`profile=:post_scaling_mehrotra`; the controller is deferred in the plan as
`:automatic_mehrotra` and resolved exactly once after scaling as
`:post_scaling_mehrotra`; and the adaptive iteration controller uses the
generic 0.50 sigma cap (`adaptive_sigma_max` remains the expert override).
`parameter_policy=:fixed` uses the supplied values exactly and records
`:user_fixed`.

This is separate from the storage/structure classification reported by
`StructureAnalysis.profile`, which describes data layout and kernel selection
rather than iteration parameters.

## Adaptive Newton parameters

`parameter_strategy=:adaptive` selects a bounded Mehrotra `sigma` from the
affine complementarity ratio, centrality, factor quality, and recent progress.
It separately selects primal and dual fraction-to-boundary values, the
backtracking contraction, and the refinement target/cap. The complete fixed
predictor/corrector path is restored after non-finite diagnostics,
rank-revealing equality factorization, or unstable progress.

When history retention is enabled, `iteration_history(result)` contains every
accepted iteration, including `sigma`, `mu`, `mu_aff`, affine and accepted
steps, the separate step safeguards, residual progress, factor/PSD-margin
proxies, regularization, refinement, and fallback provenance. See
[Adaptive Interior-Point Parameter Policy](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/adaptive-parameter-policy.md) for
the audit, equations, exact bounds, and arithmetic-specific behavior.

## Result and diagnostics

The public result exposes terminal status, objectives, certificate residuals,
the immutable execution plan, and retained primal/dual data through the v0.5
accessors. Full diagnostics add classification, presolve facts, analytical
workspace estimates, process peak RSS, selected algorithms, timings, warnings,
and refinement/fallback provenance. See [diagnostics](diagnostics.md) for the
retention and accessor contract.

## Remaining bottlenecks

1. Large dense SDP Schur matrices and their factorization dominate the heavy
   benchmark models; the Float64 numerical path is intentionally unchanged.
2. Equality-constrained LPs use dense LU. A null-space/range-space selector
   remains future work; sparse execution is intentionally restricted to
   equality-free frozen-CSC normal equations with provider-native Cholesky.
3. LP panel GEMM computes full output panels. A lower-triangle-only blocked
   BLAS-3 kernel would reduce arithmetic further.
4. General-dimensional native SOCP is production; general BigFloat work is
   serial, while exact singleton-local `2×2` arrows may use ownership-safe
   panel preparation and disjoint Schur tiles.
5. Presolve removes equality dependence and scalar-row redundancy; bound
   propagation, singleton substitution, coefficient strengthening, and chordal
   SDP decomposition remain future work.
6. Nested solves in one process are not supported because BLAS thread count is
   process-global. Use separate processes for concurrent instances.
