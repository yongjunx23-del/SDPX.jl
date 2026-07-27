# SDPX Full Code Audit, Implementation Status, and Update Plan

**Audit date:** 2026-07-25

**Repository:** `SDPX.jl`

**Scope:** numerical correctness, presolve, LP/SDP algorithms, sparse and
dense kernels, extended precision, multicore execution, automatic selection,
Julia/MOI interfaces, diagnostics, benchmarks, documentation, and release
readiness.

## Post-audit implementation update

This document preserves the evidence and recommendations recorded during the
2026-07-25 audit. The table below supersedes future-tense statements in later
sections where implementation has since landed.

| Area | Current state |
|---|---|
| Result integrity | Type-native equality presolve, dual reconstruction, original-coordinate certification, best-iterate retention, and structured termination diagnostics are implemented |
| Sparse exact-arrow path | Flat COO and packed `2x2` data are implemented; the fused path allocates neither transformed panels nor pair buffers |
| High precision | Owned, allocation-reusing BigFloat matrix/KKT/LP kernels are implemented and remain serial; Float64x4 uses exact requested worker counts |
| Memory planning | Optional packing uses the minimum usable signal from host availability, cgroup limits, and `SDPX_MEMORY_LIMIT_BYTES`, with safety headroom |
| Automatic control | Structural parameter profiles and guarded adaptive beta/gamma history are implemented; per-iteration adaptation remains opt-in |
| Interfaces and output | Active-only MOI construction, native/explicit spectrum reconstruction, atomic export, and configurable diagnostics are implemented |
| Still architectural | Lower-triangle tile ownership, persistent solve-scoped workers, general sparse/many-equality KKT, mixed-precision KKT, sparse HSD LP, and native SOCP remain future work |

Current portable evidence is in the
[extended-precision report](../bench/extended_precision_blas/REPORT.md),
[CSDR results](../bench/csdr_psd_dual/RESULTS.md),
[automatic-pipeline report](automatic-optimization-pipeline.md), and
[threading results](../bench/threading/RESULTS.md). Raw generated outputs stay
outside the tracked release surface.

## Executive decision

The first correctness and low-risk performance batch has now been
implemented. The LP orientation bug, false empty-cone infeasibility, relaxed
`Optimal` status, stale residuals, Float64 equality-presolve narrowing,
original-row validation, refinement rollback, warm-start multiplier mapping,
BigFloat precision scope, incomplete returned-state validation, and lost
termination metadata are covered by focused regressions.

The implementation also adds a cold original-coordinate result certificate,
cache-correct block-arrow updates, exact requested worker counts for
Float64x4, conservative free-memory crossover logic, active-only MOI
construction, and linear-memory spectrum export. A fresh Task_Low08 Float64
solve passes the corrected certificate.

The repository is not yet ready to make extended-precision kernels the
default. The largest remaining architectural gaps are a sparse
certificate-capable LP path, lower-triangular Schur tile ownership, a
solve-scoped scheduler, general sparse/many-equality KKT backends, and guarded
mixed-precision KKT factorization.

The largest medium-term opportunities are:

1. replace per-worker full `m x m` Schur buffers with lower-triangular tile
   ownership;
2. replace generic extended-precision KKT factorization with a guarded
   mixed-precision factorization and high-precision iterative refinement;
3. add a true sparse KKT backend with symbolic reuse;
4. avoid normal equations in difficult many-equality problems;
5. make the LP path genuinely sparse and certificate-capable;
6. finish the incidence-based MOI `ProblemBuilder`, warm-start mapping, and
   supported-subset conformance work;
7. replace repeated task creation with a solve-scoped scheduler;
8. calibrate the automatic selector from actual executed-kernel measurements,
   usable memory, and machine characteristics.

The extended-precision Schur kernels should remain disabled by default, the
adaptive beta/gamma controller should remain opt-in, and BigFloat should remain
single-threaded until the relevant acceptance gates are passed.

## Current implementation status

| Work item | Status | Evidence |
|---|---|---|
| C0.1--C0.10 and C0.12 correctness batch | Implemented | focused LP/solver/KKT/certificate tests |
| C0.11 nonfinite validation | Returned-state checks complete; internal LP direction-state coverage partial | cold certificate plus remaining Milestone 0 work |
| Cold original-coordinate certificate | Implemented | 62/62 focused tests; Task_Low08 passes |
| Arrow memory-order correction | Implemented | 3.85x at dimension 384; 20.26x at 1,024 |
| Exact Float64x4 worker cap | Implemented | 1/2/4/8 kernel tests and measurements |
| Conservative free-memory packing gate | Implemented | selector regressions |
| MOI active-only copy and pure equalities | Implemented | 87/87 focused tests |
| Native/explicit spectrum and atomic export | Implemented | 85/85 focused tests |
| Lower-triangular per-worker Schur storage | Not implemented | Milestone 2 |
| Sparse LP and general sparse KKT | Not implemented | Milestones 3--4 |
| Mixed-precision KKT | Not implemented | Milestone 3 |
| Native SOCP and complete MOI starts | Not implemented | Milestones 4 and 6 |

## Audit method and current baseline

The audit combined:

- a full static read of the solver, kernels, ingestion, pipeline, LP, MOI,
  spectrum, tests, benchmarks, and release files;
- targeted numerical reproductions for presolve, residual, precision, and
  memory-order behavior;
- review of the existing Task_Low08 and CSDR benchmark reports;
- fresh one- and four-thread package test runs;
- a complete Task_Low08 Float64 validation solve;
- Float64x4 and BigFloat old/new dense-kernel measurements;
- a memory-safe sparse fused-arrow 1/2/4/8-thread smoke test.

The synchronized test results at the audit checkpoint were:

| Configuration | Result | Elapsed |
|---|---:|---:|
| Julia 1.12.6, 4 Julia threads | 749 / 749 passed | 1m37.7s |
| Julia 1.12.6, 1 Julia thread | 740 passed, 1 expected broken | 1m39.8s |

These counts are historical because the suite subsequently expanded beyond
1,150 assertions. They remain an audit checkpoint, not a substitute for the
final clean-depot, cluster, and certificate matrix recorded for a release
commit.

Current and historical benchmark evidence:

| Benchmark | Result | Important caveat |
|---|---:|---|
| Task_Low08, native Float64, tolerance `1e-6` | 46.88 s solve, 27 iterations, objective `0.653291393898` | Fresh local result; cold certificate and original-coordinate validation pass |
| Task_Low08, prior implementation | 180.24 s | Same objective and iteration count were reported |
| Task_Low08 local 8-iteration 1/2/4/8 smoke | 28.83 / 24.11 / 17.89 / 24.51 s | Four logical CPUs; 8 threads are oversubscribed |
| Float64x4 64x512 dense Gram, new kernel | 0.204 / 0.091 / 0.049 / 0.037 s | 1/2/4/8 threads; exact lower triangle, zero relative error |
| Sparse fused-arrow smoke, 1/2/4/8 | 4.04 / 2.08 / 1.81 / 0.95 ms | 371 variables and 360 2x2 blocks, not the full requested case |
| CSDR `80/4/40/100`, Float64x4, one thread | 43.99 s per iteration after fusion | About 28.0 s Schur and 14.6 s arrow KKT; current rank-one traversal is still cache-hostile |
| BigFloat 256-bit 32x128 dense Gram | 10.16 to 9.84 ms; 1.84 MB to 560 B | Allocation win is clear; runtime gain is only 1.03x |

Current English summaries are tracked in the repository at
`bench/extended_precision_blas/REPORT.md`,
`bench/csdr_psd_dual/RESULTS.md`, and
`docs/automatic-optimization-pipeline.md`. Historical measurements below keep
their original scope and should not be read as current default-path results.

## What should be preserved

The audit confirmed that the current code already contains valuable
problem-specific engineering:

- exact active-variable incidence lists;
- compact `2 x 2` coefficient panels;
- flat COO storage that avoids empty CSC-column scans;
- active-only sparse contractions;
- a fused exact-arrow `2 x 2` Schur path that avoids a 9.08 GB pair buffer on
  the 4,100-block Float64x4 case;
- output-column sparse Schur scatter without locks or atomics;
- exact block-arrow local-variable elimination;
- Float64 BLAS `syrk!`;
- lower-triangular extended-precision `syrk!`;
- allocation-aware BigFloat dot products;
- best-iterate retention and stagnation diagnostics;
- a dedicated LP predictor-corrector loop;
- an automatic problem-classification and execution-plan layer;
- a simple native Julia API and a non-incremental MOI wrapper.

The update should evolve these components rather than replacing the whole
solver.

## Resolved release blockers

The following C0 sections preserve the original audit evidence and acceptance
criteria. C0.1--C0.10 and C0.12 are implemented and covered by focused
regressions. C0.11 is complete for every returned primal/dual object, while
full intermediate LP direction-state validation remains Milestone 0 work.
Remaining certificate and architecture work is listed after the
implementation milestones.

### C0.1: LP presolve merges inequalities with opposite directions

**Evidence:** `src/lp_solver.jl:132-165`.

The row normalization multiplies by the sign of the first nonzero coefficient.
This makes `g'x >= h` and `-g'x >= h2` look like duplicate directions even
though they impose opposite bounds.

Reproduced example:

```text
minimize x
subject to x >= 0
          -x >= -1
```

With presolve, one row was removed and the run reached `IterLimit` near
`x = -1.324e10`. Without presolve, it returned `Optimal` near
`x = 1.055e-9`.

**Required change:** normalize by a strictly positive magnitude, preserve row
orientation, and merge only positive scalar multiples.

**Acceptance tests:** positive duplicates, negative multiples, interval
constraints, contradictory bounds, near-duplicates, and all three arithmetic
families.

### C0.2: an empty reduced LP cone can be falsely declared infeasible

**Evidence:** `src/lp_solver.jl:480-484`.

If every inequality row is removed as redundant, the dedicated LP path returns
`InfeasibleCert`, including feasible equality-only models.

**Required change:** implement the equality-only case, dispatch it to a
certificate-capable backend, or return an explicit unsupported/indeterminate
status. Never return infeasible without a valid certificate.

### C0.3: SDP can report `Optimal` outside the requested tolerance

**Evidence:** `src/solve.jl:502-513`.

When the step collapses, residuals and gap below `1000 * tolerance` are mapped
to `Optimal`. This contradicts the public tolerance contract and maps directly
to `MOI.OPTIMAL`.

**Required change:** reserve `Optimal` for independently verified requested
tolerances. Add an `AlmostOptimal` or `Stalled` outcome for a useful but
uncertified point.

### C0.4: SDP residuals can describe the previous iterate

**Evidence:** termination at `src/solve.jl:344-386`, residual computation at
`src/solve.jl:424-440`, update at `src/solve.jl:645-689`.

The residuals used at the top of the loop were computed before the accepted
update. A one-iteration reproduction returned:

```text
reported p_res = 1.0       independent p_res = 0.0
reported d_res = 2.0       independent d_res = 1.4745206971
```

This also affects best-iterate selection and result fields.

**Required change:** compute current residuals before every termination
decision and perform a final independent certificate calculation.

### C0.5: scaled residuals are returned with unscaled iterates

**Evidence:** `src/solve.jl:723-740`.

Unequilibration recomputes objectives but not residuals.

**Required change:** recompute primal residual, equality residual, dual
residual, gap, complementarity, and PSD feasibility in original coordinates
after every reconstruction.

### C0.6: equality presolve discards extended-precision information

**Evidence:** `src/pipeline.jl:190-223`.

All rank and consistency calculations convert `B` and `b` to Float64. A
256-bit BigFloat matrix with a genuine `1e-100` independent direction was
reduced from rank two to rank one.

**Required change:**

- use type-native rank-revealing QR when reliable;
- use SVD or a wider validation pass for ambiguous pivots;
- scale columns before rank decisions;
- validate every proposed elimination in the original arithmetic;
- conservatively retain constraints when rank is uncertain.

Float64 may remain a guarded fast path for Float64 data only.

### C0.7: LP postsolve does not test every original inequality

**Evidence:** `src/lp_solver.jl:731-746`.

The reported primal residual compares internal kept slacks with reconstructed
kept slacks, but it does not include `max(0, -slack_original)` over all original
rows. A bad presolve decision can therefore be hidden.

**Required change:** all final validation must use every original constraint.

### C0.8: adaptive-refinement rollback can retain the worse direction

**Evidence:** `src/kkt.jl:562-580`.

The code snapshots the already-corrected direction at the start of the next
pass. If the previous correction increased the residual, rollback restores
that bad direction.

**Required change:** keep an explicit last-accepted direction and accept a
candidate only after evaluating its post-correction backward error.

### C0.9: presolve does not map warm-start equality multipliers

**Evidence:** reduction at `src/solve.jl:891`; dispatch at
`src/solve.jl:925-943`.

The original `y0` is forwarded to a reduced equality system.

**Required change:** pass `y0[equality_keep]` and reconstruct the full result as
usual.

### C0.10: the pre-ingested BigFloat API does not establish working precision

**Evidence:** `src/solve.jl:991-1032` versus the raw-data precision scope at
`src/solve.jl:1073-1101`.

`solve(problem; precision=512)` stores a number in options but can still run at
the ambient BigFloat precision.

**Required change:** wrap the complete solve of a BigFloat `SDPProblem` in the
requested precision scope and define separately whether existing input data is
rerounded.

### C0.11: nonfinite validation is incomplete

**Evidence:** `src/solve.jl:677-682` and `src/lp_solver.jl:719-721`.

The SDP check omits `y`, `Y`, and `mu`; the LP check omits `y`.

**Required change:** validate all primal, dual, complementarity, objective,
residual, and direction state.

### C0.12: equality reconstruction loses termination metadata

**Evidence:** `src/pipeline.jl:270-295`.

The compatibility constructor resets structured termination information.

**Required change:** preserve `result.termination` explicitly.

## High-value performance findings

### P1.1: repair column-major traversal in arrow KKT updates — implemented

**Evidence:** `src/kkt.jl:169-175` and `src/kkt.jl:217-223`.

The inner index varies the second matrix dimension, which is not stride one in
Julia. The same issue appears in the serial and threaded branches; nearby
partial reductions should be audited at the same time.

**Implemented:**

- make the first matrix index the innermost loop;
- hoist the invariant from `Wl` for each destination column;
- preserve deterministic accumulation order;
- specialize the common `q == 1` case;
- benchmark direct loops against a small rank-k kernel for larger `q`.

The gate passed: the focused measurements reached 3.85× at global dimension
384 and 20.26× at 1,024 with unchanged KKT solutions.

### P1.2: eliminate full per-worker Schur matrices

**Evidence:** allocation at `src/workspace.jl:388-396`; full reduction at
`src/kernels/threaded.jl:304-329`.

The extended kernel computes one triangle, but the workspace still stores full
`m x m` matrices for every bin and reduces the full matrices. This explains
the Task_Low08 memory growth with thread count and limits Float64x4 scaling.

**Implementation target:** lower-triangular tiled ownership with one writer per
tile. Prefer direct tile accumulation over task-local full matrices. If a
transition implementation is needed, use packed lower-triangle partials and a
triangular reduction.

**Expected result:** approximately half the immediate storage, removal of
`number_of_bins * m^2` reduction traffic, and useful four- to eight-thread
Float64x4 scaling.

### P1.3: add an extended-precision KKT strategy

**Evidence:** `src/kkt.jl:39-50`, `src/kernels/generic.jl:97-117`, and
`src/kernels/bigfloat.jl:163-191`.

Float64x4 reaches a generic dense Cholesky; BigFloat reaches a serial scalar
Cholesky. On Task_Low08, this dominates before the improved Schur kernel can
deliver a complete high-precision solve.

**Recommended sequence:**

1. equilibrate and estimate condition;
2. form a Float64 factor/preconditioner;
3. compute KKT residuals in Float64x4 or BigFloat;
4. perform target-precision iterative refinement;
5. monitor componentwise backward error and progress;
6. fall back to native precision if the residual does not contract;
7. add a blocked target-precision Cholesky/triangular-solve backend for the
   fallback.

No mixed-precision result may be accepted solely because the Float64 factor
succeeded.

### P1.4: avoid normal equations for difficult equality systems

**Evidence:** `src/kkt.jl:61-75`.

The current route forms `Q = Btil' * Btil`, which squares the condition number.
This is a poor match for the target class with many nearly dependent equality
constraints.

**Candidates:**

- QR-based range-space solve of `Btil`;
- direct symmetric-indefinite LDLT of the KKT system;
- sparse quasidefinite LDLT for sparse patterns;
- normal equations only when a condition estimate predicts they are safe.

The backend should expose analysis, numeric factorization, solve, refinement,
inertia/pivot information, and memory estimates.

### P1.5: add a general sparse KKT backend with symbolic reuse

The structure classifier can label a Schur complement sparse, but the
non-arrow fallback is still dense.

**Architecture:**

```text
AbstractKKTBackend
  analyze_pattern!
  factorize_numeric!
  solve!
  refine!
  condition_estimate
  memory_estimate
  diagnostics
```

Start with:

- CHOLMOD for supported Float64 SPD cases;
- a generic sparse LDLT candidate for fixed extended precision;
- optional Pardiso/MKL or another threaded sparse backend behind an extension;
- exact-arrow as a specialized backend under the same interface.

Select by predicted fill and total factorization cost, not Schur density alone.

### P1.6: reduce repeated task and synchronization overhead

The solver creates many `@sync`/`@spawn` regions per iteration for residuals,
block factors, Schur assembly, predictor, corrector, direction recovery,
line search, arrow factorization, and reductions.

**Implementation:**

- create a solve-scoped worker team;
- reuse queues and worker-local scratch;
- fuse adjacent passes over the same block schedule when data dependencies
  allow;
- use coarse tiles and work stealing for heterogeneous blocks;
- keep deterministic ownership for numerical reductions;
- avoid a task per small block.

### P1.7: respect the requested thread count — implemented

The extended kernels now schedule exactly
`min(requested_threads, Threads.nthreads())` workers and report the requested
and executed widths in diagnostics. The 1/2/4/8 regression matrix covers this
behavior.

### P1.8: replace block-count thresholds with work estimates — partially implemented

**Evidence:** `src/kernels/threaded.jl:73-78`.

Several phases require at least 256 blocks. Task_Low08 has only 32 blocks, but
some have dimension 74 and carry substantial work.

The main Schur and extended-kernel selectors now use estimated work and
packing cost rather than block count alone. A calibrated per-machine launch
and memory-traffic model, persistent workers, and first-iteration telemetry
remain future work.

### P1.9: make memory planning allocation-aware and cluster-aware

**Audit-time evidence:** `src/workspace.jl:85-87` and
`src/workspace.jl:332-338` originally budgeted from host physical memory.

The current planner uses the minimum reliable signal from host available
memory, Linux cgroup v1/v2 limits, and `SDPX_MEMORY_LIMIT_BYTES`, and leaves
general headroom before optional packing. The remaining target is full
component-aware accounting:

```text
usable = min(user_limit, cgroup_or_job_limit, currently_available)
budget = usable - input_bytes - baseline_workspace - safety_reserve
```

The solver now falls back before optional packing. Input bytes, every
simultaneously live factor/workspace, NUMA placement, and scheduler-private
buffers still need a unified lifetime model.

### P1.10: remove setup work that is known to be unused — implemented

At the audit checkpoint:

- pattern groups were built even when the crossover decision was disabled:
  `src/workspace.jl:155-166`.
- exact fused-arrow blocks allocated panel storage that the fused kernel
  does not use: `src/workspace.jl:378-385`.
- KKT equality solves allocated temporary right-hand sides:
  `src/kkt.jl:255-300`.
- best-iterate snapshots allocated new vectors and block matrices on improving
  iterations: `src/solve.jl:363-370`.
- `ExtendedPrecisionBLAS.gemm!` and `pack_columns!` had no solver/test call
  sites and therefore exposed an unsupported surface.

Pattern groups are now lazy, the fused exact-arrow path creates neither panel
nor pair storage, KKT right-hand sides and best-iterate destinations are
workspace-owned, and the extended BLAS surface has direct regression coverage.

### P1.11: complete allocation-free BigFloat workspace kernels — implemented

Internal no-alias methods mutate independently owned destinations with
MutableArithmetics across matrix multiply, triangular operations, factorization,
Schur/KKT storage, and LP Hessian assembly. Alias-safe fallbacks and focused
ownership regressions protect the public paths.

## LP roadmap

The dedicated LP path is a good architectural decision, but it is currently a
dense bounded-LP method rather than a general high-performance LP solver.

### LP1: correctness implemented; complete certificates remain

Row orientation, original-row validation, equality-only handling, honest
non-optimal outcomes, and stagnation/backward-error diagnostics are now
covered by regressions. The remaining work is rigorous unbounded and
infeasible ray export, preferably through a homogeneous self-dual embedding or
an equivalently certificate-capable design, plus more condition-aware
regularization.

### LP2: sparse data path

**Evidence:** `src/lp_solver.jl:73-89` densifies every LP.

Add:

- row-compressed and CSC incidence storage;
- bound/singleton elimination;
- duplicate and dominance detection without dense row scans;
- sparse Hessian or augmented-KKT assembly;
- symbolic factorization reuse;
- a density/aspect-ratio crossover between dense BLAS and sparse LDLT.

### LP3: improve dense factorization

The diagnostics call the backend symmetric-indefinite, but
`src/lp_solver.jl:344-367` constructs a sign-skew full system and uses LU.

Reformulate it as a symmetric system and benchmark Bunch-Kaufman/LDLT against
LU. Add residual-controlled refinement.

### LP4: integrate extended-precision kernels — implemented

The LP Hessian path now uses the owned serial BigFloat kernel and can select
the threaded blocked lower-triangular kernel for fixed-width extended
arithmetic. Diagnostics record the kernel that actually executed; the
automatic selector retains the outer-product path when packing does not
amortize.

### LP5: optimize Float64 packing

The threaded Float64 path recomputes `sqrt(weight)` for every matrix element
and performs the packing serially. Cache row scales, parallelize sufficiently
large packing, and compare with vendor BLAS thread scheduling.

## SDP algorithm roadmap

### SDP1: standardize predictor quality — implemented for adaptive mode

Adaptive mode now uses a pure affine predictor, independent primal and dual
boundary steps, global complementarity over total cone dimension, `mu_aff`,
and a safeguarded squared/cubic Mehrotra rule. Fixed mode intentionally
retains the validated historical trajectory. HKM/NT comparisons and optional
multiple centrality correctors remain research items.

### SDP2: guarded adaptive Newton policy — implemented, still opt-in

The separate typed policy now selects bounded `sigma`, independent
primal/dual fraction-to-boundary values, backtracking, and refinement limits.
It restores the complete fixed predictor/corrector path after non-finite
diagnostics, degraded equality factors, or unstable progress. Its
accepted-iteration history records:

```text
sigma
primal_fraction_to_boundary
dual_fraction_to_boundary
backtracking_factor
affine_primal_step
affine_dual_step
mu
mu_aff
residuals_before
residuals_after
regularization
refinement_count
factorization_quality
primal_psd_margin
dual_psd_margin
fallback_reason
```

Adaptive control remains off by default. It was about 2% slower on the
representative LP and improved the warmed CSDR s15 solve by 1.19x, but was
1.7% slower on Task_Low08 after a safe degraded-factor fallback. Promotion
still requires a stable runtime or robustness improvement without certificate
regression across the acceptance matrix. See
[`adaptive-parameter-policy.md`](adaptive-parameter-policy.md).

### SDP3: improve regularization and refinement

- scale regularization by a matrix norm, not `max(abs(diagonal), 1)`;
- estimate condition and perturbation size;
- continue expanding normalized backward-error coverage across backends;
- accumulate refinement residuals in wider arithmetic where useful;
- retain the implemented rollback that prevents a refined direction from
  replacing a better unrefined direction.

### SDP4: stronger structural presolve

After the type-aware equality fix:

- eliminate fixed and absent variables;
- detect empty/inconsistent PSD blocks;
- merge exact duplicate affine constraints safely;
- perform block and variable scaling without scanning empty `L x m` grids;
- add chordal decomposition for genuinely large sparse PSD blocks;
- generalize arrow elimination to models with equality constraints;
- retain reconstruction maps and validate in original coordinates.

### SDP5: native SOCP

MOI SOC constraints are currently expanded into dense arrow-shaped PSD blocks.
Add native Lorentz-cone Jordan algebra, scaling, residual, step, and dual-map
kernels. Preserve the PSD-lift path as a reference/fallback.

## Sparse and dense execution strategy

### Sparse CSDR family

For `J/K/N_a/N_mu = 40/4/20/100` and `80/4/40/100`:

1. retain active-only storage and fused `2 x 2` arrow assembly;
2. fix the arrow rank-one traversal;
3. remove unused fused-panel storage;
4. use weighted column boundaries in sparse scatter;
5. extend arrow elimination to equality-aware problems;
6. choose sparse outer products versus packed `syrk!` from measured executed
   costs, not global density alone;
7. keep BigFloat serial initially;
8. use process-level parallelism for independent BigFloat solves or parameter
   sweeps.

### Dense-Schur lattice family

For Task_Low08:

1. revalidate the complete Float64 benchmark after certificate fixes;
2. use lower-triangle tile ownership to remove full per-worker Schur copies;
3. use work-based block parallelism for the 32 heterogeneous blocks;
4. tune the dense Float64 factorization independently from Schur assembly;
5. add mixed-precision KKT before attempting a full Float64x4 result;
6. do not attempt full BigFloat Task_Low08 until a memory-safe KKT strategy is
   available.

## Automatic optimization pipeline

The execution plan should be finalized after presolve, not before it. The
pipeline should have six explicit stages:

```text
input validation
  -> structural fingerprint and classification
  -> certificate-safe presolve
  -> scaling/equilibration
  -> backend, kernel, memory, and schedule selection
  -> parameter initialization and solve
  -> original-coordinate validation and diagnostics
```

### Required classifier inputs

- cone composition: LP, SOC, PSD;
- variable, equality, block, and cone dimensions;
- coefficient nonzeros and active variables by block;
- per-block union-pattern density;
- predicted Schur nonzeros and fill;
- equality rank and condition estimate;
- arithmetic family and effective precision;
- dynamic range and scaling spread;
- actual Julia and BLAS thread availability;
- cache sizes and NUMA topology when available;
- usable memory and user memory limit;
- whether the structure matches a cached plan;
- whether a warm start or prior factorization is reusable.

### Selector output

- presolve actions and reconstruction maps;
- scaling method and iteration count;
- LP/SOCP/SDP algorithm;
- dense, sparse, arrow, or chordal KKT backend;
- Schur assembly representation;
- numeric precision strategy;
- exact thread allocation by phase;
- memory budget by component;
- beta/gamma initialization and adaptive policy;
- fallback chain;
- machine-readable explanation.

### Current heuristic thresholds to replace or calibrate

| Current rule | Location | Problem |
|---|---|---|
| Sparse storage if coefficient density `<= 0.20` or active density `<= 0.55` | `src/ingest.jl` | Ignores setup cost, arithmetic type, repeated solves, and per-block variation |
| Dense Schur if density `>= 0.15`; other non-arrow path is still a dense fallback | `src/ingest.jl` | The selector describes a distinction the backend does not implement |
| Force one thread for small Float64 SDP with blocks `<= 2` and `m < 1000` | `src/pipeline.jl` | Coarse global rule; should use phase work |
| Require at least 256 blocks for several threaded phases | `src/kernels/threaded.jl` | Makes 32 large lattice blocks serial |
| LP threaded assembly if work `>= 2e6` and BLAS width is one | `src/pipeline.jl`, `src/lp_solver.jl` | Machine- and shape-dependent |
| Extended minimum columns 32/20, work `2e5`/`5e4`, speedup 1.18/1.12 | extended selector | Hand-calibrated constants without executed-kernel telemetry |
| Extended sparse density floor 0.42/0.62 and Schur density floor 0.20/0.05 | extended selector | Uses global Schur density for every block |
| Extended packing budget 10% of currently usable memory | options/workspace | Conservative global cap; still lacks complete per-component lifetime accounting |
| Schur partials capped against currently usable memory | workspace | Still needs input/workspace/factorization lifetime accounting and NUMA placement |
| KKT BLAS threads approximately `m / 256` | `src/step.jl` | Useful measured seed, but hardware-specific |

### Calibration design

Do not run an expensive search before every solve. Use:

1. a small installation- or node-level calibration cache;
2. analytical cost models seeded by that calibration;
3. a structural fingerprint cache for repeated solves;
4. optional first-iteration telemetry;
5. hysteresis, so a marginal prediction does not switch algorithms;
6. user overrides for reproducibility.

The model should predict setup time, iteration time, memory, and accuracy risk.
Default selection should require a predicted gain above both uncertainty and
switching cost.

## Multicore and cluster plan

### Single-node multicore

- assign exact worker counts rather than using the full Julia pool;
- use a persistent solve-scoped scheduler;
- separate compute-bound and bandwidth-bound phases;
- use lower-triangle tile ownership;
- first-touch large buffers on the NUMA node that will use them;
- optionally pin workers and report affinity;
- avoid global BLAS thread mutation during concurrent solves;
- choose Julia-task width and BLAS width jointly;
- retain deterministic reduction order for reproducibility.

The current `_with_blas_threads` changes a process-global BLAS setting.
Concurrent solves can race. Either serialize those transitions, keep a fixed
BLAS width, or introduce a documented process-level solver scheduler.

### Multi-node cluster

The current solver is single-process and node-local. More Julia threads do not
make one solve multi-node.

Add two levels deliberately:

1. **Immediate:** process-level orchestration for independent solves,
   parameter sweeps, precision escalation trials, and benchmark repetitions.
   This is also the safest parallel route for BigFloat.
2. **Later:** optional distributed sparse/dense factorization through a
   proven backend such as MUMPS, Elemental, or another supported distributed
   solver. This is a separate project and should not be implied by the
   `threads` option.

Provide PBS/Slurm launch helpers that record CPU model, allocation limits,
affinity, Julia threads, BLAS threads, commit, and input fingerprint.

## BigFloat policy

The release policy is now consistent: one SDPX solve uses one BigFloat worker.
Owned MPFR destinations and allocation-free scalar kernels are validated for
the serial path; requesting additional solver threads does not enable
within-solve BigFloat parallelism. The stress campaign below remains the gate
for changing that policy, not evidence that MPFR itself can never be used
concurrently.

Required stress campaign:

- disjoint block-local MPFR operations only;
- private workspace objects with no shared mutable `BigFloat`;
- 1/2/4/8 threads;
- multiple precisions and rounding modes;
- repeated randomized runs;
- bitwise comparison where deterministic order is promised;
- original-coordinate solver certificates;
- allocation and peak-RSS tracking;
- ThreadSanitizer or lower-level diagnostics where practical.

Until that campaign passes, keep BigFloat serial. For cluster workloads, run
independent BigFloat solves in separate processes.

## MOI and public API plan

### Reduce model-building cost — active-only copy implemented

The MOI wrapper now streams only active affine terms into sparse-native
storage, avoiding the audit-time all-variable/all-constraint empty-object
construction. SOC constraints still use a dense PSD arrow lift. A reusable
public `ProblemBuilder` based on triplets/incidence remains useful:

```text
add_variable!
add_equality!
add_linear_inequality!
add_soc_block!
add_psd_block!
finalize!
```

The current `copy_to` behavior provides the active-only foundation; finalizing
this builder would make the same efficient construction API available outside
MOI.

### Complete MOI semantics

- expose correct result count after infeasible, failed, and limited solves;
- distinguish feasible points from infeasibility certificates;
- do not allow objective queries for results that do not contain a point;
- support variable and dual starts;
- map starts through presolve and scaling;
- add conservative unbounded/infeasible statuses until HSD certificates exist;
- run the supported subset of `MOI.Test`;
- make every raw option alias readable and writable;
- add a standard thread-count attribute if supported by the target MOI
  version.

### Simplified solver interface

Keep the normal user-facing options:

- tolerance;
- maximum iterations;
- time limit;
- thread count;
- precision;
- verbosity;
- diagnostics;
- warm start.

Keep beta, gamma, omega values, regularization, refinement, kernel overrides,
and backend selection in an expert options object. Return:

- status and certificate kind;
- primal and dual objectives;
- original-coordinate residuals;
- scaled residuals when scaling was used;
- iteration/restart/regularization counts;
- setup and per-phase timings;
- estimated and measured memory;
- actual executed algorithms and kernels;
- thread allocation;
- parameter history;
- warnings and fallback reasons.

## Spectrum and export plan — core implementation complete

Spectrum extraction is optional and remains outside timed solve statistics.
It now supports explicit native/Float64 reconstruction, records projection and
arithmetic metadata, checks solve status, and exports CSV, JSON, or JLD2 using
atomic replacement with informative optional-dependency errors. Remaining
work is a specialized generic/high-precision eigensolver extension where the
standard backend cannot operate natively, plus streaming strategies for very
large spectra.

## Benchmark and validation program

### Mandatory correctness tier

Before every full benchmark:

1. dense versus sparse canonical Schur comparison;
2. extended `syrk!` against a high-precision reference;
3. arrow versus dense KKT solve;
4. BigFloat aliasing tests;
5. LP presolve orientation and empty-cone tests;
6. equality rank tests across scales and precisions;
7. refinement monotonicity;
8. requested-thread enforcement;
9. original-coordinate result certificate.

### Full benchmark matrix

| Family | Problem | Arithmetic | Threads |
|---|---|---|---|
| LP dense | 80 x 400 and larger aspect-ratio sweep | Float64, Float64x4, BigFloat | 1/2/4/8 for fixed-width; BigFloat 1 |
| LP sparse | density/aspect-ratio sweep to at least 10,000 rows | Float64, Float64x4 | 1/2/4/8 |
| Sparse SDP | CSDR `40/4/20/100` | Float64x4; BigFloat | 1/2/4/8; BigFloat 1 |
| Larger sparse SDP | CSDR `80/4/40/100` | Float64, Float64x4; bounded BigFloat pilot | 1/2/4/8 |
| Dense-Schur SDP | Task_Low08 | Float64 first, then Float64x4 | 1/2/4/8 |
| Many equalities | controlled rank/condition family | all types | relevant widths |
| SOC | dimensions 2, 3, 10 and mixed cones | Float64, Float64x4 | 1/2/4/8 |

Task_Low08 extended precision may run only after the corrected Float64 baseline
passes and the mixed-precision KKT gate succeeds.

### Metrics

Record:

- model build and ingest time;
- presolve and scaling time;
- workspace construction;
- each iteration phase;
- total solve time and iteration count;
- allocations and GC;
- current and peak RSS in a fresh process;
- predicted and actual kernel/backend;
- requested and actual Julia/BLAS widths;
- objective and duality gap;
- original-coordinate primal, equality, and dual residuals;
- complementarity;
- minimum primal and dual PSD eigenvalues;
- Schur relative error;
- KKT backward error and refinement history;
- warnings, fallbacks, and status.

### Measurement protocol

- use a fresh process per configuration;
- separate compilation from measurement;
- use at least five warmed repetitions for short runs and three for long runs;
- report median, minimum, and spread, not only the minimum;
- isolate one thread count per cluster job;
- record CPU, memory limit, NUMA layout, BLAS vendor, Julia version, commit,
  input fingerprint, and options;
- record executed-kernel counters rather than selector predictions;
- store machine-readable JSON/CSV plus an English summary.

### Promotion gates

An optimization may become default only if:

1. every independent original-coordinate certificate passes;
2. no `Optimal` result exceeds the requested normalized tolerance;
3. median runtime improves by at least 10%, or robustness improves on a
   previously failing class;
4. no representative benchmark regresses by more than 5% without an explicit
   justified tradeoff;
5. allocations and peak memory do not grow unexpectedly;
6. results are stable across repetitions and thread counts;
7. fallback behavior is tested;
8. diagnostics match the code path actually executed.

Suggested numerical checks:

```text
Float64 Schur relative error <= 5e-13
Float64x4 Schur relative error <= 1e-48
KKT backward error <= precision-aware bound based on eps(T) and condition estimate
minimum PSD eigenvalue >= -scaled feasibility tolerance
objective agreement <= requested tolerance plus reference uncertainty
```

## Documentation and release work

### Resolve contradictions

The audit-time contradictions have been reconciled:

- BigFloat is consistently documented as serial within one solve;
- parameter defaults and the fixed/adaptive distinction match
  `SolverOptions`;
- SOC is described as classification plus an exact PSD arrow lift, not a
  native cone backend;
- sparse equilibration and phase timing are documented as implemented;
- execution diagnostics identify the actual Gram/Schur kernel and fallback
  reason;
- the release checklist reflects the standalone repository, new UUID, and
  retained upstream provenance.

Generating the option reference from `SolverOptions` metadata, or testing all
documented defaults mechanically, remains worthwhile maintenance work.

### Release gates

Before a public `v0.1.0` tag:

- fix all C0 blockers;
- run the complete benchmark/certificate matrix at the release commit;
- run `MOI.Test` for the declared supported subset;
- add Aqua quality checks and clean-depot tests;
- add Documenter, coverage, CompatHelper, SECURITY.md, and CODE_OF_CONDUCT.md,
  or explicitly document why an item is deferred;
- ensure TagBot and documentation keys match actual workflows;
- verify dependency licenses and General registry name availability;
- synchronize and verify local/remote history;
- mark the changelog release date and tag;
- publish reproducible benchmark artifacts and environment metadata.

## Implementation sequence

### Completion snapshot

| Milestone | Current status |
|---|---|
| 0: result correctness and honest status | Substantially complete; rigorous exported equality and unbounded rays remain |
| 1: low-risk sparse/arrow wins | Substantially complete; arrow traversal, lazy groups, fused no-buffer path, KKT RHS buffers, and telemetry are implemented |
| 2: Schur memory and multicore redesign | Partial; exact workers, work scheduling, and usable-memory caps are complete; tile ownership and persistent teams remain |
| 3: KKT abstraction and mixed precision | Not started |
| 4: sparse LP and native SOCP | Dedicated dense LP and high-precision Hessian kernels are implemented; sparse HSD LP and native SOCP remain |
| 5: algorithmic tuning | Structural profiles and guarded adaptive beta/gamma are implemented; adaptation remains correctly opt-in |
| 6: interface, diagnostics, and release | Active-only MOI copy, result certification, and spectrum export improved; conformance and release work remain |

### Milestone 0: result correctness and honest status

**Estimated effort:** 3--5 engineering days.

- C0.1 through C0.12;
- independent `validate_result`;
- new status/certificate semantics;
- regression tests for every reproduced defect;
- MOI status/result-count corrections.

**Exit gate:** no known false feasible/infeasible/optimal result; all tests and
targeted certificate checks pass.

### Milestone 1: low-risk sparse/arrow wins

**Estimated effort:** 2--4 engineering days.

- column-major arrow update and reduction;
- workspace-owned KKT RHS buffers;
- lazy pattern grouping;
- remove unused fused-arrow panels;
- preallocated best-iterate storage;
- actual-kernel counters.

**Exit gate:** at least 2x arrow-update phase gain at dimension 384 and at least
10% end-to-end gain on one large sparse acceptance case without accuracy or
memory regression.

### Milestone 2: Schur memory and multicore redesign

**Estimated effort:** 1--2 weeks.

- lower-triangle tile ownership;
- exact requested worker counts;
- work-based phase thresholds;
- persistent scheduler;
- weighted sparse scatter boundaries;
- usable-memory planner;
- NUMA/affinity diagnostics;
- concurrent-solve BLAS policy.

**Exit gate:** Task_Low08 and CSDR 1/2/4/8 scaling with no
`threads * m^2` memory growth.

### Milestone 3: KKT backend abstraction and mixed precision

**Estimated effort:** 2--4 weeks.

- dense/sparse/arrow backend interface;
- QR or direct LDLT many-equality path;
- Float64 sparse symbolic reuse;
- mixed-precision factor/refinement/fallback;
- blocked native extended-precision fallback.

**Exit gate:** complete, validated Task_Low08 Float64x4 solve and controlled
ill-conditioned fallback tests.

### Milestone 4: sparse LP and native SOCP

**Estimated effort:** 2--4 weeks.

- certificate-capable HSD LP;
- sparse LP storage and factorization;
- dense symmetric LDLT path;
- extended-precision LP `syrk!`;
- native SOC kernels and selector integration.

**Exit gate:** supported MOI tests, infeasible/unbounded certificates, and
clear crossover wins over the general SDP path.

### Milestone 5: algorithmic tuning

**Estimated effort:** 1--3 weeks.

- canonical affine predictor quality;
- HKM/NT comparison;
- repaired beta/gamma controller;
- condition-aware regularization;
- optional multiple correctors;
- calibrated scaling and initialization profiles.

**Exit gate:** adaptive control becomes default only if the promotion gates
pass across LP, CSDR, dense SDP, and Task_Low08.

### Milestone 6: interface, diagnostics, and release

**Estimated effort:** 1--2 weeks.

- incidence-based `ProblemBuilder`;
- MOI starts and complete result semantics;
- spectrum precision/export improvements;
- generated option documentation;
- CI quality, documentation, coverage, and release automation.

## Implemented and next pull-request sequence

The current working tree contains the former first batch:

1. all result/status/presolve correctness fixes;
2. independent original-coordinate validation;
3. targeted regression tests;
4. no algorithm-default changes.

It also contains the low-risk portion of the former second batch:

1. column-major arrow updates;
2. KKT right-hand-side allocation cleanup;
3. exact Float64x4 worker scheduling;
4. conservative free-memory crossover;
5. memory-safe sparse smoke revalidation.

The next performance pull request should contain:

1. lower-triangle tile ownership;
2. executed-kernel telemetry;
3. persistent solve-scoped scheduling;
4. the complete Task_Low08 and `40/4/20/100` cluster tables.

The following architectural pull request should then add:

1. a KKT backend abstraction;
2. QR/LDLT many-equality routes;
3. sparse symbolic reuse;
4. mixed-precision factorization, refinement, and fallback.

This sequence separates correctness, low-risk kernel work, and architectural
parallelism, making every performance claim independently reviewable.

## Final assessment

SDPX has a strong specialized sparse foundation and is already much faster
than its earlier implementation. The remaining gains are not primarily from
another pairwise-dot optimization. They are concentrated in:

- correctness-safe presolve and result certification;
- cache-correct arrow reduction;
- triangular Schur storage and scheduling;
- extended-precision KKT factorization;
- sparse/many-equality linear algebra;
- sparse LP and native SOC paths;
- MOI construction and result semantics.

The immediate arrow-loop correction is likely the highest return per line of
code. The mixed-precision KKT backend is the highest-impact architectural
project for making Float64x4 practical on dense lattice problems. The
certificate work is the prerequisite for trusting either speedup.
