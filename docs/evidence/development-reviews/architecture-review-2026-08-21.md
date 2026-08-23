# SDPX.jl architecture review and benchmark-driven refactor roadmap

**Review date:** 2026-08-21
**Reviewed baseline:** `2a301a10434dd013e2acbbebf5c2a11c9bb5ee6f`
**Working branch:** `refactor/benchmark-driven-core-20260821`

## Status and evidence boundary

This document records a repository-wide static review of the current source,
tests, benchmark registry, public API, native cone IR, planner, linear algebra,
KKT implementations, numerical solvers, certification path, and CI layout.
The precision-contract bug on `main` was repaired before this branch was
created.

No performance number in this document is claimed for the reviewed commit.
The repository connector used for this review can read and write GitHub but
cannot execute Julia, and GitHub App content writes did not start Actions for
the new commits. Historical benchmark records are used only to prioritize
measurement, never as a current baseline. A current performance baseline must
be produced by the canonical benchmark runner before any numerical optimization
is accepted.

## Executive assessment

SDPX already contains the foundations of an engineering-grade conic solver:

- a compact typed modeling API;
- an explicit native product-cone IR with arithmetic and precision ownership;
- fail-closed LP/SOCP/SDP route classification;
- typed planning records for formulations, KKT backends, providers, memory,
  precision policy, and diagnostics;
- native LP, Lorentz-cone SOCP, and SDP execution paths;
- original-coordinate reconstruction and certification;
- BigFloat ownership checks and optional MultiFloat/BigFloat linear-algebra
  providers;
- a provenance-aware benchmark registry with semantic gates, source/input
  fingerprints, repeated samples, fresh-process campaigns, and strict result
  comparison.

The main weakness is no longer a missing abstraction. It is incomplete
separation of existing abstractions. Policy, planning, allocation, numerical
execution, fallback, reconstruction, and diagnostics are still concentrated in
several very large files. This increases compile latency, makes execution
authority difficult to audit, and turns locally safe changes into
repository-wide risks.

The recommended strategy is therefore **not** a new solver rewrite. Preserve
the public model/IR contract and numerical methods, freeze evidence first, then
extract coherent components behind the existing typed seams. Optimize only
where phase traces and paired benchmarks identify a material cost.

## Current module and responsibility map

### 1. Public modeling and result layer

The public surface is intentionally small. `Model`, variables, affine
expressions, domains, constraints, starts, `Settings`, `Outputs`, `Result`, and
`optimize!` are exported. The implementation compiles a model once, classifies
one native cone family, invokes one family lowerer, executes one core solver,
then reconstructs and certifies in original model coordinates.

Strengths:

- one public entry point and one typed result;
- no user-visible primal/dual orientation choice;
- output retention is separated from numerical settings;
- warm starts fail closed when a native core cannot represent them;
- an `Optimal` core status is downgraded if the original-coordinate public
  certificate fails.

Debt:

- `src/public/optimize.jl` owns settings normalization, output-reference
  validation, LP/SOCP/SDP warm-start mapping, core invocation, reconstruction,
  certification, result construction, and the entry point in one file;
- similar arithmetic-copy and precision-boundary logic appears in multiple
  frontend and bridge paths;
- expert and compatibility APIs remain internally broad even though the
  exported surface is compact.

### 2. Native product-cone IR

`NativeConeProgram` is the strongest architectural boundary in the repository.
It owns arithmetic/precision, objective, sparse row-by-variable equality data,
ordered product-cone blocks, ordered affine-cone row blocks, and reconstruction
maps. PSD data remains one packed lower-authoritative block. The IR deliberately
contains no dualization, provider, factorization, or formulation decision.

Strengths:

- mathematical representation is separate from execution policy;
- cone blocks are not prematurely scalarized or lifted;
- reconstruction identity is explicit;
- route classification validates structural counts and fails closed for mixed
  unsupported cone families.

Debt:

- each family lowerer has its own substantial mapping and storage logic;
- compile/lower/reconstruct contracts are spread between `ir/`, `public/`, and
  legacy core structures;
- some structural facts are recomputed after the IR already owns enough data to
  cache them once.

### 3. Midend and planning

The midend contains `problem_features`, `auto_planner`,
`formulation_planner`, and frontend option resolution. These files express the
right design: exact structural facts feed pure formulation candidates and an
authoritative execution plan.

Strengths:

- stateless `AutoPlanner`;
- explicit problem classification and formulation candidates;
- capability, memory, storage, equality-rank, and arithmetic facts are visible;
- planner decisions are retained in diagnostics.

Debt:

- the midend is small while `src/pipeline.jl` remains very large and still owns
  option validation, precision policy, equality presolve, structural routing,
  backend/resource decisions, reconstruction policy, and diagnostics setup;
- LP planning is partly deferred until presolve, while SDP/SOCP paths freeze
  different subsets earlier;
- the distinction between `ProblemFeatures`, `ExecutionPlan`, workspace
  configuration, and executed fallback state is not consistently enforced at
  every boundary.

### 4. Execution and solver loops

The native execution paths are mature but unevenly decomposed:

- LP has a dedicated solver and sparse support;
- SOCP owns Lorentz algebra, metric assembly, singleton presolve, and a native
  interior-point path;
- SDP owns Schur assembly, workspace construction, predictor/corrector logic,
  KKT solves, refinement, termination, and certification.

Strengths:

- family-specific kernels avoid unnecessary cone lifts;
- phase diagnostics, termination metadata, regularization, refinement, and
  fallbacks are observable;
- prepared structures and sparse symbolic reuse already exist in selected
  paths.

Debt:

- `lp_solver.jl`, `solver/interior_point.jl`, `soc_native.jl`, `schur.jl`, and
  `kkt.jl` are monolithic execution units;
- initialization, iteration, termination, checkpointing, and diagnostic
  construction share mutable solver state;
- setup-time choices can be hard to distinguish from iteration-time numerical
  recovery;
- several specialized paths use parallel but not fully uniform lifecycle
  interfaces.

### 5. KKT and factorization

`KKTBackend` is a useful typed seam. Dense normal equations, dense augmented
systems, block-arrow elimination, sparse Schur, mixed precision, and LP-specific
backends are named and retained. Provider capability validation is explicit.

Strengths:

- formulation and backend names are visible in plans and diagnostics;
- sparse symbolic reuse and numeric-refactor telemetry are represented;
- mixed-precision rejection and authorized fallback are recorded;
- factorization, solve, refinement, and statistics have common conceptual
  operations.

Debt:

- formulation assembly and factorization-provider ownership are still coupled
  through `Workspace`;
- `kkt_formulations/` contains only one migrated formulation while most KKT
  code remains at the source root;
- `la_backends/` contains only the legacy adapter while provider planning and
  many execution primitives remain in `la_backend.jl`;
- backend selection, workspace buffer presence, and runtime fallback are not
  represented by one immutable `KKTPlan` plus one explicit execution state;
- normal-equation, augmented, arrow, sparse, and LP paths expose different
  symbolic/numeric lifecycle details.

### 6. Linear algebra and arithmetic ownership

The core supports Float64, BigFloat, and optional MultiFloat arithmetic, with
capability models for Cholesky, LU, rank-revealing QR, triangular solves,
multiple right-hand sides, and owned mutable scalar semantics.

Strengths:

- BigFloat precision and independent-entry ownership are treated as correctness
  properties rather than implementation details;
- provider capabilities are validated before execution;
- specialized MFLA/BFLA providers are optional extensions;
- the implementation records the provider that actually executed.

Debt:

- provider planning, validation, ownership checks, and numerical operations are
  concentrated in a single large seam;
- specialized routes retain legacy kernels, so backend behavior is not uniform;
- process-global BLAS/thread state can leak into reproducibility unless every
  benchmark pins it;
- provider absence, incomplete capability, numerical rejection, and authorized
  fallback need one stable failure taxonomy.

## Current data flow

The intended public data flow is:

```text
Model
  -> validate and compile once
NativeConeProgram
  -> classify native family once
NativeConeRoute
  -> collect exact structural features
ProblemFeatures
  -> choose formulation/provider/memory/precision policy
ExecutionPlan
  -> lower exactly one family
LP/SOCP/SDP core problem + reconstruction map
  -> allocate solve-local workspace
SolveContext
  -> initialize -> iterate -> terminate
CoreResult
  -> reconstruct original coordinates
  -> original-coordinate certificate
Result
```

The implementation broadly follows this flow, but some decisions currently
cross the intended boundaries:

- public adapters resolve numerical options again before invoking cores;
- `pipeline.jl` combines feature extraction, validation, planning, presolve,
  and configuration;
- workspace construction can finalize a backend decision;
- mixed-precision execution can move through an authorized runtime fallback;
- reconstruction and certification are implemented separately for public,
  native SOCP, and SDP result shapes.

The refactor must make these crossings explicit rather than remove legitimate
numerical recovery. The plan should authorize a finite fallback chain; the
execution state should record which member ran and why.

## Algorithm assessment

### LP

The dedicated primal-dual interior-point route supports dense and sparse data,
equality handling, diagonal reduction, direct dense factorizations, presolve,
and original-coordinate result checks. It avoids treating an LP as a generic
SDP.

Priority questions for measurement:

- dense KKT versus diagonal-reduced equality Gram crossover;
- equality rank detection and RRQR cost;
- sparse assembly and numeric-refactor reuse;
- presolve/reconstruction cost on small instances;
- allocation and setup overhead relative to iterations.

### SOCP

The native Lorentz route avoids a PSD lift and includes specialized Q3 kernels,
general Lorentz metric assembly, sparse active-column handling, singleton
substitution, and fixed-trace specialization.

Priority questions for measurement:

- metric/Schur assembly share of total time by cone count and active columns;
- many-small-cone versus few-large-cone behavior;
- sparse active-pair traversal and thread scheduling;
- reconstruction/certification overhead;
- extended-precision provider effectiveness.

### SDP

The SDP path implements primal-dual predictor/corrector iterations, dense and
sparse Schur systems, block-arrow elimination, augmented KKT support, mixed
precision, iterative refinement, PSD spectral work, equality reduction, and
certificate-aware termination.

Priority questions for measurement:

- Schur assembly versus factorization versus refinement by block structure;
- normal equations versus augmented KKT on ill-conditioned equalities;
- block-arrow eligibility and reduced-system cost;
- sparse symbolic reuse and fill growth;
- eigendecomposition, cone step, and certificate cost in high precision;
- working-precision ladders and fallback frequency.

## Numerical-stability assessment

Existing safeguards are substantial:

- original-coordinate primal, dual, gap, cone/PSD, and certificate checks;
- rank-revealing equality handling;
- regularization and iterative refinement;
- mixed-precision acceptance guards;
- precision ladders and BigFloat precision ownership;
- termination reasons and stages;
- fail-closed public status downgrade.

The main risks requiring explicit benchmark coverage are:

1. **Normal equations square conditioning.** Dense and sparse Schur paths can
   amplify equality or scaling problems. Augmented formulations need a robust
   provider and inertia-aware validation before broader use.
2. **Arithmetic ownership.** BigFloat arrays must not share mutable MPFR zeros,
   silently change precision, or stage sensitive input through Float64.
3. **Mixed precision.** Condition estimates, refinement budgets, cooldowns, and
   fallback reasons must be deterministic and visible.
4. **Sparse factorization.** Pattern reuse, pivot failure, fill growth,
   regularization, and numeric refactor failure must be separately counted.
5. **Accumulation error.** Schur/metric kernels need deterministic accumulation
   order and accuracy checks for Float64x4 and BigFloat256.
6. **Presolve reconstruction.** Every reduction must certify the restored
   original problem; a reduced-space status alone is insufficient.
7. **Precision boundaries.** Every public and expert entry point must enforce
   the same lower bound and scope. The repaired one-bit BigFloat bug is a
   concrete example of why this policy must have one owner.

## Performance and memory assessment

The current source suggests the following measurement order. These are
hypotheses until measured on the reviewed commit.

1. Cone metric and Schur assembly, especially sparse SOCP and many small blocks.
2. Dense/sparse KKT factorization and repeated numeric refactorization.
3. Equality Gram construction, rank detection, and reduction.
4. BigFloat and MultiFloat provider calls, conversions, ownership checks, and
   temporary allocation.
5. Spectral decomposition and cone-step computation in SDP.
6. Thread-local Schur buffers and memory-budget scheduling.
7. Setup, compilation, reconstruction, and certification for small problems.
8. Package load and method compilation caused by the single-module include
   graph.

Memory must be reported at three levels where available:

- Julia allocated bytes for the measured solve;
- solver-owned workspace estimate;
- fresh-process peak RSS.

A lower allocation count is not sufficient if workspace or peak RSS grows.

## Engineering-quality assessment

### Good foundations

- typed contracts and fail-closed validation;
- extensive regression and pathological tests;
- explicit diagnostics and performance trace;
- source/input/environment fingerprints in benchmarks;
- optional providers isolated through extensions;
- public data reconstructed and certified in original coordinates;
- benchmark results serialized without downcasting high-precision metrics.

### Main maintainability risks

- multiple 100–150 KB files with mixed responsibilities;
- partial directory migrations that leave old and new architecture side by side;
- a large unconditional include graph in one module;
- repeated option and precision validation across public, expert, CLI, and
  bridge layers;
- legacy and modern execution seams coexisting without one retirement table;
- benchmark one-off drivers and historical work products alongside the
  canonical registry;
- some CI coverage depends on optional providers or manually prepared sources;
- performance decisions can be encoded as thresholds without a machine-readable
  provenance record explaining the benchmark that justified them.

## Target architecture

The target is an extraction around current behavior, not a second solver.

```text
src/
  frontend/
    compile.jl
    starts.jl
    result_adapters.jl
    certification.jl
  ir/
    types.jl
    storage.jl
    route.jl
    lower_lp.jl
    lower_soc.jl
    lower_sdp.jl
    reconstruction.jl
  planning/
    features.jl
    execution_plan.jl
    formulation_plan.jl
    kkt_plan.jl
    la_plan.jl
    precision_plan.jl
    memory_plan.jl
  execution/
    context.jl
    lifecycle.jl
    initialize.jl
    termination.jl
    checkpoint.jl
  cones/
    lp/
    soc/
    sdp/
  linear_systems/
    interface.jl
    formulations/
      dense_normal.jl
      dense_augmented.jl
      sparse_schur.jl
      block_arrow.jl
      lp_reduced.jl
    factorizations/
      standard.jl
      multifloat.jl
      bigfloat.jl
      sparse.jl
      mixed_precision.jl
  diagnostics/
    trace.jl
    counters.jl
    failure_reasons.jl
```

Key contracts:

- `ProblemFeatures` contains facts only.
- `ExecutionPlan` is immutable and contains all authorized routes and fallback
  chains before numeric execution.
- `KKTPlan` separates mathematical formulation from factorization provider.
- `SolveContext` owns all mutable solve-local state and buffers.
- factor objects expose `analyze!`, `factorize!`, `solve!`, `refine!`, and
  `statistics` consistently;
- symbolic and numeric factorization states are separate;
- runtime may execute only a fallback explicitly authorized by the plan;
- every reduction owns a reconstruction map and original-coordinate certificate
  obligation;
- diagnostics observe execution but never select it.

## Benchmark contract

The canonical runner remains the single source of performance evidence. The
first fixed campaign is a nine-cell matrix:

| Family | Problem | Float64 | Float64x4 | BigFloat256 |
|---|---|---:|---:|---:|
| LP | `synthetic/lp_box` | auto | MFLA | BFLA |
| SOCP | `synthetic/soc_q3` | auto | MFLA | BFLA |
| SDP | `synthetic/sdp_dense` | auto | MFLA | BFLA |

Each row must retain at least:

- total, setup, frontend, preprocessing, presolve, core, factorization,
  refinement, and certification time where exposed;
- Julia allocated bytes, solver workspace bytes, and fresh-process peak RSS;
- primal/dual objective, relative and absolute gap, primal/dual residuals,
  affine residuals, cone/PSD violations, complementarity, and certificate;
- iterations, restarts, regularizations, refinement solves, symbolic/numeric
  factorizations, attempts, successes, failures, route, formulation, provider,
  and fallback;
- repeated-sample status/objective/path parity, median, MAD, range, and
  fresh-process stability.

The nine-cell matrix is a smoke/per-change gate, not sufficient evidence for a
production claim. Accepted optimizations must also run the relevant structural,
pathological, public, and scale ladder cases.

### Reproducibility requirements

- identical solver source, benchmark driver, project, manifest, input, and
  environment fingerprints;
- one Julia thread and one BLAS thread for the primary deterministic baseline;
- separate scaling campaigns for multithreading;
- compilation excluded from hot solve time and reported separately;
- at least three fresh processes for end-to-end claims;
- optional MFLA/BFLA versions pinned;
- no cached external input without checksum verification;
- no comparison across changed benchmark schema/driver without an explicit
  compatibility decision.

### Acceptance gates

Correctness is lexicographically prior to speed:

1. same mathematical problem, cone representation, objective sense,
   tolerances, and reconstruction;
2. semantic pass and valid original-coordinate certificate for every timed row;
3. no status, objective, route, or provider mismatch outside an explicitly
   reviewed change;
4. no worse accuracy beyond the row's absolute/relative tolerance;
5. no new fallback, factorization failure, precision change, or instability;
6. median runtime regression no greater than 3% for stable rows (5% for very
   small noisy rows);
7. allocated bytes and workspace bytes regression no greater than 3% unless a
   measured time/robustness tradeoff is documented;
8. a claimed speedup should normally be at least 5%, or a memory reduction at
   least 10%, across three fresh processes;
9. iteration growth greater than two iterations or 10% requires explicit
   accuracy and runtime justification.

## Refactor sequence and rollback commits

### Phase 0 — evidence and contracts

1. Fix cross-entry BigFloat precision validation on `main`. **Completed.**
2. Add this architecture review and target contracts.
3. Add the fixed nine-cell benchmark campaign and tests.
4. Produce a current baseline from the reviewed commit before numerical edits.

### Phase 1 — behavior-preserving decomposition

1. Split `pipeline.jl` into validation, feature extraction, presolve, planning,
   precision, and diagnostic configuration files while retaining existing types
   and function names.
2. Split public optimize code into start mapping, family adapters,
   reconstruction/certification, and entry-point orchestration.
3. Split the SDP loop into initialization, iteration, termination, checkpoint,
   and trace construction around one `SolveContext`.
4. Move existing KKT formulations and LA providers into their target folders
   without changing numerical kernels.

Every extraction is one rollbackable commit and must preserve test results,
plans, result fingerprints, benchmark semantics, and performance within noise.

### Phase 2 — one planning and linear-system authority

1. Introduce immutable `KKTPlan` and `LAPlan` records.
2. Separate formulation assembly from factorization provider state.
3. Normalize symbolic/numeric lifecycle and counters across LP/SOCP/SDP.
4. Replace implicit workspace-buffer route inference with plan-authorized
   execution.
5. Represent all fallback chains and failure reasons before iteration starts.

### Phase 3 — measured numerical optimization

Only phase traces select work. Candidate areas include:

- active-column/pair Schur and SOCP metric assembly;
- symbolic factorization and numeric-refactor reuse;
- caller-owned solve/refinement scratch;
- block-arrow equality tail and reduced-system kernels;
- reduced copying and precision-preserving buffer reuse;
- memory-budget-aware thread scheduling;
- prepared solver reuse across repeated structure;
- bounded mixed-precision factorization and refinement policy.

Each candidate receives its own frozen microbenchmark, paired end-to-end rows,
and revertible commit.

### Phase 4 — API and repository simplification

1. Keep `Settings` small; move unstable tuning controls behind an explicitly
   expert API.
2. Remove compatibility aliases only after repository-wide use search,
   deprecation, and tests.
3. Retire duplicate benchmark drivers in favor of the canonical registry.
4. Remove obsolete workflows, generated work products, and dead modules only
   after reachability and artifact-history checks.
5. Reduce unconditional package load where extension and include boundaries
   permit it without changing public behavior.

### Phase 5 — production qualification

- public Netlib, SDPLIB/DIMACS, and supported CBLIB families;
- pathological infeasible, degenerate, near-tangent, rank-deficient, scaled,
  and small-eigenvalue cases;
- Float64, Float64x4, BigFloat256 and selected higher BigFloat precisions;
- single-thread determinism and controlled multithread scaling;
- repeated solves/prepared structures;
- clean full test suite, Aqua/interface checks, documentation, and release notes.

## Immediate priorities

1. Establish the current nine-cell baseline and phase breakdown.
2. Use the baseline to choose between Schur/metric assembly, factorization,
   equality processing, or setup/certification as the first measured target.
3. Perform behavior-preserving file decomposition before algorithm changes only
   where it can be proven neutral by the same baseline.
4. Do not delete legacy code merely because a modern abstraction exists; first
   prove no plan, extension, test, benchmark, or compatibility entry reaches it.
5. Do not broaden augmented KKT or mixed-precision use until provider capability,
   inertia/refinement, and pathological accuracy gates are current.

## Definition of success

The refactor succeeds when:

- a reader can follow Model -> IR -> Plan -> SolveContext -> LinearSystem ->
  Certificate without entering unrelated files;
- each numerical solve has exactly one immutable authoritative plan and one
  explicit executed-route record;
- formulation and factorization providers can be benchmarked independently;
- Float64, Float64x4, and BigFloat256 use the same lifecycle with
  arithmetic-specific providers rather than separate solver architectures;
- accepted performance changes are linked to reproducible baseline/candidate
  artifacts;
- original-coordinate correctness remains the final authority;
- the public API is smaller than the implementation, stable, and documented;
- obsolete code is deleted only after measured replacement and reachability
  proof.
