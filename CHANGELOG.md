# Changelog

All notable changes to SDPX.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — 2026-07-27

Correctness, honesty-of-reporting, and cleanup release driven by the
2026-07-26 maintainer review (docs/maintainer-review-2026-07-26.md, which
closes with the full implementation and deferral record).

### Fixed

- All 25 fallback exception handlers now rethrow `InterruptException`,
  `OutOfMemoryError`, and `StackOverflowError` instead of absorbing them; a
  source-walking test keeps new bare `catch` blocks out of `src/`.
- `selected_algorithms` in diagnostics reports the KKT backend and Gram
  kernel that actually executed (the LP path selects its sparse Newton
  system at runtime, after the plan is frozen); the plan remains visible
  under `planned`.
- The dedicated LP path applies the same BigFloat precision-consistency
  check (and optional input rerounding) as the SDP core; previously a
  128-bit-input LP inside a 256-bit solve proceeded without a warning.
- `dense_workspace_floor_bytes` and `estimate_sdp_workspace_bytes` use
  saturating arithmetic; native-Int products could wrap negative at
  synthetic dimensions and pass every memory-budget comparison.

### Changed

- `solve_summary` publishes the shifted-Cholesky PSD bound as
  `psd_shift_lower_bound`; `minimum_psd_eigenvalue` remains as a deprecated
  alias with the same value (it was never an eigenvalue) and will be removed
  at 1.0.
- `Workspace.Qchol` and `LPWorkspace.sparse_system` are typed unions instead
  of `Any`, removing two dynamic dispatches per iteration.
- Example 03 generates its data from an explicit deterministic stream, so
  the routing decision it asserts is identical on every Julia version.

### Documentation

- Known limitations now state the one-solve-per-process contract (BLAS
  thread width is process-global), the `ingest(validate=false)` contract,
  and the experimental opt-in status of the null-space reduction and
  chordal detection (tested, not reachable from `solve`, with the measured
  reason recorded).

### Removed

- Three unreferenced internal functions (dead code, no behavioral change).

## [0.2.0] — 2026-07-27

First public release, prepared as a standalone package derived from
[SDPJSolver.jl](https://github.com/FishboneChiang/SDPJSolver.jl) (MIT,
Li-Yuan Chiang). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for what
is derived and what is original.

Because this is the first published release under the SDPX name, the list
below describes the state of the package rather than a delta against a
previous published SDPX version.

### Solver core

- Primal-dual interior-point method with the HRVW/KSH/M direction and a
  Mehrotra predictor-corrector.
- A separate, typed iteration-parameter policy layer with fixed and guarded
  adaptive implementations. The opt-in adaptive SDP path uses an affine
  predictor, bounded Mehrotra centering, independent primal/dual step
  safeguards, refinement selection, per-iteration diagnostics, and complete
  fixed-path fallback after degraded factors or unstable progress. Fixed mode
  remains the default after the Task_Low08 promotion gate.
- SDPB-style Cholesky block elimination of the KKT system, with the Schur
  complement built in symmetric-square form so it is positive definite by
  construction.
- Dedicated arrow-structured KKT path for models with shared and per-block local
  variables, and a fused compute-and-scatter Schur kernel for models whose
  blocks are all 2x2. The fused path omits both transformed-panel and packed
  pair storage when neither is consumed.
- Optional direct elimination for singleton-local `2x2` arrows. It forms the
  reduced shared Schur matrix from one combined two-row-per-block panel and a
  lower-triangular blocked SYRK, avoiding both pairwise shared contractions and
  the later sequence of dense local rank updates.
- Sparse constraint storage with a flat COO layout and precomputed column-major
  indices.
- Dedicated linear-programming path for models whose cones are all 1x1,
  including packed weighted-Hessian assembly and a Cholesky Newton solve for
  equality-free LPs.

### Arithmetic and performance

- Generic element type: `Float64`, `BigFloat`, `MultiFloats.Float64xN`, and
  `DoubleFloats.Double64`, the latter three via package extensions.
- Ownership-aware MPFR kernels built on MutableArithmetics. Solver workspaces
  keep independently owned `BigFloat` entries and reuse them in matrix
  products, vector updates, triangular solves, Cholesky factors, Schur/KKT
  right-hand sides, LP Hessian assembly, and general sparse COO Schur
  contractions.
- Multithreaded block factorisation, Schur assembly, and line search, with
  longest-processing-time scheduling, a workload crossover for small Float64
  problems, lower-triangle-only reduction where supported, and phase-aware
  BLAS thread control.
- Fixed-width extended arithmetic uses the multicore scheduler. General native
  `BigFloat` kernels remain serial, while exact singleton-local `2x2` arrows
  may parallelize disjoint block preparation and lower-triangular Schur tiles
  without sharing writable MPFR objects. An opt-in mixed path builds exact
  metrics and residuals in BigFloat while a Float64x4 panel/factorization uses
  multiple workers; failed refinement first falls back to the exact native
  reduced panel and only then to legacy pairwise assembly.
- Optional cache-blocked, panel-packed triangular `syrk!`/`gemm!` kernels for
  fixed-width extended arithmetic and BigFloat. Selection accounts for
  dimensions, active density, expected shared-Schur density, thread count,
  packing cost, arithmetic type, and available memory. Float64x4 and BigFloat
  default to conservative `:auto`; Float64 is never redirected, and rejected
  cases retain the fused
  no-panel/no-pair-buffer kernel.
- Four-lane `MultiFloatVec{4,Float64,4}` lower-triangular SYRK specialization
  for Float64x4 reduced panels. It preserves each lane's scalar reduction
  order, uses disjoint output tiles across Julia tasks, and allocates nothing
  in the arithmetic loop.
- Dense lower-triangular Float64x4 Schur products used by iterative refinement
  now assign complete output-row ranges to workers. The arithmetic loop is
  allocation-free, never shares writable output, and is not selected for
  aliased vectors or systems below the measured crossover.
- Guarded dense mixed-precision solves can refine a moderately inaccurate
  predictor before paying for native extended-precision factorization.
  Refinement defaults to the tighter of the arithmetic floor and the square
  of the requested solve tolerance; explicit `refine_tol` still overrides it.
  Float64x4 automatic mode uses a wider condition ceiling; explicit `:on`
  records conservative condition/step predictions but lets measured residual
  decrease decide. Predictor and corrector guards plus native fallback remain
  mandatory.
- The opt-in `Float64x4` dense KKT path has a guarded `Float64x2`
  intermediate fallback. Its lower-triangular blocked Cholesky, parallel
  first-touch conversion and triangular solve, equality SYRK, and disjoint
  matvec recovery avoid the multi-thousand-second native factorization seen
  on Task_Low08. Each promoted solve is accepted only after a `Float64x4`
  residual check; stale promoted factors are not reused across outer
  iterations because the measured retry passed none of its residual guards.
- Float64-to-BigFloat mixed-solve copies now write directly into reusable MPFR
  destinations with `mpfr_set_d`, eliminating one temporary BigFloat object
  per vector entry without changing rounding or scalar ownership.
- Extended-precision smoke CSV output uses stable arithmetic aliases, avoiding
  an unquoted comma in MultiFloats' concrete type name.
- Precomputed three-bit `2x2` coefficient masks remove repeated structural-zero
  tests from high-precision contraction loops. Singleton local factors cache
  their inverse, and all optional reduced paths store and compute only one
  Schur triangle.
- On the canonical medium CSDR model, the native 256-bit BigFloat reduced-arrow
  path lowers solve time from 280.011 seconds to 205.202 / 191.701 / 110.741 /
  86.752 seconds at 1 / 2 / 4 / 8 Julia threads, with identical 41-iteration
  objectives and certificates. These measurements are instance- and
  hardware-specific, not a general solver claim.
- Memory planning honors host free memory, Linux cgroup v1/v2 limits, and the
  optional `SDPX_MEMORY_LIMIT_BYTES` ceiling. Workspace estimates include
  portable object-header/allocator overhead rather than under-reporting small
  Linux workspaces.

### Robustness and diagnostics

- Added a conservative typed preprocessing pipeline with exact scalar-bound
  extraction and merging, exact fixed-variable elimination, exact
  zero/duplicate/proportional equality cleanup, original-coordinate warm-start
  and solution reconstruction, analysis-only formulation/chordal cost
  estimates, and per-stage timing/allocation diagnostics.
- Fixed equality-presolve certification so QR roundoff in a reconstructed zero
  right-hand side is not misreported as a proof of infeasibility. Exact zero
  and exact proportional contradictions remain certified; ambiguous numerical
  relations retain the original equality system.
- Added compact single-variable scalar coefficient storage for MOI bounds and
  intervals. Sparse ingestion, classification, scaling, precision
  preparation, LP extraction, chordal analysis, and null-space materialization
  now traverse active incidences rather than an `L × m` grid.
- Sparse equilibration now copies and scales only active coefficient matrices,
  reusing one read-only empty CSC matrix per block. On the canonical medium
  CSDR model this reduced median equilibration time from 1.241 to 0.633 seconds
  and cumulative allocation from 685.8 to 269.7 MB with an identical scaled
  checksum.
- Equality presolve reuses its verified normalized dependency coefficients
  when building the reconstruction map instead of factorizing the retained
  equality matrix a second time.
- Added MOI support for scalar and variable `Interval` constraints, including
  primal and signed dual reconstruction.
- Automatic stagnation detection over a rolling window of the primal residual,
  dual residual, relative duality gap, and complementarity, each normalised by
  the tolerance requested for it, so the rule scales with both the requested
  tolerance and the working precision. Distinguishes a genuine plateau from
  exhaustion of the arithmetic precision.
- Structured termination reason, measured convergence rate and projected
  iteration count exposed through `result.termination` and
  `diagnostics.termination`.
- Best-iterate retention: a stalled solve returns the best point it found rather
  than its last one.
- Restart repair confined to the side that is actually infeasible, and
  recentering as a non-destructive alternative to rescaling when a step
  collapses while the residuals and the search direction are both healthy.
- Adaptive iterative refinement driven by the KKT residual, with rollback if a
  refinement pass increases it.
- Opt-in staged BigFloat working precision. A conservative tolerance- and
  dimension-based selector may start below the requested precision, accepts
  only an independently certified result, and otherwise retries at the
  requested precision within the remaining time budget.
- Conservative automatic ownership of refinement work: exact unregularized
  reduced Float64x4 and native singleton-arrow BigFloat factorizations may
  omit the explicit residual pass when the outer tolerance is no tighter than
  `sqrt(eps(T))`; explicit, tight, regularized, and mixed paths retain it.
- Exact fraction-to-boundary step selection for models with 2x2 blocks, selected
  automatically.
- Ruiz equilibration for dense and sparse inputs, arithmetic-type-native
  presolve for dependent equality columns with consistency checks and dual
  reconstruction, and checkpoint save/resume. Warm starts are mapped from
  original coordinates through presolve and equilibration; checkpoints are
  iterate-level warm restarts that reset adaptive, stagnation, timing, and
  best-iterate history.
- Guarded adaptive Newton control driven by affine complementarity, factor
  quality, line-search behavior, and feasibility progress. Per-iteration
  values and fallback state are recorded; adaptation remains opt-in because
  the dense Task_Low08 benchmark did not improve.
- Final-result certification in the original problem coordinates, including
  equality, primal-cone, dual, and PSD checks before an `Optimal` status is
  retained.
- Optional spectrum reconstruction and atomic CSV/JSON/JLD2 export after a
  successful solve. JLD2 stores a versioned `spectrum` payload with
  `format_version`, `metadata`, and `records`.
- Mixed-precision diagnostics now identify dense Float64 and reduced-arrow
  Float64x4 backends separately, including attempts, fallback reason, and
  effective panel thread count.
- Termination diagnostics record the total number of refinement corrections
  across the complete solve, in addition to the last-iteration residual.
- Automatic parameter selection has a narrow Task_Low08-like profile for
  large equality-constrained, sparse-coefficient, dense-Schur SDPs. It uses
  `beta=0.075` and `gamma=0.8` only inside the measured structural gate;
  general large equality-constrained SDPs retain `0.1` and `0.85`.
- On the matched Task_Low08 benchmark, the profile reduces Float64 from 27 to
  24 iterations. At an equal 16-thread limit on one EPYC node, SDPX measured
  32.411 seconds versus 40.369 seconds for the MOSEK reference, with both
  original-coordinate certificates valid. This is an instance-specific
  measurement, not a general solver claim.

### Interfaces

- `solve`/`solve!` with typed `SDPProblem`, `SolverOptions` and `SDPResult`.
- MathOptInterface wrapper, usable from JuMP.
- The legacy `sdp`/`findFeasible` API of SDPJSolver.jl is preserved; the global
  setters `setArithmeticType`, `setSparseMode` and `setMode` remain as
  deprecated shims.

### Known limitations

- The sparse conformal-bootstrap benchmark bundled in `bench/` does not yet
  converge to the tolerance a reference solver reaches on the same instance.
  See `bench/csdr_psd_dual/RESULTS.md` for current evidence and
  `bench/opt2026/REPORT.md` for the historical optimization log.
- General native BigFloat execution is serial. Exact singleton-local `2x2`
  arrows are a validated exception with exclusive block/panel ownership and
  disjoint lower-triangular Schur tiles. The opt-in Float64x4 reduced-arrow
  preconditioner fails the exact refinement guard on the validated medium
  model and safely falls back to native BigFloat, so it remains off by
  default. Large non-arrow high-precision SDPs still use dense Schur/KKT
  storage and can exceed practical memory before a full solve is attempted.
- Equality-constrained LPs still use dense LU, native SOCP scaling and chordal
  SDP decomposition are not implemented, and distributed Schur
  factorization is future work.
- No general solver-to-solver performance claim is made in this release.
  Narrow, reproducible comparison measurements retain their exact instance,
  environment, convergence, and caveat labels; see [README.md](README.md).
