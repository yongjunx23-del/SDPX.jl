# Changelog

All notable changes to SDPX.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — unreleased

### User-facing

- A typed all-auto `SolveOptions` frontend and an SDPB-style CLI with integer
  BigFloat bit precision and independent duality-gap, primal-error, and
  dual-error thresholds; resolved choices and the automatic `ExecutionPlan`
  remain visible in result provenance.
- Native Lorentz SOCP through `solve_socp`/`second_order_program`, including
  the `fixed_trace_q3` specialization, with original-coordinate certification.
- JuMP/MOI vector linear cones and rotated second-order cones, lowered through
  batched sparse rows and the exact rotated-to-Lorentz primal/dual map.
- End-to-end sparse LP execution and frozen-CSC sparse SDP Schur execution.
- Certificates, executed diagnostics, and the canonical public LP/SOCP/SDP
  benchmark registry under `benchmark/`.

### Developer

- Explicit KKT formulation descriptor in `ExecutionPlan` plus a static
  formulation planner, kept independent of the selected linear-algebra
  provider.
- Provider-neutral `:standard`, MFLA, BFLA, and legacy linear-algebra routing:
  provider choice is final at planning time with no hidden runtime fallback.
  The bundled legacy provider (`src/la_backends/legacy.jl`) is included
  statically, and every `LegacyLABackend` `la_*` dispatch routes through
  `_sdpx_legacy_la_call` instead of calling historical `k*` kernels directly.
- Lower-authoritative equality Grams, lightweight MFLA LDLT metadata, and a
  reusable BFLA workspace/trusted repeated-solve seam remove redundant
  symmetric mirroring, diagnostics scans, and solve scratch allocation while
  leaving equality-rank thresholds, structured refinement, and fallback
  authorization in SDPX.
- Certification fast path: `_primal_block_backward_errors` and
  `_dual_backward_errors` asserted their `AbstractCons` subtype once
  outside the per-block loop instead of re-dispatching through the
  abstract field for every block. On the ladder's 2000-block LP row the
  certification phase dropped from 0.89 s to ~0.02 s (measured >300x on
  the isolated backward-error pass; allocation fell from 415 MB to
  <1 MB per call), lifting end-to-end ladder rows by 3.5-12.7x.
  Arithmetic order is unchanged -- certificate values are identical.
- Split the 3719-line `src/pipeline.jl` into eleven ordered files under
  `src/pipeline/` (helpers, options, classify, resources, route, plan,
  presolve, workspace_estimate, attempts, diagnostics, timing). The
  concatenation is byte-identical to the original file and every chunk
  parses independently; the split separates the pure decision helpers
  (classification, route resolution, planning, presolve) from the
  diagnostics and timing plumbing that had accumulated around them.
- Split the 2577-line `src/types.jl` into seven ordered files under
  `src/types/` (core, backends, workspaces, constraints, problems, plans,
  results). The concatenation is byte-identical to the original file and the
  include order is unchanged, so definitions and semantics are exactly as
  before; the split makes the type layer navigable and future extractions
  local.
- Retired the legacy `factor_kkt!` nothing-chain dispatcher (kkt.jl): the
  production path has dispatched through `factorize!(backend, ...)` with
  plan assertions and execution provenance since the KKT backend layer
  landed, and the chain had no production callers -- only the sparse
  regression tests, which now call `factorize!(select_backend(ws), ...)`
  directly. Behaviour unchanged; the remaining per-phase `ws.arrow`/
  `ws.sparse_kkt`/`ws.mixed_precision` branches in schur/step/kernels are
  the Phase-2 KKTPlan extraction, not part of this step.
- Repository cleanup: the former `bench/` tree is consolidated — the
  acceptance gate now lives at `benchmark/gates.jl` (with
  `benchmark/baselines/gates.json` and the benchmarking environment
  `benchmark/benchenv/`), and historical application/cluster campaigns with
  their dated reports are archived under `docs/evidence/`. The five hand-linked
  topic guides are served through Documenter under "Topic guides", and dated
  review snapshots moved to `docs/evidence/development-reviews/`. One-shot
  Codex patch automation (`.codex/` payloads and their four workflows, whose
  target branch no longer exists) and the unwired COO micro benchmark were
  removed. The optimization pipeline is defined once in
  `.github/workflows/optimization-benchmark.yml`; the push loop and the
  pull-request gate are thin callers.
- Benchmark registry and scoreboard contracts are covered by the ordinary test
  suite; public benchmark files are never downloaded.
- `PreparedSolver` reuses an immutable post-presolve `ExecutionPlan` only when
  planning is invariant to objective/RHS updates; numeric workspaces and
  factors remain solve-local.
- Removed the unused canonical snapshot/feature planner so production has one
  authoritative `ResolvedExecutionRoute -> ExecutionPlan` decision path.
- Removed all structural and benchmark parameter profiles from the automatic
  SDP/LP cold-start resolver. `parameter_policy=:auto` records a deferred
  `:automatic_mehrotra` plan and resolves exactly once after scaling. On the
  SDP path the public `recommended_parameters` resolver reports
  `:post_scaling_mehrotra` and the executed record reports
  `:post_scaling_mehrotra`: `beta`, `gamma`, `predictor`, and
  `parameter_strategy` keep the `SolverOptions` defaults or user choices,
  every SDP/LP adaptive controller uses the generic 0.50 sigma cap. The LP path
  runs a provenance-only post-scaling resolver after geometric scaling. The
  expert `adaptive_sigma_max` option still
  overrides the cap, and the old profile-taking helpers are compatibility
  forwarders only. An explicit fixed policy is recorded as `:user_fixed`.
  NativeSOC's existing local predictor/corrector controller is unchanged by
  this cleanup.
- Replaced the remaining automatic `Omega_p`/`Omega_d` identity start with a
  provider-neutral, post-scaling KKT cold start for LP, SDP, and NativeSOC. A
  planned identity-metric factor supplies one primal and one dual affine solve;
  cone-specific typed shifts then establish strict interiority. A minimal
  aggregate identity shift raises orthant/PSD starts, and Lorentz sides still
  within the typed cone-vertex envelope, to unit identity mass before the
  generic cross-centering step. This prevents an affine point at the cone
  vertex from starting at machine-epsilon scale without renormalizing an
  already balanced Lorentz point. When an accepted
  regularized or mixed-precision SDP factor leaves an original-KKT residual
  above the existing gate, bounded structured correction reuses that same
  factor; the residual gate remains fail-closed. Initialization
  factor/solve counts, residuals, shifts, margins, complementarity, route, and
  failure provenance are reported separately from Newton-iteration counters.
  Automatic initialization fails closed rather than falling back to `Omega`
  heuristics. Explicit `parameter_policy=:fixed` initialization is unchanged.

### Removed

- Removed the unreleased answer-only reduced-dual L-BFGS implementation and
  its `CertifiedObjective`, `ReducedDualReconstructionToken`, `solve_value`,
  and `reconstruct_fixed_trace_solution` API. NativeSOC primal-dual IPM is the
  sole production SOCP solver.
- Removed the DoubleFloats extension and all nonhistorical Double64 support;
  unsupported arithmetic now fails before planning instead of entering a
  generic `AbstractFloat` route.

## [0.4.0] — 2026-08-09

### Added

- A compact `SOCConstraint`/`ConicProblem`/`ConicResult` frontend with
  `second_order_program` and `solve_socp`. `Q3` uses the exact two-by-two PSD
  isomorphism; other dimensions retain the general PSD-arrow reference.
- Conservative direct and equality-implied fixed-trace analysis, including
  negative/zero-trace classification and exact `2x2` SOC candidates.
- An explicit compact fixed-trace Q3 Mehrotra/HKM backend for exact sparse
  two-local-variable PSD2 cells, with ownership-safe Float64x4/BigFloat
  scheduling, triangular equality Gram strategies, PSD2 fallback, and an
  independent original-coordinate certificate gate.
- A reproducible J40/J80 CSDR benchmark harness with immutable model/reduced
  hashes, memory-only preflight, actual CPU/NUMA/resource telemetry, and strict
  requested-versus-executed backend validation.
- A tested direct-SOC/fixed-trace example that contrasts the stable Lorentz
  frontend with the narrow native Q3 fast path and verifies its reconstructed
  original-coordinate certificate.
- `PreparedSolver` for sequential reuse of ingested constraint data and an
  optional previous-solution warm start across a few objective directions.
- Ownership-safe Q3 Nesterov--Todd scaling primitives and an explicit
  `q3_direction=:nt` solve path for Float64, Float64x4, and BigFloat256. A
  same-node J40 ABBA gate remained certificate-valid but needed 225 instead of
  191 iterations and was 16.8% slower than HKM, so NT remains research-only
  and HKM remains the production default.

### Changed

- The announced experimental-export deprecation is complete: advanced
  inspection, preprocessing, parameter-policy, and backend controls remain
  available as qualified `SDPX.name` bindings and under `SDPX.Experimental`,
  but no longer enter user modules through `using SDPX`.
- The independent docs, examples, benchmark, Convex benchmark, and CLI
  environments now accept SDPX 0.4, so release documentation and examples can
  resolve against this package version.
- Sparse `2x2` solve setup caches blocks whose affine coefficients are all
  traceless without changing the serialized model layout. Schur, fused-arrow,
  native BigFloat reduced-panel, and mixed Float64x4 panel kernels use a
  two-coordinate contraction and avoid redundant diagonal arithmetic.
- MOI/Convex dimension-three SOC constraints compile directly to exact
  two-by-two PSD blocks instead of three-by-three arrows. Same-node PBS
  medians improved by 2.38x for Float64 and 3.17x for Float64x4 in the first
  representation gate; the BigFloat-256 compact case was 25.9x faster and
  certified while the old arrow stalled.
- The fixed-trace CSDR spectral-primal smoke solve improved from 1.666 s to
  1.060 s (1.57x) at identical 21 iterations in the first same-node A/B.
- Native fixed-trace BigFloat Q3 equality Grams now select ownership-safe
  output tiles when at least two workers and a conservative large-panel work
  threshold are available. On the J40 `8,400 x 170` panel, four threads reduced
  one Gram from 9.605 s to 3.250 s (2.96x) with an identical lower triangle.
  Executed diagnostics include the actual Gram thread count and selector reason.
- Automatic algorithm selection now promotes only validated large sparse
  fixed-trace problems at least as wide as Float64x4 to native Q3. On J40,
  three eight-worker runs reduced median solver time from 187.627 to 95.444 s
  (1.97x, both CVs below 0.8%); a same-allocation one-worker pair was 9.6%
  faster, with equivalent certificates and lower allocation/RSS. Float64,
  BigFloat, smaller models, and unsupported structures keep the PSD2 default.
  The final alternating 32-thread/BLAS-1 J40 gate gave 5.26x median solver and
  5.14x end-to-end speedups over optimized PSD2 across six valid rows per
  formulation, while increasing mean active cores from 7.83 to 28.73. A
  matched BigFloat256 gate deliberately did not promote Q3: its 197.07-second
  median was 6.5% slower than PSD2's 184.99 seconds and allocated 2.66x more,
  despite equivalent high-precision certificates.
- The J40/J80 benchmark harness now writes timed non-`Optimal` and invalid-
  certificate rows before returning a nonzero exit, preserving partial phase,
  resource, and validation diagnostics from expensive limited runs.
- Fixed-trace benchmark rows now retain compact adaptive-history summaries for
  centering, complementarity, affine and accepted steps, backtracking, and
  fallback events, making iteration-count regressions directly diagnosable.
- Native Q3 high-precision local solves cache one owned reciprocal per
  Cholesky pivot and reuse it for the equality panel and predictor/corrector
  solves. The Float64 path retains direct division. Adjacent-row Float64x4
  CSDR layouts also fuse the equality-panel copy and transform; executed
  diagnostics report both selected kernels.
- Near-proportional equality detection is now a bounded diagnostic rather than
  an unbounded quadratic preprocessing scan. Exact zero, duplicate,
  proportional, and target-arithmetic rank checks are unchanged. On J80 all
  350 equalities were retained while exact-cleanup time fell from 224.42 to
  0.0735 seconds and total preprocessing from 224.75 to 0.395 seconds. The
  corresponding J40 32-thread complete-solve gate improved median solver and
  end-to-end time by 22.1% and 21.8% with identical 191-iteration objectives,
  residuals, gaps, SOC/PSD margins, and certificates.
- The retained Float64x4 equality-Gram kernel now updates four output rows by
  two columns for off-diagonal tiles while preserving each output's scalar
  accumulation order. It reduced isolated J40 Gram time by 7.2--8.4% across
  8/16/32 workers and J80 Gram time by at least 20.9% at 32 workers, with
  identical lower-triangle hashes. A strict full J40 solve reduced Gram time
  by 7.9% and complete solver/end-to-end time by 4.3%/4.2% with unchanged
  objective, 191 iterations, residuals, gap, and certificate. A four-column
  variant was rejected after only 1.0--1.4% J40 kernel gains.
- Fixed-trace release benchmark templates are capped at 32 Julia/solver
  threads with BLAS fixed to one thread. J40 uses 1/2/4/8/16/32 and J80 uses
  8/16/32; wider historical experiments are retained only as archived data.
  The direct benchmark CLI enforces the same campaign cap, successful
  launchers write an explicit `PASSED` marker, and site-specific Julia,
  environment, depot, and CSDR paths are supplied through environment
  variables instead of being embedded in the public scripts.
- User documentation is consolidated around the Documenter manual and a
  shorter README. Obsolete internal review, audit, and roadmap documents were
  removed; detailed implementation history remains available in Git history.

## [0.3.1] — 2026-08-07

### Added

- Convex.jl 0.16 is now an optional package extension with `solve_convex!`, a
  typed `convex_optimizer` factory, and `convex_semidefinite`. New PSD models
  default to an upper-triangle representation with `n(n+1)/2` variables;
  `representation=:square` and ordinary `Convex.Semidefinite` preserve the
  historical interface. Float64, Float64x4, and BigFloat regressions cover LP,
  SOCP, SDP, dual recovery, and both PSD representations.
- `linear_program` and `solve_lp` provide a direct `G*x >= h`,
  `Aeq*x = beq` frontend backed by active-only scalar rows.

### Changed

- MOI scalar inequalities now retain only active coefficients, and equality
  copy-in builds one sparse triplet matrix instead of dense column
  temporaries. On a 512-variable, 1,024-inequality, 64-equality construction
  benchmark, median copy time fell from about 35.5 ms to 11.1 ms and
  allocation from 116.0 MB to 23.0 MB.
- The matched Convex benchmark now compares native SDPX, the default Convex
  triangle representation, and the legacy Convex square representation. On
  the stable side-6 Float64 SDP, triangle modeling reduced the core dimension
  from 36 variables/21 equalities to 21 variables/6 equalities and reduced
  warmed end-to-end time from 8.66 ms to 8.45 ms with equivalent certificates.

- The Float64x4 reduced-arrow backend now uses phase-specific worker counts,
  an eight-output-row SIMD Schur microkernel, a measured narrow-triangle tile
  crossover, and a sixteen-column blocked factor with cached pivot
  reciprocals, SIMD panel rows, and at most eight trailing-update workers for
  128--256 order shared systems. Uniform `2x2` systems use contiguous block
  ownership only through 32 fine-grained tasks; wider and heterogeneous
  systems retain LPT.
  On the 1,700-block / 144-shared-variable medium CSDR model, contiguous
  ownership reduced the same-node 32-worker median from 5.010 to 4.555
  seconds, while the rejected 64-worker contiguous schedule was 1.6% slower
  than LPT. A later alternating 32-worker factor A/B reduced factor time from
  0.671 to 0.140 seconds and solve time from 4.768 to 3.420 seconds, with the
  same 41 iterations and valid certificate. Executed diagnostics expose
  effective, block-task, Schur, and factor worker counts. The Float64 path is
  unchanged.
- Narrow Float64x4 reduced-arrow models now cap a 96+ requested solver width
  at 64 workers while retaining the 32-bin contiguous schedule selected from
  the original request. On the 1,700-block / 144-shared-variable model, an
  alternating 128-thread A/B reduced solve time by 3.1% and per-solve
  allocation by 23.0%, with a neutral genuine 64-thread control. A first cap
  that accidentally expanded short phases to 64 bins was rejected and is not
  present.
- Cluster guidance now records Julia worker-sleep behavior as a major
  Float64x4 wide-pool crossover. On the same 128-core EPYC node,
  `JULIA_THREAD_SLEEP_THRESHOLD=10000000` reduced the combined full-solver
  median from 11.635 to 3.409 seconds, matched the never-sleep speed ceiling
  within 1.0%, reduced allocation by 8.2%, and preserved bit-for-bit printed
  objectives and certificates. Unlike permanent spinning, the 10 ms policy
  averaged about 12.5 CPU cores over the complete process while still
  reaching 124--126 cores during parallel bursts. This is a documented,
  hardware-specific process-start recommendation, not package-global state.
- Direct reduced-arrow workspaces now allocate full task-local Schur partial
  matrices only if the structure-specific panel build falls back. The normal
  direct Float64x4 path retains only small right-hand-side partials. A
  forward/reverse, same-node medium CSDR A/B reduced the 48-worker solve from
  2.982 to 2.940 seconds, per-solve allocation from 148.46 to 118.08 MiB, and
  process peak RSS from 3.166 to 3.028 GiB. In a 128-thread capped-pool
  control it reduced solve time from 3.480 to 3.351 seconds and allocation
  from 152.07 to 111.42 MiB. Lazy fallback matrices remain independent, and
  every baseline/candidate objective and certificate was identical at a fixed
  pool width.
- The Float64x4 direct 2x2 reduced-panel pack now caches each positive
  singleton-local factor, inverse, and solved coupling while the block data is
  hot, eliminating a later local-factor pass. A 14-solve reverse-bracketed
  medium CSDR comparison reduced KKT time by 23.6% and total solve time by
  0.77%, while also lowering allocation and peak RSS. Regularized, partial,
  Float64, BigFloat, mixed-precision, and fallback routes keep the established
  factorization.
- Float64x4 singleton-arrow linear solves now use allocation-free
  `MultiFloatVec` kernels for disjoint RHS entries and local-variable
  recovery. Every lane retains the threaded baseline's reduction order and
  every output remains task-exclusive. On the medium CSDR case, predictor and
  corrector linear-solve medians improved by 6.76% and 8.88%, producing a
  further reproducible 0.63% full-solver gain. All 28 baseline/candidate runs
  shared one exact 41-iteration certificate and one serialized solution hash.
  Executed diagnostics report the selected arrow linear-solve kernel.
- A complete 1/2/4/8/16/32/48/64/96/128-worker Float64x4 audit found the
  48-worker point fastest on the dual-EPYC 1,700-block / 144-shared-variable
  profile: 2.967 seconds versus 44.112 seconds at one worker (14.87x). Every
  requested pool became active and BLAS stayed at one thread. Wider pools
  lose to synchronization and NUMA traffic even after the 10-millisecond
  Julia wake policy removes tens of millions of context switches. Cluster
  guidance now distinguishes the 128-core scheduler reservation from the
  exact 48-worker runtime pool.
- Reduced-arrow blocks that contain every shared variable now skip redundant
  panel and coupling clears because every destination is overwritten. Partial
  coverage retains the clear path and has a sentinel regression. In an
  alternating, three-warm-up 32-worker A/B on the medium CSDR model, ten
  samples per variant reduced median Schur time from 1.542 to 1.411 seconds
  and solve time from 4.629 to 4.442 seconds, with identical 41-iteration
  objectives and certificates.
- BigFloat sparse SDPs with explicit equalities now use an ownership-safe
  block-diagonal arrow KKT path when every Schur variable belongs to exactly
  one 2x2 PSD block. Local factors and equality triangular solves own disjoint
  row blocks, while the equality Gram uses disjoint lower-triangular MPFR
  tiles. Larger or shared-variable BigFloat models retain the established
  serial general KKT route. New full-rank and duplicated-column regressions
  validate KKT residuals, object ownership, rank detection, and QR fallback.
- BigFloat block-diagonal equality solves now parallelize the two dense GEMV
  operations by assigning complete, disjoint output ranges to workers. Each
  worker owns its MPFR destinations and reduction buffers, while an
  arithmetic-work crossover keeps small panels serial. The reduction order
  within every output is unchanged and regression tests require bit-for-bit
  agreement with the serial kernel.
- All-local BigFloat equality solves now use phase-aware scheduling above 64
  requested workers. Fine-grained block, local triangular, equality GEMV,
  predictor/corrector, line-search, and update work is merged into at most 64
  ownership-safe task streams, while the tiled lower equality Gram retains the
  full requested width. On the certified J40 BigFloat512 case this reduced the
  128-worker solver from 495.811 to 425.880 seconds and peak RSS by 6.6%, with
  a bit-for-bit identical 158-iteration certificate. The measured 64-worker
  configuration remains faster than both 96 and 128 workers and is the
  recommended width for that model.
- The final immutable-candidate Task_Low08 Float64 regression remained
  `Optimal` with a valid original-coordinate certificate: 28 iterations,
  33.846 seconds solver time, `2.176e-7` relative gap, `3.316e-10` primal
  residual, and `9.534e-12` dual residual. The Float64 solve route is
  unchanged by the BigFloat scheduling work.
- A fixed BigFloat1024 J40 support gate passed its physical certificate in 157
  iterations with no restart, regularization, refinement, or fallback. It took
  553.959 seconds at 64 workers and did not improve the terminating gap over
  BigFloat512 on the once-rounded Float64x4 input, so the documented
  model-specific recommendation remains 512 bits.
- Block-local residual, Cholesky, predictor, and corrector scheduling now uses
  an arithmetic-aware cubic-work crossover in addition to the historical
  256-block threshold. This activates safe disjoint-block parallelism for
  Task_Low08's 32 medium-to-large PSD blocks; a same-node 32-Julia-thread /
  16-BLAS-thread A/B reduced median solve time from 32.062 to 28.438 seconds
  (11.3%) with unchanged iterations, backtracking trajectory, and certificate.
- Large dense `Float64` Schur assembly can use up to 25% of an ample
  scheduler-aware memory budget for task-local accumulators. The crossover is
  restricted to systems with at least 4,096 variables, 16 blocks, 16 workers,
  and 16 GiB available; extended precision and smaller systems retain the 15%
  cap. On Task_Low08 this reduced median Schur time from 8.140 to 6.701 seconds
  and the stable solve from 27.449 to 25.965 seconds. A 35% policy was rejected
  because its small median gain was not stable and used more peak memory.
  `schur_bin_report` exposes the selected fraction and effective byte budget.
- The adaptive controller now has an inspectable expert
  `adaptive_sigma_max` guard. Zero delegates to structural selection. The
  `large_lattice_dense_schur` profile selects 0.20 after a same-node
  Task_Low08 sweep: 28 instead of 29 Newton iterations, 119 instead of 174
  backtracking contractions, and a tighter valid original-coordinate
  certificate. Other profiles retain the generic 0.50 bound.
- Added a warmed, same-process Task_Low08 adaptive-cap benchmark so parameter
  experiments share input, presolve, compilation, node, and validation
  boundaries.
- Automatic scaling now preserves original coordinates when the calibrated
  `large_lattice_dense_schur` profile is paired with the expert fixed
  parameter strategy. That combination restores its validated 24-iteration
  trajectory; adaptive lattice solves continue to use Ruiz scaling.

### Fixed

- `MOI.RawSolver()` now passes `SDPResult` unchanged through the caching
  optimizer automatically inserted by Convex.jl. Previously, reading the raw
  result after a successful Convex solve failed because MOI attempted to remap
  indices in an index-free result object.

- Automatic rank-deficient equality handling now switches directly from a
  rejected normal-equation factor to the existing rank-revealing QR backend.
  This avoids an unnecessary pivoted-Cholesky probe and restores BigFloat
  compatibility with Julia 1.10, whose standard library does not implement
  generic pivoted Cholesky. Forced normal-equation mode retains its explicit
  pivoted-Cholesky behavior.

## [0.3.0] — 2026-07-31

### Added

- Optimize mode now has distinct `PrimalInfeasible` and `DualInfeasible`
  statuses. SDPX promotes a failed iterate only after a normalized homogeneous
  ray passes independent affine, PSD, objective-sign, and finite-value checks
  in original coordinates. The result, certificate API, and MathOptInterface
  statuses expose the validated ray. The certificate boundary is compatible
  with a future homogeneous self-dual iteration but does not claim that the
  current Newton system carries HSD `τ` and `κ` variables.
- Equality-only LPs now return their analytic normalized null-space ray when
  the objective is unbounded, replacing the previous numerical-error status
  with a validated `DualInfeasible` certificate.
- `SDPX.Experimental` provides namespaced access to advanced preprocessing,
  parameter-policy, inspection, and backend controls. Their historical
  top-level exports remain for the 0.3 deprecation cycle and are scheduled to
  stop being exported in 0.4; legacy SDPJSolver-style exports retain their 1.0
  compatibility window. `SDPX.api_surface()` publishes this policy for release
  tooling and downstream audits.
- The solver implementation now sits behind `src/solve.jl` as a small include
  manifest, beginning the staged decomposition of the former monolithic file
  without reordering numerical methods in the 0.3 release.
- GitHub Actions coverage upload uses Codecov OIDC with a failing upload gate,
  and pull requests from repository branches deploy Documenter previews. This
  makes the first real coverage upload and documentation deployment
  independently observable in CI.
- Failed optimization runs can now perform conservative homogeneous-ray
  diagnostics. A normalized dual ray can diagnose primal infeasibility; a
  normalized primal ray can diagnose dual infeasibility or primal
  unboundedness. In 0.3 these checks also back formal statuses when independent
  validation succeeds.
- `SolverOptions(T; ...)` accepts ASCII aliases such as `tolerance`,
  `maximum_iterations`, `time_limit`, `beta`, `gamma`,
  `primal_initial_scale`, and `dual_initial_scale`, while preserving the
  existing typed Unicode constructor.
- Exactly block-diagonal sparse SDP systems with equality constraints now use
  the local block-arrow factors directly instead of materializing the full
  Schur matrix. The transformed equality system retains fast Gram-Cholesky
  when stable and automatically switches eligible ill-conditioned systems to
  rank-revealing Householder QR.
- `Float64x4` and other fixed-width extended equality Gram matrices can use
  the existing allocation-free blocked triangular SYRK with disjoint output
  tile ownership. The equality-specific crossover rejects panels below 32
  columns or 250,000 scalar contractions in automatic mode; Float64 retains
  its unchanged vendor-BLAS SYRK path.
- Equality diagnostics now report the executed factorization, numerical rank,
  factor quality, rank-deficiency verdict, and Gram kernel. Uncertified
  terminal states explicitly warn that iterate objectives are not valid
  physical bounds.
- Public SDP defaults now use guarded adaptive Newton control, adaptive-pass
  Ruiz scaling, automatic equality-factor selection, automatic
  extended/mixed-precision kernels, and staged BigFloat working precision.
  Historical fixed behavior remains available through expert overrides and
  the legacy compatibility interface.
- A guarded equality-aware sparse Schur backend for large Float64 SDPs. It
  assembles only the lower CSC triangle with exclusive column ownership,
  reuses CHOLMOD's symbolic Cholesky analysis, solves all equality columns as
  one dense multi-right-hand-side operation, and factors the smaller equality
  normal matrix with the existing rank-revealing fallback. The route is
  selected only for sparse coefficient/equality storage, at least 10,000
  primal variables, predicted Schur density at most 10%, and an Int32-safe
  worst-case Schur factor.
- Sparse equality matrices now remain sparse through ingest, structural
  cleanup, equality slicing, Ruiz equilibration, and BigFloat rerounding.
  Float64 dependent-equality presolve uses SuiteSparse SPQR; large
  extended-precision sparse equality systems retain all numerically ambiguous
  columns instead of densifying them.
- The sparse SDP backend now caches a required equality-pivoting decision,
  factors the pivoted equality buffer in place, and reports factor nonzeros,
  symbolic reuse, regularization, and refinement counts in diagnostics.
- Equality elimination in the sparse SDP backend now applies a diagonal
  congruence to `Q = B' * S^-1 * B` before factorization and maps the Newton
  multiplier back afterward. This exact coordinate transformation avoids an
  unnecessary rank-revealing factorization on the B3 no-box model and reduces
  its equality-factor phase from 5.93 to 1.09 seconds.
- `force_gc=true` is functional again: it performs a full collection after
  every accepted iteration and, on glibc Linux, trims released allocator
  pages. The default remains off; the option is intended for very large sparse
  factor and multi-right-hand-side workloads where releasing retained
  allocator pages matters more than collection overhead. It does not guarantee
  a lower factorization-local peak RSS.
- `ActiveSparseCoefficientVector`, an active-only input representation for
  PSD blocks that touch a small subset of global variables. It retains the
  existing sparse-matrix vector interface while avoiding an `L × m` grid of
  empty references. On the 40,400-block / 40,453-variable CSDR acceptance
  case, matched Float64x4 ingestion allocations fell from 233.2 GB to
  12.8 GB and peak RSS from 136.8 GiB to 4.9 GiB; solver iterations,
  objectives, residuals, and PSD margins were unchanged.
- A language-independent command-line bridge (`bin/sdpx_solve.jl`, JSON
  schema v1 in `docs/bridge-schema.md`): numbers travel as strings above
  `Float64`, failures produce a structured result file, and a Mathematica
  package (`mathematica/SDPXLink.wl`) provides `SDPXOptimize` on top of it
  via `RunProcess`. Verified end-to-end from Mathematica 14.1: Float64 and
  256-bit BigFloat round trips (~30 correct digits), structured failures for
  invalid input, unknown precision, and a missing Julia executable, and
  temporary-file cleanup on every path. The documented upgrade path to a
  persistent server or LibraryLink keeps this schema as the contract.

### Fixed

- Structural presolve contradictions now carry an explicit proof reason and a
  valid independent diagnostics certificate instead of an unqualified
  `InfeasibleCert` with `reason=:none`.
- The precision-floor regression fixture now pins its shared variables. Its
  former version had a genuine negative-objective recession direction and was
  therefore a dual-infeasible model, which the new ray detector correctly
  exposed.
- Allocation regression gates now compare three warmed steady-state solves and
  retain the minimum, matching the Schur-kernel gate. This excludes one-time
  thread-pool/task-local initialization without loosening the byte ceilings.
- Objective equilibration now applies its scale correction only to the
  internal gap acceptance threshold. This prevents an internally accepted
  scaled gap from failing the requested tolerance after reconstruction in
  original coordinates, without feeding an artificially strict threshold
  into adaptive parameter control or stagnation detection.
- Full affine-residual certification is deferred until both feasibility and
  the relevant objective/gap condition can pass. Large SDP solves no longer
  rebuild all original-coordinate residuals on every feasible-but-not-yet-
  optimal iteration; the independent final certificate remains mandatory.
- Final primal SDP slacks are rebuilt from the returned `x` and the original
  affine data before certification. This removes congruence-amplified internal
  slack residuals without changing the iterate or objective, and makes the
  returned `X` exactly represent the original-coordinate PSD matrices.
- A pivoted but full-rank equality-normal factor no longer forces the adaptive
  parameter controller to fall back to fixed parameters. The sparse KKT result
  now distinguishes the factorization algorithm from an actual numerical rank
  loss; this matters on B3, where pivoting starts several iterations before
  the reported rank first drops.
- Sparse Schur factor quality is measured from the regularized diagonal that
  was actually factorized, rather than the unregularized diagonal. A valid
  equality-controlled zero diagonal therefore no longer disables adaptive
  parameters on the first iteration; residual refinement and actual equality
  rank loss remain the safety checks.

### Changed

- The Documenter site now includes quick-start, precision, automatic-pipeline,
  parameter, diagnostics, JuMP, and development guides. Auxiliary Julia
  environments carry explicit compatibility bounds, and CI checks release
  metadata consistency, uploads coverage, and can deploy documentation with
  the repository `GITHUB_TOKEN`.
- Acceptance baselines now reflect the guarded adaptive default: the
  closed-form SDP drops from 19 to 9 iterations and the dense gate from 10 to
  9, while both retain `Optimal` status and tighter relative gaps.
- On the 61,603-variable B3 no-box smoke case, the new sparse Schur route
  removed 328 dependent equalities, reduced first-iteration peak RSS from
  about 89.5 GB to 41.1 GB, and completed the first Newton iteration in
  103.02 seconds on the controlled node155 run, versus roughly 162--200
  seconds per iteration in the former dense run. Full trajectories,
  certificate status, and rejected alternatives are recorded in the
  [B3 report](bench/lattice_bootstrap/B3_NO_BOX_MOSEK_SDPX_REPORT.md); a
  non-optimal iterate is never reported there as a bound.
- Automatic refinement on the regularized sparse Float64 SDP route now keeps
  two guard digits beyond the requested certificate instead of always
  targeting 64 machine ulps. Explicit `refine_tol` values still override the
  policy, and original-coordinate final certification is unchanged.
- Accepted-step updates, finite-value checks, complementarity targets, dual
  objective evaluation, and best-iterate copies now use exclusive block
  ownership for immutable arithmetic on SDP models with at least 256 blocks.
  Per-block scalars are reduced in block order, and mutable `BigFloat` retains
  the safe serial path while still benefiting from fused scans.
- On the 40,400-block CSDR `J/K/Na/Nmu = 200/2/10/400` benchmark
  (Float64x4, adaptive parameters, Ruiz equilibration, 64 Julia threads, one
  BLAS thread, AMD EPYC 7742), the corrected solver reached `Optimal` in 70
  iterations with original-coordinate gap `4.00e-11`. Solve time fell from
  142.88 to 109.92 seconds and solve allocations from 5.55 to 4.11 GB, despite
  the previous run stopping nine iterations earlier and failing its requested
  original-coordinate gap. The serial update phase fell from 8.95 to 0.81
  seconds. See
  `bench/opt2026/CSDR_J200_GAP_AND_BLOCK_PASSES_2026-07-28.md`.

## [0.2.1] — 2026-07-27

Correctness, honesty-of-reporting, and cleanup release driven by the
2026-07-26 maintainer review, whose full implementation and deferral record is
preserved in Git history.

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
