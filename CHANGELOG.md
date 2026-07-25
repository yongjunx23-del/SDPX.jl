# Changelog

All notable changes to SDPX.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

First public release, prepared as a standalone package derived from
[SDPJSolver.jl](https://github.com/FishboneChiang/SDPJSolver.jl) (MIT,
Li-Yuan Chiang). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for what
is derived and what is original.

Because this is the first release under the SDPX name, the list below describes
the state of the package rather than a delta against a previous SDPX version.

### Solver core

- Primal-dual interior-point method with the HRVW/KSH/M direction and a
  Mehrotra predictor-corrector.
- SDPB-style Cholesky block elimination of the KKT system, with the Schur
  complement built in symmetric-square form so it is positive definite by
  construction.
- Dedicated arrow-structured KKT path for models with shared and per-block local
  variables, and a fused compute-and-scatter Schur kernel for models whose
  blocks are all 2x2. The fused path omits both transformed-panel and packed
  pair storage when neither is consumed.
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
- Fixed-width extended arithmetic uses the multicore scheduler. `BigFloat`
  deliberately remains serial to preserve the validated ownership model and
  avoid per-worker arbitrary-precision workspace growth.
- Optional cache-blocked, panel-packed triangular `syrk!`/`gemm!` kernels for
  fixed-width extended arithmetic and BigFloat. Selection accounts for
  dimensions, density, thread count, packing cost, expected Schur density, and
  available memory; the backend remains disabled by default. Exact-arrow
  `2x2` models bypass this optional packing route for both arithmetic families
  and use the fused no-panel/no-pair-buffer kernel.
- Memory planning honors host free memory, Linux cgroup v1/v2 limits, and the
  optional `SDPX_MEMORY_LIMIT_BYTES` ceiling.

### Robustness and diagnostics

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
- Exact fraction-to-boundary step selection for models with 2x2 blocks, selected
  automatically.
- Ruiz equilibration for dense and sparse inputs, arithmetic-type-native
  presolve for dependent equality columns with consistency checks and dual
  reconstruction, and checkpoint save/resume. Warm starts are mapped from
  original coordinates through presolve and equilibration; checkpoints are
  iterate-level warm restarts that reset adaptive, stagnation, timing, and
  best-iterate history.
- Guarded adaptive beta/gamma control driven by affine-predictor quality,
  observed complementarity reduction, line-search success, backtracking, and
  feasibility. Per-iteration values and fallback state are recorded;
  adaptation remains opt-in because the representative SDP benchmark did not
  improve.
- Final-result certification in the original problem coordinates, including
  equality, primal-cone, dual, and PSD checks before an `Optimal` status is
  retained.
- Optional spectrum reconstruction and atomic CSV/JSON/JLD2 export after a
  successful solve. JLD2 stores a versioned `spectrum` payload with
  `format_version`, `metadata`, and `records`.

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
- BigFloat execution is serial. Large non-arrow high-precision SDPs still use
  dense Schur/KKT storage and can exceed practical memory before a full solve
  is attempted.
- Equality-constrained LPs still use dense LU, native SOCP scaling and chordal
  SDP decomposition are not implemented, and distributed Schur
  factorization is future work.
- No general solver-to-solver performance claim is made in this release.
  Narrow, reproducible comparison measurements retain their exact instance,
  environment, convergence, and caveat labels; see [README.md](README.md).
