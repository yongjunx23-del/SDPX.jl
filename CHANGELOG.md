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
  blocks are all 2x2.
- Sparse constraint storage with a flat COO layout and precomputed column-major
  indices.
- Dedicated linear-programming path for models whose cones are all 1x1.

### Arithmetic and performance

- Generic element type: `Float64`, `BigFloat`, `MultiFloats.Float64xN`, and
  `DoubleFloats.Double64`, the latter three via package extensions.
- Allocation-free MPFR kernels built on MutableArithmetics.
- Multithreaded block factorisation, Schur assembly, and line search, with
  longest-processing-time scheduling and phase-aware BLAS thread control.
- `BigFloat` runs multithreaded and is verified bit-identical at 1, 2, 4 and 8
  threads.

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
- Ruiz equilibration for dense and sparse inputs, presolve for dependent
  equality rows, and checkpoint save/resume.

### Interfaces

- `solve`/`solve!` with typed `SDPProblem`, `SolverOptions` and `SDPResult`.
- MathOptInterface wrapper, usable from JuMP.
- The legacy `sdp`/`findFeasible` API of SDPJSolver.jl is preserved; the global
  setters `setArithmeticType`, `setSparseMode` and `setMode` remain as
  deprecated shims.

### Known limitations

- The sparse conformal-bootstrap benchmark bundled in `bench/` does not yet
  converge to the tolerance a reference solver reaches on the same instance. See
  `bench/opt2026/REPORT.md` for the current diagnosis and measurements.
- No solver-to-solver performance claims are made in this release; see
  [README.md](README.md).
