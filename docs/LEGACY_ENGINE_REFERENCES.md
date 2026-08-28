# Remaining legacy source reference

The public `Model` path and qualified `SDPProblem` compatibility entrypoint now
execute product-cone HSD. Historical unit/kernel tests remain in the external
archive and Git history; `test/` contains the sole public E2E.

## Deleted legacy engines

The following source files and includes have been removed:

```text
src/hsd/nonnegative_hsd.jl
src/lp_sparse.jl
src/lp_solver.jl
src/step.jl
src/solver/interior_point.jl
```

Generic certificate, checkpoint, option, precision, residual, factor, and
threading helpers that still had native callers were moved to their native
owners before deletion. The old solver loops were not copied or renamed.

Qualified `solve!(problem::SDPProblem, options)` is now a thin compatibility
wrapper over `_bridge_sdp_solve`, which constructs a public model and executes
product HSD. Legacy start/continuation keywords fail explicitly.

## Remaining file: `src/soc_native.jl`

The old SOC engine is still a production source dependency. Current source
callers include:

- `frontend/high_level_solve.jl` via `_solve_native_soc_core`;
- `pipeline/plan.jl` via `_build_native_soc_payload`, `NativeSOCPlan`, and
  `FixedTraceQ3Execution`;
- `public/optimize.jl`, `validation.jl`, and `soc_presolve.jl` via
  `NativeSOCDiagnostics`.

Retirement requires routing qualified `ConicProblem` entrypoints through the
existing `_bridge_conic_solve` product-HSD adapter, then moving only any still
required typed plan/diagnostic records to native owners. Fixed-trace Q3 remains
a local `NewtonSystem` specialization, not a separate solver.

## Completion gate

Phase 10 source retirement is complete when:

1. `soc_native.jl` and its include are removed;
2. no production source references `NativeSOCPlan`, `NativeSOCWorkspace`,
   `_solve_native_soc_core`, or `NativeSOCDiagnostics`;
3. package load succeeds;
4. the public modeling-to-certified-result E2E passes; and
5. no factor/provider fact is used as a terminal mathematical certificate.
