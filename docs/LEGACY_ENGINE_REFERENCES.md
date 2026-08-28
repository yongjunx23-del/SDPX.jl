# Remaining legacy source references

This is a temporary source-only retirement checklist. Historical unit/kernel
regressions were archived outside the repository by explicit project decision;
`test/` now contains only the black-box public E2E. Git history and the local
archive retain the removed tests.

## Current include sites

`src/SDPX.jl` still includes:

```text
soc_native.jl
step.jl
lp_sparse.jl
lp_solver.jl
solver/interior_point.jl
```

The standalone `src/hsd/nonnegative_hsd.jl` solver has already been deleted;
its seven product-HSD helpers live in `src/hsd/common_runtime.jl`.

## `lp_sparse.jl` and `lp_solver.jl`

The dedicated LP implementation is no longer a public engine selector, but the
legacy SDP loop in `solver/interior_point.jl` still calls its LP branch.
Production code outside that legacy pair has no direct `solve_lp!` caller.

Retirement sequence:

1. remove the `:lp_primal_dual` branch from the legacy interior-point loop;
2. verify no source caller remains for LP workspace/sparse-factor symbols;
3. remove both includes and both files together; and
4. retain the public general LP E2E as the external behavior gate.

## `step.jl`

Native-consumer residual, factor, right-hand-side, and BigFloat ownership
helpers were moved to `src/kernels/threaded.jl`.

`step.jl` still owns the old `newton_step!` orchestrator and private legacy
line-search/factor-quality helpers. `src/kernels/threaded.jl` retains calls to
`line_search!` and `fraction_to_boundary_search!` for the legacy path, and
`solver/interior_point.jl` consumes `newton_step!`.

Retirement sequence:

1. remove the old interior-point solve path;
2. prove no nonlegacy threaded caller needs the two retained line-search
   helpers;
3. remove `step.jl` and its include; and
4. remove any now-dead compatibility helpers from `kernels/threaded.jl`.

## `solver/interior_point.jl`

Generic helpers for certificates, checkpoints, options, scaled identities, and
precision-ladder policy were extracted to their native owner files. The file
still contains the legacy SDP solve loop.

It remains reachable from `src/public/optimize.jl` through qualified
compatibility entry points. Removing it requires retargeting or deleting those
source calls, not merely renaming the file.

`Printf` remains a runtime dependency while the legacy iteration printer is
included.

## `soc_native.jl`

The old SOC engine is still a production source dependency. Current callers
include:

- `frontend/high_level_solve.jl` via `_solve_native_soc_core`;
- `pipeline/plan.jl` via `_build_native_soc_payload`, `NativeSOCPlan`, and
  `FixedTraceQ3Execution`;
- `public/optimize.jl`, `validation.jl`, and `soc_presolve.jl` via
  `NativeSOCDiagnostics`.

Retirement requires routing these entry points through product-cone HSD and
moving only still-required typed plan/diagnostic records to native owners.
Fixed-trace Q3 remains a local specialization of the shared Newton system, not
a separate solver.

## Public selectors and certificates

- Public `engine=:legacy` and family algorithm selectors are rejected.
- The LA-provider token `:legacy` is a separate compatibility namespace and
  does not select a solver engine.
- No factor/provider status may replace an original-coordinate certificate.
- Deletion is complete only when source search, package load, and the sole
  public E2E succeed without these includes.
