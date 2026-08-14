# Bundled SDPXLegacyLA Provider: Architecture and Migration Plan

Status: architecture and migration contract for the bundled legacy LA path.
The provider is implemented in `src/la_backends/legacy.jl`; this document is
the maintainer contract for the provider, planner, call-site migration,
diagnostics gates, and deletion checklist.

## Dependency Direction

The SDPX solver owns all problem semantics; linear algebra is a replaceable
arithmetic seam underneath it. The intended dependency direction is:

```text
SDPX planner / sparse symbolic structure / KKT assembly / certification
    -> unified LA API (plan, instantiate, la_* dispatchers)
       -> StandardLinearAlgebra (LinearAlgebra/BLAS/LAPACK)
       -> bundled SDPXLegacyLA provider (owned legacy k* kernels)
       -> optional MFLA provider (MultiFloatLinearAlgebra extension)
          -> arithmetic library (LinearAlgebra, BLAS, MutableArithmetics,
             owned-scalar helpers)
```

The planner (`plan_la_backend`, `instantiate_la_backend`) remains the only
authorized entry point. Solver code selects a backend by requesting
`requested=:legacy` or by accepting an automatic plan whose fallback reason is
legacy; it never switches arithmetic providers inline. Certification and
diagnostics consume the executed-backend record (`executed_la_backend`,
`executed_la_provider`, `executed_la_ownership`, fallback reasons) instead of
inferring the backend from the presence of a type or a global.

## Extraction Boundary

### Current provider capabilities

The bundled `SDPXLegacyLA` provider owns the historical `k*` kernels behind
the `la_*` dispatcher seam. Callers use `la_*` dispatchers, never `k*`
directly:

- `kdot` (through `la_dot`)
- `kmul!`, `kmul_owned!` (GEMM/GEMV and owned-scalar variants)
- `ksyrk!` (see the mirroring note below)
- `kchol!` (in-place lower Cholesky, BigFloat/generic)
- `ktrsm!`, `ktrsv_lower!`, `ktrsv_transpose!`, `kcholsolve_owned!`
- `kaxpby!`, `kaxpby_owned!`
- `knrmInf`

The provider advertises exactly `SDPX_LEGACY_LA_CAPABILITIES`:
`:dot`, `:norminf`, `:mul`, `:mul_owned`, `:syrk`, `:chol`,
`:cholesky_factor!`, `:solve`, `:cholsolve_owned`, `:trsm`,
`:trsv_lower`, `:trsv_transpose`, `:axpby`, `:axpby_owned`.
The planner contract, provider payload, and diagnostics must agree on this
capability list.

### Future extraction (not current provider capabilities)

`kdot!`, `kdot_columns!`, `alloc_zeros`, and `copy_owned!` are implementation
helpers or shared ownership infrastructure today, not provider capabilities
exposed through the unified LA API. This first milestone does not make them
library-owned. Move them into a future extraction only when the core exposes
them through a stable boundary; until then, the provider consumes them and must
not weaken their alias-safety guarantees.

### `ksyrk!` mirroring contract

`ksyrk!`'s current public contract mirrors both triangles after the kernel
runs, even if an upstream arithmetic provider is lower-triangle-only. The
mirroring note is load-bearing: `la_syrk!` currently preserves the historical
full-matrix behavior, and some SDPX-specific call sites may still read the upper
triangle. The bundled provider must preserve the mirrored-both contract until
those call sites explicitly move to a lower-only contract. A provider that
advertises `:syrk` must therefore either mirror both triangles itself or the
provider layer must mirror the authoritative lower triangle before returning;
do not silently change the observable output shape.

### SDPX-specific: must stay in solver code

These are problem-structure code, not arithmetic libraries, and stay where
they live today:

- KKT/Schur assembly and sparse symbolic structure (`kkt.jl`, `schur.jl`,
  `kkt_sparse_backend.jl`, `mixed_precision_kkt.jl`, nullspace/chordal
  preprocessing)
- cone-native LP/SOC paths (`lp_solver.jl`, `soc_native.jl`,
  `soc_q3_kernels.jl`)
- planner, workspaces, solve-step orchestration, convergence, restarts,
  adaptive parameters, diagnostics, and certification
- legacy compatibility entry points (`sdp`, `findFeasible`, global setters)
  in `compat.jl`; those wrappers may stay for API compatibility, but they must
  not become a second backend-selection path
- kernel family files under `src/kernels/extended_precision_blas/` that provide
  the optional fixed-extended arithmetic implementation

### Direct `k*` call sites

`src/la_backend.jl` routes every `LegacyLABackend` `la_*` method through the
provider payload (`_sdpx_legacy_la_call`) and the provider-owned
`LegacyLACholeskyFactor` solve path. The architecture gate in
`cluster-probes/v041-legacy-provider/legacy_provider_gate.jl` proves this by
parsing `src/la_backend.jl` and rejecting legacy dispatch bodies that contain
direct `k*` calls or lack the provider-call marker.

## Bundled Provider Contract

The provider is discoverable, included, and instantiable through the unified
API. The probe gate asserts all of:

1. `pathof(SDPX)` resolves to the candidate tree (the PBS runner enforces this
   by source hash before Julia runs).
2. `src/la_backends/legacy.jl` is included from `src/SDPX.jl`.
3. `_provider_marker_sources` finds the provider recursively under `src/` in
   deterministic order.
4. `legacy_la_provider_identity` returns `:sdpx_legacy_la`;
   `legacy_la_provider_capabilities` returns `SDPX_LEGACY_LA_CAPABILITIES`;
   `legacy_la_provider_ownership` returns the arithmetic-specific contract
   (`:owned_mutable_scalars` for BigFloat, `:immutable_scalars` for bitstypes,
   `:legacy` for the remaining compatibility fallback).
5. `plan_la_backend(BigFloat; requested=:legacy)` selects `:legacy` with
   provider `:sdpx_legacy_la` and the advertised capabilities/ownership;
   `instantiate_la_backend` returns a `LegacyLABackend` whose provider payload
   is `SDPXLegacyLAProvider`.
6. `LegacyLABackend` `la_*` methods route through the provider and no longer
   call `k*` directly.
7. Executed diagnostics report the provider (`la_provider`,
   `la_executed_provider`, `la_ownership`, `la_executed_ownership`) after a
   real solve.

If the provider source disappears, the gate records
`PROVIDER_GATE=PROVIDER_NOT_FOUND` and fails closed. The gate is intentionally
a presence/architecture check, not a full solver suite, and it is placed in the
legacy tree so it can be reviewed and run without touching solver code.

## Phased Deletion Checklist

Do not delete compatibility `k*` wrappers in isolation. The compatibility
wrappers may remain as a documented, frozen surface until every caller has
moved and cluster differential gates pass. Delete each wrapper only after:

- [ ] the bundled `SDPXLegacyLA` provider exists and is included;
- [ ] `src/la_backend.jl` `LegacyLABackend` methods call the provider, not `k*`
      directly;
- [ ] `cluster-probes/v041-legacy-provider/legacy_provider_gate.jl` reports
      `PROVIDER_GATE=OK` and all audit rows show `direct=false` for the legacy
      dispatcher surface;
- [ ] `test/la_backend_regressions.jl` and `test/executed_diagnostics.jl`
      still pass, including planned-vs-executed LA backend identity;
- [ ] the cluster runner
      `cluster-probes/v041-legacy-provider/legacy_provider_regression.pbs`
      completes on a known-good pinned node (see node exclusions below) with
      `status=PASSED` and no `FAILED` marker;
- [ ] the small dense BigFloat standard-vs-legacy identity smoke passes on the
      cluster (`LEGACY_PROVIDER_AB full_solve_standard=ok` and
      `full_solve_legacy=ok`);
- [ ] `git diff --check`, `bash -n`, and `Meta.parseall` static checks are
      clean.

After those gates pass, delete a compatibility `k*` wrapper only when `rg`
shows no callers outside the provider, and record the deletion in the same
migration change. Do not delete wrappers merely because the provider landed.

## Diagnostics Contract

The solver already records executed backend state through `_record_la_execution!`
and diagnostics. The provider must not bypass that record. In particular:

- `planned_la_backend` and `executed_la_backend` must both equal `:legacy` for
  an explicit legacy request.
- `planned_la_fallback_reason` and `runtime_la_fallback_reason` must distinguish
  intentional legacy selection (`:requested_legacy`) from an actual runtime
  fallback. Any runtime reason other than `:none` (for migrated routes) or
  `:requested_legacy` (for explicit legacy) is a hard gate failure.
- `la_provider`/`la_executed_provider` must expose the bundled provider symbol,
  `:sdpx_legacy_la`, not `:legacy_kernels`.
- `la_ownership`/`la_executed_ownership` must match the provider's precise
  contract. For BigFloat the legacy route reports `:owned_mutable_scalars`;
  the probe defaults to that symbol and reads `SDPX_LEGACY_OWNERSHIP` so a
  different arithmetic target can override it without editing the probe.

## Node Exclusions for the Cluster Runner

The cluster runner is intentionally not submitted by this bundle. When run by
an operator, pin it to a known-good idle node. Do not submit to the known-bad
nodes from the cluster probe history: `node70`, `node71`, `node72`, `node134`,
`node135`, `node187`, and the already-down set (`node2`, `node20`, `node35`,
`node44`, `node51`, `node65`-`node69`, `node89`, `node112`, `node121`,
`node123`, `node124`, `node126`, `node130`, `node133`, `node136`, `node139`,
`node148`, `node152`, `node169`, `node174`, `node197`). Refresh the list from
`pbsnodes -a` and a 1-core probe before choosing `NODE_NAME`.

## Static Checks Only

Local verification for this bundle is static-only: `bash -n`, Julia
`Meta.parseall` with recursive `:error`/`:incomplete` rejection, and
`git diff --check`. No Julia solve, SSH, qsub, or other cluster access is part
of this task.
