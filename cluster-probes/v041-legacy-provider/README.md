# v041-legacy-provider cluster probe

Cluster-only validation for the bundled `SDPXLegacyLA` provider and the legacy
LA diagnostics contract. The probe lives in the legacy tree so it can be
reviewed and run independently of solver/core numerical changes.

## Files

- `legacy_provider_regression.pbs` - focused regression runner. `ppn=5`,
  Julia/solver threads 4, BLAS/OMP/MKL/BLIS threads 1. It verifies node
  identity, candidate source hash, runtime thread contract, then runs
  `test/la_backend_regressions.jl`, `test/executed_diagnostics.jl`,
  `legacy_provider_gate.jl`, and `bigfloat_identity_smoke.jl`.
- `legacy_provider_gate.jl` - architecture gate. Checks that the provider
  source exists anywhere under `src/` (recursively, deterministic order) and
  is included from `src/SDPX.jl`, that the unified planner selects it with the
  advertised identity/capabilities/ownership, and that `LegacyLABackend` `la_*`
  dispatch methods in `src/la_backend.jl` contain the provider-call marker
  `_sdpx_legacy_la_call` and no direct `k*` kernel calls. Reports
  `PROVIDER_GATE=OK` only when all checks pass. Standard/compatibility methods
  are ignored.
- `bigfloat_identity_smoke.jl` - small dense BigFloat standard-vs-legacy
  identity smoke: same owned inputs, both backends, real public solve,
  executed-backend identity, certificate, and objective agreement. It is not a
  benchmark and it does not duplicate the full 8700-test suite.
- `static_check.sh` - local static checks only: shell syntax, Julia
  `Meta.parseall` with recursive `:error`/`:incomplete` rejection, required
  marker checks, and `git diff --check`.

## Provider status

The bundled provider is present in the candidate tree:

- source: `src/la_backends/legacy.jl`, included from `src/SDPX.jl`;
- payload: `SDPXLegacyLAProvider` with identity `:sdpx_legacy_la`;
- BigFloat ownership: `:owned_mutable_scalars` (the provider's precise
  contract for the dense BigFloat smoke);
- capabilities: `:dot`, `:norminf`, `:mul`, `:mul_owned`, `:syrk`, `:chol`,
  `:cholesky_factor!`, `:solve`, `:cholsolve_owned`, `:trsm`,
  `:trsv_lower`, `:trsv_transpose`, `:axpby`, `:axpby_owned`, matching
  `SDPX_LEGACY_LA_CAPABILITIES`;
- provider-owned factor: `LegacyLACholeskyFactor`;
- provider-call marker: `_sdpx_legacy_la_call`.

The gate fails closed with `PROVIDER_GATE=PROVIDER_NOT_FOUND` if the provider
source disappears, and the PBS runner writes no `SUCCESS` until every contract
check passes.

## Environment variables

| Variable | Meaning |
|---|---|
| `NODE_NAME` | expected pinned compute node hostname |
| `CANDIDATE_SOURCE` | immutable SDPX candidate source root |
| `CANDIDATE_SOURCE_SHA256` | expected SDPX source-tree SHA-256 |
| `CANDIDATE_ENV` | pre-instantiated Julia environment with SDPX and test extras (`MultiFloats`, `Test`) |
| `SDPX_SITE_ENV` | site environment script exporting `JULIA_BIN` |
| `SDPX_DEPOT_PATH` | shared offline Julia depot |
| `CAMPAIGN_ID` | campaign identifier recorded in `environment.txt` |
| `OUTPUT_ROOT` | fresh result directory |
| `SDPX_LEGACY_PROVIDER_SYMBOL` | expected provider symbol; default `sdpx_legacy_la` |
| `SDPX_LEGACY_PROVIDER_CALL_MARKER` | provider-call marker that legacy dispatchers must contain; default `_sdpx_legacy_la_call` |
| `SDPX_LEGACY_OWNERSHIP` | expected BigFloat ownership symbol; default `owned_mutable_scalars` |

The environment must be prepared on a login node (compute nodes have no
package network access). Never submit from this bundle; run
`static_check.sh` locally first, then have an operator prepare the env, choose
a known-good idle node, and submit.

## Runtime contract

The runner fails closed unless `PBS_NP == 5`, Julia threads == 4, solver
threads == 4, BLAS/OMP/MKL/BLIS == 1, `pathof(SDPX)` matches the candidate,
the source-tree SHA-256 matches, `PROVIDER_GATE=OK`, and both BigFloat full
solves report their executed backend and pass. The `SUCCESS` marker is written
only after all checks pass.

Use `ppn=5`: this cluster's Gridview scheduler does not start `ppn=3` or
`ppn=4` jobs (`ppn>=5` is the empirical rule).

## Node exclusions

Pin `NODE_NAME` to a known-good idle node. Do not use the probed-bad nodes:
`node70`, `node71`, `node72`, `node134`, `node135`, `node187`, or the
already-down set (`node2`, `node20`, `node35`, `node44`, `node51`,
`node65`-`node69`, `node89`, `node112`, `node121`, `node123`, `node124`,
`node126`, `node130`, `node133`, `node136`, `node139`, `node148`, `node152`,
`node169`, `node174`, `node197`). Refresh the list with `pbsnodes -a` and a
1-core probe before choosing a node.
