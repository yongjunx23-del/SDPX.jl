# v041-unified-la cluster probes

Cluster-only integration gate for the optional
`SDPXMultiFloatLinearAlgebraExt` extension.

`MultiFloatLinearAlgebra.jl` is unregistered, so it is never added to
`test` extras/targets.  The integration environment is created once on a
login node with `Pkg.develop(path=...)` (or a pinned URL checkout) for both
the SDPX candidate and the pinned MFLA commit, then the compute node runs
the focused integration test file offline.

## Pinned upstream

```text
repo    https://github.com/yongjunx23-del/MultiFloatLinearAlgebra.jl
commit  e5eccd7a56482522acd5690800bf7438149997f5
uuid    642d9d30-8e28-45ca-9d81-256429ea358f
```

## Files

- `bootstrap_login_env.sh` - login-node only: instantiate and precompile the
  independent environment after the caller develops the SDPX candidate and
  the pinned MFLA checkout into it.
- `focused.pbs` - `ppn=5`, Julia/solver 4 threads, BLAS/OMP/MKL/BLIS 1;
  validates node, candidate identity, MFLA identity/commit, runtime contract,
  then includes `test/multifloat_linear_algebra_integration.jl`.
- `static_check.sh` - local shell/static checks only.

## Environment variables

| Variable | Meaning |
|---|---|
| `NODE_NAME` | expected pinned compute node hostname |
| `CANDIDATE_SOURCE` | immutable SDPX candidate source root |
| `CANDIDATE_SOURCE_SHA256` | expected SDPX source-tree SHA-256 |
| `CANDIDATE_ENV` | pre-instantiated independent Julia environment |
| `MFLA_SOURCE` | read-only pinned MultiFloatLinearAlgebra source root |
| `MFLA_COMMIT` | expected MFLA commit (`e5eccd7...`) |
| `SDPX_SITE_ENV` | site environment script exporting `JULIA_BIN` |
| `SDPX_DEPOT_PATH` | shared offline Julia depot |
| `CAMPAIGN_ID` | campaign identifier recorded in `environment.txt` |
| `OUTPUT_ROOT` | fresh result directory |

## Runtime contract

`focused.pbs` fails closed unless `PBS_NP == 5`, Julia threads == 4, solver
threads == 4, BLAS/OMP/MKL/BLIS == 1, `pathof(SDPX)` matches the candidate,
`pathof(MultiFloatLinearAlgebra)` is inside `MFLA_SOURCE`, the pinned commit
matches, and `plan_la_backend(Float64x4; requested=:multifloat)` selects the
provider.  The `SUCCESS` marker is written only after the integration test
file and all identity/marker checks pass.
