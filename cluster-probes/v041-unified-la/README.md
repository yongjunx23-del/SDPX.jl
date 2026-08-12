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
- `kernel_ab.pbs` - same runtime contract; runs the generic-vs-MFLA
  microbenchmark (`kernel_ab.jl`) with warmup 2 and 10 timed repeats for
  GEMM, SYRK, TRSM, and factor + repeated solve.  Inputs are generated once
  from a fixed seed and every repetition resets each destination from the
  immutable base copies, so both arms see identical input state.  SYRK uses
  an explicit lower-triangle contract (`syrk_contract=lower_triangle`) and
  residuals are computed only on the authoritative triangle.  The output
  records per-operation median seconds, allocations, true BigFloat-512
  residuals, finiteness, process RSS, and CPU utilization.
- `solver_ab.pbs` - same runtime contract; runs the small dense-KKT A/B
  (`solver_ab.jl`) with the same immutable SPD system for both arms.  The
  KKT direction verification always runs with real residuals.  The complete
  SDPX solve A/B is capability-gated: when the core exposes the expert
  `SolverOptions.linear_algebra_backend` option, the same small dense SDP is
  solved through the public `solve` frontend with `:standard` and
  `:multifloat`, and status, objective, gap, primal/dual residual, the
  original-coordinate `SDPX.result_certificate(prob, result, opts)` result,
  and iterations are validated from real solve results.  Until then the
  probe prints `full_solve=SKIPPED` and the PBS layer fails closed; no
  certificate is fabricated.
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

The kernel and solver A/B probes use explicit `:standard` and `:multifloat`
arms.  The automatic plan selection is recorded (`auto_selected`,
`auto_provider`) but is never used to choose an arm.  The MFLA arm always
uses the explicit provider: through the instantiated
`MultiFloatLABackend` when the planner accepts the pinned provider
(`mfla_route=sdpx_provider`), or through the pinned upstream MFLA kernels
directly while the planner API has not landed
(`mfla_route=direct_upstream`).

Each A/B PBS job asserts the actual `pathof(SDPX)`, the actual MFLA root and
pinned commit, the candidate source-tree SHA-256, and Julia/BLAS thread
counts, then records `rss_kib`, `user_cpu_seconds`, `system_cpu_seconds`,
`wall_seconds`, and `cpu_utilization` from `/usr/bin/time -v` in
`environment.txt`.

## Submitting both A/B jobs in parallel

`kernel_ab.pbs` and `solver_ab.pbs` are independent jobs and may run on two
different healthy idle nodes at the same time.  Each job writes only to its
own `OUTPUT_ROOT`; the candidate source, MFLA checkout, candidate
environment, and depot are read-only during a run.  Select two healthy idle
nodes (no other jobs, `nusers=0`, load average below 2) and submit:

```bash
qsub -v NODE_NAME=node_a,OUTPUT_ROOT=.../kernel_ab, \
  CANDIDATE_SOURCE=...,CANDIDATE_SOURCE_SHA256=...,CANDIDATE_ENV=..., \
  MFLA_SOURCE=...,MFLA_COMMIT=e5eccd7a56482522acd5690800bf7438149997f5, \
  SDPX_SITE_ENV=...,SDPX_DEPOT_PATH=...,CAMPAIGN_ID=... kernel_ab.pbs
qsub -v NODE_NAME=node_b,OUTPUT_ROOT=.../solver_ab, \
  CANDIDATE_SOURCE=...,CANDIDATE_SOURCE_SHA256=...,CANDIDATE_ENV=..., \
  MFLA_SOURCE=...,MFLA_COMMIT=e5eccd7a56482522acd5690800bf7438149997f5, \
  SDPX_SITE_ENV=...,SDPX_DEPOT_PATH=...,CAMPAIGN_ID=... solver_ab.pbs
```

Do not run both memory-bandwidth-heavy jobs on the same node.  Do not submit
either job before the corresponding candidate source hash and MFLA checkout
have been verified.
