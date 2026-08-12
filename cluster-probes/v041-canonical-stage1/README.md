# v041-canonical-stage1 cluster probes

Cluster-only Stage-1 validation for the canonical midend and the production
`AutoPlanner` planning boundary.  The PBS jobs run
offline on compute nodes; package setup happens once on a login node.  Nothing
here submits jobs or runs Julia locally.  Every path, the expected node, and
the expected candidate hash are supplied through environment variables; no
node name or hash value is hardcoded.

## Files

- `bootstrap_login_env.sh` - login-node only: `Pkg.instantiate()` and
  `Pkg.precompile()` for the independent Julia environment.
- `focused.pbs` - `ppn=5`, Julia/solver `4` threads, BLAS/OMP/MKL `1`;
  validates node, candidate identity and hash, then runs the compact
  `canonicalize(SDPProblem)` gate over the focused manifest below.  SOC and
  frontend-option regressions stay in the full-suite boundary for this
  milestone.
- `full.pbs` - same resource contract; `Pkg.test("SDPX"; coverage=false)`
  into a separate result directory.

## Focused manifest

`focused.pbs` includes the following test files in this exact order; keep this
list and the `files` array in `focused.pbs` in sync:

```text
canonical_conic_problem.jl
problem_features.jl
auto_planner.jl
pipeline.jl
v041_architecture_regressions.jl
executed_diagnostics.jl
public_api.jl
```

Each file is included directly (`FOCUSED_BEGIN`/`FOCUSED_END` markers), so a
per-file failure is visible in `test.log` and the job fails closed only after
recording every failing file.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `NODE_NAME` | focused, full | expected `hostname` of the pinned compute node |
| `CANDIDATE_SOURCE` | focused, full | immutable candidate source root |
| `CANDIDATE_SOURCE_SHA256` | focused, full | expected source-tree SHA-256 (see command below) |
| `CANDIDATE_ENV` | focused, full | pre-instantiated independent Julia environment (`Project.toml` + `Manifest.toml`) |
| `SDPX_SITE_ENV` | bootstrap, focused, full | site environment script exporting `JULIA_BIN` |
| `SDPX_DEPOT_PATH` | bootstrap, focused, full | shared offline Julia depot |
| `OUTPUT_ROOT` | focused | fresh result directory (must not exist) |
| `FULL_OUTPUT_ROOT` | full | fresh result directory, distinct from `OUTPUT_ROOT` |
| `BOOTSTRAP_ENV` | bootstrap | environment directory to instantiate/precompile |
| `BOOTSTRAP_LOG` | bootstrap (optional) | bootstrap log path; default `$BOOTSTRAP_ENV/bootstrap.log` |

The PBS scripts compute the candidate source-tree hash with this exact
command, so compute the expected value the same way.  The whole subshell pins
`LC_ALL=C` so `find`/`sort` use byte collation regardless of the inherited PBS
locale (an inherited `en_US.utf8` locale changes the aggregate hash):

```bash
(cd "$CANDIDATE_SOURCE" && export LC_ALL=C && find . -type f -not -path './.git' -not -path './.git/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
```

## Order

1. Prepare `CANDIDATE_ENV` on a login node (dev the candidate into it and add
   the test dependencies), then bootstrap it:

   ```bash
   export BOOTSTRAP_ENV=...
   export SDPX_SITE_ENV=...
   export SDPX_DEPOT_PATH=...
   bash cluster-probes/v041-canonical-stage1/bootstrap_login_env.sh
   ```

   The bootstrap refuses to run inside a PBS job.  Compute nodes have no
   package network access, so instantiate/precompile must happen only on a
   login node.

2. Submit the focused job by hand from the cluster campaign root.  Set
   `NODE_NAME`, `CANDIDATE_SOURCE`, `CANDIDATE_SOURCE_SHA256`,
   `CANDIDATE_ENV`, `SDPX_SITE_ENV`, `SDPX_DEPOT_PATH`, `OUTPUT_ROOT`, and
   pin the job to the node, then run `qsub focused.pbs`.

3. Submit the full job with `FULL_OUTPUT_ROOT` set to a fresh directory
   distinct from `OUTPUT_ROOT`, then run `qsub full.pbs`.

## Outputs and SUCCESS semantics

Each PBS job writes into its own result root:

- `environment.txt` - PBS job id, hostname/node check, Julia binary/version,
  candidate path and realpath, tree hash and expected hash, commit/archive
  metadata when present, Julia/solver/BLAS threads, PBS `ppn`, CPU info, and
  start/finish timestamps.
- `process.time.txt` - `/usr/bin/time -v` output, including maximum resident
  set size.
- `test.log` - focused markers (`FOCUSED_BEGIN`/`FOCUSED_END`/
  `FOCUSED_FAILED`) or full `Pkg.test` output.
- `SUCCESS` - written only after every focused test (or the full suite)
  passes.
- `FAILED` - written on any failure or abort.

Local static verification is intentionally limited to shell syntax checks:
`bash -n` on the shell/PBS files and `git diff --check`.  No Julia, Pkg, SSH,
or qsub command is run locally.
