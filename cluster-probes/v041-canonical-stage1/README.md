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
- `submit_parallel.sh` - dry-run by default; submits the focused job and the
  full job concurrently to two caller-provided distinct healthy idle nodes.
- `aggregate_parallel.py` - correctness-only gate over the two result roots;
  writes deterministic `summary.json` and concise `summary.md`.
- `static_check.sh` - local shell/Python static checks only.

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

The focused matrix is exactly the seven files above; the full matrix is
exactly one `Pkg.test("SDPX"; coverage=false)`.  They are intentionally not
duplicated into two file lists.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `NODE_NAME` | focused, full | expected `hostname` of the pinned compute node |
| `CANDIDATE_SOURCE` | focused, full | immutable candidate source root |
| `CANDIDATE_SOURCE_SHA256` | focused, full | expected source-tree SHA-256 (see command below) |
| `CANDIDATE_ENV` | focused, full | pre-instantiated independent Julia environment (`Project.toml` + `Manifest.toml`) |
| `SDPX_SITE_ENV` | bootstrap, focused, full | site environment script exporting `JULIA_BIN` |
| `SDPX_DEPOT_PATH` | bootstrap, focused, full | shared offline Julia depot |
| `CAMPAIGN_ID` | focused, full, submit, aggregate | shared campaign identifier; recorded in `environment.txt` and the campaign manifest |
| `OUTPUT_ROOT` | focused | fresh result directory (must not exist) |
| `FULL_OUTPUT_ROOT` | full | fresh result directory, distinct from `OUTPUT_ROOT` |
| `BOOTSTRAP_ENV` | bootstrap | environment directory to instantiate/precompile |
| `BOOTSTRAP_LOG` | bootstrap (optional) | bootstrap log path; default `$BOOTSTRAP_ENV/bootstrap.log` |
| `CAMPAIGN_ROOT` | submit | existing campaign root; result roots and the manifest may live beneath it |
| `CAMPAIGN_MANIFEST` | submit (optional) | manifest path; default `$CAMPAIGN_ROOT/$CAMPAIGN_ID.conf` |

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
   `CAMPAIGN_ID`, `NODE_NAME`, `CANDIDATE_SOURCE`,
   `CANDIDATE_SOURCE_SHA256`, `CANDIDATE_ENV`, `SDPX_SITE_ENV`,
   `SDPX_DEPOT_PATH`, `OUTPUT_ROOT`, and pin the job to the node, then run
   `qsub focused.pbs`.

3. Submit the full job with the same `CAMPAIGN_ID` and identity variables,
   `FULL_OUTPUT_ROOT` set to a fresh directory distinct from `OUTPUT_ROOT`,
   then run `qsub full.pbs`.

For the streamlined path, use the parallel submit helper instead of steps 2-3:

```bash
export CAMPAIGN_ID=...
export CAMPAIGN_ROOT=...
export CANDIDATE_SOURCE=...
export CANDIDATE_SOURCE_SHA256=...
export CANDIDATE_ENV=...
export SDPX_SITE_ENV=...
export SDPX_DEPOT_PATH=...
export OUTPUT_ROOT="$CAMPAIGN_ROOT/focused"
export FULL_OUTPUT_ROOT="$CAMPAIGN_ROOT/full"
bash cluster-probes/v041-canonical-stage1/submit_parallel.sh \
  FOCUSED_NODE FULL_NODE --submit
```

`submit_parallel.sh` requires two distinct node names and fails closed on
missing/equal nodes or required variables.  The caller must provide nodes it
has already verified as healthy and idle; the helper does not query PBS node
state, does not select or migrate nodes, does not cancel jobs, and does not
touch any other job.  Both jobs receive the same `CAMPAIGN_ID`, candidate
path/realpath/tree hash, environment, site env, depot, and thread contract;
only the pinned node and result root differ.  Result roots may live inside
`CAMPAIGN_ROOT` (this is the normal layout) but never inside the candidate
source, candidate env, or depot.  Dry-run prints both `qsub` commands and the
manifest; `--submit` writes `CAMPAIGN_MANIFEST` only after both submissions
succeed, and never auto-cancels a successfully submitted first job if the
second fails.

## Parallel aggregation

After both outputs are pulled (or with the manifest and result roots on the
login result root), run:

```bash
python3 cluster-probes/v041-canonical-stage1/aggregate_parallel.py \
  --manifest "$CAMPAIGN_ROOT/$CAMPAIGN_ID.conf" \
  --summary-root "$CAMPAIGN_ROOT/summary" FOCUSED_ROOT FULL_ROOT
```

`aggregate_parallel.py` writes one row per job into deterministic
`summary.json` plus a concise `summary.md`, and exits 1 unless every job has a
non-empty `SUCCESS`, `environment.txt`, `test.log`, and `process.time.txt`,
and `environment.txt` matches the manifest on campaign id, candidate
source/realpath/env/tree hash, expected node/job id, and thread/ppn contract.
The two rows must also agree on those shared identity fields.  `FAILED` is
reported as `TEST_FAILURE`; a missing `SUCCESS`/`FAILED` pair with no timing
evidence is `MISSING_REPORT`; only a `/usr/bin/time` `Command terminated by
signal` line converts that to `TIMING_NOISE`.  The gate is correctness-only:
no ratios, CVs, medians, or timing comparisons are made.  The summary root
gets `AGGREGATE_GATE_PASS` or `AGGREGATE_GATE_FAILED`, and
`aggregate_parallel.py --self-test` covers PASS, deterministic JSON, test
failure, timing noise, missing report, hash/job-id mismatch, and missing test
log/environment in a synthetic temp tree.

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

Local static verification is intentionally limited to `static_check.sh`
(`bash -n` on the shell/PBS files, Python syntax and synthetic aggregation
self-tests) plus `git diff --check`.  No Julia, Pkg, SSH, or qsub command is
run locally.
