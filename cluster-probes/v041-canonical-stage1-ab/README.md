# v041-canonical-stage1-ab

Cluster-only unchanged-hot-path A/B for `SDPX-v041-blocklocal-auto`:

- baseline: rc1 `302dca8` (`Avoid same-dimension complementarity normalization divisions`)
- candidate: canonical Stage-1 `d6daf70` (`Introduce canonical conic semantic IR`)

`d6daf70` adds `src/midend/canonical_problem.jl`, `src/public_api.jl`, and
`test/canonical_conic_problem.jl` on top of `302dca8`; it does not edit the
existing solver hot path.  This probe verifies on one pinned node that the
candidate keeps the numerical trajectory, statuses, certificates, routes, and
per-category solve times of generated LP/SOCP/SDP cases unchanged.

## Policy

- Cluster-only: no Julia, Pkg, SSH, or qsub on a development machine.  Do not
  run `run_generated_pathological.jl`, `Pkg.test`, or any solver locally.
- The two arms run sequentially inside one PBS job on the same pinned node, in
  fixed order `baseline` then `candidate`.
- Both arms use the same runner code, campaign, case filter, arithmetic,
  tolerances, iteration/time limits, warmup/repetition scheme, resource class,
  and thread environment.  Only `--source` and the Julia project differ.
- This directory never commits anything; it is a runner/diagnostics artifact.

## Files

| File | Purpose |
|---|---|
| `stage1_ab.pbs` | one PBS job: identity checks, baseline arm, candidate arm, analysis gate |
| `analyze_ab.py` | machine-checkable cross-arm gate; writes `ab_report.csv` and `ab_provenance.txt`; `--self-test` runs synthetic CSV scenarios |
| `submit_stage1_ab.sh` | dry-run by default; `--submit` pins the job to `NODE_NAME` |
| `static_check.sh` | local bash/Python static checks only |

## Environment variables

Required at submit time:

| Variable | Meaning |
|---|---|
| `NODE_NAME` | healthy pinned compute node; must equal the job's `hostname` |
| `BASELINE_SOURCE` | immutable rc1 baseline checkout (`302dca8`) |
| `BASELINE_SOURCE_SHA256` | expected full-tree hash (command below); used only as the PBS immutable gate |
| `BASELINE_SOURCE_COMMIT` | expected baseline commit, `302dca8543f598e20e82ba3ea302cf26e10e4bc8` |
| `BASELINE_ENV` | pre-instantiated baseline Julia environment |
| `CANDIDATE_SOURCE` | immutable Stage-1 candidate checkout (`d6daf70`) |
| `CANDIDATE_SOURCE_SHA256` | expected candidate full-tree hash |
| `CANDIDATE_SOURCE_COMMIT` | expected candidate commit, `d6daf709ee6b064408165a999a76cc5b0414b2a3` |
| `CANDIDATE_ENV` | pre-instantiated candidate Julia environment |
| `RUNNER_SOURCE` | fixed `bench/public_conic_suite` tree used by both arms |
| `RUNNER_SOURCE_SHA256` | expected runner full-tree hash |
| `SDPX_SITE_ENV` | site env script exporting `JULIA_BIN` |
| `SDPX_DEPOT_PATH` | shared offline Julia depot |
| `RESULT_ROOT` | fresh result root outside both sources |

Optional job variables with stable defaults:

| Variable | Default |
|---|---|
| `ARITHMETIC` | `float64,float64x4` |
| `CASE_FILTER` | `lp_row_scaling,socp_many_tiny,sdp_hilbert` |
| `REPETITIONS` | `4` (rep 1 is per-case warmup; reps 2-4 are timed) |
| `WARMUP` | `true` |
| `TIME_LIMIT` | `900` |
| `MAX_ITERATIONS` | `300` |

## Immutable candidate layout

Each source must follow the repo's immutable layout:

```text
<parent>/<candidate-dir>/
<parent>/metadata/source_commit.txt
<parent>/metadata/archive_sha256.txt
```

The runner's `gate_identity` requires a non-empty archive SHA, and the PBS
fails early when either `metadata` file is missing.  Compute expected tree
hashes with the exact command used by the PBS (whole subshell pinned to
`LC_ALL=C`):

```bash
(cd "$SOURCE" && export LC_ALL=C && find . -type f -not -path './.git' -not -path './.git/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
```

`arms.conf` records these as `baseline_tree_sha256`, `candidate_tree_sha256`,
and `runner_tree_sha256`.  Do not confuse them with the runner CSV
`source_sha256`: the runner writes a *subset* hash over
`src/bench/Project/Manifest` using path+NUL+bytes, while the PBS gate uses a
*full-tree* `sha256sum` aggregation.  The analyzer never compares the two; it
only requires each arm's subset hash to be non-empty and identical across that
arm's rows, then records both subset hashes in `ab_provenance.txt` and the
analysis output.  Commit, archive, `pathof(SDPX)`, and full-tree hashes remain
the independent identity guarantees.

## Resource contract

PBS `ppn=5`, Julia `-t 4`, solver threads `4`, BLAS/OMP/MKL/BLIS threads `1`.
This is the existing `regular` resource class enforced by
`run_generated_pathological.jl`; a 1-thread variant would require a new
resource class and is intentionally out of scope for this runner-only probe.
The analyzer verifies the per-row thread/ppn fields and that both arms ran on
the same host in the same PBS job.

## Warmup and timed repetitions

Each arm runs `--warmup=true --repetitions=4`.  The harness-level warmup covers
the first campaign row per arithmetic, but case-specific JIT can still land in
each case's first repetition, so repetition 1 is defined as that case's
warmup and is discarded from timing.  Repetitions 2-4 are the three timed rows;
the analyzer uses their median and CV.  Repetition 1 is still fully checked
for correctness, config, status, certificate, route, endpoint, and workspace
equality; only timing excludes it.

## Case matrix

With `CASE_FILTER=lp_row_scaling,socp_many_tiny,sdp_hilbert`:

- Float64: `lp_row_scaling` (n16 decades6), `socp_many_tiny`, `sdp_hilbert` -> 3 cases x 4 reps (timed 2-4)
- Float64x4: the above plus `lp_row_scaling` (n16 decades16) -> 4 cases x 4 reps (timed 2-4)

28 rows per arm total (7 case x 4 repetitions).  Every row is generated offline from
`bench/public_conic_suite/generators/SDPXPathologicalBenchmarks.jl`; input
hashes (`input_sha256`) must match exactly across arms.

## Outputs

`RESULT_ROOT/`:

- `environment.txt` - node, job, paths, expected/actual hashes and commits, threads, Julia version, CPU, start/finish
- `arms.conf` - machine-readable arm/config record consumed by `analyze_ab.py`
- `baseline/` and `candidate/` - each with `results.csv`, `failures.csv`, `benchmark_manifest.toml`, `report.md`, `runner.log`, and `SUCCESS`/`FAILED`
- `baseline.process.time.txt`, `candidate.process.time.txt` - `/usr/bin/time -v`, including peak RSS
- `baseline.runner.log`, `candidate.runner.log`
- `analyze.log` - analyzer output
- `ab_report.csv` - per-category timing medians, ratios, CVs, gate
- `ab_provenance.txt` - subset `source_sha256` per arm plus tree hashes, commits, archive hashes, and source paths
- `SUCCESS` (root) or `FAILED` (root)

After both arms finish, the PBS also appends
`baseline_source_sha256_subset` and `candidate_source_sha256_subset` to
`environment.txt` from the unique per-arm CSV value.

## Analysis gates

`analyze_ab.py` fails unless:

- every runner row passes `gate_pass` in both arms and both arm `SUCCESS`
  markers exist;
- row matrices are identical, each case group has exactly `1..4`
  repetitions, and `input_sha256` and all config/thread/node fields match per
  row across arms;
- correctness, status, certificate, route, endpoint, and workspace equality
  cover all four rows, including the discarded warmup row;
- status, certificate, route, and fallback fields match exactly per row
  (`normalized_status`, `certificate_valid`, `certificate_type`,
  `planned_backend`, `executed_backend`, `fallback`, `fallback_reason`, ...);
- `workspace_bytes` and iteration counters match exactly per row (Stage-1 does
  not change workspace); peak RSS is reported only and has no threshold here;
- each arm's runner `source_sha256` subset hash is non-empty and identical
  across that arm's rows; it is reported but never compared to the full-tree
  hash;
- objective/residual/certificate endpoints agree within arithmetic bounds
  (default Float64 `atol=1e-9 rtol=1e-5`, Float64x4 `atol=1e-22 rtol=1e-5`;
  override with `--endpoint-atol`/`--endpoint-rtol`);
- iteration counters are identical (`--no-strict-iterations` is a
  diagnosis-only escape hatch);
- per `(arithmetic, family)` candidate median `total_seconds` is at most 10%
  above baseline (`ratio <= 1.10`) and both arm CVs are below 5%, where medians
  and CVs use only repetitions 2-4.

The analysis gate never accepts a run where the runner itself failed
certificate, status, objective, resource, identity, or route gates.

`analyze_ab.py --self-test` verifies the analyzer itself against synthetic
CSVs without Julia: a clean PASS, a PASS where full-tree and subset hashes
differ, and a FAIL exercising input/config/certificate/timing gates.

## Static verification

```bash
bash cluster-probes/v041-canonical-stage1-ab/static_check.sh
git diff --check
```

No Julia is executed locally.  Submit with the helper only from the cluster
login node:

```bash
export NODE_NAME=... BASELINE_SOURCE=... # and all other variables above
bash cluster-probes/v041-canonical-stage1-ab/submit_stage1_ab.sh        # dry-run
bash cluster-probes/v041-canonical-stage1-ab/submit_stage1_ab.sh --submit
```
