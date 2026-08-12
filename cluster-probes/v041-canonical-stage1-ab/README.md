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
| `stage1_ab.pbs` | one PBS job: identity checks, baseline arm, candidate arm, analysis gate; `SCREEN_MODE=true` selects the lean single-family screen |
| `analyze_ab.py` | machine-checkable cross-arm gate; writes `ab_report.csv` and `ab_provenance.txt`; `--screen` relaxes CV to the development band; `--self-test` runs synthetic CSV scenarios |
| `aggregate_ab.py` | multi-node aggregate gate over completed A/B result roots; reports per-node family ratios and fails unless all settings/identity gates and aggregated ratios pass |
| `submit_stage1_ab.sh` | dry-run by default; `--submit` pins the job to `NODE_NAME` |
| `submit_screen_parallel.sh` | dry-run by default; submits three single-family screen jobs on three explicit nodes with shared identity variables |
| `screen_summarize.py` | concise gate over three disjoint screen result roots; writes `screen_summary.csv`/`screen_summary.txt`; `--self-test` runs synthetic scenarios |
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
| `AB_RUNNER_ROOT` | local probe/runner directory containing `analyze_ab.py` (shared filesystem path reachable from the compute node) |

Optional job variables with stable defaults:

| Variable | Default |
|---|---|
| `ARITHMETIC` | `float64,float64x4` |
| `CASE_FILTER` | `lp_row_scaling,socp_many_tiny,sdp_hilbert` |
| `TIMING_BATCH_SIZE` | `10` (timed runs per batch) |
| `TIMED_BATCHES` | `3` (batch timing samples) |
| `REPETITIONS` | derived `1 + 10*3 = 31` (rep 1 warmup; reps 2-31 timed); explicit values must equal `1 + TIMING_BATCH_SIZE*TIMED_BATCHES` or the job fails closed |
| `WARMUP` | `true` |
| `TIME_LIMIT` | `900` |
| `MAX_ITERATIONS` | `300` |
| `ARM_ORDER` | `baseline_first` (default) or `candidate_first`; forwarded by the submit helper as a single token |

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

Each arm runs `--warmup=true` with the derived `REPETITIONS`.  The
harness-level warmup covers the first campaign row per arithmetic, but
case-specific JIT can still land in each case's first repetition, so
repetition 1 is defined as that case's warmup and is discarded from timing.
Timed repetitions `2..N` are grouped into `TIMED_BATCHES` consecutive batches
of `TIMING_BATCH_SIZE` runs; each batch sum is one timing sample.  The
analyzer uses the batch medians and CVs.  Repetition 1 is still fully checked
for correctness, config, status, certificate, route, endpoint, and workspace
equality; only timing excludes it.  Explicit `REPETITIONS` must equal
`1 + TIMING_BATCH_SIZE*TIMED_BATCHES`; the PBS job fails closed otherwise.

## Case matrix

With `CASE_FILTER=lp_row_scaling,socp_many_tiny,sdp_hilbert`:

- Float64: `lp_row_scaling` (n16 decades6), `socp_many_tiny`, `sdp_hilbert` -> 3 cases x 31 reps (timed batches 3 x 10)
- Float64x4: the above plus `lp_row_scaling` (n16 decades16) -> 4 cases x 31 reps (timed batches 3 x 10)

217 rows per arm total (7 cases x 31 repetitions).  Every row is generated offline from
`bench/public_conic_suite/generators/SDPXPathologicalBenchmarks.jl`; input
hashes (`input_sha256`) must match exactly across arms.

## Outputs

`RESULT_ROOT/`:

- `environment.txt` - node, job, paths, expected/actual hashes and commits, threads, Julia version, CPU, start/finish
- `arms.conf` - machine-readable arm/config record consumed by `analyze_ab.py`
- `AB_RUNNER_ROOT`, its realpath, and `analyze_ab.py` SHA-256 recorded in `environment.txt` and `arms.conf`
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
- row matrices are identical, each case group has exactly `1..N`
  repetitions, and `input_sha256` and all config/thread/node fields match per
  row across arms;
- correctness, status, certificate, route, endpoint, iteration, and workspace
  equality cover all N rows, including the discarded warmup row;
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
- per `(arithmetic, family)` candidate median batch sum is at most 10% above
  baseline (`ratio <= 1.10`) and both arm CVs are below 5%; medians and CVs
  use only the `TIMED_BATCHES` batch sums from repetitions 2..N.

The analysis gate never accepts a run where the runner itself failed
certificate, status, objective, resource, identity, or route gates.

PBS copies the submitted script to the mom spool, so `$0` cannot locate the
analyzer; `stage1_ab.pbs` therefore requires `AB_RUNNER_ROOT` and runs
`$AB_RUNNER_ROOT/analyze_ab.py`.  The submit helper forwards it and validates
that the file exists.  `environment.txt` and `arms.conf` record its realpath
and SHA-256 for provenance.

`analyze_ab.py --self-test` verifies the analyzer itself against synthetic
CSVs without Julia: a clean PASS, a PASS where full-tree and subset hashes
differ, and a FAIL exercising input/config/certificate/timing gates.

## Multi-node aggregate

After multiple completed result roots exist, run:

```bash
python3 cluster-probes/v041-canonical-stage1-ab/aggregate_ab.py RESULT_ROOT_1 RESULT_ROOT_2 ...
```

Each root must contain `arms.conf`, `analyze.log`, `ab_report.csv`, and
`baseline/`/`candidate/` result CSVs, plus root `SUCCESS` (root `FAILED` is
rejected).  The aggregate tool requires every root's local analyzer gate to
have passed via an explicit, non-truncated `AB_GATE_PASS` line in
`analyze.log`, and requires all roots to agree on commits, tree hashes, archive
hashes, `ab_runner_sha256`, arithmetic, case filter, batch/repetition settings,
resource/thread config, and runner source.  It reports each node's per-family
ratio and worst within-arm CV (`max(baseline_cv, candidate_cv)`), plus the
aggregate median/min/max ratio and node count; every family ratio must be
<= 1.10 with worst within-arm CV < 0.05, and the aggregated median ratio must
be <= 1.10.  `aggregate_ab.py --self-test` covers a clean pass, a missing pass
marker, and a provenance mismatch.  It is Python 3.6 compatible and performs no
Julia/Pkg/SSH/qsub work.

### Order-balanced screening

`aggregate_ab.py --ordered` accepts two completed result roots per node, one
`baseline_first` and one `candidate_first` (paired by `hostname`/`expected_node`
from `environment.txt`).  Real timing-only roots may have root `FAILED` and no
root `SUCCESS` because the analyzer exits 1; this is accepted only when both
arm `SUCCESS` markers exist, no arm `FAILED` exists, `analyze.log` has no
non-timing FAIL lines, and `max_endpoint_norm=0` and `max_iteration_delta=0`
each appear exactly once.  Both root markers absent/present or a root `FAILED`
with non-timing failures are rejected.  Each node's two order ratios are
combined geometrically (`sqrt(r_baseline_first * r_candidate_first)`); the
combined ratio must be <= 1.10, the aggregate geometric median <= 1.10, and
the one-sided 95% log-space t upper bound (df=n-1) <= 1.10 for n>=3.
Within-arm batch CV is reported, warned at >= 0.05, and hard-fails at >= 0.20.
Roots that predate `arm_order` require the explicit
`--legacy-baseline-first` flag and are accepted only when fixed
analyzer/runner provenance is present and baseline artifacts precede candidate
artifacts; otherwise rerun both orders.

`aggregate_ab.py --self-test` also covers a three-host order-balanced PASS with
timing-only root `FAILED` roots and an order-balanced FAIL from a non-timing
root `FAILED`.

## Development parallel screen

The lean development-stage screen triages a candidate across all three
families in parallel: LP, SOCP, and SDP each run on its own healthy pinned
node, and each node's PBS job runs baseline and candidate sequentially inside
the same job (same-node paired A/B).  It is a screening tool, not a release
confirmation: it intentionally uses the analyzer's `--screen` CV policy and
requires no order-balanced repeats.

Submit with three explicit nodes and shared identity variables:

```bash
export BASELINE_SOURCE=... BASELINE_SOURCE_SHA256=... BASELINE_SOURCE_COMMIT=...
export BASELINE_ENV=... CANDIDATE_SOURCE=... CANDIDATE_SOURCE_SHA256=...
export CANDIDATE_SOURCE_COMMIT=... CANDIDATE_ENV=... RUNNER_SOURCE=...
export RUNNER_SOURCE_SHA256=... SDPX_SITE_ENV=... SDPX_DEPOT_PATH=...
export AB_RUNNER_ROOT=... RESULT_ROOT_PREFIX=/public/.../screen-20260812
bash cluster-probes/v041-canonical-stage1-ab/submit_screen_parallel.sh \
  NODE_LP NODE_SOCP NODE_SDP        # dry-run
bash cluster-probes/v041-canonical-stage1-ab/submit_screen_parallel.sh \
  NODE_LP NODE_SOCP NODE_SDP --submit
```

The helper requires three distinct nodes, derives
`RESULT_ROOT_PREFIX/lp`, `/socp`, and `/sdp`, and refuses to reuse any result
root.  Each job runs with `SCREEN_MODE=true`, `CASE_FILTER` limited to one of
`lp_row_scaling`, `socp_many_tiny`, `sdp_hilbert`, and the default screen
timing scheme warmup 1 + 3 batches x 5 (16 repetitions).  Float64 and
Float64x4 remain on the PBS-side default `float64,float64x4`; do not forward
comma-containing values through `-v`.

After all three jobs finish, summarize and gate the disjoint family roots:

```bash
python3 cluster-probes/v041-canonical-stage1-ab/screen_summarize.py \
  "$RESULT_ROOT_PREFIX/lp" "$RESULT_ROOT_PREFIX/socp" "$RESULT_ROOT_PREFIX/sdp"
```

`screen_summarize.py` requires each root's `screen_mode=true`, explicit
`AB_GATE_PASS`, baseline/candidate arm `SUCCESS`, no FAILED markers, identity
and thread settings identical across roots, exactly
`{lp, socp, sdp} x {float64, float64x4}` family coverage, ratio <= 1.10, and
`max_endpoint_norm=0`/`max_iteration_delta=0`.  Within-arm worst batch CV is
reported per family: < 5% passes, 5%-20% is a warning (does not fail the
screen), and >= 20% fails.  It writes `screen_summary.csv` and
`screen_summary.txt` (use `--output-dir` to place them elsewhere).

Result roots produced before `SCREEN_MODE` was recorded are rejected by
default.  The explicit `--legacy-screen-evidence` flag accepts them only when
all other immutable identity, 16-repetition timing, thread, dual-arm SUCCESS,
exact endpoint/iteration, and `AB_GATE_PASS` evidence is present; accepted
legacy roots are always reported with a provenance warning.
Pre-SCREEN_MODE roots lack `screen_mode=true`; `--legacy-screen-evidence`
accepts them only when all other screen gates still pass and records the
acceptance as a warning.  New screen roots must always carry
`screen_mode=true`.

### Development screen vs release confirmation

The screen is a first-pass parallel triage; its CV band is deliberately wider
and it does not control for arm order.  A candidate only graduates to release
confirmation after the screen is clean, and release confirmation must then use
the strict default path:

- same pinned node, sequential baseline/candidate pair per job;
- two orders per node (`baseline_first` and `candidate_first`), three or more
  nodes, combined geometrically;
- warmup 1 + 3 batches x 10 (31 repetitions), both within-arm CVs < 5%;
- `aggregate_ab.py --ordered` one-sided 95% upper bound <= 1.10.

Never claim a candidate as confirmed from a screen root alone; the screen only
decides whether the more expensive order-balanced confirmation is worth
running.

## Single-token CASE_FILTER

`submit_stage1_ab.sh` forwards `CASE_FILTER` only when it is a single token
(e.g. `CASE_FILTER=socp_many_tiny`) so SOCP-only runs are possible; a value
containing a comma, equals sign, or whitespace is rejected at submit time.
Comma-containing `ARITHMETIC` defaults remain on the PBS side and are not
forwarded.

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

For the parallel development screen, use `submit_screen_parallel.sh` as shown
above and summarize with `screen_summarize.py`.  `screen_summarize.py
--self-test` and the extended `analyze_ab.py --self-test`/`aggregate_ab.py
--self-test` cover the synthetic pass, CV warning band, hard CV failure,
ratio/endpoint/identity/thread/coverage failures, and order-balanced release
evidence without Julia.
