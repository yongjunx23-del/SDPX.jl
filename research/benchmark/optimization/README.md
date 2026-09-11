# Dependent benchmark profiling and optimization

This directory contains the long-horizon optimization path. It is deliberately
separate from the correctness E2E suite and never restores the obsolete
`hsd_allocation.jl --check` hard gate.

## Dependency graph

```text
main push / manual dispatch
  -> Benchmark catalog gate
     -> exact-head catalog-manifest-v2 artifact
        -> Benchmark profile and optimization seed
           -> hotspot-selection.toml
              -> measure_target.sh
                 -> bounded Pi candidate loop
```

The profile workflow listens only to a successful `Benchmark catalog gate`
`workflow_run`. It checks out and measures
`github.event.workflow_run.head_sha`, not the workflow's default-branch
`github.sha`. Missing, failed, or mismatched artifacts fail closed.

## Catalog gate

```bash
SDPX_CATALOG_OUTPUT=/tmp/catalog-manifest-v2.toml \
  julia --project=benchmark/bootstrap/benchenv \
  benchmark/optimization/catalog_build.jl
```

Every registered case is represented in the manifest. Build-only physics
catalogs are built, fingerprinted, and witness-validated but never selected
for solving. A case is solve-eligible only with an optimal/reference contract,
certificate policy, and independent objective oracle. The optional V2 adapter
(`benchmark/general/v2/GeneralBenchmarkV2.jl`) is consumed when present; its
transform exactness and reference metadata remain in the manifest.

## Profile and target selection

```bash
SDPX_PROFILE_FIXTURE=1 SDPX_OPTIMIZATION_TEST_MODE=1 \
  SDPX_PROFILE_OUTPUT=/tmp/profile.toml \
  julia --project=. benchmark/optimization/profile_catalog.jl
SDPX_HOTSPOT_MANIFEST=/tmp/profile.toml SDPX_PROFILE_FIXTURE=1 \
  SDPX_OPTIMIZATION_TEST_MODE=1 \
  benchmark/optimization/measure_target.sh

Fixture mode is test-only and requires both switches; either one alone is
rejected. Automatic dependent profiling is readiness-gated by the repository
variable `SDPX_ENABLE_DEPENDENT_OPTIMIZATION=true`, which may be enabled only
after the catalog workflow has published a complete solve-eligible schema-v9
receipt. Manual `workflow_dispatch` additionally requires the explicit
`enable_experimental=true` input. No fixture may satisfy the live gate.
```

The live profile uses one untimed warmup followed by exactly three timed
repetitions. Every sample must independently pass status, certificate,
semantic, objective/reference-interval, deterministic-iteration/objective,
identity, route, provider, and declared-trajectory checks. JIT/setup are
reported separately and excluded from the solver metric. The selector uses the median core time
when available, otherwise the solver-call time. Build-only, xfail, uncertified,
failed, fallback, and missing-metric rows cannot be selected. Ties are resolved
by descending allocation telemetry and then lexical `case_key`.

Each target row records deterministic identity, status, certificate and
semantic result, objective, iterations, requested/planned/executed route,
phase timings when available, allocations, RSS, transform exactness, and
failure taxonomy. The target manifest is immutable evidence, not a tuning
input.

## Exact target measurement

`measure_target.jl` consumes the selected case manifest and emits one
machine-readable `METRIC` line. It requires exactly three measured samples,
valid certificates, semantic success, and identical per-sample iterations.
`measure_target.sh` pins Julia GC to one thread for MultiFloats and fixes Julia,
OpenBLAS, and MKL thread defaults. Allocation audits are secondary evidence;
nonzero allocations never suppress correctness or make a candidate pass.

## Bounded optimization loop

`dependent_optimize_workflow.js` is a `workflowScriptPath` template for Pi
subagents. The parent launches it only after the catalog/profile artifacts
exist. It runs up to six rounds and twelve hours, with three parallel scouts,
three isolated candidate worktrees, an independent evidence reviewer, and a
fresh-worktree integration validation per round. Two consecutive rounds with
less than 2% accepted improvement stop the loop. Failed correctness,
certificate, identity, determinism, timeout, RSS, or performance evidence is
not retried as success.

Children only run commands and return bounded reports. They must not mutate
GitHub Actions, embed credentials, merge, or push. The parent agent owns the
final merge gate and verifies tree identity, the frozen CSDR trajectory SHA,
full `Pkg.test`, and the complete regression corpus before publication.

## Evidence layout and resume

A run should retain:

```text
catalog-manifest-v2.toml
catalog-manifest-v2.sha256
profile-catalog.toml
hotspot-selection.toml
hotspot-selection.json
measure-target.toml
hsd-allocation.log                 # secondary, non-blocking
candidate/<round>/<candidate>/     # patch, logs, receipts, test output
```

GitHub artifacts are named by exact source SHA and catalog run ID. A live
hotspot manifest is accepted only when its source/tree SHA equals checked-out
HEAD; fixture manifests require explicit test mode. Optional CFT/PMP2SDP absence
is recorded as `optional_unavailable` and controlled by
`SDPX_REQUIRE_OPTIONAL_CFT`, never silently treated as a pass. To resume,
redownload the successful catalog artifact, verify its source SHA and checksum,
rerun profile selection, then invoke the Pi template with that immutable
manifest. Never retarget from a candidate's result and never reuse an artifact
from another source commit.

## Existing workflow compatibility

`optimization-benchmark.yml` remains reusable for legacy manual comparisons,
but its allocation audit is evidence-only. The new catalog/profile workflows
are the dependency-aware path; they do not alter solver source or physics
formulas.
