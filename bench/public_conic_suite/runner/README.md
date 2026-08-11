# Generated pathological P0 cluster runner

Cluster-only campaign for repository-generated LP/SOCP/SDP pathological cases
across `Float64`, `Float64x4`, and `BigFloat`.  Everything is computed offline
from `../generators/SDPXPathologicalBenchmarks.jl` and `campaign.jl`; there are
no downloads, no external binaries, and no Pkg operations at runtime.

## Files

- `result_schema.jl` - canonical CSV column contract, normalized-status mapping,
  CSV/TOML/Markdown writers, and SUCCESS/FAILED markers.
- `campaign.jl` - P0 case matrix, resource matrix, deterministic input hashes.
- `run_generated_pathological.jl` - the compute entry point.  Refuses to run
  unless `JULIA_PKG_OFFLINE=true`, records the full audit row per repetition,
  evaluates all gates, and writes `SUCCESS` only when every row passes.
- `static_check.jl` - solver-free preflight that validates the schema and
  campaign matrix without loading SDPX/JuMP.
- `generated_pathological.pbs` - queue `sugon`, regular class: `ppn=5`,
  Julia `-t 4`, solver threads 4, BLAS threads 1, `float64,float64x4`.
- `generated_pathological_bigfloat.pbs` - queue `sugon`, bigfloat class:
  `ppn=1`, Julia `-t 1`, solver threads 1, BLAS threads 1, `bigfloat256`.
- `submit_generated_pathological.sh` - dry-runs or submits both PBS jobs.
- `static_check.sh` - bash-only static verification for this directory.

## Cluster usage

```bash
export SDPX_SOURCE=/path/to/immutable/SDPX-v041-auto-conic
export SDPX_CAMPAIGN_ROOT=/unique/result/root/outside-the-candidate
export SDPX_SITE_ENV=/path/to/site-env.sh     # required; exports JULIA_BIN
export SDPX_ENVIRONMENT=/path/to/matching-environment
export SDPX_DEPOT_PATH=/path/to/offline-julia-depot
export REGULAR_NODE_NAME=healthy-node-a        # pinned with ppn=5
export BIGFLOAT_NODE_NAME=healthy-node-b       # pinned with ppn=1; may equal A

./bench/public_conic_suite/runner/static_check.sh
./bench/public_conic_suite/runner/submit_generated_pathological.sh --submit
```

`SDPX_CAMPAIGN_ROOT` must be a fresh path outside the candidate source; the
submit script refuses to reuse an existing root and does not create result
directories.  The two PBS jobs create `regular/` and `bigfloat/` under it
themselves, so they never collide.  Each job writes `environment.txt`,
`runner.log`, `process.time.txt`, `results.csv`, `failures.csv`,
`benchmark_manifest.toml`, `report.md`, and either `SUCCESS` or `FAILED`.
`/usr/bin/time -v` output is redirected with `-o` to each result directory's
`process.time.txt`.

`SDPX_SITE_ENV`, `SDPX_ENVIRONMENT`, and `SDPX_DEPOT_PATH` are all required
(no defaults), `NODE_NAME` is required inside each PBS script and must equal
`hostname` or the job aborts.  The submit script pins nodes with
`qsub -l nodes=<REGULAR_NODE_NAME>:ppn=5` and
`-l nodes=<BIGFLOAT_NODE_NAME>:ppn=1`.  Compute runs offline.

The candidate commit comes from `git rev-parse HEAD` with a fallback to the
standard immutable layout `dirname(source)/metadata/source_commit.txt` and then
`candidate_metadata.toml`; it is required to be non-empty.  If present,
`dirname(source)/metadata/archive_sha256.txt` (also `archive_sha.txt` /
`archive_sha`) is reported as `archive_sha`.  Official PBS runs are immutable
and strict: `gate_identity` requires a non-empty `archive_sha` from
`metadata/archive_sha256.txt`, so local git-checkout trials are permitted only
as diagnostics and cannot pass the official gate unless the archive hash is
present.

## Gates

`SUCCESS` is written only when every requested row passes all of:

- identity: candidate `pathof(SDPX)` lives under `SDPX_SOURCE` and source,
  input, environment, candidate-pathof, git commit, archive, active project,
  Project, and Manifest hashes are all non-empty;
- resource: `ppn`/Julia/BLAS/solver threads match the class contract;
- route: planned backend equals executed, or the difference is an explicitly
  authorized LP deferred resolution (`post_presolve`,
  `analytic_equality_only`, `resolved_no_iteration`, `not_resolved`) or a
  non-`none` fallback reason. Archive baselines that predate the
  `planned_backend`/`executed_backend` fields (reported as
  `not_available`/`not_executed`) pass only when the legacy actual
  `solver_algorithm`, `kkt_backend`, and `gram_kernel` fields are real and no
  unexpected fallback is recorded;
- status: certificate-aware normalized status equals the analytic expectation;
  the weak-infeasible `delta=0` case may be honestly `unresolved` but a raw
  Optimal without a certificate is always rejected;
- objective: objective error is within `10 * tolerance` of the analytic oracle
  when one exists;
- certificate: SDPX's original-coordinate certificate is valid and scaled
  residual/gap are within tolerance.

## Audit fields

Every row records raw and normalized status, planned/executed backend and
formulation, fallback reason/resolution, objective and relative gap,
original-coordinate affine/cone residuals, primal/dual cone margins,
certificate type/residual/failures, phase timings, workspace and peak RSS,
thread configuration, candidate pathof identity, source/archive/input
SHA-256 fingerprints, active project path plus Project/Manifest hashes, and
environment hashes.  See `result_schema.jl` for the exact column order.
Numeric correctness fields are kept in the target arithmetic (`Float64`,
`Float64x4`, or `BigFloat`); only timings, RSS, and display values are
`Float64`.  `lp_formulation` is a dedicated column for the executed LP Newton
formulation; planned/executed `formulation` columns derive from the actual
execution plan/selected backend.  Cone margins are normalized lower bounds
derived from SDPX's target-arithmetic PSD certificate details
(`-required_shift / scale`), never a generic LAPACK eigmin.

Rows whose analytic margin is below an arithmetic's precision carry
`min_bits` in `campaign.jl` and are skipped for that arithmetic via
`campaign_rows_for(bits)`, so the Float64 suite only executes cases it can
honestly certify.  P0 v5 keeps analytic margins well above the ladder's
default tolerance: Float64 runs LP row scaling at `decades=6`,
`socp_near_infeasible` uses a linear separation of exactly `epsilon`
(`1e-2` Float64 as an audited certificate-stability floor / `1e-12`
extended), and the weak-infeasible SDP stays at
the audited `delta=1e-8` severity for every arithmetic.
Hilbert and small-eigenvalue PSD matrices are built as explicitly typed
`Matrix{JuMP.GenericAffExpr{T,typeof(t)}}` matrices via
`_typed_affine_psd_matrix`, never `Matrix{Any}`.

## Static verification

Local development must not execute Julia or numeric code.  Use:

```bash
bash bench/public_conic_suite/runner/static_check.sh
git diff --check
```

The static check covers shell/PBS syntax, the PBS resource contract, offline
enforcement, schema audit coverage, and SUCCESS gating.

## Cluster validation checklist

- `static_check.jl` passes on the login node (no solver deps required).
- Both PBS headers use `#PBS -q sugon`.
- Regular job: `PBS_NP=5`, `julia_threads=4`, `solver_threads=4`,
  `blas_threads=1` in `results.csv`.
- BigFloat job: `PBS_NP=1`, all thread columns `1`, arithmetic `bigfloat256`.
- `results.csv` row shape matches `RESULT_COLUMNS` exactly.
- `SUCCESS` exists only if `failures.csv` is header-only and `gate_pass=true`
  on every row.
- Both jobs reuse the same immutable `SDPX_SOURCE`; `candidate_pathof_match`
  must be `true`.
- `regular/` and `bigfloat/` are separate result dirs under one fresh
  `SDPX_CAMPAIGN_ROOT`; `process.time.txt` exists in each.
- `gate_route` is true on every row; LP `planned_backend=:lp_deferred` may
  resolve via authorized `backend_resolution` values, and
  `resolved_no_iteration`/`not_resolved` must report `executed_backend=:not_executed`.
