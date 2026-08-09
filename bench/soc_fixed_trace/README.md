# Fixed-trace CSDR J40/J80 benchmark

This directory contains the compute-node harness for the two immutable
spectral-primal CSDR traces:

| case | source parameters | PSD2 blocks | variables | equalities |
| --- | --- | ---: | ---: | ---: |
| J40 | `J40/Na20/Nmu200/Nx3` | 4,200 | 8,400 | 170 |
| J80 | `J80/Na40/Nmu800/Nx3` | 32,800 | 65,600 | 350 |

The equality count is after the fixed `twice_subtraction` reduction.  A model
file is immutable input; the driver records its SHA-256 and refuses an
`--expected-hash` mismatch.  Never compare rows with different model hashes.

## Driver

`benchmark.jl` accepts a serialized CSDR payload, an `SDPX.SDPProblem`, or a
`(c,A,C,B,b)` tuple.  If a release directory is needed to deserialize the
payload, pass `--release=/path/to/CSDR-release`; the loader invokes the release's
low-energy-variable elimination helper when necessary.  The source payload is
normally `Float64x4`.  A requested `BigFloat256` run creates an independent
256-bit `SDPProblem` while preserving the sparse active-variable layout; it
does not merely change solver options on the x4 model.

Reference SDP and native fixed-trace SOC are separate modes:

```bash
# PSD reference (one process, all output in English)
julia --project=. -t 8 bench/soc_fixed_trace/benchmark.jl \
  --case=J40 --model=/data/J40.bin --mode=sdp \
  --arithmetic=Float64x4 --reps=3 --warmup=1 \
  --expected-hash="$J40_SHA" --output=results/j40-sdp.toml

# Native Q3/SOC path; native socp requires original coordinates
julia --project=. -t 8 bench/soc_fixed_trace/benchmark.jl \
  --case=J40 --model=/data/J40.bin --mode=socp --scaling=none \
  --arithmetic=Float64x4 --reps=3 --warmup=1 \
  --expected-hash="$J40_SHA" --output=results/j40-socp.toml
```

Arithmetic labels are exactly `Float64`, `Float64x4`, and `BigFloat256`.
`--precision-bits=256` and `working_precision_policy=:fixed` are used for the
BigFloat run.  `--preflight` (the default) checks the expected geometry, every
block's exact fixed trace, and all 2x2 SOC candidates before timing.
`--no-preflight` is rejected for serialized J40/J80 runs; only the synthetic
loader smoke may omit the external provenance checks.

Before a high-precision J80 solve, use `--preflight-only`. It performs model
loading, exact low-energy elimination, arithmetic conversion, hash/geometry
validation, and reports the compact Q3 workspace estimate, the equality-arrow
SDP floor, and the conservative SDP campaign estimate without allocating a
Newton workspace or entering an iteration. The selected estimate must fit in
70% of scheduler/cgroup/configuration-aware free memory. An unavailable zero
estimate is rejected rather than interpreted as zero bytes. This is the
mandatory memory gate for BigFloat256.

The campaign default is `1e-12` for all three normalized stopping tolerances
and 500 iterations.  The selected best Float64x4 and BigFloat256 configurations
must also be repeated at `1e-20`; never compare rows with different tolerance
or precision fields.

`--q3-gram-strategy=output_tiles` and `row_bins` are expert benchmark
overrides. The former assigns each output tile to one worker and keeps no
replicated Gram storage. The latter assigns contiguous panel rows to workers,
stores one packed lower triangle per worker, and performs a deterministic
tree reduction. Production `auto` retains output tiles. Controlled J40
BigFloat256 measurements found row bins slower, so they remain only a
research/reference override and are not part of the release campaign.
For BigFloat, automatic output tiles require at least two selected workers, 32
equality columns, and 250,000 triangular contractions; smaller panels retain
the pairwise kernel. The report records the executed Gram worker count and
selection reason, so an allocated-but-idle thread request is visible.
Even when forced, their private storage must fit the configured memory fraction.

The protocol is: load and hash → preflight → `warmup` complete solves → `reps`
timed cold solves.  Every timed and warm-up result is certified with
`SDPX.result_certificate` in original coordinates.  A failed certificate or
non-`Optimal` status exits non-zero. Timed failures are still serialized before
the nonzero exit, including `benchmark_valid=false`, `execution_valid`, and the
validation error; this preserves partial phase and resource evidence from an
expensive `TimeLimit` or `IterationLimit` run. Warm-up failures remain strict
and stop before timing. TOML/CSV rows include status, iterations,
objectives, certificate residuals, minimum primal/dual PSD2 margins, allocated
bytes, GC time, and the solver's phase timers.  The report and its companion
manifest include requested and actual Julia/solver threads, BLAS threads,
process peak RSS, total/available memory, CPU model/count, cpuset, NUMA policy,
per-process NUMA residency, PBS job/node metadata, model hash, and observed
geometry.  Every timed solve also records process CPU seconds, mean active
cores, utilization relative to the requested solver thread count, and
voluntary/involuntary context switches.  This distinguishes cores that were
actually busy from scheduler slots that were merely reserved.
Rows also summarize the complete adaptive parameter history (first, last,
minimum, and maximum `sigma`, `mu`, affine/accepted steps, total backtracking,
and fallback count). This makes a slow complementarity tail distinguishable
from an expensive linear-algebra iteration without storing a large opaque
Julia object in the portable TOML/CSV result.
The provenance digest includes both numerical source and the fixed-trace
benchmark drivers/launchers, so changing a timing or validation gate changes
the recorded digest even when `src/` is unchanged.

The release campaign is capped at 32 Julia/solver workers. Use
`1/2/4/8/16/32` for J40 and `8/16/32` for J80, with one configuration per PBS
job and BLAS fixed at one thread. Recommendations come from complete-solve
medians, not from the largest scheduler allocation. Retired 64/96/128-worker
experiments remain documented only in `WORKLOG.md`; no current PBS template
requests more than 32 slots. Both the portable launcher and the direct J40/J80
benchmark driver reject wider requests; this is a release-campaign policy, not
a global limit on `SolverOptions` for unrelated SDPX models.

Near-proportional equality detection is diagnostic-only and bounded. On the
immutable J80 Float64x4 model, collision-checked exact cleanup kept all 350
equalities while the old exhaustive diagnostic spent 224.42 seconds. Bounded
diagnostics reduced that stage to 0.0735 seconds and complete preprocessing to
0.395 seconds without changing the model. Reports include an explicit warning
whenever this approximate diagnostic is skipped.

The same target-arithmetic audit found no J80 equality reduction to exploit:
rank presolve retained all 350 columns, all 65,600 equality-row templates were
distinct, and all 32,800 ordered two-row Q3 block templates were distinct. A
sampled full-rank shortcut was slower than the complete target-arithmetic rank
check. Equality dropping, support grouping, and sampled rank proposals are
therefore rejected for this benchmark rather than enabled behind heuristics.

`--synthetic --case=synthetic` builds a tiny fixed-trace Q3 instance for local
syntax or CI checks.  It is intentionally not a J40/J80 benchmark:

```bash
julia --project=. -t 1 bench/soc_fixed_trace/benchmark.jl \
  --synthetic --case=synthetic --mode=socp --scaling=none \
  --arithmetic=Float64 --reps=1 --warmup=0
```

## PBS protocol

Generate the immutable inputs first with `generate_j40.pbs` and
`generate_j80.pbs`.  Both use the pinned CSDR release and an output directory
that must not already exist; this prevents a later generation attempt from
overwriting a model used in a published comparison.  The model SHA is written
next to the binary and must be passed to every solver job.

`j40_sdp.pbs`, `j40_socp.pbs`, `j80_sdp.pbs`, and `j80_socp.pbs` are portable
one-configuration templates.  They reserve a single node, set
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=BLIS_NUM_THREADS=1`,
and reject `SDPX_THREADS > PBS_NP`.  Set the model, hash, and optional site
paths with `qsub -v CSDR_MODEL=...,SDPX_EXPECTED_HASH=...` (or the equivalent
environment exported by your scheduler). The optional site variables are
`JULIA_BIN`, `SDPX_SHARED_ENV`, `SDPX_SHARED_DEPOT`, and `CSDR_RELEASE`. No
private cluster path is embedded in the portable shell launchers: without an
explicit shared environment they use `julia`, the repository project, and the
caller's existing depot path.
The launcher refuses to run on a login node unless `ALLOW_INTERACTIVE=1` is
explicitly set for a smoke test.
Each job writes an isolated output directory and never changes production
symlinks. It prepends a job-private compiled depot and reads packages/artifacts
from the shared depot afterward, preventing two immutable candidates with the
same package UUID/version from racing on one `.ji` cache. A successful
single-configuration launcher writes an explicit `PASSED` marker only after
the timed report and manifest have both been completed.

`SDPX_THREAD_POLICY=exclusive` starts an explicit `N,0` Julia pool with one GC
worker, enables Julia CPU affinity inside the PBS cpuset, and raises the worker
sleep threshold. A same-node J80 control improved completed iterations by
51.2% and equality-Gram throughput by 49.1%. J40 and J80 SDP/SOCP templates
therefore select `exclusive` by default, while every template permits an
explicit override. Reports record the policy, default/interactive pool widths,
affinity flag, and sleep threshold; do not combine rows from different
policies in one scaling curve.

`pair_modes.pbs` runs SDP and SOC in isolated Julia subprocesses.  Use
`SDPX_PAIR_ORDER=sdp-socp` on one allocation and `socp-sdp` on a paired
allocation when alternating order is desired.  `csdr_thread_scaling.pbs` is a
single scaling point; submit an array/resubmit with another `SDPX_THREADS`
value rather than changing thread counts in one process.  `run_cluster.pbs`
is a portable single-point alias.  The older `focused_regression.pbs`,
`full_regression.pbs`, `task_low08_regression.pbs`, and
`csdr_smoke_compare.pbs` remain regression/release-gate utilities and are not
part of the J40/J80 timing table.

## Manifests

`manifest.template.toml` is a reviewable schema for archived reports.  A real
run writes the same fields to `<output>.manifest.toml`; generated result files
are intentionally not tracked.  Keep the original model file and its SHA-256
next to any published timing claim.
