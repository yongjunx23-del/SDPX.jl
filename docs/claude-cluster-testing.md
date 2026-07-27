# SDPX Cluster Installation and Test Handoff

This document is a self-contained handoff for an independent Claude session.
It describes the SDPX installation that was verified on 2026-07-27 and the
commands needed to reproduce its validation. Use English in all new logs,
scripts, result summaries, and code changes.

## 1. Connect and select the release

From the local machine, connect through the configured SSH alias:

```bash
ssh hpc
```

The canonical SDPX root is:

```text
/public/home/yongjunxu/projects/SDPX.jl
```

The promoted release is selected atomically through:

```text
/public/home/yongjunxu/projects/SDPX.jl/current
```

Set reusable shell variables:

```bash
BASE=/public/home/yongjunxu/projects/SDPX.jl
SOURCE="$(readlink -f "$BASE/current")"
RELEASE="$(dirname "$SOURCE")"
COMMIT="$(cat "$RELEASE/metadata/commit.txt")"
ENVIRONMENT="$RELEASE/environment"
JOBDIR="$BASE/jobs/$COMMIT"
```

Resolve `current` once at the start of a campaign and record `COMMIT` in every
result. Do not edit the release source or environment in place. A newer commit
must be installed as a new immutable release and pass validation before the
symlink is repointed.

## 2. Verify the existing installation

The site Julia helper is:

```text
/public/home/yongjunxu/projects/csdr-clarabel-multifloat/src/hpc_julia_clarabel_env.sh
```

Load Julia and the shared package depot:

```bash
source /public/home/yongjunxu/projects/csdr-clarabel-multifloat/src/hpc_julia_clarabel_env.sh
export JULIA_DEPOT_PATH="$BASE/depot${JULIA_DEPOT_PATH:+:$JULIA_DEPOT_PATH}"
```

Verify the immutable release metadata and resolved environment:

```bash
test "$(cat "$RELEASE/metadata/commit.txt")" = "$COMMIT"
test -d "$SOURCE"
test -f "$ENVIRONMENT/Project.toml"
test -f "$ENVIRONMENT/Manifest.toml"
(
  cd "$RELEASE"
  sha256sum -c metadata/environment.sha256
)
```

The authoritative environment checksums are stored in
`$RELEASE/metadata/environment.sha256`; do not copy values from an older
release.

A lightweight package-load check may run on the login node:

```bash
export JULIA_PKG_OFFLINE=true
"$JULIA_BIN" \
  --project="$ENVIRONMENT" \
  --startup-file=no \
  -t 1 \
  -e 'using SDPX, MultiFloats; println("SDPX load: passed")'
```

Do not run full tests or benchmarks on the login node.

The environment is already installed. Only if a new immutable release has no
resolved environment, run its release-specific setup script on the login node:

```bash
bash "$JOBDIR/setup-release.sh"
```

Do not run `Pkg.add`, `Pkg.instantiate`, or package downloads from a compute
node. Compute jobs use `JULIA_PKG_OFFLINE=true`.

## 3. Verified benchmark inputs

### Sparse `J/K/N_a/N_mu = 40/4/20/100`

```text
path: /public/home/yongjunxu/projects/SDPX.jl/data/csdr-40-4-20-100/8622330ad83989a3c183e82b7b0cedc5bbcd1572e96c35e8cc19750129dc4d4a.bin
size: 614325622 bytes
sha256: 8622330ad83989a3c183e82b7b0cedc5bbcd1572e96c35e8cc19750129dc4d4a
```

### Dense lattice `Task_Low08`

```text
path: /public/home/yongjunxu/projects/SDPX.jl/data/task-low08/5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3.bin
size: 4640880 bytes
sha256: 5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3
```

Verify before every new campaign:

```bash
stat -c '%s %n' \
  "$BASE/data/csdr-40-4-20-100/8622330ad83989a3c183e82b7b0cedc5bbcd1572e96c35e8cc19750129dc4d4a.bin" \
  "$BASE/data/task-low08/5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3.bin"

sha256sum \
  "$BASE/data/csdr-40-4-20-100/8622330ad83989a3c183e82b7b0cedc5bbcd1572e96c35e8cc19750129dc4d4a.bin" \
  "$BASE/data/task-low08/5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3.bin"
```

Do not copy either input into a Git repository or release source tree.

## 4. Submit the complete validation job

The validated PBS script requests eight cores, 32 GiB, and two hours from the
`normal` queue:

```bash
cd "$JOBDIR"
VALIDATION_JOB="$(qsub validate-cluster-release.pbs)"
printf 'validation_job=%s\n' "$VALIDATION_JOB"
```

Monitor without running another copy:

```bash
qstat -f "$VALIDATION_JOB" |
  egrep 'job_state|resources_used.walltime|resources_used.mem|Exit_status'
```

The script performs, in order:

1. `Pkg.test("SDPX"; coverage=false)` with four Julia threads;
2. analytic Float64x4 and 256-bit BigFloat smoke solves with one thread;
3. the required Float64 `Task_Low08` solve with eight Julia and eight
   factorization BLAS threads selected internally;
4. the sparse scheduler correctness benchmark with up to eight Julia threads;
5. SHA-256 generation for every result artifact.

Results are written to:

```text
/public/home/yongjunxu/projects/SDPX.jl/results/<commit>/<PBS-job-id>/
```

After completion, require all of the following:

```bash
qstat -xf "$VALIDATION_JOB" | grep -E 'job_state|exit_status'
cat "$BASE/results/$COMMIT/$VALIDATION_JOB/validation-status.txt"
sha256sum -c \
  "$BASE/results/$COMMIT/$VALIDATION_JOB/artifact-sha256.txt"
```

The expected status is `exit_status=0` and
`validation_status=passed`.

Numerical acceptance gates:

- all package tests pass;
- analytic Float64x4 and BigFloat solves return `Optimal`;
- `Task_Low08` returns `Optimal`;
- `abs(objective - 0.653291393898) <= 1e-6`;
- relative gap, primal residual, and dual residual are at most `1e-6`;
- maximum equality residual is at most `1e-9`;
- the original-coordinate primal and dual PSD checks pass their
  problem-scaled tolerance; record the raw minimum eigenvalues as diagnostics
  rather than replacing the scale-aware check with a fixed absolute cutoff;
- sparse Schur validation has no meaningful relative error;
- peak memory remains below the PBS request.

The 2026-07-26 Task_Low08 optimization campaign used baseline job
`193930.node220`, optimized comparison jobs `193934.node220` and
`193936.node220`, focused validation job `193935.node220`, and complete
package-test job `193938.node220` (1,948/1,948 tests passed). Its final
Float64 result was:

```text
status: Optimal
iterations: 27
objective: 0.653291393898
relative gap: 4.61e-7
maximum equality residual: 2.06e-12
minimum primal PSD eigenvalue: -3.26e-15
minimum dual PSD eigenvalue: 2.12e-14
best solve time: 37.03 s
best-run peak RSS: 5.63 GiB
```

The latest-release baseline was 51.59 s on the same node. The promoted path is
1.39x faster. See
`bench/opt2026/TASK_LOW08_CLUSTER_REPORT_2026-07-26.md` for detailed phase,
kernel, memory, and numerical comparisons.

Test counts may increase in newer commits; the exit status and tests contained
in the candidate release are authoritative.

### 2026-07-27 Task_Low08 precision campaign

The current v0.2.0 candidate adds a narrowly gated automatic Task_Low08
parameter profile, a lower-only threaded Float64x4 refinement product, and a
guarded Float64 → Float64x2 → Float64x4 dense-KKT hierarchy. The full local
package suite passed 2,107 of 2,107 assertions.

The fair same-node Float64 reference used the same 394-column equality basis,
`1e-6` tolerance, warm-up, timing boundaries, and thread limits. At one
thread, MOSEK / SDPX solver medians were 75.783 / 88.123 seconds. At 16
threads, they were 40.369 / 32.411 seconds; SDPX was 19.7% faster in solver
time and used 6.19 rather than 12.66 GB peak RSS. The measured SDPX
complete-solve optimum was 16 Julia plus 16 OpenBLAS threads with interleaved
NUMA allocation.

The full Float64x4 run used 16 Julia threads, one BLAS thread,
`extended_precision_blas=:auto`, `mixed_precision_kkt=:on`, and a 64 GiB
request:

```text
status: Optimal
iterations: 55
solve time: 2241.803 s
relative gap: 7.08808e-13
primal / dual residual: 8.95078e-20 / 5.48435e-17
maximum equality violation: 2.43274e-17
minimum primal / dual PSD eigenvalue: -1.10118e-11 / -3.98301e-14
peak RSS: 20.934 GiB
native Float64x4 fallback: not entered
```

Do not submit a native 256-bit BigFloat Task_Low08 solve in a 64 GiB job. The
conservative workspace estimate is 95.720 GiB before operating-system
headroom; request at least 128 GiB. Use the medium CSDR exact-arrow model for
routine BigFloat regression, where the retained 256-bit eight-thread solve is
approximately 87 seconds and certificate-equivalent to the legacy path.

See
`bench/opt2026/TASK_LOW08_PRECISION_CLUSTER_REPORT_2026-07-27.md` for the
complete timing tables, rejected experiments, crossovers, and raw job IDs.

### Latest promoted release

The v0.2.0 cluster release includes the dense Task_Low08 work and the medium
sparse `2x2` block-arrow optimization. Resolve the immutable release selected
by `current` and read its metadata instead of relying on a copied commit:

```bash
SOURCE="$(readlink -f /public/home/yongjunxu/projects/SDPX.jl/current)"
RELEASE="$(dirname "$SOURCE")"
printf 'source: %s\n' "$SOURCE"
cat "$RELEASE/metadata/validation.txt"
```

The medium CSDR acceptance campaign recorded:

```text
exact reduced-Schur validator: 194115.node220
exact MultiFloatVec SYRK comparison: 194141.node220
final Float64x4 scaling jobs:
  194145.node220, 194149.node220, 194150.node220, 194144.node220
BigFloat diagnostics:
  194139.node220, 194143.node220, 194148.node220, 194151.node220
full package test: 194154.node220 (2,027 of 2,027 tests)
final Task_Low08 Float64 regression: 194155.node220
medium CSDR Float64x4 status: Optimal
medium CSDR Float64x4 best time: 11.727777 s (8 Julia threads)
medium CSDR Float64x4 scaling:
  51.479394 / 31.342820 / 19.349405 / 11.727777 s
  at 1 / 2 / 4 / 8 Julia threads
medium CSDR BigFloat256 status: Optimal
medium CSDR native BigFloat256 reduced-arrow time:
  205.202262 / 191.701491 / 110.741381 / 86.752160 s
  at 1 / 2 / 4 / 8 Julia threads
medium CSDR native BigFloat192 reduced-arrow time: 80.703301 s at 8 threads
medium CSDR mixed BigFloat256 diagnostic: 323.897000 s; exact refinement
  stalled and the solver safely fell back to native BigFloat
canonical model manifest:
  df62be289368abb162e43cddba72cd13efe79cbf441d1596454a658b4175592b
```

The validated Float64x4 default is the exact two-row reduced panel with the
four-lane `MultiFloatVec{4,Float64,4}` lower-triangular SYRK. Float64x4 and
BigFloat use `extended_precision_blas=:auto`; Float64 is unchanged. General
native BigFloat remains serial, while exact singleton-local `2x2` arrows may
use exclusive block/panel ownership and disjoint Schur tiles. The mixed arrow
factor remains opt-in.

The release is selected by:

```text
/public/home/yongjunxu/projects/SDPX.jl/current
```

The independent medium benchmark and audit are stored at:

```text
/public/home/yongjunxu/projects/sdpx-benchmarks/csdr-medium-j32-k4
```

See
`bench/opt2026/CSDR_MEDIUM_J32_K4_CLUSTER_2026-07-26.md` for the complete
algorithm, timing, allocation, memory, scaling, and numerical-validation
report. The latest Task_Low08 results remain documented separately in
`bench/opt2026/TASK_LOW08_CLUSTER_REPORT_2026-07-26.md`.

OpenBLAS remains the selected Linux backend. Controlled Task_Low08
comparisons found both MKL 2025.2 and BLISBLAS 0.2 slower on the AMD EPYC 7742
nodes. The alternative backends remain isolated benchmark dependencies.

## 5. Submit the 614 MB high-precision benchmark

Run this only as a PBS job. The script requests eight cores, 64 GiB, and eight
hours:

```bash
cd "$JOBDIR"
SPARSE_JOB="$(qsub sparse-benchmark.pbs)"
printf 'sparse_job=%s\n' "$SPARSE_JOB"
```

It runs Float64x4 sequentially with 1, 2, 4, and 8 Julia threads, always with
one BLAS thread. It then runs native BigFloat at 256-bit precision with one
Julia thread. Each configuration uses three timed repetitions. A separate
normal-queue job may sweep the experimental mixed reduced-arrow mode over
1/2/4/8 workers; reserve eight PBS cores even for the smaller widths.

Check completion:

```bash
qstat -xf "$SPARSE_JOB" | grep -E 'job_state|exit_status'
RESULT="$BASE/results/$COMMIT/$SPARSE_JOB"
cat "$RESULT/benchmark-status.txt"
cat "$RESULT/float64x4-schur.csv"
cat "$RESULT/bigfloat-sparse-schur.csv"
sha256sum -c "$RESULT/artifact-sha256.txt"
```

The last successful benchmark was job `193886.node220`:

| Arithmetic | Threads | Minimum kernel time | Peak RSS |
|---|---:|---:|---:|
| Float64x4 | 1 | 8.007 s | 2,111,808 KiB |
| Float64x4 | 2 | 3.971 s | 2,043,828 KiB |
| Float64x4 | 4 | 2.044 s | 2,086,460 KiB |
| Float64x4 | 8 | 1.077 s | 2,074,924 KiB |

The eight-thread speedup over one thread was `7.44x`. Sampled Float64x4 Schur
relative error was at most `9.62e-65`.

For BigFloat, the fused-arrow Schur kernel changed from `54.967 s` to
`9.771 s`, a `5.63x` speedup, with zero reported relative error. The isolated
BigFloat process peaked at `7,188,924 KiB`.

This 614 MB job is a Schur/kernel benchmark, not a complete optimization solve.
Do not report it as a converged full SDP solve.

## 6. Resource and arithmetic rules

- Keep general native BigFloat solver work serial. Exact singleton-local `2x2`
  arrows may sweep 1/2/4/8 Julia threads because their native block preparation
  and triangular Schur tiles have exclusive ownership. In mixed reduced-arrow
  experiments, the Float64x4 panel and factorization use workers.
- Use separate PBS jobs for independent BigFloat cases.
- For Float64x4 scheduler scaling, use 1/2/4/8 Julia threads and one BLAS
  thread; do not enable full Julia and BLAS widths simultaneously.
- Do not exceed the script's `SDPX_MEMORY_LIMIT_BYTES` planning ceiling.
- Do not modify or delete historical data, results, or releases.
- Do not expose passwords, private keys, tokens, complete environments,
  compute-node hostnames, or private topology in logs or reports.
- Record the full commit, Julia version, thread counts, memory request, input
  checksum, PBS job ID, status, numerical errors, and peak RSS.

## 7. Testing a newer SDPX candidate

Do not overwrite this validated release. A newer commit must have its own:

```text
$BASE/releases/<new-full-commit>/source
$BASE/releases/<new-full-commit>/environment
$BASE/jobs/<new-full-commit>/
$BASE/results/<new-full-commit>/<PBS-job-id>/
```

Update the `COMMIT` embedded in copied PBS scripts, resolve dependencies on the
login node, and run the complete validation job before the 614 MB benchmark.
Do not repoint `current` until every numerical and memory gate passes.

If a new candidate fails, preserve its release, logs, result artifacts, and
environment manifest for diagnosis. Do not replace the validated release
merely because a job was submitted or started.
