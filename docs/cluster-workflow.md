# Cluster Deployment and Execution

This guide describes a generic single-node PBS workflow for SDPX. It keeps
source releases immutable, preserves previous campaigns, separates the Julia
environment from the source tree, and promotes a release only after numerical
validation. Site-specific accounts, login hosts, filesystem paths, module
commands, and node topology belong in private operator documentation.

The examples use PBS (`qsub`, `qstat`, and `qdel`) and its conventional
`normal` queue. Adapt resource syntax if the local PBS installation differs.
Do not place passwords, private keys, access tokens, complete process
environments, or private site topology in source control or result logs.

## Required site variables

Define these values in the shell or a private site-environment helper before
using the examples:

```bash
export CLUSTER=pbs-login
export SDPX_REMOTE_BASE=/absolute/path/to/SDPX.jl
# export SDPX_SITE_ENV=/absolute/path/to/site-environment.sh  # optional
export JULIA_BIN=/absolute/path/to/julia  # unless SDPX_SITE_ENV defines it
```

- `CLUSTER` is an SSH alias or login hostname.
- `SDPX_REMOTE_BASE` is a new canonical deployment root owned by the user.
- `SDPX_SITE_ENV` is optional; it may load compiler/runtime modules and define
  `JULIA_BIN` or an existing shared `JULIA_DEPOT_PATH`.
- `JULIA_BIN` must resolve to the Julia executable used on login and compute
  nodes.

Always pass an explicit `--project=...`; a site helper may otherwise select an
unrelated default project. Never reuse an older solver checkout as the release
root or overwrite its manifests. Historical inputs may be copied or referenced
only after their documented size and checksum have been verified.

## Immutable release layout

Use one directory per full Git commit:

```text
$SDPX_REMOTE_BASE/
├── current -> releases/<full-commit>/source
├── data/
│   └── csdr-40-4-20-100/
├── depot/
├── incoming/
├── jobs/
│   └── <full-commit>/
├── releases/
│   └── <full-commit>/
│       ├── environment/
│       │   ├── Project.toml
│       │   └── Manifest.toml
│       ├── metadata/
│       │   ├── archive-sha256.txt
│       │   ├── commit.txt
│       │   ├── environment-sha256.txt
│       │   └── previous-current.txt
│       └── source/
└── results/
    └── <full-commit>/
        └── <PBS-job-id>/
```

The `source` directory is exactly one `git archive`. The generated
`environment` is separate so package resolution never modifies the source
tree. A job writes only below its unique result directory. The `current`
symlink is an activation pointer, not a working checkout.

This layout has three useful properties:

1. a result names the exact source commit and resolved Julia environment;
2. a failed candidate cannot corrupt the active release;
3. rollback changes one symlink and does not delete releases or results.

## Prepare a release locally

Deploy only committed source. First confirm the intended branch, tests, and
working tree:

```bash
cd /path/to/SDPX.jl
git status --short --branch
git diff --check
git rev-parse HEAD
```

Do not archive a dirty tree. After the final commit is available, create an
archive whose top-level directory is `source`:

```bash
COMMIT="$(git rev-parse HEAD)"
ARCHIVE="/tmp/sdpx-${COMMIT}.tar.gz"

git archive \
  --format=tar.gz \
  --prefix=source/ \
  --output="$ARCHIVE" \
  "$COMMIT"

ARCHIVE_SHA="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
printf 'commit=%s\narchive=%s\nsha256=%s\n' \
  "$COMMIT" "$ARCHIVE" "$ARCHIVE_SHA"
```

The archive intentionally excludes ignored benchmark inputs and results. Move
large inputs separately and verify them by content hash.

## Install a candidate without overwriting anything

Inspect the configured target before the first write:

```bash
: "${CLUSTER:?set CLUSTER to the PBS login SSH alias}"
: "${SDPX_REMOTE_BASE:?set SDPX_REMOTE_BASE to the deployment root}"

ssh "$CLUSTER" \
  "test ! -e '$SDPX_REMOTE_BASE/releases/$COMMIT' &&
   test ! -e '$SDPX_REMOTE_BASE/releases/$COMMIT.tmp'"
```

Create only the new canonical root and its children:

```bash
ssh "$CLUSTER" \
  "mkdir -p \
    '$SDPX_REMOTE_BASE/releases' \
    '$SDPX_REMOTE_BASE/incoming' \
    '$SDPX_REMOTE_BASE/depot' \
    '$SDPX_REMOTE_BASE/data' \
    '$SDPX_REMOTE_BASE/results' \
    '$SDPX_REMOTE_BASE/jobs'"
```

Upload to a temporary name, verify the archive remotely, and extract into a
new temporary release:

```bash
scp "$ARCHIVE" \
  "$CLUSTER:$SDPX_REMOTE_BASE/incoming/$COMMIT.tar.gz.part"

ssh "$CLUSTER" "
  set -eu
  uploaded='$SDPX_REMOTE_BASE/incoming/$COMMIT.tar.gz.part'
  actual=\$(sha256sum \"\$uploaded\" | awk '{print \$1}')
  test \"\$actual\" = '$ARCHIVE_SHA'
  mv \"\$uploaded\" '$SDPX_REMOTE_BASE/incoming/$COMMIT.tar.gz'
  mkdir '$SDPX_REMOTE_BASE/releases/$COMMIT.tmp'
  tar -xzf '$SDPX_REMOTE_BASE/incoming/$COMMIT.tar.gz' \
    -C '$SDPX_REMOTE_BASE/releases/$COMMIT.tmp'
  mkdir -p \
    '$SDPX_REMOTE_BASE/releases/$COMMIT.tmp/environment' \
    '$SDPX_REMOTE_BASE/releases/$COMMIT.tmp/metadata'
  printf '%s\n' '$COMMIT' \
    > '$SDPX_REMOTE_BASE/releases/$COMMIT.tmp/metadata/commit.txt'
  printf '%s\n' '$ARCHIVE_SHA' \
    > '$SDPX_REMOTE_BASE/releases/$COMMIT.tmp/metadata/archive-sha256.txt'
  mv '$SDPX_REMOTE_BASE/releases/$COMMIT.tmp' \
    '$SDPX_REMOTE_BASE/releases/$COMMIT'
"
```

Every final path is new. If a temporary or final release already exists, stop
and inspect it. Do not delete or overwrite it automatically.

## Resolve the Julia environment on the login node

Compute nodes cannot be assumed to reach Julia package servers. Resolve,
download, and precompile dependencies on the login node before submitting a
job. Use a release-specific environment and a private depot layered in front
of any site-provided shared cache:

```bash
BASE="$SDPX_REMOTE_BASE"
RELEASE="$BASE/releases/$COMMIT"
SOURCE="$RELEASE/source"
ENVIRONMENT="$RELEASE/environment"

if [ -n "${SDPX_SITE_ENV:-}" ]; then
  source "$SDPX_SITE_ENV"
fi
: "${JULIA_BIN:?set JULIA_BIN directly or through SDPX_SITE_ENV}"

export JULIA_DEPOT_PATH="$BASE/depot${JULIA_DEPOT_PATH:+:$JULIA_DEPOT_PATH}"
export SDPX_RELEASE_SOURCE="$SOURCE"

"$JULIA_BIN" \
  --startup-file=no \
  --project="$ENVIRONMENT" \
  -e '
    using Pkg
    Pkg.develop(PackageSpec(path=ENV["SDPX_RELEASE_SOURCE"]))
    Pkg.add([
        PackageSpec(name="MultiFloats", version="3"),
        PackageSpec(name="DoubleFloats", version="1"),
        PackageSpec(name="JLD2"),
        PackageSpec(name="JuMP", version="1"),
        PackageSpec(name="StableRNGs"),
    ])
    Pkg.instantiate()
    Pkg.precompile()
    Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
  '

sha256sum \
  "$ENVIRONMENT/Project.toml" \
  "$ENVIRONMENT/Manifest.toml" \
  > "$RELEASE/metadata/environment-sha256.txt"
```

The explicit packages include optional backends and the complete test target.
Do not copy either manifest from a historical SDPX directory: those projects
have a different package UUID and dependency graph.

After this step, compute jobs should set `JULIA_PKG_OFFLINE=true`. If an
offline package load fails, return to the login node and complete the
environment; do not run `Pkg.add` from a compute node.

## External benchmark data

Store large inputs outside both Git and the release source. Name immutable
inputs by their SHA-256 digest when practical.

The validated `J/K/N_a/N_mu = 40/4/20/100` Float64x4 input has:

```text
size:   614325622 bytes
sha256: 8622330ad83989a3c183e82b7b0cedc5bbcd1572e96c35e8cc19750129dc4d4a
```

Its canonical cluster path is:

```text
$SDPX_REMOTE_BASE/data/csdr-40-4-20-100/8622330ad83989a3c183e82b7b0cedc5bbcd1572e96c35e8cc19750129dc4d4a.bin
```

Upload it to a `.part` file, verify both size and hash remotely, then rename it
with `mv -n`. If the final path already exists, verify it instead of replacing
it.

The validated Task_Low08 Float64 input has:

```text
size:   4640880 bytes
sha256: 5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3
```

Store it at a site-selected immutable path, for example:

```text
$SDPX_REMOTE_BASE/data/task-low08/5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3.bin
```

Set `TASK_LOW08_INPUT` to that verified path before submitting validation.

## Preprocessing and model classification

Leave `presolve=:auto`, `scaling=:auto`, `formulation=:auto`, and
`chordal_decomposition=:auto` for ordinary cluster runs. The typed
preprocessing report is available at:

```julia
report = result.diagnostics.presolve.preprocessing
```

Record its input/output dimensions, extracted-bound counts, fixed-variable
count, exact equality cleanup, formulation estimate, chordal decision,
elapsed time, allocation, and warnings with the benchmark result.

Task_Low08 should report no scalar bounds, equality reduction from 482 to 394,
and rejected chordal analysis because the aggregate PSD pattern is 99.84%
dense. The medium `J/K/Na/Nmu = 32/4/16/100` CSDR model should report no
structural change. Treat any different classification as an input or release
regression before comparing solve times.

Sparse equilibration copies only active coefficient matrices and shares one
read-only empty CSC object per block. Keep the scaled-problem checksum in an
A/B result when changing this code. On the canonical medium model the
validated Float64x4 medians were 1.241 seconds and 685.8 MB before the change,
and 0.633 seconds and 269.7 MB after it.

## Thread and memory policy

Use one process per solve. The following is a conservative starting policy;
adjust PBS resources after measuring the target node:

| Arithmetic and workload | Reserved PBS cores | Julia threads per solve | BLAS threads | Strategy |
|---|---:|---:|---:|---|
| Float64x4 sparse/block-arrow | 8 | sweep 1, 2, 4, 8 | 1 | SDPX Julia block and Schur scheduling |
| Medium Float64x4 CSDR, 1,700 uniform `2x2` blocks / 144 shared variables | 128 site-specific placement reservation | 48 (re-sweep 32/48/64 after material geometry changes) | 1 | exact Julia compute pool, 10 ms worker-sleep threshold, CPUs 0--47 and NUMA nodes 0--3; do not count the unused reservation as speedup |
| BigFloat, general native | site's minimum allocation | 1 | 1 | one serial solve; use separate jobs for independent cases |
| BigFloat, exact singleton-local `2x2` arrow | 8 | sweep 1, 2, 4, 8 | 1 | native block preparation and disjoint reduced-Schur tiles may be parallel; residual/refinement remains serial |
| BigFloat, all-local `2x2` cells plus equalities | measured node allocation | sweep through 64; use 96/128 only for a larger measured panel | 1 | block/GEMV/triangular tasks cap at 64; tiled lower equality Gram may use the full width |
| BigFloat, experimental mixed arrow | 8 | sweep 1, 2, 4, 8 | 1 | the Float64x4 reduced panel/factorization is parallel; BigFloat residual/refinement remains serial |
| Task_Low08 Float64 validation | 8 | 8 | 8 | sparse assembly plus dense OpenBLAS KKT factorization |
| Task_Low08 Float64 performance on dual EPYC 7742 | 16 | 16 | 16 | OpenBLAS with `numactl --interleave=all`; measured, hardware-specific |
| Task_Low08 Float64x4 mixed KKT | 16 | 16 | 1 | lower-only sparse Schur plus owned-row refinement; request at least 64 GiB |
| Task_Low08 native BigFloat, 256 bit | 1 | 1 | 1 | request at least 128 GiB; the measured 95.720 GiB estimate is unsafe in a 64 GiB job |
| Small package validation | 8 | 4 for tests, 1 for high-precision smoke | 1 | sequential validation phases |

General native `BigFloat` uses one solver thread. Exact singleton-local `2x2`
arrows and all-local 2x2 equality-cell systems may use additional cores only
for ownership-safe block work and disjoint triangular Schur or Gram tiles.
For the all-local equality path, fine-grained MPFR phases automatically use at
most 64 ownership tasks while the equality Gram may use a wider requested
allocation. On the certified J40 BigFloat512 model, 64 workers took 368.704
seconds; uniform 128 workers took 495.811 seconds, and phase-capped 128 workers
took 425.880 seconds with the exact same certificate. A 96-worker crossover
took 398.303 seconds, so start at 64 workers for this geometry. The opt-in
mixed reduced-arrow
backend may also use those cores for its Float64x4 panel and factorization,
while exact BigFloat residual and refinement work remains serial. It must be
benchmarked rather than assumed to be faster.

The fixed BigFloat1024 support gate (job 196296) also passed at 64 workers:
157 iterations, 553.959 seconds solver time, 4,268,480 KiB peak RSS, valid
physical certificate, and no restart, regularization, refinement, or fallback.
It was slower and did not terminate with a tighter relative gap than the
512-bit run because this archived model is rounded once to Float64x4. Retain
512 bits for routine runs of this input; use 1,024 bits only when the input and
requested certificate actually carry enough precision to justify it.

Do not start a Float64x4 block-parallel run with both Julia and BLAS set to the
full allocation. Keep BLAS and OMP at one thread so the 1/2/4/8 comparison
changes only SDPX's scheduler width.

For the canonical medium CSDR geometry, use an exact compute pool and pin it
explicitly. The validated node60 launch shape is:

```bash
export JULIA_EXCLUSIVE=1
export JULIA_THREAD_SLEEP_THRESHOLD=10000000
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
numactl --physcpubind=0-47 --membind=0-3 \
  julia --threads=48,0 --gcthreads=1,0 --project=. solve_medium.jl
```

The final exact-pool sweep measured 44.112 / 22.954 / 12.450 / 6.946 / 4.275 /
3.286 / 2.967 / 3.074 / 3.540 / 3.403 seconds at 1 / 2 / 4 / 8 / 16 / 32 /
48 / 64 / 96 / 128 workers. Every requested worker was observed active and
BLAS remained at one thread. The 48-worker process reached 45.38 sampled CPU
cores, placed about 3,009 MiB across local NUMA nodes 0--3, and was 14.87x
faster than one worker. Wider pools lost time to barriers, worker wake-up, and
NUMA traffic. At 64 workers, forced NUMA interleaving was 5.4% slower than
bound local allocation. Reserve 128 cores only when required by this site's
placement policy; create the measured 48-worker runtime pool for this
144-column system. Re-run the 32/48/64 bracket when the shared dimension or
block count changes materially.

`JULIA_THREAD_SLEEP_THRESHOLD` is consumed before Julia starts. The measured
10-millisecond value removed the wide-pool wake-up collapse without the full
CPU consumption of permanent spinning. It is a site profile for exclusive
jobs, not a package default. Record process CPU time, active-worker samples,
BLAS width, RSS, affinity, NUMA placement, and wall time in every new sweep.

Use full-model warm-ups for source-to-source comparisons. On a fresh
Float64x4 process the first solve can spend about 32 seconds compiling while
the steady solve is below five seconds. The medium campaign uses three
warm-ups and at least five measured solves per process, alternates source
order, and reports the combined median. Benchmark drivers must parse the
warm-up count as a nonnegative integer; treating it as a Boolean can silently
disable warm-up when a value greater than one is requested.

For dense Task_Low08 Float64x4 runs, `mixed_precision_kkt=:on` is an expert
configuration: Float64 factorization is accepted only while target-precision
predictor and corrector residual corrections decrease monotonically. If those
corrections fail, the solver may allocate a guarded `Float64x2` factor
workspace (0.596 GiB for Task_Low08), rebuild it from the current Schur
matrix, and accept each solve only after a `Float64x4` residual check. Native
`Float64x4` factorization remains the final fallback. Keep
`extended_precision_blas=:auto`, Julia at 16 threads, and BLAS at one thread.
The 16-thread conservative planner estimate is 43.339 GiB and the measured
process peak is approximately 21 GiB; do not use 32 task-local Schur workers
in a 64 GiB request because the estimate rises to 70.120 GiB.

For dense-Schur Float64 jobs, Julia and BLAS widths are separate tuning
parameters. Task_Low08 has only 32 PSD blocks, while its `6119 x 6119` dense
Cholesky remains a level-3 LAPACK operation. The measured dual-EPYC optimum
used 16 Julia threads, 16 OpenBLAS threads, and interleaved NUMA allocation.
Thirty-two Julia threads were within 1%, while 64--128 were slower and 128
used substantially more memory.
Treat these values as a site profile, not universal solver defaults.

Set a planning ceiling below the PBS memory request:

```bash
export SDPX_MEMORY_LIMIT_BYTES=28GiB  # for a 32 GiB validation job
export SDPX_MEMORY_LIMIT_BYTES=56GiB  # for a 64 GiB benchmark job
```

SDPX also reads host and Linux cgroup memory signals, retains headroom, and may
reduce Schur workers or reject optional packing.

## Validation job

This example candidate gate uses the `normal` queue, one node, eight reserved
cores, 32 GiB, and a two-hour limit. Adapt resource values to site policy
without changing the numerical gates:

```bash
#!/bin/bash
#PBS -N sdpx_validate
#PBS -q normal
#PBS -l nodes=1:ppn=8
#PBS -l mem=32gb
#PBS -l walltime=02:00:00
#PBS -j oe

set -euo pipefail

: "${SDPX_COMMIT:?submit with qsub -v SDPX_COMMIT=<full-commit>}"
: "${SDPX_REMOTE_BASE:?submit with SDPX_REMOTE_BASE set}"
: "${TASK_LOW08_INPUT:?submit with the checksum-verified Task_Low08 path}"

BASE="$SDPX_REMOTE_BASE"
RELEASE="$BASE/releases/$SDPX_COMMIT"
SOURCE="$RELEASE/source"
ENVIRONMENT="$RELEASE/environment"
RESULT="$BASE/results/$SDPX_COMMIT/$PBS_JOBID"

mkdir -p "$RESULT"

if [ -n "${SDPX_SITE_ENV:-}" ]; then
  source "$SDPX_SITE_ENV"
fi
: "${JULIA_BIN:?set JULIA_BIN directly or through SDPX_SITE_ENV}"

export JULIA_DEPOT_PATH="$BASE/depot${JULIA_DEPOT_PATH:+:$JULIA_DEPOT_PATH}"
export JULIA_PKG_OFFLINE=true
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export SDPX_MEMORY_LIMIT_BYTES=28GiB

{
  printf 'commit=%s\n' "$SDPX_COMMIT"
  printf 'pbs_job_id=%s\n' "$PBS_JOBID"
  printf 'pbs_cores=%s\n' "${PBS_NP:-8}"
  printf 'node_class=%s\n' "${SDPX_NODE_CLASS:-unspecified}"
  printf 'pbs_memory_request=%s\n' "32gb"
  printf 'openblas_threads=%s\n' "$OPENBLAS_NUM_THREADS"
  printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
  printf 'memory_limit=%s\n' "$SDPX_MEMORY_LIMIT_BYTES"
  "$JULIA_BIN" --version
} > "$RESULT/environment.txt"

"$JULIA_BIN" \
  --project="$ENVIRONMENT" \
  --startup-file=no \
  -e 'using Pkg; Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)' \
  > "$RESULT/package-status.txt"

"$JULIA_BIN" \
  --project="$ENVIRONMENT" \
  --startup-file=no \
  -t 4 \
  -e 'using Pkg; Pkg.test("SDPX"; coverage=false)' \
  2>&1 | tee "$RESULT/tests.log"

"$JULIA_BIN" \
  --project="$ENVIRONMENT" \
  --startup-file=no \
  -t 1 \
  "$SOURCE/bench/extended_precision_blas/small_solve_validation.jl" \
  "$RESULT/small-solve.csv" \
  2>&1 | tee "$RESULT/small-solve.log"

/usr/bin/time \
  -v \
  -o "$RESULT/task-low08.time.txt" \
  "$JULIA_BIN" \
  --project="$ENVIRONMENT" \
  --startup-file=no \
  -t 4 \
  "$SOURCE/bench/lattice_bootstrap/benchmark_sdpx_float64_solve.jl" \
  "$TASK_LOW08_INPUT" \
  "$RESULT/task-low08-float64.json" \
  4 600 0.1 0.85 100 0.001 sdpb 100 1e-6 \
  2>&1 | tee "$RESULT/task-low08.log"

"$JULIA_BIN" \
  --project="$ENVIRONMENT" \
  --startup-file=no \
  -t 8 \
  "$SOURCE/bench/threading/benchmark_sparse_schur_scheduler.jl" \
  "$RESULT/sparse-threading.csv" \
  2>&1 | tee "$RESULT/sparse-threading.log"

find "$RESULT" \
  -type f \
  ! -name artifact-sha256.txt \
  -exec sha256sum {} + \
  | sort \
  > "$RESULT/artifact-sha256.txt"
```

Submit from a release-specific job directory so PBS's default output filename
is unique:

```bash
cd "$SDPX_REMOTE_BASE/jobs/$COMMIT"
PBS_VARS="SDPX_COMMIT=$COMMIT,SDPX_REMOTE_BASE=$SDPX_REMOTE_BASE"
PBS_VARS="$PBS_VARS,TASK_LOW08_INPUT=$TASK_LOW08_INPUT"
if [ -n "${SDPX_SITE_ENV:-}" ]; then
  PBS_VARS="$PBS_VARS,SDPX_SITE_ENV=$SDPX_SITE_ENV"
fi
qsub -v "$PBS_VARS" validate.pbs
```

Do not log `env`, `set`, `export -p`, or `printenv`. Those commands can capture
unrelated credentials or tokens inherited by a scheduler job. The template
records only explicitly approved reproducibility fields.
`SDPX_NODE_CLASS` should be a public, non-identifying label such as
`zen4-64c`; do not substitute a hostname. Keep detailed hostname, `lscpu`,
NUMA, and node-memory captures only in private campaign storage, not in public
result artifacts.

## Candidate acceptance gates

Inspect the PBS exit status, complete log, structured result files, and peak
RSS. Queue state alone is not validation.

The candidate may be promoted only when:

- the full test suite exits successfully;
- the analytic Float64x4 and 256-bit BigFloat solves in both kernel modes
  return `Optimal`;
- analytic objective relative error is at most `1e-18` for Float64x4 and
  `1e-28` for BigFloat;
- analytic primal and dual residuals, relative gap, and minimum PSD eigenvalue
  remain within the tolerances recorded by the driver;
- Task_Low08 returns `Optimal` with
  `abs(objective - 0.653291393898) <= 1e-6`;
- Task_Low08 has relative gap, primal residual, and dual residual at most
  `1e-6`;
- Task_Low08 maximum equality residual is at most `1e-9`;
- the original-coordinate certificate reports `primal_psd.ok` and
  `dual_psd.ok`; record raw minimum eigenvalues as diagnostics, but do not
  replace the solver's scale-aware `tolerance * max(norm(block, Inf), 1)` PSD
  allowance with a fixed absolute threshold;
- the sparse scheduler benchmark reports no meaningful Schur error and no
  unexpected thread or memory selection;
- no phase exceeds its memory request or reports an ownership/aliasing
  warning.

The local release-candidate suite contained 1,272 passing assertions with four
Julia threads and no failures when this workflow was written. Treat exit
status and the tests present in the candidate commit as authoritative; a
later commit may legitimately change the count.

The Float64 Task_Low08 gate must pass before any extended-precision lattice
run. A successful small BigFloat solve or Schur benchmark is not evidence that
a full BigFloat Task_Low08 solve fits memory or has completed.

The final `c9d6514` cross-problem gate (job 196277, node58) used 16 Julia and
16 OpenBLAS threads inside a 64-core reservation. It returned `Optimal` in 28
iterations: primal/dual objectives `0.6532912655025964` /
`0.6532910479425099`, relative gap `2.176e-7`, primal/dual residuals
`3.316e-10` / `9.534e-12`, and maximum original equality residual
`3.316e-10`. The original-coordinate certificate was valid; minimum primal
and dual PSD eigenvalues were `-7.126e-11` and `1.975e-15`. Solver and driver
times were 33.846 and 35.585 seconds, and `/usr/bin/time` recorded 4,469,000
KiB peak RSS. This measurement is a correctness gate for the BigFloat
scheduler change; it is not a replacement for the warmed backend comparison.

## Sparse 1/2/4/8 and BigFloat benchmark job

After the validation gate, use a separate `normal`-queue job with one node,
eight cores, 64 GiB, and up to eight hours. Run the Float64x4
`40/4/20/100` Schur benchmark sequentially with Julia `-t 1`, `-t 2`,
`-t 4`, and `-t 8`; keep BLAS and OMP at one. Measure both
`extended_precision_blas=:off` and `:auto`, use at least three timed
repetitions, and retain sampled Schur errors and peak RSS.

The body of the benchmark job can use:

```bash
: "${SDPX_REMOTE_BASE:?set SDPX_REMOTE_BASE}"
INPUT="$SDPX_REMOTE_BASE/data/csdr-40-4-20-100/8622330ad83989a3c183e82b7b0cedc5bbcd1572e96c35e8cc19750129dc4d4a.bin"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export SDPX_MEMORY_LIMIT_BYTES=56GiB

for threads in 1 2 4 8; do
  for mode in off auto; do
    /usr/bin/time \
      -v \
      -o "$RESULT/float64x4-${threads}t-${mode}.time.txt" \
      "$JULIA_BIN" \
      --project="$ENVIRONMENT" \
      --startup-file=no \
      -t "$threads" \
      "$SOURCE/bench/extended_precision_blas/benchmark_schur.jl" \
      --problem=sparse \
      --input="$INPUT" \
      --output="$RESULT/float64x4-schur.csv" \
      --arithmetic=float64x4 \
      --mode="$mode" \
      --repetitions=3 \
      --validation-samples=128 \
      --memory-fraction=0.10 \
      2>&1 | tee "$RESULT/float64x4-${threads}t-${mode}.log"
  done
done
```

Run the BigFloat comparison in the same allocation only with Julia `-t 1`:

```bash
export SPARSE_INPUT="$SDPX_REMOTE_BASE/data/csdr-40-4-20-100/8622330ad83989a3c183e82b7b0cedc5bbcd1572e96c35e8cc19750129dc4d4a.bin"
export BIGFLOAT_BITS=256
export BENCH_SAMPLES=3
export BENCH_OUTPUT="$RESULT/bigfloat-sparse-schur.csv"

/usr/bin/time \
  -v \
  -o "$RESULT/bigfloat-sparse-schur.time.txt" \
  "$JULIA_BIN" \
  --project="$ENVIRONMENT" \
  --startup-file=no \
  -t 1 \
  "$SOURCE/bench/bigfloat_sparse_schur/benchmark.jl" \
  2>&1 | tee "$RESULT/bigfloat-sparse-schur.log"
```

For exact sparse `2x2` block-arrow problems, the current runtime uses the fused
compute-and-scatter kernel and bypasses packed Gram panels. Historical packed
BigFloat measurements are useful implementation history, but they are not a
substitute for measuring the current fused path on the target node.

The historical `80/4/40/100` input is a later scale-up case, not a replacement
for the requested `40/4/20/100` gate. Run it only after the smaller target
passes and request at least 128 GiB with a longer wall-time. Verify its input
hash before use and keep all output under the new commit's result directory.
The documented artifact has:

```text
size:   2353958026 bytes
sha256: ed65290d3517ef24b334abca8b8043b7b7fadbf6911ab02c904113e364203e42
```

Store it under a checksum-named site-selected path such as:

```text
$SDPX_REMOTE_BASE/data/csdr-80-4-40-100/ed65290d3517ef24b334abca8b8043b7b7fadbf6911ab02c904113e364203e42.bin
```

## Promotion

Do not move `current` when a job is merely queued or running. After all
acceptance gates pass, record the previous target and atomically replace only
the symlink:

```bash
: "${SDPX_REMOTE_BASE:?set SDPX_REMOTE_BASE}"
BASE="$SDPX_REMOTE_BASE"
RELEASE="$BASE/releases/$COMMIT"

cd "$BASE"

if [ -L current ]; then
  readlink current > "$RELEASE/metadata/previous-current.txt"
elif [ -e current ]; then
  printf 'current exists but is not a symlink\n' >&2
  exit 1
else
  printf 'none\n' > "$RELEASE/metadata/previous-current.txt"
fi

ln -s "releases/$COMMIT/source" ".current-$COMMIT"
mv -Tf ".current-$COMMIT" current
readlink -f current
```

Once active, make the archived source read-only:

```bash
chmod -R a-w "$RELEASE/source"
```

Jobs should continue to use their explicit release path, not `current`, so a
later promotion cannot change a running campaign.

## Rollback

Rollback never deletes the failed release or its diagnostics. Read the
candidate's recorded previous target, verify that it still exists, and replace
the symlink atomically:

```bash
: "${SDPX_REMOTE_BASE:?set SDPX_REMOTE_BASE}"
BASE="$SDPX_REMOTE_BASE"
FAILED_COMMIT=<full-commit>
PREVIOUS="$(
  cat "$BASE/releases/$FAILED_COMMIT/metadata/previous-current.txt"
)"

test "$PREVIOUS" != none
test -d "$BASE/$PREVIOUS"

cd "$BASE"
ln -s "$PREVIOUS" ".current-rollback-$FAILED_COMMIT"
mv -Tf ".current-rollback-$FAILED_COMMIT" current
readlink -f current
```

Preserve the failed candidate, PBS output, structured results, input hashes,
and environment manifest for diagnosis.

## Result collection

Copy a completed result directory back without deleting the cluster original:

```bash
: "${CLUSTER:?set CLUSTER to the PBS login SSH alias}"
: "${SDPX_REMOTE_BASE:?set SDPX_REMOTE_BASE}"
rsync -av \
  "$CLUSTER:$SDPX_REMOTE_BASE/results/$COMMIT/$JOB_ID/" \
  "./results/$COMMIT/$JOB_ID/"
```

Verify `artifact-sha256.txt` after transfer. Keep raw multi-gigabyte inputs,
Julia depots, manifests created for cluster campaigns, and machine-specific
result directories outside Git. Commit reusable drivers and concise English
summaries only.
