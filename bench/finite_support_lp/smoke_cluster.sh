#!/bin/bash
#PBS -N sdpx_finite_lp_smoke
#PBS -q normal
#PBS -l nodes=1:ppn=8
#PBS -l mem=16gb
#PBS -l walltime=01:30:00
#PBS -j oe

set -euo pipefail

: "${SDPX_SOURCE:?set SDPX_SOURCE to an immutable release source directory}"
: "${SDPX_ENVIRONMENT:?set SDPX_ENVIRONMENT to the matching Julia environment}"
: "${SDPX_MODEL_DIR:?set SDPX_MODEL_DIR to the staged model directory}"
: "${SDPX_RESULT_DIR:?set SDPX_RESULT_DIR to a unique result directory}"
: "${JULIA_BIN:?set JULIA_BIN to the cluster Julia executable}"

mkdir -p "$SDPX_RESULT_DIR"
export JULIA_PKG_OFFLINE=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export SDPX_WARMUP=1
export SDPX_MAX_ITERATIONS="${SDPX_MAX_ITERATIONS:-120}"
export SDPX_TIME_LIMIT="${SDPX_TIME_LIMIT:-300}"

{
  printf 'pbs_job_id=%s\n' "$PBS_JOBID"
  printf 'host=%s\n' "$(hostname)"
  printf 'source=%s\n' "$SDPX_SOURCE"
  printf 'environment=%s\n' "$SDPX_ENVIRONMENT"
  printf 'model_directory=%s\n' "$SDPX_MODEL_DIR"
  taskset -pc $$ || true
  numactl --show 2>/dev/null || true
  lscpu | grep -E 'Model name|Socket|NUMA|CPU\(s\)' || true
  "$JULIA_BIN" --version
} > "$SDPX_RESULT_DIR/environment.txt"

run_one() {
  local arithmetic="$1"
  local output="$SDPX_RESULT_DIR/control_${arithmetic}.log"
  SDPX_BLAS_THREADS=1 /usr/bin/time -v \
    "$JULIA_BIN" --project="$SDPX_ENVIRONMENT" --startup-file=no -t 1 \
    "$SDPX_SOURCE/bench/finite_support_lp/benchmark.jl" \
    "$SDPX_MODEL_DIR/low_plus_tail_fixed_y_mu1_1_min.txt" \
    "$arithmetic" min 1 > "$output" 2>&1
}

for arithmetic in Float64 Float64x4 BigFloat256; do
  run_one "$arithmetic"
done

# Run only the focused reduced-LP regression test on the compute node.
SDPX_BLAS_THREADS=1 "$JULIA_BIN" --project="$SDPX_ENVIRONMENT" \
  --startup-file=no -t 2 -e \
  'using Pkg; Pkg.test("SDPX"; test_args=["lp_regressions.jl"], coverage=false)' \
  > "$SDPX_RESULT_DIR/lp_regressions.log" 2>&1

sha256sum "$SDPX_RESULT_DIR"/*.log > "$SDPX_RESULT_DIR/SHA256SUMS"
touch "$SDPX_RESULT_DIR/PASSED"
