#!/bin/bash
#PBS -N sdpx_finite_lp
#PBS -q normal
#PBS -l nodes=1:ppn=64
#PBS -l mem=64gb
#PBS -l walltime=08:00:00
#PBS -j oe

set -euo pipefail

: "${SDPX_SOURCE:?set SDPX_SOURCE to an immutable release source directory}"
: "${SDPX_ENVIRONMENT:?set SDPX_ENVIRONMENT to the matching Julia environment}"
: "${SDPX_MODEL_DIR:?set SDPX_MODEL_DIR to the checksum-verified model directory}"
: "${SDPX_RESULT_DIR:?set SDPX_RESULT_DIR to a unique result directory}"

if [ -n "${SDPX_SITE_ENV:-}" ]; then
  source "$SDPX_SITE_ENV"
fi
: "${JULIA_BIN:?set JULIA_BIN directly or through SDPX_SITE_ENV}"
: "${SDPX_DEPOT_PATH:?set SDPX_DEPOT_PATH directly or through SDPX_SITE_ENV}"

mkdir -p "$SDPX_RESULT_DIR"
SDPX_BENCH_SOURCE="${SDPX_BENCH_SOURCE:-$SDPX_SOURCE}"
export JULIA_PKG_OFFLINE=true
export JULIA_DEPOT_PATH="$SDPX_DEPOT_PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export SDPX_WARMUP=1
export SDPX_MAX_ITERATIONS="${SDPX_MAX_ITERATIONS:-600}"
export SDPX_TIME_LIMIT="${SDPX_TIME_LIMIT:-1800}"

PRIMARY_MODEL="$SDPX_MODEL_DIR/all_spins_fixed_y_mu12_1_min.txt"
SECONDARY_MODEL="$SDPX_MODEL_DIR/all_spins_fixed_y_mu1_1_min.txt"
CONTROL_MODEL="$SDPX_MODEL_DIR/low_plus_tail_fixed_y_mu1_1_min.txt"

{
  printf 'pbs_job_id=%s\n' "$PBS_JOBID"
  printf 'host=%s\n' "$(hostname)"
  printf 'reserved_cores=%s\n' "${PBS_NP:-64}"
  printf 'source=%s\n' "$SDPX_SOURCE"
  printf 'benchmark_source=%s\n' "$SDPX_BENCH_SOURCE"
  printf 'environment=%s\n' "$SDPX_ENVIRONMENT"
  printf 'model_directory=%s\n' "$SDPX_MODEL_DIR"
  taskset -pc $$ || true
  numactl --show 2>/dev/null || true
  lscpu | grep -E 'Model name|Socket|NUMA|CPU\(s\)' || true
  "$JULIA_BIN" --version
} > "$SDPX_RESULT_DIR/environment.txt"

run_one() {
  local model="$1"
  local arithmetic="$2"
  local threads="$3"
  local direction="$4"
  local label="$5"
  local repetitions="${6:-1}"
  local blas_threads=1
  if [ "$arithmetic" = "Float64" ]; then
    blas_threads="$threads"
  fi
  local repetition
  for repetition in $(seq 1 "$repetitions"); do
    local output="$SDPX_RESULT_DIR/${label}_${arithmetic}_t${threads}_${direction}_r${repetition}.log"
    SDPX_BLAS_THREADS="$blas_threads" \
    /usr/bin/time -v \
      "$JULIA_BIN" \
      --project="$SDPX_ENVIRONMENT" \
      --startup-file=no \
      -t "$threads" \
      "$SDPX_BENCH_SOURCE/bench/finite_support_lp/benchmark.jl" \
      "$model" "$arithmetic" "$direction" "$threads" \
      > "$output" 2>&1
  done
}

# A cheap correctness gate catches parser, precision, and reduced-system errors
# before the primary model consumes significant node time.
for arithmetic in Float64 Float64x4 BigFloat256; do
  run_one "$CONTROL_MODEL" "$arithmetic" 1 min control 1
done

for arithmetic in Float64 Float64x4 BigFloat256; do
  for threads in 1 4 8; do
    run_one "$SECONDARY_MODEL" "$arithmetic" "$threads" min secondary 1
  done
done

for threads in 1 2 4 8 16; do
  run_one "$PRIMARY_MODEL" Float64 "$threads" min primary 3
  run_one "$PRIMARY_MODEL" Float64 "$threads" max primary 3
done

for threads in 1 2 4 8 16; do
  run_one "$PRIMARY_MODEL" Float64x4 "$threads" min primary 2
  run_one "$PRIMARY_MODEL" Float64x4 "$threads" max primary 2
done

for threads in 1 2 4 8; do
  run_one "$PRIMARY_MODEL" BigFloat256 "$threads" min primary 1
  run_one "$PRIMARY_MODEL" BigFloat256 "$threads" max primary 1
done

sha256sum "$SDPX_RESULT_DIR"/*.log > "$SDPX_RESULT_DIR/SHA256SUMS"
touch "$SDPX_RESULT_DIR/PASSED"
