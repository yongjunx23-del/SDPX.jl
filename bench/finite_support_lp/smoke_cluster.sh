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
: "${SDPX_DEPOT_PATH:?set SDPX_DEPOT_PATH to the Julia depot}"

mkdir -p "$SDPX_RESULT_DIR"
export JULIA_PKG_OFFLINE=true
export JULIA_DEPOT_PATH="$SDPX_DEPOT_PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export SDPX_WARMUP=1
export SDPX_MAX_ITERATIONS="${SDPX_MAX_ITERATIONS:-120}"
export SDPX_TIME_LIMIT="${SDPX_TIME_LIMIT:-300}"

if [ "${SDPX_MINIMAL_ENV:-0}" = "1" ]; then
  # The site image contains the solver's runtime dependencies but may not have
  # optional test-only packages from the full release environment.  Build a
  # job-local project that keeps only SDPX and MultiFloats while reusing the
  # checked-in manifest entries for their transitive dependencies.
  minimal_environment="$SDPX_RESULT_DIR/environment"
  mkdir -p "$minimal_environment"
  cp "$SDPX_ENVIRONMENT/Manifest.toml" "$minimal_environment/Manifest.toml"
  sed -i '/^path = /c\\path = "'"$SDPX_SOURCE"'"' "$minimal_environment/Manifest.toml"
  sed -i '/\[\[deps.MathOptInterface\]\]/,/^\[\[/{s/7b57dbe5d2c988a0c7a0ea977045e844e3d0b263/9f23c8c1667bd0b0e611110aaf80aa91c1bdf274/; s/version = "1.51.2"/version = "1.51.1"/}' "$minimal_environment/Manifest.toml"
  sed -i '/\[\[deps.ForwardDiff\]\]/,/^\[\[/{s/73d5084cae45f9d0857776ad78cf303fec09eb02/2c5d0b0e12088cde2cf84afb2784415b1ea3dfee/; s/version = "1.4.3"/version = "1.4.1"/}' "$minimal_environment/Manifest.toml"
  cat > "$minimal_environment/Project.toml" <<EOF
name = "SDPXFiniteSupportLP"
uuid = "d9b6ef2d-0d78-4a4f-9d2f-9fd0c4a8bf0c"
version = "0.1.0"

[deps]
SDPX = "9c19f76d-03c5-4610-b403-7c8fdd8897fd"
MultiFloats = "bdf0d083-296b-4888-a5b6-7498122e68a5"
EOF
  export SDPX_ENVIRONMENT="$minimal_environment"
fi

if [ "${SDPX_INSTANTIATE:-0}" = "1" ]; then
  # Keep package installation isolated from the shared cluster depot.  This is
  # useful when the locked manifest is newer than the preloaded site image.
  mkdir -p "$SDPX_RESULT_DIR/depot"
  export JULIA_DEPOT_PATH="$SDPX_RESULT_DIR/depot:$JULIA_DEPOT_PATH"
  unset JULIA_PKG_OFFLINE
  "$JULIA_BIN" --project="$SDPX_ENVIRONMENT" --startup-file=no \
    -e 'using Pkg; Pkg.instantiate()' \
    > "$SDPX_RESULT_DIR/instantiate.log" 2>&1
  export JULIA_PKG_OFFLINE=true
fi

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

# Run only the focused reduced-LP regression test on the compute node.  Including
# the file directly avoids invoking the full package test suite at this gate.
SDPX_BLAS_THREADS=1 "$JULIA_BIN" --project="$SDPX_ENVIRONMENT" \
  --startup-file=no -t 2 -e \
  "using SDPX; include(\"$SDPX_SOURCE/test/lp_regressions.jl\")" \
  > "$SDPX_RESULT_DIR/lp_regressions.log" 2>&1

sha256sum "$SDPX_RESULT_DIR"/*.log > "$SDPX_RESULT_DIR/SHA256SUMS"
touch "$SDPX_RESULT_DIR/PASSED"
