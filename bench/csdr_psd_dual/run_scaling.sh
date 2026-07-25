#!/bin/bash

# Reproducible 1/2/4/8-thread Float64x4 comparison on one serialized CSDR
# reduced-direct PSD dual. OpenBLAS/OMP stay at one thread so the only changing
# resource is SDPX's Julia block parallelism.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
REFERENCE_ROOT="${CLARABEL_CSDR_ROOT:?set CLARABEL_CSDR_ROOT to the reference-data directory}"
DEPOT_HEAD="${JULIA_DEPOT_HEAD:-$(cd "$REPO/.." && pwd)/.julia-depot}"
RUN_DIR="${RUN_DIR:-$HERE/results/$(date -u +%Y%m%dT%H%M%SZ)}"
REPS="${REPS:-3}"
TOL="${TOL:-1e-7}"
SDPX_EQUILIBRATE="${SDPX_EQUILIBRATE:-1}"
SDPX_BETA="${SDPX_BETA:-0.01}"
SDPX_OMEGA_P="${SDPX_OMEGA_P:-10}"
SDPX_OMEGA_D="${SDPX_OMEGA_D:-10}"
SDPX_PREDICTOR="${SDPX_PREDICTOR:-sdpb}"
SDPX_MAX_RESTARTS="${SDPX_MAX_RESTARTS:-10}"

mkdir -p "$RUN_DIR" "$DEPOT_HEAD"
DATA="$RUN_DIR/problem.bin"
CSV="$RUN_DIR/timings.csv"

export CLARABEL_CSDR_ROOT="$REFERENCE_ROOT"
export JULIA_DEPOT_PATH="$DEPOT_HEAD:${JULIA_DEPOT_PATH:-$HOME/.julia}"
export JULIA_LOAD_PATH="$REPO:$REFERENCE_ROOT/julia:@stdlib"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"$JULIA_BIN" --startup-file=no -t 1 \
  "$HERE/prepare_problem.jl" "$DATA" |
  tee "$RUN_DIR/prepare.log"

"$JULIA_BIN" --startup-file=no -t 1 \
  "$HERE/benchmark_solver.jl" \
  --solver=clarabel --input="$DATA" --output="$CSV" \
  --reps="$REPS" --tol="$TOL" |
  tee "$RUN_DIR/clarabel.log"

for threads in 1 2 4 8; do
  "$JULIA_BIN" --startup-file=no -t "$threads" \
    "$HERE/benchmark_solver.jl" \
    --solver=sdpx --input="$DATA" --output="$CSV" \
    --reps="$REPS" --tol="$TOL" \
    --equilibrate="$SDPX_EQUILIBRATE" \
    --beta="$SDPX_BETA" \
    --omega-p="$SDPX_OMEGA_P" --omega-d="$SDPX_OMEGA_D" \
    --predictor="$SDPX_PREDICTOR" \
    --max-restarts="$SDPX_MAX_RESTARTS" |
    tee "$RUN_DIR/sdpx-${threads}t.log"
done

echo "Benchmark complete: $CSV"
