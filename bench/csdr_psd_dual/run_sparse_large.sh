#!/bin/bash

# Larger CSDR benchmark: Clarabel, SDPX dense 1-thread, and incidence-aware
# SDPX sparse at 1/2/4/8 threads. Tiny 2x2 blocks may not amortize task overhead.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
REFERENCE_ROOT="${CLARABEL_CSDR_ROOT:?set CLARABEL_CSDR_ROOT to the reference-data directory}"
DEPOT_HEAD="${JULIA_DEPOT_HEAD:-$(cd "$REPO/.." && pwd)/.julia-depot}"
RUN_DIR="${RUN_DIR:-$HERE/results/$(date -u +%Y%m%dT%H%M%SZ)-sparse-large}"
REPS="${REPS:-3}"
TOL="${TOL:-1e-7}"
BETA="${BETA:-0.1}"
GAMMA="${GAMMA:-0.85}"
REFINE_STEPS="${REFINE_STEPS:-1}"

mkdir -p "$RUN_DIR" "$DEPOT_HEAD"
DATA="$RUN_DIR/problem.bin"
CSV="$RUN_DIR/timings.csv"

export CLARABEL_CSDR_ROOT="$REFERENCE_ROOT"
export JULIA_DEPOT_PATH="$DEPOT_HEAD:${JULIA_DEPOT_PATH:-$HOME/.julia}"
export JULIA_LOAD_PATH="$REPO:$REFERENCE_ROOT/julia:@stdlib"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CSDR_J="${CSDR_J:-4}"
export CSDR_K="${CSDR_K:-1}"
export CSDR_NA="${CSDR_NA:-2}"
export CSDR_NMU="${CSDR_NMU:-60}"

"$JULIA_BIN" --startup-file=no -t 1 \
  "$HERE/prepare_problem.jl" "$DATA" |
  tee "$RUN_DIR/prepare.log"

"$JULIA_BIN" --startup-file=no -t 1 \
  "$HERE/benchmark_solver.jl" \
  --solver=clarabel --input="$DATA" --output="$CSV" \
  --reps="$REPS" --tol="$TOL" |
  tee "$RUN_DIR/clarabel.log"

"$JULIA_BIN" --startup-file=no -t 1 \
  "$HERE/benchmark_solver.jl" \
    --solver=sdpx --input="$DATA" --output="$CSV" \
    --reps="$REPS" --tol="$TOL" --sparse=0 --equilibrate=1 \
    --beta="$BETA" --gamma="$GAMMA" --refine-steps="$REFINE_STEPS" |
  tee "$RUN_DIR/sdpx-dense-1t.log"

for threads in 1 2 4 8; do
  "$JULIA_BIN" --startup-file=no -t "$threads" \
    "$HERE/benchmark_solver.jl" \
      --solver=sdpx --input="$DATA" --output="$CSV" \
      --reps="$REPS" --tol="$TOL" --sparse=1 --equilibrate=0 \
      --beta="$BETA" --gamma="$GAMMA" --refine-steps="$REFINE_STEPS" |
    tee "$RUN_DIR/sdpx-sparse-${threads}t.log"
done

echo "Benchmark complete: $CSV"
