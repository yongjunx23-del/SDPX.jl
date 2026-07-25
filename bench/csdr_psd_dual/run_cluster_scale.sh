#!/bin/bash

# Scaled sparse CSDR benchmark for a workstation or one multi-core cluster node.
# The Julia threads are the only computational threads: BLAS and OMP stay at 1.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
: "${CLARABEL_CSDR_ROOT:?Set CLARABEL_CSDR_ROOT to the reference CSDR checkout}"

RUN_DIR="${RUN_DIR:-$HERE/results/$(date -u +%Y%m%dT%H%M%SZ)-cluster-scale}"
THREAD_LIST="${THREAD_LIST:-1 2 4 8}"
REPS="${REPS:-3}"
TOL="${TOL:-1e-7}"
RUN_CLARABEL="${RUN_CLARABEL:-1}"
AVAILABLE_CPUS="${SLURM_CPUS_PER_TASK:-999999}"
DEPOT_HEAD="${JULIA_DEPOT_HEAD:-$(cd "$REPO/.." && pwd)/.julia-depot}"

mkdir -p "$RUN_DIR" "$DEPOT_HEAD"
DATA="$RUN_DIR/problem.bin"
CSV="$RUN_DIR/timings.csv"

export JULIA_DEPOT_PATH="$DEPOT_HEAD${JULIA_DEPOT_PATH:+:$JULIA_DEPOT_PATH}"
export JULIA_LOAD_PATH="$REPO:$CLARABEL_CSDR_ROOT/julia:@stdlib"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CSDR_J="${CSDR_J:-8}"
export CSDR_K="${CSDR_K:-2}"
export CSDR_NA="${CSDR_NA:-4}"
export CSDR_NMU="${CSDR_NMU:-120}"

"$JULIA_BIN" --startup-file=no -t 1 \
  "$HERE/prepare_problem.jl" "$DATA" |
  tee "$RUN_DIR/prepare.log"

if [[ "$RUN_CLARABEL" == "1" ]]; then
  "$JULIA_BIN" --startup-file=no -t 1 \
    "$HERE/benchmark_solver.jl" \
      --solver=clarabel --input="$DATA" --output="$CSV" \
      --reps="$REPS" --tol="$TOL" |
    tee "$RUN_DIR/clarabel.log"
fi

for threads in $THREAD_LIST; do
  if (( threads > AVAILABLE_CPUS )); then
    continue
  fi
  "$JULIA_BIN" --startup-file=no -t "$threads" \
    "$HERE/benchmark_solver.jl" \
      --solver=sdpx --input="$DATA" --output="$CSV" \
      --reps="$REPS" --tol="$TOL" \
      --sparse=1 --equilibrate=0 \
      --parameter-policy=auto --step-rule=backtrack \
      --refine-steps=1 |
    tee "$RUN_DIR/sdpx-${threads}t.log"
done

echo "Benchmark complete: $CSV"
