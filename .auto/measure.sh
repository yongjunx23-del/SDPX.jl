#!/bin/bash
set -euo pipefail
cd /tmp/sdpx-opt-20260910
JULIA=/Users/xuyongjun/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# CSDR alpha3 Float64x4 (primary): median of 3 guarded solves
out=$($JULIA --startup-file=no --threads=4 --gcthreads=1 --heap-size-hint=4G \
  --project=/tmp/sdpx-scientific-core-env-20260907 \
  benchmark/autoresearch/csdr_alpha3_x4.jl 2>&1) || { echo "CSDR_RUN_FAILED"; echo "$out" | tail -5; exit 1; }
echo "$out" | grep -E "^METRIC" || { echo "NO_METRICS"; echo "$out" | tail -5; exit 1; }
echo "$out" | grep -E "^METRIC|^CSDR"

# Float64 LP general catalog (secondary): single tier-small LP
$JULIA --startup-file=no --threads=1 --project=/tmp/sdpx-scientific-core-env-20260907 \
  .auto/lp_bench.jl 2>&1 | grep -E "^METRIC" || echo "LP_METRICS_FAILED"
