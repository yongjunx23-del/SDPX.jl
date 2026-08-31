#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
PROJECT="${SDPX_PRECISION_PROJECT:-$ROOT/benchmark/general/precision_env}"
THREADS="${SDPX_PRECISION_THREADS:-1}"
GC_THREADS="${SDPX_PRECISION_GC_THREADS:-1}"
PRECISIONS="${SDPX_PRECISIONS:-Float64 Float64x2 Float64x3 Float64x4 BigFloat256 BigFloat512 BigFloat1024}"
DEFAULT_IDS="lp_afiro_style socp_portfolio_small socp_ill_scaled_small rsoc_epigraph_small sdp_maxcut_k4 exp_unit_small power_epigraph_small mixed_orthant_exp_small"
IDS="${SDPX_PRECISION_IDS:-$DEFAULT_IDS}"
IDS="${IDS//,/ }"
status=0
for precision in $PRECISIONS; do
  for id in $IDS; do
    echo "=== PRECISION $precision CASE $id ==="
    if ! SDPX_PRECISION="$precision" SDPX_PRECISION_IDS="$id" \
        SDPX_PRECISION_THREADS="$THREADS" \
        "$JULIA_BIN" --startup-file=no --gcthreads="$GC_THREADS" \
        --threads="$THREADS" --project="$PROJECT" \
        "$ROOT/benchmark/general/precision_matrix.jl"; then
      status=1
    fi
  done
done
exit "$status"
