#!/usr/bin/env bash
set -euo pipefail
if [ -z "${SDPX_HOTSPOT_MANIFEST:-}" ]; then
  for arg in "$@"; do
    case "$arg" in --manifest=*) export SDPX_HOTSPOT_MANIFEST="${arg#--manifest=}" ;; esac
  done
fi
: "${SDPX_HOTSPOT_MANIFEST:?set SDPX_HOTSPOT_MANIFEST or pass --manifest=...}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# MultiFloats on Julia 1.12 requires a single GC thread for safe reproducibility.
exec julia --startup-file=no --gcthreads=1 --project="${SDPX_PROJECT:-$ROOT}" \
  "$ROOT/benchmark/optimization/measure_target.jl" "$@"
