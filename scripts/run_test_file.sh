#!/usr/bin/env bash
# Run one or more SDPX test files against a persistent test environment so the
# edit-test loop and the self-loop gate step are reproducible and fast.
#
# Usage:
#   scripts/run_test_file.sh test/hsd_certificates.jl
#   scripts/run_test_file.sh test/hsd_certificates.jl test/hsd_product_certificates.jl
#
# Environment:
#   JULIA_DEPOT_PATH   defaults to /Users/xuyongjun/Desktop/project/SDPX/.julia-depot
#   SDPX_TEST_ENV      persistent env dir (default /tmp/sdpx-testenv)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${JULIA_DEPOT_PATH:=/Users/xuyongjun/Desktop/project/SDPX/.julia-depot}"
: "${SDPX_TEST_ENV:=/tmp/sdpx-testenv}"
[ "$#" -ge 1 ] || { echo "usage: $0 <test-file> [...]" >&2; exit 2; }
for f in "$@"; do
  [ -f "$ROOT/$f" ] || { echo "test file not found: $ROOT/$f" >&2; exit 2; }
done

mkdir -p "$SDPX_TEST_ENV"
export JULIA_DEPOT_PATH

{
  echo "using Pkg"
  echo "Pkg.activate(\"$SDPX_TEST_ENV\")"
  echo "Pkg.develop(path=\"$ROOT\")"
  echo 'Pkg.add(["MultiFloats","Test","LinearAlgebra","SparseArrays"])'
  echo "using SDPX, Test"
  for f in "$@"; do
    echo "println(\"===== $f\")"
    echo "include(\"$ROOT/$f\")"
  done
  echo 'println("ALL_DONE")'
} > "$SDPX_TEST_ENV/driver.jl"

julia "$SDPX_TEST_ENV/driver.jl" 2>&1
