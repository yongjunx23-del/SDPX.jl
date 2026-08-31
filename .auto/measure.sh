#!/bin/bash
set -euo pipefail
# CSDR alpha3 4-thread solve benchmark. Emits METRIC lines for autoresearch.
# Env: PRECOND_VARIANT (baseline|colnorm|elim|ruiz), PRECOND_SCALING (auto|none|equilibrate)
cd "$(dirname "$0")/.."
export JULIA_DEPOT_PATH="$(pwd)/.auto/depot:$HOME/Desktop/project/SDPX/.julia-depot:$HOME/.julia"
export JULIA_NUM_THREADS=4
VARIANT="${PRECOND_VARIANT:-baseline}"
SCALING="${PRECOND_SCALING:-auto}"
# fast pre-check: input exists and checksum matches
INPUT=/tmp/csdr-alpha9-twice/solve-alpha3.bin
[ -f "$INPUT" ] || { echo "missing $INPUT" >&2; exit 1; }
EXPECTED=2e7bac1da3aa0fdf441eb08ec105c9c90397c482d78173ee6c41b030c53f97d7
ACTUAL=$(shasum -a 256 "$INPUT" | cut -d' ' -f1)
[ "$ACTUAL" = "$EXPECTED" ] || { echo "input checksum mismatch" >&2; exit 1; }
PRECOND_VARIANT="$VARIANT" PRECOND_SCALING="$SCALING" \
  julia --startup-file=no --project=.auto/csdr-env benchmark/autoresearch/csdr_alpha3_precond.jl
