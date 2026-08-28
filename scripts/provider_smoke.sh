#!/usr/bin/env bash
set -euo pipefail

# Explicit installed-provider smoke for MultiFloatLinearAlgebra (MFLA) and
# BigFloatLinearAlgebra (BFLA).  This is intentionally separate from
# `Pkg.test`: MFLA is unregistered, so the ordinary test environment must
# never clone it, and the smoke must run against the real installed/local
# provider checkouts.
#
# Environment:
#   SDPX_MFLA_PROJECT  path to the MultiFloatLinearAlgebra checkout
#   SDPX_BFLA_PROJECT  path to the BigFloatLinearAlgebra checkout
#   SDPX_PROVIDER_SMOKE_ENV  optional pre-created Julia environment; when
#                            unset a temporary environment is used and removed

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_PROVIDERS_ROOT="$(cd "$ROOT/.." && pwd)"

: "${SDPX_MFLA_PROJECT:=$LOCAL_PROVIDERS_ROOT/MultiFloatLinearAlgebra.jl}"
: "${SDPX_BFLA_PROJECT:=$LOCAL_PROVIDERS_ROOT/BigFloatLinearAlgebra.jl}"

if [ ! -d "$SDPX_MFLA_PROJECT" ]; then
  echo "MFLA checkout not found at $SDPX_MFLA_PROJECT (set SDPX_MFLA_PROJECT)" >&2
  exit 1
fi
if [ ! -d "$SDPX_BFLA_PROJECT" ]; then
  echo "BFLA checkout not found at $SDPX_BFLA_PROJECT (set SDPX_BFLA_PROJECT)" >&2
  exit 1
fi

SMOKE_ENV="${SDPX_PROVIDER_SMOKE_ENV:-}"
KEEP_ENV=0
if [ -z "$SMOKE_ENV" ]; then
  SMOKE_ENV="$(mktemp -d "${TMPDIR:-/tmp}/sdpx-provider-smoke.XXXXXX")"
  KEEP_ENV=1
fi
mkdir -p "$SMOKE_ENV/depot"

cleanup() {
  if [ "$KEEP_ENV" -eq 1 ]; then
    rm -rf "$SMOKE_ENV"
  fi
}
trap cleanup EXIT

export SDPX_MFLA_PROJECT
export SDPX_BFLA_PROJECT
export JULIA_DEPOT_PATH="$SMOKE_ENV/depot:${JULIA_DEPOT_PATH:-$HOME/.julia}"
export JULIA_PKG_OFFLINE=true

julia --startup-file=no --project="$SMOKE_ENV" -e '
using Pkg
Pkg.develop(path=ARGS[1])
Pkg.develop(path=ARGS[2])
Pkg.develop(path=ARGS[3])
Pkg.add(["MultiFloats", "GenericLinearAlgebra"])
' "$ROOT" "$SDPX_MFLA_PROJECT" "$SDPX_BFLA_PROJECT"

julia --startup-file=no --project="$SMOKE_ENV" -t1 \
  "$ROOT/validation/providers/provider_smoke.jl"

echo "provider smoke completed: mfla=$SDPX_MFLA_PROJECT bfla=$SDPX_BFLA_PROJECT"
