#!/bin/bash
# Login-node bootstrap for the parameter-provenance Stage-1 independent Julia
# environment.  Only instantiates and precompiles the env named by
# BOOTSTRAP_ENV.  Compute nodes have no package network access, so this must
# run on a login node.
set -euo pipefail

: "${BOOTSTRAP_ENV:?set BOOTSTRAP_ENV to the independent Julia environment}"
: "${SDPX_SITE_ENV:?set SDPX_SITE_ENV to the site environment script}"
: "${SDPX_DEPOT_PATH:?set SDPX_DEPOT_PATH to the offline Julia depot}"

if [ -n "${PBS_JOBID:-}" ] || [ "${PBS_ENVIRONMENT:-}" = "PBS_BATCH" ]; then
  echo "bootstrap refuses to run inside a PBS job; use a login node" >&2
  exit 1
fi
test -f "$BOOTSTRAP_ENV/Project.toml" || {
  echo "BOOTSTRAP_ENV has no Project.toml: $BOOTSTRAP_ENV" >&2
  exit 1
}

source "$SDPX_SITE_ENV"
: "${JULIA_BIN:?set JULIA_BIN directly or through SDPX_SITE_ENV}"
export JULIA_DEPOT_PATH="$SDPX_DEPOT_PATH"
unset JULIA_PKG_OFFLINE

LOG="${BOOTSTRAP_LOG:-$BOOTSTRAP_ENV/bootstrap.log}"
{
  echo "started=$(date -Iseconds)"
  echo "hostname=$(hostname)"
  echo "julia_bin=$JULIA_BIN"
  echo "julia_version=$("$JULIA_BIN" --version 2>/dev/null | head -n 1)"
  echo "bootstrap_env=$BOOTSTRAP_ENV"
  echo "julia_depot_path=$JULIA_DEPOT_PATH"
} > "$LOG"

"$JULIA_BIN" --project="$BOOTSTRAP_ENV" --startup-file=no \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' >> "$LOG" 2>&1

echo "finished=$(date -Iseconds)" >> "$LOG"
echo "bootstrap done: $BOOTSTRAP_ENV (log: $LOG)"
