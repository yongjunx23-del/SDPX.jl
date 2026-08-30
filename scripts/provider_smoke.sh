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

TARGET="${SDPX_PROVIDER_SMOKE_TARGET:-all}"
case "$TARGET" in
  all|mfla|bfla) ;;
  *)
    echo "SDPX_PROVIDER_SMOKE_TARGET must be all, mfla, or bfla" >&2
    exit 2
    ;;
esac

if [ "$TARGET" = all ] || [ "$TARGET" = mfla ]; then
  if [ ! -d "$SDPX_MFLA_PROJECT" ]; then
    echo "MFLA checkout not found at $SDPX_MFLA_PROJECT (set SDPX_MFLA_PROJECT)" >&2
    exit 1
  fi
fi
if [ "$TARGET" = all ] || [ "$TARGET" = bfla ]; then
  if [ ! -d "$SDPX_BFLA_PROJECT" ]; then
    echo "BFLA checkout not found at $SDPX_BFLA_PROJECT (set SDPX_BFLA_PROJECT)" >&2
    exit 1
  fi
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

# Local development may opt into an offline run when all dependencies are
# already cached. CI and clean checkouts must resolve registered dependencies,
# so online resolution is the default.
export JULIA_PKG_OFFLINE="${JULIA_PKG_OFFLINE:-false}"

DEVELOP_ARGS=("$ROOT")
if [ "$TARGET" = all ] || [ "$TARGET" = mfla ]; then
  DEVELOP_ARGS+=("$SDPX_MFLA_PROJECT")
fi
if [ "$TARGET" = all ] || [ "$TARGET" = bfla ]; then
  DEVELOP_ARGS+=("$SDPX_BFLA_PROJECT")
fi

julia --startup-file=no --project="$SMOKE_ENV" -e '
using Pkg
for path in ARGS
    Pkg.develop(path=path)
end
Pkg.add(["MultiFloats", "GenericLinearAlgebra"])
' "${DEVELOP_ARGS[@]}"

# Julia 1.12 can exhaust its inference compiler when the MFLA fixed-width
# specializations and the BFLA/MPFR specialization are compiled in the same
# process.  Each target is an independent provider contract, so run `all` as
# two fresh processes.  This changes no solver/provider route and makes the
# documented smoke command reproducible.
case "$TARGET" in
  all)
    for provider_target in mfla bfla; do
      SDPX_PROVIDER_SMOKE_TARGET="$provider_target" \
        julia --startup-file=no --project="$SMOKE_ENV" -t1 \
          "$ROOT/validation/providers/provider_smoke.jl"
    done
    ;;
  mfla|bfla)
    SDPX_PROVIDER_SMOKE_TARGET="$TARGET" \
      julia --startup-file=no --project="$SMOKE_ENV" -t1 \
        "$ROOT/validation/providers/provider_smoke.jl"
    ;;
esac

echo "provider smoke completed: target=$TARGET mfla=$SDPX_MFLA_PROJECT bfla=$SDPX_BFLA_PROJECT"
