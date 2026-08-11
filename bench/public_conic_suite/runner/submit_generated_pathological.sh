#!/bin/bash
# Submit the two generated-pathological P0 PBS jobs for one candidate.
#
# Required:
#   SDPX_SOURCE            immutable candidate checkout
#   SDPX_CAMPAIGN_ROOT     unique campaign root OUTSIDE the candidate source;
#                          regular/ and bigfloat/ result dirs are created under it
#   SDPX_SITE_ENV          site environment script (exports JULIA_BIN)
#   SDPX_ENVIRONMENT       Julia environment with SDPX/MultiFloats installed
#   SDPX_DEPOT_PATH        offline Julia depot
#   REGULAR_NODE_NAME      healthy node pinned with ppn=5
#   BIGFLOAT_NODE_NAME     healthy node pinned with ppn=1 (may equal regular)
#
# Pass --submit to actually qsub; otherwise this prints the commands.
set -euo pipefail

: "${SDPX_SOURCE:?set SDPX_SOURCE to the immutable candidate checkout}"
: "${SDPX_CAMPAIGN_ROOT:?set SDPX_CAMPAIGN_ROOT to a unique result root outside the candidate source}"
: "${SDPX_SITE_ENV:?set SDPX_SITE_ENV to the site environment script}"
: "${SDPX_ENVIRONMENT:?set SDPX_ENVIRONMENT to the matching Julia environment}"
: "${SDPX_DEPOT_PATH:?set SDPX_DEPOT_PATH to the offline Julia depot}"
: "${REGULAR_NODE_NAME:?set REGULAR_NODE_NAME to the pinned regular-compute node}"
: "${BIGFLOAT_NODE_NAME:?set BIGFLOAT_NODE_NAME to the pinned bigfloat-compute node}"

SDPX_SOURCE="$(cd "$SDPX_SOURCE" && pwd)"
SDPX_CAMPAIGN_ROOT="$(cd "$(dirname "$SDPX_CAMPAIGN_ROOT")" && pwd)/$(basename "$SDPX_CAMPAIGN_ROOT")"
case "$SDPX_CAMPAIGN_ROOT" in
  "$SDPX_SOURCE"|"$SDPX_SOURCE"/*)
    echo "SDPX_CAMPAIGN_ROOT must not live inside the candidate source: $SDPX_CAMPAIGN_ROOT" >&2
    exit 1
    ;;
esac

if [ -e "$SDPX_CAMPAIGN_ROOT" ]; then
  echo "SDPX_CAMPAIGN_ROOT already exists; refusing to reuse a campaign root: $SDPX_CAMPAIGN_ROOT" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

SUBMIT=0
for argument in "$@"; do
  if [ "$argument" = "--submit" ]; then
    SUBMIT=1
  else
    echo "unknown argument: $argument" >&2
    exit 2
  fi
done

submit_one() {
  local script="$1"
  local node="$2"
  local ppn="$3"
  local result_dir="$4"
  local vars="SDPX_SOURCE=$SDPX_SOURCE"
  vars="$vars,SDPX_SITE_ENV=$SDPX_SITE_ENV"
  vars="$vars,SDPX_ENVIRONMENT=$SDPX_ENVIRONMENT"
  vars="$vars,SDPX_DEPOT_PATH=$SDPX_DEPOT_PATH"
  vars="$vars,NODE_NAME=$node"
  vars="$vars,SDPX_RESULT_DIR=$result_dir"
  if [ -n "${JULIA_BIN:-}" ]; then
    vars="$vars,JULIA_BIN=$JULIA_BIN"
  fi
  if [ "$SUBMIT" = "1" ]; then
    qsub -v "$vars" -l "nodes=$node:ppn=$ppn" "$script"
  else
    echo "qsub -v \"$vars\" -l nodes=$node:ppn=$ppn $script"
  fi
}

echo "campaign_root=$SDPX_CAMPAIGN_ROOT"
echo "regular_result_dir=$SDPX_CAMPAIGN_ROOT/regular"
echo "bigfloat_result_dir=$SDPX_CAMPAIGN_ROOT/bigfloat"
submit_one "$HERE/generated_pathological.pbs" \
  "$REGULAR_NODE_NAME" 5 "$SDPX_CAMPAIGN_ROOT/regular"
submit_one "$HERE/generated_pathological_bigfloat.pbs" \
  "$BIGFLOAT_NODE_NAME" 1 "$SDPX_CAMPAIGN_ROOT/bigfloat"

if [ "$SUBMIT" != "1" ]; then
  echo "dry-run: re-run with --submit to submit both jobs"
fi
