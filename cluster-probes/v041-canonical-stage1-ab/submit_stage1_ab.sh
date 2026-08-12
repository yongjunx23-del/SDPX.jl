#!/bin/bash
# Submit the canonical Stage-1 unchanged-hot-path A/B as one pinned-node job.
# Dry-run by default; pass --submit to call qsub.
set -euo pipefail

: "${NODE_NAME:?set NODE_NAME to the healthy pinned compute node}"
: "${BASELINE_SOURCE:?set BASELINE_SOURCE to the immutable rc1 baseline checkout}"
: "${BASELINE_SOURCE_SHA256:?set BASELINE_SOURCE_SHA256}"
: "${BASELINE_SOURCE_COMMIT:?set BASELINE_SOURCE_COMMIT}"
: "${BASELINE_ENV:?set BASELINE_ENV}"
: "${CANDIDATE_SOURCE:?set CANDIDATE_SOURCE to the immutable Stage-1 candidate checkout}"
: "${CANDIDATE_SOURCE_SHA256:?set CANDIDATE_SOURCE_SHA256}"
: "${CANDIDATE_SOURCE_COMMIT:?set CANDIDATE_SOURCE_COMMIT}"
: "${CANDIDATE_ENV:?set CANDIDATE_ENV}"
: "${RUNNER_SOURCE:?set RUNNER_SOURCE to bench/public_conic_suite}"
: "${RUNNER_SOURCE_SHA256:?set RUNNER_SOURCE_SHA256}"
: "${SDPX_SITE_ENV:?set SDPX_SITE_ENV}"
: "${SDPX_DEPOT_PATH:?set SDPX_DEPOT_PATH}"
: "${RESULT_ROOT:?set RESULT_ROOT to a fresh result root outside both sources}"
: "${AB_RUNNER_ROOT:?set AB_RUNNER_ROOT to the local probe/runner directory}"

for var in BASELINE_SOURCE CANDIDATE_SOURCE RUNNER_SOURCE \
           BASELINE_ENV CANDIDATE_ENV SDPX_SITE_ENV AB_RUNNER_ROOT; do
  value="${!var}"
  [ -e "$value" ] || {
    echo "$var does not exist: $value" >&2
    exit 1
  }
done
[ -d "$(dirname "$RESULT_ROOT")" ] || {
  echo "RESULT_ROOT parent does not exist: $(dirname "$RESULT_ROOT")" >&2
  exit 1
}

BASELINE_SOURCE="$(cd "$BASELINE_SOURCE" && pwd)"
CANDIDATE_SOURCE="$(cd "$CANDIDATE_SOURCE" && pwd)"
RUNNER_SOURCE="$(cd "$RUNNER_SOURCE" && pwd)"
AB_RUNNER_ROOT="$(cd "$AB_RUNNER_ROOT" && pwd)"
BASELINE_ENV="$(cd "$BASELINE_ENV" && pwd)"
CANDIDATE_ENV="$(cd "$CANDIDATE_ENV" && pwd)"
SDPX_SITE_ENV="$(cd "$(dirname "$SDPX_SITE_ENV")" && pwd)/$(basename "$SDPX_SITE_ENV")"
RESULT_ROOT="$(cd "$(dirname "$RESULT_ROOT")" 2>/dev/null && pwd)/$(basename "$RESULT_ROOT")"

case "$RESULT_ROOT" in
  "$BASELINE_SOURCE"|"$BASELINE_SOURCE"/*|"$CANDIDATE_SOURCE"|"$CANDIDATE_SOURCE"/*|"$AB_RUNNER_ROOT"|"$AB_RUNNER_ROOT"/*)
    echo "RESULT_ROOT must not live inside either source: $RESULT_ROOT" >&2
    exit 1
    ;;
esac
[ -f "$AB_RUNNER_ROOT/analyze_ab.py" ] || {
  echo "AB_RUNNER_ROOT has no analyze_ab.py: $AB_RUNNER_ROOT" >&2
  exit 1
}
if [ -e "$RESULT_ROOT" ]; then
  echo "RESULT_ROOT already exists; refusing to reuse: $RESULT_ROOT" >&2
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

vars="NODE_NAME=$NODE_NAME"
vars="$vars,BASELINE_SOURCE=$BASELINE_SOURCE"
vars="$vars,BASELINE_SOURCE_SHA256=$BASELINE_SOURCE_SHA256"
vars="$vars,BASELINE_SOURCE_COMMIT=$BASELINE_SOURCE_COMMIT"
vars="$vars,BASELINE_ENV=$BASELINE_ENV"
vars="$vars,CANDIDATE_SOURCE=$CANDIDATE_SOURCE"
vars="$vars,CANDIDATE_SOURCE_SHA256=$CANDIDATE_SOURCE_SHA256"
vars="$vars,CANDIDATE_SOURCE_COMMIT=$CANDIDATE_SOURCE_COMMIT"
vars="$vars,CANDIDATE_ENV=$CANDIDATE_ENV"
vars="$vars,RUNNER_SOURCE=$RUNNER_SOURCE"
vars="$vars,RUNNER_SOURCE_SHA256=$RUNNER_SOURCE_SHA256"
vars="$vars,SDPX_SITE_ENV=$SDPX_SITE_ENV"
vars="$vars,SDPX_DEPOT_PATH=$SDPX_DEPOT_PATH"
vars="$vars,RESULT_ROOT=$RESULT_ROOT"
vars="$vars,AB_RUNNER_ROOT=$AB_RUNNER_ROOT"
# Optional scalar overrides are forwarded only when set.  ARITHMETIC stays on
# its comma-containing PBS default to avoid -v splitting.  CASE_FILTER is
# forwarded only when it is a single token (e.g. socp_many_tiny), so
# comma-containing defaults stay on the PBS side.
for var in REPETITIONS TIMING_BATCH_SIZE TIMED_BATCHES WARMUP TIME_LIMIT MAX_ITERATIONS CASE_FILTER ARM_ORDER; do
  if [ -n "${!var:-}" ]; then
    if [ "$var" = "CASE_FILTER" ] || [ "$var" = "ARM_ORDER" ]; then
      case "${!var}" in
        *[,=[:space:]]*)
          echo "$var must be a single token (no comma/space): ${!var}" >&2
          exit 1
          ;;
      esac
    fi
    vars="$vars,$var=${!var}"
  fi
done

if [ "$SUBMIT" = "1" ]; then
  qsub -v "$vars" -l "nodes=$NODE_NAME:ppn=5" "$HERE/stage1_ab.pbs"
else
  echo "qsub -v \"$vars\" -l nodes=$NODE_NAME:ppn=5 $HERE/stage1_ab.pbs"
  echo "dry-run: re-run with --submit to submit"
fi
