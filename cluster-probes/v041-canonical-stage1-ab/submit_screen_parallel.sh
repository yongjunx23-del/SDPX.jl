#!/bin/bash
# Submit the lean development-stage parallel A/B screen: three disjoint
# single-family jobs (LP/SOCP/SDP), each pinned to its own healthy node, each
# running baseline and candidate sequentially inside one PBS job.  The three
# nodes are explicit positional arguments; all identity/provenance variables
# are shared across the three jobs.  Dry-run by default; pass --submit to call
# qsub for all three jobs.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 LP_NODE SOCP_NODE SDP_NODE [--submit]" >&2
  exit 2
fi
LP_NODE="$1"
SOCP_NODE="$2"
SDP_NODE="$3"
shift 3
SUBMIT=0
for argument in "$@"; do
  case "$argument" in
    --submit)
      SUBMIT=1
      ;;
    *)
      echo "unknown argument: $argument" >&2
      exit 2
      ;;
  esac
done

if [ "$LP_NODE" = "$SOCP_NODE" ] || \
   [ "$LP_NODE" = "$SDP_NODE" ] || \
   [ "$SOCP_NODE" = "$SDP_NODE" ]; then
  echo "LP/SOCP/SDP nodes must be three distinct healthy nodes" >&2
  exit 1
fi

: "${BASELINE_SOURCE:?set BASELINE_SOURCE to the immutable baseline checkout}"
: "${BASELINE_SOURCE_SHA256:?set BASELINE_SOURCE_SHA256}"
: "${BASELINE_SOURCE_COMMIT:?set BASELINE_SOURCE_COMMIT}"
: "${BASELINE_ENV:?set BASELINE_ENV}"
: "${CANDIDATE_SOURCE:?set CANDIDATE_SOURCE to the immutable candidate checkout}"
: "${CANDIDATE_SOURCE_SHA256:?set CANDIDATE_SOURCE_SHA256}"
: "${CANDIDATE_SOURCE_COMMIT:?set CANDIDATE_SOURCE_COMMIT}"
: "${CANDIDATE_ENV:?set CANDIDATE_ENV}"
: "${RUNNER_SOURCE:?set RUNNER_SOURCE to bench/public_conic_suite}"
: "${RUNNER_SOURCE_SHA256:?set RUNNER_SOURCE_SHA256}"
: "${SDPX_SITE_ENV:?set SDPX_SITE_ENV}"
: "${SDPX_DEPOT_PATH:?set SDPX_DEPOT_PATH}"
: "${AB_RUNNER_ROOT:?set AB_RUNNER_ROOT to the local probe/runner directory}"
: "${RESULT_ROOT_PREFIX:?set RESULT_ROOT_PREFIX; per-family roots are derived as PREFIX/lp, PREFIX/socp, PREFIX/sdp}"

for var in BASELINE_SOURCE CANDIDATE_SOURCE RUNNER_SOURCE \
           BASELINE_ENV CANDIDATE_ENV SDPX_SITE_ENV AB_RUNNER_ROOT; do
  value="${!var}"
  [ -e "$value" ] || {
    echo "$var does not exist: $value" >&2
    exit 1
  }
done
[ -d "$(dirname "$RESULT_ROOT_PREFIX")" ] || {
  echo "RESULT_ROOT_PREFIX parent does not exist: $(dirname "$RESULT_ROOT_PREFIX")" >&2
  exit 1
}

BASELINE_SOURCE="$(cd "$BASELINE_SOURCE" && pwd)"
CANDIDATE_SOURCE="$(cd "$CANDIDATE_SOURCE" && pwd)"
RUNNER_SOURCE="$(cd "$RUNNER_SOURCE" && pwd)"
AB_RUNNER_ROOT="$(cd "$AB_RUNNER_ROOT" && pwd)"
BASELINE_ENV="$(cd "$BASELINE_ENV" && pwd)"
CANDIDATE_ENV="$(cd "$CANDIDATE_ENV" && pwd)"
SDPX_SITE_ENV="$(cd "$(dirname "$SDPX_SITE_ENV")" && pwd)/$(basename "$SDPX_SITE_ENV")"
RESULT_ROOT_PREFIX="$(cd "$(dirname "$RESULT_ROOT_PREFIX")" 2>/dev/null && pwd)/$(basename "$RESULT_ROOT_PREFIX")"

RESULT_ROOT_LP="${RESULT_ROOT_LP:-$RESULT_ROOT_PREFIX/lp}"
RESULT_ROOT_SOCP="${RESULT_ROOT_SOCP:-$RESULT_ROOT_PREFIX/socp}"
RESULT_ROOT_SDP="${RESULT_ROOT_SDP:-$RESULT_ROOT_PREFIX/sdp}"
if [ "$RESULT_ROOT_LP" = "$RESULT_ROOT_SOCP" ] || \
   [ "$RESULT_ROOT_LP" = "$RESULT_ROOT_SDP" ] || \
   [ "$RESULT_ROOT_SOCP" = "$RESULT_ROOT_SDP" ]; then
  echo "LP/SOCP/SDP result roots must be distinct" >&2
  exit 1
fi
for root in "$RESULT_ROOT_LP" "$RESULT_ROOT_SOCP" "$RESULT_ROOT_SDP"; do
  if [ -e "$root" ]; then
    echo "RESULT_ROOT already exists; refusing to reuse: $root" >&2
    exit 1
  fi
  case "$root" in
    "$BASELINE_SOURCE"|"$BASELINE_SOURCE"/*|"$CANDIDATE_SOURCE"|"$CANDIDATE_SOURCE"/*|"$AB_RUNNER_ROOT"|"$AB_RUNNER_ROOT"/*)
      echo "RESULT_ROOT must not live inside either source or the runner root: $root" >&2
      exit 1
      ;;
  esac
done

[ -f "$AB_RUNNER_ROOT/analyze_ab.py" ] || {
  echo "AB_RUNNER_ROOT has no analyze_ab.py: $AB_RUNNER_ROOT" >&2
  exit 1
}
grep -qF -- '--screen' "$AB_RUNNER_ROOT/analyze_ab.py" || {
  echo "analyze_ab.py does not support --screen; update the runner probe first" >&2
  exit 1
}
[ -f "$AB_RUNNER_ROOT/stage1_ab.pbs" ] || {
  echo "AB_RUNNER_ROOT has no stage1_ab.pbs: $AB_RUNNER_ROOT" >&2
  exit 1
}

HERE="$(cd "$(dirname "$0")" && pwd)"

# Shared identity/provenance variables.  SCREEN_MODE, the lean timing scheme
# (warmup 1 + 3 batches x 5), and the single-family CASE_FILTER are per screen
# contract; ARITHMETIC stays on the PBS-side default float64,float64x4.
base_vars="SCREEN_MODE=true"
base_vars="$base_vars,BASELINE_SOURCE=$BASELINE_SOURCE"
base_vars="$base_vars,BASELINE_SOURCE_SHA256=$BASELINE_SOURCE_SHA256"
base_vars="$base_vars,BASELINE_SOURCE_COMMIT=$BASELINE_SOURCE_COMMIT"
base_vars="$base_vars,BASELINE_ENV=$BASELINE_ENV"
base_vars="$base_vars,CANDIDATE_SOURCE=$CANDIDATE_SOURCE"
base_vars="$base_vars,CANDIDATE_SOURCE_SHA256=$CANDIDATE_SOURCE_SHA256"
base_vars="$base_vars,CANDIDATE_SOURCE_COMMIT=$CANDIDATE_SOURCE_COMMIT"
base_vars="$base_vars,CANDIDATE_ENV=$CANDIDATE_ENV"
base_vars="$base_vars,RUNNER_SOURCE=$RUNNER_SOURCE"
base_vars="$base_vars,RUNNER_SOURCE_SHA256=$RUNNER_SOURCE_SHA256"
base_vars="$base_vars,SDPX_SITE_ENV=$SDPX_SITE_ENV"
base_vars="$base_vars,SDPX_DEPOT_PATH=$SDPX_DEPOT_PATH"
base_vars="$base_vars,AB_RUNNER_ROOT=$AB_RUNNER_ROOT"
base_vars="$base_vars,TIMING_BATCH_SIZE=${TIMING_BATCH_SIZE:-5}"
base_vars="$base_vars,TIMED_BATCHES=${TIMED_BATCHES:-3}"

# Optional scalar overrides are forwarded only when set.  ARITHMETIC stays on
# its comma-containing PBS default to avoid -v splitting.  CASE_FILTER is set
# per family below and must remain a single token.
for var in REPETITIONS WARMUP TIME_LIMIT MAX_ITERATIONS ARM_ORDER; do
  if [ -n "${!var:-}" ]; then
    case "${!var}" in
      *[,=[:space:]]*)
        echo "$var must be a single token (no comma/space): ${!var}" >&2
        exit 1
        ;;
    esac
    base_vars="$base_vars,$var=${!var}"
  fi
done

submit_one() {
  local node="$1"
  local case_filter="$2"
  local result_root="$3"
  local job_name="$4"
  local vars="$base_vars,NODE_NAME=$node,CASE_FILTER=$case_filter,RESULT_ROOT=$result_root"
  if [ "$SUBMIT" = "1" ]; then
    qsub -N "$job_name" -v "$vars" -l "nodes=$node:ppn=5" "$HERE/stage1_ab.pbs"
  else
    echo "qsub -N $job_name -v \"$vars\" -l nodes=$node:ppn=5 $HERE/stage1_ab.pbs"
  fi
}

submit_one "$LP_NODE" lp_row_scaling "$RESULT_ROOT_LP" sdpx_screen_lp
submit_one "$SOCP_NODE" socp_many_tiny "$RESULT_ROOT_SOCP" sdpx_screen_socp
submit_one "$SDP_NODE" sdp_hilbert "$RESULT_ROOT_SDP" sdpx_screen_sdp

if [ "$SUBMIT" = "0" ]; then
  echo "dry-run: re-run with --submit to submit all three screen jobs"
fi
