#!/bin/bash
# Submit one parameter-provenance Stage-1 focused job and one full Pkg.test
# job to two distinct compute nodes concurrently, sharing one immutable
# campaign identity.
#
# The two nodes must already have been verified by the caller as healthy and
# idle.  This helper does not query PBS node state, does not pick a node, does
# not cancel or migrate any job, and does not touch any other job.  It only
# pins each submitted job to the caller-provided node and fails closed on
# missing/equal nodes and missing required variables.
#
# Dry-run by default; pass --submit to call qsub for both jobs.
set -euo pipefail

usage() {
  echo "usage: $0 FOCUSED_NODE FULL_NODE [--submit]" >&2
  exit 2
}

if [ "$#" -lt 2 ]; then
  usage
fi
FOCUSED_NODE="$1"
FULL_NODE="$2"
shift 2

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

if [ -z "$FOCUSED_NODE" ] || [ -z "$FULL_NODE" ]; then
  echo "FOCUSED_NODE and FULL_NODE must be non-empty node names" >&2
  exit 1
fi
if [ "$FOCUSED_NODE" = "$FULL_NODE" ]; then
  echo "FOCUSED_NODE and FULL_NODE must be two distinct, already-verified healthy idle nodes" >&2
  exit 1
fi
for node in "$FOCUSED_NODE" "$FULL_NODE"; do
  case "$node" in
    *[,=[:space:]]*)
      echo "node name must be a single PBS token (no comma/equals/space): $node" >&2
      exit 1
      ;;
  esac
done

: "${CAMPAIGN_ID:?set CAMPAIGN_ID to a single campaign token shared by both jobs}"
: "${CANDIDATE_SOURCE:?set CANDIDATE_SOURCE to the immutable candidate source}"
: "${CANDIDATE_SOURCE_SHA256:?set CANDIDATE_SOURCE_SHA256 to the expected source-tree SHA-256}"
: "${CANDIDATE_ENV:?set CANDIDATE_ENV to the pre-instantiated Julia environment}"
: "${SDPX_SITE_ENV:?set SDPX_SITE_ENV to the site environment script}"
: "${SDPX_DEPOT_PATH:?set SDPX_DEPOT_PATH to the offline Julia depot}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh focused result directory}"
: "${FULL_OUTPUT_ROOT:?set FULL_OUTPUT_ROOT to a fresh, distinct full result directory}"
: "${CAMPAIGN_ROOT:?set CAMPAIGN_ROOT to an existing campaign directory outside the sources}"

case "$CAMPAIGN_ID" in
  *[,=[:space:]]*)
    echo "CAMPAIGN_ID must be a single PBS token (no comma/equals/space): $CAMPAIGN_ID" >&2
    exit 1
    ;;
esac
case "$CANDIDATE_SOURCE_SHA256" in
  *[!0-9a-fA-F]*)
    echo "CANDIDATE_SOURCE_SHA256 must be a hex SHA-256 digest: $CANDIDATE_SOURCE_SHA256" >&2
    exit 1
    ;;
esac
[ "${#CANDIDATE_SOURCE_SHA256}" -eq 64 ] || {
  echo "CANDIDATE_SOURCE_SHA256 must be exactly 64 hex characters" >&2
  exit 1
}

[ -e "$CANDIDATE_SOURCE" ] || {
  echo "CANDIDATE_SOURCE does not exist: $CANDIDATE_SOURCE" >&2
  exit 1
}
[ -e "$CANDIDATE_ENV" ] || {
  echo "CANDIDATE_ENV does not exist: $CANDIDATE_ENV" >&2
  exit 1
}
[ -e "$SDPX_SITE_ENV" ] || {
  echo "SDPX_SITE_ENV does not exist: $SDPX_SITE_ENV" >&2
  exit 1
}
[ -e "$SDPX_DEPOT_PATH" ] || {
  echo "SDPX_DEPOT_PATH does not exist: $SDPX_DEPOT_PATH" >&2
  exit 1
}
test -f "$CANDIDATE_SOURCE/src/SDPX.jl" || {
  echo "CANDIDATE_SOURCE has no src/SDPX.jl: $CANDIDATE_SOURCE" >&2
  exit 1
}
test -f "$CANDIDATE_ENV/Project.toml" || {
  echo "CANDIDATE_ENV has no Project.toml: $CANDIDATE_ENV" >&2
  exit 1
}
test -d "$CAMPAIGN_ROOT" || {
  echo "CAMPAIGN_ROOT is not a directory: $CAMPAIGN_ROOT" >&2
  exit 1
}

CANDIDATE_SOURCE="$(cd "$CANDIDATE_SOURCE" && pwd)"
CANDIDATE_ENV="$(cd "$CANDIDATE_ENV" && pwd)"
SDPX_SITE_ENV="$(cd "$(dirname "$SDPX_SITE_ENV")" && pwd)/$(basename "$SDPX_SITE_ENV")"
SDPX_DEPOT_PATH="$(cd "$SDPX_DEPOT_PATH" && pwd)"
CAMPAIGN_ROOT="$(cd "$CAMPAIGN_ROOT" && pwd)"
CANDIDATE_SOURCE_REALPATH="$(cd "$CANDIDATE_SOURCE" && pwd -P)"

for root in "$OUTPUT_ROOT" "$FULL_OUTPUT_ROOT"; do
  [ -d "$(dirname "$root")" ] || {
    echo "result root parent does not exist: $(dirname "$root")" >&2
    exit 1
  }
  [ ! -e "$root" ] || {
    echo "result root already exists; refusing to reuse: $root" >&2
    exit 1
  }
  case "$root" in
    "$CANDIDATE_SOURCE"|"$CANDIDATE_SOURCE"/*|"$CANDIDATE_ENV"|"$CANDIDATE_ENV"/*|"$SDPX_DEPOT_PATH"|"$SDPX_DEPOT_PATH"/*)
      echo "result root must not live inside the candidate source, candidate env, or depot: $root" >&2
      exit 1
      ;;
  esac
done

OUTPUT_ROOT="$(cd "$(dirname "$OUTPUT_ROOT")" && pwd)/$(basename "$OUTPUT_ROOT")"
FULL_OUTPUT_ROOT="$(cd "$(dirname "$FULL_OUTPUT_ROOT")" && pwd)/$(basename "$FULL_OUTPUT_ROOT")"
if [ "$OUTPUT_ROOT" = "$FULL_OUTPUT_ROOT" ]; then
  echo "OUTPUT_ROOT and FULL_OUTPUT_ROOT must be distinct directories" >&2
  exit 1
fi

CAMPAIGN_MANIFEST="${CAMPAIGN_MANIFEST:-$CAMPAIGN_ROOT/$CAMPAIGN_ID.conf}"
case "$CAMPAIGN_MANIFEST" in
  "$CANDIDATE_SOURCE"|"$CANDIDATE_SOURCE"/*|"$CANDIDATE_ENV"|"$CANDIDATE_ENV"/*)
    echo "CAMPAIGN_MANIFEST must not live inside the candidate source or env: $CAMPAIGN_MANIFEST" >&2
    exit 1
    ;;
esac
[ ! -e "$CAMPAIGN_MANIFEST" ] || {
  echo "CAMPAIGN_MANIFEST already exists; refusing to overwrite: $CAMPAIGN_MANIFEST" >&2
  exit 1
}

for var in CANDIDATE_SOURCE CANDIDATE_ENV SDPX_SITE_ENV SDPX_DEPOT_PATH \
           CAMPAIGN_ROOT CAMPAIGN_MANIFEST OUTPUT_ROOT FULL_OUTPUT_ROOT; do
  value="${!var}"
  case "$value" in
    *[,=[:space:]]*)
      echo "$var must be a single PBS token (no comma/equals/space): $value" >&2
      exit 1
      ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
test -f "$HERE/focused.pbs" || {
  echo "missing focused.pbs next to submit_parallel.sh: $HERE/focused.pbs" >&2
  exit 1
}
test -f "$HERE/full.pbs" || {
  echo "missing full.pbs next to submit_parallel.sh: $HERE/full.pbs" >&2
  exit 1
}

# Shared immutable campaign identity.  Both jobs receive the same candidate
# path/hash/environment/campaign id; only the pinned node, result root, and
# PBS job name differ.
common_vars="CAMPAIGN_ID=$CAMPAIGN_ID"
common_vars="$common_vars,CANDIDATE_SOURCE=$CANDIDATE_SOURCE"
common_vars="$common_vars,CANDIDATE_SOURCE_SHA256=$CANDIDATE_SOURCE_SHA256"
common_vars="$common_vars,CANDIDATE_ENV=$CANDIDATE_ENV"
common_vars="$common_vars,SDPX_SITE_ENV=$SDPX_SITE_ENV"
common_vars="$common_vars,SDPX_DEPOT_PATH=$SDPX_DEPOT_PATH"

submit_one() {
  local kind="$1"
  local node="$2"
  local result_root="$3"
  local job_name="$4"
  local vars="$common_vars,NODE_NAME=$node"
  if [ "$kind" = "focused" ]; then
    vars="$vars,OUTPUT_ROOT=$result_root"
  else
    vars="$vars,FULL_OUTPUT_ROOT=$result_root"
  fi
  if [ "$SUBMIT" = "1" ]; then
    qsub -N "$job_name" -v "$vars" -l "nodes=$node:ppn=5" "$HERE/$kind.pbs"
  else
    echo "qsub -N $job_name -v \"$vars\" -l nodes=$node:ppn=5 $HERE/$kind.pbs"
  fi
}

FOCUSED_JOB_ID=""
FULL_JOB_ID=""
if [ "$SUBMIT" = "1" ]; then
  FOCUSED_JOB_ID="$(submit_one focused "$FOCUSED_NODE" "$OUTPUT_ROOT" sdpx_parameter_provenance_stage1_focused)" || {
    echo "focused qsub failed; no full job was submitted" >&2
    exit 1
  }
  case "$FOCUSED_JOB_ID" in
    ""|*[[:space:]]*)
      echo "focused qsub returned no clean job id: $FOCUSED_JOB_ID" >&2
      exit 1
      ;;
  esac
  FULL_JOB_ID="$(submit_one full "$FULL_NODE" "$FULL_OUTPUT_ROOT" sdpx_parameter_provenance_stage1_full)" || {
    echo "full qsub failed; focused job $FOCUSED_JOB_ID is already submitted and is NOT cancelled; manage it manually" >&2
    exit 1
  }
  case "$FULL_JOB_ID" in
    ""|*[[:space:]]*)
      echo "full qsub returned no clean job id: $FULL_JOB_ID" >&2
      exit 1
      ;;
  esac
fi

if [ "$SUBMIT" = "1" ]; then
  SUBMITTED_AT="$(date -Iseconds)"
else
  SUBMITTED_AT="dry-run"
fi

manifest_text="manifest_version=1
campaign_id=$CAMPAIGN_ID
submitted_at=$SUBMITTED_AT
focused_node=$FOCUSED_NODE
focused_job_id=${FOCUSED_JOB_ID:-unsubmitted}
focused_root=$OUTPUT_ROOT
full_node=$FULL_NODE
full_job_id=${FULL_JOB_ID:-unsubmitted}
full_root=$FULL_OUTPUT_ROOT
candidate_source=$CANDIDATE_SOURCE
candidate_source_realpath=$CANDIDATE_SOURCE_REALPATH
candidate_source_sha256=$CANDIDATE_SOURCE_SHA256
candidate_env=$CANDIDATE_ENV
sdp_site_env=$SDPX_SITE_ENV
sdp_depot_path=$SDPX_DEPOT_PATH
ppn=5
julia_threads=4
solver_threads=4
blas_threads=1"

if [ "$SUBMIT" = "1" ]; then
  tmp_manifest="$CAMPAIGN_MANIFEST.tmp.$$"
  printf '%s\n' "$manifest_text" > "$tmp_manifest"
  mv "$tmp_manifest" "$CAMPAIGN_MANIFEST"
  echo "campaign manifest written: $CAMPAIGN_MANIFEST"
  echo "submitted focused job $FOCUSED_JOB_ID on $FOCUSED_NODE and full job $FULL_JOB_ID on $FULL_NODE"
else
  echo "campaign manifest (dry-run; would be written to $CAMPAIGN_MANIFEST):"
  printf '%s\n' "$manifest_text"
  echo "dry-run: re-run with --submit to submit both jobs"
fi
