#!/usr/bin/env bash

# Run the SDP reference and native SOC implementation as separate Julia
# subprocesses.  The PBS-safe order can be alternated across allocations
# (`sdp-socp` then `socp-sdp`) to avoid treating thermal drift as a solver
# effect. Comma-separated spellings remain accepted for interactive use.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${CSDR_MODEL:?set CSDR_MODEL}"
CASE="${SDPX_CASE:?set SDPX_CASE to J40 or J80}"
ARITHMETIC="${SDPX_ARITHMETIC:-Float64x4}"
THREADS="${SDPX_THREADS:-1}"
PPN="${PBS_NP:-${SDPX_PPN:-1}}"
EXPECTED_HASH="${SDPX_EXPECTED_HASH:?set SDPX_EXPECTED_HASH}"
ORDER="${SDPX_PAIR_ORDER:-sdp-socp}"
ROOT="${SDPX_OUTPUT_ROOT:-$HERE/results/pair-${CASE}-${PBS_JOBID:-interactive}-$(date -u +%Y%m%dT%H%M%SZ)}"
JULIA_BIN="${JULIA_BIN:-julia}"
REPO="$(cd "$HERE/../.." && pwd)"
SHARED_ENV="${SDPX_SHARED_ENV:-}"
SHARED_DEPOT="${SDPX_SHARED_DEPOT:-}"
CSDR_RELEASE="${CSDR_RELEASE:-}"
ORIGINAL_DEPOT="${JULIA_DEPOT_PATH:-}"
THREAD_POLICY="${SDPX_THREAD_POLICY:-exclusive}"

[[ -n "${PBS_JOBID:-}" || "${ALLOW_INTERACTIVE:-0}" == 1 ]] || {
    echo "Refusing paired benchmark outside PBS; set ALLOW_INTERACTIVE=1 for a local smoke test." >&2
    exit 2
}
[[ "$THREADS" =~ ^[0-9]+$ && "$PPN" =~ ^[0-9]+$ && THREADS -le PPN ]] || {
    echo "SDPX_THREADS=$THREADS must be <= allocated slots=$PPN." >&2
    exit 2
}
(( THREADS <= 32 )) || {
    echo "The validated fixed-trace campaign is capped at 32 Julia threads." >&2
    exit 2
}
[[ "$THREAD_POLICY" == default || "$THREAD_POLICY" == exclusive ]] || {
    echo "SDPX_THREAD_POLICY must be default or exclusive." >&2
    exit 2
}
JULIA_EXECUTABLE="$(command -v "$JULIA_BIN" 2>/dev/null || true)"
[[ -n "$JULIA_EXECUTABLE" ]] || {
    echo "Julia executable not found: $JULIA_BIN (set JULIA_BIN explicitly)." >&2
    exit 2
}
JULIA_BIN="$JULIA_EXECUTABLE"
case "$ORDER" in
    sdp-socp|sdp,socp)
        FIRST=sdp
        SECOND=socp
        ;;
    socp-sdp|socp,sdp)
        FIRST=socp
        SECOND=sdp
        ;;
    *)
        echo "SDPX_PAIR_ORDER must be sdp-socp or socp-sdp." >&2
        exit 2
        ;;
esac
mkdir -p "$ROOT"
JOB_DEPOT="${SDPX_JOB_DEPOT:-$ROOT/.julia-depot}"
mkdir -p "$JOB_DEPOT"
export JULIA_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
JULIA_THREAD_ARGS=(--threads="$THREADS")
if [[ "$THREAD_POLICY" == exclusive ]]; then
    export JULIA_EXCLUSIVE=1
    export JULIA_THREAD_SLEEP_THRESHOLD=10000000
    JULIA_THREAD_ARGS=(--threads="$THREADS,0" --gcthreads="1,0")
fi
if [[ -n "$SHARED_ENV" && -f "$SHARED_ENV/Project.toml" ]]; then
    export JULIA_PROJECT="$SHARED_ENV"
    export JULIA_LOAD_PATH="$SHARED_ENV:$REPO:@stdlib"
    READ_DEPOT="${SHARED_DEPOT:-$ORIGINAL_DEPOT}"
    export JULIA_DEPOT_PATH="$JOB_DEPOT${READ_DEPOT:+:$READ_DEPOT}"
else
    export JULIA_PROJECT="$REPO"
    export JULIA_LOAD_PATH="$REPO:@stdlib"
    export JULIA_DEPOT_PATH="$JOB_DEPOT${ORIGINAL_DEPOT:+:$ORIGINAL_DEPOT}"
fi

{
    echo "case=$CASE"
    echo "arithmetic=$ARITHMETIC"
    echo "requested_threads=$THREADS"
    echo "allocated_slots=$PPN"
    echo "blas_threads=1"
    echo "thread_policy=$THREAD_POLICY"
    echo "julia_bin=$(readlink -f "$JULIA_BIN")"
    echo "job_depot=$JOB_DEPOT"
    echo "job_id=${PBS_JOBID:-interactive}"
    echo "node=$(hostname)"
    grep '^Cpus_allowed_list' /proc/self/status 2>/dev/null || true
    command -v numactl >/dev/null 2>&1 && numactl --show 2>/dev/null || true
} > "$ROOT/resources.txt"

run_mode() {
    local mode="$1"
    local scaling=none
    [[ "$mode" == sdp ]] && scaling="${SDPX_SDP_SCALING:-none}"
    /usr/bin/time -v -o "$ROOT/$mode.time" \
        "$JULIA_BIN" --startup-file=no --project="$JULIA_PROJECT" \
        "${JULIA_THREAD_ARGS[@]}" \
        "$HERE/benchmark.jl" --case="$CASE" --model="$MODEL" \
        --release="$CSDR_RELEASE" \
        --mode="$mode" --scaling="$scaling" --arithmetic="$ARITHMETIC" \
        --threads="$THREADS" --reps="${SDPX_REPS:-3}" \
        --warmup="${SDPX_WARMUP:-1}" --expected-hash="$EXPECTED_HASH" \
        --tolerance="${SDPX_TOLERANCE:-1e-12}" \
        --max-iterations="${SDPX_MAX_ITERATIONS:-500}" \
        --time-limit-seconds="${SDPX_TIME_LIMIT_SECONDS:-43200}" \
        --precision-bits="${SDPX_PRECISION_BITS:-256}" \
        --output="$ROOT/$mode.toml" --manifest="$ROOT/$mode.manifest.toml" \
        > "$ROOT/$mode.log" 2>&1
}

run_mode "$FIRST"
run_mode "$SECOND"
echo PASSED > "$ROOT/PASSED"
echo "PASSED pair_order=$ORDER output=$ROOT"
