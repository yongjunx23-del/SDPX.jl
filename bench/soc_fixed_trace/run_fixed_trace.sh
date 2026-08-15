#!/usr/bin/env bash

# Portable compute-node launcher used by the one-configuration PBS templates.
# The PBS file supplies the case/mode/arithmetic defaults; this script owns the
# resource checks and keeps every Julia process isolated from the shell.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
SHARED_ENV="${SDPX_SHARED_ENV:-}"
SHARED_DEPOT="${SDPX_SHARED_DEPOT:-}"
CSDR_RELEASE="${CSDR_RELEASE:-}"
ORIGINAL_DEPOT="${JULIA_DEPOT_PATH:-}"
MODEL="${CSDR_MODEL:?set CSDR_MODEL to an immutable serialized CSDR model}"
CASE="${SDPX_CASE:?set SDPX_CASE to J40 or J80}"
MODE="${SDPX_MODE:?set SDPX_MODE to sdp or socp}"
ARITHMETIC="${SDPX_ARITHMETIC:-Float64x4}"
THREADS="${SDPX_THREADS:-1}"
PPN="${PBS_NP:-${SDPX_PPN:-1}}"
REPS="${SDPX_REPS:-3}"
WARMUP="${SDPX_WARMUP:-1}"
EXPECTED_HASH="${SDPX_EXPECTED_HASH:?set SDPX_EXPECTED_HASH to the model SHA-256}"
OUTPUT_ROOT="${SDPX_OUTPUT_ROOT:-$HERE/results/${CASE}-${MODE}-${ARITHMETIC}-${PBS_JOBID:-interactive}-$(date -u +%Y%m%dT%H%M%SZ)}"
DRIVER="${SDPX_DRIVER:-benchmark.jl}"
THREAD_POLICY="${SDPX_THREAD_POLICY:-default}"

[[ "${ALLOW_INTERACTIVE:-0}" == "1" || -n "${PBS_JOBID:-}" ]] || {
    echo "Refusing to run fixed-trace benchmarks outside a PBS compute allocation." >&2
    echo "Set ALLOW_INTERACTIVE=1 only for a deliberate local smoke test." >&2
    exit 2
}
[[ "$THREADS" =~ ^[0-9]+$ && "$PPN" =~ ^[0-9]+$ ]] || {
    echo "SDPX_THREADS and PBS_NP/SDPX_PPN must be integers." >&2
    exit 2
}
(( THREADS >= 1 && THREADS <= PPN )) || {
    echo "Requested Julia threads ($THREADS) exceed allocated slots ($PPN)." >&2
    exit 2
}
(( THREADS <= 32 )) || {
    echo "The validated fixed-trace campaign is capped at 32 Julia threads." >&2
    exit 2
}
[[ "$CASE" == J40 || "$CASE" == J80 ]] || { echo "SDPX_CASE must be J40 or J80." >&2; exit 2; }
[[ "$MODE" == sdp || "$MODE" == socp ]] || { echo "SDPX_MODE must be sdp or socp." >&2; exit 2; }
[[ "$DRIVER" == benchmark.jl || "$DRIVER" == diagnose_native.jl ]] || {
    echo "SDPX_DRIVER must be benchmark.jl or diagnose_native.jl." >&2
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
if [[ "$MODE" == socp && "${SDPX_SCALING:-none}" != none ]]; then
    echo "Native socp requires SDPX_SCALING=none." >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
JOB_DEPOT="${SDPX_JOB_DEPOT:-$OUTPUT_ROOT/.julia-depot}"
mkdir -p "$JOB_DEPOT"
export JULIA_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
JULIA_THREAD_ARGS=(--threads="$THREADS")
if [[ "$THREAD_POLICY" == exclusive ]]; then
    # High-core J80 templates select this policy by default after a same-node
    # control improved equality-Gram throughput by 49.1%.  The explicit pool
    # avoids an unused interactive pool, one GC worker avoids competing with
    # ownership-partitioned numerical tasks, and Julia binds worker i to CPU i
    # inside the PBS cpuset.  Other launchers retain the ordinary Julia policy.
    export JULIA_EXCLUSIVE=1
    export JULIA_THREAD_SLEEP_THRESHOLD=10000000
    JULIA_THREAD_ARGS=(--threads="$THREADS,0" --gcthreads="1,0")
fi
if [[ -n "$SHARED_ENV" && -f "$SHARED_ENV/Project.toml" ]]; then
    export JULIA_PROJECT="$SHARED_ENV"
    # The instantiated environment must precede the candidate.  SDPX declares
    # MultiFloats as a weak dependency, so the reverse order shadows the
    # shared manifest and breaks Float64x4 loading on clean compute nodes.
    export JULIA_LOAD_PATH="$SHARED_ENV:$REPO:@stdlib"
    # Candidate packages share a UUID/version while their source trees differ.
    # A job-private first depot prevents concurrent candidates from racing on
    # the same precompile pidfile/cache; immutable packages and artifacts still
    # resolve from the shared second depot.
    READ_DEPOT="${SHARED_DEPOT:-$ORIGINAL_DEPOT}"
    export JULIA_DEPOT_PATH="$JOB_DEPOT${READ_DEPOT:+:$READ_DEPOT}"
else
    export JULIA_PROJECT="$REPO"
    export JULIA_LOAD_PATH="$REPO:@stdlib"
    export JULIA_DEPOT_PATH="$JOB_DEPOT${ORIGINAL_DEPOT:+:$ORIGINAL_DEPOT}"
fi

{
    echo "case=$CASE"
    echo "mode=$MODE"
    echo "arithmetic=$ARITHMETIC"
    echo "requested_threads=$THREADS"
    echo "allocated_slots=$PPN"
    echo "blas_threads=1"
    echo "thread_policy=$THREAD_POLICY"
    echo "julia_exclusive=${JULIA_EXCLUSIVE:-0}"
    echo "julia_thread_sleep_threshold=${JULIA_THREAD_SLEEP_THRESHOLD:-default}"
    echo "julia_bin=$(readlink -f "$JULIA_BIN")"
    echo "job_depot=$JOB_DEPOT"
    echo "job_id=${PBS_JOBID:-interactive}"
    echo "node=$(hostname)"
    if [[ -r /proc/self/status ]]; then grep '^Cpus_allowed_list' /proc/self/status || true; fi
    if command -v numactl >/dev/null 2>&1; then numactl --show || true; fi
    if command -v lscpu >/dev/null 2>&1; then lscpu | grep -E 'Model name|Socket|NUMA node' || true; fi
} > "$OUTPUT_ROOT/resources.txt"

TIME_BIN="/usr/bin/time"
[[ -x "$TIME_BIN" ]] || TIME_BIN="time"
# `set -u` on the cluster's older Bash treats an expansion of an empty array
# as an unbound variable. Use `env` as a no-op executable prefix so both NUMA
# branches have a nonempty command vector.
NUMA_PREFIX=(env)
if [[ "${SDPX_NUMA_INTERLEAVE:-0}" == "1" && -x "$(command -v numactl 2>/dev/null || true)" ]]; then
    NUMA_PREFIX=(numactl --interleave=all)
fi

"$TIME_BIN" -v -o "$OUTPUT_ROOT/benchmark.time" \
    "${NUMA_PREFIX[@]}" \
    "$JULIA_BIN" --startup-file=no --project="$JULIA_PROJECT" \
    "${JULIA_THREAD_ARGS[@]}" \
    "$HERE/$DRIVER" \
    --case="$CASE" --model="$MODEL" --mode="$MODE" \
    --release="$CSDR_RELEASE" \
    --arithmetic="$ARITHMETIC" --threads="$THREADS" \
    --reps="$REPS" --warmup="$WARMUP" \
    --preflight-only="${SDPX_PREFLIGHT_ONLY:-false}" \
    --expected-hash="$EXPECTED_HASH" \
    --tolerance="${SDPX_TOLERANCE:-1e-12}" \
    --max-iterations="${SDPX_MAX_ITERATIONS:-500}" \
    --time-limit-seconds="${SDPX_TIME_LIMIT_SECONDS:-43200}" \
    --precision-bits="${SDPX_PRECISION_BITS:-256}" \
    --scaling="${SDPX_SCALING:-none}" \
    --output="$OUTPUT_ROOT/report.toml" \
    --manifest="$OUTPUT_ROOT/manifest.toml" \
    > "$OUTPUT_ROOT/benchmark.log" 2>&1

echo PASSED > "$OUTPUT_ROOT/PASSED"
echo "PASSED case=$CASE mode=$MODE arithmetic=$ARITHMETIC threads=$THREADS output=$OUTPUT_ROOT"
