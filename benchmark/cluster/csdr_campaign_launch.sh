#!/usr/bin/env bash
# Submit the CSDR build -> solve -> validate -> controller dependency graph.
#
# `--dry-run` is the required planning path: it writes manifests under the
# campaign root and prints every qsub command without invoking qsub. The
# launcher never names a compute node; placement remains a live PBS gate.

set -euo pipefail

readonly SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly MANIFEST_TOOL="$SCRIPT_DIR/csdr_campaign_manifest.sh"
readonly BUILD_PBS="$SCRIPT_DIR/csdr_cache_build.pbs"
readonly SOLVE_PBS="$SCRIPT_DIR/csdr_cache_solve.pbs"
readonly PREFLIGHT_PBS="$SCRIPT_DIR/csdr_campaign_preflight.pbs"
readonly VALIDATOR="$SCRIPT_DIR/csdr_campaign_validate.sh"
readonly CONTROLLER_PBS="$SCRIPT_DIR/csdr_campaign_controller.pbs"

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-}"
SDPX_SOURCE_ROOT="${SDPX_SOURCE_ROOT:-}"
SDPX_ENV_ROOT="${SDPX_ENV_ROOT:-}"
SDPX_DEPLOYED_COMMIT="${SDPX_DEPLOYED_COMMIT:-}"
MFLA_DEPLOYED_COMMIT="${MFLA_DEPLOYED_COMMIT:-}"
CSDR_SOURCE_TREE_SHA256="${CSDR_SOURCE_TREE_SHA256:-}"
CSDR_SOURCE_ROOT="${CSDR_SOURCE_ROOT:-}"
CSDR_DRIVER="${CSDR_DRIVER:-}"
CSDR_CONTROLLER="${CSDR_CONTROLLER:-}"
MANIFEST_ROOT=""
RUN_ID="${CSDR_RUN_ID:-}"
DRY_RUN=0
MAX_CORES=768
STAGE=alpha
SELECTED_Q=""
SELECTED_J=""
SELECTED_NMU=""
FENCE_KEYS=""

usage() {
    cat <<'EOF'
usage: csdr_campaign_launch.sh --campaign-root PATH [options]

Options:
  --campaign-root PATH       immutable campaign/output root (required)
  --sdpx-source-root PATH    SDPX checkout used by compute jobs
  --sdpx-env-root PATH       Julia environment used by compute jobs
  --sdpx-deployed-commit HEX validated 40-hex SDPX release commit
  --mfla-deployed-commit HEX validated 40-hex MultiFloats release commit
  --csdr-source-tree-sha256 HEX validated 64-hex CSDR snapshot tree hash
  --controller PATH           csdr_convergence_cli.jl on the immutable release
  --stage alpha|j|nmu|fence adaptive campaign stage (default alpha)
  --q COUNT                   selected alpha count (one action; required for j/nmu)
  --selected-j J              selected J (required for j/nmu)
  --selected-nmu N            selected N_mu (required for nmu)
  --missing KEY[,KEY...]      fence-stage missing predecessor keys
  --manifest-root PATH       manifest directory (default CAMPAIGN_ROOT/manifests)
  --run-id ID                immutable run identifier (default UTC timestamp+pid)
  --expected-sdpx-commit HEX optional immutable release commit gate
  --dry-run                  generate manifests and print qsub commands only
  --help

The launcher uses at most 768 simultaneously allocated cores: build and large
solve arrays are throttled at 12x64, small solves at 48x16, and validation at
96x8. It never hard-codes a node or bypasses PBS's live placement gate. The
certified launcher submits only the selected stage; it does not launch the
full Cartesian grid by default.
EOF
}

EXPECTED_COMMIT="${CSDR_EXPECTED_SDPX_COMMIT:-}"
while (($#)); do
    case "$1" in
        --campaign-root)
            (($# >= 2)) || { echo "--campaign-root requires a path" >&2; exit 2; }
            CAMPAIGN_ROOT=$2
            shift 2
            ;;
        --sdpx-source-root)
            (($# >= 2)) || { echo "--sdpx-source-root requires a path" >&2; exit 2; }
            SDPX_SOURCE_ROOT=$2
            shift 2
            ;;
        --sdpx-env-root)
            (($# >= 2)) || { echo "--sdpx-env-root requires a path" >&2; exit 2; }
            SDPX_ENV_ROOT=$2
            shift 2
            ;;
        --sdpx-deployed-commit)
            (($# >= 2)) || { echo "--sdpx-deployed-commit requires a value" >&2; exit 2; }
            SDPX_DEPLOYED_COMMIT=$2
            shift 2
            ;;
        --mfla-deployed-commit)
            (($# >= 2)) || { echo "--mfla-deployed-commit requires a value" >&2; exit 2; }
            MFLA_DEPLOYED_COMMIT=$2
            shift 2
            ;;
        --csdr-source-tree-sha256)
            (($# >= 2)) || { echo "--csdr-source-tree-sha256 requires a value" >&2; exit 2; }
            CSDR_SOURCE_TREE_SHA256=$2
            shift 2
            ;;
        --controller)
            (($# >= 2)) || { echo "--controller requires a path" >&2; exit 2; }
            CSDR_CONTROLLER=$2
            shift 2
            ;;
        --stage)
            (($# >= 2)) || { echo "--stage requires a value" >&2; exit 2; }
            STAGE=$2
            shift 2
            ;;
        --q|--selected-q)
            (($# >= 2)) || { echo "--q requires a value" >&2; exit 2; }
            SELECTED_Q=$2
            shift 2
            ;;
        --selected-j)
            (($# >= 2)) || { echo "--selected-j requires a value" >&2; exit 2; }
            SELECTED_J=$2
            shift 2
            ;;
        --selected-nmu)
            (($# >= 2)) || { echo "--selected-nmu requires a value" >&2; exit 2; }
            SELECTED_NMU=$2
            shift 2
            ;;
        --missing)
            (($# >= 2)) || { echo "--missing requires a value" >&2; exit 2; }
            FENCE_KEYS=$2
            shift 2
            ;;
        --manifest-root)
            (($# >= 2)) || { echo "--manifest-root requires a path" >&2; exit 2; }
            MANIFEST_ROOT=$2
            shift 2
            ;;
        --run-id)
            (($# >= 2)) || { echo "--run-id requires a value" >&2; exit 2; }
            RUN_ID=$2
            shift 2
            ;;
        --expected-sdpx-commit)
            (($# >= 2)) || { echo "--expected-sdpx-commit requires a value" >&2; exit 2; }
            EXPECTED_COMMIT=$2
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$STAGE" in alpha|j|nmu|fence) ;; *) echo "invalid --stage: $STAGE" >&2; exit 2 ;; esac
# Canonicalize hexadecimal identities once so driver reports, shell sidecars,
# and controller exact comparisons all use the same lowercase representation.
SDPX_DEPLOYED_COMMIT=$(printf '%s' "$SDPX_DEPLOYED_COMMIT" | tr '[:upper:]' '[:lower:]')
MFLA_DEPLOYED_COMMIT=$(printf '%s' "$MFLA_DEPLOYED_COMMIT" | tr '[:upper:]' '[:lower:]')
CSDR_SOURCE_TREE_SHA256=$(printf '%s' "$CSDR_SOURCE_TREE_SHA256" | tr '[:upper:]' '[:lower:]')
EXPECTED_COMMIT=$(printf '%s' "$EXPECTED_COMMIT" | tr '[:upper:]' '[:lower:]')
if [[ "$STAGE" == j || "$STAGE" == nmu ]]; then
    [[ "$SELECTED_Q" =~ ^(2|3|5|9|17|33)$ ]] || {
        echo "$STAGE stage requires --q in {2,3,5,9,17,33}" >&2
        exit 2
    }
fi
if [[ "$STAGE" == j ]]; then
    [[ "$SELECTED_J" =~ ^(40|80|160|320)$ ]] || {
        echo "j stage requires --selected-j in {40,80,160,320}" >&2
        exit 2
    }
fi
if [[ "$STAGE" == nmu ]]; then
    [[ "$SELECTED_J" =~ ^(40|80|160|320)$ ]] || {
        echo "nmu stage requires --selected-j in {40,80,160,320}" >&2
        exit 2
    }
    [[ "$SELECTED_NMU" =~ ^(400|800|1600|3200)$ ]] || {
        echo "nmu stage requires --selected-nmu in {400,800,1600,3200}" >&2
        exit 2
    }
fi
if [[ "$STAGE" == fence ]]; then
    [[ -n "$FENCE_KEYS" ]] || { echo "fence stage requires --missing" >&2; exit 2; }
fi

: "${CAMPAIGN_ROOT:?--campaign-root (or CAMPAIGN_ROOT) is required}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run id contains unsafe characters" >&2; exit 2; }
MANIFEST_ROOT="${MANIFEST_ROOT:-$CAMPAIGN_ROOT/manifests}/${STAGE}-${RUN_ID}"
if ((DRY_RUN == 0)); then
    : "${SDPX_SOURCE_ROOT:?--sdpx-source-root (or SDPX_SOURCE_ROOT) is required for submission}"
    : "${SDPX_ENV_ROOT:?--sdpx-env-root (or SDPX_ENV_ROOT) is required for submission}"
    : "${CSDR_CONTROLLER:?--controller (or CSDR_CONTROLLER) is required for submission}"
    [[ "$SDPX_DEPLOYED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || {
        echo "SDPX_DEPLOYED_COMMIT must be a validated 40-hex commit" >&2
        exit 2
    }
    [[ "$MFLA_DEPLOYED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || {
        echo "MFLA_DEPLOYED_COMMIT must be a validated 40-hex commit" >&2
        exit 2
    }
    [[ "$CSDR_SOURCE_TREE_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || {
        echo "CSDR_SOURCE_TREE_SHA256 must be a validated 64-hex hash" >&2
        exit 2
    }
fi

mkdir -p "$MANIFEST_ROOT"
solve_manifest="$MANIFEST_ROOT/csdr-grid-solve.tsv"
build_manifest="$MANIFEST_ROOT/csdr-grid-build.tsv"
small_manifest="$MANIFEST_ROOT/csdr-grid-small16.tsv"
medium_manifest="$MANIFEST_ROOT/csdr-grid-medium64.tsv"
high_manifest="$MANIFEST_ROOT/csdr-grid-high64.tsv"

manifest_args=(--stage "$STAGE")
[[ -n "$SELECTED_Q" ]] && manifest_args+=(--q "$SELECTED_Q")
[[ -n "$SELECTED_J" ]] && manifest_args+=(--selected-j "$SELECTED_J")
[[ -n "$SELECTED_NMU" ]] && manifest_args+=(--selected-nmu "$SELECTED_NMU")
[[ -n "$FENCE_KEYS" ]] && manifest_args+=(--missing "$FENCE_KEYS")

"$MANIFEST_TOOL" --phase solve "${manifest_args[@]}" --output "$solve_manifest"
"$MANIFEST_TOOL" --phase build "${manifest_args[@]}" --output "$build_manifest"

{ head -n 1 "$solve_manifest"; awk -F '\t' 'NR > 1 && $13 == "small16"' "$solve_manifest"; } > "$small_manifest"
{ head -n 1 "$solve_manifest"; awk -F '\t' 'NR > 1 && $13 == "medium64"' "$solve_manifest"; } > "$medium_manifest"
{ head -n 1 "$solve_manifest"; awk -F '\t' 'NR > 1 && ($13 == "high64" || $13 == "large64")' "$solve_manifest"; } > "$high_manifest"

solve_rows=$(( $(wc -l < "$solve_manifest") - 1 ))
build_rows=$(( $(wc -l < "$build_manifest") - 1 ))
small_rows=$(( $(wc -l < "$small_manifest") - 1 ))
medium_rows=$(( $(wc -l < "$medium_manifest") - 1 ))
high_rows=$(( $(wc -l < "$high_manifest") - 1 ))
frontier_rows=$(awk -F '\t' 'NR > 1 && $13 == "resource_frontier" {n++} END {print n + 0}' "$solve_manifest")
frontier_manifest="$MANIFEST_ROOT/resource-frontier.tsv"
if ((frontier_rows > 0)); then
    [[ ! -e "$frontier_manifest" ]] || { echo "refusing to overwrite immutable frontier manifest" >&2; exit 3; }
    { head -n 1 "$solve_manifest"; awk -F '\t' 'NR > 1 && $13 == "resource_frontier"' "$solve_manifest"; } > "$frontier_manifest"
    echo "resource_frontier_rows=$frontier_rows" >&2
    if ((DRY_RUN == 0)); then
        while IFS=$'\t' read -r phase stage case_id j na nmu alpha_level alpha_count frontier_max_count predicted_peak_gib nred solver_threads resource_class requested_cores memory_gb walltime cache_alpha_level cache_alpha_count cache_rel result_rel build_rel predecessor_key; do
            [[ "$resource_class" == resource_frontier ]] || continue
            result_root="$CAMPAIGN_ROOT/$result_rel"
            mkdir -p "$result_root"
            if [[ ! -e "$result_root/result.toml" ]]; then
                frontier_tmp="$result_root/result.toml.part.$$"
                {
                    echo 'status="RESOURCE_FRONTIER"'
                    echo "l_max=$j"; echo "N_a=$na"; echo "N_mu=$nmu"; echo 'N_x=1'
                    echo "alpha_level=$alpha_level"; echo "alpha_count=$alpha_count"
                    echo 'solve_arithmetic="Float64x2"'; echo 'precompute_precision_bits=256'
                    echo 'memory_estimate_gate_valid=false'; echo 'cache_digest_gate_valid=false'
                    echo 'certificate_valid=false'; echo 'numerical_gate_valid=false'
                    echo 'la_planned_provider="multifloat_linear_algebra"'; echo 'la_executed_provider="multifloat_linear_algebra"'
                    echo 'fallback_reason="none"'; echo 'la_fallback_reason="none"'
                    echo "sdpx_commit=\"$SDPX_DEPLOYED_COMMIT\""; echo "mfla_commit=\"$MFLA_DEPLOYED_COMMIT\""
                    echo "csdr_source_tree_sha256=\"$CSDR_SOURCE_TREE_SHA256\""; echo 'cache_sha256=""'
                    echo 'resource_frontier="unresolved_at_resource_frontier"'
                } > "$frontier_tmp"
                mv "$frontier_tmp" "$result_root/result.toml"
                touch "$result_root/SOLVE_COMPLETE"
            fi
        done < <(tail -n +2 "$solve_manifest")
    fi
fi
compute_rows=$((small_rows + medium_rows + high_rows))
if ((compute_rows > 0 && build_rows == 0)); then
    echo "selected stage has feasible solves but no cache-build rows" >&2
    exit 2
fi
((small_rows + medium_rows + high_rows + frontier_rows == solve_rows)) || { echo "class split is incomplete" >&2; exit 2; }

small_core_sum=$(awk -F '\t' 'NR > 1 {s += $14} END {print s + 0}' "$small_manifest")
medium_core_sum=$(awk -F '\t' 'NR > 1 {s += $14} END {print s + 0}' "$medium_manifest")
high_core_sum=$(awk -F '\t' 'NR > 1 {s += $14} END {print s + 0}' "$high_manifest")
((small_core_sum <= MAX_CORES && medium_core_sum <= MAX_CORES && high_core_sum <= MAX_CORES)) || { echo "class manifest exceeds core cap" >&2; exit 2; }
build_wave_cores=$((16 * 48))
small_wave_cores=$((16 * 48))
large_wave_cores=$((64 * 12))
validate_wave_cores=$((8 * 96))
((build_wave_cores <= MAX_CORES && small_wave_cores <= MAX_CORES &&
  large_wave_cores <= MAX_CORES && validate_wave_cores <= MAX_CORES)) || {
    echo "array throttle arithmetic exceeds cap" >&2
    exit 2
}
((small_wave_cores == MAX_CORES && large_wave_cores == MAX_CORES)) || {
    echo "solve wave arithmetic is not exactly capped at MAX_CORES" >&2
    exit 2
}
solve_total_core_work=$((small_core_sum + medium_core_sum + high_core_sum))
solve_peak_core_allocation=$large_wave_cores
((solve_peak_core_allocation <= MAX_CORES)) || {
    echo "serialized solve peak exceeds core cap" >&2
    exit 2
}

print_qsub() {
    printf '+ qsub'
    local arg
    for arg in "$@"; do printf ' %q' "$arg"; done
    printf '\n'
}

submit_qsub() {
    local name=$1
    shift
    if ((DRY_RUN)); then
        print_qsub "$@" >&2
        printf 'dry_run_job=%s\n' "$name" >&2
        echo "dry-${name}"
    else
        command qsub "$@"
    fi
}

qsub_vars=(
    "CAMPAIGN_ROOT=$CAMPAIGN_ROOT"
    "SDPX_SOURCE_ROOT=$SDPX_SOURCE_ROOT"
    "SDPX_ENV_ROOT=$SDPX_ENV_ROOT"
    "CSDR_SOURCE_ROOT=${CSDR_SOURCE_ROOT:-$CAMPAIGN_ROOT/csdr-source}"
    "CSDR_DRIVER=${CSDR_DRIVER:-$CAMPAIGN_ROOT/driver/g0_max_bigfloat256_float64x2.jl}"
    "CSDR_CONTROLLER=$CSDR_CONTROLLER"
    "CSDR_STAGE=$STAGE"
    "CSDR_RUN_ID=$RUN_ID"
    "SDPX_DEPLOYED_COMMIT=$SDPX_DEPLOYED_COMMIT"
    "MFLA_DEPLOYED_COMMIT=$MFLA_DEPLOYED_COMMIT"
    "CSDR_SOURCE_TREE_SHA256=$CSDR_SOURCE_TREE_SHA256"
)
[[ -n "$EXPECTED_COMMIT" ]] && qsub_vars+=("CSDR_EXPECTED_SDPX_COMMIT=$EXPECTED_COMMIT")
qsub_var_string=$(IFS=,; echo "${qsub_vars[*]}")

echo "campaign_root=$CAMPAIGN_ROOT"
echo "manifest_root=$MANIFEST_ROOT"
echo "stage=$STAGE run_id=$RUN_ID selected_q=${SELECTED_Q:-none} selected_j=${SELECTED_J:-none} selected_nmu=${SELECTED_NMU:-none}"
echo "solve_rows=$solve_rows build_rows=$build_rows small16_rows=$small_rows medium64_rows=$medium_rows high64_rows=$high_rows frontier_rows=$frontier_rows"
echo "small16_core_sum=$small_core_sum medium64_core_sum=$medium_core_sum high64_core_sum=$high_core_sum solve_total_core_work=$solve_total_core_work solve_peak_core_allocation=$solve_peak_core_allocation max_cores=$MAX_CORES"
((DRY_RUN)) && echo "mode=dry-run (qsub is not invoked)"

build_range="0-$((build_rows - 1))%48"
build_job=""
if ((build_rows > 0)); then
    build_job=$(submit_qsub build \
        -N csdr-cache-build \
        -t "$build_range" \
        -v "$qsub_var_string,CSDR_BUILD_MANIFEST=$build_manifest" \
        "$BUILD_PBS")
fi

small_preflight=""
if ((small_rows > 0)); then
    small_range="0-$((small_rows - 1))%48"
    small_preflight=$(submit_qsub preflight-small \
        -N csdr-preflight-small16 -t "$small_range" \
        -W "depend=afterokarray:${build_job}" \
        -l nodes=1:ppn=16,mem=96gb,walltime=02:00:00 \
        -v "$qsub_var_string,CSDR_PREFLIGHT_MANIFEST=$small_manifest" \
        "$PREFLIGHT_PBS")
fi
small_job=""
if ((small_rows > 0)); then
    small_range="0-$((small_rows - 1))%48"
    small_job=$(submit_qsub solve-small \
        -N csdr-solve-small16 \
        -t "$small_range" \
        -W "depend=afterokarray:${small_preflight}" \
        -l nodes=1:ppn=16,mem=96gb,walltime=04:00:00 \
        -v "$qsub_var_string,CSDR_SOLVE_MANIFEST=$small_manifest" \
        "$SOLVE_PBS")
fi

medium_preflight=""
if ((medium_rows > 0)); then
    medium_range="0-$((medium_rows - 1))%12"
    medium_preflight=$(submit_qsub preflight-medium \
        -N csdr-preflight-medium64 -t "$medium_range" \
        -W "depend=afterokarray:${small_job:-$build_job}" \
        -l nodes=1:ppn=64,mem=160gb,walltime=02:00:00 \
        -v "$qsub_var_string,CSDR_PREFLIGHT_MANIFEST=$medium_manifest" \
        "$PREFLIGHT_PBS")
fi

medium_job=""
medium_dependency="$medium_preflight"
if ((medium_rows > 0)); then
    medium_range="0-$((medium_rows - 1))%12"
    medium_job=$(submit_qsub solve-medium \
        -N csdr-solve-medium64 \
        -t "$medium_range" \
        -W "depend=afterokarray:${medium_dependency}" \
        -l nodes=1:ppn=64,mem=160gb,walltime=12:00:00 \
        -v "$qsub_var_string,CSDR_SOLVE_MANIFEST=$medium_manifest" \
        "$SOLVE_PBS")
fi

high_preflight=""
if ((high_rows > 0)); then
    high_range="0-$((high_rows - 1))%12"
    high_preflight=$(submit_qsub preflight-high \
        -N csdr-preflight-high64 -t "$high_range" \
        -W "depend=afterokarray:${medium_job:-${small_job:-$build_job}}" \
        -l nodes=1:ppn=64,mem=224gb,walltime=02:00:00 \
        -v "$qsub_var_string,CSDR_PREFLIGHT_MANIFEST=$high_manifest" \
        "$PREFLIGHT_PBS")
fi

high_job=""
high_dependency="$high_preflight"
if ((high_rows > 0)); then
    high_range="0-$((high_rows - 1))%12"
    high_job=$(submit_qsub solve-high \
        -N csdr-solve-high64 \
        -t "$high_range" \
        -W "depend=afterokarray:${high_dependency}" \
        -l nodes=1:ppn=64,mem=224gb,walltime=48:00:00 \
        -v "$qsub_var_string,CSDR_SOLVE_MANIFEST=$high_manifest" \
        "$SOLVE_PBS")
fi

solve_dependencies=()
[[ -n "$small_job" ]] && solve_dependencies+=("afteranyarray:${small_job}")
[[ -n "$medium_job" ]] && solve_dependencies+=("afteranyarray:${medium_job}")
[[ -n "$high_job" ]] && solve_dependencies+=("afteranyarray:${high_job}")
validation_dep=$(IFS=,; echo "${solve_dependencies[*]}")
validate_range="0-$((solve_rows - 1))%96"
validate_args=(-N csdr-campaign-validate -t "$validate_range")
[[ -n "$validation_dep" ]] && validate_args+=(-W "depend=${validation_dep}")
validate_args+=(-v "$qsub_var_string,CSDR_SOLVE_MANIFEST=$solve_manifest" "$VALIDATOR")
validate_job=$(submit_qsub validate "${validate_args[@]}")

aggregate_job=$(submit_qsub aggregate \
    -N csdr-campaign-aggregate \
    -W "depend=afteranyarray:${validate_job}" \
    -v "$qsub_var_string,CSDR_SOLVE_MANIFEST=$solve_manifest" \
    "$VALIDATOR")

controller_job=$(submit_qsub controller \
    -N csdr-campaign-controller \
    -W "depend=afterany:${aggregate_job}" \
    -v "$qsub_var_string,CSDR_SOLVE_MANIFEST=$solve_manifest" \
    "$CONTROLLER_PBS")

echo "build_job=$build_job"
echo "small_preflight_job=$small_preflight"
echo "medium_preflight_job=$medium_preflight"
echo "high_preflight_job=$high_preflight"
echo "small_solve_job=$small_job"
echo "medium_solve_job=$medium_job"
echo "high_solve_job=$high_job"
echo "validation_job=$validate_job"
echo "aggregate_job=$aggregate_job"
echo "controller_job=$controller_job"
