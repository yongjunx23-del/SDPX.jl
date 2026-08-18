#!/usr/bin/env bash
# Validate immutable solve artifacts or aggregate a campaign summary.
#
# In array mode set CSDR_SOLVE_MANIFEST and PBS_ARRAY_INDEX/PBS_ARRAYID. In
# controller mode omit the array index; all rows are checked and a summary is
# atomically published. The validator is intentionally shell-only so it can run
# after a PBS dependency without changing the Julia/SDPX source tree.
#PBS -N csdr-campaign-validate
#PBS -q normal
#PBS -l nodes=1:ppn=8
#PBS -l mem=8gb
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

: "${CAMPAIGN_ROOT:?set CAMPAIGN_ROOT}"
: "${CSDR_SOLVE_MANIFEST:?set CSDR_SOLVE_MANIFEST}"
: "${SDPX_SOURCE_ROOT:?set SDPX_SOURCE_ROOT}"
: "${SDPX_ENV_ROOT:?set SDPX_ENV_ROOT}"
: "${SDPX_DEPLOYED_COMMIT:?set SDPX_DEPLOYED_COMMIT}"
: "${MFLA_DEPLOYED_COMMIT:?set MFLA_DEPLOYED_COMMIT}"
: "${CSDR_SOURCE_TREE_SHA256:?set CSDR_SOURCE_TREE_SHA256}"
[[ "$SDPX_DEPLOYED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "SDPX_DEPLOYED_COMMIT must be 40-hex" >&2
    exit 2
}
[[ "$MFLA_DEPLOYED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "MFLA_DEPLOYED_COMMIT must be 40-hex" >&2
    exit 2
}
[[ "$CSDR_SOURCE_TREE_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || {
    echo "CSDR_SOURCE_TREE_SHA256 must be 64-hex" >&2
    exit 2
}

readonly ARRAY_INDEX="${CSDR_VALIDATE_INDEX:-${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}}"
readonly VALIDATION_STAGE="${CSDR_STAGE:-unscoped}"
readonly VALIDATION_RUN="${CSDR_RUN_ID:-manual-$$}"
readonly VALIDATION_ROOT="${CAMPAIGN_ROOT}/validation/${VALIDATION_STAGE}-${VALIDATION_RUN}"

toml_value() {
    local key=$1 file=$2
    sed -n -E \
        "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*\"?([^\"[:space:]]+)\"?.*$/\1/p" \
        "$file" | tail -n 1
}

require_value() {
    local expected=$1 key=$2 file=$3 actual
    actual=$(toml_value "$key" "$file")
    [[ -n "$actual" && "$actual" == "$expected" ]] || {
        echo "$file: $key expected=$expected actual=${actual:-<missing>}" >&2
        return 1
    }
}

validate_row() {
    local index=$1 row
    row=$(sed -n "$((index + 2))p" "$CSDR_SOLVE_MANIFEST")
    [[ -n "$row" ]] || { echo "missing solve manifest row $index" >&2; return 1; }
    local phase stage case_id j na nmu alpha_level alpha_count frontier_max_count
    local predicted_peak_gib nred solver_threads resource_class requested_cores memory_gb walltime
    local cache_alpha_level cache_alpha_count cache_rel result_rel build_rel predecessor_key
    local result_root result cache_hash expected_hash cache_selection_origin peak_rss_bytes driver_peak process_peak request_bytes
    IFS=$'\t' read -r phase stage case_id j na nmu alpha_level alpha_count frontier_max_count \
        predicted_peak_gib nred solver_threads resource_class requested_cores memory_gb walltime \
        cache_alpha_level cache_alpha_count cache_rel result_rel build_rel predecessor_key <<< "$row"
    [[ "$phase" == solve ]] || { echo "row $index is not solve" >&2; return 1; }
    result_root="$CAMPAIGN_ROOT/$result_rel"
    result="$result_root/result.toml"
    cache_hash="$result_root/input.sha256"
    [[ -s "$result" ]] || { echo "$case_id: missing result.toml" >&2; return 1; }
    [[ -s "$result_root/provenance.txt" ]] || {
        echo "$case_id: missing provenance.txt" >&2
        return 1
    }
    [[ -s "$cache_hash" ]] || { echo "$case_id: missing input.sha256" >&2; return 1; }
    [[ -e "$result_root/SOLVE_COMPLETE" ]] || {
        echo "$case_id: missing SOLVE_COMPLETE" >&2
        return 1
    }

    local expected_cache="$CAMPAIGN_ROOT/$cache_rel"
    [[ -s "$expected_cache" ]] || { echo "$case_id: missing cache" >&2; return 1; }
    expected_hash=$(sha256sum "$expected_cache" | awk '{print $1}')
    grep -q "^$expected_hash  $expected_cache$" "$cache_hash" || {
        echo "$case_id: cache hash absent from input.sha256" >&2
        return 1
    }

    require_value Optimal status "$result"
    require_value true certificate_valid "$result"
    require_value true numerical_gate_valid "$result"
    require_value "$j" l_max "$result"
    require_value "$na" N_a "$result"
    require_value "$nmu" N_mu "$result"
    require_value "$alpha_count" alpha_count "$result"
    require_value Float64x2 solve_arithmetic "$result"
    require_value "$solver_threads" solver_threads "$result"
    require_value 2 cache_contract_version "$result"
    require_value 2 cache_schema_version "$result"
    ((alpha_count <= cache_alpha_count)) || {
        echo "$case_id: requested alpha count exceeds cache maximum" >&2
        return 1
    }
    require_value true memory_estimate_gate_valid "$result"
    require_value true cache_digest_gate_valid "$result"
    require_value "$cache_alpha_level" cache_alpha_level "$result"
    require_value "$cache_alpha_count" cache_alpha_count "$result"
    require_value "$cache_alpha_level" cache_max_alpha_level "$result"
    require_value "$cache_alpha_count" cache_max_alpha_count "$result"
    cache_selection_origin=$(toml_value cache_alpha_selection_origin "$result")
    [[ -n "$cache_selection_origin" && "$cache_selection_origin" != "none" ]] || {
        echo "$case_id: cache alpha selection origin is missing" >&2
        return 1
    }
    require_value "$expected_hash" cache_sha256 "$result"
    driver_peak=$(toml_value peak_rss_bytes "$result"); process_peak=$(toml_value process_peak_rss_bytes "$result")
    [[ "$driver_peak" =~ ^[0-9]+$ ]] || driver_peak=0
    [[ "$process_peak" =~ ^[0-9]+$ ]] || process_peak=0
    peak_rss_bytes=$((driver_peak > process_peak ? driver_peak : process_peak))
    ((peak_rss_bytes > 0)) || { echo "$case_id: missing peak RSS evidence" >&2; return 1; }
    ((peak_rss_bytes <= 176 * 1024 * 1024 * 1024)) || { echo "$case_id: peak RSS exceeds 70% node budget" >&2; return 1; }
    request_bytes=$((memory_gb * 1024 * 1024 * 1024))
    ((request_bytes * 4 >= peak_rss_bytes * 5)) || { echo "$case_id: PBS memory request lacks 1.25x RSS headroom" >&2; return 1; }
    require_value "$SDPX_DEPLOYED_COMMIT" sdpx_commit "$result"
    require_value "$MFLA_DEPLOYED_COMMIT" mfla_commit "$result"
    require_value "$CSDR_SOURCE_TREE_SHA256" csdr_source_tree_sha256 "$result"
    require_value "$expected_hash" cache_sha256 "$result_root/provenance.txt"
    require_value multifloat_linear_algebra la_planned_provider "$result"
    require_value multifloat_linear_algebra la_executed_provider "$result"
    require_value none fallback_reason "$result"
    require_value none la_fallback_reason "$result"
    local mfla_module_hash mfla_package_hash
    mfla_module_hash=$(toml_value multifloat_linear_algebra_module_sha256 "$result")
    mfla_package_hash=$(toml_value multifloat_linear_algebra_package_tree_sha256 "$result")
    [[ "$mfla_module_hash" =~ ^[0-9a-fA-F]{64}$ &&
       "$mfla_package_hash" =~ ^[0-9a-fA-F]{64}$ ]] || {
        echo "$case_id: MFLA module/package fingerprints are not 64-hex" >&2
        return 1
    }
    require_value SDPX orchestration_owner "$result_root/provenance.txt"
    require_value "$resource_class" resource_class "$result_root/provenance.txt"
    require_value "$requested_cores" allocated_slots "$result_root/provenance.txt"
    require_value "$memory_gb" requested_memory_gb "$result_root/provenance.txt"
    require_value "$walltime" requested_walltime "$result_root/provenance.txt"
    require_value "${SDPX_SOURCE_ROOT:?set SDPX_SOURCE_ROOT for validation}" \
        sdpx_source_root "$result_root/provenance.txt"
    require_value "${SDPX_ENV_ROOT:?set SDPX_ENV_ROOT for validation}" \
        sdpx_env_root "$result_root/provenance.txt"
    require_value "$SDPX_DEPLOYED_COMMIT" sdpx_deployed_commit "$result_root/provenance.txt"
    require_value "$MFLA_DEPLOYED_COMMIT" mfla_commit "$result_root/provenance.txt"
    require_value "$CSDR_SOURCE_TREE_SHA256" csdr_source_tree_sha256 "$result_root/provenance.txt"
    require_value "$cache_alpha_level" cache_alpha_level "$result_root/provenance.txt"
    require_value "$cache_alpha_count" cache_alpha_count "$result_root/provenance.txt"
    require_value multifloat_linear_algebra la_planned_provider "$result_root/provenance.txt"
    require_value multifloat_linear_algebra la_executed_provider "$result_root/provenance.txt"
    require_value none fallback_reason "$result_root/provenance.txt"
    require_value none la_fallback_reason "$result_root/provenance.txt"
    local driver_hash_result driver_hash_provenance expected_driver
    driver_hash_result=$(toml_value driver_sha256 "$result")
    driver_hash_provenance=$(toml_value driver_sha256 "$result_root/provenance.txt")
    [[ "$driver_hash_result" =~ ^[0-9a-fA-F]{64}$ &&
       "$driver_hash_result" == "$driver_hash_provenance" ]] || {
        echo "$case_id: driver hash missing or differs between result/provenance" >&2
        return 1
    }
    expected_driver="${CSDR_DRIVER:-$CAMPAIGN_ROOT/driver/g0_max_bigfloat256_float64x2.jl}"
    if [[ -f "$expected_driver" ]]; then
        [[ "$driver_hash_result" == "$(sha256sum "$expected_driver" | awk '{print $1}')" ]] || {
            echo "$case_id: driver hash differs from deployed driver" >&2
            return 1
        }
    fi
    require_value "$SDPX_DEPLOYED_COMMIT" sdpx_commit "$result_root/provenance.txt"
    local active_sdpx source_hash_file build_source_hash
    active_sdpx=$(toml_value active_sdpx "$result_root/provenance.txt")
    case "$active_sdpx" in
        "$SDPX_SOURCE_ROOT"/*) ;;
        *) echo "$case_id: active SDPX path mismatch: $active_sdpx" >&2; return 1 ;;
    esac
    source_hash_file=$(toml_value source_hash_file "$result_root/provenance.txt")
    [[ -s "$source_hash_file" ]] || {
        echo "$case_id: missing source hash file $source_hash_file" >&2
        return 1
    }
    build_source_hash="$CAMPAIGN_ROOT/$build_rel/source.sha256"
    cmp -s "$build_source_hash" "$source_hash_file" || {
        echo "$case_id: source hash differs from cache build" >&2
        return 1
    }

    local validation="$result_root/validation.toml"
    if [[ -e "$validation" ]]; then
        grep -q '^status=PASS$' "$validation" || {
            echo "$case_id: existing validation is not PASS" >&2
            return 1
        }
    else
        local tmp="$validation.part.${PBS_JOBID:-$$}"
        {
            echo "status=PASS"
            echo "case_id=$case_id"
            echo "manifest_index=$index"
            echo "result=$result"
            echo "cache_sha256=$expected_hash"
            echo "validated_utc=$(date -u +%FT%TZ)"
        } > "$tmp"
        mv "$tmp" "$validation"
    fi
    printf '%s\tPASS\t%s\n' "$case_id" "$result"
}

aggregate() {
    mkdir -p "$VALIDATION_ROOT"
    local summary="$VALIDATION_ROOT/campaign_summary.tsv"
    local summary_toml="$VALIDATION_ROOT/campaign_summary.toml"
    local frontier_summary="$VALIDATION_ROOT/resource-frontier.tsv"
    local tmp="$summary.part.${PBS_JOBID:-$$}"
    local frontier_tmp="$frontier_summary.part.${PBS_JOBID:-$$}"
    [[ ! -e "$summary" && ! -e "$summary_toml" && ! -e "$frontier_summary" ]] || {
        echo "refusing to overwrite immutable campaign summary: $VALIDATION_ROOT" >&2
        return 3
    }
    local total=0 passed=0 failed=0 frontier_count=0 index
    printf 'case_id\tstatus\tresult\n' > "$tmp"
    printf 'phase\tJ\tN_mu\talpha_count\treason\tevidence\n' > "$frontier_tmp"
    while IFS= read -r index; do
        [[ -n "$index" ]] || continue
        total=$((total + 1))
        if line=$(validate_row "$index"); then
            printf '%s\n' "$line" >> "$tmp"
            passed=$((passed + 1))
        else
            row=$(sed -n "$((index + 2))p" "$CSDR_SOLVE_MANIFEST")
            case_id=$(printf '%s\n' "$row" | cut -f 3)
            stage=$(printf '%s\n' "$row" | cut -f 2)
            j=$(printf '%s\n' "$row" | cut -f 4)
            nmu=$(printf '%s\n' "$row" | cut -f 6)
            alpha_count=$(printf '%s\n' "$row" | cut -f 8)
            build_rel=$(printf '%s\n' "$row" | cut -f 21)
            result_rel=$(printf '%s\n' "$row" | cut -f 20)
            status=FAIL
            evidence="$CAMPAIGN_ROOT/$result_rel/result.toml"
            preflight_evidence="$CAMPAIGN_ROOT/preflight/$stage/j${j}/nmu${nmu}/q${alpha_count}/preflight.toml"
            if [[ -s "$preflight_evidence" ]] &&
               grep -q '^resource_frontier="unresolved_at_resource_frontier"' "$preflight_evidence"; then
                status=RESOURCE_FRONTIER
                evidence="$preflight_evidence"
            elif [[ -s "$CAMPAIGN_ROOT/$build_rel/resource_frontier.toml" ||
                    -s "$CAMPAIGN_ROOT/$result_rel/resource_frontier.toml" ]]; then
                status=RESOURCE_FRONTIER
                evidence="$CAMPAIGN_ROOT/$build_rel/resource_frontier.toml"
                [[ -s "$evidence" ]] || evidence="$CAMPAIGN_ROOT/$result_rel/resource_frontier.toml"
            elif [[ -s "$CAMPAIGN_ROOT/$result_rel/result.toml" ]] &&
                 grep -q '^status="RESOURCE_FRONTIER"' "$CAMPAIGN_ROOT/$result_rel/result.toml" &&
                 grep -q '^resource_frontier=' "$CAMPAIGN_ROOT/$result_rel/result.toml"; then
                status=RESOURCE_FRONTIER
                evidence="$CAMPAIGN_ROOT/$result_rel/result.toml"
            fi
            printf '%s\t%s\t%s\n' "$case_id" "$status" "$evidence" >> "$tmp"
            if [[ "$status" == RESOURCE_FRONTIER ]]; then
                printf 'resource_frontier\t%s\t%s\t%s\truntime_resource_frontier\t%s\n' \
                    "$j" "$nmu" "$alpha_count" "$evidence" >> "$frontier_tmp"
                frontier_count=$((frontier_count + 1))
            fi
            failed=$((failed + 1))
        fi
    done < <(awk -F '\t' 'NR > 1 && $1 == "solve" {print NR - 2}' "$CSDR_SOLVE_MANIFEST")
    mv "$tmp" "$summary"
    {
        echo "total=$total"
        echo "passed=$passed"
        echo "failed=$failed"
        echo "completed_utc=$(date -u +%FT%TZ)"
    } > "$summary_toml"
    if ((frontier_count > 0)); then
        mv "$frontier_tmp" "$frontier_summary"
    else
        rm -f "$frontier_tmp"
    fi
    ((failed == 0)) || return 1
}

if [[ -n "$ARRAY_INDEX" ]]; then
    mkdir -p "$VALIDATION_ROOT"
    row_output="$VALIDATION_ROOT/row-${ARRAY_INDEX}.tsv"
    [[ ! -e "$row_output" ]] || {
        echo "refusing to overwrite immutable validation row: $row_output" >&2
        exit 3
    }
    validate_row "$ARRAY_INDEX" > "$row_output"
else
    aggregate
fi
