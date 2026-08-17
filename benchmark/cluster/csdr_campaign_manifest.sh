#!/usr/bin/env bash
# Canonical adaptive campaign manifest generator.
# One solve row represents one mathematical point; build rows are deduplicated
# by (J,N_mu,cache_max_alpha_count). The certified default is alpha seed
# (J=40,N_mu=400,q=2); later actions are explicit controller selections.
set -euo pipefail

PHASE=solve
STAGE=alpha
OUTPUT=-
SELECTED_Q=
SELECTED_J=
SELECTED_NMU=
FENCE_KEYS=

while (($#)); do
    case "$1" in
        --phase) (($# >= 2)) || exit 2; PHASE=$2; shift 2 ;;
        --stage) (($# >= 2)) || exit 2; STAGE=$2; shift 2 ;;
        --q|--selected-q) (($# >= 2)) || exit 2; SELECTED_Q=$2; shift 2 ;;
        --selected-j) (($# >= 2)) || exit 2; SELECTED_J=$2; shift 2 ;;
        --selected-nmu) (($# >= 2)) || exit 2; SELECTED_NMU=$2; shift 2 ;;
        --missing) (($# >= 2)) || exit 2; FENCE_KEYS=$2; shift 2 ;;
        --output) (($# >= 2)) || exit 2; OUTPUT=$2; shift 2 ;;
        --help|-h)
            printf '%s\n' \
                'stage alpha: default seed J40/Nmu400/q2; pass --q for one alpha action' \
                'stage j: one J at --selected-j/--q, Nmu400' \
                'stage nmu: one Nmu at --selected-j/--selected-nmu/--q' \
                'stage fence: --missing J:Nmu:q[,J:Nmu:q...]'
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$PHASE" in solve|build) ;; *) echo "bad phase: $PHASE" >&2; exit 2 ;; esac
case "$STAGE" in alpha|j|nmu|fence) ;; *) echo "bad stage: $STAGE" >&2; exit 2 ;; esac

readonly COLUMNS=$'phase	stage	case_id	J	N_a	N_mu	alpha_level	alpha_count	frontier_max_count	predicted_peak_gib	nred	solver_threads	resource_class	requested_cores	memory_gb	walltime	cache_alpha_level	cache_alpha_count	cache_rel	result_rel	build_rel	predecessor_key'

level_of() {
    case "$1" in
        2) echo 1 ;; 3) echo 2 ;; 5) echo 3 ;; 9) echo 4 ;; 17) echo 5 ;; 33) echo 6 ;;
        *) return 1 ;;
    esac
}

frontier_of() {
    case "$1:$2" in
        40:400|40:800|40:1600|40:3200) echo 33 ;;
        80:400|80:800|80:1600) echo 33 ;; 80:3200) echo 17 ;;
        160:400) echo 33 ;; 160:800) echo 17 ;; 160:1600) echo 9 ;; 160:3200) echo 5 ;;
        320:400) echo 9 ;; 320:800) echo 5 ;; 320:1600) echo 3 ;; 320:3200) echo 2 ;;
        *) return 1 ;;
    esac
}

na_of() { echo $((3 * $1 / 8)); }
valid_q() { case "$1" in 2|3|5|9|17|33) return 0 ;; *) return 1 ;; esac; }
valid_j() { case "$1" in 40|80|160|320) return 0 ;; *) return 1 ;; esac; }
valid_nmu() { case "$1" in 400|800|1600|3200) return 0 ;; *) return 1 ;; esac; }

# Versioned conservative memory lookup in GiB. Values are upper bounds for the
# certified driver contract at node_memory_gb=176 (70% of a 256-GiB node).
peak_of() {
    case "$1:$2:$3" in
        40:400:*) echo 12 ;; 40:800:*) echo 24 ;; 40:1600:*) echo 48 ;; 40:3200:*) echo 96 ;;
        80:400:*) echo 24 ;; 80:800:*) echo 48 ;; 80:1600:*) echo 96 ;; 80:3200:*) echo 160 ;;
        160:400:*) echo 48 ;; 160:800:*) echo 96 ;; 160:1600:*) echo 160 ;; 160:3200:*) echo 176 ;;
        320:400:*) echo 96 ;; 320:800:*) echo 160 ;; 320:1600:*) echo 176 ;; 320:3200:*) echo 176 ;;
        *) return 1 ;;
    esac
}

emit_frontier() {
    local stage=$1 j=$2 nmu=$3 q=$4 maxq=$5 predecessor=${6:-none}
    local na level maxlevel peak nred case_id cache_rel result_rel build_rel
    na=$(na_of "$j"); level=$(level_of "$q"); maxlevel=$(level_of "$maxq")
    peak=$(peak_of "$j" "$nmu" "$q"); nred=$((2 * q * na - 3)); ((nred < 0)) && nred=0
    case_id="${stage}-j${j}-nmu${nmu}-q${q}-resource-frontier"
    cache_rel="cache/j${j}/nmu${nmu}/alpha${maxq}.bin"
    build_rel="build/j${j}/nmu${nmu}/alpha${maxq}"
    result_rel="results/${stage}/j${j}/nmu${nmu}/q${q}/resource_frontier"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        solve "$stage" "$case_id" "$j" "$na" "$nmu" "$level" "$q" "$maxq" "$peak" "$nred" \
        0 resource_frontier 0 0 00:00:00 "$maxlevel" "$maxq" "$cache_rel" "$result_rel" "$build_rel" "$predecessor"
}

emit_point() {
    local stage=$1 j=$2 nmu=$3 q=$4 predecessor=${5:-none}
    valid_j "$j" && valid_nmu "$nmu" && valid_q "$q" || return 2
    local maxq maxlevel level na peak nred class cores memory wall case_id cache_rel result_rel build_rel
    maxq=$(frontier_of "$j" "$nmu")
    if ((q > maxq)); then emit_frontier "$stage" "$j" "$nmu" "$q" "$maxq" "$predecessor"; return 0; fi
    maxlevel=$(level_of "$maxq"); level=$(level_of "$q"); na=$(na_of "$j"); peak=$(peak_of "$j" "$nmu" "$q")
    nred=$((2 * q * na - 3)); ((nred < 0)) && nred=0
    if ((peak <= 32 && nred <= 512)); then
        class=small16; cores=16; memory=96; wall=04:00:00
    elif ((peak <= 96)); then
        class=medium64; cores=64; memory=160; wall=12:00:00
    else
        class=high64; cores=64; memory=224; wall=48:00:00
    fi
    case_id="${stage}-j${j}-nmu${nmu}-q${q}"
    cache_rel="cache/j${j}/nmu${nmu}/alpha${maxq}.bin"
    build_rel="build/j${j}/nmu${nmu}/alpha${maxq}"
    result_rel="results/${stage}/j${j}/nmu${nmu}/q${q}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        solve "$stage" "$case_id" "$j" "$na" "$nmu" "$level" "$q" "$maxq" "$peak" "$nred" \
        "$cores" "$class" "$cores" "$memory" "$wall" "$maxlevel" "$maxq" "$cache_rel" "$result_rel" "$build_rel" "$predecessor"
}

emit_solve() {
    printf '%s\n' "$COLUMNS"
    case "$STAGE" in
        alpha)
            [[ -z "$SELECTED_Q" ]] && SELECTED_Q=2
            emit_point alpha 40 400 "$SELECTED_Q" alpha-seed
            ;;
        j)
            [[ -n "$SELECTED_Q" && -n "$SELECTED_J" ]] || { echo 'stage j requires --q and --selected-j' >&2; return 2; }
            emit_point j "$SELECTED_J" 400 "$SELECTED_Q" "alpha-j40-nmu400-q${SELECTED_Q}"
            ;;
        nmu)
            [[ -n "$SELECTED_Q" && -n "$SELECTED_J" && -n "$SELECTED_NMU" ]] || {
                echo 'stage nmu requires --q, --selected-j, and --selected-nmu' >&2; return 2;
            }
            emit_point nmu "$SELECTED_J" "$SELECTED_NMU" "$SELECTED_Q" \
                "j-j${SELECTED_J}-nmu400-q${SELECTED_Q}"
            ;;
        fence)
            [[ -n "$FENCE_KEYS" ]] || { echo 'stage fence requires --missing' >&2; return 2; }
            local key
            while IFS= read -r key; do
                [[ -n "$key" ]] || continue
                if [[ "$key" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                    emit_point fence "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" fence-missing
                elif [[ "$key" =~ ^J=([0-9]+),Nmu=([0-9]+),alpha=([0-9]+)$ ]]; then
                    emit_point fence "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" fence-missing
                elif [[ "$key" =~ ^j([0-9]+)-nmu([0-9]+)-q([0-9]+)$ ]]; then
                    emit_point fence "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" fence-missing
                else
                    echo "unparseable fence predecessor key: $key" >&2; return 2
                fi
            done < <(printf '%s' "$FENCE_KEYS" | tr ',' '\n')
            ;;
    esac
}

emit_build() {
    printf '%s\n' "$COLUMNS"
    local tmp phase stage case_id j na nmu level q maxq peak nred threads class cores memory wall cache_level cache_q cache_rel result_rel build_rel predecessor
    local -A seen=()
    tmp=$(mktemp)
    emit_solve > "$tmp"
    while IFS=$'\t' read -r phase stage case_id j na nmu level q maxq peak nred threads class cores memory wall cache_level cache_q cache_rel result_rel build_rel predecessor; do
        [[ "$phase" == solve && "$class" != resource_frontier ]] || continue
        local key="${j}:${nmu}:${cache_q}"
        [[ -n "${seen[$key]:-}" ]] && continue
        seen[$key]=1
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            build "$stage" "build-j${j}-nmu${nmu}-q${cache_q}" "$j" "$na" "$nmu" "$cache_level" "$cache_q" "$maxq" "$peak" "$nred" \
            16 build16 16 224 48:00:00 "$cache_level" "$cache_q" "$cache_rel" - "$build_rel" none
    done < <(tail -n +2 "$tmp")
    rm -f "$tmp"
}

if [[ "$OUTPUT" == - ]]; then
    if [[ "$PHASE" == build ]]; then emit_build; else emit_solve; fi
else
    mkdir -p "$(dirname "$OUTPUT")"
    [[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite immutable manifest: $OUTPUT" >&2; exit 3; }
    tmp="${OUTPUT}.part.$$"
    trap 'rm -f "$tmp"' EXIT
    if [[ "$PHASE" == build ]]; then emit_build > "$tmp"; else emit_solve > "$tmp"; fi
    mv "$tmp" "$OUTPUT"
    trap - EXIT
fi
