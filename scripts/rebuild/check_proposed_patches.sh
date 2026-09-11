#!/usr/bin/env bash
# Check every proposed integration patch, in the repository it actually targets.
#
# WHY THIS EXISTS. The patches in `docs/evidence/proposed/` are spread across
# three repositories. Running `git apply --check` for all of them from `SDPX.jl`
# reports five bogus failures ("src/factor_caches.jl: No such file or
# directory") for patches that are perfectly fine in MFLA or BFLA -- and it is
# easy to read those as "the patch is corrupt", which is a different and much
# more alarming conclusion. This script keeps the mapping in one place so the
# check is run in the right tree, and runs the structural validator too, because
# `git apply` is not the only way a recorded patch goes wrong (see
# validate_patches.py for the two defect shapes that motivated it).
#
# Usage:
#   scripts/rebuild/check_proposed_patches.sh            # check
#   scripts/rebuild/check_proposed_patches.sh --apply    # apply for real
#   scripts/rebuild/check_proposed_patches.sh --list     # print the mapping
#
# Exit status 0 only if every patch is structurally valid AND applies cleanly.

set -uo pipefail

WS="/Users/xuyongjun/Desktop/project/SDPX"
SDPX="$WS/SDPX.jl"
MFLA="$WS/MultiFloatLinearAlgebra.jl"
BFLA="$WS/BigFloatLinearAlgebra.jl"
PROPOSED="$SDPX/docs/evidence/proposed"

MODE="check"
case "${1:-}" in
    --apply) MODE="apply" ;;
    --list)  MODE="list" ;;
    "")      MODE="check" ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
esac

# patch<TAB>repo-key. The repo key selects the working tree below.
MAPPING="
B02_wire_rectangular_rrqr.patch|BFLA
I02_adapt_B02_driver.patch|BFLA
I02_M01_IP2_record_at_commit.patch|MFLA
I02_M01_IP4_sparse_solve_dense_order.patch|MFLA
M02_wire_gemm_candidate.patch|MFLA
I02_adapt_S07_driver.patch|SDPX
I02_fix_refactor_numeric_lease.patch|SDPX
S07_wire_session_layer.patch|SDPX
product_cone_solve_nzrange.patch|SDPX
"

repo_dir() {
    case "$1" in
        SDPX) echo "$SDPX" ;;
        MFLA) echo "$MFLA" ;;
        BFLA) echo "$BFLA" ;;
        *)    echo "" ;;
    esac
}

if [ "$MODE" = "list" ]; then
    printf '%-48s %s\n' "PATCH" "REPO"
    echo "$MAPPING" | while IFS='|' read -r p r; do
        [ -z "${p:-}" ] && continue
        printf '%-48s %s\n' "$p" "$r"
    done
    exit 0
fi

echo "=== 1. structural validation (scripts/rebuild/validate_patches.py) ==="
python3 "$SDPX/scripts/rebuild/validate_patches.py" "$PROPOSED"
struct_rc=$?
echo

echo "=== 2. git apply --check, in each patch's own repository ==="
fail=0
total=0
echo "$MAPPING" | while IFS='|' read -r p r; do
    [ -z "${p:-}" ] && continue
    total=$((total + 1))
    path="$PROPOSED/$p"
    dir="$(repo_dir "$r")"
    if [ ! -f "$path" ]; then
        echo "  MISSING  $p  (expected in $PROPOSED)"
        fail=$((fail + 1))
        continue
    fi
    if git -C "$dir" apply --check "$path" 2>/tmp/cpp_err; then
        if [ "$MODE" = "apply" ]; then
            if git -C "$dir" apply "$path" 2>/tmp/cpp_err; then
                echo "  APPLIED  [$r] $p"
            else
                echo "  APPLY-FAILED [$r] $p: $(head -1 /tmp/cpp_err)"
                fail=$((fail + 1))
            fi
        else
            echo "  applies       [$r] $p"
        fi
    elif git -C "$dir" apply --check --reverse "$path" >/dev/null 2>&1; then
        # THE PATCH IS ALREADY IN THE TREE. This is the benign refusal, and
        # conflating it with a corrupt or stale patch was a real defect in the
        # first version of this script: once I02 applied the SDPX and BFLA
        # patches, the check reported six REFUSED and "RESULT: FAIL", which reads
        # as "six patches are broken" when in fact six patches had landed. The
        # reverse check is the standard way to tell the two apart.
        echo "  already-in    [$r] $p"
    else
        echo "  REFUSED       [$r] $p: $(head -1 /tmp/cpp_err)"
        fail=$((fail + 1))
    fi
done

# The loop above runs in a subshell, so recompute the counts here rather than
# trusting variables that were never updated in this shell. Three states, not two:
# applies cleanly / already applied (benign) / genuinely refused (a failure).
refused=0; already=0; clean=0
while IFS='|' read -r p r; do
    [ -z "${p:-}" ] && continue
    dir="$(repo_dir "$r")"
    if git -C "$dir" apply --check "$PROPOSED/$p" >/dev/null 2>&1; then
        clean=$((clean + 1))
    elif git -C "$dir" apply --check --reverse "$PROPOSED/$p" >/dev/null 2>&1; then
        already=$((already + 1))
    else
        refused=$((refused + 1))
    fi
done <<< "$MAPPING"

echo
echo "clean=$clean  already-in=$already  refused=$refused"
if [ "$struct_rc" -ne 0 ] || [ "$refused" -ne 0 ]; then
    echo "RESULT: FAIL ($refused patch(es) genuinely do not apply; structural rc=$struct_rc)"
    exit 1
fi
if [ "$already" -ne 0 ]; then
    echo "RESULT: OK ($clean apply cleanly, $already already in the tree, 0 refused)"
    exit 0
fi
echo "RESULT: OK (all mapped patches structurally valid and apply cleanly)"
