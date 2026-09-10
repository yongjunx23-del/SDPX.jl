#!/usr/bin/env bash
# Verify one task report against the workspace, in the order that catches things.
#
#   bash SDPX.jl/scripts/rebuild/verify_task.sh <ID>
#
# WHY A SCRIPT. The goal requires, for each of twelve remaining reports: validate the
# report, independently re-run the task's driver, then commit explicit paths and run
# the full suite on the now-clean tree. Done by hand twelve times, that becomes
# "validate the ones that look suspicious and re-run the ones that look quick" —
# which is exactly the sampling that let M01 ship a report claiming 6/6 against its
# own red 195-line file.
#
# WHAT IT DELIBERATELY DOES NOT DO: it never runs `git add`. It prints the exact
# staging command, with paths taken from the task's write allowlist, after checking
# that no OTHER worker has dirty files in the same repository. The parent's `git add
# -A` has twice swept a live worker's files into a commit; a script that stages for
# you would reintroduce that with better manners.

set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SDPX="$ROOT/SDPX.jl"; MFLA="$ROOT/MultiFloatLinearAlgebra.jl"; BFLA="$ROOT/BigFloatLinearAlgebra.jl"
ENV="$ROOT/rebuild-env"
PACKET="$HOME/Downloads/sdpx_infrastructure_review"
export JULIA_DEPOT_PATH="$ROOT/rebuild-env-depot:$HOME/.julia"
export JULIA_NUM_THREADS=1

ID="${1:-}"
[ -n "$ID" ] || { sed -n '2,12p' "$0"; exit 2; }

# Task -> (repository, driver, write allowlist). Kept here rather than parsed out of
# tasks.json so that a mismatch between this table and the packet is visible.
case "$ID" in
  M02) REPO="$MFLA"; DRIVER="test/rebuild/M02.jl"
       ALLOW=("src/planning/gemm_plan.jl" "src/kernels/packing.jl" "src/kernels/gemm_microkernels.jl" "src/kernels/gemm_schedule.jl" "test/rebuild/M02.jl") ;;
  M03) REPO="$MFLA"; DRIVER="test/rebuild/M03.jl"
       ALLOW=("src/factorizations/pivot_policy.jl" "src/factorizations/panel_updates.jl" "src/factorizations/qr_workspace.jl" "test/rebuild/M03.jl") ;;
  P02) REPO="$MFLA"; DRIVER="test/rebuild/P02.jl"
       ALLOW=("ext/rebuild/mfla_sparse_adapter.jl" "test/rebuild/P02.jl") ;;
  B02) REPO="$BFLA"; DRIVER="test/rebuild/B02.jl"
       ALLOW=("src/caches/rectangular_qr.jl" "src/kernels/qr_panel.jl" "src/solves/least_squares.jl" "test/rebuild/B02.jl") ;;
  B03) REPO="$BFLA"; DRIVER="test/rebuild/B03.jl"
       ALLOW=("src/planning/kernel_plan.jl" "src/kernels/native_level3.jl" "src/kernels/native_triangular.jl" "src/kernels/worker_scratch.jl" "test/rebuild/B03.jl") ;;
  B04) REPO="$BFLA"; DRIVER="test/rebuild/B04.jl"
       ALLOW=("src/caches/common.jl" "src/caches/cholesky.jl" "src/caches/lu.jl" "src/caches/ldlt.jl" "test/rebuild/B04.jl") ;;
  P03) REPO="$BFLA"; DRIVER="test/rebuild/P03.jl"
       ALLOW=("ext/rebuild/bfla_sparse_adapter.jl" "test/rebuild/P03.jl" "docs/rebuild/sparse_backend_contract.md") ;;
  S07) REPO="$SDPX"; DRIVER="test/rebuild/S07.jl"
       ALLOW=("src/session/update.jl" "src/session/replay.jl" "src/session/cancellation.jl" "test/rebuild/S07.jl") ;;
  *)   echo "unknown task id: $ID (expected one of M02 M03 P02 B02 B03 B04 P03 S07)"; exit 2 ;;
esac

echo "=== $ID  repo=$(basename "$REPO")  head=$(git -C "$REPO" rev-parse --short HEAD)"
echo

echo "--- 1. report schema ------------------------------------------------------"
if [ -f "$ROOT/rebuild-reports/$ID/report.json" ]; then
    python3 "$SDPX/scripts/rebuild/validate_reports.py" "$ROOT" --only "$ID"
    SCHEMA_RC=$?
else
    echo "NO REPORT YET: rebuild-reports/$ID/report.json does not exist"
    SCHEMA_RC=9
fi
echo

echo "--- 2. independent driver re-run -----------------------------------------"
mkdir -p "$ROOT/rebuild-reports/$ID"
if [ -f "$REPO/$DRIVER" ]; then
    LOG="$ROOT/rebuild-reports/$ID/verify_driver.log"
    ( cd "$REPO" && julia --project="$ENV" -t1 "$DRIVER" ) > "$LOG" 2>&1
    RC=$?
    FAILCOLS=$(grep -cE '\|\s+[0-9]+\s+(Fail|Error)' "$LOG" || true)
    echo "exit=$RC  failcols=$FAILCOLS  log=$LOG"
    echo "  testsets=$(grep -c '^Test Summary' "$LOG" || true)"
else
    echo "NO DRIVER YET: $REPO/$DRIVER does not exist"
    RC=9
fi
echo

echo "--- 3. concurrent-worker guard -------------------------------------------"
# `git status --porcelain` reports a brand-new directory as ONE entry with a
# trailing slash (e.g. `?? src/planning/`), not as the files inside it. Comparing
# that entry against an allowlist of file paths made the guard fire on every task
# that creates a directory — which is most of them, so it would have been ignored
# exactly when it mattered. Expand directories to their files first.
DIRTY=()
while IFS= read -r line; do
    [ -n "$line" ] || continue
    path="${line:3}"
    if [ "${path: -1}" = "/" ]; then
        while IFS= read -r inner; do
            [ -n "$inner" ] && DIRTY+=("$inner")
        done < <(git -C "$REPO" ls-files --others --exclude-standard -- "$path")
    else
        DIRTY+=("$path")
    fi
done < <(git -C "$REPO" status --porcelain)
STRAY=()
for path in "${DIRTY[@]:-}"; do
    [ -n "$path" ] || continue
    keep=0
    for a in "${ALLOW[@]}"; do [ "$path" = "$a" ] && keep=1; done
    [ "$keep" = 0 ] && STRAY+=("$path")
done
if [ "${#STRAY[@]}" -eq 0 ]; then
    echo "clean: every dirty path in $(basename "$REPO") belongs to $ID"
else
    echo "WARNING: $(basename "$REPO") also has dirty paths NOT in $ID's allowlist:"
    for s in "${STRAY[@]}"; do echo "    $s"; done
    echo "  -> another worker is probably live in this repo. Do NOT commit yet:"
    echo "     staging is safe (explicit paths), but a commit now would freeze"
    echo "     another task's half-written files into your history."
fi
echo

echo "--- 4. staging command (NOT executed) ------------------------------------"
echo "    git -C $(basename "$REPO") add \\"
for a in "${ALLOW[@]}"; do
    if [ -e "$REPO/$a" ]; then echo "        $a \\"; else echo "        # MISSING: $a"; fi
done
echo "        # (add rebuild-reports/$ID/ by hand if you keep reports in-repo)"
echo
echo "--- 5. after committing, the gate this task cannot self-certify ----------"
if [ "$REPO" = "$SDPX" ]; then
    echo "    SDPX's Pkg.test() ABORTS on a dirty tree, so it could not run before the"
    echo "    commit. Run it now, on the clean tree:"
    echo "      julia --project=$SDPX -t1 -e 'using Pkg; Pkg.test()'"
    echo "    Expect 170 testsets / Pass=9392 / Broken=7 / Total=9399 and 0 Fail columns."
else
    echo "    julia --project=$REPO -t1 -e 'using Pkg; Pkg.test()'"
    case "$REPO" in
      "$MFLA") echo "    Expect 23 testsets / 4164 assertions, 'tests passed'." ;;
      "$BFLA") echo "    Expect 10864/10864, 'tests passed'." ;;
    esac
fi
echo
[ "$SCHEMA_RC" = 0 ] && [ "$RC" = 0 ] && echo "RESULT: schema ok, driver exit 0." || echo "RESULT: NOT clean (schema_rc=$SCHEMA_RC driver_rc=$RC) — read the logs above."
