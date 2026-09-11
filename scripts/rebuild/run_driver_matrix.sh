#!/usr/bin/env bash
# Run every task driver on the CURRENT revisions, one process each.
#
#   bash SDPX.jl/scripts/rebuild/run_driver_matrix.sh [OUTDIR]
#
# WHY RE-RUN INSTEAD OF QUOTING THE OLD NUMBERS. The batch reports under
# `rebuild-reports/<TASK>/report.json` were written at the time each task finished,
# and several of them assert a state that I01 has since changed: S02's acceptance
# still says "src/SDPX.jl does not include solver/loop.jl", S03's says "the five new
# files are not included", S04's says "Pkg.test() not_run". Those statements were
# true when written. A handover that quotes them would be describing three
# revisions ago. This script produces one coherent set of numbers at one revision,
# and the handover cites this rather than the reports.
#
# ONE PROCESS PER LEG, ALWAYS -t1. Q01 measured that the MultiFloat and BigFloat
# legs exhaust the Julia 1.12 inference compiler when co-resident, and MFLA
# threading interacts with `-t` in a way that changes results. Do not parallelise
# this script by removing `-t1`; run the legs sequentially as written.

set -u

# Aggregated verdict across legs. Initialised here because `set -u` is on and these
# are incremented inside run(); see run() for why they exist at all.
MATRIX_LEGS_RUN=0
MATRIX_LEGS_FAILED=0
MATRIX_FAILED_LEGS=""

# Overridable so the same matrix can be run against a PINNED revision set built by
# scripts/rebuild/pin_revisions_env.sh. That matters because `rebuild-env`
# resolves the three packages by dev path with no `git-tree-sha1`, so a run
# against the live trees measures whatever a worker last saved:
#
#   SDPX_REBUILD_ROOT=/tmp/sdpxpin SDPX_REBUILD_ENV=/tmp/sdpxpin-env \
#   SDPX_REPO=/tmp/sdpxpin/SDPX.jl MFLA_REPO=/tmp/sdpxpin/MultiFloatLinearAlgebra.jl \
#   BFLA_REPO=/tmp/sdpxpin/BigFloatLinearAlgebra.jl \
#   scripts/rebuild/run_driver_matrix.sh /tmp/mx-pinned
#
# (pin_revisions_env.sh names the pinned directories with the canonical
# repository names, so a pin is a drop-in workspace root.)
#
ROOT="${SDPX_REBUILD_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SDPX="${SDPX_REPO:-$ROOT/SDPX.jl}"
MFLA="${MFLA_REPO:-$ROOT/MultiFloatLinearAlgebra.jl}"
BFLA="${BFLA_REPO:-$ROOT/BigFloatLinearAlgebra.jl}"
ENV="${SDPX_REBUILD_ENV:-$ROOT/rebuild-env}"
OUT="${1:-$ROOT/rebuild-reports/I01_prework/driver_matrix}"
# Isolated SDPX + MultiFloats environment for A01's no-provider leg.
# Override when the host default environment contains optional providers.
DEFAULT_ENV="${SDPX_REBUILD_DEFAULT_ENV:-$SDPX}"
export SDPX_S06_REBUILD_ENV="${SDPX_S06_REBUILD_ENV:-$ENV}"
export JULIA_DEPOT_PATH="${SDPX_REBUILD_DEPOT:-$ROOT/rebuild-env-depot}:$HOME/.julia"
export JULIA_NUM_THREADS=1

mkdir -p "$OUT"
echo "workspace   $ROOT"
echo "env         $ENV"
echo "outdir      $OUT"
# Print the revisions AND the working-tree state actually used. A revision alone
# is not enough: a dirty tree means the run is not attributable to that commit,
# and the reader of this log has no other way to know.
for r in "$SDPX" "$MFLA" "$BFLA"; do
    printf '%-28s %s  dirty_paths=%s\n' "$(basename "$r")" \
        "$(git -C "$r" rev-parse --short HEAD 2>/dev/null || echo '??')" \
        "$(git -C "$r" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
done
echo "julia       $(julia --version)"
echo

run() {  # run <name> <workdir> <project> <script> [ENVS="K=V ..."] [ARGS="--flag ..."]
    local name="$1" wd="$2" proj="$3" script="$4" envs="${5:-}" args="${6:-}"
    local log="$OUT/$name.log"
    printf '%-26s ' "$name"
    # Batch-5 drivers are added here as their tasks are dispatched. A driver that
    # does not exist yet is SKIPPED, not failed: this script is meant to be runnable
    # throughout a wave so the parent can verify each report the moment it lands,
    # and "not written yet" is not a red result.
    if [ ! -f "$wd/$script" ]; then
        echo "SKIP  driver not written yet: $wd/$script"
        return 0
    fi
    ( cd "$wd" && env $envs julia --project="$proj" -t1 "$script" $args ) > "$log" 2>&1
    local rc=$?
    # A driver that exits 0 while its summary shows failures is the failure mode
    # this whole rebuild keeps hitting, so report both independently.
    local fails fail_lines
    fails=$(grep -cE '\|\s+[0-9]+\s+(Fail|Error)' "$log" 2>/dev/null || true)
    fail_lines=$(grep -coE '(Test Failed|^ERROR:)' "$log" 2>/dev/null || true)
    echo "exit=$rc  failcols=${fails:-0}  fail_lines=${fail_lines:-0}  log=$log"
    # ...and then AGGREGATE them. Until this existed the script ended in `echo`, so
    # its exit status was echo's and was 0 no matter what happened: `MATRIX_EXIT=0`
    # was printed for a run in which B03 exited 1. A summary whose verdict cannot
    # be anything but "ok" is not a verdict, and a caller writing
    # `... ; echo MATRIX_EXIT=$?` reads exactly that vacuous value.
    MATRIX_LEGS_RUN=$((MATRIX_LEGS_RUN + 1))
    if [ "$rc" -ne 0 ] || [ "${fails:-0}" -ne 0 ] || [ "${fail_lines:-0}" -ne 0 ]; then
        MATRIX_LEGS_FAILED=$((MATRIX_LEGS_FAILED + 1))
        MATRIX_FAILED_LEGS="$MATRIX_FAILED_LEGS $name"
    fi
}

# A01's default leg requires SDPX + MultiFloats but no optional provider.
# Use SDPX_REBUILD_DEFAULT_ENV when that is not the SDPX project environment.
# Its LOAD_PATH is isolated so global @v#.# providers cannot contaminate it.
# The explicitly selected provider legs continue to use the full rebuild env.
run S01            "$SDPX" "$ENV"  test/rebuild/S01.jl
run S02            "$SDPX" "$ENV"  test/rebuild/S02.jl
run S03            "$SDPX" "$ENV"  test/rebuild/S03.jl
run S04            "$SDPX" "$ENV"  test/rebuild/S04.jl
run S05_none       "$SDPX" "$ENV"  test/rebuild/S05.jl "S05_LIVE_PROVIDER=none"
run S05_mfla       "$SDPX" "$ENV"  test/rebuild/S05.jl "S05_LIVE_PROVIDER=mfla"
run S05_bfla       "$SDPX" "$ENV"  test/rebuild/S05.jl "S05_LIVE_PROVIDER=bfla"
run S06            "$SDPX" "$ENV"  test/rebuild/S06.jl
run A01_default    "$SDPX" "$DEFAULT_ENV" test/rebuild/A01.jl "JULIA_LOAD_PATH=@:@stdlib"
run A01_all        "$SDPX" "$ENV"  test/rebuild/A01.jl "" "--provider=all"
run A01_mfla       "$SDPX" "$ENV"  test/rebuild/A01.jl "" "--provider=mfla"
run A01_bfla       "$SDPX" "$ENV"  test/rebuild/A01.jl "" "--provider=bfla"
run A01_qdldl      "$SDPX" "$ENV"  test/rebuild/A01.jl "" "--provider=qdldl"
run P01_none       "$SDPX" "$ENV"  test/provider_contracts/sparse_contract.jl "P01_PROVIDER_LEG=none"
run P01_mfla       "$SDPX" "$ENV"  test/provider_contracts/sparse_contract.jl "P01_PROVIDER_LEG=mfla"
run P01_bfla       "$SDPX" "$ENV"  test/provider_contracts/sparse_contract.jl "P01_PROVIDER_LEG=bfla"
run Q01_rules      "$SDPX" "$ENV"  test/rebuild/dependency_rules.jl
run M01            "$MFLA" "$ENV"  test/rebuild/M01.jl
run B01_sandbox    "$BFLA" "$ENV"  test/rebuild/B01.jl
run B01_wired      "$BFLA" "$ENV"  test/rebuild/B01_wired.jl

# --- batch 5 ------------------------------------------------------------------
# Dispatched in waves, one task per repository per wave. Skipped until written.
# The MFLA/BFLA legs must stay separate PROCESSES with -t1 (the two-process rule).
# B03 is the threading task and is run alone on purpose: a contended measurement
# window on this 4-core host is a different measurement.
run M02            "$MFLA" "$ENV"  test/rebuild/M02.jl
run M03            "$MFLA" "$ENV"  test/rebuild/M03.jl
run P02            "$MFLA" "$ENV"  test/rebuild/P02.jl
run B02            "$BFLA" "$ENV"  test/rebuild/B02.jl
# B03's mode is DETECTED, not flagged: `detect_mode()` asks whether
# `plan_gemm!`/`plan_cholesky_trail!` are bound in `BigFloatLinearAlgebra`. Since
# I02 wired B03's four kernel files into BFLA's entry point (they are loaded at
# `src/BigFloatLinearAlgebra.jl:89-92` at revision f087a72; `plan_gemm!` is defined
# at `src/kernels/native_level3.jl:157`), the detected mode is now **WIRED**, and
# WIRED mode refuses to run without `B03_WIRED_INJECT_TEST=1`.
#
# Without that variable this leg exited 1 with a LoadError and was the ONLY red leg
# in the 28-leg run — recorded as `B03 exit=1`, and simultaneously reported as
# `MATRIX_EXIT=0` because nothing aggregated the result. Both defects are fixed
# here and above. The env var is not papering over a failure: the parent re-ran
# this driver at the frozen revisions with the variable set and got exit=0,
# fail_lines=0 in all four non-perf phases plus the default invocation.
#
# There is deliberately no UNWIRED leg. UNWIRED is not selectable by a flag — it is
# what the driver infers when the names are ABSENT — so it is unreachable in a tree
# where the wiring is real, and adding a leg that cannot run would be a leg that
# always "passes" by skipping.
run B03            "$BFLA" "$ENV"  test/rebuild/B03.jl "B03_WIRED_INJECT_TEST=1"
run B04            "$BFLA" "$ENV"  test/rebuild/B04.jl
run P03            "$BFLA" "$ENV"  test/rebuild/P03.jl
run S07            "$SDPX" "$ENV"  test/rebuild/S07.jl

echo
echo "legs_run=$MATRIX_LEGS_RUN  legs_failed=$MATRIX_LEGS_FAILED  failed_legs=${MATRIX_FAILED_LEGS:- none}"
if [ "$MATRIX_LEGS_FAILED" -ne 0 ]; then
    echo "MATRIX_EXIT=1"
    echo "done WITH FAILURES. logs in $OUT"
    exit 1
fi
echo "MATRIX_EXIT=0"
echo "done. logs in $OUT"
