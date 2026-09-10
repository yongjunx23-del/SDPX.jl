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

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SDPX="$ROOT/SDPX.jl"
MFLA="$ROOT/MultiFloatLinearAlgebra.jl"
BFLA="$ROOT/BigFloatLinearAlgebra.jl"
ENV="$ROOT/rebuild-env"
OUT="${1:-$ROOT/rebuild-reports/I01_prework/driver_matrix}"
export JULIA_DEPOT_PATH="$ROOT/rebuild-env-depot:$HOME/.julia"
export JULIA_NUM_THREADS=1

mkdir -p "$OUT"
echo "workspace   $ROOT"
echo "env         $ENV"
echo "outdir      $OUT"
for r in SDPX.jl MultiFloatLinearAlgebra.jl BigFloatLinearAlgebra.jl; do
    printf '%-28s %s\n' "$r" "$(git -C "$ROOT/$r" rev-parse --short HEAD)"
done
echo "julia       $(julia --version)"
echo

run() {  # run <name> <workdir> <project> <script> [ENVS="K=V ..."] [ARGS="--flag ..."]
    local name="$1" wd="$2" proj="$3" script="$4" envs="${5:-}" args="${6:-}"
    local log="$OUT/$name.log"
    printf '%-26s ' "$name"
    ( cd "$wd" && env $envs julia --project="$proj" -t1 "$script" $args ) > "$log" 2>&1
    local rc=$?
    # A driver that exits 0 while its summary shows failures is the failure mode
    # this whole rebuild keeps hitting, so report both independently.
    local fails fail_lines
    fails=$(grep -cE '\|\s+[0-9]+\s+(Fail|Error)' "$log" 2>/dev/null || true)
    fail_lines=$(grep -coE '(Test Failed|^ERROR:)' "$log" 2>/dev/null || true)
    echo "exit=$rc  failcols=${fails:-0}  fail_lines=${fail_lines:-0}  log=$log"
}

# A01 is provider-GATED: its default leg asserts the providers are ABSENT, which
# is only true under SDPX's own project. Running that leg under $ENV makes three
# assertions fail by design, and A01b recorded exactly that ("NO selector" -> 1
# fail). So the default leg uses --project=$SDPX and the provider legs use $ENV.
# Getting this wrong looks like a regression and is not one.
run S01            "$SDPX" "$ENV"  test/rebuild/S01.jl
run S02            "$SDPX" "$ENV"  test/rebuild/S02.jl
run S03            "$SDPX" "$ENV"  test/rebuild/S03.jl
run S04            "$SDPX" "$ENV"  test/rebuild/S04.jl
run S05_none       "$SDPX" "$ENV"  test/rebuild/S05.jl "S05_LIVE_PROVIDER=none"
run S05_mfla       "$SDPX" "$ENV"  test/rebuild/S05.jl "S05_LIVE_PROVIDER=mfla"
run S05_bfla       "$SDPX" "$ENV"  test/rebuild/S05.jl "S05_LIVE_PROVIDER=bfla"
run S06            "$SDPX" "$ENV"  test/rebuild/S06.jl
run A01_default    "$SDPX" "$SDPX" test/rebuild/A01.jl
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

echo
echo "done. logs in $OUT"
