#!/bin/bash
# Static preflight for the canonical Stage-1 unchanged-hot-path A/B probe.
# No Julia, Pkg, SSH, or qsub is executed here.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
fail=0

check() {
  if ! "$@"; then
    echo "FAIL: $*"
    fail=1
  fi
}

echo "checking shell syntax"
for script in "$HERE"/*.sh "$HERE"/*.pbs; do
  check bash -n "$script"
done

echo "checking Python syntax"
check python3 -c \
  "import ast, pathlib; ast.parse(pathlib.Path('$HERE/analyze_ab.py').read_text())"

echo "checking required files"
for file in README.md stage1_ab.pbs analyze_ab.py submit_stage1_ab.sh static_check.sh; do
  check test -s "$HERE/$file"
done

echo "checking PBS resource contract"
check grep -qF '#PBS -q sugon' "$HERE/stage1_ab.pbs"
check grep -qF '#PBS -l nodes=1:ppn=5' "$HERE/stage1_ab.pbs"
check grep -qF -- '-t 4' "$HERE/stage1_ab.pbs"
check grep -qF -- '--resource-class=regular' "$HERE/stage1_ab.pbs"
check grep -qF 'OPENBLAS_NUM_THREADS=1' "$HERE/stage1_ab.pbs"
check grep -qF 'JULIA_PKG_OFFLINE=true' "$HERE/stage1_ab.pbs"
check grep -qF '/usr/bin/time -v' "$HERE/stage1_ab.pbs"
check grep -qF 'already exists; refusing' "$HERE/stage1_ab.pbs"
check grep -qF 'NODE_NAME:?set NODE_NAME' "$HERE/stage1_ab.pbs"
check grep -qF 'baseline_tree_sha256' "$HERE/stage1_ab.pbs"
check grep -qF 'candidate_tree_sha256' "$HERE/stage1_ab.pbs"
check grep -qF 'runner_tree_sha256' "$HERE/stage1_ab.pbs"
check grep -qF 'baseline_source_sha256_subset' "$HERE/stage1_ab.pbs"
check grep -qF 'candidate_source_sha256_subset' "$HERE/stage1_ab.pbs"

echo "checking A/B configuration"
check grep -qF 'run_generated_pathological.jl' "$HERE/stage1_ab.pbs"
check grep -qF -- '--arithmetic="$ARITHMETIC"' "$HERE/stage1_ab.pbs"
check grep -qF -- '--case-filter="$CASE_FILTER"' "$HERE/stage1_ab.pbs"
check grep -qF -- '--repetitions="$REPETITIONS"' "$HERE/stage1_ab.pbs"
check grep -qF -- '--warmup="$WARMUP"' "$HERE/stage1_ab.pbs"
check grep -qF 'lp_row_scaling,socp_many_tiny,sdp_hilbert' "$HERE/stage1_ab.pbs"
check grep -qF 'float64,float64x4' "$HERE/stage1_ab.pbs"
check grep -qF 'REPETITIONS:-4' "$HERE/stage1_ab.pbs"
check grep -qF 'run_arm baseline' "$HERE/stage1_ab.pbs"
check grep -qF 'run_arm candidate' "$HERE/stage1_ab.pbs"
check grep -qF 'analyze_ab.py' "$HERE/stage1_ab.pbs"
check grep -qF 'arms.conf' "$HERE/stage1_ab.pbs"

echo "checking analysis gates"
check grep -qF '1.10' "$HERE/analyze_ab.py"
check grep -qF '0.05' "$HERE/analyze_ab.py"
check grep -qF 'input_sha256' "$HERE/analyze_ab.py"
check grep -qF 'objective_primal' "$HERE/analyze_ab.py"
check grep -qF 'certificate_residual' "$HERE/analyze_ab.py"
check grep -qF 'planned_backend' "$HERE/analyze_ab.py"
check grep -qF 'executed_backend' "$HERE/analyze_ab.py"
check grep -qF 'hostname' "$HERE/analyze_ab.py"
check grep -qF 'workspace_bytes' "$HERE/analyze_ab.py"
check grep -qF 'baseline_source_sha256_subset' "$HERE/analyze_ab.py"
check grep -qF 'candidate_source_sha256_subset' "$HERE/analyze_ab.py"
check grep -qF 'ab_provenance.txt' "$HERE/analyze_ab.py"
check grep -qF 'key[4] == 1' "$HERE/analyze_ab.py"
check grep -qF 'range(2, repetitions + 1)' "$HERE/analyze_ab.py"
check grep -qF '1000.0' "$HERE/analyze_ab.py"
check grep -qF -- '--self-test' "$HERE/analyze_ab.py"
check grep -qF 'AB_GATE_PASS' "$HERE/analyze_ab.py"
check grep -qF 'ab_report.csv' "$HERE/analyze_ab.py"

echo "checking submit script"
check grep -qF -- '--submit' "$HERE/submit_stage1_ab.sh"
check grep -qF 'qsub -v' "$HERE/submit_stage1_ab.sh"
check grep -qF 'nodes=$NODE_NAME:ppn=5' "$HERE/submit_stage1_ab.sh"

echo "checking policy documentation"
check grep -qF 'Cluster-only' "$HERE/README.md"
check grep -qF 'Do not' "$HERE/README.md"
check grep -qF 'full-tree' "$HERE/README.md"
check grep -qF 'subset' "$HERE/README.md"
check grep -qF 'workspace_bytes' "$HERE/README.md"
check grep -qF 'warmup' "$HERE/README.md"
check grep -qF 'timed 2-4' "$HERE/README.md"

echo "running analyzer synthetic self-test"
check python3 "$HERE/analyze_ab.py" --self-test

if [ "$fail" -ne 0 ]; then
  echo "static check FAILED"
  exit 1
fi
echo "static check PASSED"
