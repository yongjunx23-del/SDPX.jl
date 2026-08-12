#!/bin/bash
# Static preflight for the parallel focused/full Stage-1 submission layer.
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
  "import ast, pathlib; ast.parse(pathlib.Path('$HERE/aggregate_parallel.py').read_text())"

echo "checking required files"
for file in README.md focused.pbs full.pbs bootstrap_login_env.sh \
            submit_parallel.sh aggregate_parallel.py static_check.sh; do
  check test -s "$HERE/$file"
done

echo "checking PBS resource and identity contract"
for job in focused full; do
  check grep -qF '#PBS -q sugon' "$HERE/$job.pbs"
  check grep -qF '#PBS -l nodes=1:ppn=5' "$HERE/$job.pbs"
  check grep -qF 'OPENBLAS_NUM_THREADS=1' "$HERE/$job.pbs"
  check grep -qF 'JULIA_PKG_OFFLINE=true' "$HERE/$job.pbs"
  check grep -qF '/usr/bin/time -v' "$HERE/$job.pbs"
  check grep -qF 'already exists; refusing' "$HERE/$job.pbs"
  check grep -qF 'NODE_NAME:?set NODE_NAME' "$HERE/$job.pbs"
  check grep -qF 'CAMPAIGN_ID:?set CAMPAIGN_ID' "$HERE/$job.pbs"
  check grep -qF 'echo "campaign_id=$CAMPAIGN_ID"' "$HERE/$job.pbs"
done
check grep -qF -- '-t 4' "$HERE/focused.pbs"
check grep -qF -- '-t 4' "$HERE/full.pbs"
check grep -qF 'canonical_conic_problem.jl' "$HERE/focused.pbs"
check grep -qF 'problem_features.jl' "$HERE/focused.pbs"
check grep -qF 'auto_planner.jl' "$HERE/focused.pbs"
check grep -qF 'pipeline.jl' "$HERE/focused.pbs"
check grep -qF 'v041_architecture_regressions.jl' "$HERE/focused.pbs"
check grep -qF 'executed_diagnostics.jl' "$HERE/focused.pbs"
check grep -qF 'public_api.jl' "$HERE/focused.pbs"
check grep -qF 'Pkg.test("SDPX"; coverage=false)' "$HERE/full.pbs"

echo "checking parallel submit helper"
check grep -qF -- '--submit' "$HERE/submit_parallel.sh"
check grep -qF 'must be two distinct, already-verified healthy idle nodes' "$HERE/submit_parallel.sh"
check grep -qF 'qsub -N "$job_name" -v "$vars"' "$HERE/submit_parallel.sh"
check grep -qF 'nodes=$node:ppn=5' "$HERE/submit_parallel.sh"
check grep -qF 'vars="$vars,OUTPUT_ROOT=$result_root"' "$HERE/submit_parallel.sh"
check grep -qF 'vars="$vars,FULL_OUTPUT_ROOT=$result_root"' "$HERE/submit_parallel.sh"
check grep -qF 'CAMPAIGN_ROOT' "$HERE/submit_parallel.sh"
check grep -qF 'CAMPAIGN_MANIFEST' "$HERE/submit_parallel.sh"
check grep -qF 'must not live inside the candidate source, candidate env, or depot' "$HERE/submit_parallel.sh"

echo "checking aggregation helper"
check python3 -c \
  "import ast, pathlib; ast.parse(pathlib.Path('$HERE/aggregate_parallel.py').read_text())"
check grep -qF 'AGGREGATE_GATE_PASS' "$HERE/aggregate_parallel.py"
check grep -qF 'AGGREGATE_GATE_FAILED' "$HERE/aggregate_parallel.py"
check grep -qF 'summary.json' "$HERE/aggregate_parallel.py"
check grep -qF 'summary.md' "$HERE/aggregate_parallel.py"
check grep -qF 'MISSING_REPORT' "$HERE/aggregate_parallel.py"
check grep -qF 'TIMING_NOISE' "$HERE/aggregate_parallel.py"
check grep -qF 'TEST_FAILURE' "$HERE/aggregate_parallel.py"
check grep -qF 'campaign_id' "$HERE/aggregate_parallel.py"
check grep -qF 'SELF_TEST_PASS' "$HERE/aggregate_parallel.py"
check grep -qF -- '--self-test' "$HERE/aggregate_parallel.py"
check grep -qF 'json.dumps' "$HERE/aggregate_parallel.py"

echo "checking README"
check grep -qF 'submit_parallel.sh' "$HERE/README.md"
check grep -qF 'aggregate_parallel.py' "$HERE/README.md"
check grep -qF 'static_check.sh' "$HERE/README.md"
check grep -qF 'CAMPAIGN_ID' "$HERE/README.md"
check grep -qF 'CAMPAIGN_ROOT' "$HERE/README.md"
check grep -qF 'no ratios, CVs, medians, or timing comparisons' "$HERE/README.md"

echo "running aggregation synthetic self-test"
check python3 "$HERE/aggregate_parallel.py" --self-test

if [ "$fail" -ne 0 ]; then
  echo "static check FAILED"
  exit 1
fi
echo "static check PASSED"
