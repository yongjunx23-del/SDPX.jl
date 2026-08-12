#!/bin/bash
# Static preflight for the v041-unified-la cluster probe.  No Julia, Pkg,
# SSH, or qsub is executed here.
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

echo "checking required files"
for file in README.md focused.pbs bootstrap_login_env.sh static_check.sh; do
  check test -s "$HERE/$file"
done

echo "checking PBS resource and identity contract"
check grep -qF '#PBS -q sugon' "$HERE/focused.pbs"
check grep -qF '#PBS -l nodes=1:ppn=5' "$HERE/focused.pbs"
check grep -qF 'OPENBLAS_NUM_THREADS=1' "$HERE/focused.pbs"
check grep -qF 'JULIA_PKG_OFFLINE=true' "$HERE/focused.pbs"
check grep -qF '/usr/bin/time -v' "$HERE/focused.pbs"
check grep -qF 'already exists; refusing' "$HERE/focused.pbs"
check grep -qF 'NODE_NAME:?set NODE_NAME' "$HERE/focused.pbs"
check grep -qF 'CAMPAIGN_ID:?set CAMPAIGN_ID' "$HERE/focused.pbs"
check grep -qF 'MFLA_COMMIT:?set MFLA_COMMIT' "$HERE/focused.pbs"
check grep -qF 'PBS_NP contract failed' "$HERE/focused.pbs"
check grep -qF 'JULIA_NUM_THREADS contract failed' "$HERE/focused.pbs"
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/focused.pbs"
check grep -qF 'runtime_contract=ok' "$HERE/focused.pbs"
check grep -qF 'SUCCESS' "$HERE/focused.pbs"
check grep -qF -- '-t 4' "$HERE/focused.pbs"
check grep -qF 'multifloat_linear_algebra_integration.jl' "$HERE/focused.pbs"
check grep -qF ':solve,' "$HERE/focused.pbs"
check grep -qF ':cholesky_factor!' "$HERE/focused.pbs"
check grep -qF '642d9d30-8e28-45ca-9d81-256429ea358f' "$HERE/focused.pbs"

echo "checking README"
check grep -qF 'e5eccd7a56482522acd5690800bf7438149997f5' "$HERE/README.md"
check grep -qF '642d9d30-8e28-45ca-9d81-256429ea358f' "$HERE/README.md"
check grep -qF 'bootstrap_login_env.sh' "$HERE/README.md"
check grep -qF 'focused.pbs' "$HERE/README.md"
check grep -qF 'static_check.sh' "$HERE/README.md"

echo "checking bootstrap guard"
check grep -qF 'bootstrap refuses to run inside a PBS job' "$HERE/bootstrap_login_env.sh"

if [ "$fail" -ne 0 ]; then
  echo "static check FAILED"
  exit 1
fi
echo "static check PASSED"
