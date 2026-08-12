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
for file in README.md focused.pbs kernel_ab.pbs solver_ab.pbs \
            kernel_ab.jl solver_ab.jl bootstrap_login_env.sh static_check.sh; do
  check test -s "$HERE/$file"
done

echo "checking PBS resource and identity contract"
for job in focused kernel_ab solver_ab; do
  check grep -qF '#PBS -q sugon' "$HERE/$job.pbs"
  check grep -qF '#PBS -l nodes=1:ppn=5' "$HERE/$job.pbs"
  check grep -qF 'OPENBLAS_NUM_THREADS=1' "$HERE/$job.pbs"
  check grep -qF 'JULIA_PKG_OFFLINE=true' "$HERE/$job.pbs"
  check grep -qF '/usr/bin/time -v' "$HERE/$job.pbs"
  check grep -qF 'already exists; refusing' "$HERE/$job.pbs"
  check grep -qF 'NODE_NAME:?set NODE_NAME' "$HERE/$job.pbs"
  check grep -qF 'CAMPAIGN_ID:?set CAMPAIGN_ID' "$HERE/$job.pbs"
  check grep -qF 'MFLA_COMMIT:?set MFLA_COMMIT' "$HERE/$job.pbs"
  check grep -qF 'PBS_NP contract failed' "$HERE/$job.pbs"
  check grep -qF 'JULIA_NUM_THREADS contract failed' "$HERE/$job.pbs"
  check grep -qF 'runtime_contract=ok' "$HERE/$job.pbs"
  check grep -qF 'SUCCESS' "$HERE/$job.pbs"
done
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/focused.pbs"
check grep -qF 'KERNEL_AB ' "$HERE/kernel_ab.pbs"
check grep -qF 'SOLVER_AB ' "$HERE/solver_ab.pbs"
check grep -qF 'requested=requested,' "$HERE/kernel_ab.jl"
check grep -qF ':multifloat' "$HERE/kernel_ab.jl"
check grep -qF ':multifloat' "$HERE/solver_ab.jl"
check grep -qF 'runtime_contract=ok' "$HERE/focused.pbs"
check grep -qF 'SUCCESS' "$HERE/focused.pbs"
check grep -qF -- '-t 4' "$HERE/focused.pbs"
check grep -qF 'multifloat_linear_algebra_integration.jl' "$HERE/focused.pbs"
check grep -qF ':solve,' "$HERE/focused.pbs"
check grep -qF ':cholesky_factor!' "$HERE/focused.pbs"
check grep -qF 'auto selection ignores provider presence' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF '642d9d30-8e28-45ca-9d81-256429ea358f' "$HERE/focused.pbs"

echo "checking README"
check grep -qF 'e5eccd7a56482522acd5690800bf7438149997f5' "$HERE/README.md"
check grep -qF '642d9d30-8e28-45ca-9d81-256429ea358f' "$HERE/README.md"
check grep -qF 'bootstrap_login_env.sh' "$HERE/README.md"
check grep -qF 'focused.pbs' "$HERE/README.md"
check grep -qF 'kernel_ab.pbs' "$HERE/README.md"
check grep -qF 'solver_ab.pbs' "$HERE/README.md"
check grep -qF 'static_check.sh' "$HERE/README.md"

echo "checking bootstrap guard"
check grep -qF 'bootstrap refuses to run inside a PBS job' "$HERE/bootstrap_login_env.sh"

if [ "$fail" -ne 0 ]; then
  echo "static check FAILED"
  exit 1
fi
echo "static check PASSED"
