#!/bin/bash
# Static preflight for the v041-unified-la cluster probes.  Only shell
# syntax, Julia parse-only syntax (`Meta.parseall`), and git diff checks are
# executed here.  No probe script, Julia solve, SSH, or qsub is executed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PROBE_REL="cluster-probes/v041-unified-la"
fail=0

check() {
  if ! "$@"; then
    echo "FAIL: $*"
    fail=1
  fi
}

check_not() {
  if "$@"; then
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
check grep -qF 'CANDIDATE_PATHOF ' "$HERE/kernel_ab.pbs"
check grep -qF 'CANDIDATE_PATHOF ' "$HERE/solver_ab.pbs"
check grep -qF 'MFLA_ROOT ' "$HERE/kernel_ab.pbs"
check grep -qF 'MFLA_ROOT ' "$HERE/solver_ab.pbs"
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/kernel_ab.pbs"
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/solver_ab.pbs"
check grep -qF 'Maximum resident set size' "$HERE/kernel_ab.pbs"
check grep -qF 'Maximum resident set size' "$HERE/solver_ab.pbs"
check grep -qF 'rss_kib=' "$HERE/kernel_ab.pbs"
check grep -qF 'rss_kib=' "$HERE/solver_ab.pbs"
check grep -qF 'cpu_utilization=' "$HERE/kernel_ab.pbs"
check grep -qF 'cpu_utilization=' "$HERE/solver_ab.pbs"
check grep -qF 'mfla_commit_expected' "$HERE/kernel_ab.pbs"
check grep -qF 'mfla_commit_expected' "$HERE/solver_ab.pbs"
check grep -qF 'e5eccd7a56482522acd5690800bf7438149997f5' "$HERE/kernel_ab.pbs"
check grep -qF 'e5eccd7a56482522acd5690800bf7438149997f5' "$HERE/solver_ab.pbs"
check grep -qF 'full_solve=ok' "$HERE/solver_ab.pbs"
check grep -qF 'full_solve=SKIPPED' "$HERE/solver_ab.pbs"
check grep -qF 'SOLVER_AB kkt_verification=ok' "$HERE/solver_ab.pbs"
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/focused.pbs"
check grep -qF 'SUCCESS' "$HERE/focused.pbs"
check grep -qF -- '-t 4' "$HERE/focused.pbs"
check grep -qF 'multifloat_linear_algebra_integration.jl' "$HERE/focused.pbs"
check grep -qF ':solve,' "$HERE/focused.pbs"
check grep -qF ':cholesky_factor!' "$HERE/focused.pbs"
check grep -qF 'auto selection ignores provider presence' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF '642d9d30-8e28-45ca-9d81-256429ea358f' "$HERE/focused.pbs"

echo "checking kernel A/B contracts"
check grep -qF 'using Random' "$HERE/kernel_ab.jl"
check grep -qF 'Random.seed!(0x1234)' "$HERE/kernel_ab.jl"
check grep -qF 'copyto!(' "$HERE/kernel_ab.jl"
check grep -qF 'LinearAlgebra.cholesky' "$HERE/kernel_ab.jl"
check grep -qF 'LinearAlgebra.ldiv!' "$HERE/kernel_ab.jl"
check grep -qF 'transpose(R0) * R0' "$HERE/kernel_ab.jl"
check grep -qF 'Symmetric(copy(SPD0), :L)' "$HERE/kernel_ab.jl"
check grep -qF ':cholesky_factor!' "$HERE/kernel_ab.jl"
check grep -qF ':solve!' "$HERE/kernel_ab.jl"
check grep -qF ':direct_upstream' "$HERE/kernel_ab.jl"
check grep -qF ':sdpx_provider' "$HERE/kernel_ab.jl"
check grep -qF 'syrk_contract' "$HERE/kernel_ab.jl"
check grep -qF ':lower_triangle' "$HERE/kernel_ab.jl"
check grep -qF '_lower_triangle_residual_metrics' "$HERE/kernel_ab.jl"
check grep -qF '@allocated' "$HERE/kernel_ab.jl"
check grep -qF 'Sys.maxrss()' "$HERE/kernel_ab.jl"
check grep -qF 'max_relative_residual' "$HERE/kernel_ab.jl"
check grep -qF 'CANDIDATE_PATHOF ' "$HERE/kernel_ab.jl"
check grep -qF 'MFLA_ROOT ' "$HERE/kernel_ab.jl"
check grep -qF 'RUNTIME_CONTRACT julia=' "$HERE/kernel_ab.jl"

echo "checking solver A/B contracts"
check grep -qF 'hasfield(SolverOptions{T}, :linear_algebra_backend)' "$HERE/solver_ab.jl"
check grep -qF 'linear_algebra_backend=requested' "$HERE/solver_ab.jl"
check grep -qF 'SDPX.solve(prob, options)' "$HERE/solver_ab.jl"
check grep -qF 'SDPX.resolve_solve_options' "$HERE/solver_ab.jl"
check grep -qF 'SDPX.result_certificate(prob, result, core_opts)' "$HERE/solver_ab.jl"
check grep -qF 'certificate.valid' "$HERE/solver_ab.jl"
check_not grep -qF 'la_backend=requested' "$HERE/solver_ab.jl"
check grep -qF 'full_solve=ok' "$HERE/solver_ab.jl"
check grep -qF 'full_solve=SKIPPED' "$HERE/solver_ab.jl"
check grep -qF 'gap_rel' "$HERE/solver_ab.jl"
check grep -qF 'p_res' "$HERE/solver_ab.jl"
check grep -qF 'd_res' "$HERE/solver_ab.jl"
check grep -qF 'iterations' "$HERE/solver_ab.jl"
check grep -qF 'certificate_valid' "$HERE/solver_ab.jl"
check grep -qF 'max_relative_residual' "$HERE/solver_ab.jl"
check grep -qF 'max_relative_error_vs_reference' "$HERE/solver_ab.jl"
check grep -qF 'kkt_verification=ok' "$HERE/solver_ab.jl"
check_not grep -qF 'status=:Optimal' "$HERE/solver_ab.jl"
check grep -qF 'Random.seed!(0xabcd)' "$HERE/solver_ab.jl"
check grep -qF 'transpose(R0) * R0' "$HERE/solver_ab.jl"
check grep -qF 'LinearAlgebra.cholesky' "$HERE/solver_ab.jl"
check grep -qF 'LinearAlgebra.ldiv!' "$HERE/solver_ab.jl"

echo "checking README"
check grep -qF 'e5eccd7a56482522acd5690800bf7438149997f5' "$HERE/README.md"
check grep -qF '642d9d30-8e28-45ca-9d81-256429ea358f' "$HERE/README.md"
check grep -qF 'bootstrap_login_env.sh' "$HERE/README.md"
check grep -qF 'focused.pbs' "$HERE/README.md"
check grep -qF 'kernel_ab.pbs' "$HERE/README.md"
check grep -qF 'solver_ab.pbs' "$HERE/README.md"
check grep -qF 'static_check.sh' "$HERE/README.md"
check grep -qF 'lower_triangle' "$HERE/README.md"
check grep -qF 'full_solve=SKIPPED' "$HERE/README.md"
check grep -qF 'Submitting both A/B jobs in parallel' "$HERE/README.md"
check grep -qF 'cpu_utilization' "$HERE/README.md"
check grep -qF 'direct_upstream' "$HERE/README.md"

echo "checking bootstrap guard"
check grep -qF 'bootstrap refuses to run inside a PBS job' "$HERE/bootstrap_login_env.sh"

echo "checking Julia syntax with Meta.parseall (parse-only, no package loading)"
JULIA_CMD="${JULIA_BIN:-}"
if [ -z "$JULIA_CMD" ] && command -v julia >/dev/null 2>&1; then
  JULIA_CMD="$(command -v julia)"
fi
if [ -n "$JULIA_CMD" ]; then
  for file in "$HERE"/*.jl; do
    check "$JULIA_CMD" --startup-file=no -e \
      "Meta.parseall(read(\"$file\", String)); println(\"parsed\")"
  done
else
  echo "julia not found; skipping Meta.parseall syntax check"
fi

echo "checking commit scope (only $PROBE_REL may change)"
check test -d "$ROOT"
out_of_scope=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  path="$(printf '%s' "$line" | sed -E 's/^.{3}//')"
  case "$path" in
    "$PROBE_REL"|"$PROBE_REL/"*) ;;
    *)
      echo "FAIL: out-of-scope change: $line"
      out_of_scope=1
      ;;
  esac
done < <(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)
[ "$out_of_scope" -eq 0 ] || fail=1

echo "checking git whitespace"
check git -C "$ROOT" diff --check

if [ "$fail" -ne 0 ]; then
  echo "static check FAILED"
  exit 1
fi
echo "static check PASSED"
