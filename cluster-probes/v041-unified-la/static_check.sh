# Static preflight for the v041-unified-la cluster probes.  Only shell
# syntax, Julia parse-only syntax (`Meta.parseall` with recursive
# `:error`/`:incomplete` rejection), and git diff checks are executed.
# No probe script, Julia solve, SSH, or qsub is executed.
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
            bigfloat_generic_probe.pbs kernel_ab.jl solver_ab.jl \
            bigfloat_generic_probe.jl bootstrap_login_env.sh static_check.sh; do
  check test -s "$HERE/$file"
done

echo "checking PBS resource and identity contract"
for job in focused kernel_ab solver_ab bigfloat_generic_probe; do
  check grep -qF '#PBS -q sugon' "$HERE/$job.pbs"
  if [ "$job" != "bigfloat_generic_probe" ]; then
    check grep -qF '#PBS -l nodes=1:ppn=5' "$HERE/$job.pbs"
  fi
  check grep -qF 'OPENBLAS_NUM_THREADS=1' "$HERE/$job.pbs"
  check grep -qF 'JULIA_PKG_OFFLINE=true' "$HERE/$job.pbs"
  check grep -qF '/usr/bin/time -v' "$HERE/$job.pbs"
  check grep -qF 'already exists; refusing' "$HERE/$job.pbs"
  check grep -qF 'NODE_NAME:?set NODE_NAME' "$HERE/$job.pbs"
  check grep -qF 'CAMPAIGN_ID:?set CAMPAIGN_ID' "$HERE/$job.pbs"
  if [ "$job" != "bigfloat_generic_probe" ]; then
    check grep -qF 'MFLA_COMMIT:?set MFLA_COMMIT' "$HERE/$job.pbs"
    check grep -qF '$(dirname "$MFLA_SOURCE")/COMMIT' "$HERE/$job.pbs"
  fi
  check grep -qF 'PBS_NP contract failed' "$HERE/$job.pbs"
  check grep -qF 'JULIA_NUM_THREADS contract failed' "$HERE/$job.pbs"
  check grep -qF 'runtime_contract=ok' "$HERE/$job.pbs"
  check grep -qF 'SUCCESS' "$HERE/$job.pbs"
done
for job in kernel_ab solver_ab bigfloat_generic_probe; do
  check grep -qF 'if (w > 0) { printf "%.3f", (u+s)/w } else { printf "0.000" }' "$HERE/$job.pbs"
  check_not grep -qF 'printf "%.3f", w > 0 ? (u+s)/w : 0' "$HERE/$job.pbs"
done
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/focused.pbs"
check grep -qF 'KERNEL_AB ' "$HERE/kernel_ab.pbs"
check grep -qF 'SOLVER_AB ' "$HERE/solver_ab.pbs"
check grep -qF 'BIGFLOAT_AB ' "$HERE/bigfloat_generic_probe.pbs"
check grep -qF 'CANDIDATE_PATHOF ' "$HERE/kernel_ab.pbs"
check grep -qF 'CANDIDATE_PATHOF ' "$HERE/solver_ab.pbs"
check grep -qF 'CANDIDATE_PATHOF ' "$HERE/bigfloat_generic_probe.pbs"
check grep -qF 'MFLA_ROOT ' "$HERE/kernel_ab.pbs"
check grep -qF 'MFLA_ROOT ' "$HERE/solver_ab.pbs"
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/kernel_ab.pbs"
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/solver_ab.pbs"
check grep -qF 'RUNTIME_CONTRACT julia=1 plan=1 blas=1' "$HERE/bigfloat_generic_probe.pbs"
check grep -qF 'Maximum resident set size' "$HERE/kernel_ab.pbs"
check grep -qF 'Maximum resident set size' "$HERE/solver_ab.pbs"
check grep -qF 'Maximum resident set size' "$HERE/bigfloat_generic_probe.pbs"
check grep -qF 'rss_kib=' "$HERE/kernel_ab.pbs"
check grep -qF 'rss_kib=' "$HERE/solver_ab.pbs"
check grep -qF 'rss_kib=' "$HERE/bigfloat_generic_probe.pbs"
check grep -qF 'cpu_utilization=' "$HERE/kernel_ab.pbs"
check grep -qF 'cpu_utilization=' "$HERE/solver_ab.pbs"
check grep -qF 'cpu_utilization=' "$HERE/bigfloat_generic_probe.pbs"
check grep -qF 'mfla_commit_expected' "$HERE/kernel_ab.pbs"
check grep -qF 'mfla_commit_expected' "$HERE/solver_ab.pbs"
check grep -qF 'e5eccd7a56482522acd5690800bf7438149997f5' "$HERE/kernel_ab.pbs"
check grep -qF 'e5eccd7a56482522acd5690800bf7438149997f5' "$HERE/solver_ab.pbs"
check grep -qF '#PBS -l nodes=1:ppn=1' "$HERE/bigfloat_generic_probe.pbs"
check grep -qF 'JULIA_NUM_THREADS=1' "$HERE/bigfloat_generic_probe.pbs"
check grep -qF 'SDPX_SOLVER_THREADS=1' "$HERE/bigfloat_generic_probe.pbs"
check grep -qF 'BIGFLOAT_AB full_solve=ok' "$HERE/bigfloat_generic_probe.pbs"
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
check grep -qF 'function _sorted_summary(record)' "$HERE/solver_ab.jl"
check grep -qF 'collect(pairs(record))' "$HERE/solver_ab.jl"
check grep -qF 'string(first(pair))' "$HERE/solver_ab.jl"
check_not grep -qF 'sort(collect(standard); by=first)' "$HERE/solver_ab.jl"
check_not grep -qF 'sort(collect(multifloat); by=first)' "$HERE/solver_ab.jl"
check_not grep -qF 'sort(collect(std_full); by=first)' "$HERE/solver_ab.jl"
check_not grep -qF 'sort(collect(mf_full); by=first)' "$HERE/solver_ab.jl"
check_not grep -qF 'status=:Optimal' "$HERE/solver_ab.jl"
check grep -qF 'Random.seed!(0xabcd)' "$HERE/solver_ab.jl"
check grep -qF 'transpose(R0) * R0' "$HERE/solver_ab.jl"
check grep -qF 'LinearAlgebra.cholesky' "$HERE/solver_ab.jl"
check grep -qF 'LinearAlgebra.ldiv!' "$HERE/solver_ab.jl"

echo "checking BigFloat generic probe contracts"
check grep -qF 'SDPX.alloc_zeros' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'SDPX.copy_owned!' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'linear_algebra_backend=requested' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'SDPX.solve(prob, options)' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'SDPX.resolve_solve_options' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'SDPX.result_certificate(prob, result, core_opts)' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'certificate.valid' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'BigFloat("1e-40")' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'BigFloat("1e-30")' "$HERE/bigfloat_generic_probe.jl"
check grep -qF ':la_fallback_reason' "$HERE/bigfloat_generic_probe.jl"
check grep -qF ':planned_la_fallback_reason' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'runtime_la_fallback_reason' "$HERE/bigfloat_generic_probe.jl"
check grep -qF ':requested_legacy' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'unauthorized runtime LA fallback' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'verification.executed_la_backend == requested' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'function _sorted_summary(record)' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'collect(pairs(record))' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'string(first(pair))' "$HERE/bigfloat_generic_probe.jl"
check_not grep -qF 'sort(collect(std_full); by=first)' "$HERE/bigfloat_generic_probe.jl"
check_not grep -qF 'sort(collect(legacy_full); by=first)' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'WORKING_BITS' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'REFERENCE_BITS' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'setprecision(BigFloat, WORKING_BITS)' "$HERE/bigfloat_generic_probe.jl"
check grep -qF '_assert_owned_independent!' "$HERE/bigfloat_generic_probe.jl"
check grep -qF '_assert_source_unchanged!' "$HERE/bigfloat_generic_probe.jl"
check grep -qF '_assert_deterministic' "$HERE/bigfloat_generic_probe.jl"
check grep -qF '_expect_factor_failure' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'full_solve_standard' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'full_solve_legacy' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'BIGFLOAT_AB full_solve=ok' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'iterations' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'fallback_reason' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'RUNTIME_CONTRACT julia=1 plan=1 blas=1' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'A_snapshot = _owned_copy(A0)' "$HERE/bigfloat_generic_probe.jl"
check grep -qF 'destination slots' "$HERE/bigfloat_generic_probe.jl"

echo "checking authorized peripheral comment and expectation updates"
check grep -qF 'config.provider === :generic_linear_algebra' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF 'config.ownership === :owned_mutable_scalars' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check_not grep -qF 'config.provider === :none' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF 'rhs = T.(randn(5))' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF 'q = T.(randn(4))' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF 'dy = T.(randn(4))' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF 'transpose(A) * rhs' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF 'A * dy' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF 'lower * transpose(lower)' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF 'transpose(P) * P' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF '_max_relative_error_lower' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check grep -qF 'transpose(R) * R' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check_not grep -qF 'lower_reference' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
check_not grep -qF 'S = T.(randn(n, n))' "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl"
for comment_file in "$HERE/kernel_ab.jl" "$HERE/solver_ab.jl" "$HERE/bigfloat_generic_probe.jl" \
                    "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl" \
                    "$(dirname "$HERE")/../ext/SDPXMultiFloatLinearAlgebraExt.jl"; do
  check grep -qE '^#=$' "$comment_file"
  check grep -qE '^=#$' "$comment_file"
  check_not grep -qE '^#=+#' "$comment_file"
done

echo "checking README"
check grep -qF 'e5eccd7a56482522acd5690800bf7438149997f5' "$HERE/README.md"
check grep -qF '642d9d30-8e28-45ca-9d81-256429ea358f' "$HERE/README.md"
check grep -qF 'bootstrap_login_env.sh' "$HERE/README.md"
check grep -qF 'focused.pbs' "$HERE/README.md"
check grep -qF 'kernel_ab.pbs' "$HERE/README.md"
check grep -qF 'solver_ab.pbs' "$HERE/README.md"
check grep -qF 'bigfloat_generic_probe.pbs' "$HERE/README.md"
check grep -qF 'bigfloat_generic_probe.jl' "$HERE/README.md"
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
  for file in "$HERE"/*.jl \
              "$(dirname "$HERE")/../test/multifloat_linear_algebra_integration.jl" \
              "$(dirname "$HERE")/../ext/SDPXMultiFloatLinearAlgebraExt.jl"; do
    check "$JULIA_CMD" --startup-file=no -e \
      "ex = Meta.parseall(read(\"$file\", String)); bad = Symbol[]; function walk(e); e isa Expr || return; e.head in (:error, :incomplete) && push!(bad, e.head); foreach(walk, e.args); end; walk(ex); isempty(bad) || error(\"AST error/incomplete nodes: \$bad\"); println(\"parsed\")"
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
    "test/multifloat_linear_algebra_integration.jl") ;;
    "ext/SDPXMultiFloatLinearAlgebraExt.jl") ;;
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
