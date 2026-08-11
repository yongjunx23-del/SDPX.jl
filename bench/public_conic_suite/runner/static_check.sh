#!/bin/bash
# Static preflight for the generated pathological P0 runner.
# No Julia, Pkg, or numeric execution happens here.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SUITE="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$SUITE/../.." && pwd)"
fail=0

check() {
  if ! "$@"; then
    echo "FAIL: $*"
    fail=1
  fi
}

must_absent() {
  local label="$1"
  shift
  if "$@"; then
    echo "FAIL: $label unexpectedly matched"
    fail=1
  fi
}

echo "checking shell syntax"
for script in "$HERE"/*.sh "$HERE"/*.pbs; do
  check bash -n "$script"
done

echo "checking required files"
for file in \
  result_schema.jl \
  campaign.jl \
  run_generated_pathological.jl \
  static_check.jl \
  static_check.sh \
  generated_pathological.pbs \
  generated_pathological_bigfloat.pbs \
  submit_generated_pathological.sh; do
  check test -s "$HERE/$file"
done

echo "checking PBS resource contract"
check grep -q '#PBS -q sugon' "$HERE/generated_pathological.pbs"
check grep -q '#PBS -q sugon' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q '#PBS -l nodes=1:ppn=5' "$HERE/generated_pathological.pbs"
check grep -q -- '-t 4' "$HERE/generated_pathological.pbs"
check grep -q -- '--resource-class=regular' "$HERE/generated_pathological.pbs"
check grep -q '#PBS -l nodes=1:ppn=1' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q -- '-t 1' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q -- '--resource-class=bigfloat' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q 'OPENBLAS_NUM_THREADS=1' "$HERE/generated_pathological.pbs"
check grep -q 'OPENBLAS_NUM_THREADS=1' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q 'ppn=1' "$HERE/campaign.jl"
check grep -q 'ppn=5' "$HERE/campaign.jl"
check grep -q 'NODE_NAME' "$HERE/generated_pathological.pbs"
check grep -q 'NODE_NAME' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q '/usr/bin/time -v' "$HERE/generated_pathological.pbs"
check grep -q '/usr/bin/time -v' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q 'already exists; refusing' "$HERE/generated_pathological.pbs"
check grep -q 'already exists; refusing' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q 'NODE_NAME:?set NODE_NAME' "$HERE/generated_pathological.pbs"
check grep -q 'NODE_NAME:?set NODE_NAME' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q 'SDPX_SITE_ENV:?set' "$HERE/generated_pathological.pbs"
check grep -q 'SDPX_ENVIRONMENT:?set' "$HERE/generated_pathological.pbs"
check grep -q 'SDPX_DEPOT_PATH:?set' "$HERE/generated_pathological.pbs"
check grep -q '/usr/bin/time -v -o' "$HERE/generated_pathological.pbs"
check grep -q '/usr/bin/time -v -o' "$HERE/generated_pathological_bigfloat.pbs"

echo "checking campaign root and node pinning"
check grep -q 'SDPX_CAMPAIGN_ROOT:?set' "$HERE/submit_generated_pathological.sh"
check grep -q 'must not live inside the candidate source' \
  "$HERE/submit_generated_pathological.sh"
check grep -q 'qsub -v .* -l "nodes=\$node:ppn=\$ppn"' \
  "$HERE/submit_generated_pathological.sh"
check grep -q 'REGULAR_NODE_NAME:?set' "$HERE/submit_generated_pathological.sh"
check grep -q 'BIGFLOAT_NODE_NAME:?set' "$HERE/submit_generated_pathological.sh"
must_absent "submit pre-creates result dirs" \
  grep -q 'mkdir -p "\$SDPX_CAMPAIGN_ROOT' "$HERE/submit_generated_pathological.sh"

echo "checking candidate metadata"
check grep -q 'source_commit.txt' "$HERE/run_generated_pathological.jl"
check grep -q 'archive_sha256.txt' "$HERE/run_generated_pathological.jl"

echo "checking generator integration"
check grep -q 'generators.*SDPXPathologicalBenchmarks' "$HERE/run_generated_pathological.jl"
check grep -q 'using .SDPXPathologicalBenchmarks' "$HERE/run_generated_pathological.jl"
check grep -q 'campaign_rows_for' "$HERE/campaign.jl"
check grep -q 'campaign_rows_for' "$HERE/run_generated_pathological.jl"

echo "checking CLI convention"
check grep -q 'values\["--" \* name\]' "$HERE/run_generated_pathological.jl"

echo "checking offline enforcement"
check grep -q 'JULIA_PKG_OFFLINE=true' "$HERE/generated_pathological.pbs"
check grep -q 'JULIA_PKG_OFFLINE=true' "$HERE/generated_pathological_bigfloat.pbs"
check grep -q 'JULIA_PKG_OFFLINE' "$HERE/run_generated_pathological.jl"
must_absent "Pkg operations in runner" grep -RInE 'Pkg\.|using Pkg|Base\.download|Downloads|HTTP\.' "$HERE" --include='*.jl'

echo "checking schema audit coverage"
for column in \
  raw_status raw_moi_status raw_primal_status raw_dual_status \
  normalized_status expected_status expected_normalized_status \
  planned_backend executed_backend planned_formulation executed_formulation \
  lp_formulation \
  kkt_backend gram_kernel equality_method solver_algorithm \
  backend_resolution fallback fallback_reason \
  objective_primal objective_dual objective_expected objective_error \
  objective_relative_error relative_gap \
  primal_residual_original dual_residual_original \
  primal_affine_residual_original dual_affine_residual_original \
  primal_cone_violation_original dual_cone_violation_original \
  primal_residual_scaled_original dual_residual_scaled_original \
  equality_backward_error_original dual_backward_error_original \
  complementarity_relative cone_margin_primal cone_margin_dual \
  certificate_type certificate_valid certificate_residual \
  certificate_failures certificate_validation_precision_bits \
  setup_seconds solve_seconds total_seconds compile_seconds \
  presolve_seconds factorization_seconds schur_assembly_seconds \
  refinement_seconds iterations restarts regularizations refinement_steps \
  workspace_bytes process_peak_rss_bytes \
  julia_threads blas_threads blas_vendor solver_threads pbs_ppn \
  source_sha256 input_sha256 environment_sha256 \
  archive_sha environment_project_path environment_project_sha256 \
  environment_manifest_sha256 gate_route \
  candidate_pathof candidate_pathof_sha256 candidate_pathof_match \
  sdpx_git_sha julia_version os cpu_model \
  gate_identity gate_resource gate_status gate_objective gate_certificate \
  gate_failures gate_pass; do
  check grep -q ":$column" "$HERE/result_schema.jl"
done

echo "checking SUCCESS gating"
check grep -q 'write_success_marker' "$HERE/run_generated_pathological.jl"
check grep -q 'gate_pass = !isempty(all_rows) && isempty(failures)' \
  "$HERE/run_generated_pathological.jl"
check grep -q 'isempty(failures)' "$HERE/run_generated_pathological.jl"
check grep -q '"environment_sha256" => combined_environment_sha256' \
  "$HERE/run_generated_pathological.jl"
check grep -q 'environment_sha256' "$HERE/result_schema.jl"
check grep -q 'Base.active_project' "$HERE/run_generated_pathological.jl"
check grep -q 'project_sha256' "$HERE/run_generated_pathological.jl"
check grep -q 'manifest_sha256' "$HERE/run_generated_pathological.jl"
check grep -q '!isempty(ctx.archive_sha)' "$HERE/run_generated_pathological.jl"
check grep -q '!isempty(ctx.git_sha)' "$HERE/run_generated_pathological.jl"
check grep -q '!isempty(ctx.environment_project_path)' \
  "$HERE/run_generated_pathological.jl"
check grep -q '!isempty(ctx.candidate_pathof_sha256)' \
  "$HERE/run_generated_pathological.jl"

echo "checking weak status and route gates"
check grep -q 'unresolved_or_certified_infeasible' "$HERE/result_schema.jl"
check grep -q 'HONEST_UNRESOLVED' "$HERE/result_schema.jl"
check grep -q 'allow_unresolved && normalized_raw in HONEST_UNRESOLVED' \
  "$HERE/result_schema.jl"
check grep -q 'natural_objective' "$HERE/result_schema.jl"
check grep -q 'natural_objective(result.pObj' \
  "$HERE/run_generated_pathological.jl"
check grep -q 'MOI.MAX_SENSE' "$HERE/run_generated_pathological.jl"
check grep -q 'lp_klee_minty.*sdp_hilbert.*sdp_small_eigenvalue' \
  "$HERE/static_check.jl"
check grep -q 'gate_route = true' "$HERE/run_generated_pathological.jl"
check grep -q 'resolved_no_iteration' "$HERE/run_generated_pathological.jl"
check grep -q 'executed_backend_text == "not_executed"' \
  "$HERE/run_generated_pathological.jl"
check grep -q 'certificate_cone_margins' "$HERE/run_generated_pathological.jl"
check grep -q 'required_shift' "$HERE/run_generated_pathological.jl"

echo "checking schema functions"
check grep -q 'function normalize_status' "$HERE/result_schema.jl"
check grep -q 'function expected_normalized_status' "$HERE/result_schema.jl"
check grep -q 'function check_columns' "$HERE/result_schema.jl"
check grep -q 'allow_unresolved' "$HERE/result_schema.jl"
check grep -q '"unresolved"' "$HERE/result_schema.jl"

echo "checking review regressions"
check grep -q 'min_bits' "$HERE/campaign.jl"
check grep -q 'campaign_rows_for' "$HERE/campaign.jl"
check grep -q '_candidate_metadata' "$HERE/run_generated_pathological.jl"
check grep -q 'candidate_metadata.toml' "$HERE/run_generated_pathological.jl"
check grep -q 'merged = merge(base_fields, overrides)' \
  "$HERE/run_generated_pathological.jl"
check grep -q 'isfinite(relative_gap)' "$HERE/run_generated_pathological.jl"
check grep -q 'relative_gap <= tol_T' "$HERE/run_generated_pathological.jl"
check grep -q '_typed_value' "$HERE/run_generated_pathological.jl"

echo "checking whitespace"
(
  cd "$REPO"
  git diff --check
)

if [ "$fail" = "1" ]; then
  echo "static check FAILED"
  exit 1
fi
echo "static check PASSED"
