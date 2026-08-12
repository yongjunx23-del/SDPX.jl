#!/bin/bash
# Static preflight for the v041-legacy-provider probe.  Only shell syntax,
# Julia parse-only syntax (`Meta.parseall` with recursive
# `:error`/`:incomplete` rejection), marker checks, and git diff checks are
# executed.  No probe script, Julia solve, SSH, or qsub is executed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PROBE_REL="cluster-probes/v041-legacy-provider"
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
for file in README.md legacy_provider_regression.pbs legacy_provider_gate.jl \
            bigfloat_identity_smoke.jl static_check.sh; do
  check test -s "$HERE/$file"
done

echo "checking PBS resource and identity contract"
for marker in \
  '#PBS -q sugon' \
  '#PBS -l nodes=1:ppn=5' \
  'OPENBLAS_NUM_THREADS=1' \
  'JULIA_PKG_OFFLINE=true' \
  'SDPX_LEGACY_PROVIDER_SYMBOL="${SDPX_LEGACY_PROVIDER_SYMBOL:-sdpx_legacy_la}"' \
  'SDPX_LEGACY_PROVIDER_CALL_MARKER="${SDPX_LEGACY_PROVIDER_CALL_MARKER:-_sdpx_legacy_la_call}"' \
  'SDPX_LEGACY_OWNERSHIP="${SDPX_LEGACY_OWNERSHIP:-owned_mutable_scalars}"' \
  'NODE_NAME:?set NODE_NAME' \
  'CAMPAIGN_ID:?set CAMPAIGN_ID' \
  'OUTPUT_ROOT already exists; refusing' \
  'PBS_NP contract failed' \
  'JULIA_NUM_THREADS contract failed' \
  'runtime_contract=ok' \
  'SUCCESS'; do
  check grep -qF -- "$marker" "$HERE/legacy_provider_regression.pbs"
done
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/legacy_provider_regression.pbs"
check grep -qF 'legacy_provider_regression.pbs' "$HERE/README.md"
check grep -qF 'legacy_provider_gate.jl' "$HERE/README.md"
check grep -qF 'bigfloat_identity_smoke.jl' "$HERE/README.md"
check grep -qF 'static_check.sh' "$HERE/README.md"
check grep -qF 'SDPX_LEGACY_PROVIDER_SYMBOL' "$HERE/README.md"
check grep -qF 'SDPX_LEGACY_PROVIDER_CALL_MARKER' "$HERE/README.md"
check grep -qF 'SDPX_LEGACY_OWNERSHIP' "$HERE/README.md"

echo "checking gate contracts"
check grep -qF 'using SDPX' "$HERE/legacy_provider_gate.jl"
check grep -qF 'using LinearAlgebra' "$HERE/legacy_provider_gate.jl"
check grep -qF 'PROVIDER_GATE=PROVIDER_NOT_FOUND' "$HERE/legacy_provider_gate.jl"
check grep -qF 'PROVIDER_GATE=OK' "$HERE/legacy_provider_gate.jl"
check grep -qF 'PROVIDER_INCLUDED' "$HERE/legacy_provider_gate.jl"
check grep -qF 'PROVIDER_DIRECT_K' "$HERE/legacy_provider_gate.jl"
check grep -qF 'walkdir' "$HERE/legacy_provider_gate.jl"
check grep -qF 'sort!(dirnames)' "$HERE/legacy_provider_gate.jl"
check grep -qF 'sort!(filenames)' "$HERE/legacy_provider_gate.jl"
check grep -qF 'sort!(sources)' "$HERE/legacy_provider_gate.jl"
check grep -qF 'ROUTED_LEGACY_OPERATIONS' "$HERE/legacy_provider_gate.jl"
check grep -qF 'EXPECTED_PROVIDER_CALL_MARKER' "$HERE/legacy_provider_gate.jl"
check grep -qF 'EXPECTED_LEGACY_OWNERSHIP' "$HERE/legacy_provider_gate.jl"
check grep -qF '_is_legacy_dispatch_signature' "$HERE/legacy_provider_gate.jl"
check grep -qF 'LegacyLACholeskyFactor' "$HERE/legacy_provider_gate.jl"
check grep -qF 'SDPXLegacyLAProvider' "$HERE/legacy_provider_gate.jl"
check grep -qF 'SDPX.SDPX_LEGACY_LA_CAPABILITIES' "$HERE/legacy_provider_gate.jl"
check grep -qF 'legacy_la_provider_identity' "$HERE/legacy_provider_gate.jl"
check grep -qF 'legacy_la_provider_arithmetic' "$HERE/legacy_provider_gate.jl"
check grep -qF 'legacy_la_provider_capabilities' "$HERE/legacy_provider_gate.jl"
check grep -qF 'legacy_la_provider_ownership' "$HERE/legacy_provider_gate.jl"
check grep -qF 'legacy_la_provider_supports' "$HERE/legacy_provider_gate.jl"
check grep -qF '_sdpx_legacy_la_call' "$HERE/legacy_provider_gate.jl"
check grep -qF 'owned_mutable_scalars' "$HERE/legacy_provider_gate.jl"
check grep -qF 'plan_la_backend' "$HERE/legacy_provider_gate.jl"
check grep -qF 'instantiate_la_backend' "$HERE/legacy_provider_gate.jl"
check grep -qF 'LegacyLABackend' "$HERE/legacy_provider_gate.jl"
check grep -qF 'la_backend_provider' "$HERE/legacy_provider_gate.jl"
check grep -qF 'la_backend_ownership' "$HERE/legacy_provider_gate.jl"
check grep -qF 'relpath(source, joinpath(root, "src"))' "$HERE/legacy_provider_gate.jl"
check grep -qF 'kdot!' "$HERE/legacy_provider_gate.jl"
check grep -qF 'kdot_columns!' "$HERE/legacy_provider_gate.jl"
check grep -qF 'alloc_zeros' "$HERE/legacy_provider_gate.jl"
check grep -qF 'copy_owned!' "$HERE/legacy_provider_gate.jl"
echo "checking smoke contracts"
check grep -qF 'full_solve_standard=ok' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'full_solve_legacy=ok' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'identity=ok' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'WORKING_BITS' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'REFERENCE_BITS' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'setprecision(BigFloat, WORKING_BITS)' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'SDPX.alloc_zeros' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'SDPX.copy_owned!' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'SDPX.solve(prob, options)' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'SDPX.resolve_solve_options' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'SDPX.result_certificate' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'certificate.valid' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'planned_la_backend' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'executed_la_backend' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'la_provider' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'la_executed_provider' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'la_ownership' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'la_executed_ownership' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF ':requested_legacy' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'unauthorized runtime LA fallback' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'objective mismatch' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'Random.seed!(0x3141)' "$HERE/bigfloat_identity_smoke.jl"
check grep -qF 'RUNTIME_CONTRACT julia=4 plan=4 blas=1' "$HERE/bigfloat_identity_smoke.jl"
check_not grep -qF 'legacy_solution == standard_solution' "$HERE/bigfloat_identity_smoke.jl"
check_not grep -qF 'x == y' "$HERE/bigfloat_identity_smoke.jl"

echo "checking README"
check grep -qF 'ppn=5' "$HERE/README.md"
check grep -qF 'does not duplicate the full' "$HERE/README.md"
check grep -qF 'node70' "$HERE/README.md"
check grep -qF 'node187' "$HERE/README.md"
check grep -qF 'pbsnodes -a' "$HERE/README.md"
check grep -qF 'SDPXLegacyLA' "$HERE/README.md"
check grep -qF 'src/la_backends/legacy.jl' "$HERE/README.md"
check grep -qF 'SDPXLegacyLAProvider' "$HERE/README.md"
check grep -qF 'LegacyLACholeskyFactor' "$HERE/README.md"
check grep -qF '_sdpx_legacy_la_call' "$HERE/README.md"
check grep -qF 'SDPX_LEGACY_LA_CAPABILITIES' "$HERE/README.md"
check grep -qF ':owned_mutable_scalars' "$HERE/README.md"

echo "checking migration document"
check test -s "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'SDPXLegacyLA' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'ksyrk!' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'mirrors both triangles' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'Future extraction' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'kdot!' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'kdot_columns!' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'alloc_zeros' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'copy_owned!' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'not current provider capabilities' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'Phased Deletion Checklist' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'PROVIDER_GATE=OK' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'node70' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'src/la_backends/legacy.jl' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'SDPXLegacyLAProvider' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'SDPX_LEGACY_LA_CAPABILITIES' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'LegacyLACholeskyFactor' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF 'legacy_la_provider_identity' "$ROOT/docs/legacy-la-provider-migration.md"
check grep -qF ':owned_mutable_scalars' "$ROOT/docs/legacy-la-provider-migration.md"

echo "checking Julia syntax with Meta.parseall (parse-only, no package loading)"
JULIA_CMD="${JULIA_BIN:-}"
if [ -z "$JULIA_CMD" ] && command -v julia >/dev/null 2>&1; then
  JULIA_CMD="$(command -v julia)"
fi
if [ -n "$JULIA_CMD" ]; then
  for file in "$HERE"/*.jl; do
    check "$JULIA_CMD" --startup-file=no -e \
      "ex = Meta.parseall(read(\"$file\", String)); bad = Symbol[]; function walk(e); e isa Expr || return; e.head in (:error, :incomplete) && push!(bad, e.head); foreach(walk, e.args); end; walk(ex); isempty(bad) || error(\"AST error/incomplete nodes: \$bad\"); println(\"parsed\")"
  done
else
  echo "julia not found; skipping Meta.parseall syntax check"
fi

echo "checking commit scope"
check test -d "$ROOT"
out_of_scope=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  path="$(printf '%s' "$line" | sed -E 's/^.{3}//')"
  case "$path" in
    "$PROBE_REL"|"$PROBE_REL/"*) ;;
    "docs/legacy-la-provider-migration.md") ;;
    "src/SDPX.jl"|"src/types.jl"|"src/la_backend.jl"|\
    "src/la_backends/legacy.jl"|"src/kkt.jl"|\
    "src/solver/interior_point.jl"|"src/public_api.jl"|\
    "test/la_backend_regressions.jl") ;;
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
