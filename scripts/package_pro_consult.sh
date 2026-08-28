#!/bin/bash
# Package the complete SDPX.jl source for a GPT Pro consultation round.
# Usage: scripts/package_pro_consult.sh <output-dir>
set -euo pipefail
SHA=$(git rev-parse HEAD)
if [ "$#" -ne 1 ]; then
  echo "usage: scripts/package_pro_consult.sh <output-dir>" >&2
  exit 2
fi
OUT="$1"
mkdir -p "$OUT"
git archive "$SHA" --prefix=source/ | gzip > "$OUT/context-$SHA.tar.gz"
shasum -a 256 "$OUT/context-$SHA.tar.gz" > "$OUT/context-$SHA.sha256"
cp docs/PLAN.md "$OUT/PLAN.md"
cp docs/design/CANONICAL_FORM.md "$OUT/"
cp docs/design/HSD_FORMULATION.md "$OUT/"
cp docs/design/NEWTON_SYSTEM.md "$OUT/"
cat > "$OUT/PROGRESS_SINCE_LAST_AUDIT.md" <<EOF
# Progress since the 6.0/10 maturity audit

## Landed (all merged to wave/v4-a0)
1. P1.5 stabilization (audit findings B1 trial-mu, B3 terminal-result ownership,
   B4 verification ordering, I1 dkappa dual-candidate, I3 RSOC O(n)/alias/precision,
   I6 ray tolerance, F8 provenance SHA + include-uniqueness test)
2. Wave B: expanded-KKT inertia control, static/dynamic signed regularization,
   same-factor multi-RHS refinement certified vs unregularized equations;
   rank-one PSD expanded gap CLOSED (optimal -0.5, valid certificate);
   BFLA TransposeOp adapter fix; BFLA/MFLA capability-contract routing
3. Wave C: refinement stagnation-recovery ladder + arithmetic-floor diagnosis;
   cone-preserving Ruiz equilibration (EquilibrationMap, prepared);
   minimal reversible structural presolve (prepared); equality policy selector
4. Wave D: DualInfeasible as certificate-backed first-class termination +
   cone x route status matrix; expanded-route gap diagnosis
5. Cleanup: reconstruction-stack block-scope hazard fixed; KKT exception
   narrowing; NativeHSDPlan records actual kkt_route; canonical layout typed
   (no Any); dispatch chain deepseek(merge)->sol-high
EOF
echo "packaged at $OUT (SHA $SHA)"
