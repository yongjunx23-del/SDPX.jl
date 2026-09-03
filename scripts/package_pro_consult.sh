#!/bin/bash
# Package the complete SDPX.jl source for a GPT Pro consultation round.
# Usage: scripts/package_pro_consult.sh <output-dir>
set -euo pipefail
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "refusing to package a dirty tracked tree" >&2
  exit 1
fi
SHA=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)
if [ "$#" -ne 1 ]; then
  echo "usage: scripts/package_pro_consult.sh <output-dir>" >&2
  exit 2
fi
OUT="$1"
mkdir -p "$OUT"
git archive "$SHA" --prefix=source/ | gzip > "$OUT/context-$SHA.tar.gz"
shasum -a 256 "$OUT/context-$SHA.tar.gz" > "$OUT/context-$SHA.sha256"
cp HANDOVER.md "$OUT/HANDOVER.md"
cp docs/design/CANONICAL_FORM.md "$OUT/"
cp docs/design/HSD_FORMULATION.md "$OUT/"
cp docs/design/NEWTON_SYSTEM.md "$OUT/"
cat > "$OUT/REVIEW_CONTEXT.md" <<EOF
# SDPX review context

- Source SHA: \`$SHA\`
- Branch: \`$BRANCH\`
- Packaged UTC: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`
- Tracked tree: clean

Current plan and handover: \`HANDOVER.md\`.
Frozen contracts: \`CANONICAL_FORM.md\`, \`HSD_FORMULATION.md\`, and
\`NEWTON_SYSTEM.md\`. Review claims against the archived source, not against
historical conversation context.
EOF
echo "packaged at $OUT (SHA $SHA)"
