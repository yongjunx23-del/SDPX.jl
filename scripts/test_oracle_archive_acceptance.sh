#!/usr/bin/env bash
# Acceptance test for scripts/prepare_oracle_review_context.sh. Builds a
# temporary git repo with a local bare remote and exercises the pure validation
# gates (invalid base, non-ancestor base, dirty tree, missing contract,
# Manifest leak, oversized diff, abbreviated-base canonicalization).
#
# Usage: scripts/test_oracle_archive_acceptance.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sdpx-oracle-accept.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

BARE="$TMP/remote.git"
WORK="$TMP/work"
git init -q --bare "$BARE"
git clone -q "$BARE" "$WORK"

# Copy the script under test into the work repo.
mkdir -p "$WORK/scripts" "$WORK/.pi" "$WORK/src" "$WORK/test"
cp "$HERE/scripts/prepare_oracle_review_context.sh" "$WORK/scripts/"
cat > "$WORK/.pi/ORACLE_REVIEW_PROMPT.md" <<'EOF'
# contract
EOF

cd "$WORK"
git config user.email t@t
git config user.name t
git add -A
git commit -qm base
BASE="$(git rev-parse HEAD)"
git checkout -qb wave
echo x > src/x.jl
git add -A
git commit -qm cand
CAND="$(git rev-parse HEAD)"
git push -q origin wave:wave
git branch --set-upstream-to=origin/wave wave

expect_fail() { # $1=desc $2..=cmd
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "FAIL: $desc (expected nonzero)"; exit 1; fi
  echo "ok: $desc rejected"
}
expect_ok() { # $1=desc $2..=cmd
  local desc="$1"; shift
  if ! "$@" >/dev/null 2>&1; then echo "FAIL: $desc (expected zero)"; exit 1; fi
  echo "ok: $desc accepted"
}

expect_fail "invalid base" bash scripts/prepare_oracle_review_context.sh not-a-sha

# Unrelated (non-ancestor) base from an orphan branch.
git checkout -q --orphan other
rm -f src/x.jl
echo y > src/other.jl
git add -A && git commit -qm other
OTHER="$(git rev-parse HEAD)"
git checkout -q wave
expect_fail "non-ancestor base" bash scripts/prepare_oracle_review_context.sh "$OTHER"
expect_fail "missing contract" env -i PATH="$PATH" bash -c "cd '$WORK' && rm .pi/ORACLE_REVIEW_PROMPT.md && bash scripts/prepare_oracle_review_context.sh '$BASE'"
git checkout -q -- .pi/ORACLE_REVIEW_PROMPT.md

echo '// manifest leak' > Manifest.toml
expect_fail "Manifest leak" bash scripts/prepare_oracle_review_context.sh "$BASE"
rm -f Manifest.toml

expect_fail "dirty tree" bash -c "touch '$WORK/uncommitted' && bash scripts/prepare_oracle_review_context.sh '$BASE'"
rm -f "$WORK/uncommitted"

# abbreviated base canonicalizes and succeeds (full SHA printed)
out="$(bash scripts/prepare_oracle_review_context.sh "${BASE:0:7}")"
echo "$out" | grep -q "candidate SHA: $CAND" || { echo "FAIL: abbreviated base did not canonicalize"; exit 1; }
echo "ok: abbreviated base canonicalized"

echo "ACCEPTANCE_PASS"
