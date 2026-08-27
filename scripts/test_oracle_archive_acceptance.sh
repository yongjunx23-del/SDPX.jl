#!/usr/bin/env bash
# Acceptance test for the Oracle review context and archive builders. Builds a
# temporary git repo with a local bare remote and exercises pure validation
# gates plus the real git-archive staging path, its identity files and digest
# manifest, contract pinning, and receipt adversarial cases.
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

# Copy the scripts under test into the fixture repo. The fixture's ignore file
# models the generated files allowed by the production worktree gate.
mkdir -p "$WORK/scripts" "$WORK/.pi" "$WORK/src" "$WORK/test"
cp "$HERE/scripts/prepare_oracle_review_context.sh" "$WORK/scripts/"
cp "$HERE/scripts/build_oracle_archive.sh" "$WORK/scripts/"
cp "$HERE/scripts/oracle_receipt.sh" "$WORK/scripts/"
cat > "$WORK/.pi/ORACLE_REVIEW_PROMPT.md" <<'EOF'
# contract
EOF
cat > "$WORK/.gitignore" <<'EOF'
.pi/oracle-review-context.md
.pi/oracle-review-context.sha256
.pi/oracle-tree-manifest
.pi/oracle-archive/
.pi/oracle-archive-manifest
.pi/oracle-receipts/
.pi/oracle-review-receipt.json
.pi/WAVE.yaml
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
git branch --set-upstream-to=origin/wave wave >/dev/null

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

# Existing pure-gate checks.
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

# Abbreviated base canonicalizes and succeeds (full SHA printed).
out="$(bash scripts/prepare_oracle_review_context.sh "${BASE:0:7}")"
echo "$out" | grep -q "candidate SHA: $CAND" || { echo "FAIL: abbreviated base did not canonicalize"; exit 1; }
echo "ok: abbreviated base canonicalized"

# Build the actual production staging path through build_oracle_archive.sh.
# This invokes prepare_oracle_review_context.sh against the pushed candidate,
# then exports the candidate with git archive and adds generated identity files.
bash scripts/build_oracle_archive.sh "$BASE" > "$TMP/build.out"
grep -q "candidate SHA: $CAND" "$TMP/build.out" || { echo "FAIL: archive builder used wrong candidate"; exit 1; }
STAGE="$WORK/.pi/oracle-archive"
[ -d "$STAGE" ] || { echo "FAIL: staged archive missing"; exit 1; }

# Exercise the archive boundary itself: tar the staged root and extract it,
# rather than validating only the builder's source directory.
EXTRACT="$TMP/extracted"
mkdir -p "$EXTRACT"
COPYFILE_DISABLE=1 tar -cf "$TMP/oracle.tar" -C "$STAGE" .
COPYFILE_DISABLE=1 tar -xf "$TMP/oracle.tar" -C "$EXTRACT"

[ "$(find "$EXTRACT" -name '._*' -type f | wc -l | tr -d ' ')" = 0 ] || { echo "FAIL: AppleDouble files in extracted archive"; exit 1; }
[ ! -e "$EXTRACT/.git" ] || { echo "FAIL: .git leaked into extracted archive"; exit 1; }
for identity in \
  .pi/oracle-review-context.md \
  .pi/oracle-review-context.sha256 \
  .pi/oracle-tree-manifest \
  .pi/oracle-archive-manifest; do
  [ -f "$EXTRACT/$identity" ] || { echo "FAIL: missing archive identity file: $identity"; exit 1; }
done
echo "ok: extracted archive has no AppleDouble/.git and all identity files"

# Validate every archive-manifest record: exactly mode, SHA-256 and path; the
# path is confined, refers to a regular file, has the recorded mode, and has
# the recorded content digest. The archive manifest intentionally excludes
# itself because it is copied into the stage after being generated.
verify_manifest() {
  local root="$1"
  python3 - "$root" <<'PY'
import hashlib, os, stat, sys
root = sys.argv[1]
manifest = os.path.join(root, ".pi", "oracle-archive-manifest")
seen = set()
with open(manifest, encoding="utf-8") as stream:
    for number, line in enumerate(stream, 1):
        fields = line.rstrip("\n").split(" ", 2)
        if len(fields) != 3:
            raise SystemExit(f"manifest line {number} does not have mode sha256 path")
        mode, digest, rel = fields
        if len(mode) != 3 or any(char not in "01234567" for char in mode):
            raise SystemExit(f"manifest line {number} has invalid mode")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SystemExit(f"manifest line {number} has invalid SHA-256")
        if not rel or rel.startswith("/") or rel == "." or ".." in rel.split("/"):
            raise SystemExit(f"manifest line {number} has unsafe path: {rel}")
        if rel in seen:
            raise SystemExit(f"manifest duplicate path: {rel}")
        seen.add(rel)
        path = os.path.join(root, rel)
        if not os.path.isfile(path) or os.path.islink(path):
            raise SystemExit(f"manifest path is not a regular file: {rel}")
        actual_mode = format(stat.S_IMODE(os.stat(path).st_mode), "03o")
        if actual_mode != mode:
            raise SystemExit(f"mode mismatch for {rel}: {actual_mode} != {mode}")
        actual_digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if actual_digest != digest:
            raise SystemExit(f"digest mismatch for {rel}")
actual_files = set()
for directory, _, names in os.walk(root):
    for name in names:
        rel = os.path.relpath(os.path.join(directory, name), root)
        if rel != ".pi/oracle-archive-manifest":
            actual_files.add(rel)
if seen != actual_files:
    raise SystemExit(f"manifest membership mismatch: missing={sorted(actual_files-seen)} extra={sorted(seen-actual_files)}")
print(f"verified {len(seen)} mode+SHA-256 manifest records")
PY
}
verify_manifest "$EXTRACT"

# Tampering with one extracted source must be detected by the same verifier.
cp "$EXTRACT/src/x.jl" "$TMP/x.jl.original"
printf '\nTAMPERED\n' >> "$EXTRACT/src/x.jl"
expect_fail "tampered archive source" verify_manifest "$EXTRACT"
# Restore the extracted copy for the remaining checks.
cp "$TMP/x.jl.original" "$EXTRACT/src/x.jl"
verify_manifest "$EXTRACT" >/dev/null

echo "ok: archive manifest verifies mode+SHA-256 and detects tampering"

# Pin the original contract outside the candidate tree, mutate the candidate
# contract, and ensure the builder rejects the mismatched trusted pin.
PIN="$TMP/oracle-contract-pin.json"
CONTRACT_BLOB="$(git rev-parse HEAD:.pi/ORACLE_REVIEW_PROMPT.md)"
CONTRACT_SHA="$(shasum -a 256 .pi/ORACLE_REVIEW_PROMPT.md | awk '{print $1}')"
python3 - "$PIN" "$CONTRACT_BLOB" "$CONTRACT_SHA" <<'PY'
import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump({"approved_contract_blob": sys.argv[2], "approved_contract_sha256": sys.argv[3]}, stream)
PY
printf '\nweakened contract\n' >> .pi/ORACLE_REVIEW_PROMPT.md
git add .pi/ORACLE_REVIEW_PROMPT.md
git commit -qm "test: mutate contract for pin rejection"
git push -q origin wave:wave
expect_fail "mismatched trusted contract pin" env ORACLE_CONTRACT_PIN="$PIN" bash scripts/build_oracle_archive.sh "$BASE"

echo "ok: trusted contract pin rejects mismatched blob"

# Receipt adversarial cases: conflicting begin, invalid verdict, traversal, and
# reviewed identity mismatch must all fail closed.
RECEIPT="scripts/oracle_receipt.sh"
JOB="acceptance-job"
rm -rf .pi/oracle-receipts
bash "$RECEIPT" begin "$BASE" "$CAND" tree-accept "$JOB" context-accept archive-accept contract-accept >/dev/null
expect_fail "conflicting receipt begin" bash "$RECEIPT" begin "$BASE" "$CAND" tree-accept "$JOB" context-accept archive-accept different-contract
expect_fail "invalid receipt verdict" bash "$RECEIPT" complete "$JOB" "$TMP/response.md" https://example.test BANANA "$BASE" "$CAND" tree-accept
expect_fail "receipt path traversal" bash "$RECEIPT" begin "$BASE" "$CAND" tree-accept ../escape context-accept archive-accept contract-accept
expect_fail "receipt mismatched identity" bash "$RECEIPT" complete "$JOB" "$TMP/response.md" https://example.test APPROVE "$BASE" "$CAND" wrong-tree

echo "ok: receipt adversarial cases fail closed"
echo "ACCEPTANCE_PASS"
