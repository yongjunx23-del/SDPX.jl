#!/usr/bin/env bash
# Build a faithful, closed review archive from the frozen candidate tree.
#
# pi-oracle packages the filesystem, which is not a closed git snapshot: it can
# include AppleDouble metadata, .git, ignored files, and files outside the tree,
# and it can omit generated .pi/ identity files. This builder instead stages the
# candidate tree from `git archive`, adds the generated review identity files,
# emits a full mode/type/digest manifest, and scans the staged tree before
# submission. Dispatch pi-oracle with files=[".pi/oracle-archive"].
#
# Usage:
#   scripts/build_oracle_archive.sh <base-sha>
#
# Environment:
#   ORACLE_ARCHIVE_DIR   staging dir (default .pi/oracle-archive)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IN="${1:-}"
: "${ORACLE_ARCHIVE_DIR:=$ROOT/.pi/oracle-archive}"
[ -n "$BASE_IN" ] || { echo "usage: $0 <base-sha>" >&2; exit 2; }

cd "$ROOT"

# Reuse the context generator: it enforces clean tree, pushed candidate,
# ancestor base, full-SHA canonicalization, leak rejection, and emits the
# context + sidecar + tree manifest.
bash "$ROOT/scripts/prepare_oracle_review_context.sh" "$BASE_IN" >/dev/null

CANDIDATE_SHA="$(git rev-parse HEAD)"
CANDIDATE_TREE="$(git rev-parse HEAD^{tree})"

# Disable macOS AppleDouble metadata generation for every copy/tar below.
export COPYFILE_DISABLE=1

rm -rf "$ORACLE_ARCHIVE_DIR"
mkdir -p "$ORACLE_ARCHIVE_DIR/.pi"

# 1. Faithful candidate tree from git archive (no .git, no AppleDouble, no
#    ignored files, no external files).
git archive "$CANDIDATE_SHA" | tar -x -C "$ORACLE_ARCHIVE_DIR"

# 2. Add the generated review identity files (context, sidecar, tree manifest).
cp "$ROOT/.pi/oracle-review-context.md" "$ORACLE_ARCHIVE_DIR/.pi/"
cp "$ROOT/.pi/oracle-review-context.sha256" "$ORACLE_ARCHIVE_DIR/.pi/"
cp "$ROOT/.pi/oracle-tree-manifest" "$ORACLE_ARCHIVE_DIR/.pi/"

# 3. Full manifest: mode/type/digest for every archived file, tree-relative.
MANIFEST="$ROOT/.pi/oracle-archive-manifest"
( cd "$ORACLE_ARCHIVE_DIR" && find . -type f | sort | while read -r f; do
    mode="$(stat -f '%Lp' "$f")"
    sha="$(shasum -a 256 "$f" | awk '{print $1}')"
    printf '%s %s %s\n' "$mode" "$sha" "${f#./}"
  done ) > "$MANIFEST"
cp "$MANIFEST" "$ORACLE_ARCHIVE_DIR/.pi/oracle-archive-manifest"

# 4. Post-build scan: reject AppleDouble and any non-tree stray file.
LEAKS="$(find "$ORACLE_ARCHIVE_DIR" -name '._*' -o -name 'Manifest.toml' 2>/dev/null)"
[ -z "$LEAKS" ] || { echo "staged archive leaks non-tree files:" >&2; printf '%s\n' "$LEAKS" >&2; exit 1; }

MANIFEST_SHA="$(shasum -a 256 "$MANIFEST" | awk '{print $1}')"
printf 'oracle archive stage: %s\n' "$ORACLE_ARCHIVE_DIR"
printf 'candidate SHA: %s\n' "$CANDIDATE_SHA"
printf 'candidate tree: %s\n' "$CANDIDATE_TREE"
printf 'archive manifest: %s (%s)\n' "$MANIFEST" "$MANIFEST_SHA"
printf 'dispatch pi-oracle with files=[".pi/oracle-archive"]\n'
