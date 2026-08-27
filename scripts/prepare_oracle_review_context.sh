#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SHA="${1:-}"
OUTPUT="${2:-$ROOT/.pi/oracle-review-context.md}"
MAX_DIFF_BYTES="${ORACLE_DIFF_MAX_BYTES:-20971520}"

if [ -z "$BASE_SHA" ]; then
  echo "usage: $0 <base-sha> [output-path]" >&2
  exit 2
fi

cd "$ROOT"
git cat-file -e "${BASE_SHA}^{commit}"
CANDIDATE_SHA="$(git rev-parse HEAD)"
CANDIDATE_TREE="$(git rev-parse HEAD^{tree})"
BASE_TREE="$(git rev-parse "${BASE_SHA}^{tree}")"

if ! git merge-base --is-ancestor "$BASE_SHA" "$CANDIDATE_SHA"; then
  echo "base SHA is not an ancestor of candidate: $BASE_SHA -> $CANDIDATE_SHA" >&2
  exit 1
fi

STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [ -n "$STATUS" ]; then
  echo "candidate worktree is not clean; freeze and commit it before oracle review:" >&2
  printf '%s\n' "$STATUS" >&2
  exit 1
fi

git diff --check "${BASE_SHA}...${CANDIDATE_SHA}"

TMP_DIFF="$(mktemp "${TMPDIR:-/tmp}/sdpx-oracle-diff.XXXXXX")"
cleanup() {
  rm -f "$TMP_DIFF"
}
trap cleanup EXIT

git diff --no-ext-diff --find-renames --submodule=log "${BASE_SHA}...${CANDIDATE_SHA}" > "$TMP_DIFF"
DIFF_BYTES="$(wc -c < "$TMP_DIFF" | tr -d ' ')"
if [ "$DIFF_BYTES" -gt "$MAX_DIFF_BYTES" ]; then
  echo "review diff is ${DIFF_BYTES} bytes, above ORACLE_DIFF_MAX_BYTES=${MAX_DIFF_BYTES}; split the candidate into atomic PRs" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
{
  echo "# SDPX Oracle Review Context"
  echo
  echo "Generated at (UTC): $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo
  echo "- Base SHA: \`$BASE_SHA\`"
  echo "- Base tree: \`$BASE_TREE\`"
  echo "- Candidate SHA: \`$CANDIDATE_SHA\`"
  echo "- Candidate tree: \`$CANDIDATE_TREE\`"
  echo "- Branch: \`$(git branch --show-current)\`"
  echo "- Worktree clean before context generation: yes"
  echo "- Text diff bytes: \`$DIFF_BYTES\`"
  echo
  echo "## Commits"
  echo '```text'
  git log --format='%H %s' "${BASE_SHA}..${CANDIDATE_SHA}"
  echo '```'
  echo
  echo "## Diff stat"
  echo '```text'
  git diff --stat "${BASE_SHA}...${CANDIDATE_SHA}"
  echo '```'
  echo
  echo "## Candidate diff"
  echo '```diff'
  cat "$TMP_DIFF"
  echo '```'
} > "$OUTPUT"
chmod 600 "$OUTPUT"

if command -v shasum >/dev/null 2>&1; then
  CONTEXT_SHA="$(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
else
  CONTEXT_SHA="$(sha256sum "$OUTPUT" | awk '{print $1}')"
fi

printf 'oracle review context: %s\n' "$OUTPUT"
printf 'candidate SHA: %s\n' "$CANDIDATE_SHA"
printf 'candidate tree: %s\n' "$CANDIDATE_TREE"
printf 'context SHA-256: %s\n' "$CONTEXT_SHA"
