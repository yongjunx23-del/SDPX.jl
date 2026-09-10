#!/usr/bin/env bash
# setup_identity.sh -- P0-00 baseline/candidate worktree setup (no mutations).
#
# Given a repo root and two full commit SHAs, this script:
#   1. verifies both commit objects exist in the repo,
#   2. saves the current worktree status (porcelain + HEAD),
#   3. creates a DETACHED baseline worktree and a candidate branch worktree
#      under an evidence parent dir,
#   4. records the julia version and the baseline..candidate diffstat.
#
# Safety contract (hard rules, enforced below):
#   - NEVER runs `git reset --hard`, `git clean -fd`, force push, or any
#     command that mutates the user's working tree.
#   - NEVER checks anything out inside the given repo root.
#   - Only additive worktree operations (`git worktree add --detach`) plus
#     read-only inspection (`status`, `rev-parse`, `diff --stat`, `show`).
#
# Env overrides (all optional; CLI args take precedence over env):
#   SDPX_REPO_ROOT, SDPX_BASELINE_SHA, SDPX_CANDIDATE_SHA,
#   EVIDENCE_PARENT (default: <repo>/.auto/evidence), RUN_ID
#   (default: UTC timestamp), JULIA_BIN (default: first `julia` on PATH).
#
# No absolute machine paths are baked in: every path recorded in the
# evidence files is either relative or derived at runtime.
set -u

usage() {
    echo "usage: setup_identity.sh [--repo-root PATH] [--baseline fullSHA] [--candidate fullSHA] [--evidence-parent PATH] [--run-id ID]" >&2
    echo "       env overrides: SDPX_REPO_ROOT SDPX_BASELINE_SHA SDPX_CANDIDATE_SHA EVIDENCE_PARENT RUN_ID JULIA_BIN" >&2
}

REPO_ROOT="${1:+}"
# Parse long options; positional fallback: repo baseline candidate [evidence].
BASELINE=""; CANDIDATE=""; EVIDENCE=""; RUN_ID_IN=""
ARGS=("$@")
i=0
while [ "$i" -lt "$#" ]; do
    case "${ARGS[$i]}" in
        --repo-root) REPO_ROOT="${ARGS[$((i+1))]:-}"; i=$((i+2)) ;;
        --baseline) BASELINE="${ARGS[$((i+1))]:-}"; i=$((i+2)) ;;
        --candidate) CANDIDATE="${ARGS[$((i+1))]:-}"; i=$((i+2)) ;;
        --evidence-parent) EVIDENCE="${ARGS[$((i+1))]:-}"; i=$((i+2)) ;;
        --run-id) RUN_ID_IN="${ARGS[$((i+1))]:-}"; i=$((i+2)) ;;
        -h|--help) usage; exit 0 ;;
        *) break ;;
    esac
done
# Positional fallback for the remaining tokens.
REST=("${ARGS[@]:$i}")
[ -z "$REPO_ROOT" ] && [ "${#REST[@]}" -ge 1 ] && REPO_ROOT="${REST[0]}"
[ -z "$BASELINE" ] && [ "${#REST[@]}" -ge 2 ] && BASELINE="${REST[1]}"
[ -z "$CANDIDATE" ] && [ "${#REST[@]}" -ge 3 ] && CANDIDATE="${REST[2]}"
[ -z "$EVIDENCE" ] && [ "${#REST[@]}" -ge 4 ] && EVIDENCE="${REST[3]}"

REPO_ROOT="${REPO_ROOT:-${SDPX_REPO_ROOT:-}}"
BASELINE="${BASELINE:-${SDPX_BASELINE_SHA:-}}"
CANDIDATE="${CANDIDATE:-${SDPX_CANDIDATE_SHA:-}}"
EVIDENCE="${EVIDENCE:-${EVIDENCE_PARENT:-}}"
RUN_ID="${RUN_ID_IN:-${RUN_ID:-}}"
JULIA_BIN="${JULIA_BIN:-julia}"

[ -z "$REPO_ROOT" ] && { echo "error: repo root required" >&2; usage; exit 2; }
[ -z "$BASELINE" ] && { echo "error: baseline full SHA required" >&2; usage; exit 2; }
[ -z "$CANDIDATE" ] && { echo "error: candidate full SHA required" >&2; usage; exit 2; }
case "$BASELINE" in *[!0-9a-f]*|????????????????????????????????????????) :;; *) echo "error: baseline must be a full 40-hex SHA" >&2; exit 2;; esac
[ "${#BASELINE}" -eq 40 ] || { echo "error: baseline must be a full 40-hex SHA" >&2; exit 2; }
[ "${#CANDIDATE}" -eq 40 ] || { echo "error: candidate must be a full 40-hex SHA" >&2; exit 2; }
case "$CANDIDATE" in *[!0-9a-f]*) echo "error: candidate must be a full 40-hex SHA" >&2; exit 2;; esac

[ -d "$REPO_ROOT/.git" ] || [ -f "$REPO_ROOT/.git" ] || { echo "error: not a git repo: $REPO_ROOT" >&2; exit 2; }

if [ -z "$RUN_ID" ]; then RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"; fi
if [ -z "$EVIDENCE" ]; then EVIDENCE="$REPO_ROOT/.auto/evidence"; fi
OUTDIR="$EVIDENCE/$RUN_ID/identity"
mkdir -p "$OUTDIR" || { echo "error: cannot create $OUTDIR" >&2; exit 1; }

git -C "$REPO_ROOT" cat-file -e "${BASELINE}^{commit}" 2>/dev/null \
    || { echo "error: baseline object $BASELINE not found in $REPO_ROOT" >&2; exit 1; }
git -C "$REPO_ROOT" cat-file -e "${CANDIDATE}^{commit}" 2>/dev/null \
    || { echo "error: candidate object $CANDIDATE not found in $REPO_ROOT" >&2; exit 1; }

# 1. Save original worktree status (read-only).
git -C "$REPO_ROOT" rev-parse HEAD > "$OUTDIR/orig_HEAD.txt"
git -C "$REPO_ROOT" status --porcelain=v1 > "$OUTDIR/orig_status.txt"
git -C "$REPO_ROOT" rev-parse HEAD^{tree} > "$OUTDIR/orig_tree.txt"

# 2. Additive worktrees only. Never touch the user's checkout.
BASE_WT="$EVIDENCE/$RUN_ID/worktree-baseline"
CAND_WT="$EVIDENCE/$RUN_ID/worktree-candidate"
[ -e "$BASE_WT" ] || git -C "$REPO_ROOT" worktree add --detach "$BASE_WT" "$BASELINE" > "$OUTDIR/baseline_add.log" 2>&1 \
    || { echo "error: baseline worktree add failed (see $OUTDIR/baseline_add.log)" >&2; exit 1; }
[ -e "$CAND_WT" ] || git -C "$REPO_ROOT" worktree add --detach "$CAND_WT" "$CANDIDATE" > "$OUTDIR/candidate_add.log" 2>&1 \
    || { echo "error: candidate worktree add failed (see $OUTDIR/candidate_add.log)" >&2; exit 1; }

# 3. Record versions and diffstat.
if command -v "$JULIA_BIN" >/dev/null 2>&1; then
    "$JULIA_BIN" --startup-file=no --version > "$OUTDIR/julia_version.txt" 2>&1
else
    echo "missing: JULIA_BIN=$JULIA_BIN not found" > "$OUTDIR/julia_version.txt"
fi
git -C "$REPO_ROOT" diff --stat "$BASELINE" "$CANDIDATE" > "$OUTDIR/diffstat.txt"
git -C "$REPO_ROOT" rev-list --count "$BASELINE..$CANDIDATE" > "$OUTDIR/commit_count.txt"
{
    echo "run_id=$RUN_ID"
    echo "repo_root=$REPO_ROOT"
    echo "baseline=$BASELINE"
    echo "candidate=$CANDIDATE"
    echo "baseline_worktree=$BASE_WT"
    echo "candidate_worktree=$CAND_WT"
} > "$OUTDIR/setup.env"

echo "identity setup complete: $OUTDIR"
echo "  baseline worktree:  $BASE_WT"
echo "  candidate worktree: $CAND_WT"
