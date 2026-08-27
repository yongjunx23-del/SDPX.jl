#!/usr/bin/env bash
# Durable oracle-job receipt writer. Gives ORACLE_PENDING an executable,
# atomic, idempotent, confined state and makes /oracle-read idempotent w.r.t.
# identity. One job id has one immutable identity; stale, mismatched, malformed
# or replayed responses never become COMPLETE or pass G5.
#
# Usage:
#   scripts/oracle_receipt.sh begin <base-sha> <candidate-sha> <candidate-tree> <job-id> <context-sha256> <archive-sha256> <contract-blob>
#   scripts/oracle_receipt.sh complete <job-id> <response-path> <conversation-url> <verdict> <reviewed-base> <reviewed-candidate> <reviewed-tree>
#   scripts/oracle_receipt.sh get <job-id>

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECEIPT_DIR="$ROOT/.pi/oracle-receipts"
mkdir -p "$RECEIPT_DIR"

# Job ids must be safe path components (no separators, no traversal).
safe_job() {
  case "$1" in
    */*|*\\*|*..*|*[!A-Za-z0-9_-]*) return 1 ;;
    *) return 0 ;;
  esac
}

VERDICTS="APPROVE APPROVE_WITH_CHANGES NEEDS_REWORK"

_cmd="${1:-}"
case "$_cmd" in
  begin)
    [ "$#" -eq 8 ] || { echo "begin needs 7 args: base candidate tree job context-sha archive-sha contract-blob" >&2; exit 2; }
    base="$2"; cand="$3"; ctree="$4"; job="$5"; ctxsha="$6"; arcsha="$7"; cblob="$8"
    safe_job "$job" || { echo "unsafe job id: $job" >&2; exit 2; }
    python3 - "$RECEIPT_DIR" "$job" "$base" "$cand" "$ctree" "$ctxsha" "$arcsha" "$cblob" <<'PY'
import json, sys, os, datetime
d, job, base, cand, ctree, ctx, arc, cblob = sys.argv[1:]
out = os.path.join(d, job + ".json")
rec = {
  "state": "PENDING",
  "base_sha": base,
  "candidate_sha": cand,
  "candidate_tree": ctree,
  "oracle_job_id": job,
  "context_sha256": ctx,
  "archive_sha256": arc,
  "contract_blob": cblob,
  "submitted_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
# Exclusive-create: if a receipt already exists, only accept an identical
# identity (idempotent re-begin); reject a conflicting one.
if os.path.exists(out):
    with open(out) as f:
        existing = json.load(f)
    same = all(existing.get(k) == rec[k] for k in
               ("base_sha","candidate_sha","candidate_tree","context_sha256","archive_sha256","contract_blob"))
    if not same:
        sys.exit("conflicting receipt already exists for job " + job)
    print("PENDING receipt already exists (idempotent):", out)
    sys.exit(0)
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(rec, f, indent=2, sort_keys=True)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, out)
os.chmod(out, 0o600)
print("PENDING receipt:", out)
PY
    ;;
  complete)
    [ "$#" -eq 8 ] || { echo "complete needs 7 args: job response-path conv-url verdict reviewed-base reviewed-candidate reviewed-tree" >&2; exit 2; }
    job="$2"; rpath="$3"; conv="$4"; verdict="$5"; rbase="$6"; rcand="$7"; rtree="$8"
    safe_job "$job" || { echo "unsafe job id: $job" >&2; exit 2; }
    case " $VERDICTS " in *" $verdict "*) ;; *) echo "invalid verdict: $verdict" >&2; exit 2 ;; esac
    python3 - "$RECEIPT_DIR" "$job" "$rpath" "$conv" "$verdict" "$rbase" "$rcand" "$rtree" <<'PY'
import json, sys, os, datetime
d, job, rpath, conv, verdict, rbase, rcand, rtree = sys.argv[1:]
out = os.path.join(d, job + ".json")
if not os.path.exists(out):
    sys.exit("no PENDING receipt for job " + job)
with open(out) as f:
    rec = json.load(f)
if rec.get("state") != "PENDING":
    sys.exit("receipt for job " + job + " is not PENDING (state=" + str(rec.get("state")) + ")")
# Identity validation: the reviewed base/candidate/tree must match the receipt.
if rec.get("base_sha") != rbase or rec.get("candidate_sha") != rcand or rec.get("candidate_tree") != rtree:
    sys.exit("reviewed identity does not match receipt for job " + job)
rec["state"] = "COMPLETE"
rec["response_path"] = rpath
rec["conversation_url"] = conv
rec["verdict"] = verdict
rec["reviewed_base_sha"] = rbase
rec["reviewed_candidate_sha"] = rcand
rec["reviewed_candidate_tree"] = rtree
rec["completed_at"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(rec, f, indent=2, sort_keys=True)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, out)
os.chmod(out, 0o600)
print("COMPLETE receipt updated:", out)
PY
    ;;
  get)
    [ "$#" -eq 2 ] || { echo "get needs <job>" >&2; exit 2; }
    safe_job "$2" || { echo "unsafe job id: $2" >&2; exit 2; }
    cat "$RECEIPT_DIR/$2.json"
    ;;
  *)
    echo "usage: $0 {begin|complete|get} ..." >&2; exit 2
    ;;
esac
