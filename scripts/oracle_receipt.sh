#!/usr/bin/env bash
# Durable oracle-job receipt writer. Gives ORACLE_PENDING an executable,
# atomic, idempotent state and makes /oracle-read idempotent w.r.t. identity.
#
# Usage:
#   scripts/oracle_receipt.sh begin <base-sha> <candidate-sha> <job-id> <context-sha256> <archive-sha256> <contract-blob>
#   scripts/oracle_receipt.sh complete <job-id> <response-path> <conversation-url> <verdict>
#   scripts/oracle_receipt.sh get <job-id>
#
# Receipts live under .pi/oracle-receipts/<job-id>.json (gitignored). A pending
# receipt is written atomically before entering ORACLE_PENDING so a lost wake-up,
# lead restart or duplicate submission is detected deterministically.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECEIPT_DIR="$ROOT/.pi/oracle-receipts"
mkdir -p "$RECEIPT_DIR"

_cmd="${1:-}"
case "$_cmd" in
  begin)
    [ "$#" -eq 7 ] || { echo "begin needs 6 args: base candidate job context-sha archive-sha contract-blob" >&2; exit 2; }
    base="$2"; cand="$3"; job="$4"; ctxsha="$5"; arcsha="$6"; cblob="$7"
    python3 - "$RECEIPT_DIR" "$job" "$base" "$cand" "$ctxsha" "$arcsha" "$cblob" <<'PY'
import json, sys, os
d, job, base, cand, ctx, arc, cblob = sys.argv[1:]
rec = {
  "state": "PENDING",
  "base_sha": base,
  "candidate_sha": cand,
  "oracle_job_id": job,
  "context_sha256": ctx,
  "archive_sha256": arc,
  "contract_blob": cblob,
  "submitted_at": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
}
out = os.path.join(d, job + ".json")
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(rec, f, indent=2, sort_keys=True)
os.replace(tmp, out)
os.chmod(out, 0o600)
print("PENDING receipt:", out)
PY
    ;;
  complete)
    [ "$#" -eq 5 ] || { echo "complete needs <job> <response-path> <conversation-url> <verdict>" >&2; exit 2; }
    job="$2"; rpath="$3"; conv="$4"; verdict="$5"
    python3 - "$RECEIPT_DIR" "$job" "$rpath" "$conv" "$verdict" <<'PY'
import json, sys, os
d, job, rpath, conv, verdict = sys.argv[1:]
out = os.path.join(d, job + ".json")
with open(out) as f:
    rec = json.load(f)
rec["state"] = "COMPLETE"
rec["response_path"] = rpath
rec["conversation_url"] = conv
rec["verdict"] = verdict
rec["completed_at"] = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(rec, f, indent=2, sort_keys=True)
os.replace(tmp, out)
os.chmod(out, 0o600)
print("COMPLETE receipt updated:", out)
PY
    ;;
  get)
    [ "$#" -eq 2 ] || { echo "get needs <job>" >&2; exit 2; }
    cat "$RECEIPT_DIR/$2.json"
    ;;
  *)
    echo "usage: $0 {begin|complete|get} ..." >&2; exit 2
    ;;
esac
