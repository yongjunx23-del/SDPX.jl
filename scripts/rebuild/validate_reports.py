#!/usr/bin/env python3
"""Validate task reports against the packet template and the evidence on disk.

    validate_reports.py [WORKSPACE] [--only ID ...]

WHY THIS EXISTS. A worker report in this rebuild once claimed "6/6" while its own
log read 192/192 and the file was red at line 195. The claim was caught by a human
re-running the driver, which does not scale to the twelve reports still to come and
depends on someone remembering to be suspicious. This script mechanically checks the
things that are checkable without re-running anything:

  * the report's key set equals the template's, so a missing section is an error
    rather than a silent omission;
  * every enum value is from the documented vocabulary;
  * every artifact path a report cites EXISTS on disk — a report that points at a
    log which was never written is the cheapest possible false claim;
  * a command with a null exit code may not claim `pass`, and a command with
    exit 0 may not claim `fail`;
  * a `not_run` item may not carry a measured-looking number in its note, and a
    `pass` item may not describe itself as not run;
  * `performance.measured == true` requires raw samples that exist.

WHAT IT CANNOT DO, stated so nobody reads more into a green run than is there: it
does not check that a command was actually run, that a number is correct, or that a
test is the right test. Those need the driver re-run. This is a filter for the
cheap lies and the honest omissions, not a substitute for measurement.
"""

import argparse
import json
import os
import re
import sys

TEMPLATE = os.path.join(
    os.path.expanduser("~"),
    "Downloads", "sdpx_infrastructure_review", "templates", "agent_report.json",
)

STATUS_VOCAB = {"not_started", "in_progress", "blocked", "needs_review", "accepted"}
TEST_VOCAB = {"pass", "fail", "not_run", "unsupported"}
ACCEPT_VOCAB = {"verified", "partially_verified", "not_verified"}
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
NUMBER_RE = re.compile(r"\d")
NOT_RUN_WORDS = re.compile(r"\bnot_run\b|\bnot run\b|\bunrun\b|\bdid not run\b", re.I)
# A non-zero exit can BE the passing result in this packet: proving that a file is
# inert means grepping for an include and finding none (`grep` exits 1), and proving
# that a guard is live means it exits non-zero. The report has to say so, though —
# an unexplained non-zero exit with `result: pass` is still an error.
EXPECTED_NONZERO = re.compile(
    r"\bnot a failure\b|\bis the (?:RESULT|result|expected)\b|expected\b|"
    r"\bis the point\b|\bby design\b|\bexit(?:s|ed)? non-?zero\b|"
    # Proving ABSENCE by grep is the other standard pattern here: the command exits
    # 1 because it found nothing, and finding nothing is the claim. D01's
    # "grep -c Quadratic ... -> 0 occurrences" is exactly this.
    r"\b0 occurrences\b|\bzero occurrences\b|\bno matches?\b|\bnot found\b|"
    r"\babsent\b|\bexits? 1\b|\breturned 1\b",
    re.I,
)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def check_report(report, template, workspace, report_path):
    """Return a list of (severity, message)."""
    problems = []
    add = lambda sev, msg: problems.append((sev, msg))

    missing = sorted(set(template) - set(report))
    extra = sorted(set(report) - set(template))
    if missing:
        add("ERROR", f"missing template keys: {missing}")
    if extra:
        add("WARN", f"keys not in template: {extra}")

    if report.get("status") not in STATUS_VOCAB:
        add("ERROR", f"status {report.get('status')!r} not in {sorted(STATUS_VOCAB)}")

    for key in ("base_commits", "candidate_commits"):
        for repo, sha in (report.get(key) or {}).items():
            if sha is None:
                add("WARN", f"{key}.{repo} is null")
            elif not SHA_RE.match(str(sha)):
                add("ERROR", f"{key}.{repo}={sha!r} is not a SHA")

    def artifact(rel, where):
        """Check an artifact field.

        The template intends these to be PATHS. Several workers instead used them
        as free-text notes, which made the first version of this check emit one
        "path does not exist" error per line of prose and bury the real findings.
        So: take the leading token, check it if it looks like a path, and report the
        prose separately.

        A missing path under /tmp is downgraded to a warning. /tmp is ephemeral by
        construction, and this project has already lost a validated patch and a
        driver it left there — that is a known, recorded limitation, not a new
        report defect. A missing path anywhere else is an error.
        """
        if rel in (None, "", "null"):
            return
        text = str(rel).strip()
        token = text.split()[0] if text.split() else ""
        # `origin/main=4b46cda;` contains a slash and is not a path. A shell
        # assignment or a `k=v;` fragment is prose that happens to contain one.
        looks_like_path = (
            bool(token)
            and "/" in token
            and not any(ch in token for ch in "<>\"'")
            and not any(ch in token for ch in "=;,")
        )
        if not looks_like_path:
            add("WARN", f"{where} is not a path; it reads as prose: {text[:70]!r}")
            return
        if len(text.split()) > 1:
            add("WARN", f"{where} has prose after the path: {text[:70]!r}")
        # Reports in this packet record some artifacts workspace-relative and some
        # repo-relative (Q01's `benchmark/rebuild/measure_result.toml` lives under
        # SDPX.jl/). Resolving only against the workspace root produced a false
        # "does not exist" for a file that is present. Try every repo root before
        # calling it missing.
        roots = [workspace] + [
            os.path.join(workspace, r) for r in
            ("SDPX.jl", "MultiFloatLinearAlgebra.jl", "BigFloatLinearAlgebra.jl")
        ]
        candidates = [token] if os.path.isabs(token) else [os.path.join(r, token) for r in roots]
        if any(os.path.exists(c) for c in candidates):
            return
        if token.startswith("/tmp/") or token.startswith("/private/var/folders/"):
            add("WARN", f"{where} pointed at ephemeral storage that is gone: {token}")
        else:
            add("ERROR", f"{where} cites a path that does not exist under any repo root: {token}")

    for i, cmd in enumerate(report.get("commands") or []):
        where = f"commands[{i}]"
        if not cmd.get("command"):
            add("WARN", f"{where} has no command text")
        rc, result = cmd.get("exit_code"), cmd.get("result")
        note = str(cmd.get("note") or cmd.get("stdout_artifact") or "")
        if result not in TEST_VOCAB:
            add("ERROR", f"{where}.result={result!r} not in {sorted(TEST_VOCAB)}")
        if rc is None and result == "pass":
            add("ERROR", f"{where} claims pass with a null exit_code")
        if rc == 0 and result == "fail":
            add("ERROR", f"{where} has exit_code 0 but result 'fail'")
        if rc not in (None, 0) and result == "pass":
            if EXPECTED_NONZERO.search(note):
                add("WARN", f"{where} claims pass with exit_code {rc} and justifies it")
            else:
                add("ERROR", f"{where} claims pass with exit_code {rc} and does not explain why")
        artifact(cmd.get("stdout_artifact"), f"{where}.stdout_artifact")
        artifact(cmd.get("stderr_artifact"), f"{where}.stderr_artifact")

    for i, test in enumerate(report.get("numeric_tests") or []):
        where = f"numeric_tests[{i}]"
        status = test.get("status")
        if status not in TEST_VOCAB:
            add("ERROR", f"{where}.status={status!r} not in {sorted(TEST_VOCAB)}")
        note = test.get("note") or ""
        if status == "not_run" and NUMBER_RE.search(note) and not NOT_RUN_WORDS.search(note):
            add("WARN", f"{where} is not_run but its note states a number without saying so: {note[:70]!r}")
        if status == "pass" and NOT_RUN_WORDS.search(note):
            # Not an error. Q01's entries legitimately pass while DISCLOSING that a
            # sub-item was not run ("native_allocator_bytes (not_run with reason:
            # ...)", "prepared-update replay ... stays not_run"). Punishing that
            # would punish exactly the honesty the packet asks for. But the two
            # readings are indistinguishable to a script, so this asks a human to
            # confirm the ENTRY ran rather than asserting that it did not.
            add("WARN", f"{where} passes while mentioning a not_run item; "
                        f"confirm the entry itself ran: {note[:70]!r}")

    for i, acc in enumerate(report.get("acceptance") or []):
        where = f"acceptance[{i}]"
        if acc.get("state") not in ACCEPT_VOCAB:
            add("ERROR", f"{where}.state={acc.get('state')!r} not in {sorted(ACCEPT_VOCAB)}")
        if not (acc.get("evidence") or "").strip():
            add("ERROR", f"{where} has no evidence")
        if acc.get("state") != "verified" and not (acc.get("criterion") or "").strip():
            add("WARN", f"{where} has no criterion text")

    perf = report.get("performance") or {}
    if perf.get("measured") is True:
        for field in ("raw_samples_artifact", "arithmetic", "machine",
                      "threads_requested", "threads_executed"):
            if perf.get(field) in (None, "", "null"):
                add("ERROR", f"performance.measured is true but {field} is null")
        artifact(perf.get("raw_samples_artifact"), "performance.raw_samples_artifact")
        if perf.get("failures_included") is None:
            add("WARN", "performance.failures_included is null")

    if not (report.get("limitations") or []):
        add("WARN", "limitations is empty; every report in this packet has had some")

    n_open = len(report.get("open_findings") or [])
    n_partial = sum(1 for a in (report.get("acceptance") or [])
                    if a.get("state") != "verified")
    if n_partial and not n_open:
        add("WARN", f"{n_partial} acceptance item(s) are not verified but open_findings is empty")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace", nargs="?", default=os.getcwd())
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)

    template = load(TEMPLATE)
    reports_root = os.path.join(args.workspace, "rebuild-reports")
    found = sorted(
        d for d in os.listdir(reports_root)
        if os.path.isfile(os.path.join(reports_root, d, "report.json"))
    )
    if args.only:
        found = [d for d in found if d in set(args.only)]
    if not found:
        print("no reports found")
        return 1

    total = {"ERROR": 0, "WARN": 0}
    for task in found:
        path = os.path.join(reports_root, task, "report.json")
        try:
            report = load(path)
        except Exception as exc:  # noqa: BLE001 - report the parse failure, do not hide it
            print(f"{task:6s} PARSE ERROR: {exc}")
            total["ERROR"] += 1
            continue
        problems = check_report(report, template, args.workspace, path)
        errors = [m for s, m in problems if s == "ERROR"]
        warns = [m for s, m in problems if s == "WARN"]
        total["ERROR"] += len(errors)
        total["WARN"] += len(warns)
        mark = "ok  " if not errors else "FAIL"
        print(f"{task:6s} {mark} errors={len(errors)} warnings={len(warns)}")
        for msg in errors:
            print(f"         ERROR {msg}")
        for msg in warns:
            print(f"         warn  {msg}")

    print()
    print(f"reports={len(found)} errors={total['ERROR']} warnings={total['WARN']}")
    return 1 if total["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
