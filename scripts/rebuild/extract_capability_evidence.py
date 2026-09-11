#!/usr/bin/env python3
"""Extract PER-CAPABILITY evidence from the packet (reports + logs).

    python3 scripts/rebuild/extract_capability_evidence.py [WORKSPACE] [--out PATH]

WHY A SEPARATE EXTRACTOR FROM `gen_support_matrix.py`.
`gen_support_matrix.py` renders a *curated* mapping (capability -> the artifacts
that exercise it) whose curation is the reviewable part. This script is the part
that must not be curated: it reads every `rebuild-reports/<ID>/report.json` and
every log those reports cite or leave beside them, and emits the raw per-artifact
verdicts in the report's OWN vocabulary (`pass` / `fail` / `not_run` /
`unsupported`; `verified` / `partially_verified` / `not_verified`). Nothing is
summarised away.

FOUR RULES, because each of them is a way this table could lie:

1. A missing measurement is `not_run`, NEVER `pass` and NEVER `0`. An artifact
   that could not be found is emitted with `log_verdict: "not_run"` and a reason
   that distinguishes "the report cited no artifact" from "the cited path is
   gone". Those are different facts and are never merged.
2. `unsupported` and `not_run` are DIFFERENT facts and are never merged: the
   first means the platform or dependency cannot provide the capability, the
   second means nobody ran it.
3. A log whose `Test Summary` rows carry a non-zero `Fail` or `Error` column is
   `fail` for that artifact. When the report's prose says `pass` anyway, the case
   is recorded in `disagreements` rather than silently resolved — the log is the
   primary source only when it is present, and it is present for a minority of
   commands (see the `logs` block in the output for exactly how many).
4. `Test Summary` tables are the machine-readable part; a log with no such table
   yields `not_run`, not `pass`. An exit code of 0 is not a test verdict.

WHAT THIS CANNOT DO, stated so a green run is not read as more than it is: it
does not re-run any driver, so it cannot tell whether the log it reads was
produced at the revision the report names. That is `run_matrix.jl`'s job.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

COLNUM_RE = re.compile(r"(Pass|Fail|Error|Broken|Total)\s*=\s*(\d+)")
# Absolute paths inside free-text artifact fields. Reports in this packet embed
# one or more paths in prose ("/tmp/a.log (run 2) and /tmp/b.log (run 3)"), so
# token-splitting loses the second one; a path regex does not.
ABSPATH_RE = re.compile(r"(/[^\s,;()\[\]'\"`]+)")


def candidate_paths(report, cmd, workspace, task):
    """Every path worth trying for a command's log, and where each came from.

    Two sources, kept distinct in the output because they carry different weight:
      * `cited`     -- a path the report itself names in stdout_artifact /
                       stderr_artifact. A citation that does not resolve is a
                       report defect, recorded as such.
      * `discovered`-- a `.log` sitting in the task's own log directory. The
                       packet writes many logs to `/tmp/<task>/`, which is wiped
                       between sessions, so discovery is what makes this usable
                       after the fact. A discovered log is evidence about the
                       package, not about the report's citation, and the two are
                       never conflated.
    """
    seen, out = set(), []

    def add(path, source):
        path = os.path.normpath(path)
        if path in seen:
            return
        seen.add(path)
        out.append((path, source))

    for field in ("stdout_artifact", "stderr_artifact"):
        raw = cmd.get(field)
        if not raw:
            continue
        for token in ABSPATH_RE.findall(str(raw)):
            add(token, "cited")
        head = str(raw).strip().split()[0] if str(raw).strip().split() else ""
        if head and "/" in head and not head.startswith("/"):
            roots = [workspace] + [
                os.path.join(workspace, r) for r in
                ("SDPX.jl", "MultiFloatLinearAlgebra.jl", "BigFloatLinearAlgebra.jl")
            ]
            for root in roots:
                add(os.path.join(root, head), "cited")

    for sub in ("logs", ""):
        directory = os.path.join(workspace, "rebuild-reports", task, sub)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith(".log"):
                add(os.path.join(directory, name), "discovered")
    # A few tasks keep their logs under /tmp/<task>/; try that too, since it is
    # the packet's other convention and /tmp survives within a session.
    for candidate in (f"/tmp/{task.lower()}", f"/tmp/{task}"):
        if os.path.isdir(candidate):
            for name in sorted(os.listdir(candidate)):
                if name.endswith(".log"):
                    add(os.path.join(candidate, name), "discovered")
    return out


def summaries(text):
    """Parse every `Test Summary` table row out of a log.

    Two shapes occur in this packet's logs and both are handled:
        <name>  |  <Pass>  <Fail>  <Error>  <Broken>     (column layout)
        <name>  |  Pass=.. Total=..                      (D01/A01 style)
    A summary-looking line that yields no counts is still returned, with
    `counts: {}`, so a table cannot shrink silently.
    """
    rows, pending = [], False
    for line in text.splitlines():
        if line.strip().startswith("Test Summary"):
            pending = True
            continue
        if not pending:
            continue
        if not line.strip():
            if rows:
                pending = False
            continue
        if "|" not in line:
            pending = False
            continue
        name, _, rest = line.partition("|")
        cols = {k: int(v) for k, v in COLNUM_RE.findall(rest)}
        if not cols:
            nums = re.findall(r"\d+", rest)
            if len(nums) >= 2:
                cols = {"Pass": int(nums[0]), "Total": int(nums[1])}
        rows.append({"name": name.strip(), "counts": cols})
    return rows


def verdict(rows):
    """pass / fail / not_run for a set of summary rows (used for aggregation)."""
    if not rows:
        return "not_run"
    failed = sum(r["counts"].get("Fail", 0) + r["counts"].get("Error", 0) for r in rows)
    return "fail" if failed else "pass"


# `Pkg.test()` prints this INSTEAD of a `Test Summary` for the suite as a whole
# when it fails, so a row-counting parser alone reports such a run as clean. The
# five SDPX runs in this packet that ended this way all carry the line.
TESTS_DID_NOT_PASS = re.compile(r"tests? did not pass", re.I)
# SDPX's own gate: `test/runtests.jl` refuses to run while `git status
# --porcelain` is non-empty, so the suite aborts before the remaining suites
# execute. That is an INFRASTRUCTURE event, and the packet's rule is explicit --
# a dirty tree is `not_run` with a reason, never a numeric `fail`.
WORKTREE_DIRTY_ABORT = re.compile(
    r"worktree became dirty|source worktree is dirty|became dirty at", re.I)
# A report that says in its own words that the non-zero exit IS the result (a
# fail-closed guard firing, a planted mutation being caught) is not in conflict
# with its log, and Q02 must not silently turn an expected failure into a defect.
EXPECTED_FAILURE = re.compile(
    r"EXPECTED FAILURE|expected to fail|is the (?:expected )?result|by design|"
    r"fail-closed|planted|mutation|deliberate", re.I)


def classify_log(text, rows):
    """Return (verdict, reason) for ONE log.

    The order matters and each branch is a distinct fact:
      1. the worktree-cleanliness abort  -> `not_run` (infrastructure)
      2. `Some tests did not pass`       -> `fail`
      3. a Fail/Error column > 0         -> `fail`
      4. a Test Summary table, all clean -> `pass`
      5. no Test Summary at all          -> `not_run` (not test evidence)
    """
    if WORKTREE_DIRTY_ABORT.search(text):
        return "not_run", "worktree_cleanliness_abort"
    if TESTS_DID_NOT_PASS.search(text):
        return "fail", "pkgtest_reported_tests_did_not_pass"
    if rows:
        failed = sum(r["counts"].get("Fail", 0) + r["counts"].get("Error", 0) for r in rows)
        if failed:
            return "fail", f"test_summary_rows_with_{failed}_fail_or_error"
        return "pass", "all_test_summary_rows_clean"
    return "not_run", "no_test_summary_table"


def read_logs(report, cmd, workspace, task):
    """Return (rows, files, cited_missing, reason) with a PER-FILE verdict.

    The per-file verdict is what matters and aggregating over sibling logs was a
    real defect in the first version of this script: S04's dirty-tree `Pkg.test()`
    failure lives in `/tmp/s04/pkgtest.log` while `/tmp/s04/S04_after_fix.log`
    (a different command, a passing run) sat in the same directory. An OR over
    the directory reported `pass` for the failing command. Each file therefore
    carries its own `verdict`, and the command-level verdict is the WORST of the
    files it cited -- `fail` beats `pass` beats `not_run` -- which is the only
    direction that cannot turn a failure into a pass.
    """
    rows, files, cited_missing = [], [], []
    cited_names = set()
    raw_fields = [str(cmd.get(f) or "") for f in ("stdout_artifact", "stderr_artifact")]
    for raw in raw_fields:
        for token in ABSPATH_RE.findall(raw):
            cited_names.add(os.path.basename(token).strip())
        head = raw.strip().split()[0] if raw.strip().split() else ""
        if head and "/" in head:
            cited_names.add(os.path.basename(head).strip())

    for path, source in candidate_paths(report, cmd, workspace, task):
        if source == "cited" and not os.path.isfile(path):
            cited_missing.append(path)
            continue
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError as exc:
            files.append({"path": path, "source": source, "read": False, "verdict": "not_run",
                          "reason": f"{type(exc).__name__}: {exc}"})
            continue
        found = summaries(text)
        file_verdict, file_reason = classify_log(text, found)
        files.append({
            "path": path,
            "source": source,
            "read": True,
            "test_summary_rows": len(found),
            "verdict": file_verdict,
            "verdict_reason": file_reason,
            "expected_failure": bool(EXPECTED_FAILURE.search(str(cmd.get("note") or ""))),
            # `evidence_kind` keeps a log that IS test evidence apart from one
            # that merely exists. A precompile-gate or surface-diff log contains
            # no `Test Summary` table, so the absence of failures in it is not a
            # test result and must not be counted as one.
            "evidence_kind": "test_summary" if found else "no_test_summary",
            # `primary` marks a file whose name the command itself cited. It is
            # the file most likely to be THIS command's log; a sibling log in the
            # same directory is corroboration, not the command's own artifact.
            "primary": os.path.basename(path) in cited_names,
        })
        rows.extend(found)

    if not files:
        reason = ("no readable log: the report cites no artifact and none was found "
                  "beside it" if not cited_missing else
                  "every cited path is gone: " + ", ".join(cited_missing[:3]))
        return rows, files, cited_missing, reason
    return rows, files, cited_missing, None


def worst_verdict(files):
    """Command-level verdict = worst per-file verdict. fail > pass > not_run.

    Only files carrying a `Test Summary` table participate; a log with no test
    table is `not_run` as evidence and cannot make a command look green.
    """
    verdicts = {f.get("verdict") for f in files
                if f.get("read") and f.get("evidence_kind") == "test_summary"}
    if not verdicts:
        return "not_run"
    if "fail" in verdicts:
        return "fail"
    if "pass" in verdicts:
        return "pass"
    return "not_run"


def extract(workspace):
    reports_root = os.path.join(workspace, "rebuild-reports")
    out = {"workspace": workspace, "reports": [], "disagreements": [],
           "not_run_infrastructure": [], "expected_failures": [], "unreadable": [],
           "logs": {"commands": 0, "with_readable_log": 0, "with_summary_rows": 0,
                    "cited_path_missing": 0}}
    if not os.path.isdir(reports_root):
        out["unreadable"].append({"artifact": reports_root, "reason": "no rebuild-reports/"})
        return out
    for task in sorted(os.listdir(reports_root)):
        path = os.path.join(reports_root, task, "report.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - record, never hide
            out["unreadable"].append({"artifact": path, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        entry = {
            "task": task,
            "status": report.get("status"),
            "base_commits": report.get("base_commits"),
            "commands": [],
            "numeric_tests": report.get("numeric_tests") or [],
            "acceptance": report.get("acceptance") or [],
            "report_path": path,
        }
        for i, cmd in enumerate(report.get("commands") or []):
            rows, files, cited_missing, reason = read_logs(
                report, cmd, workspace, task)
            log_verdict = worst_verdict(files)
            if not files:
                log_verdict = "not_run"
            reported = cmd.get("result")
            out["logs"]["commands"] += 1
            if files:
                out["logs"]["with_readable_log"] += 1
            if rows:
                out["logs"]["with_summary_rows"] += 1
            out["logs"]["cited_path_missing"] += len(cited_missing)
            # A disagreement is recorded when a file whose NAME THE COMMAND
            # CITED was read and its Test Summary rows contradict the report.
            # Sibling logs found beside it are not the command's artifact and
            # cannot create a disagreement -- that distinction is what stopped
            # the first version of this script from reporting S04's failing
            # `Pkg.test()` as a pass.
            primary = [f for f in files
                       if f.get("primary") and f.get("read")
                       and f.get("evidence_kind") == "test_summary"]
            primary_verdict = worst_verdict(primary)
            # Three exclusions, each a distinct fact and each recorded rather
            # than silently dropped, so the disagreement count stays interpretable:
            #   * `worktree_cleanliness_abort` is INFRASTRUCTURE. The packet's rule
            #     is that a dirty tree is `not_run` with a reason, not a failure --
            #     and the report saying `fail` about it is not a contradiction of
            #     the log, it is a difference of vocabulary that this table
            #     deliberately preserves on both sides.
            #   * a report whose own note says the failure is expected (a
            #     fail-closed guard, a planted mutation) is describing a pass.
            #   * a file with no `Test Summary` is not test evidence and gets a
            #     warning, not a disagreement.
            infra_abort = any(f.get("verdict_reason") == "worktree_cleanliness_abort"
                              for f in primary)
            # A command's OWN note can say the dirty-tree abort is the expected
            # result. S02 commands[3] does exactly that: `result: pass`, exit 1,
            # and a note reading "exit_code 1 is EXPECTED and by design ... the
            # substantive suite evidence was subsequently re-taken by the PARENT".
            # The log is `not_run`, the report says `pass`, and BOTH are honest --
            # so this is recorded as a disclosed infrastructure abort rather than
            # as a disagreement, and the report's label is left as reported.
            expected = (any(f.get("expected_failure") for f in primary)
                        or bool(EXPECTED_FAILURE.search(str(cmd.get("note") or "")))
                        or bool(EXPECTED_FAILURE.search(str(cmd.get("command") or ""))))
            if primary and reported in ("pass", "fail") and primary_verdict != reported:
                record = {"task": task, "commands_index": i, "reported": reported,
                          "log_verdict": primary_verdict,
                          "artifact": ", ".join(f["path"] for f in primary)}
                if infra_abort:
                    record["explained"] = (
                        "the log shows the worktree-cleanliness abort: SDPX's "
                        "test/runtests.jl refuses to run on a dirty tree, so this "
                        "is `not_run` with a reason (not an infrastructure FAIL and "
                        "not a numeric fail). The report's own `fail` label is kept "
                        "as reported."
                    )
                    out["not_run_infrastructure"].append(record)
                elif expected:
                    record["explained"] = "the report's own note says this failure is the expected result"
                    out["expected_failures"].append(record)
                else:
                    out["disagreements"].append(record)
            entry["commands"].append({
                "index": i,
                "command": cmd.get("command"),
                "exit_code": cmd.get("exit_code"),
                "reported_result": reported,
                "files": files,
                "cited_paths_missing": cited_missing,
                "log_summary_rows": rows,
                "log_verdict": log_verdict,
                "log_verdict_primary": primary_verdict if primary else None,
                "log_verdict_reason": reason,
            })
        out["reports"].append(entry)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace", nargs="?", default=os.getcwd())
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    data = extract(args.workspace)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.out}")
        print(f"reports={len(data['reports'])} commands={data['logs']['commands']} "
              f"with_readable_log={data['logs']['with_readable_log']} "
              f"with_summary_rows={data['logs']['with_summary_rows']} "
              f"cited_missing={data['logs']['cited_path_missing']} "
              f"disagreements={len(data['disagreements'])} "
              f"not_run_infrastructure={len(data['not_run_infrastructure'])} "
              f"expected_failures={len(data['expected_failures'])}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
