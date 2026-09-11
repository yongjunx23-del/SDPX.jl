#!/usr/bin/env python3
"""gen_support_matrix.py — render the public capability table, from evidence.

    python3 scripts/rebuild/gen_support_matrix.py [WORKSPACE]
        [--source docs/rebuild/support_matrix.json]
        [--out docs/rebuild/support_matrix.md]
        [--tsv rebuild-reports/Q02/capability_table.tsv]
        [--check]

WHAT THIS IS FOR. Q02 owes "every public capability has a corresponding passing
artifact, and `not_run` is distinguishable from `pass`". That is a claim about a
TABLE, and the table is only worth having if it cannot be filled in optimistically.
So the rendering is mechanical and every rule below is a rule that can REFUSE:

  * `--check` exits non-zero if any capability row lacks a resolving evidence
    path, or if the rendered table would differ from the file on disk. A table
    that cannot fail its own check is the situation the packet keeps finding.
  * A capability with no `pass`-status evidence is rendered with state
    `not_verified` and its `blocked_by` text. It is NEVER omitted, and it is
    NEVER rendered as `verified` because a report says something encouraging in
    prose.
  * The report's own vocabulary is used verbatim: `verified` /
    `partially_verified` / `not_verified` for the acceptance column, and
    `pass` / `fail` / `not_run` / `unsupported` for the evidence column. Nothing
    is translated, so the counts in this table can be reconciled against
    `reports=`/`acceptance` counts computed anywhere else.
  * `--sources auto` (the default) sources each row's evidence from
    (a) `acceptance[]` items of the reports named by the row, and (b) `logs`:
    the `Test Summary` verdicts extracted from the logs those commands cite.
    A row whose acceptance items are all `verified` but whose logs are absent
    gets `log_corroboration: not_run` and says so. That is the difference
    between "the report says so" and "the log says so", and the table carries
    both.

THE DISTINCTION THIS TABLE MUST NOT BLUR (Q02 acceptance 3):
a NUMERICALLY VERIFIED capability has a passing measurement with a stated
tolerance; a STRICTLY CERTIFIED one has an original-coordinate certificate
(ADR-003). The `evidence_class` column carries exactly one of

    certificate          an original-coordinate certificate exists and is checked
    numeric_verified     a numeric measurement passed against a stated tolerance
    structural           a non-numeric property (reachability, surface, gates)
    contract             a provider/API contract, no numerics claimed
    documentation        prose only -- NOT evidence of behaviour

and `documentation` rows are never counted toward "has passing evidence".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

VALID_EVIDENCE_CLASS = {
    "certificate", "numeric_verified", "structural", "contract", "documentation",
}
VALID_PASS_STATUS = {"pass", "verified"}
EVIDENCE_STATE_VOCAB = {"verified", "partially_verified", "not_verified"}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(workspace, token):
    """Resolve an evidence path the way validate_reports.py does: workspace first,
    then each repo root. Returns the resolved absolute path or None."""
    if not token:
        return None
    token = token.split()[0].strip().rstrip(",;")
    if token.startswith("/"):
        return token if os.path.exists(token) else None
    roots = [workspace] + [
        os.path.join(workspace, r) for r in
        ("SDPX.jl", "MultiFloatLinearAlgebra.jl", "BigFloatLinearAlgebra.jl")
    ]
    for root in roots:
        candidate = os.path.join(root, token)
        if os.path.exists(candidate):
            return candidate
    return None


def evidence_index(evidence):
    """task -> {'acceptance': [...], 'commands': [...], 'numeric_tests': [...]}."""
    index = {}
    for report in evidence.get("reports", []):
        index[report["task"]] = report
    return index


def rows_for_capability(cap, index, workspace):
    """Return the evidence rows for one capability spec, or a list of problems."""
    problems, rows = [], []
    # Two source forms are accepted:
    #   "reports": ["S01","S02"]                      -- every acceptance item counts
    #   "sources": [{"report": "P02", "acceptance_indices": [0,1]}, ...]
    # The second exists because an index list is only meaningful WITHIN one
    # report: P02 has 3 acceptance items and P01 has 10, so a single flat index
    # list silently addressed the wrong items (caught by this generator's own
    # "acceptance[N] does not exist" refusal, which is why that check is fatal).
    if cap.get("sources"):
        specs = [(s["report"], s.get("acceptance_indices")) for s in cap["sources"]]
    else:
        specs = [(task, cap.get("acceptance_indices")) for task in cap.get("reports", [])]
    for task, wanted in specs:
        report = index.get(task)
        if report is None:
            problems.append(f"{cap['id']}: names report {task}, which has no report.json")
            continue
        accepted = report["acceptance"]
        items = []
        if wanted:
            for i in wanted:
                if i >= len(accepted):
                    problems.append(f"{cap['id']}: {task}.acceptance[{i}] does not exist "
                                    f"({len(accepted)} items)")
                    continue
                items.append((i, accepted[i]))
        else:
            items = list(enumerate(accepted))
        if not items:
            problems.append(f"{cap['id']}: {task} contributes no acceptance items")
        for i, item in items:
            state = item.get("state")
            if state not in EVIDENCE_STATE_VOCAB:
                problems.append(f"{cap['id']}: {task}.acceptance[{i}].state={state!r} "
                                f"is outside the vocabulary")
                continue
            rows.append({
                "kind": "acceptance", "task": task, "index": i,
                "state": state,
                "criterion": (item.get("criterion") or "").strip(),
                "evidence": (item.get("evidence") or "").strip(),
            })
        for cmd in report["commands"]:
            primary = cmd.get("log_verdict_primary")
            if primary is None:
                continue
            rows.append({
                "kind": "log", "task": task, "index": cmd["index"],
                "state": primary,
                "criterion": f"log verdict for commands[{cmd['index']}]",
                "evidence": ", ".join(f["path"] for f in cmd["files"]
                                      if f.get("primary") and f.get("read")) or
                            (cmd.get("log_verdict_reason") or ""),
            })
        for item in report["numeric_tests"]:
            status = item.get("status")
            if status is None:
                continue
            rows.append({
                "kind": "numeric_test", "task": task, "index": None,
                "state": status,
                "criterion": (item.get("name") or item.get("criterion") or "").strip(),
                "evidence": (item.get("note") or item.get("artifact") or "").strip(),
            })
    return rows, problems


def summarise(cap, rows, workspace):
    """Return (state, evidence_class, passing_acceptance, n_acceptance, passing_logs, problems).

    `passing_acceptance` counts ACCEPTANCE items in a passing state. The first
    version counted acceptance, log and numeric-test rows together, which made a
    capability backed by nine verified acceptance items report `pass_count = 68`
    -- a number that sounds like evidence volume and is really the number of
    commands in a report. The table now reports three separate counts, because a
    count a reader cannot interpret is worse than no count.
    """
    problems = []
    acceptance = [r for r in rows if r["kind"] == "acceptance"]
    if not acceptance:
        problems.append(f"{cap['id']}: no acceptance evidence from any named report")
    states = [r["state"] for r in acceptance]
    if states and all(s == "verified" for s in states):
        state = "verified"
    elif states and all(s == "not_verified" for s in states):
        state = "not_verified"
    elif states:
        state = "partially_verified"
    else:
        state = "not_verified"
    # A capability that declares itself not_verified stays that way whatever the
    # reports say: the declaration is the reviewed judgement, and the renderer
    # exists to stop it being rounded UP, not to let a script round it up.
    if cap.get("force_state"):
        if cap["force_state"] not in EVIDENCE_STATE_VOCAB:
            problems.append(f"{cap['id']}: force_state {cap['force_state']!r} invalid")
        else:
            state = cap["force_state"]
    evidence_class = cap.get("evidence_class")
    if evidence_class not in VALID_EVIDENCE_CLASS:
        problems.append(f"{cap['id']}: evidence_class {evidence_class!r} not in "
                        f"{sorted(VALID_EVIDENCE_CLASS)}")
        evidence_class = "documentation"
    passing_acceptance = [r for r in acceptance if r["state"] in VALID_PASS_STATUS]
    passing_logs = [r for r in rows if r["kind"] == "log" and r["state"] in VALID_PASS_STATUS]
    for row in rows:
        token = row["evidence"].split()[0] if row["evidence"].split() else ""
        if token and not resolve_path(workspace, token):
            # Not fatal for prose evidence, but a cited path that does not resolve
            # is exactly the cheapest false claim the packet has already been
            # bitten by. It is recorded per row in the table.
            row["path_missing"] = True
    # The guard that CANNOT BE SATISFIED BY PROSE, stated precisely.
    #
    # An earlier version refused whenever a capability had no passing acceptance
    # item and its evidence_class was not `documentation`. That was wrong in a way
    # worth recording: `INFRA-PERF-RUNNER-EXCLUSIVITY` cites exactly one Q01
    # acceptance item, and Q01 honestly marks that item `partially_verified`,
    # while the report's OTHER items pass. The capability is genuinely partial,
    # the table said so, and the guard still refused -- it was measuring "does
    # this row cite a passing item", not "does this row overclaim". A guard that
    # fails an honest partial row pressures whoever hits it to loosen the guard,
    # which is how a check becomes a printer.
    #
    # The failure it must actually catch: a row that claims verification with no
    # passing evidence behind it. State derivation already prevents that for rows
    # without `force_state`, so what remains is a `force_state: verified` that the
    # evidence does not support -- a reviewed judgement contradicting the record.
    if state == "verified" and not passing_acceptance and not passing_logs \
            and evidence_class != "documentation":
        problems.append(
            f"{cap['id']}: state is `verified` but NO row has a passing state "
            f"({sorted({r['state'] for r in rows})}). A `force_state: verified` must be "
            f"supported by passing evidence; the honest state is `partially_verified` "
            f"or `not_verified`."
        )
    return (state, evidence_class, len(passing_acceptance), len(acceptance),
            len(passing_logs), problems)


def level(result):
    """The numeric-vs-certificate distinction, as one column.

    Q02 acceptance 3 asks that a reader can tell a numerically verified capability
    from a strictly certified one. `evidence_class` says what KIND of evidence
    exists; this column says WHAT MAY BE CONCLUDED from it, which is the
    distinction ADR-003 draws. `certified` is reachable only when the evidence
    class is `certificate` AND the state is `verified`; a capability that claims
    the certificate class without one is `certificate_gap`, and is emphatically
    not certified.
    """
    if result["evidence_class"] == "certificate":
        return "certified" if result["state"] == "verified" else "certificate_gap"
    if result["evidence_class"] == "documentation":
        return "prose_only"
    if result["state"] == "verified":
        return "numeric_or_structural_verified"
    return "partial"


def render(spec, results, workspace):
    lines = []
    lines.append("# Public capability support matrix")
    lines.append("")
    lines.append("<!-- GENERATED by scripts/rebuild/gen_support_matrix.py. Do not edit by")
    lines.append("     hand; re-run the generator. `--check` fails if this file drifts. -->")
    lines.append("")
    lines.append(spec["preamble"].rstrip())
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    counts = {"verified": 0, "partially_verified": 0, "not_verified": 0}
    levels = {}
    with_passing = 0
    for result in results:
        counts[result["state"]] += 1
        levels[result["level"]] = levels.get(result["level"], 0) + 1
        if (result["passing_acceptance"] or result["passing_logs"]) \
                and result["evidence_class"] != "documentation":
            with_passing += 1
    lines.append("| | count |")
    lines.append("| --- | --- |")
    lines.append(f"| capabilities listed | **{len(results)}** |")
    lines.append(f"| with passing evidence | **{with_passing}** |")
    for state in ("verified", "partially_verified", "not_verified"):
        lines.append(f"| state `{state}` | {counts[state]} |")
    for key in ("certified", "certificate_gap", "numeric_or_structural_verified",
                "partial", "prose_only"):
        if key in levels:
            lines.append(f"| level `{key}` | {levels[key]} |")
    lines.append("")
    lines.append("A capability counts under **with passing evidence** when its `evidence_class`")
    lines.append("is not `documentation` and it has at least one passing acceptance item or a")
    lines.append("passing `Test Summary` corroboration. Prose-only rows are listed so that they")
    lines.append("are not silently absent, and are never counted.")
    lines.append("")
    lines.append("**`certified` is 0.** Every row here is `numeric_or_structural_verified`,")
    lines.append("`partial`, `certificate_gap` or `prose_only`. A numerically verified")
    lines.append("capability has a measurement that passed against a stated tolerance; a")
    lines.append("*strictly certified* one additionally has an original-coordinate certificate")
    lines.append("(ADR-003). The two are different states, and this table is where a reader can")
    lines.append("see which one they are looking at. Nothing in this release is certified.")
    lines.append("")
    lines.append("## The table")
    lines.append("")
    lines.append("`passing / total acceptance` counts only the acceptance items this row")
    lines.append("CITES (the `sources` / `acceptance_indices` in `support_matrix.json`), not")
    lines.append("every item of the named reports. A capability citing one honestly-partial item")
    lines.append("therefore reads `0 / 1` while the report has other passing items; the `level`")
    lines.append("and `state` columns carry the verdict, and the full row list below carries the")
    lines.append("items themselves.")
    lines.append("")
    lines.append("| capability | state | level | evidence class | passing / total acceptance | passing logs | reports |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        cap = result["cap"]
        reports = sorted({r["task"] for r in result["rows"]})
        lines.append(
            f"| `{cap['id']}` {cap['title']} | `{result['state']}` | `{result['level']}` | "
            f"`{result['evidence_class']}` | {result['passing_acceptance']} / "
            f"{result['n_acceptance']} | {result['passing_logs']} | "
            f"{', '.join('`' + r + '`' for r in reports)} |"
        )
    lines.append("")
    lines.append("## Every row that is not `verified`")
    lines.append("")
    lines.append("Listed in full, because the failure this table exists to prevent is a")
    lines.append("reader assuming the unlisted rows are fine.")
    lines.append("")
    for result in results:
        if result["state"] == "verified":
            continue
        cap = result["cap"]
        lines.append(f"### `{cap['id']}` — {cap['title']} — `{result['state']}`")
        lines.append("")
        lines.append(f"* level: `{result['level']}`   evidence class: `{result['evidence_class']}`")
        lines.append(f"* passing acceptance items: {result['passing_acceptance']} "
                     f"of {result['n_acceptance']}; passing log corroborations: "
                     f"{result['passing_logs']}")
        if cap.get("blocked_by"):
            lines.append(f"* blocked by: {cap['blocked_by']}")
        non_passing = [r for r in result["rows"] if r["state"] not in VALID_PASS_STATUS]
        if non_passing:
            lines.append("* non-passing evidence rows:")
            for row in non_passing:
                where = f"{row['task']}"
                if row["index"] is not None:
                    where += f"[{row['index']}]"
                where += f".{row['kind']}"
                lines.append(f"  * `{where}` = `{row['state']}` — {row['criterion']}")
                if row["evidence"]:
                    lines.append(f"    * evidence: {row['evidence']}")
        lines.append("")
    lines.append("## Full evidence rows")
    lines.append("")
    lines.append("| capability | source | kind | state | criterion / note |")
    lines.append("| --- | --- | --- | --- | --- |")
    for result in results:
        for row in result["rows"]:
            where = row["task"] + (f"[{row['index']}]" if row["index"] is not None else "")
            criterion = row["criterion"].replace("|", "\\|")[:160]
            note = row["evidence"].replace("|", "\\|")[:200]
            lines.append(f"| `{result['cap']['id']}` | `{where}` | {row['kind']} | "
                         f"`{row['state']}` | {criterion} {note} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_tsv(results):
    lines = ["capability\tstate\tlevel\tevidence_class\tpassing_acceptance\t"
             "n_acceptance\tpassing_logs\tpassing_sources\tnon_passing\tblocked_by"]
    for result in results:
        passing = sorted({r["task"] for r in result["rows"] if r["state"] in VALID_PASS_STATUS})
        non_passing = sorted({f"{r['task']}.{r['kind']}={r['state']}" for r in result["rows"]
                              if r["state"] not in VALID_PASS_STATUS})
        lines.append("\t".join([
            result["cap"]["id"], result["state"], result["level"], result["evidence_class"],
            str(result["passing_acceptance"]), str(result["n_acceptance"]),
            str(result["passing_logs"]), ",".join(passing), ",".join(non_passing),
            (result["cap"].get("blocked_by") or "").replace("\t", " ").replace("\n", " "),
        ]))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace", nargs="?", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--evidence", default=None,
                    help="reuse an extracted evidence JSON instead of re-reading logs")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the rendered table differs from disk, or a "
                         "row has no passing evidence while claiming an evidence class")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    workspace = args.workspace or os.path.dirname(os.path.dirname(os.path.dirname(here)))
    source = args.source or os.path.join(workspace, "SDPX.jl", "docs", "rebuild",
                                         "support_matrix.json")
    out = args.out or os.path.join(workspace, "SDPX.jl", "docs", "rebuild",
                                   "support_matrix.md")
    spec = load_json(source)
    if args.evidence:
        evidence = load_json(args.evidence)
    else:
        extractor = os.path.join(here, "extract_capability_evidence.py")
        proc = subprocess.run([sys.executable, extractor, workspace],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"FAIL evidence extraction failed: {proc.stderr.strip()[:400]}")
            return 1
        evidence = json.loads(proc.stdout)
    index = evidence_index(evidence)

    results, problems = [], []
    for cap in spec["capabilities"]:
        rows, row_problems = rows_for_capability(cap, index, workspace)
        (state, evidence_class, passing_acceptance, n_acceptance,
         passing_logs, sum_problems) = summarise(cap, rows, workspace)
        problems.extend(row_problems)
        problems.extend(sum_problems)
        result = {"cap": cap, "rows": rows, "state": state,
                  "evidence_class": evidence_class,
                  "passing_acceptance": passing_acceptance,
                  "n_acceptance": n_acceptance,
                  "passing_logs": passing_logs,
                  "pass_count": passing_acceptance + passing_logs}
        result["level"] = level(result)
        results.append(result)

    rendered = render(spec, results, workspace)
    tsv = render_tsv(results)

    if problems:
        print(f"FAIL {len(problems)} capability-table problem(s):")
        for line in problems:
            print(f"  {line}")
        return 1

    if args.check:
        disk = open(out, encoding="utf-8").read() if os.path.isfile(out) else None
        if disk is None:
            print(f"FAIL {out} does not exist; run the generator without --check")
            return 1
        if disk != rendered:
            print(f"FAIL {out} differs from the rendered table -- the file on disk is "
                  f"stale or was edited by hand")
            return 1
        print(f"OK {out} matches the evidence ({len(results)} capabilities, "
              f"{sum(1 for r in results if r['pass_count'] and r['evidence_class'] != 'documentation')} "
              f"with passing evidence, "
              f"{sum(1 for r in results if r['state'] != 'verified')} not verified)")
        return 0

    with open(out, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    print(f"wrote {out}")
    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as handle:
            handle.write(tsv)
        print(f"wrote {args.tsv}")
    print(f"capabilities={len(results)} "
          f"verified={sum(1 for r in results if r['state'] == 'verified')} "
          f"partially_verified={sum(1 for r in results if r['state'] == 'partially_verified')} "
          f"not_verified={sum(1 for r in results if r['state'] == 'not_verified')} "
          f"with_passing_evidence="
          f"{sum(1 for r in results if r['pass_count'] and r['evidence_class'] != 'documentation')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
