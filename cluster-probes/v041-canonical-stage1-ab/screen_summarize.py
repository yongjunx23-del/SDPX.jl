#!/usr/bin/env python3
"""Concise gate for the development-stage parallel A/B screen.

Reads exactly three completed, disjoint screen result roots (LP, SOCP, SDP),
each produced by `stage1_ab.pbs` with `SCREEN_MODE=true` and one single-family
`CASE_FILTER` on its own pinned node.  This tool performs no solver/numeric
execution and is Python 3.6 compatible.

For every root:
- require root SUCCESS and baseline/candidate arm SUCCESS, no FAILED markers;
- require arms.conf, environment.txt, analyze.log, ab_report.csv and both arm
  results.csv files;
- require `screen_mode=true` and an explicit, non-truncated `AB_GATE_PASS` in
  analyze.log (the screen analyzer already fails closed on ratio > 1.10,
  CV >= 0.20, and nonzero endpoint/iteration deltas);
- `--legacy-screen-evidence` explicitly accepts pre-SCREEN_MODE roots whose
  arms.conf has no `screen_mode=true`: all other evidence (root/arm markers,
  analyze log gate, endpoint/iteration evidence, identity, thread contract,
  family coverage, ratio/CV gates) still applies and the root is not silently
  weakened.  Production screen roots still require `screen_mode=true` unless
  the flag is passed.
- require max_endpoint_norm=0 and max_iteration_delta=0 evidence, each exactly
  once;
- require identity/thread settings to agree across all three roots (commits,
  tree/archive/runner hashes, arithmetic, timing scheme, resource/thread
  contract, arm order, analyzer SHA).

Across the three roots:
- require case_filter to be exactly {lp_row_scaling, socp_many_tiny,
  sdp_hilbert}, one family per root, on three distinct nodes;
- require ab_report.csv family rows to cover exactly
  {lp, socp, sdp} x {float64, float64x4};
- require every family ratio <= 1.10;
- classify within-arm worst batch CV as pass (< 0.05), warning
  (0.05 .. 0.20), or fail (>= 0.20); warnings do not fail the screen.

Writes `screen_summary.csv` and `screen_summary.txt` into --output-dir
(default: current directory) and prints a compact text table.
"""

import argparse
import csv
import math
import pathlib
import sys

EXPECTED_FAMILIES = ("lp_row_scaling", "socp_many_tiny", "sdp_hilbert")
EXPECTED_ARITHMETIC = ("float64", "float64x4")
FAMILY_SHORT = {
    "lp_row_scaling": "lp",
    "socp_many_tiny": "socp",
    "sdp_hilbert": "sdp",
}
REPORT_FAMILIES = frozenset(FAMILY_SHORT.values())

RATIO_LIMIT = 1.10
CV_WARN_FLOOR = 0.05
CV_HARD_LIMIT = 0.20

EXPECTED_TIMING = {
    "timing_batch_size": "5",
    "timed_batches": "3",
    "repetitions": "16",
    "warmup": "true",
}
EXPECTED_THREADS = {
    "resource_class": "regular",
    "julia_threads": "4",
    "solver_threads": "4",
    "blas_threads": "1",
    "ppn": "5",
}

IDENTITY_KEYS = (
    "baseline_source",
    "baseline_commit",
    "baseline_tree_sha256",
    "baseline_archive_sha",
    "candidate_source",
    "candidate_commit",
    "candidate_tree_sha256",
    "candidate_archive_sha",
    "runner_source",
    "runner_tree_sha256",
    "ab_runner_root",
    "ab_runner_sha256",
    "arithmetic",
    "timing_batch_size",
    "timed_batches",
    "repetitions",
    "warmup",
    "resource_class",
    "julia_threads",
    "solver_threads",
    "blas_threads",
    "ppn",
    "time_limit",
    "max_iterations",
    "arm_order",
)


def _read_lines(path):
    with path.open(errors="replace") as stream:
        return stream.read().splitlines()


def _read_conf(path):
    config = {}
    if not path.is_file():
        return config
    for line in _read_lines(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            config[key.strip()] = value.strip()
    return config


def _float(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RuntimeError("non-numeric {0}: {1!r}".format(label, value))
    if not math.isfinite(number):
        raise RuntimeError("non-finite {0}: {1!r}".format(label, value))
    return number


def _check_analyze_log(root, failures):
    """Require a clean screen AB_GATE_PASS and exact zero endpoint/iteration
    evidence; returns (endpoint_norm, iteration_delta)."""
    lines = _read_lines(root / "analyze.log")
    text = "".join(lines)
    endpoint_norm = None
    iteration_delta = None
    endpoint_seen = 0
    iteration_seen = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("max_endpoint_norm="):
            endpoint_seen += 1
            try:
                endpoint_norm = float(stripped.split("=", 1)[1])
            except ValueError:
                endpoint_norm = math.inf
        elif stripped.startswith("max_iteration_delta="):
            iteration_seen += 1
            try:
                iteration_delta = float(stripped.split("=", 1)[1])
            except ValueError:
                iteration_delta = math.inf
    if "AB_GATE_PASS" not in text:
        failures.append("{0}: analyze.log missing AB_GATE_PASS".format(root))
    pass_lines = [
        line.strip() for line in lines if "AB_GATE_PASS" in line
    ]
    if pass_lines and pass_lines[-1] != "AB_GATE_PASS":
        failures.append(
            "{0}: AB_GATE_PASS marker truncated or polluted".format(root),
        )
    if "AB_GATE_FAILED" in text:
        failures.append("{0}: analyze.log contains AB_GATE_FAILED".format(root))
    if endpoint_seen != 1:
        failures.append(
            "{0}: max_endpoint_norm evidence must appear exactly once "
            "(got {1})".format(root, endpoint_seen),
        )
    elif endpoint_norm != 0.0:
        failures.append(
            "{0}: max_endpoint_norm must be exactly 0 "
            "(got {1})".format(root, endpoint_norm),
        )
    if iteration_seen != 1:
        failures.append(
            "{0}: max_iteration_delta evidence must appear exactly once "
            "(got {1})".format(root, iteration_seen),
        )
    elif iteration_delta != 0.0:
        failures.append(
            "{0}: max_iteration_delta must be exactly 0 "
            "(got {1})".format(root, iteration_delta),
        )
    return endpoint_norm, iteration_delta


def _read_report(path, failures):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        failures.append("{0}: ab_report.csv has no rows".format(path))
        return {}
    report = {}
    for row in rows:
        if row.get("level") != "family":
            continue
        key = (row.get("arithmetic", ""), row.get("family", ""))
        if key in report:
            failures.append(
                "{0}: duplicate family row {1}".format(path, key),
            )
        report[key] = row
    return report


def _classify_cv(worst_cv):
    if worst_cv < CV_WARN_FLOOR:
        return "pass"
    if worst_cv < CV_HARD_LIMIT:
        return "warn"
    return "fail"


def _summarize_roots(
    raw_roots,
    output_dir,
    failures,
    legacy_screen_evidence=False,
):
    nodes = {}
    common = None
    root_families = {}
    root_meta = {}
    summary_rows = []
    warnings = []

    for raw in raw_roots:
        root = pathlib.Path(raw).resolve()
        label = str(root)
        for name in (
            "arms.conf",
            "environment.txt",
            "analyze.log",
            "ab_report.csv",
            "baseline/results.csv",
            "candidate/results.csv",
            "baseline/SUCCESS",
            "candidate/SUCCESS",
            "SUCCESS",
        ):
            if not (root / name).is_file():
                failures.append("{0}: missing {1}".format(label, name))
        for name in (
            "FAILED",
            "baseline/FAILED",
            "candidate/FAILED",
        ):
            if (root / name).is_file():
                failures.append("{0}: {1} present".format(label, name))
        if not (root / "arms.conf").is_file():
            continue
        config = _read_conf(root / "arms.conf")
        env = _read_conf(root / "environment.txt")
        screen_mode = config.get("screen_mode", "")
        if screen_mode != "true":
            if legacy_screen_evidence:
                warnings.append(
                    "WARN {0}: legacy screen root without screen_mode=true; "
                    "accepted only via --legacy-screen-evidence".format(label),
                )
            else:
                failures.append(
                    "{0}: arms.conf screen_mode must be true "
                    "(got {1!r})".format(label, screen_mode),
                )
        case_filter = config.get("case_filter", "")
        if case_filter not in EXPECTED_FAMILIES:
            failures.append(
                "{0}: case_filter {1!r} is not a screen family".format(
                    label,
                    case_filter,
                ),
            )
        hostname = env.get("hostname", "")
        expected_node = env.get("expected_node", "")
        if not hostname:
            failures.append("{0}: environment.txt missing hostname".format(label))
        if expected_node and hostname and expected_node != hostname:
            failures.append(
                "{0}: expected_node {1!r} != hostname {2!r}".format(
                    label,
                    expected_node,
                    hostname,
                ),
            )
        if hostname:
            if hostname in nodes:
                failures.append(
                    "{0}: node {1!r} already used by {2}".format(
                        label,
                        hostname,
                        nodes[hostname],
                    ),
                )
            nodes[hostname] = label
        for key in IDENTITY_KEYS:
            if not config.get(key, ""):
                failures.append(
                    "{0}: arms.conf missing identity key {1}".format(label, key),
                )
        for key, expected in EXPECTED_TIMING.items():
            if config.get(key, "") != expected:
                failures.append(
                    "{0}: {1}={2!r} != screen default {3!r}".format(
                        label,
                        key,
                        config.get(key),
                        expected,
                    ),
                )
        for key, expected in EXPECTED_THREADS.items():
            if config.get(key, "") != expected:
                failures.append(
                    "{0}: {1}={2!r} != screen contract {3!r}".format(
                        label,
                        key,
                        config.get(key),
                        expected,
                    ),
                )
        arithmetic = set(config.get("arithmetic", "").split(","))
        if arithmetic != set(EXPECTED_ARITHMETIC):
            failures.append(
                "{0}: arithmetic {1!r} must be exactly {2}".format(
                    label,
                    config.get("arithmetic"),
                    ",".join(EXPECTED_ARITHMETIC),
                ),
            )
        if common is None:
            common = config
        elif any(
            config.get(key) != common.get(key)
            for key in IDENTITY_KEYS
        ):
            failures.append(
                "{0}: identity/thread mismatch with first screen root".format(
                    label,
                ),
            )
        endpoint_norm, iteration_delta = _check_analyze_log(root, failures)
        report = _read_report(root / "ab_report.csv", failures)
        root_families[label] = case_filter
        root_meta[label] = {
            "hostname": hostname,
            "config": config,
            "report": report,
            "endpoint_norm": endpoint_norm,
            "iteration_delta": iteration_delta,
        }

    expected_product = set()
    for family in EXPECTED_FAMILIES:
        for arithmetic in EXPECTED_ARITHMETIC:
            expected_product.add((arithmetic, family))

    seen_families = set()
    for label, case_filter in sorted(root_families.items()):
        if case_filter in seen_families:
            failures.append(
                "{0}: duplicate screen family {1!r}".format(label, case_filter),
            )
        seen_families.add(case_filter)
        meta = root_meta[label]
        report_family = FAMILY_SHORT.get(case_filter)
        if report_family not in REPORT_FAMILIES:
            failures.append(
                "{0}: cannot map case_filter {1!r} to a report family".format(
                    label,
                    case_filter,
                ),
            )
        required_rows = {
            (arithmetic, report_family) for arithmetic in EXPECTED_ARITHMETIC
        }
        report_keys = set(meta["report"])
        unknown_family_names = sorted({
            key[1] for key in report_keys if key[1] not in REPORT_FAMILIES
        })
        if unknown_family_names:
            failures.append(
                "{0}: ab_report contains unknown family names {1}".format(
                    label,
                    unknown_family_names,
                ),
            )
        if report_keys != required_rows:
            failures.append(
                "{0}: ab_report family rows {1} != required {2}".format(
                    label,
                    sorted(report_keys),
                    sorted(required_rows),
                ),
            )
    if seen_families != set(EXPECTED_FAMILIES):
        failures.append(
            "screen coverage must be exactly {0}, got {1}".format(
                ",".join(EXPECTED_FAMILIES),
                ",".join(sorted(seen_families)),
            ),
        )

    for label, meta in sorted(root_meta.items()):
        for key, row in sorted(meta["report"].items()):
            ratio = _float(row.get("ratio"), "ratio")
            baseline_cv = _float(row.get("baseline_cv"), "baseline_cv")
            candidate_cv = _float(row.get("candidate_cv"), "candidate_cv")
            worst_cv = max(baseline_cv, candidate_cv)
            cv_class = _classify_cv(worst_cv)
            row_gate = "pass"
            if ratio > RATIO_LIMIT:
                failures.append(
                    "{0} {1}: ratio {2:.6f} > {3}".format(
                        label,
                        ",".join(key),
                        ratio,
                        RATIO_LIMIT,
                    ),
                )
                row_gate = "fail"
            if cv_class == "fail":
                failures.append(
                    "{0} {1}: worst within-arm CV {2:.6f} >= hard limit {3}".format(
                        label,
                        ",".join(key),
                        worst_cv,
                        CV_HARD_LIMIT,
                    ),
                )
                row_gate = "fail"
            elif cv_class == "warn":
                warnings.append(
                    "WARN {0} {1}: worst within-arm CV {2:.6f} in warning band "
                    "{3:.6f}..{4:.6f}".format(
                        label,
                        ",".join(key),
                        worst_cv,
                        CV_WARN_FLOOR,
                        CV_HARD_LIMIT,
                    ),
                )
                if row_gate == "pass":
                    row_gate = "warn"
            summary_rows.append({
                "root": pathlib.Path(label).name,
                "node": meta["hostname"],
                "family": FAMILY_SHORT.get(key[1], key[1]),
                "arithmetic": key[0],
                "baseline_median": row.get("baseline_median", ""),
                "candidate_median": row.get("candidate_median", ""),
                "ratio": ratio,
                "baseline_cv": baseline_cv,
                "candidate_cv": candidate_cv,
                "worst_cv": worst_cv,
                "cv_class": cv_class,
                "endpoint_norm": meta["endpoint_norm"],
                "iteration_delta": meta["iteration_delta"],
                "gate": row_gate,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "screen_summary.csv"
    with csv_path.open("w", newline="") as stream:
        if summary_rows:
            writer = csv.DictWriter(stream, fieldnames=sorted(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        else:
            stream.write("root,node,family,arithmetic,baseline_median,"
                         "candidate_median,ratio,baseline_cv,candidate_cv,"
                         "worst_cv,cv_class,endpoint_norm,iteration_delta,"
                         "gate\n")

    text_path = output_dir / "screen_summary.txt"
    table = _format_table(summary_rows)
    with text_path.open("w") as stream:
        stream.write(table)
        stream.write("\n")
        for message in warnings:
            stream.write(message + "\n")
        if failures:
            for message in failures:
                stream.write("  FAIL: " + message + "\n")
        stream.write(
            "SCREEN_SUMMARY_{0}\n".format(
                "FAILED" if failures else "PASS",
            ),
        )
    print(table)
    for message in warnings:
        print(message)
    print(
        "SCREEN_COVERAGE roots={0} families={1} nodes={2} "
        "arithmetic={3}".format(
            len(root_meta),
            ",".join(FAMILY_SHORT[f] for f in EXPECTED_FAMILIES),
            len(nodes),
            ",".join(EXPECTED_ARITHMETIC),
        ),
    )
    print(
        "SCREEN_IDENTITY {0}".format(
            "ok" if common is not None and not any(
                "identity/thread mismatch" in message
                or "missing identity key" in message
                for message in failures
            ) else "failed",
        ),
    )
    print("SCREEN_RATIO {0}".format("ok" if not any(
        "ratio" in message and ">" in message for message in failures
    ) else "failed"))
    print("SCREEN_ENDPOINT_ITERATION {0}".format("ok" if not any(
        "max_endpoint_norm" in message or "max_iteration_delta" in message
        for message in failures
    ) else "failed"))
    if failures:
        for message in failures:
            print("  FAIL:", message)
        print("SCREEN_SUMMARY_FAILED")
        return 1
    if warnings:
        print("SCREEN_SUMMARY_PASS_WITH_WARNINGS")
    else:
        print("SCREEN_SUMMARY_PASS")
    return 0


def _format_table(rows):
    header = [
        "root", "node", "family", "arithmetic", "ratio",
        "baseline_cv", "candidate_cv", "worst_cv", "cv_class",
        "endpoint_norm", "iteration_delta", "gate",
    ]
    widths = {name: len(name) for name in header}
    for row in rows:
        for name in header:
            value = row[name]
            if isinstance(value, float):
                text = "{0:.6g}".format(value)
            else:
                text = str(value)
            widths[name] = max(widths[name], len(text))
    lines = []
    lines.append(" ".join(
        name.ljust(widths[name]) for name in header
    ))
    lines.append(" ".join(
        "-" * widths[name] for name in header
    ))
    for row in rows:
        cells = []
        for name in header:
            value = row[name]
            if isinstance(value, float):
                text = "{0:.6g}".format(value)
            else:
                text = str(value)
            cells.append(text.ljust(widths[name]))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def _write_synthetic_root(
    root,
    *,
    family,
    node,
    screen_mode=True,
    ratio=1.0,
    cv=0.01,
    endpoint_norm="0",
    iteration_delta="0",
    candidate_commit=None,
    julia_threads="4",
):
    """Write a minimal screen-looking result root for self-tests."""
    root = pathlib.Path(root)
    for arm in ("baseline", "candidate"):
        (root / arm).mkdir(parents=True)
    (root / "SUCCESS").touch()
    (root / "baseline" / "SUCCESS").touch()
    (root / "candidate" / "SUCCESS").touch()
    if candidate_commit is None:
        candidate_commit = "2" * 40
    screen_line = "screen_mode=true\n" if screen_mode else ""
    (root / "arms.conf").write_text(
        screen_line +
        "baseline_source=/synthetic/baseline\n"
        "baseline_commit=" + "1" * 40 + "\n"
        "baseline_tree_sha256=" + "3" * 64 + "\n"
        "baseline_archive_sha=" + "6" * 64 + "\n"
        "candidate_source=/synthetic/candidate\n"
        "candidate_commit=" + candidate_commit + "\n"
        "candidate_tree_sha256=" + "4" * 64 + "\n"
        "candidate_archive_sha=" + "7" * 64 + "\n"
        "runner_source=/synthetic/runner\n"
        "runner_tree_sha256=" + "5" * 64 + "\n"
        "ab_runner_root=/synthetic/ab\n"
        "ab_runner_sha256=" + "a" * 64 + "\n"
        "arithmetic=float64,float64x4\n"
        "case_filter=" + family + "\n"
        "timing_batch_size=5\n"
        "timed_batches=3\n"
        "repetitions=16\n"
        "warmup=true\n"
        "resource_class=regular\n"
        "julia_threads=" + julia_threads + "\n"
        "solver_threads=4\n"
        "blas_threads=1\n"
        "ppn=5\n"
        "time_limit=900\n"
        "max_iterations=300\n"
        "arm_order=baseline_first\n",
    )
    (root / "environment.txt").write_text(
        "hostname=" + node + "\n"
        "expected_node=" + node + "\n",
    )
    (root / "analyze.log").write_text(
        "AB_GATE_PASS\n"
        "max_endpoint_norm=" + endpoint_norm + "\n"
        "max_iteration_delta=" + iteration_delta + "\n",
    )
    report_family = FAMILY_SHORT[family]
    report = (
        "level,arithmetic,family,problem,baseline_median,candidate_median,"
        "ratio,baseline_cv,candidate_cv,gate\n"
    )
    for arithmetic in EXPECTED_ARITHMETIC:
        report += (
            "family,{0},{1},,10.0,{2:.6f},{3:.6f},{4:.6f},{4:.6f},pass\n"
        ).format(arithmetic, report_family, 10.0 * ratio, ratio, cv)
    (root / "ab_report.csv").write_text(report)
    for arm in ("baseline", "candidate"):
        (root / arm / "results.csv").write_text(
            "run_id,arithmetic,family,problem,repetition,gate_pass\n"
            "r1,float64," + family + ",p,1,true\n",
        )


def _patch_arms_conf(path, key, value):
    """Rewrite one key in a synthetic arms.conf (self-test only)."""
    lines = path.read_text().splitlines()
    found = False
    for index, line in enumerate(lines):
        current, sep, _ = line.partition("=")
        if sep and current.strip() == key:
            lines[index] = key + "=" + value
            found = True
            break
    if not found:
        raise RuntimeError("self-test key not found: " + key)
    path.write_text("\n".join(lines) + "\n")


def _self_test():
    import shutil
    import tempfile

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="screen_summarize_selftest_"))
    try:
        def make_roots(prefix, **overrides):
            roots = []
            for index, family in enumerate(EXPECTED_FAMILIES):
                root = tmp / (prefix + str(index))
                kwargs = {"family": family, "node": "screen-node" + str(index)}
                kwargs.update(overrides)
                _write_synthetic_root(root, **kwargs)
                roots.append(str(root))
            return roots

        failures = []
        out_dir = tmp / "out"
        roots = make_roots("pass_")
        code = _summarize_roots(roots, out_dir, failures)
        expected = 0
        print(
            "SELF_TEST pass: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        roots = make_roots("warn_", cv=0.10)
        code = _summarize_roots(roots, out_dir, [])
        expected = 0
        print(
            "SELF_TEST warn_cv: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        roots = make_roots("ratio_", ratio=1.15)
        code = _summarize_roots(roots, out_dir, [])
        expected = 1
        print(
            "SELF_TEST fail_ratio: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        roots = make_roots("cv_", cv=0.25)
        code = _summarize_roots(roots, out_dir, [])
        expected = 1
        print(
            "SELF_TEST fail_cv: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        roots = make_roots("ident_")
        _patch_arms_conf(
            pathlib.Path(roots[0]) / "arms.conf",
            "candidate_commit",
            "9" * 40,
        )
        code = _summarize_roots(roots, out_dir, [])
        expected = 1
        print(
            "SELF_TEST fail_identity: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        roots = make_roots("thread_")
        _patch_arms_conf(
            pathlib.Path(roots[1]) / "arms.conf",
            "julia_threads",
            "8",
        )
        code = _summarize_roots(roots, out_dir, [])
        expected = 1
        print(
            "SELF_TEST fail_threads: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        roots = make_roots("endpoint_", endpoint_norm="1e-9")
        code = _summarize_roots(roots, out_dir, [])
        expected = 1
        print(
            "SELF_TEST fail_endpoint: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        roots = make_roots("legacy_", screen_mode=False)
        code = _summarize_roots(roots, out_dir, [])
        expected = 1
        print(
            "SELF_TEST legacy_screen_fail_closed: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        roots = make_roots("legacy_ok_", screen_mode=False)
        code = _summarize_roots(
            roots,
            out_dir,
            [],
            legacy_screen_evidence=True,
        )
        expected = 0
        print(
            "SELF_TEST legacy_screen_accepted: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        # Two roots with the same family produce a coverage failure.
        dup_root = tmp / "dup"
        _write_synthetic_root(
            dup_root,
            family="lp_row_scaling",
            node="screen-node-dup",
        )
        roots = [
            str(tmp / "pass_0"),
            str(dup_root),
            str(tmp / "pass_2"),
        ]
        code = _summarize_roots(roots, out_dir, [])
        expected = 1
        print(
            "SELF_TEST fail_coverage: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        print("SELF_TEST_PASS")
        return 0
    finally:
        shutil.rmtree(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_roots", nargs="*")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--legacy-screen-evidence",
        action="store_true",
        help=(
            "accept pre-SCREEN_MODE roots without arms.conf screen_mode=true; "
            "all other gates still apply"
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if len(args.result_roots) != 3:
        parser.error(
            "exactly three screen result roots are required "
            "(LP, SOCP, SDP)",
        )
    failures = []
    code = _summarize_roots(
        args.result_roots,
        pathlib.Path(args.output_dir).resolve(),
        failures,
        legacy_screen_evidence=args.legacy_screen_evidence,
    )
    if failures and code == 0:
        code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
