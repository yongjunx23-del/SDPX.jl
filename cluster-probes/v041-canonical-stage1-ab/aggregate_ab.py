#!/usr/bin/env python3
"""Multi-node aggregate gate for completed Stage-1 A/B result roots.

Inputs are two or more completed result directories, each containing
arms.conf, analyze.log, ab_report.csv, and baseline/candidate results.csv
from one pinned-node run.  This tool performs no solver/numeric execution and
is Python 3.6 compatible.

For each input:
- require the local analyzer gate to have passed: an explicit non-empty
  `AB_GATE_PASS` line in analyze.log (a missing/truncated marker is rejected),
  root SUCCESS present, root FAILED absent, and no arm FAILED markers;
- require matching commits/tree hashes/settings across every input
  (baseline/candidate commit, tree hashes, archive hashes, arithmetic,
  case_filter, timing_batch_size/timed_batches/repetitions, resource/thread
  config, runner source/tree, and analyzer SHA);
- load ab_report.csv family rows and require every family ratio <= 1.10 and
  both within-arm batch CVs < 0.05;
- record each node's per-family ratio.

Across nodes, for each family key shared by all inputs, aggregate the
same-node paired ratios: report median/min/max/node count/worst within-arm
CV and fail unless the aggregated median ratio <= 1.10.
"""

import argparse
import csv
import math
import pathlib
import statistics
import sys

RATIO_LIMIT = 1.10
CV_LIMIT = 0.05
ORDERED_CV_LIMIT = 0.20
ORDERED_WARN_CV_LIMIT = 0.05
ORDERED_MIN_NODES = 3
# Student-t one-sided 95% critical values by degrees of freedom (df = n-1),
# stored with fixed precision so Python 3.6 needs no statistics.t or scipy.
ORDERED_T_TABLE = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    15: 1.753,
    20: 1.725,
    30: 1.697,
    40: 1.684,
    60: 1.671,
}

ARMS_KEYS = (
    "baseline_commit",
    "candidate_commit",
    "baseline_tree_sha256",
    "candidate_tree_sha256",
    "runner_tree_sha256",
    "baseline_archive_sha",
    "candidate_archive_sha",
    "runner_source",
    "baseline_source",
    "candidate_source",
    "arithmetic",
    "case_filter",
    "repetitions",
    "timing_batch_size",
    "timed_batches",
    "resource_class",
    "julia_threads",
    "solver_threads",
    "blas_threads",
    "ppn",
    "max_iterations",
    "time_limit",
    "ab_runner_sha256",
    "arm_order",
)


def _read_lines(path):
    with path.open(errors="replace") as stream:
        return stream.read().splitlines()


def _read_arms_conf(path):
    config = {}
    for line in _read_lines(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            config[key.strip()] = value.strip()
    return config


def _read_environment(root):
    """Read key=value provenance lines from environment.txt; used for node
    pairing because the runner does not emit hostname in arms.conf."""
    env = {}
    path = root / "environment.txt"
    if not path.is_file():
        return env
    for line in _read_lines(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            env[key.strip()] = value.strip()
    return env


def _read_report(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("no rows in {0}".format(path))
    families = {}
    for row in rows:
        if row.get("level") != "family":
            continue
        family = (row["arithmetic"], row["family"])
        if family in families:
            raise RuntimeError("duplicate family row in {0}".format(path))
        families[family] = row
    return families


def _float(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RuntimeError("non-numeric {0}: {1!r}".format(label, value))
    if not math.isfinite(number):
        raise RuntimeError("non-finite {0}: {1!r}".format(label, value))
    return number


def _check_report(analyzer_log, report_path, failures):
    lines = _read_lines(analyzer_log)
    if "AB_GATE_FAILED" in "".join(lines):
        failures.append("analyze.log reports AB_GATE_FAILED")
    if "AB_GATE_PASS" not in "".join(lines):
        failures.append("analyze.log missing explicit AB_GATE_PASS marker")
    pass_lines = [
        line.strip() for line in lines if "AB_GATE_PASS" in line
    ]
    if pass_lines and pass_lines[-1] != "AB_GATE_PASS":
        failures.append("analyze.log AB_GATE_PASS marker is truncated or polluted")
    try:
        report = _read_report(report_path)
    except RuntimeError as err:
        failures.append(str(err))
        return {}
    if not report:
        failures.append("ab_report.csv has no family rows")
        return report
    for key, row in report.items():
        ratio = _float(row.get("ratio"), "ratio")
        baseline_cv = _float(row.get("baseline_cv"), "baseline_cv")
        candidate_cv = _float(row.get("candidate_cv"), "candidate_cv")
        if ratio > RATIO_LIMIT:
            failures.append(
                "family ratio {0}: {1:.6f} > {2}".format(
                    key,
                    ratio,
                    RATIO_LIMIT,
                ),
            )
        for label, cv in (("baseline_cv", baseline_cv),
                          ("candidate_cv", candidate_cv)):
            if cv >= CV_LIMIT:
                failures.append(
                    "family {0} {1}: {2:.6f} >= {3}".format(
                        key,
                        label,
                        cv,
                        CV_LIMIT,
                    ),
                )
    return report


def _t_upper_one_sided(df):
    """Conservative one-sided 95% t value; for a df not in the table, use the
    largest stored df that is <= the requested df (t decreases as df grows,
    so this keeps the interval conservative)."""
    if df in ORDERED_T_TABLE:
        return ORDERED_T_TABLE[df]
    candidates = sorted(
        candidate for candidate in ORDERED_T_TABLE if candidate <= df
    )
    if candidates:
        return ORDERED_T_TABLE[candidates[-1]]
    return ORDERED_T_TABLE[min(ORDERED_T_TABLE)]


def _has_non_timing_failures(log_lines):
    """Return True when analyze.log contains a non-timing FAIL line."""
    for line in log_lines:
        stripped = line.strip()
        if not stripped.startswith("FAIL:"):
            continue
        if stripped.startswith("FAIL: timing gate failed for "):
            continue
        return True
    return False


def _ordered_report_evidence(log_lines):
    """Return (non_timing, endpoint_ok, iteration_ok, endpoint_seen,
    iteration_seen).  Only exact timing-gate failures may be tolerated; any
    other FAIL line or missing/nonzero endpoint/iteration evidence is a hard
    failure.  Each evidence marker must appear exactly once and parse to an
    exact zero."""
    non_timing = _has_non_timing_failures(log_lines)
    endpoint_ok = True
    iteration_ok = True
    endpoint_seen = False
    iteration_seen = False
    for line in log_lines:
        stripped = line.strip()
        if stripped.startswith("max_endpoint_norm="):
            if endpoint_seen:
                endpoint_ok = False
                continue
            endpoint_seen = True
            try:
                endpoint_ok = float(stripped.split("=", 1)[1]) == 0.0
            except ValueError:
                endpoint_ok = False
        elif stripped.startswith("max_iteration_delta="):
            if iteration_seen:
                iteration_ok = False
                continue
            iteration_seen = True
            try:
                iteration_ok = int(stripped.split("=", 1)[1]) == 0
            except ValueError:
                iteration_ok = False
    return non_timing, endpoint_ok, iteration_ok, endpoint_seen, iteration_seen


def _legacy_baseline_first_ok(root, config, label, failures):
    """Explicit legacy baseline-first screening: arm_order is absent, analyzer
    provenance is fixed, and baseline artifacts precede candidate artifacts."""
    ok = True
    runner_sha = config.get("ab_runner_sha256", "").strip()
    runner_tree = config.get("runner_tree_sha256", "").strip()
    if not runner_sha or not runner_tree:
        failures.append(
            "{0}: legacy root missing ab_runner_sha256/runner_tree_sha256 "
            "provenance".format(label),
        )
        ok = False
    baseline_success = root / "baseline" / "SUCCESS"
    candidate_success = root / "candidate" / "SUCCESS"
    if not baseline_success.is_file() or not candidate_success.is_file():
        failures.append(
            "{0}: legacy root needs baseline/candidate SUCCESS markers".format(label),
        )
        return False
    try:
        precedes = baseline_success.stat().st_mtime <= candidate_success.stat().st_mtime
    except OSError:
        precedes = False
    if not precedes:
        failures.append(
            "{0}: legacy root baseline artifacts do not precede candidate".format(
                label,
            ),
        )
        ok = False
    return ok


def _write_synthetic_root(
    root,
    *,
    gate_pass=True,
    truncated_pass=False,
    provenance_mismatch=False,
    arm_order="baseline_first",
    host="synthetic-node",
    timing_only=False,
    non_timing_failure=False,
    include_arm_order=True,
):
    """Write a minimal completed-looking result root for self-tests."""
    root = pathlib.Path(root)
    for arm in ("baseline", "candidate"):
        (root / arm).mkdir(parents=True)
    if timing_only or non_timing_failure:
        (root / "FAILED").touch()
    else:
        (root / "SUCCESS").touch()
    for arm in ("baseline", "candidate"):
        (root / arm / "SUCCESS").touch()
        (root / arm / "results.csv").write_text(
            "run_id,arithmetic,family,problem,repetition,gate_pass\n"
            "r1,float64,lp,lp_row_scaling,1,true\n",
        )
    (root / "ab_report.csv").write_text(
        "level,arithmetic,family,problem,baseline_median,candidate_median,"
        "ratio,baseline_cv,candidate_cv,gate\n"
        "family,float64,lp,,10.0,10.0,1.0,0.01,0.01,pass\n",
    )
    (root / "baseline.process.time.txt").write_text(
        "Maximum resident set size (kbytes): 100000\n",
    )
    (root / "candidate.process.time.txt").write_text(
        "Maximum resident set size (kbytes): 100000\n",
    )
    if timing_only:
        (root / "analyze.log").write_text(
            "AB_GATE_FAILED\n"
            "  FAIL: timing gate failed for ('float64', 'lp'): ratio=1.05\n"
            "max_endpoint_norm=0\n"
            "max_iteration_delta=0\n",
        )
    elif non_timing_failure:
        (root / "analyze.log").write_text(
            "AB_GATE_FAILED\n"
            "  FAIL: certificate_valid differs\n"
            "max_endpoint_norm=0\n"
            "max_iteration_delta=0\n",
        )
    elif gate_pass:
        marker = "AB_GATE_PASS"
        if truncated_pass:
            marker = "AB_GATE_PASS\nFAIL: analyzer failed\nAB_GATE_FAILED"
        (root / "analyze.log").write_text(marker + "\n")
    else:
        (root / "analyze.log").write_text(
            "AB_GATE_FAILED\nFAIL: synthetic failure\n",
        )
    extra_sha = "z" * 64 if provenance_mismatch else "a" * 64
    arms_conf = (
        "baseline_commit=" + "1" * 40 + "\n"
        "candidate_commit=" + "2" * 40 + "\n"
        "baseline_tree_sha256=" + "3" * 64 + "\n"
        "candidate_tree_sha256=" + "4" * 64 + "\n"
        "runner_tree_sha256=" + "5" * 64 + "\n"
        "baseline_archive_sha=" + "6" * 64 + "\n"
        "candidate_archive_sha=" + "7" * 64 + "\n"
        "runner_source=/synthetic/runner\n"
        "baseline_source=/synthetic/baseline\n"
        "candidate_source=/synthetic/candidate\n"
        "arithmetic=float64\n"
        "case_filter=lp_row_scaling\n"
        "repetitions=31\n"
        "timing_batch_size=10\n"
        "timed_batches=3\n"
        "resource_class=regular\n"
        "julia_threads=4\n"
        "solver_threads=4\n"
        "blas_threads=1\n"
        "ppn=5\n"
        "max_iterations=300\n"
        "time_limit=900\n"
        "ab_runner_sha256=" + extra_sha + "\n"
    )
    if include_arm_order:
        arms_conf += "arm_order=" + arm_order + "\n"
    (root / "arms.conf").write_text(arms_conf)
    (root / "environment.txt").write_text(
        "hostname=" + host + "\n"
        "expected_node=" + host + "\n",
    )


def _self_test():
    import shutil
    import tempfile

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="aggregate_ab_selftest_"))
    try:
        good = tmp / "good"
        _write_synthetic_root(good, gate_pass=True)
        ok_root = tmp / "ok"
        _write_synthetic_root(ok_root, gate_pass=True)
        ordered_roots = []
        for host_index in range(3):
            base = tmp / ("ordered_host" + str(host_index))
            alt = tmp / ("ordered_host" + str(host_index) + "_alt")
            _write_synthetic_root(
                base,
                timing_only=True,
                arm_order="baseline_first",
                host="synthetic-node" + str(host_index),
            )
            _write_synthetic_root(
                alt,
                timing_only=True,
                arm_order="candidate_first",
                host="synthetic-node" + str(host_index),
            )
            ordered_roots.append(str(base))
            ordered_roots.append(str(alt))
        code = _run([str(good), str(ok_root)])
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

        missing_pass = tmp / "missing_pass"
        _write_synthetic_root(missing_pass, gate_pass=False)
        code = _run([str(good), str(missing_pass)])
        expected = 1
        print(
            "SELF_TEST missing_pass_marker: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        mismatch = tmp / "mismatch"
        _write_synthetic_root(mismatch, gate_pass=True, provenance_mismatch=True)
        code = _run([str(good), str(mismatch)])
        expected = 1
        print(
            "SELF_TEST provenance_mismatch: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1
        code = _run_ordered(ordered_roots)
        expected = 0
        print(
            "SELF_TEST ordered_pass: exit={0} expected={1} {2}".format(
                code,
                expected,
                "PASS" if code == expected else "FAIL",
            ),
        )
        if code != expected:
            return 1

        bad_root = tmp / "ordered_bad"
        _write_synthetic_root(
            bad_root,
            non_timing_failure=True,
            arm_order="candidate_first",
            host="synthetic-node2",
        )
        ordered_bad_roots = []
        for host_index in range(3):
            base = tmp / ("ordered_bad_host" + str(host_index))
            alt = tmp / ("ordered_bad_host" + str(host_index) + "_alt")
            _write_synthetic_root(
                base,
                timing_only=True,
                arm_order="baseline_first",
                host="synthetic-node" + str(host_index),
            )
            _write_synthetic_root(
                alt,
                timing_only=True,
                arm_order="candidate_first",
                host="synthetic-node" + str(host_index),
            )
            ordered_bad_roots.append(str(base))
            ordered_bad_roots.append(str(alt))
        ordered_bad_roots[-1] = str(bad_root)
        code = _run_ordered(ordered_bad_roots)
        expected = 1
        print(
            "SELF_TEST ordered_non_timing_fail: exit={0} expected={1} {2}".format(
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


def _run(result_dirs):
    if len(result_dirs) < 2:
        raise SystemExit("need at least two completed result directories")

    nodes = []
    common = None
    failures = []
    for raw in result_dirs:
        root = pathlib.Path(raw).resolve()
        label = str(root)
        conf_path = root / "arms.conf"
        report_path = root / "ab_report.csv"
        log_path = root / "analyze.log"
        baseline_csv = root / "baseline" / "results.csv"
        candidate_csv = root / "candidate" / "results.csv"
        for path, name in (
            (conf_path, "arms.conf"),
            (report_path, "ab_report.csv"),
            (log_path, "analyze.log"),
            (baseline_csv, "baseline/results.csv"),
            (candidate_csv, "candidate/results.csv"),
            (root / "SUCCESS", "root SUCCESS"),
            (root / "FAILED", "root FAILED"),
        ):
            if name == "root FAILED":
                if path.is_file():
                    failures.append("{0}: root FAILED present".format(label))
            elif not path.is_file():
                failures.append("{0}: missing {1}".format(label, name))
        if not conf_path.is_file():
            continue
        config = _read_arms_conf(conf_path)
        if not all(key in config for key in ARMS_KEYS):
            missing = [key for key in ARMS_KEYS if key not in config]
            failures.append(
                "{0}: arms.conf missing keys {1}".format(label, missing),
            )
            continue
        if common is None:
            common = config
        elif any(
            config.get(key) != common.get(key)
            for key in ARMS_KEYS
            if key != "arm_order"
        ):
            failures.append(
                "{0}: settings/identity mismatch with first result root".format(
                    label,
                ),
            )
        if report_path.is_file() and log_path.is_file():
            report = _check_report(log_path, report_path, failures)
        else:
            report = {}
        for arm in ("baseline", "candidate"):
            marker = root / arm / "FAILED"
            if marker.is_file():
                failures.append("{0}: {1}/FAILED present".format(label, arm))
            success = root / arm / "SUCCESS"
            if not success.is_file():
                failures.append("{0}: {1}/SUCCESS missing".format(label, arm))
        nodes.append({"label": label, "config": config, "report": report})

    if common is None:
        failures.append("no valid result roots supplied")
    family_keys = None
    for node in nodes:
        keys = set(node["report"])
        if family_keys is None:
            family_keys = keys
        elif keys != family_keys:
            failures.append(
                "{0}: family set differs from first node".format(node["label"]),
            )

    print("node,family,ratio,worst_cv")
    aggregates = {}
    for node in nodes:
        for key, row in node["report"].items():
            ratio = _float(row.get("ratio"), "ratio")
            cv_values = [
                _float(row.get("baseline_cv"), "baseline_cv"),
                _float(row.get("candidate_cv"), "candidate_cv"),
            ]
            worst_cv = max(cv_values)
            aggregates.setdefault(key, []).append(ratio)
            print(
                "{0},{1},{2:.6f},{3:.6f}".format(
                    node["label"],
                    ",".join(key),
                    ratio,
                    worst_cv,
                ),
            )

    aggregated_ok = True
    for key in sorted(aggregates):
        ratios = aggregates[key]
        median = statistics.median(ratios)
        minimum = min(ratios)
        maximum = max(ratios)
        ok = median <= RATIO_LIMIT
        aggregated_ok = aggregated_ok and ok
        print(
            "AGGREGATE {0}: node_count={1} median={2:.6f} min={3:.6f} "
            "max={4:.6f} gate={5}".format(
                ",".join(key),
                len(ratios),
                median,
                minimum,
                maximum,
                "pass" if ok else "fail",
            ),
        )

    if failures or not aggregated_ok:
        for message in failures:
            print("  FAIL:", message)
        print("AGGREGATE_GATE_FAILED")
        return 1
    print("AGGREGATE_GATE_PASS")
    return 0


def _run_ordered(result_dirs, legacy_baseline_first=False):
    """Order-balanced screening across two result roots per node (one per
    arm_order).  Local analyzer CV failures are accepted only when all other
    correctness/identity evidence is clean; per-node geometric combined ratios
    must be <= 1.10, the geometric aggregate must be <= 1.10, and for n>=3 the
    one-sided 95% t upper bound must be <= 1.10.
    """
    nodes = []
    failures = []
    common = None
    host_order = {}
    for raw in result_dirs:
        root = pathlib.Path(raw).resolve()
        label = str(root)
        conf_path = root / "arms.conf"
        report_path = root / "ab_report.csv"
        log_path = root / "analyze.log"
        baseline_csv = root / "baseline" / "results.csv"
        candidate_csv = root / "candidate" / "results.csv"
        for path, name in (
            (conf_path, "arms.conf"),
            (report_path, "ab_report.csv"),
            (log_path, "analyze.log"),
            (baseline_csv, "baseline/results.csv"),
            (candidate_csv, "candidate/results.csv"),
        ):
            if not path.is_file():
                failures.append("{0}: missing {1}".format(label, name))
        if not conf_path.is_file():
            continue
        config = _read_arms_conf(conf_path)
        required_keys = [
            key for key in ARMS_KEYS if key != "arm_order"
        ]
        if legacy_baseline_first and "arm_order" not in config:
            if not _legacy_baseline_first_ok(root, config, label, failures):
                continue
            config["arm_order"] = "baseline_first"
        elif "arm_order" not in config:
            failures.append(
                "{0}: arms.conf missing arm_order; rerun both orders or "
                "pass --legacy-baseline-first explicitly".format(label),
            )
            continue
        if not all(key in config for key in required_keys):
            missing = [key for key in required_keys if key not in config]
            failures.append(
                "{0}: arms.conf missing keys {1}".format(label, missing),
            )
            continue
        if common is None:
            common = config
        elif any(
            config.get(key) != common.get(key)
            for key in ARMS_KEYS
            if key != "arm_order"
        ):
            failures.append(
                "{0}: settings/identity mismatch with first result root".format(
                    label,
                ),
            )
        env = _read_environment(root)
        host = env.get("hostname", env.get("expected_node", "unknown"))
        host_order.setdefault(host, []).append(
            {"label": label, "order": config.get("arm_order", "")},
        )
        report = {}
        if report_path.is_file() and log_path.is_file():
            log_lines = _read_lines(log_path)
            non_timing, endpoint_ok, iteration_ok, endpoint_seen, iteration_seen = _ordered_report_evidence(
                log_lines,
            )
            if non_timing:
                failures.append(
                    "{0}: analyze.log contains non-timing FAIL lines".format(label),
                )
            if not endpoint_seen:
                failures.append(
                    "{0}: max_endpoint_norm evidence missing in analyze.log".format(
                        label,
                    ),
                )
            elif not endpoint_ok:
                failures.append(
                    "{0}: max_endpoint_norm != 0 or repeated in analyze.log".format(
                        label,
                    ),
                )
            if not iteration_seen:
                failures.append(
                    "{0}: max_iteration_delta evidence missing in analyze.log".format(
                        label,
                    ),
                )
            elif not iteration_ok:
                failures.append(
                    "{0}: max_iteration_delta != 0 or repeated in analyze.log".format(
                        label,
                    ),
                )
            try:
                report = _read_report(report_path)
            except RuntimeError as err:
                failures.append("{0}: {1}".format(label, err))
                report = {}
        else:
            failures.append("{0}: missing analyze.log or ab_report.csv".format(label))
        root_success = (root / "SUCCESS").is_file()
        root_failed = (root / "FAILED").is_file()
        if root_success and root_failed:
            failures.append(
                "{0}: both root SUCCESS and FAILED present".format(label),
            )
        elif not root_success and not root_failed:
            failures.append("{0}: missing root SUCCESS/FAILED marker".format(label))
        elif root_failed:
            log_lines = _read_lines(log_path) if log_path.is_file() else []
            non_timing, endpoint_ok, iteration_ok, endpoint_seen, iteration_seen = _ordered_report_evidence(
                log_lines,
            )
            gate_failed = "AB_GATE_FAILED" in "".join(log_lines)
            timing_only = gate_failed and not non_timing and \
                          endpoint_seen and iteration_seen and \
                          endpoint_ok and iteration_ok
            if not timing_only:
                failures.append(
                    "{0}: root FAILED without clean timing-only evidence".format(
                        label,
                    ),
                )
        else:
            log_lines = _read_lines(log_path) if log_path.is_file() else []
            if "AB_GATE_PASS" not in "".join(log_lines):
                failures.append(
                    "{0}: root SUCCESS without AB_GATE_PASS evidence".format(
                        label,
                    ),
                )
        for arm in ("baseline", "candidate"):
            marker = root / arm / "FAILED"
            if marker.is_file():
                failures.append("{0}: {1}/FAILED present".format(label, arm))
            success = root / arm / "SUCCESS"
            if not success.is_file():
                failures.append("{0}: {1}/SUCCESS missing".format(label, arm))
        nodes.append({"label": label, "config": config, "report": report})

    for host, entries in host_order.items():
        orders = sorted(entry["order"] for entry in entries)
        if orders != ["baseline_first", "candidate_first"]:
            failures.append(
                "{0}: missing exactly baseline_first and candidate_first "
                "result roots".format(host),
            )

    family_keys = None
    for node in nodes:
        keys = set(node["report"])
        if family_keys is None:
            family_keys = keys
        elif keys != family_keys:
            failures.append(
                "{0}: family set differs from first node".format(node["label"]),
            )

    host_pairs = {}
    for node in nodes:
        order = node["config"].get("arm_order", "")
        env = _read_environment(pathlib.Path(node["label"]))
        host = env.get("hostname", env.get("expected_node", "unknown"))
        host_pairs.setdefault(host, {})[order] = node

    print("host,family,combined_ratio,worst_cv")
    combined = {}
    for host in sorted(host_pairs):
        pair = host_pairs[host]
        orders = sorted(pair)
        if orders != ["baseline_first", "candidate_first"]:
            failures.append(
                "{0}: host missing one arm order; cannot combine".format(host),
            )
            continue
        keys = set(pair["baseline_first"]["report"]) & \
               set(pair["candidate_first"]["report"])
        for key in sorted(keys):
            a = _float(
                pair["baseline_first"]["report"][key].get("ratio"),
                "ratio",
            )
            b = _float(
                pair["candidate_first"]["report"][key].get("ratio"),
                "ratio",
            )
            if not math.isfinite(a) or not math.isfinite(b) or a <= 0.0 or b <= 0.0:
                failures.append(
                    "{0} {1}: paired ratios must be finite and positive "
                    "(got {2:.9g}, {3:.9g})".format(host, ",".join(key), a, b),
                )
                continue
            combined_ratio = math.sqrt(a * b)
            cvs = [
                _float(
                    pair["baseline_first"]["report"][key].get("baseline_cv"),
                    "baseline_cv",
                ),
                _float(
                    pair["baseline_first"]["report"][key].get("candidate_cv"),
                    "candidate_cv",
                ),
                _float(
                    pair["candidate_first"]["report"][key].get("baseline_cv"),
                    "baseline_cv",
                ),
                _float(
                    pair["candidate_first"]["report"][key].get("candidate_cv"),
                    "candidate_cv",
                ),
            ]
            worst_cv = max(cvs)
            if worst_cv >= ORDERED_CV_LIMIT:
                failures.append(
                    "{0} {1}: within-arm worst CV {2:.6f} >= hard limit {3}".format(
                        host,
                        ",".join(key),
                        worst_cv,
                        ORDERED_CV_LIMIT,
                    ),
                )
            elif worst_cv >= ORDERED_WARN_CV_LIMIT:
                print(
                    "WARN {0} {1}: within-arm worst CV {2:.6f} >= warning {3}".format(
                        host,
                        ",".join(key),
                        worst_cv,
                        ORDERED_WARN_CV_LIMIT,
                    ),
                )
            combined.setdefault(key, []).append(combined_ratio)
            print(
                "{0},{1},{2:.6f},{3:.6f}".format(
                    host,
                    ",".join(key),
                    combined_ratio,
                    worst_cv,
                ),
            )

    ordered_ok = True
    for key in sorted(combined):
        values = combined[key]
        if len(values) < ORDERED_MIN_NODES:
            failures.append(
                "order-balanced {0}: only {1} nodes; need at least {2}".format(
                    ",".join(key),
                    len(values),
                    ORDERED_MIN_NODES,
                ),
            )
            continue
        geo = 1.0
        for value in values:
            geo *= value
        geo = geo ** (1.0 / len(values))
        log_values = [math.log(value) for value in values]
        mean_log = sum(log_values) / len(log_values)
        stdev_log = statistics.stdev(log_values) if len(log_values) > 1 else 0.0
        t_value = _t_upper_one_sided(len(values) - 1)
        one_sided_log_upper = mean_log + t_value * stdev_log / math.sqrt(len(values))
        one_sided_upper = math.exp(one_sided_log_upper)
        node_ok = all(value <= RATIO_LIMIT for value in values)
        ok = node_ok and geo <= RATIO_LIMIT and \
             one_sided_upper <= RATIO_LIMIT
        ordered_ok = ordered_ok and ok
        print(
            "ORDERED {0}: node_count={1} geometric={2:.6f} "
            "one_sided_95_upper={3:.6f} gate={4}".format(
                ",".join(key),
                len(values),
                geo,
                one_sided_upper,
                "pass" if ok else "fail",
            ),
        )

    if failures or not ordered_ok:
        for message in failures:
            print("  FAIL:", message)
        print("ORDERED_GATE_FAILED")
        return 1
    print("ORDERED_GATE_PASS")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="*")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ordered", action="store_true")
    parser.add_argument("--legacy-baseline-first", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if not args.result_dirs:
        parser.error("result_dirs is required unless --self-test is used")
    if args.ordered:
        return _run_ordered(
            args.result_dirs,
            legacy_baseline_first=args.legacy_baseline_first,
        )
    return _run(args.result_dirs)


if __name__ == "__main__":
    sys.exit(main())
