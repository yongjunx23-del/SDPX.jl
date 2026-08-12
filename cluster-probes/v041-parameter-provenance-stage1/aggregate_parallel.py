#!/usr/bin/env python3
"""Correctness-only aggregate gate for the parallel focused/full Stage-1 run.

Reads the key=value campaign manifest written by submit_parallel.sh and two
completed result roots (focused, full), which may have been pulled from the
login node or any other host.  It writes one row per job and requires SUCCESS,
no FAILED, matching source/environment/campaign identity, the expected
focused/full markers in test.log, and the recorded runtime contract.  Report
status distinguishes a real test failure (FAILED marker), a missing report (no
markers and no timing evidence), and timing noise (no markers plus a
terminated-by-signal /usr/bin/time line).  This gate makes no performance
claims: no ratios, medians, CVs, or timing comparisons.

Stdlib only (Python 3.6 compatible).  No Julia, Pkg, SSH, or qsub.
"""

import argparse
import json
import pathlib
import shutil
import sys
import tempfile

MANIFEST_VERSION = "1"
JOBS = ("focused", "full")
FOCUSED_FILES = ("executed_diagnostics.jl", "auto_planner.jl")

REQUIRED_MANIFEST_KEYS = (
    "manifest_version",
    "campaign_id",
    "focused_node",
    "focused_job_id",
    "focused_root",
    "full_node",
    "full_job_id",
    "full_root",
    "candidate_source",
    "candidate_source_realpath",
    "candidate_source_sha256",
    "candidate_env",
    "sdp_site_env",
    "sdp_depot_path",
    "ppn",
    "julia_threads",
    "solver_threads",
    "blas_threads",
)

# environment.txt key -> campaign manifest key.
ENV_IDENTITY = (
    ("campaign_id", "campaign_id"),
    ("candidate_source", "candidate_source"),
    ("candidate_source_realpath", "candidate_source_realpath"),
    ("candidate_env", "candidate_env"),
    ("candidate_source_sha256", "candidate_source_sha256"),
    ("candidate_source_sha256_expected", "candidate_source_sha256"),
    ("julia_threads", "julia_threads"),
    ("solver_threads", "solver_threads"),
    ("blas_threads", "blas_threads"),
    ("pbs_ppn", "ppn"),
)

JOB_MANIFEST_KEYS = {
    "focused": ("focused_node", "focused_job_id"),
    "full": ("full_node", "full_job_id"),
}

CROSS_ROW_KEYS = (
    "campaign_id",
    "candidate_source",
    "candidate_source_realpath",
    "candidate_source_sha256",
    "candidate_env",
    "julia_threads",
    "solver_threads",
    "blas_threads",
    "pbs_ppn",
    "runtime_contract",
)


def _read_conf(path):
    """Read a key=value configuration file; missing files return {}."""
    config = {}
    if not path.is_file():
        return config
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return config
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            config[key.strip()] = value.strip()
    return config


def _timing_noise(process_time_path):
    """Detect a killed-by-signal /usr/bin/time line; this is the only
    accepted evidence that a missing SUCCESS/FAILED pair is timing noise."""
    if not process_time_path.is_file():
        return False, ""
    try:
        lines = process_time_path.read_text(errors="replace").splitlines()
    except OSError:
        return False, ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Command terminated by signal"):
            return True, stripped
    return False, ""


def _marker_issues(kind, test_log_text):
    """Verify the focused/full marker and runtime-contract lines in test.log."""
    issues = []
    if "RUNTIME_CONTRACT julia=4 plan=4 blas=1" not in test_log_text:
        issues.append(
            "test.log missing RUNTIME_CONTRACT julia=4 plan=4 blas=1",
        )
    if kind == "focused":
        for file in FOCUSED_FILES:
            begin = "FOCUSED_BEGIN {0}".format(file)
            end = "FOCUSED_END {0}".format(file)
            if begin not in test_log_text:
                issues.append("focused test.log missing {0}".format(begin))
            if end not in test_log_text:
                issues.append("focused test.log missing {0}".format(end))
        if "FOCUSED_FAILED" in test_log_text:
            issues.append("focused test.log contains FOCUSED_FAILED")
    else:
        if "FULL_BEGIN Pkg.test" not in test_log_text:
            issues.append("full test.log missing FULL_BEGIN Pkg.test")
        if "FULL_END Pkg.test" not in test_log_text:
            issues.append("full test.log missing FULL_END Pkg.test")
    return issues


def _identity_issues(kind, env, manifest):
    issues = []
    node_key, job_key = JOB_MANIFEST_KEYS[kind]
    expected_node = manifest.get(node_key, "")
    expected_job = manifest.get(job_key, "")
    if not expected_node:
        issues.append("manifest missing {0}".format(node_key))
    if not expected_job:
        issues.append("manifest missing {0}".format(job_key))
    for env_key, manifest_key in ENV_IDENTITY:
        expected = manifest.get(manifest_key, "")
        actual = env.get(env_key, "")
        if not expected:
            issues.append("manifest missing {0}".format(manifest_key))
        elif not actual:
            issues.append("environment.txt missing {0}".format(env_key))
        elif actual != expected:
            issues.append(
                "{0}: got {1!r}, expected {2!r}".format(
                    env_key,
                    actual,
                    expected,
                ),
            )
    for env_key, expected in (
        ("hostname", expected_node),
        ("expected_node", expected_node),
        ("pbs_job_id", expected_job),
    ):
        actual = env.get(env_key, "")
        if not expected:
            continue
        if not actual:
            issues.append("environment.txt missing {0}".format(env_key))
        elif actual != expected:
            issues.append(
                "{0}: got {1!r}, expected {2!r}".format(
                    env_key,
                    actual,
                    expected,
                ),
            )
    actual_hash = env.get("candidate_source_sha256", "")
    expected_hash = env.get("candidate_source_sha256_expected", "")
    if actual_hash and expected_hash and actual_hash != expected_hash:
        issues.append(
            "candidate_source_sha256 and its expected value disagree in environment.txt",
        )
    if env.get("runtime_contract", "") != "ok":
        issues.append(
            "environment.txt does not record runtime_contract=ok (got {0!r})".format(
                env.get("runtime_contract", ""),
            ),
        )
    return issues


def _build_row(kind, root_arg, manifest, failures):
    root = pathlib.Path(root_arg)
    label = kind
    row = {
        "job": kind,
        "root": str(root),
        "node": manifest.get(JOB_MANIFEST_KEYS[kind][0], ""),
        "expected_job_id": manifest.get(JOB_MANIFEST_KEYS[kind][1], ""),
        "pbs_job_id": "",
        "status": "MISSING_REPORT",
        "success_present": False,
        "success_bytes": 0,
        "failed_present": False,
        "test_log_present": False,
        "test_log_bytes": 0,
        "process_time_present": False,
        "environment_present": False,
        "timing_evidence": "",
        "candidate_source_sha256": "",
        "runtime_contract": "",
        "identity_match": False,
        "identity_issues": [],
        "marker_issues": [],
        "identity_values": {},
        "report_complete": False,
    }
    if not root.is_dir():
        failures.append(
            "{0}: result root is not a directory: {1}".format(label, root),
        )
        return row

    env = _read_conf(root / "environment.txt")
    row["environment_present"] = (root / "environment.txt").is_file()
    row["candidate_source_sha256"] = env.get("candidate_source_sha256", "")
    row["pbs_job_id"] = env.get("pbs_job_id", "")
    row["runtime_contract"] = env.get("runtime_contract", "")
    row["identity_values"] = {
        env_key: env.get(env_key, "") for env_key, _ in ENV_IDENTITY
    }
    row["identity_values"]["hostname"] = env.get("hostname", "")
    row["identity_values"]["expected_node"] = env.get("expected_node", "")
    row["identity_values"]["pbs_job_id"] = row["pbs_job_id"]
    row["identity_values"]["runtime_contract"] = row["runtime_contract"]

    success_path = root / "SUCCESS"
    failed_path = root / "FAILED"
    test_log_path = root / "test.log"
    process_path = root / "process.time.txt"
    row["success_present"] = success_path.is_file()
    row["failed_present"] = failed_path.is_file()
    row["test_log_present"] = test_log_path.is_file()
    row["process_time_present"] = process_path.is_file()
    try:
        row["success_bytes"] = (
            success_path.stat().st_size if row["success_present"] else 0
        )
        row["test_log_bytes"] = (
            test_log_path.stat().st_size if row["test_log_present"] else 0
        )
    except OSError:
        row["success_bytes"] = 0
        row["test_log_bytes"] = 0

    if row["failed_present"]:
        row["status"] = "TEST_FAILURE"
    elif row["success_present"] and row["success_bytes"] > 0:
        row["status"] = "PASSED"
    elif row["success_present"]:
        failures.append("{0}: SUCCESS marker is empty".format(label))
        row["status"] = "MISSING_REPORT"
    else:
        noise, evidence = _timing_noise(process_path)
        if noise:
            row["status"] = "TIMING_NOISE"
            row["timing_evidence"] = evidence
        else:
            row["status"] = "MISSING_REPORT"

    test_log_text = ""
    if row["test_log_present"]:
        try:
            test_log_text = test_log_path.read_text(errors="replace")
        except OSError:
            test_log_text = ""
    marker_issues = _marker_issues(kind, test_log_text)
    row["marker_issues"] = marker_issues

    issues = _identity_issues(kind, env, manifest)
    row["identity_issues"] = issues
    row["identity_match"] = not issues

    if not row["environment_present"]:
        failures.append("{0}: missing environment.txt".format(label))
    if issues:
        failures.append(
            "{0}: identity mismatch: {1}".format(label, "; ".join(issues)),
        )
    if marker_issues:
        failures.append(
            "{0}: marker mismatch: {1}".format(
                label,
                "; ".join(marker_issues),
            ),
        )
    if row["status"] != "PASSED":
        failures.append(
            "{0}: status {1} (SUCCESS={2}, FAILED={3})".format(
                label,
                row["status"],
                row["success_present"],
                row["failed_present"],
            ),
        )

    row["report_complete"] = (
        row["environment_present"]
        and row["identity_match"]
        and not marker_issues
        and row["runtime_contract"] == "ok"
        and row["status"] == "PASSED"
        and row["test_log_present"]
        and row["test_log_bytes"] > 0
        and row["process_time_present"]
    )
    if not row["report_complete"]:
        failures.append(
            "{0}: report incomplete (test.log present={1}, process.time.txt "
            "present={2}, identity match={3}, markers={4}, "
            "runtime contract={5!r})".format(
                label,
                row["test_log_present"],
                row["process_time_present"],
                row["identity_match"],
                not marker_issues,
                row["runtime_contract"],
            ),
        )
    return row


def _markdown(summary):
    lines = [
        "# v041 parameter-provenance Stage-1 parallel focused/full gate",
        "",
        "Campaign: {0}".format(summary["campaign_id"]),
        "Gate: {0}".format(summary["gate"]),
        "",
        "| job | node | pbs job id | status | SUCCESS | FAILED | "
        "test.log bytes | report complete | identity | markers | runtime |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} |".format(
                row["job"],
                row["node"],
                row["pbs_job_id"],
                row["status"],
                row["success_present"],
                row["failed_present"],
                row["test_log_bytes"],
                row["report_complete"],
                row["identity_match"],
                not row["marker_issues"],
                row["runtime_contract"],
            ),
        )
    if summary["failures"]:
        lines.append("")
        lines.append("Failures:")
        for failure in summary["failures"]:
            lines.append("- " + failure)
    lines.append("")
    return "\n".join(lines)


def _run(manifest_arg, focused_root, full_root, summary_root_arg, quiet=False):
    failures = []
    manifest_path = pathlib.Path(manifest_arg)
    manifest = _read_conf(manifest_path)
    if not manifest_path.is_file():
        failures.append(
            "campaign manifest not found: {0}".format(manifest_path),
        )
    else:
        missing_keys = [
            key for key in REQUIRED_MANIFEST_KEYS if not manifest.get(key)
        ]
        if missing_keys:
            failures.append(
                "campaign manifest missing keys: {0}".format(
                    ", ".join(missing_keys),
                ),
            )
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            failures.append(
                "unsupported manifest_version: {0}".format(
                    manifest.get("manifest_version"),
                ),
            )
        if (
            manifest.get("focused_node")
            and manifest.get("full_node")
            and manifest["focused_node"] == manifest["full_node"]
        ):
            failures.append(
                "campaign manifest uses the same node for focused and full",
            )
        if (
            manifest.get("focused_job_id")
            and manifest.get("full_job_id")
            and manifest["focused_job_id"] == manifest["full_job_id"]
        ):
            failures.append(
                "campaign manifest uses the same PBS job id for focused and full",
            )
    if focused_root == full_root:
        failures.append("focused and full result roots must be distinct")

    rows = []
    for kind in JOBS:
        root = focused_root if kind == "focused" else full_root
        rows.append(_build_row(kind, root, manifest, failures))

    cross_a = rows[0]["identity_values"]
    cross_b = rows[1]["identity_values"]
    for key in CROSS_ROW_KEYS:
        if cross_a.get(key) and cross_b.get(key) and cross_a[key] != cross_b[key]:
            failures.append(
                "focused and full results disagree on {0}".format(key),
            )

    gate = "PASS" if not failures else "FAIL"
    summary = {
        "tool": "aggregate_parallel.py",
        "gate": gate,
        "campaign_id": manifest.get("campaign_id", ""),
        "manifest": str(manifest_path),
        "focused_root": str(focused_root),
        "full_root": str(full_root),
        "rows": rows,
        "failures": failures,
    }

    summary_root = pathlib.Path(summary_root_arg)
    if summary_root.exists():
        raise SystemExit("summary root already exists: {0}".format(summary_root))
    summary_root.mkdir(parents=True)
    json_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (summary_root / "summary.json").write_text(json_text)
    (summary_root / "summary.md").write_text(_markdown(summary))
    marker = (
        "AGGREGATE_GATE_PASS" if gate == "PASS" else "AGGREGATE_GATE_FAILED"
    )
    (summary_root / marker).write_text("gate={0}\n".format(gate))

    if not quiet:
        for failure in failures:
            print("  FAIL:", failure)
        print(marker)
        print("summary.json: {0}".format(summary_root / "summary.json"))
        print("summary.md: {0}".format(summary_root / "summary.md"))
    return 0 if gate == "PASS" else 1


def _synthetic_markers(kind):
    runtime = "RUNTIME_CONTRACT julia=4 plan=4 blas=1\n"
    if kind == "focused":
        return runtime + (
            "FOCUSED_BEGIN executed_diagnostics.jl\n"
            "FOCUSED_END executed_diagnostics.jl\n"
            "FOCUSED_BEGIN auto_planner.jl\n"
            "FOCUSED_END auto_planner.jl\n"
        )
    return runtime + "FULL_BEGIN Pkg.test\nFULL_END Pkg.test\n"


def _write_synthetic_root(
    root,
    node,
    job_id,
    status,
    hash_hex,
    campaign_id="synthetic",
    env_job=None,
    env_host=None,
    missing_test_log=False,
    missing_process=False,
    missing_env=False,
    kind="focused",
    markers=None,
    runtime_contract="ok",
):
    root = pathlib.Path(root)
    root.mkdir(parents=True)
    if not missing_env:
        env_lines = (
            "campaign_id={0}\n".format(campaign_id)
            + "pbs_job_id={0}\n".format(job_id if env_job is None else env_job)
            + "hostname={0}\n".format(node if env_host is None else env_host)
            + "expected_node={0}\n".format(node)
            + "candidate_source=/synthetic/candidate\n"
            + "candidate_source_realpath=/synthetic/candidate\n"
            + "candidate_env=/synthetic/env\n"
            + "candidate_source_sha256={0}\n".format(hash_hex)
            + "candidate_source_sha256_expected={0}\n".format(hash_hex)
            + "julia_threads=4\n"
            + "solver_threads=4\n"
            + "blas_threads=1\n"
            + "pbs_ppn=5\n"
        )
        if runtime_contract is not None:
            env_lines += "runtime_contract={0}\n".format(runtime_contract)
        (root / "environment.txt").write_text(env_lines)
    if status == "PASSED":
        (root / "SUCCESS").write_text("synthetic\n")
    elif status == "TEST_FAILURE":
        (root / "FAILED").write_text('status = "FAILED"\n')
    if status == "TIMING_NOISE":
        (root / "process.time.txt").write_text(
            "Command terminated by signal 15\n"
            "Elapsed (wall clock) time (m:ss): 0:30\n",
        )
    elif not missing_process:
        (root / "process.time.txt").write_text(
            "Maximum resident set size (kbytes): 1000\n"
            "Elapsed (wall clock) time (m:ss): 0:10\n",
        )
    if not missing_test_log:
        if markers is None:
            markers = _synthetic_markers(kind)
        (root / "test.log").write_text(markers)


def _write_campaign(
    base,
    name,
    focused_status="PASSED",
    full_status="PASSED",
    focused_node="syn-node-focused",
    full_node="syn-node-full",
    focused_job="1000.sugon",
    full_job="1001.sugon",
    focused_env_job=None,
    full_env_job=None,
    focused_env_host=None,
    full_env_host=None,
    focused_hash="a" * 64,
    full_hash=None,
    focused_env_hash=None,
    full_env_hash=None,
    focused_missing_test_log=False,
    full_missing_test_log=False,
    focused_missing_process=False,
    full_missing_process=False,
    focused_missing_env=False,
    full_missing_env=False,
    focused_markers=None,
    full_markers=None,
    focused_runtime_contract="ok",
    full_runtime_contract="ok",
):
    base = pathlib.Path(base)
    base.mkdir(parents=True)
    focused_root = base / "focused"
    full_root = base / "full"
    manifest = base / "manifest.conf"
    if full_hash is None:
        full_hash = focused_hash
    manifest.write_text(
        "manifest_version=1\n"
        "campaign_id={0}\n".format(name)
        + "submitted_at=synthetic\n"
        + "focused_node={0}\n".format(focused_node)
        + "focused_job_id={0}\n".format(focused_job)
        + "focused_root={0}\n".format(focused_root)
        + "full_node={0}\n".format(full_node)
        + "full_job_id={0}\n".format(full_job)
        + "full_root={0}\n".format(full_root)
        + "candidate_source=/synthetic/candidate\n"
        + "candidate_source_realpath=/synthetic/candidate\n"
        + "candidate_source_sha256={0}\n".format(focused_hash)
        + "candidate_env=/synthetic/env\n"
        + "sdp_site_env=/synthetic/site.sh\n"
        + "sdp_depot_path=/synthetic/depot\n"
        + "ppn=5\njulia_threads=4\nsolver_threads=4\nblas_threads=1\n",
    )
    _write_synthetic_root(
        focused_root,
        focused_node,
        focused_job,
        focused_status,
        focused_hash if focused_env_hash is None else focused_env_hash,
        campaign_id=name,
        env_job=focused_env_job,
        env_host=focused_env_host,
        missing_test_log=focused_missing_test_log,
        missing_process=focused_missing_process,
        missing_env=focused_missing_env,
        kind="focused",
        markers=focused_markers,
        runtime_contract=focused_runtime_contract,
    )
    _write_synthetic_root(
        full_root,
        full_node,
        full_job,
        full_status,
        full_hash if full_env_hash is None else full_env_hash,
        campaign_id=name,
        env_job=full_env_job,
        env_host=full_env_host,
        missing_test_log=full_missing_test_log,
        missing_process=full_missing_process,
        missing_env=full_missing_env,
        kind="full",
        markers=full_markers,
        runtime_contract=full_runtime_contract,
    )
    return manifest, focused_root, full_root


def _run_scenario(tmp, name, expect_code, **kwargs):
    base = tmp / name
    manifest, focused, full = _write_campaign(base, name, **kwargs)
    summary = tmp / ("summary_" + name)
    code = _run(manifest, focused, full, summary, quiet=True)
    if code != expect_code:
        print(
            "SELF_TEST {0}: FAIL (exit {1}, expected {2})".format(
                name,
                code,
                expect_code,
            ),
        )
        return False
    print("SELF_TEST {0}: PASS".format(name))
    return True


def _self_test():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="aggregate_parallel_selftest_"))
    try:
        ok = True

        base = tmp / "pass"
        manifest, focused, full = _write_campaign(base, "campaign-pass")
        summary1 = tmp / "summary_pass_1"
        summary2 = tmp / "summary_pass_2"
        code1 = _run(manifest, focused, full, summary1, quiet=True)
        code2 = _run(manifest, focused, full, summary2, quiet=True)
        ok = ok and code1 == 0 and code2 == 0
        json1 = (summary1 / "summary.json").read_bytes()
        json2 = (summary2 / "summary.json").read_bytes()
        ok = ok and json1 == json2
        print(
            "SELF_TEST pass_and_deterministic_json: {0}".format(
                "PASS" if ok else "FAIL",
            ),
        )
        if not ok:
            return 1

        ok = ok and _run_scenario(
            tmp,
            "test_failure",
            1,
            focused_status="TEST_FAILURE",
        )
        ok = ok and _run_scenario(
            tmp,
            "timing_noise",
            1,
            focused_status="TIMING_NOISE",
        )
        ok = ok and _run_scenario(
            tmp,
            "missing_report",
            1,
            focused_status="MISSING_REPORT",
            focused_missing_process=True,
        )
        ok = ok and _run_scenario(
            tmp,
            "hash_mismatch",
            1,
            focused_env_hash="f" * 64,
        )
        ok = ok and _run_scenario(
            tmp,
            "job_id_mismatch",
            1,
            focused_env_job="9999.sugon",
        )
        ok = ok and _run_scenario(
            tmp,
            "missing_test_log",
            1,
            focused_missing_test_log=True,
        )
        ok = ok and _run_scenario(
            tmp,
            "missing_environment",
            1,
            focused_missing_env=True,
        )
        ok = ok and _run_scenario(
            tmp,
            "focused_marker_missing",
            1,
            focused_markers=(
                "FOCUSED_BEGIN executed_diagnostics.jl\n"
                "FOCUSED_END executed_diagnostics.jl\n"
                "FOCUSED_BEGIN auto_planner.jl\n"
            ),
        )
        ok = ok and _run_scenario(
            tmp,
            "full_marker_missing",
            1,
            full_markers="FULL_BEGIN Pkg.test\n",
        )
        ok = ok and _run_scenario(
            tmp,
            "runtime_contract_missing",
            1,
            focused_runtime_contract=None,
        )
        ok = ok and _run_scenario(
            tmp,
            "runtime_line_missing",
            1,
            focused_markers=(
                "FOCUSED_BEGIN executed_diagnostics.jl\n"
                "FOCUSED_END executed_diagnostics.jl\n"
                "FOCUSED_BEGIN auto_planner.jl\n"
                "FOCUSED_END auto_planner.jl\n"
            ),
        )

        if not ok:
            return 1
        print("SELF_TEST_PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the parallel v041 parameter-provenance Stage-1 "
            "focused/full run; writes deterministic summary.json and "
            "summary.md."
        ),
    )
    parser.add_argument(
        "--manifest",
        help="campaign manifest.conf written by submit_parallel.sh",
    )
    parser.add_argument(
        "--summary-root",
        help="fresh directory that receives summary.json/summary.md",
    )
    parser.add_argument("focused_root", nargs="?", help="focused result root")
    parser.add_argument("full_root", nargs="?", help="full Pkg.test result root")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic temp-directory scenarios and exit",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not (
        args.manifest
        and args.summary_root
        and args.focused_root
        and args.full_root
    ):
        parser.error(
            "--manifest, --summary-root, focused_root, and full_root are "
            "required unless --self-test is used",
        )
    return _run(
        args.manifest,
        args.focused_root,
        args.full_root,
        args.summary_root,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
