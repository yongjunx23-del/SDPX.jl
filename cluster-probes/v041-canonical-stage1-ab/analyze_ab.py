#!/usr/bin/env python3
"""Cross-arm gate for the canonical Stage-1 unchanged-hot-path A/B.

Reads RESULT_ROOT/arms.conf and RESULT_ROOT/{baseline,candidate}/results.csv.
Fails unless:
- every row in both arms passes the runner gates and both arm SUCCESS markers
  exist;
- row matrices are identical, each case group has exactly the configured
  repetitions, and input/config/thread/node fields match per row across arms;
- repetition 1 is the per-case warmup: it is fully checked for correctness,
  config, status, certificate, route, endpoint, and workspace equality, but it
  is excluded from timing medians/CV and from the regression ratio;
- each arm's runner `source_sha256` (a subset hash over
  src/bench/Project/Manifest) is non-empty and identical across that arm's
  rows; it is reported but deliberately never compared to the PBS full-tree
  hash (`*_tree_sha256` in arms.conf), which is only an immutable-identity
  gate enforced before the arms run;
- status, certificate, route, and fallback fields are identical per row;
- workspace and iteration counters are identical
  (--no-strict-iterations is diagnosis-only for iterations);
- numeric objective/residual/certificate endpoints agree within arithmetic
  bounds;
- per (arithmetic, family) candidate median total_seconds is at most 10% above
  baseline and both arm CVs are below 5%.

`analyze_ab.py --self-test` runs three synthetic CSV scenarios without touching
Julia: a clean PASS, a PASS where full-tree and subset hashes differ, and a
FAIL exercising input/config/certificate/timing gates.
"""

import argparse
import csv
import math
import pathlib
import statistics
import sys

MISSING_STRINGS = {"", "missing", "nan", "inf", "-inf", "+inf", "infinity",
                   "-infinity"}

EXACT_COLUMNS = (
    "normalized_status",
    "expected_status",
    "certificate_valid",
    "certificate_type",
    "certificate_failures",
    "planned_backend",
    "executed_backend",
    "planned_formulation",
    "executed_formulation",
    "lp_formulation",
    "kkt_backend",
    "gram_kernel",
    "equality_method",
    "solver_algorithm",
    "backend_resolution",
    "fallback",
    "fallback_reason",
    "raw_status",
    "raw_primal_status",
    "raw_dual_status",
)

NUMERIC_COLUMNS = (
    "objective_primal",
    "objective_dual",
    "objective_error",
    "objective_relative_error",
    "relative_gap",
    "primal_residual_original",
    "dual_residual_original",
    "primal_affine_residual_original",
    "dual_affine_residual_original",
    "primal_cone_violation_original",
    "dual_cone_violation_original",
    "primal_residual_scaled_original",
    "dual_residual_scaled_original",
    "equality_backward_error_original",
    "dual_backward_error_original",
    "complementarity_relative",
    "cone_margin_primal",
    "cone_margin_dual",
    "certificate_residual",
)

EXACT_INT_COLUMNS = ("iterations", "restarts", "regularizations",
                     "refinement_steps", "workspace_bytes")

CONFIG_COLUMNS = (
    "input_sha256",
    "max_iterations",
    "time_limit_seconds",
    "tolerance_primal",
    "tolerance_dual",
    "tolerance_gap",
    "resource_class",
    "julia_threads",
    "blas_threads",
    "solver_threads",
    "pbs_ppn",
    "pbs_job_id",
    "cpu_model",
    "julia_version",
    "hostname",
)

ROW_CONFIG = (
    ("resource_class", "resource_class", "str"),
    ("julia_threads", "julia_threads", "int"),
    ("solver_threads", "solver_threads", "int"),
    ("blas_threads", "blas_threads", "int"),
    ("ppn", "pbs_ppn", "int"),
    ("max_iterations", "max_iterations", "int"),
    ("time_limit", "time_limit_seconds", "float"),
)

DEFAULT_ENDPOINT = {
    "float64": (1.0e-9, 1.0e-5),
    "float64x4": (1.0e-22, 1.0e-5),
}
FALLBACK_ENDPOINT = (1.0e-6, 1.0e-4)

TIMING_RATIO_LIMIT = 1.10
CV_LIMIT = 0.05


def _parse_float(value):
    text = str(value).strip().lower()
    if text in MISSING_STRINGS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _endpoint_close(baseline, candidate, atol, rtol):
    b = _parse_float(baseline)
    c = _parse_float(candidate)
    if b is None and c is None:
        return True, 0.0
    if b is None or c is None:
        return False, math.inf
    diff = abs(c - b)
    bound = max(atol, rtol * max(abs(b), abs(c)))
    if bound == 0.0:
        return diff == 0.0, 0.0
    return diff <= bound, diff / bound


def _row_key(row):
    return (
        row["arithmetic"],
        row["family"],
        row["problem"],
        row["severity"],
        int(row["repetition"]),
    )


def _read_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    by_key = {}
    for row in rows:
        key = _row_key(row)
        if key in by_key:
            raise RuntimeError(f"duplicate row key in {path}: {key}")
        by_key[key] = row
    return by_key


def _read_arms_conf(root):
    config = {}
    for line in (root / "arms.conf").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            config[key.strip()] = value.strip()
    return config


def _summarize(values):
    if not values:
        return None
    median = statistics.median(values)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = stdev / mean if mean else math.inf
    return median, mean, stdev, cv


def _config_scalar_equal(expected, actual, kind):
    if kind == "int":
        return _parse_int(expected) == _parse_int(actual)
    if kind == "float":
        return _parse_float(expected) == _parse_float(actual)
    return str(expected) == str(actual)


def _max_rss_from_time(path):
    try:
        for line in path.read_text(errors="replace").splitlines():
            if "Maximum resident set size" in line and ":" in line:
                token = line.split(":", 1)[1].strip().split()[0]
                return float(token)
    except (OSError, ValueError):
        pass
    return None


def _row_max_rss(by_key):
    values = []
    for row in by_key.values():
        value = _parse_float(row.get("process_peak_rss_bytes"))
        if value is not None and value > 0:
            values.append(value)
    return max(values) if values else None


def analyze_root(root, opts):
    root = pathlib.Path(root).resolve()
    config = _read_arms_conf(root)
    try:
        repetitions = int(config["repetitions"])
    except (KeyError, ValueError) as err:
        raise SystemExit(f"arms.conf repetitions invalid: {err}") from err

    baseline = _read_rows(root / "baseline" / "results.csv")
    candidate = _read_rows(root / "candidate" / "results.csv")
    failures = []

    for arm in ("baseline", "candidate"):
        arm_root = root / arm
        if not (arm_root / "SUCCESS").is_file():
            failures.append(f"{arm}/SUCCESS missing; runner gates did not pass")
        if (arm_root / "FAILED").is_file():
            failures.append(f"{arm}/FAILED present")
        if not (root / f"{arm}.process.time.txt").is_file():
            failures.append(f"{arm}.process.time.txt missing")

    if set(baseline) != set(candidate):
        failures.append(
            "row matrix mismatch baseline vs candidate: "
            f"only_baseline={sorted(set(baseline) - set(candidate))} "
            f"only_candidate={sorted(set(candidate) - set(baseline))}",
        )
    common = sorted(set(baseline) & set(candidate))

    for arm, by_key in (("baseline", baseline), ("candidate", candidate)):
        groups = {}
        for key in by_key:
            groups.setdefault(key[:4], []).append(key[4])
        for group, reps in sorted(groups.items()):
            if sorted(reps) != list(range(1, repetitions + 1)):
                failures.append(
                    f"{arm} {group} has repetitions {sorted(reps)}; "
                    f"expected 1..{repetitions}",
                )

    max_endpoint_norm = 0.0
    max_iteration_delta = 0
    for key in common:
        b = baseline[key]
        c = candidate[key]
        for arm, row in (("baseline", b), ("candidate", c)):
            if str(row.get("gate_pass", "")).strip().lower() != "true":
                failures.append(
                    f"{arm} {key}: gate_pass false "
                    f"({row.get('gate_failures')})",
                )
            if str(row.get("candidate_pathof_match", "")).strip().lower() != "true":
                failures.append(f"{arm} {key}: candidate_pathof_match false")
            if str(row.get("exception", "")).strip():
                failures.append(
                    f"{arm} {key}: exception {row.get('exception')}",
                )
            for config_key, row_column, kind in ROW_CONFIG:
                if not _config_scalar_equal(
                    config.get(config_key),
                    row.get(row_column),
                    kind,
                ):
                    failures.append(
                        f"{arm} {key}: {row_column} "
                        f"{row.get(row_column)!r} != arms.conf "
                        f"{config_key}={config.get(config_key)!r}",
                    )
            if key[0] not in config.get("arithmetic", "").split(","):
                failures.append(
                    f"{arm} {key}: arithmetic {key[0]} not in arms.conf "
                    f"arithmetic={config.get('arithmetic')!r}",
                )
        if b.get("sdpx_git_sha") != config.get("baseline_commit"):
            failures.append(f"baseline {key}: sdpx_git_sha mismatch")
        if c.get("sdpx_git_sha") != config.get("candidate_commit"):
            failures.append(f"candidate {key}: sdpx_git_sha mismatch")
        if b.get("archive_sha") != config.get("baseline_archive_sha"):
            failures.append(
                f"baseline {key}: archive_sha {b.get('archive_sha')!r} "
                f"!= arms.conf baseline_archive_sha="
                f"{config.get('baseline_archive_sha')!r}",
            )
        if c.get("archive_sha") != config.get("candidate_archive_sha"):
            failures.append(
                f"candidate {key}: archive_sha {c.get('archive_sha')!r} "
                f"!= arms.conf candidate_archive_sha="
                f"{config.get('candidate_archive_sha')!r}",
            )
        if not str(b.get("archive_sha", "")).strip() or \
           not str(c.get("archive_sha", "")).strip():
            failures.append(f"{key}: archive_sha missing in a row")

        for column in CONFIG_COLUMNS:
            if b.get(column) != c.get(column):
                failures.append(f"{key}: config column {column} differs")
        for column in EXACT_COLUMNS:
            if b.get(column) != c.get(column):
                failures.append(f"{key}: endpoint column {column} differs")

        for column in EXACT_INT_COLUMNS:
            bi = _parse_int(b.get(column))
            ci = _parse_int(c.get(column))
            if column == "iterations" and bi is not None and ci is not None:
                max_iteration_delta = max(max_iteration_delta, abs(bi - ci))
            if not opts.no_strict_iterations and \
               (bi is None or ci is None or bi != ci):
                failures.append(
                    f"{key}: {column} differs "
                    f"(baseline={b.get(column)!r} candidate={c.get(column)!r})",
                )

        atol, rtol = DEFAULT_ENDPOINT.get(key[0], FALLBACK_ENDPOINT)
        if opts.endpoint_atol is not None:
            atol = opts.endpoint_atol
        if opts.endpoint_rtol is not None:
            rtol = opts.endpoint_rtol
        for column in NUMERIC_COLUMNS:
            ok, norm = _endpoint_close(b.get(column), c.get(column), atol, rtol)
            max_endpoint_norm = max(max_endpoint_norm, norm)
            if not ok:
                failures.append(
                    f"{key}: {column} endpoint differs beyond "
                    f"atol={atol:g} rtol={rtol:g} "
                    f"(baseline={b.get(column)!r} candidate={c.get(column)!r})",
                )

    subset_hashes = {}
    for arm, by_key in (("baseline", baseline), ("candidate", candidate)):
        values = sorted({
            str(row.get("source_sha256", "")).strip()
            for row in by_key.values()
        })
        if not values or values == [""]:
            failures.append(f"{arm}: source_sha256 missing or empty in all rows")
            subset_hashes[arm] = ""
        elif len(values) > 1:
            failures.append(
                f"{arm}: source_sha256 differs across rows: {values}",
            )
            subset_hashes[arm] = ""
        else:
            subset_hashes[arm] = values[0]

    family_totals_b, family_totals_c = {}, {}
    case_totals_b, case_totals_c = {}, {}
    for arm, totals, case_totals in (
        ("baseline", family_totals_b, case_totals_b),
        ("candidate", family_totals_c, case_totals_c),
    ):
        for key, row in (baseline if arm == "baseline" else candidate).items():
            total = _parse_float(row.get("total_seconds"))
            if total is None or not math.isfinite(total):
                failures.append(f"{arm} {key}: non-finite total_seconds")
                continue
            if key[4] == 1:
                continue
            family_key = (key[0], key[1])
            case_key = (key[0], key[1], key[2])
            totals.setdefault(family_key, {}).setdefault(key[4], 0.0)
            totals[family_key][key[4]] += total
            case_totals.setdefault(case_key, {})[key[4]] = total

    report_rows = []
    family_keys = sorted(set(family_totals_b) | set(family_totals_c))
    for category in family_keys:
        incomplete = False
        for arm, totals in (("baseline", family_totals_b),
                            ("candidate", family_totals_c)):
            by_rep = totals.get(category)
            if by_rep is None or \
               sorted(by_rep) != list(range(2, repetitions + 1)):
                failures.append(
                    f"{arm} timing repetitions missing for {category}: "
                    f"{sorted(by_rep) if by_rep else []}",
                )
                incomplete = True
        if incomplete:
            continue
        by_rep_b = family_totals_b[category]
        by_rep_c = family_totals_c[category]
        values_b = [by_rep_b[r] for r in range(2, repetitions + 1)]
        values_c = [by_rep_c[r] for r in range(2, repetitions + 1)]
        summary_b = _summarize(values_b)
        summary_c = _summarize(values_c)
        if summary_b is None or summary_c is None:
            failures.append(f"timing summary missing for {category}")
            continue
        median_b, mean_b, _, cv_b = summary_b
        median_c, mean_c, _, cv_c = summary_c
        ratio = median_c / median_b if median_b else math.inf
        gate_ok = ratio <= TIMING_RATIO_LIMIT and cv_b < CV_LIMIT and \
                  cv_c < CV_LIMIT
        if not gate_ok:
            failures.append(
                f"timing gate failed for {category}: ratio={ratio:.6f} "
                f"baseline_cv={cv_b:.6f} candidate_cv={cv_c:.6f}",
            )
        report_rows.append(
            ("family", category[0], category[1], "", median_b, median_c,
             ratio, cv_b, cv_c, "pass" if gate_ok else "fail"),
        )

    case_keys = sorted(set(case_totals_b) | set(case_totals_c))
    for category in case_keys:
        if category not in case_totals_b or \
           category not in case_totals_c:
            failures.append(f"case timing missing for {category}")
            continue
        by_rep_b = case_totals_b[category]
        by_rep_c = case_totals_c[category]
        if sorted(by_rep_b) != list(range(2, repetitions + 1)) or \
           sorted(by_rep_c) != list(range(2, repetitions + 1)):
            failures.append(f"case timing repetitions missing for {category}")
            continue
        values_b = [by_rep_b[r] for r in range(2, repetitions + 1)]
        values_c = [by_rep_c[r] for r in range(2, repetitions + 1)]
        summary_b = _summarize(values_b)
        summary_c = _summarize(values_c)
        if summary_b is None or summary_c is None:
            continue
        median_b, _, _, cv_b = summary_b
        median_c, _, _, cv_c = summary_c
        ratio = median_c / median_b if median_b else math.inf
        report_rows.append(
            ("case", category[0], category[1], category[2], median_b, median_c,
             ratio, cv_b, cv_c, "diagnostic"),
        )

    provenance = {
        "baseline_source_sha256_subset": subset_hashes.get("baseline", ""),
        "candidate_source_sha256_subset": subset_hashes.get("candidate", ""),
        "baseline_tree_sha256": config.get("baseline_tree_sha256", ""),
        "candidate_tree_sha256": config.get("candidate_tree_sha256", ""),
        "runner_tree_sha256": config.get("runner_tree_sha256", ""),
        "baseline_commit": config.get("baseline_commit", ""),
        "candidate_commit": config.get("candidate_commit", ""),
        "baseline_archive_sha": config.get("baseline_archive_sha", ""),
        "candidate_archive_sha": config.get("candidate_archive_sha", ""),
        "baseline_source_path": config.get("baseline_source", ""),
        "candidate_source_path": config.get("candidate_source", ""),
        "runner_source_path": config.get("runner_source", ""),
    }
    with (root / "ab_provenance.txt").open("w") as stream:
        for key in sorted(provenance):
            stream.write(f"{key}={provenance[key]}\n")

    with (root / "ab_report.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "level", "arithmetic", "family", "problem", "baseline_median",
            "candidate_median", "ratio", "baseline_cv", "candidate_cv",
            "gate",
        ])
        for row in report_rows:
            writer.writerow(row)

    print("level,arithmetic,family,problem,baseline_median,candidate_median,"
          "ratio,baseline_cv,candidate_cv,gate")
    for row in report_rows:
        print(",".join(
            str(value) if not isinstance(value, float) else f"{value:.9g}"
            for value in row
        ))
    for key in sorted(provenance):
        print(f"{key}={provenance[key]}")
    print(
        f"baseline_rss_time={_max_rss_from_time(root / 'baseline.process.time.txt')!r} "
        f"candidate_rss_time={_max_rss_from_time(root / 'candidate.process.time.txt')!r}",
    )
    print(
        f"baseline_rss_rows={_row_max_rss(baseline)!r} "
        f"candidate_rss_rows={_row_max_rss(candidate)!r}",
    )
    print(f"max_endpoint_norm={max_endpoint_norm:.9g}")
    print(f"max_iteration_delta={max_iteration_delta}")

    if failures:
        print("AB_GATE_FAILED")
        for message in failures:
            print("  FAIL:", message)
        return 1
    print("AB_GATE_PASS")
    return 0


def _write_synthetic_run(root, *, corrupt, tree_differs):
    """Write a self-contained synthetic A/B result without Julia."""
    root = pathlib.Path(root)
    for arm in ("baseline", "candidate"):
        (root / arm).mkdir(parents=True)
    (root / "SUCCESS").touch()
    (root / "baseline" / "SUCCESS").touch()
    (root / "candidate" / "SUCCESS").touch()
    (root / "baseline.process.time.txt").write_text(
        "Maximum resident set size (kbytes): 100000\n",
    )
    (root / "candidate.process.time.txt").write_text(
        "Maximum resident set size (kbytes): 100000\n",
    )

    if tree_differs:
        baseline_tree = "a" * 64
        candidate_tree = "b" * 64
    else:
        baseline_tree = "1" * 64
        candidate_tree = "2" * 64
    subset_baseline = "f" * 64
    subset_candidate = "e" * 64
    arms_conf = (
        "repetitions=4\n"
        "arithmetic=float64\n"
        "case_filter=lp_row_scaling,sdp_hilbert\n"
        f"baseline_source=/synthetic/baseline\n"
        f"baseline_commit={'9' * 40}\n"
        f"baseline_archive_sha={'c' * 64}\n"
        f"baseline_tree_sha256={baseline_tree}\n"
        f"candidate_source=/synthetic/candidate\n"
        f"candidate_commit={'8' * 40}\n"
        f"candidate_archive_sha={'d' * 64}\n"
        f"candidate_tree_sha256={candidate_tree}\n"
        "runner_source=/synthetic/runner\n"
        f"runner_tree_sha256={'7' * 64}\n"
        "resource_class=regular\n"
        "julia_threads=4\n"
        "solver_threads=4\n"
        "blas_threads=1\n"
        "ppn=5\n"
        "max_iterations=300\n"
        "time_limit=900.0\n"
        "warmup=true\n"
    )
    (root / "arms.conf").write_text(arms_conf)

    def row(arm, family, problem, rep, total):
        git_sha = "9" * 40 if arm == "baseline" else "8" * 40
        archive = "c" * 64 if arm == "baseline" else "d" * 64
        subset = subset_baseline if arm == "baseline" else subset_candidate
        input_sha = "5" * 64
        certificate_valid = "true"
        tolerance_primal = "1.0e-8"
        if corrupt and arm == "candidate":
            input_sha = "6" * 64
            certificate_valid = "false"
            tolerance_primal = "2.0e-8"
        return {
            "run_id": f"{arm}-{family}-{problem}-r{rep}",
            "arithmetic": "float64",
            "family": family,
            "problem": problem,
            "severity": "synthetic",
            "repetition": str(rep),
            "gate_pass": "true",
            "gate_failures": "",
            "candidate_pathof_match": "true",
            "exception": "",
            "sdpx_git_sha": git_sha,
            "source_sha256": subset,
            "archive_sha": archive,
            "input_sha256": input_sha,
            "max_iterations": "300",
            "time_limit_seconds": "900.0",
            "tolerance_primal": tolerance_primal,
            "tolerance_dual": "1.0e-8",
            "tolerance_gap": "1.0e-8",
            "resource_class": "regular",
            "julia_threads": "4",
            "blas_threads": "1",
            "solver_threads": "4",
            "pbs_ppn": "5",
            "pbs_job_id": "12345",
            "cpu_model": "synthetic cpu",
            "julia_version": "1.10.0",
            "hostname": "node1",
            "normalized_status": "certified_optimal",
            "expected_status": "optimal",
            "certificate_valid": certificate_valid,
            "certificate_type": "optimal_certificate",
            "certificate_failures": "",
            "planned_backend": "dense_cholesky",
            "executed_backend": "dense_cholesky",
            "planned_formulation": "sdp_native",
            "executed_formulation": "sdp_native",
            "lp_formulation": "not_applicable",
            "kkt_backend": "dense_cholesky",
            "gram_kernel": "dense",
            "equality_method": "cholesky",
            "solver_algorithm": "sdp_primal_dual",
            "backend_resolution": "planned",
            "fallback": "false",
            "fallback_reason": "none",
            "raw_status": "Optimal",
            "raw_primal_status": "feasible_point",
            "raw_dual_status": "feasible_point",
            "iterations": "10",
            "restarts": "0",
            "regularizations": "0",
            "refinement_steps": "0",
            "workspace_bytes": "4096",
            "objective_primal": "1.0",
            "objective_dual": "1.0",
            "objective_error": "0.0",
            "objective_relative_error": "0.0",
            "relative_gap": "1.0e-12",
            "primal_residual_original": "1.0e-14",
            "dual_residual_original": "1.0e-14",
            "primal_affine_residual_original": "1.0e-14",
            "dual_affine_residual_original": "1.0e-14",
            "primal_cone_violation_original": "-1.0e-15",
            "dual_cone_violation_original": "-1.0e-15",
            "primal_residual_scaled_original": "1.0e-14",
            "dual_residual_scaled_original": "1.0e-14",
            "equality_backward_error_original": "1.0e-15",
            "dual_backward_error_original": "1.0e-15",
            "complementarity_relative": "1.0e-12",
            "cone_margin_primal": "1.0e-3",
            "cone_margin_dual": "1.0e-3",
            "certificate_residual": "1.0e-12",
            "total_seconds": str(total),
            "process_peak_rss_bytes": "1000000",
        }

    for arm in ("baseline", "candidate"):
        rows = []
        families = {
            # Repetition 1 is an intentionally abnormal per-case warmup time;
            # the analyzer must discard it from timing while still checking
            # its correctness/config/certificate/workspace fields.
            "lp": ("lp_row_scaling", [1000.0, 10.0, 10.2, 10.1]),
            "sdp": ("sdp_hilbert", [500.0, 5.0, 5.1, 5.05]),
        }
        for family, (problem, times) in families.items():
            for rep, total in enumerate(times, start=1):
                candidate_total = total
                if corrupt:
                    candidate_total = total * 1.5
                value = candidate_total if arm == "candidate" else total
                rows.append(row(arm, family, problem, rep, value))
        with (root / arm / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _self_test():
    import shutil
    import tempfile

    from types import SimpleNamespace

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stage1_ab_selftest_"))
    try:
        scenarios = (
            ("pass", False, False),
            ("tree_subset_ok", False, True),
            ("fail_gates", True, False),
        )
        wanted = (0, 0, 1)
        opts = SimpleNamespace(
            endpoint_atol=None,
            endpoint_rtol=None,
            no_strict_iterations=False,
        )
        ok = True
        for (name, corrupt, tree_differs), expected in zip(scenarios, wanted):
            root = tmp / name
            _write_synthetic_run(root, corrupt=corrupt, tree_differs=tree_differs)
            code = analyze_root(root, opts)
            status = "PASS" if code == expected else "FAIL"
            if code != expected:
                ok = False
            print(f"SELF_TEST {name}: exit={code} expected={expected} {status}")
        if ok:
            print("SELF_TEST_PASS")
            return 0
        print("SELF_TEST_FAILED")
        return 1
    finally:
        shutil.rmtree(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", nargs="?")
    parser.add_argument("--endpoint-atol", type=float)
    parser.add_argument("--endpoint-rtol", type=float)
    parser.add_argument("--no-strict-iterations", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    from types import SimpleNamespace

    opts = SimpleNamespace(
        endpoint_atol=args.endpoint_atol,
        endpoint_rtol=args.endpoint_rtol,
        no_strict_iterations=args.no_strict_iterations,
    )
    if args.self_test:
        return _self_test()
    if not args.result_root:
        parser.error("result_root is required unless --self-test is used")
    return analyze_root(pathlib.Path(args.result_root).resolve(), opts)


if __name__ == "__main__":
    sys.exit(main())
