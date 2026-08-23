#!/usr/bin/env python3

"""Summarize finite-support LP key/value logs without third-party packages."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = (
    "build_seconds",
    "solve_seconds",
    "validation_seconds",
    "allocated_bytes",
    "peak_rss_bytes",
    "iterations",
    "primal_objective",
    "dual_objective",
    "relative_gap",
    "reported_primal_residual",
    "reported_dual_residual",
    "equality_absolute",
    "equality_normalized",
    "nonnegative_violation",
    "minimum_scalar_psd",
    "timing_gram_assembly",
    "timing_kkt_factorization",
    "timing_predictor_corrector",
)


def parse_log(path: Path) -> dict[str, str]:
    record: dict[str, str] = {"log": path.name}
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.replace("_", "").isalnum():
            record[key] = value.strip()
    return record


def finite_number(record: dict[str, str], field: str) -> float | None:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = sorted({field for record in records for field in record})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    logs = sorted(args.result_directory.glob("*.log"))
    records = [parse_log(path) for path in logs]
    records = [record for record in records if "arithmetic" in record]
    if not records:
        raise SystemExit("no completed benchmark records found")
    write_csv(args.result_directory / "runs.csv", records)

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    group_fields = (
        "arithmetic",
        "variables",
        "equalities",
        "direction",
        "julia_threads_requested",
        "blas_threads",
        "executed_kkt",
        "executed_gram",
    )
    for record in records:
        groups[tuple(record.get(field, "") for field in group_fields)].append(record)

    summaries: list[dict[str, object]] = []
    for key, members in sorted(groups.items()):
        summary: dict[str, object] = dict(zip(group_fields, key))
        summary["samples"] = len(members)
        summary["statuses"] = ";".join(sorted({row.get("status", "") for row in members}))
        summary["all_certificates_valid"] = all(
            row.get("certificate_valid") == "true" for row in members
        )
        for field in NUMERIC_FIELDS:
            values = [
                value
                for member in members
                if (value := finite_number(member, field)) is not None
            ]
            if values:
                summary[f"median_{field}"] = statistics.median(values)
                summary[f"minimum_{field}"] = min(values)
                summary[f"maximum_{field}"] = max(values)
        summaries.append(summary)
    write_csv(args.result_directory / "summary.csv", summaries)
    print(f"parsed {len(records)} runs into {len(summaries)} configurations")


if __name__ == "__main__":
    main()
