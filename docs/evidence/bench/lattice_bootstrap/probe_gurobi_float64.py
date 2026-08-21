#!/usr/bin/env python3
"""Probe Gurobi against the exact Task_Low08 PSD-cone model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import cvxpy as cp
import gurobipy as gp


def load_source_module(source_path: Path):
    spec = importlib.util.spec_from_file_location("lattice_low08_source", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = load_source_module(args.source.resolve())
    started = time.perf_counter()
    data = source.parse_wolfram_task(args.input.resolve(), 0.8)
    parsed_at = time.perf_counter()
    problem, _ = source.build_problem(data)
    built_at = time.perf_counter()

    supported = True
    error_type = ""
    error_message = ""
    attempted_at = time.perf_counter()
    try:
        problem.solve(solver=cp.GUROBI, verbose=False)
    except Exception as error:  # The expected CVXPY cone-support rejection.
        supported = False
        error_type = type(error).__name__
        error_message = str(error)
    finished = time.perf_counter()

    result = {
        "task_name": data.name,
        "arithmetic": "Float64",
        "solver": "Gurobi",
        "gurobi_version": ".".join(map(str, gp.gurobi.version())),
        "cvxpy_version": cp.__version__,
        "python_version": platform.python_version(),
        "variables": data.variable_count,
        "equalities": int(data.linear_coefficients.shape[0]),
        "psd_blocks": len(data.psd_dimensions),
        "native_sdp_supported": supported,
        "status": "not_applicable_no_psd_cone_support" if not supported else problem.status,
        "error_type": error_type,
        "error_message": error_message,
        "timing_seconds": {
            "parse": parsed_at - started,
            "model_construction": built_at - parsed_at,
            "compatibility_probe": finished - attempted_at,
            "total": finished - started,
        },
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
