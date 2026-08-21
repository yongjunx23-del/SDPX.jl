#!/usr/bin/env python3
"""Run a fresh, validated Float64 MOSEK solve of Task_Low08."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import cvxpy as cp
import mosek
import numpy as np


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
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = load_source_module(source_path)

    started = time.perf_counter()
    data = source.parse_wolfram_task(input_path, 0.8)
    parsed_at = time.perf_counter()
    problem, variables = source.build_problem(data)
    built_at = time.perf_counter()

    parameters = {
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": data.tolerance,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": data.tolerance,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": data.tolerance,
        "MSK_IPAR_NUM_THREADS": args.threads,
    }
    objective = problem.solve(
        solver=cp.MOSEK,
        verbose=not args.quiet,
        mosek_params=parameters,
    )
    solved_at = time.perf_counter()
    if variables.value is None:
        raise RuntimeError(
            f"MOSEK returned {problem.status!r} without a primal solution"
        )

    values = np.asarray(variables.value, dtype=np.float64).ravel()
    diagnostics = source._validate_solution(data, values)
    finished = time.perf_counter()
    result = {
        "task_name": data.name,
        "arithmetic": "Float64",
        "solver": "MOSEK",
        "mosek_version": ".".join(map(str, mosek.Env.getversion())),
        "cvxpy_version": cp.__version__,
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "threads": args.threads,
        "status": problem.status,
        "objective": float(objective),
        "variables": data.variable_count,
        "equalities": int(data.linear_coefficients.shape[0]),
        "psd_blocks": len(data.psd_dimensions),
        "solver_stats": {
            "solver_name": problem.solver_stats.solver_name,
            "solve_time_seconds": problem.solver_stats.solve_time,
            "setup_time_seconds": problem.solver_stats.setup_time,
            "num_iters": problem.solver_stats.num_iters,
        },
        "timing_seconds": {
            "parse": parsed_at - started,
            "model_construction": built_at - parsed_at,
            "canonicalization_and_solve": solved_at - built_at,
            "validation": finished - solved_at,
            "total": finished - started,
        },
        "diagnostics": diagnostics,
        "variable_values": values.tolist(),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {
        "variable_values", "diagnostics"
    }}, indent=2))
    print(json.dumps({"diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
