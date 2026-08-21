#!/usr/bin/env python3
"""Run the lattice-bootstrap Task_Low08 SDP with CVXPY/Clarabel Float64."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np


def load_source_module(source_directory: Path):
    source = source_directory / "solve_task_low08.py"
    spec = importlib.util.spec_from_file_location("lattice_low08_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    args = parser.parse_args()

    source_directory = args.source_directory.resolve()
    source_module = load_source_module(source_directory)
    parsed_start = time.perf_counter()
    data = source_module.parse_wolfram_task(
        source_directory / "Task_Low08.wl",
        0.8,
    )
    parsed_end = time.perf_counter()
    problem, variables = source_module.build_problem(data)
    built_end = time.perf_counter()
    objective = problem.solve(
        solver=cp.CLARABEL,
        verbose=False,
        max_iter=args.max_iterations,
        time_limit=args.time_limit,
        tol_gap_abs=data.tolerance,
        tol_gap_rel=data.tolerance,
        tol_feas=data.tolerance,
        reduced_tol_gap_abs=data.tolerance,
        reduced_tol_gap_rel=data.tolerance,
        reduced_tol_feas=data.tolerance,
        chordal_decomposition_enable=True,
        presolve_enable=True,
    )
    solved_end = time.perf_counter()
    if variables.value is None:
        raise RuntimeError(f"Clarabel returned {problem.status} without a solution")
    variable_values = np.asarray(variables.value, dtype=np.float64).ravel()
    diagnostics = source_module._validate_solution(data, variable_values)
    validated_end = time.perf_counter()

    result = {
        "task_name": data.name,
        "arithmetic": "Float64",
        "solver": "Clarabel",
        "cvxpy_version": cp.__version__,
        "status": problem.status,
        "objective": float(objective),
        "solver_stats": {
            "solver_name": problem.solver_stats.solver_name,
            "solve_time_seconds": problem.solver_stats.solve_time,
            "num_iters": problem.solver_stats.num_iters,
        },
        "timing_seconds": {
            "parse": parsed_end - parsed_start,
            "model_construction": built_end - parsed_end,
            "solve_and_canonicalization": solved_end - built_end,
            "validation": validated_end - solved_end,
            "total": validated_end - parsed_start,
        },
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "objective": result["objective"],
        "solver_stats": result["solver_stats"],
        "timing_seconds": result["timing_seconds"],
        "max_absolute_linear_residual": diagnostics[
            "max_absolute_linear_residual"
        ],
        "minimum_psd_eigenvalue": diagnostics["minimum_psd_eigenvalue"],
    }, indent=2))


if __name__ == "__main__":
    main()
