#!/usr/bin/env python3
"""Time a MOSEK Task_Low08 solve on this machine.

The repository's full MOSEK driver also unpacks dual variables, which fails on
CVXPY 1.9 + MOSEK 11.2 with a KeyError inside CVXPY's cone-stuffing inverse
(a version skew against the MOSEK 11.1.3 used for the archived report). The
primal solve itself is unaffected, so this harness times the solve and reports
the objective and primal validation, tolerating that unpacking failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("lattice_low08_source", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = load_source(args.source.resolve())
    data = source.parse_wolfram_task(args.input.resolve(), 0.8)
    problem, variables = source.build_problem(data)

    parameters = {
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": data.tolerance,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": data.tolerance,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": data.tolerance,
        "MSK_IPAR_NUM_THREADS": args.threads,
    }

    started = time.perf_counter()
    try:
        problem.solve(solver=cp.MOSEK, verbose=False, mosek_params=parameters)
    except KeyError:
        # Dual unpacking only; the primal solution is already attached.
        pass
    elapsed = time.perf_counter() - started

    stats = problem.solver_stats
    values = np.asarray(variables.value, dtype=np.float64).ravel()
    diagnostics = source._validate_solution(data, values)
    objective = float(values[0])

    record = {
        "tolerance": data.tolerance,
        "threads": args.threads,
        "wall_seconds": elapsed,
        "solver_seconds": getattr(stats, "solve_time", None),
        "iterations": getattr(stats, "num_iters", None),
        "status": problem.status,
        "objective_w0": objective,
        "diagnostics": {
            key: (float(value) if np.isscalar(value) else None)
            for key, value in vars(diagnostics).items()
        }
        if hasattr(diagnostics, "__dict__")
        else str(diagnostics),
    }
    print(json.dumps(record, indent=2, default=str))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
