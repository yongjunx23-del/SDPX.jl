#!/usr/bin/env python3
"""Run a repeated, certificate-checked MOSEK baseline from the SDPX binary.

This driver deliberately uses the same compact Task_Low08 input and the same
independent equality columns as the paired SDPX driver.  It reports model
construction, shared equality presolve, CVXPY canonicalization, MOSEK task
construction plus optimization, MOSEK's internal presolve/optimizer times,
solution inversion, validation, and an end-to-end per-run boundary separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import mosek
import numpy as np
import scipy.linalg
import scipy.sparse as sp


MAGIC = b"LATSDP01"


@dataclass
class Block:
    dimension: int
    coefficient_map: sp.csc_matrix
    constant_vector: np.ndarray


@dataclass
class ProblemData:
    variables: int
    equalities: int
    tolerance: float
    blocks: list[Block]
    equality_matrix: sp.csc_matrix
    equality_rhs: np.ndarray


def _read_exact(stream, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise EOFError(f"Expected {size} bytes, received {len(value)}")
    return value


def _read_scalar(stream, code: str):
    return struct.unpack("<" + code, _read_exact(stream, struct.calcsize(code)))[0]


def read_problem(path: Path) -> ProblemData:
    with path.open("rb") as stream:
        if _read_exact(stream, len(MAGIC)) != MAGIC:
            raise ValueError("Unexpected lattice benchmark file format")
        variable_count = int(_read_scalar(stream, "q"))
        equality_count = int(_read_scalar(stream, "q"))
        block_count = int(_read_scalar(stream, "q"))
        tolerance = float(_read_scalar(stream, "d"))
        blocks: list[Block] = []
        for _ in range(block_count):
            dimension = int(_read_scalar(stream, "i"))
            entry_count = int(_read_scalar(stream, "q"))
            rows: list[int] = []
            columns: list[int] = []
            values: list[float] = []
            constant = np.zeros(dimension * dimension, dtype=np.float64)
            for _ in range(entry_count):
                row = int(_read_scalar(stream, "i"))
                column = int(_read_scalar(stream, "i"))
                slot = int(_read_scalar(stream, "i"))
                value = float(_read_scalar(stream, "d"))
                positions = (row + column * dimension,)
                if row != column:
                    positions += (column + row * dimension,)
                if slot == 0:
                    for position in positions:
                        constant[position] += value
                else:
                    for position in positions:
                        rows.append(position)
                        columns.append(slot - 1)
                        values.append(value)
            coefficient_map = sp.csc_matrix(
                (values, (rows, columns)),
                shape=(dimension * dimension, variable_count),
            )
            blocks.append(Block(dimension, coefficient_map, constant))

        equality_nnz = int(_read_scalar(stream, "q"))
        equality_rows = np.empty(equality_nnz, dtype=np.int64)
        equality_columns = np.empty(equality_nnz, dtype=np.int64)
        equality_values = np.empty(equality_nnz, dtype=np.float64)
        for index in range(equality_nnz):
            equality_rows[index] = int(_read_scalar(stream, "i"))
            equality_columns[index] = int(_read_scalar(stream, "i"))
            equality_values[index] = float(_read_scalar(stream, "d"))
        equality_constants = np.frombuffer(
            _read_exact(stream, 8 * equality_count),
            dtype="<f8",
        ).copy()
        if stream.read(1):
            raise ValueError("Unexpected trailing data")
    equality_matrix = sp.csc_matrix(
        (equality_values, (equality_rows, equality_columns)),
        shape=(equality_count, variable_count),
    )
    return ProblemData(
        variables=variable_count,
        equalities=equality_count,
        tolerance=tolerance,
        blocks=blocks,
        equality_matrix=equality_matrix,
        equality_rhs=-equality_constants,
    )


def equality_basis(data: ProblemData):
    dense_transpose = np.asarray(data.equality_matrix.transpose().todense())
    _, upper, permutation = scipy.linalg.qr(
        dense_transpose,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(upper))
    tolerance = (
        max(dense_transpose.shape)
        * np.finfo(np.float64).eps
        * (float(diagonal.max()) if diagonal.size else 0.0)
    )
    rank = int(np.count_nonzero(diagonal > tolerance))
    independent = np.asarray(permutation[:rank], dtype=np.int64)
    dependent = np.asarray(permutation[rank:], dtype=np.int64)
    dependency_residual = 0.0
    if dependent.size:
        coefficients = scipy.linalg.solve_triangular(
            upper[:rank, :rank],
            upper[:rank, rank:],
            lower=False,
            check_finite=False,
        )
        predicted = coefficients.transpose() @ data.equality_rhs[independent]
        dependency_residual = float(
            np.max(np.abs(predicted - data.equality_rhs[dependent]))
        )
    return independent, rank, tolerance, dependency_residual


def build_cvxpy_problem(data: ProblemData, equality_ids: np.ndarray):
    variables = cp.Variable(data.variables)
    constraints: list[cp.Constraint] = [
        data.equality_matrix[equality_ids, :] @ variables
        == data.equality_rhs[equality_ids]
    ]
    psd_constraints: list[cp.Constraint] = []
    for block in data.blocks:
        vector = block.coefficient_map @ variables + block.constant_vector
        matrix = cp.reshape(
            vector,
            (block.dimension, block.dimension),
            order="F",
        )
        constraint = matrix >> 0
        constraints.append(constraint)
        psd_constraints.append(constraint)
    problem = cp.Problem(cp.Minimize(variables[0]), constraints)
    return problem, variables, constraints[0], psd_constraints


def validate_solution(
    data: ProblemData,
    values: np.ndarray,
    equality_dual: np.ndarray,
    psd_duals: list[np.ndarray],
) -> dict[str, Any]:
    equality_residual = data.equality_matrix @ values - data.equality_rhs
    primal_minimum = math.inf
    dual_minimum = math.inf
    dual_residual = np.zeros(data.variables, dtype=np.float64)
    dual_residual[0] = 1.0
    dual_residual += data.equality_matrix.transpose() @ equality_dual
    dual_objective = -float(np.dot(data.equality_rhs, equality_dual))
    primal_block_minima: list[float] = []
    dual_block_minima: list[float] = []
    for block, dual in zip(data.blocks, psd_duals, strict=True):
        primal = np.asarray(
            block.coefficient_map @ values + block.constant_vector
        ).reshape((block.dimension, block.dimension), order="F")
        primal_eigenvalue = float(np.linalg.eigvalsh(primal)[0])
        dual_array = np.asarray(dual, dtype=np.float64)
        dual_eigenvalue = float(np.linalg.eigvalsh(dual_array)[0])
        primal_block_minima.append(primal_eigenvalue)
        dual_block_minima.append(dual_eigenvalue)
        dual_vector = dual_array.reshape(-1, order="F")
        dual_residual -= block.coefficient_map.transpose() @ dual_vector
        dual_objective -= float(np.dot(block.constant_vector, dual_vector))
        primal_minimum = min(primal_minimum, primal_eigenvalue)
        dual_minimum = min(dual_minimum, dual_eigenvalue)
    primal_objective = float(values[0])
    relative_gap = abs(primal_objective - dual_objective) / max(
        1.0,
        abs(primal_objective),
        abs(dual_objective),
    )
    return {
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "relative_gap": relative_gap,
        "maximum_equality_violation": float(np.max(np.abs(equality_residual))),
        "equality_residual_l2": float(np.linalg.norm(equality_residual)),
        "dual_residual_l2": float(np.linalg.norm(dual_residual)),
        "dual_residual_maximum": float(np.max(np.abs(dual_residual))),
        "minimum_primal_psd_eigenvalue": primal_minimum,
        "minimum_dual_psd_eigenvalue": dual_minimum,
        "primal_block_minimum_eigenvalues": primal_block_minima,
        "dual_block_minimum_eigenvalues": dual_block_minima,
    }


def _safe_douinf(task: mosek.Task, item) -> float | None:
    try:
        return float(task.getdouinf(item))
    except Exception:
        return None


def _safe_intinf(task: mosek.Task, item) -> int | None:
    try:
        return int(task.getintinf(item))
    except Exception:
        return None


def task_statistics(task: mosek.Task) -> dict[str, Any]:
    doubles = {
        "optimizer_time": mosek.dinfitem.optimizer_time,
        "presolve_time": mosek.dinfitem.presolve_time,
        "presolve_lindep_time": mosek.dinfitem.presolve_lindep_time,
        "presolve_elimination_time": mosek.dinfitem.presolve_eli_time,
        "interior_point_time": mosek.dinfitem.intpnt_time,
        "ordering_time": mosek.dinfitem.intpnt_order_time,
        "factorization_flops": mosek.dinfitem.intpnt_factor_num_flops,
        "primal_objective_internal": mosek.dinfitem.intpnt_primal_obj,
        "dual_objective_internal": mosek.dinfitem.intpnt_dual_obj,
        "primal_feasibility_internal": mosek.dinfitem.intpnt_primal_feas,
        "dual_feasibility_internal": mosek.dinfitem.intpnt_dual_feas,
    }
    integers = {
        "interior_point_iterations": mosek.iinfitem.intpnt_iter,
        "interior_point_threads": mosek.iinfitem.intpnt_num_threads,
        "factor_dense_dimension": mosek.iinfitem.intpnt_factor_dim_dense,
    }
    return {
        **{name: _safe_douinf(task, item) for name, item in doubles.items()},
        **{name: _safe_intinf(task, item) for name, item in integers.items()},
    }


def run_once(problem, chain, inverse_data, solver_data, parameters):
    started = time.perf_counter()
    raw = chain.solve_via_data(
        problem,
        solver_data,
        warm_start=False,
        verbose=False,
        solver_opts={"mosek_params": dict(parameters)},
    )
    optimized = time.perf_counter()
    statistics = task_statistics(raw["task"])
    solution = chain.invert(raw, inverse_data)
    problem.unpack(solution)
    inverted = time.perf_counter()
    return statistics, optimized - started, inverted - optimized


def machine_metadata(numa_policy: str) -> dict[str, Any]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "node": socket.gethostname(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "numa_policy": numa_policy,
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("basis_output", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--numa-policy", default="default")
    args = parser.parse_args()

    process_started = time.perf_counter()
    read_started = time.perf_counter()
    data = read_problem(args.input.resolve())
    read_seconds = time.perf_counter() - read_started

    presolve_started = time.perf_counter()
    equality_ids, rank, rank_tolerance, dependency_residual = equality_basis(data)
    presolve_seconds = time.perf_counter() - presolve_started
    args.basis_output.parent.mkdir(parents=True, exist_ok=True)
    args.basis_output.write_text(
        "\n".join(str(int(index) + 1) for index in equality_ids) + "\n",
        encoding="utf-8",
    )

    model_started = time.perf_counter()
    problem, variables, equality_constraint, psd_constraints = (
        build_cvxpy_problem(data, equality_ids)
    )
    model_seconds = time.perf_counter() - model_started

    parameters = {
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": args.tolerance,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": args.tolerance,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": args.tolerance,
        "MSK_IPAR_NUM_THREADS": args.threads,
        "MSK_IPAR_LOG": 0,
    }
    canonicalization_started = time.perf_counter()
    solver_data, chain, inverse_data = problem.get_problem_data(
        cp.MOSEK,
        solver_opts={"mosek_params": dict(parameters)},
    )
    canonicalization_seconds = time.perf_counter() - canonicalization_started

    # Full untimed optimizer warm-up. The timed repetitions still build a
    # fresh MOSEK Task from identical canonical data and disable warm starts.
    run_once(problem, chain, inverse_data, solver_data, parameters)

    runs: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        statistics, task_and_optimize_seconds, inversion_seconds = run_once(
            problem,
            chain,
            inverse_data,
            solver_data,
            parameters,
        )
        validation_started = time.perf_counter()
        values = np.asarray(variables.value, dtype=np.float64).ravel()
        equality_dual_reduced = np.asarray(
            equality_constraint.dual_value,
            dtype=np.float64,
        ).ravel()
        equality_dual = np.zeros(data.equalities, dtype=np.float64)
        equality_dual[equality_ids] = equality_dual_reduced
        psd_duals = [
            np.asarray(constraint.dual_value, dtype=np.float64)
            for constraint in psd_constraints
        ]
        certificate = validate_solution(
            data,
            values,
            equality_dual,
            psd_duals,
        )
        validation_seconds = time.perf_counter() - validation_started
        optimizer_seconds = statistics.get("optimizer_time")
        task_overhead = (
            task_and_optimize_seconds - optimizer_seconds
            if optimizer_seconds is not None
            else None
        )
        runs.append(
            {
                "repetition": repetition,
                "status": problem.status,
                "problem_value": float(problem.value),
                "task_construction_and_optimize_seconds": task_and_optimize_seconds,
                "task_construction_overhead_seconds": task_overhead,
                "solution_inversion_seconds": inversion_seconds,
                "validation_seconds": validation_seconds,
                "end_to_end_from_canonical_seconds": (
                    task_and_optimize_seconds
                    + inversion_seconds
                    + validation_seconds
                ),
                "mosek": statistics,
                "certificate": certificate,
            }
        )

    optimizer_times = [
        run["mosek"]["optimizer_time"]
        for run in runs
        if run["mosek"]["optimizer_time"] is not None
    ]
    output = {
        "benchmark": "Task_Low08 matched Float64 baseline",
        "solver": "MOSEK",
        "arithmetic": "Float64",
        "mosek_version": ".".join(map(str, mosek.Env.getversion())),
        "cvxpy_version": cp.__version__,
        "python_version": platform.python_version(),
        "threads": args.threads,
        "tolerance": args.tolerance,
        "repetitions": args.repetitions,
        "warmup": "one complete cold-start solve from cached canonical data",
        "variables": data.variables,
        "equalities": data.equalities,
        "presolved_equalities": rank,
        "psd_blocks": len(data.blocks),
        "equality_rank_tolerance": rank_tolerance,
        "equality_dependency_residual": dependency_residual,
        "timing_seconds": {
            "input_read": read_seconds,
            "shared_equality_presolve": presolve_seconds,
            "model_construction": model_seconds,
            "canonicalization": canonicalization_seconds,
            "median_optimizer": float(np.median(optimizer_times)),
            "median_end_to_end_from_canonical": float(
                np.median(
                    [run["end_to_end_from_canonical_seconds"] for run in runs]
                )
            ),
            "process_total_including_warmup": time.perf_counter() - process_started,
        },
        "allocated_bytes": None,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "machine": machine_metadata(args.numa_policy),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": [run["status"] for run in runs],
                "median_optimizer_seconds": output["timing_seconds"][
                    "median_optimizer"
                ],
                "median_end_to_end_seconds": output["timing_seconds"][
                    "median_end_to_end_from_canonical"
                ],
                "objective": runs[-1]["certificate"]["primal_objective"],
                "relative_gap": runs[-1]["certificate"]["relative_gap"],
                "maximum_equality_violation": runs[-1]["certificate"][
                    "maximum_equality_violation"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
