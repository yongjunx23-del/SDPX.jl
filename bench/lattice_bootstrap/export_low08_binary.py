#!/usr/bin/env python3
"""Export Task_Low08 to a compact Float64 binary file for Julia benchmarks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np


MAGIC = b"LATSDP01"
ENTRY_DTYPE = np.dtype(
    [
        ("row", "<i4"),
        ("column", "<i4"),
        ("slot", "<i4"),
        ("value", "<f8"),
    ],
    align=False,
)
LINEAR_ENTRY_DTYPE = np.dtype(
    [
        ("row", "<i4"),
        ("column", "<i4"),
        ("value", "<f8"),
    ],
    align=False,
)


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
    args = parser.parse_args()

    source_directory = args.source_directory.resolve()
    source_module = load_source_module(source_directory)
    data = source_module.parse_wolfram_task(
        source_directory / "Task_Low08.wl",
        0.8,
    )
    equality_matrix = data.linear_coefficients.tocoo()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    block_summaries: list[dict[str, int | float]] = []
    upper_coefficient_nnz = 0

    with args.output.open("wb") as output:
        output.write(MAGIC)
        output.write(
            struct.pack(
                "<qqqd",
                data.variable_count,
                equality_matrix.shape[0],
                len(data.psd_dimensions),
                data.tolerance,
            )
        )

        for dimension, coefficients in zip(
            data.psd_dimensions,
            data.psd_coefficients,
            strict=True,
        ):
            coordinate = coefficients.tocoo()
            matrix_rows = coordinate.row // dimension
            matrix_columns = coordinate.row % dimension
            upper_mask = matrix_rows <= matrix_columns

            rows = matrix_rows[upper_mask].astype("<i4", copy=False)
            columns = matrix_columns[upper_mask].astype("<i4", copy=False)
            slots = coordinate.col[upper_mask].astype("<i4", copy=False)
            values = coordinate.data[upper_mask].astype("<f8", copy=False)
            records = np.empty(values.size, dtype=ENTRY_DTYPE)
            records["row"] = rows
            records["column"] = columns
            records["slot"] = slots
            records["value"] = values

            output.write(struct.pack("<iq", dimension, records.size))
            records.tofile(output)
            upper_coefficient_nnz += int(records.size)
            structural_upper_nnz = int(
                np.unique(rows.astype(np.int64) * dimension + columns).size
            )
            full_upper_dimension = dimension * (dimension + 1) // 2
            block_summaries.append(
                {
                    "dimension": dimension,
                    "coefficient_nnz": int(records.size),
                    "structural_upper_nnz": structural_upper_nnz,
                    "full_upper_dimension": full_upper_dimension,
                    "structural_density": (
                        structural_upper_nnz / full_upper_dimension
                    ),
                }
            )

        linear_records = np.empty(
            equality_matrix.nnz,
            dtype=LINEAR_ENTRY_DTYPE,
        )
        linear_records["row"] = equality_matrix.row.astype("<i4", copy=False)
        linear_records["column"] = equality_matrix.col.astype("<i4", copy=False)
        linear_records["value"] = equality_matrix.data.astype("<f8", copy=False)
        output.write(struct.pack("<q", linear_records.size))
        linear_records.tofile(output)
        np.asarray(data.linear_constants, dtype="<f8").tofile(output)

    structural_upper_nnz = sum(
        int(block["structural_upper_nnz"]) for block in block_summaries
    )
    full_upper_dimension = sum(
        int(block["full_upper_dimension"]) for block in block_summaries
    )
    print(
        json.dumps(
            {
                "task_name": data.name,
                "arithmetic": "Float64",
                "variables": data.variable_count,
                "equalities": equality_matrix.shape[0],
                "equality_nnz": equality_matrix.nnz,
                "psd_blocks": len(data.psd_dimensions),
                "upper_coefficient_nnz": upper_coefficient_nnz,
                "structural_upper_nnz": structural_upper_nnz,
                "full_upper_dimension": full_upper_dimension,
                "structural_density": (
                    structural_upper_nnz / full_upper_dimension
                ),
                "output_bytes": args.output.stat().st_size,
                "blocks": block_summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
