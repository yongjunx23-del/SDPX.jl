# SDPX bridge schema (version 1)

The language-independent contract between SDPX and foreign runtimes. The first
transport is a command-line bridge (`bin/sdpx_solve.jl`, used by
`mathematica/SDPXLink.wl`); the schema — not the transport — is the stable
part, and it is versioned so a future persistent-server or LibraryLink
transport can speak the same files.

```
consumer ── problem.json ──▶  julia --project=bin bin/sdpx_solve.jl in.json out.json  ──▶ result.json
```

Exit code `0` means `result.json` contains a solve outcome; any other exit
code means it contains a structured error. **The result file is written in
both cases** — a consumer never has to parse a Julia stack trace.

## Numbers

Every numeric field accepts either a JSON number or a **string** in scientific
notation (`"4.898979485566356196394568149411e0"`). Strings are the only safe
carrier above `Float64`: JSON numbers are IEEE doubles in most parsers and
silently round anything wider. The bridge always *emits* strings. `BigFloat`
strings are parsed at `settings.precision_bits`, so digits beyond that
precision are rounded, not lost silently at 53 bits.

## Problem file

```jsonc
{
  "sdpx_schema": 1,                  // required, exactly 1
  "precision": "auto",               // auto | Float64 | Float64x2/x3/x4 | BigFloat
  "objective": [2.0, 3.0],           // c, length m
  "blocks": [                        // one per PSD block: Σᵢ xᵢ Aᵢ − C ⪰ 0
    {
      "dimension": 2,
      "constant":     {"rows": [1,2], "cols": [2,1], "values": [1.0, 1.0]},   // C, sparse COO, 1-based
      "coefficients": [              // one entry per variable with nonzeros
        {"variable": 1, "rows": [1], "cols": [1], "values": [1.0]},
        {"variable": 2, "rows": [2], "cols": [2], "values": [1.0]}
      ]
    }
  ],
  "equalities": {                    // optional: Bᵀx = b, B is m×n in COO
    "rows": [1], "cols": [1], "values": [1.0], "rhs": [0.5]
  },
  "settings": {
    "dualityGapThreshold": "auto",   // or e.g. "1e-80"
    "primalErrorThreshold": "auto",
    "dualErrorThreshold": "auto",
    "maximumIterations": "auto",
    "maxRuntime": "auto",
    "threads": "auto",
    "verbosity": "auto",
    "precision_bits": "auto",        // BigFloat only; CLI --precision=N sets this
    "algorithm": "auto",
    "presolve": "auto",
    "scaling": "auto",
    "sparse": "auto",
    "return_matrices": false,        // include X/Y blocks in the result
    "certificate": true              // include the independent certificate
  }
}
```

COO matrices are dense on arrival — the schema carries structure, the solver
decides storage. Matrices are stored as written; symmetry is the caller's
responsibility (SDPX's ingest symmetrizes within its tolerance and errors on
gross asymmetry).

## Result file

```jsonc
{
  "sdpx_schema": 1,
  "success": true,                   // false ⇒ only "status"/"error" are meaningful
  "error": null,
  "precision": "Float64",
  "status": "Optimal",               // SDPX SolveStatus name as a string
  "optimal": true,
  "message": "",
  "objective": "4.89897950663398",   // all numbers as strings
  "dual_objective": "…", "relative_gap": "…",
  "primal_residual": "…", "dual_residual": "…",
  "iterations": 17,
  "x": ["…"], "y": ["…"],
  "certificate": {                   // present unless settings.certificate=false
    "valid": true, "gap": "…",
    "primal_residual": "…", "dual_residual": "…",
    "primal_psd": true, "dual_psd": true, "failures": []
  },
  "X": [["…"]], "Y": [["…"]],        // only with return_matrices; column-major
  "block_dimensions": [2]            //   flattened blocks, reshape k×k
}
```

`status` is honest: an iteration-limited solve reports `"IterLimit"` with
`optimal: false` and `success: true` — a solver outcome, not a bridge error.
`success: false` is reserved for "no solve happened": unreadable input,
schema violations, unknown precision, internal errors.

## Upgrade path (deliberately not implemented yet)

One process per solve costs Julia startup plus package load (~3–5 s) per
call, which is negligible against high-precision solves and dominant for tiny
ones. When that matters, the planned steps — in order, each speaking this
same schema — are:

1. **Persistent server**: a long-lived Julia process reading length-prefixed
   schema-v1 JSON over stdin/stdout or a socket; the Mathematica side keeps
   `RunProcess` semantics behind the same `SDPXOptimize` signature.
2. **LibraryLink / WSTP**: in-process transport for zero-copy of large dense
   blocks; the schema then only describes the *layout*, not the bytes.

Consumers that stick to the file contract above will not notice the change.

## All-auto policy and SDPB-style CLI

Schema-v1 remains backward compatible with the original `tolerance`,
`maximum_iterations` and `time_limit` keys.  New code should prefer the
independent threshold names above.  Missing policy fields resolve through the
same `SolveOptions` midend used by the Julia API.

The friendly wrapper `bin/sdpx.jl` accepts:

```bash
sdpx problem.json result.json \
  --precision=840 \
  --dualityGapThreshold=1e-80 \
  --primalErrorThreshold=1e-80 \
  --dualErrorThreshold=1e-80
```

The response additionally contains `resolved_options` and a compact `plan`
record (`algorithm`, storage/classification, KKT backend, Gram kernel, schedule
and threads).
