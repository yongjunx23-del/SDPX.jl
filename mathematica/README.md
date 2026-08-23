# SDPX from Mathematica

`SDPXLink.wl` calls the SDPX.jl solver through a command-line bridge:
the problem is exported as JSON, Julia runs `bin/sdpx_solve.jl` via
`RunProcess`, and the result is imported back. Numbers above `Float64`
travel as strings, so `BigFloat` solves round-trip their full precision
(measured: ~30 correct digits at 256 bits through the whole loop). The
schema is documented in [`docs/bridge-schema.md`](../docs/src/bridge-schema.md),
together with the upgrade path to a persistent server or LibraryLink —
this first version deliberately keeps one process per solve.

## Setup (once)

From the SDPX.jl repository root:

```bash
julia --project=bin -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Requirements: Julia ≥ 1.10 on `PATH` (or pass `"JuliaExecutable"`),
Mathematica ≥ 12 (tested on 14.1).

## Use

```wolfram
<< "/path/to/SDPX.jl/mathematica/SDPXLink.wl"

(* minimise 2 x1 + 3 x2  subject to  {{x1, -1}, {-1, x2}} ⪰ 0 *)
c = {2, 3};
A = {{ {{1, 0}, {0, 0}}, {{0, 0}, {0, 1}} }};   (* blocks ▸ variables ▸ matrix *)
C = {{{0, 1}, {1, 0}}};

result = SDPXOptimize[c, A, C]
result["Objective"]        (* 4.898979506633980 — exact optimum 2√6 *)
result["x"]
result["Certificate"]

(* high precision: strings in, ~30 digits back *)
SDPXOptimize[c, A, C, "Precision" -> "BigFloat", "PrecisionBits" -> 256,
    "Tolerance" -> "1e-30"]["Objective"]

(* equalities Bᵀx = b, sparse blocks as SparseArray *)
SDPXOptimize[c, A, C, B, b, "MaxIterations" -> 500]
```

`SDPXOptimize` returns an `Association` (`"Status"`, `"Objective"`, `"x"`,
`"y"`, `"Certificate"`, optionally `"X"`/`"Y"` with
`"ReturnMatrices" -> True`), or a `Failure` whose message carries the
solver's own error text. A solve that stops at its iteration limit is a
*result* (`"Status" -> "IterLimit"`, `"Optimal" -> False`), not a `Failure`;
`Failure` means no solve happened.

Options: `"Precision"` (`"Float64"`, `"Float64x2"`, `"Float64x4"`,
`"BigFloat"`), `"PrecisionBits"`, `"Tolerance"`, `"MaxIterations"`,
`"TimeLimit"`, `"Verbosity"`, `"ReturnMatrices"`, `"JuliaExecutable"`,
`"SDPXDirectory"` (needed when the package file was copied away from the
repository), `"KeepFiles"` (retain the JSON pair for debugging).

Temporary files live in `$TemporaryDirectory` under unique names and are
deleted on every path unless `"KeepFiles" -> True`.

See [`examples/SDPXExample.nb`](examples/SDPXExample.nb) for a worked
notebook. The same JSON round trip is covered by
[`test/cli_bridge.jl`](../test/cli_bridge.jl) in CI.
