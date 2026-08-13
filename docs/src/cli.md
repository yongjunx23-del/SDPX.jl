# SDPX command-line frontend (v0.5 development)

The v0.5 development frontend intentionally resembles the small set of
high-value options used in SDPB workflows while keeping SDPX's LP/SOCP/SDP
automatic pipeline in control.

## Setup

From the repository root:

```bash
julia bin/setup_cli.jl
./bin/sdpx --help
```

`bin/setup_cli.jl` creates the isolated CLI environment and develops the local
checkout into it.  The solver core still has no JSON dependency.

## Normal use: everything automatic

```bash
./bin/sdpx problem.json
```

The result is written to `problem.result.json`.  The output contains both
`resolved_options` and `plan`, so `auto` is inspectable rather than opaque.

## SDPB-style high precision

```bash
./bin/sdpx problem.json result.json \
  --precision=840 \
  --dualityGapThreshold=1e-80 \
  --primalErrorThreshold=1e-80 \
  --dualErrorThreshold=1e-80
```

An integer `--precision=N` means BigFloat with `N` **bits**, matching the
meaning users normally expect from SDPB's precision flag.  High-precision
coefficients in the JSON input should be strings so that they are parsed only
after the BigFloat precision scope has been established.

The three stopping thresholds are independent.  If any is omitted or set to
`auto`, the frontend selects a conservative target using roughly one third of
the available decimal precision, with a Float64 floor of `1e-8`.  For example,
840 bits resolves to approximately `1e-84`.  This is only a stopping-policy
heuristic; final success remains governed by original-coordinate
certification.

## Automatic controls

The ordinary frontend exposes policies, not low-level IPM constants:

```text
--algorithm=auto|lp|socp|sdp
--presolve=auto|on|off
--scaling=auto|none|equilibrate
--sparse=auto|on|off
--formulation=auto|primal|dual
--chordalDecomposition=auto|on|off
--equalitySolver=auto|normal_equations|qr
--workingPrecisionPolicy=auto|fixed
--threads=auto|N
--certificate=auto|on|off
```

Parameters such as centering constants, Q3 Gram strategies, mixed-precision
condition limits and refinement micro-policy remain expert/internal controls.
They should not become command-line flags unless a benchmarked scientific
workflow proves that users need to override them.

## Julia frontend

The same policy boundary is available without the CLI:

```julia
using SDPX

options = SolveOptions()  # every field is :auto
result = solve(problem, options)
```

High-precision policy can be explicit:

```julia
options = SolveOptions(
    precision=840,
    duality_gap_threshold="1e-80",
    primal_error_threshold="1e-80",
    dual_error_threshold="1e-80",
)
```

For an already-ingested problem, its arithmetic has already been chosen.  A
request for 840 bits therefore requires that the problem itself was created as
BigFloat at the desired precision.  The CLI does not have this issue because
it parses the model inside the requested precision scope.

`SolverOptions{T}` remains available as the expert resolved interface and is
not removed in v0.4.1.
