# SDPX command-line frontend

The command-line frontend resembles the small set of high-value options used
in SDPB workflows while keeping SDPX's LP/SOCP/SDP automatic pipeline in
control.

## Setup

From the repository root:

```bash
julia bin/setup_cli.jl
./bin/sdpx --help
```

`bin/setup_cli.jl` creates the isolated CLI environment and develops the local
checkout into it. The solver core still has no JSON dependency.

## Normal use: everything automatic

```bash
./bin/sdpx problem.json
```

The result is written to `problem.result.json`. The output contains both
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
meaning users normally expect from SDPB's precision flag. High-precision
coefficients in the JSON input should be strings so that they are parsed only
after the BigFloat precision scope has been established.

The three stopping thresholds are independent. If any is omitted or set to
`auto`, the frontend selects a conservative target using roughly one third of
the available decimal precision, with a Float64 floor of `1e-8`. For example,
840 bits resolves to approximately `1e-84`. This is only a stopping-policy
heuristic; final success remains governed by original-coordinate
certification.

## Automatic controls

The ordinary frontend exposes policies, not low-level IPM constants:

```text
--algorithm=auto|lp|socp|sdp
--presolve=auto|on|off
--scaling=auto|none|equilibrate
--sparse=auto|on|off
--formulation=auto|primal|normal_equations|augmented
--equalitySolver=auto|normal_equations|qr
--workingPrecisionPolicy=auto|fixed
--threads=auto|N
--certificate=auto|on|off
```

Under `:auto`, the static formulation planner selects the dense KKT
formulation before execution. `normal_equations` and `augmented` are expert
overrides; `primal` preserves the historical primal-formulation policy without
disabling sparse or block-arrow routes. `dual` is not a production formulation
and fails closed.

Parameters such as centering constants, Q3 Gram strategies, mixed-precision
condition limits, and refinement micro-policy remain expert/internal controls.
They should not become command-line flags unless a benchmarked scientific
workflow proves that users need to override them.

## Julia frontend

The typed `Model` frontend exposes the same policy boundary without JSON:

```julia
using SDPX

model = Model(Float64)
x = variable!(model, :x, 1; domain=Nonnegative())
constraint!(model, :lower, x[1] - 1, Nonnegative())
objective!(model, Minimize(), x[1])
settings = Settings(model; verbosity=0)
result = optimize!(model; settings=settings)
```

High-precision policy can be explicit:

```julia
model = Model(BigFloat; precision_bits=840)
settings = Settings(
    model;
    tolerances=Tolerances(
        BigFloat;
        primal=big"1e-80",
        dual=big"1e-80",
        gap=big"1e-80",
    ),
)
```

The model's arithmetic is chosen at construction. A request for 840 bits
therefore requires `Model(BigFloat; precision_bits=840)`, while the CLI parses
JSON inside the requested precision scope. `Settings.algorithm` is a
read-only diagnostic label whose only accepted value is `:auto`: native
product HSD is the only public engine, and the family selectors `:lp`,
`:socp`, and `:sdp` are deprecated and rejected. (The CLI's own
`--algorithm` flag lowers into the qualified `SolveOptions` frontend record
and is unchanged.)

The CLI implementation lowers its JSON policy through qualified compatibility
option records, but those records are not part of the public Julia quickstart
surface.
