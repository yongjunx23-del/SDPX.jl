# API reference

The stable-intent surface. Anything not documented here, and anything
prefixed with `_`, is internal and may change without notice.

## Solving

```@docs
solve
solve!
ingest
SolverOptions
SDPResult
SolveStatus
```

## Inspection and certificates

```@docs
solve_summary
result_certificate
recommended_parameters
```

Additional introspection entry points (experimental; result shapes may
change): `structure_summary` and `analyze_structure` report the detected
problem structure, `classify_problem` the cone/storage classification, and
`build_execution_plan` the pre-solve execution plan. Their docstrings live on
the internal bindings; see the repository's `docs/` design notes for the
concepts.

## Spectrum helpers

```@docs
reconstruct_spectrum
export_spectrum
```

## MathOptInterface

`SDPX.Optimizer` implements MathOptInterface, so SDPX can be used from JuMP:

```julia
using JuMP, LinearAlgebra, SDPX
model = Model(() -> SDPX.Optimizer(sparse=:auto, verbosity=0))
```

See the repository's `docs/julia-interface.md` for the full JuMP guide,
including `GenericModel{Float64x4}` for extended precision.

## Legacy interface

Preserved from SDPJSolver.jl for source compatibility; prefer the typed API
above for new code.

```@docs
sdp
findFeasible
```
