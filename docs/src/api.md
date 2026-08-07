# API reference

The stable-intent surface. Anything not documented here, and anything
prefixed with `_`, is internal and may change without notice.

## Solving

```@docs
solve
solve!
ingest
linear_program
solve_lp
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

The optimize-mode ray diagnostic is intentionally outside the stable-intent
export surface while its result schema is still experimental:

```@docs
SDPX.infeasibility_diagnosis
```

Validated optimize-mode rays produce the formal `PrimalInfeasible` or
`DualInfeasible` members of `SolveStatus`. The direct primal-dual iteration is
not yet a full homogeneous self-dual embedding; the certificate metadata says
which generator was used.

Additional introspection entry points live under `SDPX.Experimental`
(experimental; result shapes may change):
`SDPX.Experimental.structure_summary` and
`SDPX.Experimental.analyze_structure` report the detected problem structure,
`SDPX.Experimental.classify_problem` the cone/storage classification, and
`SDPX.Experimental.build_execution_plan` the pre-solve execution plan.
The historical top-level exports are deprecated in 0.3 and scheduled to stop
being exported in 0.4.

## Spectrum helpers

```@docs
reconstruct_spectrum
export_spectrum
```

## MathOptInterface

`SDPX.Optimizer` implements MathOptInterface, so SDPX can be used from JuMP
or as the solver for a Convex.jl DCP model:

```julia
using JuMP, LinearAlgebra, SDPX
model = Model(() -> SDPX.Optimizer(sparse=:auto, verbosity=0))
```

See the repository's `docs/julia-interface.md` for the full JuMP guide,
including `GenericModel{Float64x4}` for extended precision.
See `docs/convex-interface.md` for Convex atom coverage, typed models, and
frontend-overhead benchmarks.

```@docs
Optimizer
convex_optimizer
convex_semidefinite
solve_convex!
```

## Legacy interface

Preserved from SDPJSolver.jl for source compatibility; prefer the typed API
above for new code.

```@docs
sdp
findFeasible
```
