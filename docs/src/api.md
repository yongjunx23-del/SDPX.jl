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
SOCConstraint
ConicProblem
ConicResult
second_order_program
solve_socp
PreparedSolver
prepare
SolveOptions
SolverOptions
SDPResult
SolveStatus
```

## Inspection and certificates

```@docs
solve_summary
result_certificate
SDPX.Experimental.recommended_parameters
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
`SDPX.Experimental.analyze_fixed_trace` conservatively detects direct and
equality-implied constant-trace PSD blocks.
`SDPX.Experimental.resolve_solve_options` shows how the small all-auto
`SolveOptions` policy is lowered to the resolved numerical configuration.
The historical top-level experimental exports completed their deprecation
cycle in 0.4. Qualified `SDPX.name` bindings remain available, while new code
should use `SDPX.Experimental.name`.

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

See [jump.md](jump.md) for the full JuMP guide, including
`GenericModel{Float64x4}` for extended precision, and [convex.md](convex.md)
for Convex atom coverage, typed models, and frontend-overhead benchmarks.

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
