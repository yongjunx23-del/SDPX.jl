# LP / SOCP / SDP architecture boundary for v0.4.1 -> v0.5

## Target

```text
                 Frontend
 Julia arrays / JuMP-MOI / CLI / file loaders
                       |
                       v
             Canonical cone semantics
          Linear | Lorentz | PositiveSemidefinite
                       |
                       v
                  Midend
 analyze -> presolve -> scale -> formulate -> plan
                       |
                  ExecutionPlan
                       |
            +----------+----------+
            |          |          |
            v          v          v
           LP         SOCP        SDP
          kernel     kernel      kernel
            \          |          /
             +---------+---------+
                       v
             system assembly / KKT
                       v
        factorization -> refinement
                       v
          original-coordinate certificate
```

## Current-to-target mapping

The current tree is kept running while code migrates behind stable boundaries.

| Current code | Target responsibility |
|---|---|
| `ingest.jl`, `lp_api.jl`, `soc.jl`, `moi_wrapper.jl` | frontend/canonicalization |
| `preprocessing.jl`, `pipeline.jl`, `adaptive_parameters.jl` | midend |
| `lp_solver.jl`, `soc_native_q3.jl`, `schur.jl`, `step.jl` | cone/system backend |
| `kkt*.jl`, mixed-precision kernels | linear-system backend |
| `validation.jl` | authoritative certificate |

The new files `frontend/solve_options.jl` and `midend/resolve_options.jl` are the
first deliberate boundary.  Do not duplicate that policy elsewhere.

## Public vs expert options

Public policy (`SolveOptions`) stays small and defaults entirely to `auto`.
Resolved expert state (`SolverOptions{T}`) may remain large because it is an
internal numerical contract.  New microkernel knobs should default to internal
policy and should not be added to `SolveOptions` without a user-level need.

## Planner ownership

`ExecutionPlan` must become authoritative.  Deterministic choices made after
plan construction are architecture bugs.  Runtime numerical fallback is
allowed, but it must be recorded as:

```text
planned_backend
planned_kkt_formulation
executed_backend
fallback_reason
```

and the original-coordinate certificate remains the final status authority.

## Cone formulation ownership

SOC -> PSD is a **formulation transform**, not a frontend fact.  Native SOC and
arrow specialized Q3 implementations therefore share the same Lorentz semantic
IR.  The planner may choose an exact PSD-arrow reference route when necessary,
but diagnostics and benchmarks must label that formulation explicitly.
