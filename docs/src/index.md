# SDPX.jl

SDPX is an experimental arbitrary-precision conic optimizer written in Julia.
It solves LP, SOC, rotated SOC, SDP, exponential-cone, and power-cone models
through one native product-cone homogeneous self-dual engine.

Supported arithmetic includes Float64, fixed-width MultiFloat, and BigFloat.
Every claimed terminal status is checked in the original model coordinates;
provider or factorization success alone is never treated as a mathematical
certificate.

> SDPX is pre-1.0. APIs and internal layouts may change while the unified engine
> and provider routes complete release qualification.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

## First solve

```julia
using SDPX

model = Model(Float64)
w = variable!(model, :w, 3; domain=Reals())
constraint!(model, :normalization, w[1] - 1, ZeroCone())
constraint!(model, :recurrence, w[1] - w[2] - w[3], ZeroCone())
constraint!(model, :moment, [w[1] w[2]; w[2] w[3]], PSDCone())
objective!(model, Maximize(), w[2])

result = optimize!(
    model;
    settings=Settings(model; limits=Limits(iterations=200), verbosity=0),
    outputs=Outputs(:all, :all, :all; objectives=true, certificate=:summary),
)

status(result)
value(result, w)
primal_objective(result)
certificate(result).valid
```

The same public path accepts mixed supported cone blocks. Nonpositive and
rotated-Lorentz rows are exact canonical transforms; structural PSD/Q3 and
fixed-size kernels are internal specializations, not alternate solvers.

## Result policy

A terminal result is promoted only after independent original-coordinate
verification:

- `:optimal` requires primal/dual feasibility, cone membership, objective gap,
  and complementarity checks;
- `:primal_infeasible` requires a normalized dual ray;
- `:dual_infeasible` requires a normalized primal recession ray.

NaN, infinity, invalid tolerances, stale factor state, or unavailable
certificate data fail closed.

## Documentation map

### Using the solver

- [Quick start](quickstart.md)
- [Command line](cli.md)
- [Precision](precision.md)
- [JuMP and MOI](jump.md)
- [Lorentz and rotated-Lorentz cones](socp.md)
- [Parameters](parameters.md)
- [Diagnostics and certificates](diagnostics.md)

### Understanding execution

- [Automatic pipeline](pipeline.md)
- [Preprocessing](preprocessing.md)
- [Sparse execution](sparse-execution.md)
- [Architecture](architecture.md)
- [Linear-algebra providers](providers.md)
- [Threading](threading.md)

### Operating and validating

- [Benchmarks and evidence](benchmarks.md)
- [Cluster workflow](cluster-workflow.md)
- [Bridge schema](bridge-schema.md)
- [Development](development.md)
- [Qualified internals](internals.md)

The active implementation and handover plan is
[`HANDOVER.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/HANDOVER.md).
Frozen mathematical contracts are under
[`docs/design/`](https://github.com/yongjunx23-del/SDPX.jl/tree/main/docs/design).
Historical plans and reviews are retained in Git history rather than presented
as current documentation.
