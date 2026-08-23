# API reference

The v0.5 public surface is deliberately small: build one typed `Model`, solve
it with `optimize!`, and inspect the returned result through the accessors
below. Anything not documented here, and anything prefixed with `_`, is an
internal implementation detail and may change without notice.

## Modeling

```@docs
Model
variable!
constraint!
objective!
set_start!
set_dual_start!
set_dual_slack_start!
Reals
Nonnegative
Nonpositive
ZeroCone
LorentzCone
RotatedLorentzCone
ExponentialCone
PSDCone
Minimize
Maximize
```

## Settings and solve

```@docs
Settings
Tolerances
Limits
Outputs
optimize!
execution_plan
```

## Results and diagnostics

```@docs
status
value
dual
dual_slack
primal_objective
dual_objective
certificate
diagnostics
iteration_history
performance_trace
```

## MathOptInterface

`SDPX.Optimizer` implements MathOptInterface, so SDPX can be used from JuMP
or any other MOI client that emits the supported linear, Lorentz, rotated
Lorentz, and positive-semidefinite cone sets:

```julia
using JuMP, LinearAlgebra, SDPX
model = Model(() -> SDPX.Optimizer(sparse=:auto, verbosity=0))
```

The MOI adapter is the only secondary frontend. It uses the same typed native
solver and does not add a parallel SDPX-specific modeling or solve API.

```@docs
Optimizer
```
