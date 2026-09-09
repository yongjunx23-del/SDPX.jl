# API reference

```@meta
CurrentModule = SDPX
```

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
Reals
Nonnegative
Nonpositive
ZeroCone
LorentzCone
RotatedLorentzCone
ExponentialCone
PowerCone
PSDCone
Minimize
Maximize
variable_by_name
constraint_by_name
variable_names
constraint_names
num_variables
num_constraints
```

## Starts

The native product-HSD route is cold-start only at present, but models retain
typed start values for qualified integrations and future continuation routes:

```@docs
set_start!
set_dual_start!
set_dual_slack_start!
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

## Structure cache lifecycle

The cross-solve cache stores only immutable symbolic structure; numeric values
are always reallocated for each solve. These controls expose explicit cache
invalidation and observability without relaxing solver or certificate
semantics.

```@docs
set_structure_cache_enabled!
clear_structure_cache!
structure_cache_stats
symbolic_analysis_count
symbolic_analysis_counts
symbolic_analysis_delta
```

## Accuracy contract

The accuracy contract exposes, for a solved model, the effective arithmetic
precision and class of the retained result, without guessing from coefficient
counts or the ambient scope.

```@docs
AccuracyContract
AccuracyClass
AccuracyVerified
AccuracyUnsupported
AccuracyNumericalFailure
AccuracyInfrastructureFailure
UnsupportedAccuracyContext
accuracy_contract
accuracy_class
```

## Nonsymmetric backend selection

These types control and describe the optional experimental half-Power
factor-pair backend. The default dense-metric route is unchanged unless a
backend is explicitly selected.

```@docs
NonsymmetricBackendChoice
NativeNonsymmetricBackend
ExperimentalHalfPowerFactorPairBackend
UnsupportedBackendError
```

## Results and diagnostics

```@docs
status
value
dual
dual_slack
primal_objective
dual_objective
objective_value
dual_objective_value
primal_residual
dual_residual
relative_gap
iterations
solve_time
is_optimal
is_primal_infeasible
is_dual_infeasible
primal_status
dual_status
termination_status
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
