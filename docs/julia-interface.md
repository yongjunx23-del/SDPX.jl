# JuMP and MathOptInterface Guide

SDPX exposes a non-incremental
[MathOptInterface](https://jump.dev/MathOptInterface.jl/stable/) optimizer.
JuMP builds a cached model, then SDPX converts the completed model into its
native block-SDP representation before solving it. The wrapper calls the same
`ingest` and `solve!` core as the native API.

## Simplified native interface

Most applications can use the one-call interface:

```julia
result = solve(
    c, A, C, B, b;
    tolerance=1e-8,
    maximum_iterations=300,
    time_limit=60.0,
    threads=4,
    precision=:float64,
    verbosity=1,
    diagnostics=true,
    warm_start=nothing,
)
```

`precision` accepts `:float64`, `:bigfloat`, or a concrete extended type such
as `Float64x4`. The automatic pipeline classifies the cone structure and
arithmetic, removes dependent equalities, scales the model, selects the
kernel and schedule, and tunes the initialization. `solve!` with an explicit
`SolverOptions` remains the expert interface.

`time_limit` is end-to-end for solver work: it includes automatic-pipeline
setup, and `solve(c, A, C, B, b; ...)` also charges raw input ingestion. Native
warm starts are specified in original input coordinates and are mapped through
equality presolve and the selected LP/SDP equilibration automatically.

## Float64 example

```julia
using JuMP, LinearAlgebra, SDPX   # `Symmetric` comes from LinearAlgebra;
                                  # `MOI` is re-exported by JuMP

model = Model(() -> SDPX.Optimizer(
    sparse=:auto,
    verbosity=0,
    tol_gap=1e-8,
))

@variable(model, x[1:2])
@constraint(
    model,
    Symmetric([x[1] -1.0; -1.0 x[2]]) in PSDCone(),
)
@objective(model, Min, 2x[1] + 3x[2])

optimize!(model)

@assert termination_status(model) == MOI.OPTIMAL
println("objective = ", objective_value(model))
println("x = ", value.(x))
```

The analytic solution is
`x = [sqrt(3/2), sqrt(2/3)]` with objective `2sqrt(6)`.

## Float64x4 example

Use a typed JuMP model and typed optimizer when the coefficients are not
ordinary `Float64` values:

```julia
using JuMP, LinearAlgebra, SDPX   # `Symmetric` comes from LinearAlgebra
using MultiFloats: Float64x4

const T = Float64x4
model = GenericModel{T}(
    () -> SDPX.Optimizer{T}(
        sparse=:auto,
        verbosity=0,
        tol_gap=T(1e-8),
        tol_primal=T(1e-8),
        tol_dual=T(1e-8),
    ),
)

@variable(model, x[1:2])
@constraint(
    model,
    Symmetric([x[1] T(-1); T(-1) x[2]]) in PSDCone(),
)
@objective(model, Min, T(2) * x[1] + T(3) * x[2])

optimize!(model)
```

The same pattern works for other `AbstractFloat` coefficient types supported
by the native solver. `BigFloat` solves are deliberately serial. Fixed-width
types such as `Float64x4` can use Julia threads.

## Supported model forms

The initial wrapper supports:

- minimization, maximization, and feasibility objective senses;
- scalar affine and single-variable objectives;
- free scalar variables;
- scalar affine or single-variable constraints in `MOI.EqualTo`;
- scalar affine or single-variable constraints in `MOI.GreaterThan` and
  `MOI.LessThan`; pure scalar-cone models use the dedicated LP engine;
- vector affine or vector-of-variables constraints in `MOI.SecondOrderCone`;
- vector affine or vector-of-variables constraints in
  `MOI.PositiveSemidefiniteConeTriangle`;
- `MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}`.

The wrapper converts MathOptInterface's column-major upper-triangle ordering
to full symmetric SDPX blocks. It also applies the required `sqrt(2)`
off-diagonal scaling for scaled PSD cones in both primal and dual results.

The following result attributes are available through JuMP or
MathOptInterface:

- termination, primal, dual, and raw statuses;
- primal and dual objective values and relative gap;
- variable primal values;
- equality and PSD constraint primal/dual values;
- solve time and barrier iteration count;
- solver name, version, and raw solver result.

## Options

Options can be passed to the constructor:

```julia
model = Model(() -> SDPX.Optimizer(
    tolerance=1e-8,
    threads=4,
    parameter_policy=:auto,
    max_iterations=300,
    time_limit=60.0,
    verbose=0,
))
```

They can also be set after model construction:

```julia
set_optimizer_attribute(model, "tol_gap", 1e-8)
set_optimizer_attribute(model, "tol_primal", 1e-8)
set_optimizer_attribute(model, "tol_dual", 1e-8)
set_time_limit_sec(model, 60.0)
set_silent(model)
```

Convenience aliases are:

| Interface name | `SolverOptions` field | Meaning |
|---|---|---|
| `beta` | `β` | complementarity reduction target |
| `gamma` | `γ` | line-search backtracking factor |
| `omega_p` | `Ωp` | initial primal PSD scale |
| `omega_d` | `Ωd` | initial dual PSD scale |
| `tol_gap` | `ϵ_gap` | relative gap tolerance |
| `tol_primal` | `ϵ_primal` | primal residual tolerance |
| `tol_dual` | `ϵ_dual` | dual residual tolerance |
| `max_iter` | `iter_max` | iteration limit |
| `max_iterations` | `iter_max` | iteration limit |
| `time_limit` | `max_time` | end-to-end pipeline wall-clock limit in seconds |
| `num_threads` | `threads` | maximum Julia tasks used by one solve |
| `precision` | `precision_bits` | BigFloat working precision in bits |
| `verbose` | `verbosity` | output level from 0 to 3 |

Setting the raw `"tolerance"` attribute updates the gap, primal, and dual
tolerances together.

Every exact `SolverOptions` field name is also accepted as a raw optimizer
attribute, including `sparse`, `equilibrate`, `predictor`, `refine_steps`,
`max_restarts`, `extended_precision_blas`, and
`extended_precision_memory_fraction`. `parameter_policy=:auto` (the default)
selects a calibrated structural profile without a pilot solve; use
`:fixed` in expert mode to preserve explicitly supplied `beta`, `gamma`, and
initialization scales.
For the dedicated LP engine, automatic mode selects the faster
`beta=1/50, gamma=99/100` profile only when a row-scale-invariant
initial-distance indicator is at most `1000`; distant starts retain the
configured conservative values. The selected profile and parameters are
available in `result.diagnostics.plan`.
`parameter_strategy=:adaptive` enables guarded per-iteration `beta`/`gamma`
adaptation and records every selection in `result.parameter_history`. It
currently remains opt-in because the representative SDP benchmark did not
beat the fixed strategy. `extended_precision_blas=:auto` separately enables the conservative packed
Float64x4/BigFloat Schur crossover. Its default is `:off`. Unknown names are
rejected.

Native SDP checkpoints are iterate-level warm restarts, not full execution
snapshots. Resume restores the iterate and iteration/restart counters but
resets adaptive-parameter, stagnation, phase-timing, and best-iterate history.
The dedicated LP path does not currently support checkpoint resume.

Julia must be started with at least as many threads as a solve may request:

```bash
julia -t 4
```

## Current limitations

- The wrapper is non-incremental. Modifying and re-solving a JuMP model causes
  the finalized SDPX representation to be rebuilt.
- At least one scalar, SOC, or PSD cone constraint is required.
- Rotated SOC and other nonsymmetric cones still rely on MOI bridges or future
  native support. SOC constraints currently use an exact PSD arrow lift.
- Sparse and dense coefficient storage both support internal equilibration.
  Sparse derived caches are rebuilt after scaling.
- LP unboundedness and general conic infeasibility certificates are not yet
  available in every numerical-breakdown case.
- The specialized block-arrow factorization currently applies to sparse
  models without explicit equality columns. Equality-constrained models use
  the generic KKT path.

The native `SDPProblem` API remains preferable when bootstrap code already has
final block arrays and wants to avoid modeling-layer conversion.
