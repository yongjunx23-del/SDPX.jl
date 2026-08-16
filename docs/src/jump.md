# JuMP and MathOptInterface

SDPX provides a non-incremental [MathOptInterface](https://jump.dev/MathOptInterface.jl/stable/)
optimizer. JuMP builds a cached model, then SDPX converts the completed model
into the same typed cone representation used by the public `Model` route before
solving it. The wrapper performs one route classification and one family
lowering; it does not create a second modeling or numerical route.

```julia
using JuMP, LinearAlgebra, SDPX

model = Model(() -> SDPX.Optimizer(
    sparse=:auto,
    tolerance=1e-8,
    threads=4,
    verbosity=0,
))

@variable(model, x[1:2])
@constraint(
    model,
    Symmetric([x[1] -1.0; -1.0 x[2]]) in PSDCone(),
)
@objective(model, Min, 2x[1] + 3x[2])
optimize!(model)
```

The analytic solution is `x = [sqrt(3/2), sqrt(2/3)]` with objective
`2sqrt(6)`.

Use `GenericModel{T}` and `SDPX.Optimizer{T}` for extended coefficient types:

```julia
using JuMP, LinearAlgebra, SDPX
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

## Supported model forms

The wrapper supports:

- minimization, maximization, and feasibility objective senses;
- scalar affine and single-variable objectives;
- free scalar variables;
- scalar affine or single-variable constraints in `MOI.EqualTo`,
  `MOI.GreaterThan`, `MOI.LessThan`, and `MOI.Interval`; pure scalar-cone
  models use the dedicated LP engine;
- vector affine or vector-of-variables constraints in
  `MOI.Nonnegatives`, `MOI.Nonpositives`, and `MOI.Zeros` (through batched
  linear rows);
- vector affine or vector-of-variables constraints in
  `MOI.SecondOrderCone` and `MOI.RotatedSecondOrderCone` (through native
  Lorentz SOC and the exact rotated map);
- vector affine or vector-of-variables constraints in
  `MOI.PositiveSemidefiniteConeTriangle` and
  `MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}`.

The wrapper converts MathOptInterface's column-major upper-triangle ordering
to full symmetric SDPX blocks and applies the required `sqrt(2)` off-diagonal
scaling for scaled PSD cones in both primal and dual results.

Result attributes available through JuMP or MathOptInterface include
termination, primal, dual, and raw statuses; primal and dual objective values
and relative gap; variable primal values; equality and PSD constraint
primal/dual values; solve time and barrier iteration count; and solver name,
version, and raw solver result.

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

The adapter accepts these compatibility aliases. They lower to qualified
engine options; native v0.5 applications should use `Settings` instead:

| Adapter attribute | Qualified engine field | Meaning |
|---|---|---|
| `beta` | `β` | complementarity reduction target |
| `gamma` | `γ` | line-search backtracking factor |
| `omega_p` | `Ωp` | expert fixed-policy primal PSD scale |
| `omega_d` | `Ωd` | expert fixed-policy dual PSD scale |
| `tol_gap` | `ϵ_gap` | relative gap tolerance |
| `tol_primal` | `ϵ_primal` | primal residual tolerance |
| `tol_dual` | `ϵ_dual` | dual residual tolerance |
| `max_iter` / `max_iterations` | `iter_max` | iteration limit |
| `time_limit` | `max_time` | end-to-end pipeline wall-clock limit in seconds |
| `num_threads` | `threads` | maximum Julia tasks used by one solve |
| `precision` | `precision_bits` | BigFloat working precision in bits |
| `verbose` | `verbosity` | output level from 0 to 3 |

Setting the raw `"tolerance"` attribute updates the gap, primal, and dual
tolerances together. Every exact qualified engine field name is also accepted
as a raw optimizer attribute, including `sparse`, `scaling`, `predictor`,
`refine_steps`, `max_restarts`, `extended_precision_blas`, and
`extended_precision_memory_fraction`. Unknown names are rejected. The
attribute bridge does not change the public pure-route contract: mixed
non-free cone families fail before lowering.

`parameter_policy=:auto` (the default) runs the generic automatic Mehrotra
controller: `beta`, `gamma`, `predictor`, and `parameter_strategy` keep the
adapter defaults or user choices. After presolve and scaling, the
selected KKT/provider route constructs a primal/dual affine point and applies
typed cone-interior shifts plus deterministic complementarity mass balancing;
`omega_p` and `omega_d` are ignored. The public resolver reports
`profile=:post_scaling_mehrotra`,
the plan records the neutral deferred identity `:automatic_mehrotra`, and
executed diagnostics record `:post_scaling_mehrotra`. Use `:fixed` in expert
mode to preserve supplied values exactly with provenance `:user_fixed`.
`parameter_strategy=:adaptive` (the default) enables the bounded Mehrotra
controller, independent primal/dual step safeguards, adaptive refinement
limits, and complete fixed-path fallback. See the
[adaptive parameter policy](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-parameter-policy.md).

Native SDP checkpoints are iterate-level warm restarts, not full execution
snapshots. The dedicated LP path does not currently support checkpoint
resume. Julia must be started with at least as many threads as a solve may
request (`julia -t 4`).

## Current limitations

- The wrapper is non-incremental. Modifying and re-solving a JuMP model
  causes the finalized SDPX representation to be rebuilt.
- At least one scalar, SOC, or PSD cone constraint is required.
- Rotated SOC constraints use the exact sparse map
  `(u,v,w) -> (u+v,u-v,sqrt(2)w)` into NativeSOC; MOI primal and dual getters
  apply the inverse and adjoint maps. `Nonnegatives`, `Nonpositives`, and
  `Zeros` vector sets are lowered in batches. General Lorentz SOC constraints
  use NativeSOC directly; strict local fixed-trace Q3 products may select the
  compact specialization. Mixed PSD+SOC models fail clearly rather than
  silently lifting one cone family, and other nonsymmetric cones remain
  unsupported.
- Sparse and dense coefficient storage both support internal equilibration;
  sparse derived caches are rebuilt after scaling.
- LP unboundedness and general conic infeasibility certificates are not yet
  available in every numerical-breakdown case.

Bootstrap integrations that already have final block arrays may continue to use
the qualified `SDPProblem`/loader internals, but new native code should use the
typed `Model` route. No production route converts SOC blocks into PSD blocks.
