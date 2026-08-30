# JuMP and MathOptInterface

SDPX provides a non-incremental [MathOptInterface](https://jump.dev/MathOptInterface.jl/stable/)
optimizer. JuMP builds a cached model, then SDPX converts the completed model
into the same canonical product-cone program used by the public `Model` route.
The wrapper is a one-shot frontend; it does not create a second numerical
engine.

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
  `MOI.GreaterThan`, `MOI.LessThan`, and `MOI.Interval`; scalar rows become
  product-cone blocks in the shared HSD engine;
- vector affine or vector-of-variables constraints in
  `MOI.Nonnegatives`, `MOI.Nonpositives`, and `MOI.Zeros` (through batched
  linear rows);
- vector affine or vector-of-variables constraints in
  `MOI.SecondOrderCone` and `MOI.RotatedSecondOrderCone` (through native
  Lorentz SOC and the exact rotated map);
- vector affine or vector-of-variables constraints in
  `MOI.PositiveSemidefiniteConeTriangle`;
- vector affine constraints in
  `MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}`. Scaled PSD product
  variables are reported unsupported rather than entering an unavailable
  product-scaling path.

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
| `tol_gap` | `ϵ_gap` | relative gap tolerance |
| `tol_primal` | `ϵ_primal` | primal residual tolerance |
| `tol_dual` | `ϵ_dual` | dual residual tolerance |
| `max_iter` / `max_iterations` | `iter_max` | iteration limit |
| `time_limit` | `max_time` | solve-phase wall-clock limit in seconds; setup time is reported separately |
| `num_threads` | `threads` | maximum Julia tasks used by one solve |
| `precision` | `precision_bits` | BigFloat working precision in bits |
| `verbose` | `verbosity` | output level from 0 to 3 |
| `scaling` | `scaling` | `:auto`, `:none`, or `:equilibrate` |
| `presolve` | `presolve` | `:auto`, `:on`, or `:off` |
| `algorithm` | `algorithm` | `:auto` only; family values (`:lp`, `:socp`, `:sdp`) are deprecated and rejected |
| `sparse` | `sparse` | sparse-storage policy |
| `formulation` | `formulation` | `:auto`, `:normal_equations`, or `:augmented` |
| `equality_solver` | `equality_solver` | `:auto`, `:normal_equations`, or `:qr` |
| `linear_algebra_backend` | `linear_algebra_backend` | provider policy |
| `working_precision_policy` | `working_precision_policy` | `:auto` or `:fixed` |
| `diagnostics` | `diagnostics` | retain execution diagnostics |
| `timing` | `timing` | record phase timings |
| `certification` | `certification` | retain post-solve certification |

Setting the raw `"tolerance"` attribute updates the gap, primal, and dual
tolerances together. The exact qualified fields listed above are accepted as
raw optimizer attributes; unknown names and fields without a lossless public
Settings mapping are rejected with `MOI.UnsupportedAttribute`. This includes
expert interior-point controls such as `beta`, `gamma`, `omega_p`, `omega_d`,
`predictor`, `parameter_strategy`, `refine_steps`, `max_restarts`,
`extended_precision_blas`, `mixed_precision_kkt`, and `checkpoint_path`.

The raw `"engine"` attribute accepts `:auto` (default) or `:native_hsd` only;
the historical `:legacy` selector is rejected with a migration error because
native product HSD is the only public engine. The raw `"algorithm"` attribute
accepts `:auto` only; family selectors are rejected with a migration error.
Use the qualified `SolverOptions` interface for those expert controls instead
of silently falling back to defaults through JuMP.

The MOI bridge uses the public automatic Mehrotra controller after presolve
and scaling. It constructs a primal/dual affine point and applies typed
cone-interior shifts plus deterministic complementarity mass balancing. The
typed Model API records this as `profile=:post_scaling_mehrotra`; use the
qualified low-level interface when a fixed expert trajectory is required. See
the
[adaptive parameter policy](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/adaptive-parameter-policy.md).

Checkpoint helpers are qualified compatibility utilities rather than a second
MOI solve path. Julia must be started with at least as many threads as a solve
may request (`julia -t 4`).

## Current limitations

- The wrapper is non-incremental (`MOI.supports_incremental_interface`
  returns `false`). Modifying and re-solving a JuMP model causes the
  finalized SDPX representation to be rebuilt.
- `MOI.ExponentialCone` and `MOI.PowerCone` constraints remain unsupported
  and fail closed (during discovery or copy). The direct `Model` route can
  execute primal Exp/Power blocks through native HSD, but the MOI adapter
  does not claim that surface until the standard MOI conformance tests are
  green for it.
- At least one scalar, SOC, or PSD cone constraint is required.
- Rotated SOC constraints use the exact sparse map
  `(u,v,w) -> (u+v,u-v,sqrt(2)w)` into a canonical Lorentz block; MOI primal
  and dual getters apply the inverse and adjoint maps. `Nonnegatives`,
  `Nonpositives`, and `Zeros` vector sets are lowered in batches. Verified
  fixed-trace Q3 structure may select a local specialization inside the shared
  product-cone runtime. Mixed supported cone blocks retain their native
  coordinates; no cone is silently lifted to PSD.
- Sparse and dense coefficient storage both support internal equilibration;
  sparse derived caches are rebuilt after scaling.
- LP unboundedness and general conic infeasibility certificates are not yet
  available in every numerical-breakdown case.

Integrations that already have final block arrays may use qualified loader
internals, but new code should use the typed `Model` route. No production route
converts SOC blocks into PSD blocks.
