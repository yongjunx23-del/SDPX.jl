# Parameters

The public policy is typed and attached to the same `Model` that is passed to
`optimize!`:

```julia
model = SDPX.Model(Float64)
x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
SDPX.constraint!(model, :lower, x[1] - 1, SDPX.Nonnegative())
SDPX.objective!(model, SDPX.Minimize(), x[1])
settings = SDPX.Settings(
    model;
    tolerances=SDPX.Tolerances(Float64; primal=1e-8, dual=1e-8, gap=1e-8),
    limits=SDPX.Limits(iterations=200, time=Inf, threads=1),
    algorithm=:auto,
    presolve=:auto,
    scaling=:auto,
    sparse=:auto,
    diagnostics=:summary,
    certification=true,
    verbosity=1,
)
result = SDPX.optimize!(model; settings=settings)
```

`Settings{T}` accepts only typed policy fields: `tolerances`, `limits`,
`scaling`, `formulation`, `provider`, `presolve`, `algorithm`, `sparse`,
`equality_solver`, `working_precision_policy`, `diagnostics`, `verbosity`,
`timing`, `certification`, and `blas_threads`. Route selection is pure and
fail-closed: `:lp` is valid only for a pure LP model, `:socp` only for a pure
SOC/RSOC model, and `:sdp` only for a pure SDP model; `:auto` follows the
classified family. Public formulation values are `:auto`,
`:variable_space_schur`, and `:dense_augmented_kkt`.

The default controller adapts centering, fraction-to-boundary values,
backtracking, and refinement from the measured Newton iteration. Presolve,
scaling, kernel selection, working precision, and scheduling also default to
automatic policies. The detailed tables below retain the mature
`SolverOptions{T}` field names for algorithm and compatibility notes only;
that qualified low-level record is not the public quickstart interface.

## Interior-point method and initialization

| Parameter | Default | Meaning |
|---|---:|---|
| `β` | `0.1` | Fixed SDP centering/complementarity reduction target and the safe fallback for adaptive `sigma`. |
| `γ` | `0.9` | Fixed backtracking reduction factor and exact fraction-to-boundary safety. |
| `Ωp` | `1` | Expert fixed-policy primal PSD identity scale. Ignored by the default automatic KKT cold start. |
| `Ωd` | `1` | Expert fixed-policy dual PSD identity scale. Ignored by the default automatic KKT cold start. |
| `predictor` | `:classic` | Predictor rule: `:classic` or `:sdpb`. |
| `refine_steps` | `1` | Number of iterative-refinement passes for the KKT predictor/corrector solutions. |
| `step_rule` | `:auto` | `:backtrack`, exact `2x2`-optimized `:fraction_to_boundary`, or `:auto`. |
| `parameter_policy` | `:auto` | Automatic cold-start Mehrotra controller; `:fixed` preserves supplied values exactly. |
| `parameter_strategy` | `:adaptive` | Guarded per-iteration Mehrotra policy with fixed fallback when cold-start or stability diagnostics are unreliable. |
| `adaptive_sigma_max` | `0` | Expert adaptive-centering cap; zero uses the generic 0.5 maximum. |
| `refine_policy` | `:auto` | `:auto`/`:adaptive` stop KKT refinement from its residual; `:fixed` runs exactly `refine_steps` passes. |

With `parameter_policy=:auto`, the automatic Mehrotra controller keeps `β`,
`γ`, `predictor`, and `parameter_strategy` from the defaults or user choices.
After presolve and scaling, the solver builds an identity-metric KKT system,
solves one primal and one dual affine right-hand side with the same factor,
then shifts the cone variables in their native coordinates. A minimal identity
shift raises orthant/PSD starts, and Lorentz sides still at the typed
cone-vertex envelope, to unit identity mass before complementarity
cross-centering; bounded structured correction reuses an accepted SDP factor
when an original-KKT residual remains above the existing gate.
`Ωp` and `Ωd` do not participate in this path. The public compatibility
resolver reports `profile=:post_scaling_mehrotra`, the plan records the deferred
identity `:automatic_mehrotra`, and executed diagnostics record
`:post_scaling_mehrotra` plus the separate cold-start report.
`parameter_policy=:fixed` uses the supplied values exactly and records
`:user_fixed`.

## Convergence and stopping

| Parameter | Default | Meaning |
|---|---:|---|
| `ϵ_gap` | `1e-10` | Relative primal-dual gap tolerance. |
| `ϵ_primal` | `1e-10` | Primal residual tolerance. |
| `ϵ_dual` | `1e-10` | Dual residual tolerance. |
| `iter_max` | `200` | Maximum outer iterations (legacy keyword `iterMax`). |
| `max_time` | `Inf` | End-to-end wall-clock limit in seconds, including automatic-pipeline setup and model compilation. |
| `callback` | `nothing` | Called after every iteration; returning `true` stops with `UserStopped`. |

## Restarts and numerical safeguards

| Parameter | Default | Meaning |
|---|---:|---|
| `restart` | `true` | Rescale the collapsed side and continue after step-size collapse. |
| `min_step` | `1e-10` | A backtracking step below this value triggers a convergence-tail check or restart. |
| `omega_step` | `1e5` | Per-restart multiplier applied to the collapsed `X` or `Y` side. |
| `max_restarts` | `5` | Maximum restarts in the new API. |
| `max_omega` | `1e50` | Compatibility field for the qualified legacy engine; public `Settings` has no equivalent. |

For fixed-exponent types such as `Float64x4`, SDPX limits the effective
restart multiplier and returns `NumericalBreakdown` when an iterate becomes
non-finite.

## Precision, equilibration, and storage

| Parameter | Default | Meaning |
|---|---:|---|
| `precision_bits` | `997` | Working precision for `BigFloat` only. |
| `working_precision_policy` | `:auto` | May start lower and retry at `precision_bits` unless certification passes. |
| `minimum_working_precision_bits` | `192` | Lower bound for the staged BigFloat selector. |
| `convert_inputs` | `false` | Normalize independent `BigFloat` storage to `precision_bits`; cannot recover lost digits. |
| `scaling` | `:auto` | LP geometric scaling and adaptive-pass Ruiz congruence/variable scaling for SDP. |
| `sparse` | `:auto` | Storage selection during model compilation; distinguishes sparse coefficients from aggregate PSD/Schur density. |
| `formulation` | `:auto` | Static dense KKT formulation selection (`:normal_equations`, `:augmented`, `:primal`). |
| `linear_algebra_backend` | `:auto` | Resolves once during planning: `:standard`, `:bfla`, `:multifloat`, `:legacy`. |
| `extended_precision_blas` | type-dependent | Conservative `:auto` for fixed-width extended types and BigFloat, `:off` for Float64. |
| `extended_precision_memory_fraction` | `0.10` | Maximum fraction of available memory for packed extended-precision panels. |
| `mixed_precision_kkt` | type-dependent | `:auto` for BigFloat and fixed-width extended arithmetic, `:off` for Float64. |
| `mixed_precision_condition_limit` | type-dependent | `1e14` for `Float64x4`, `1e8` otherwise. |
| `mixed_precision_refine_max_steps` | `32` | Maximum correction solves before native fallback. |
| `mixed_precision_memory_fraction` | `0.10` | Maximum fraction of available memory for Float64 factors and conversion scratch. |
| `equality_solver` | `:auto` | Normal equations with rank-revealing QR fallback when factor diagnostics justify it. |
| `force_gc` | `false` | Run a full collection after each accepted iteration and return free allocator pages where supported. |

Unless `refine_tol` is explicitly positive, dense mixed-precision refinement
uses `max(64 * eps(T), min(ϵ_gap, ϵ_primal, ϵ_dual)^2)`. Native KKT refinement
normally targets `64 * eps(T)`; the large regularized sparse Float64 SDP route
uses a looser tolerance retaining two guard digits beyond the requested
certificate. A positive user-supplied `refine_tol` always takes precedence.

Sparse equilibration rebuilds derived sparse caches after scaling. On the
public route, request it with `Settings(model; scaling=:equilibrate)`; the
model and all warm-start values remain in their original coordinates:

```julia
settings = SDPX.Settings(model; scaling=:equilibrate)
result = SDPX.optimize!(model; settings=settings)
```

Warm starts use the exported modeling API: `set_start!` sets a variable block,
`set_dual_start!` sets a constraint block, and `set_dual_slack_start!` sets a
variable block's dual slack. A layout-compatible SDP can also continue from a
previous certified result with
`optimize!(model; settings=settings, warm_start=previous_result)`. Values are
always supplied in original input coordinates; callers should not pre-scale
them.

## Output and timing

| Parameter | Default | Meaning |
|---|---:|---|
| `verbosity` | `1` | `0` is silent; values of `1` or higher print iteration information. |
| `timing` | `false` | Records total and phase-level timing. |

## Qualified compatibility defaults

The mature low-level engine retains historical names such as `sdp`,
`SolverOptions`, `β`, `γ`, `Ωp`, and `Ωd` for compatibility and algorithm
audits. Those names are qualified implementation details; they are not
additional v0.5 public entry points. Public code should express stopping and
resource policy with `Tolerances`, `Limits`, and `Settings` as shown above.

The full Newton-method audit, exact diagnostic fields, controller bounds,
fallback rules, arithmetic behavior, and fixed-versus-adaptive results are in
[Adaptive Interior-Point Parameter Policy](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/adaptive-parameter-policy.md).
