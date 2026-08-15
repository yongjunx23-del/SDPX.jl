# Parameters

Most applications should configure only:

```julia
result = solve(
    problem;
    tolerance=1e-8,
    maximum_iterations=200,
    time_limit=Inf,
    threads=Threads.nthreads(),
    precision=nothing,
    verbosity=1,
    diagnostics=true,
    warm_start=nothing,
)
```

The default controller adapts centering, fraction-to-boundary values,
backtracking, and refinement from the measured Newton iteration. Presolve,
scaling, kernel selection, working precision, and scheduling also default to
automatic policies. The tables below describe the expert `SolverOptions{T}`
surface and its defaults.

## Interior-point method and initialization

| Parameter | Default | Meaning |
|---|---:|---|
| `β` | `0.1` | Fixed SDP centering/complementarity reduction target and the safe fallback for adaptive `sigma`. |
| `γ` | `0.9` | Fixed backtracking reduction factor and exact fraction-to-boundary safety. |
| `Ωp` | `1` | Initial primal PSD matrices: `X_l=Ωp*I`. |
| `Ωd` | `1` | Initial dual PSD matrices: `Y_l=Ωd*I`. |
| `predictor` | `:classic` | Predictor rule: `:classic` or `:sdpb`. |
| `refine_steps` | `1` | Number of iterative-refinement passes for the KKT predictor/corrector solutions. |
| `step_rule` | `:auto` | `:backtrack`, exact `2x2`-optimized `:fraction_to_boundary`, or `:auto`. |
| `parameter_policy` | `:auto` | Cold-start structural policy; `:fixed` preserves supplied values. |
| `parameter_strategy` | `:adaptive` | Guarded per-iteration Mehrotra policy with fixed fallback when cold-start or stability diagnostics are unreliable. |
| `adaptive_sigma_max` | `0` | Expert adaptive-centering cap; zero selects by structure. |
| `refine_policy` | `:auto` | `:auto`/`:adaptive` stop KKT refinement from its residual; `:fixed` runs exactly `refine_steps` passes. |

## Convergence and stopping

| Parameter | Default | Meaning |
|---|---:|---|
| `ϵ_gap` | `1e-10` | Relative primal-dual gap tolerance. |
| `ϵ_primal` | `1e-10` | Primal residual tolerance. |
| `ϵ_dual` | `1e-10` | Dual residual tolerance. |
| `termination` | `:relative` | Scale-normalized stopping tests; `:legacy` is the legacy absolute/nonnegative-gap convention. |
| `iter_max` | `200` | Maximum outer iterations (legacy keyword `iterMax`). |
| `max_time` | `Inf` | End-to-end wall-clock limit in seconds, including automatic-pipeline setup and (for the raw-array one-call interface) ingestion. |
| `callback` | `nothing` | Called after every iteration; returning `true` stops with `UserStopped`. |

## Restarts and numerical safeguards

| Parameter | Default | Meaning |
|---|---:|---|
| `restart` | `true` | Rescale the collapsed side and continue after step-size collapse. |
| `min_step` | `1e-10` | A backtracking step below this value triggers a convergence-tail check or restart. |
| `omega_step` | `1e5` | Per-restart multiplier applied to the collapsed `X` or `Y` side. |
| `max_restarts` | `5` | Maximum restarts in the new API. |
| `max_omega` | `1e50` | Compatibility field; the new `solve!` loop does not read it directly. |

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
| `sparse` | `:auto` | Storage selection during ingestion; distinguishes sparse coefficients from aggregate PSD/Schur density. |
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

Sparse equilibration rebuilds derived sparse caches after scaling. Use
`scaling=:equilibrate` (or the legacy `sdp(...; equilibrate=true)` keyword,
which maps to it):

```julia
prob = ingest(c, A, C, B, b; sparse=true)
opts = SolverOptions{T}(scaling=:equilibrate)
```

Warm starts are supplied in original input coordinates. SDPX maps `x0`, `X0`,
and `Y0` through the selected equilibration and maps `y0` through equality
presolve as needed; callers should not pre-scale them.

## Output, timing, and checkpoints

| Parameter | Default | Meaning |
|---|---:|---|
| `verbosity` | `1` | `0` is silent; values of `1` or higher print iteration information. |
| `timing` | `false` | Records total and phase-level timing. |
| `checkpoint_every` | `0` | Save an iterate-level warm-restart checkpoint every N iterations; `0` disables. |
| `checkpoint_path` | `""` | Atomic checkpoint destination used by the SDP path. |
| `mode` | `OPTIMIZE` | `OPTIMIZE` or the internal feasibility mode `FEASIBILITY`. |

Resume restores the primal/dual iterate, centering targets, and
iteration/restart counters. It intentionally reinitializes adaptive-parameter
history, stagnation windows, phase timers, and best-iterate history; a resumed
adaptive solve is therefore not bit-for-bit equivalent to an uninterrupted
run. Checkpoint resume is not currently supported by the dedicated LP path.

## Legacy `sdp(...)` defaults

The common legacy call is equivalent to:

```julia
sdp(c, A, C, B, b;
    β=0.1, γ=0.9, Ωp=1, Ωd=1,
    ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10,
    iterMax=200, prec=300,
    restart=true, minStep=1e-10,
    maxOmega=1e50, OmegaStep=1e5,
    sparse=:auto, verbosity=1,
    termination=:relative,
    equilibrate=false, refine_steps=1, predictor=:classic,
    max_time=Inf, callback=nothing)
```

Two differences matter:

1. `prec=300` is expressed in decimal digits and is converted internally to
   approximately `997` bits. It affects `BigFloat` only.
2. The legacy wrapper derives its restart limit from `maxOmega/OmegaStep`;
   their defaults produce `max_restarts=10`, while directly constructing
   `SolverOptions` defaults to `5`.

The full Newton-method audit, exact diagnostic fields, controller bounds,
fallback rules, arithmetic behavior, and fixed-versus-adaptive results are in
[Adaptive Interior-Point Parameter Policy](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-parameter-policy.md).
