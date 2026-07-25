# SDPX Solver Parameters

These defaults come from `SolverOptions{T}`. Differences in the legacy
`sdp(...)` wrapper are listed separately.

## Interior-point method and initialization

| Parameter | Default | Meaning |
|---|---:|---|
| `β` | `0.1` | Centering/complementarity reduction target. Each step targets `β*μ`; smaller values are usually more aggressive. |
| `γ` | `0.9` | Backtracking reduction factor. An infeasible trial step is reduced as `t ← γ*t`. |
| `Ωp` | `1` | Initial primal PSD matrices: `X_l=Ωp*I`. |
| `Ωd` | `1` | Initial dual PSD matrices: `Y_l=Ωd*I`. |
| `predictor` | `:classic` | Predictor rule: `:classic` or `:sdpb`. |
| `refine_steps` | `1` | Number of iterative-refinement passes for the KKT predictor/corrector solutions. |
| `step_rule` | `:auto` | `:backtrack`, exact `2x2`-optimized `:fraction_to_boundary`, or `:auto` (fraction-to-boundary for blocks up to `2x2`, backtracking otherwise). |
| `parameter_policy` | `:auto` | Cold-start structural policy. `:auto` may select calibrated initial `β`, `γ`, `Ωp`, and `Ωd`; `:fixed` preserves the supplied values. |
| `parameter_strategy` | `:fixed` | Per-iteration policy. `:adaptive` updates `β` and `γ` with guarded fallback and records their history; it remains opt-in because the current SDP benchmark gate did not improve. |
| `refine_policy` | `:auto` | `:auto`/`:adaptive` stop KKT refinement from its residual; `:fixed` always runs exactly `refine_steps` passes. |

## Convergence and stopping

| Parameter | Default | Meaning |
|---|---:|---|
| `ϵ_gap` | `1e-10` | Relative primal-dual gap tolerance. |
| `ϵ_primal` | `1e-10` | Primal residual tolerance. |
| `ϵ_dual` | `1e-10` | Dual residual tolerance. |
| `termination` | `:relative` | Uses scale-normalized stopping tests. `:legacy` provides a legacy-like absolute/nonnegative-gap convention while retaining modern inclusive boundaries and post-solve certification. |
| `iter_max` | `200` | Maximum outer iterations. The legacy keyword is `iterMax`. |
| `max_time` | `Inf` | End-to-end wall-clock limit in seconds. It includes automatic-pipeline setup; the public raw-array one-call interface also includes ingestion. |
| `callback` | `nothing` | Called after every iteration. Returning `true` stops with `UserStopped`. |

The limit is therefore not only a barrier-iteration budget. Presolve,
classification, scaling selection, and workspace setup consume it before the
first iteration.

## Restarts and numerical safeguards

| Parameter | Default | Meaning |
|---|---:|---|
| `restart` | `true` | Whether to rescale the collapsed side and continue after step-size collapse. |
| `min_step` | `1e-10` | A backtracking step below this value triggers a convergence-tail check or restart. |
| `omega_step` | `1e5` | Per-restart multiplier applied to the collapsed `X` or `Y` side. |
| `max_restarts` | `5` | Maximum restarts in the new API. |
| `max_omega` | `1e50` | Compatibility field. The new `solve!` loop does not read it directly. |

For fixed-exponent types such as `Float64x4`, SDPX also limits the effective
restart multiplier and returns `NumericalBreakdown` when an iterate becomes
non-finite.

## Precision, equilibration, and storage

| Parameter | Default | Meaning |
|---|---:|---|
| `precision_bits` | `997` | Working precision for `BigFloat` only. It does not affect fixed-width `Float64x4`. |
| `convert_inputs` | `false` | Normalize independent `BigFloat` storage to `precision_bits`. This cannot recover digits already lost when the source was created. |
| `equilibrate` | `false` | Apply PSD-block diagonal congruence scaling and variable scaling before solving. Dense and sparse coefficient storage are supported. |
| `scaling` | `:auto` | Pipeline selector: LP geometric scaling for the dedicated LP path; for SDP, Ruiz scaling when `equilibrate=true`, otherwise none. |
| `sparse` | `:auto` | Storage selection used during ingestion. `:auto` distinguishes sparse coefficient storage from aggregate PSD and Schur density; `true`/`:sparse` and `false`/`:dense` force a path. |
| `extended_precision_blas` | `:off` | Extended-precision Schur backend: `:off`, conservative `:auto`, or diagnostic `:on`. Float32/Float64 always retain their existing BLAS route. |
| `extended_precision_memory_fraction` | `0.10` | Maximum fraction of currently available memory that the crossover may reserve for packed extended-precision panels. The cap respects host free memory, cgroups, and `SDPX_MEMORY_LIMIT_BYTES`, and conservatively keeps half of reported free memory outside the packing budget. |
| `mixed_precision_kkt` | `:off` | Opt-in dense KKT acceleration for BigFloat and fixed-width extended types: `:off`, guarded `:auto`, or size-override `:on`. Float64 is never redirected. |
| `mixed_precision_condition_limit` | `1e8` | Maximum conservative Float64 condition estimate accepted for mixed KKT refinement. |
| `mixed_precision_refine_max_steps` | `32` | Maximum target-precision correction solves before native extended-precision fallback. |
| `mixed_precision_memory_fraction` | `0.10` | Maximum fraction of reliably available memory used for persistent Float64 factors and conversion scratch. |
| `force_gc` | `false` | Retained A/B compatibility field. The main solve path does not currently read it. |

Sparse equilibration rebuilds the derived sparse caches after scaling, so the
following combination is supported:

```julia
prob = ingest(c, A, C, B, b; sparse=true)
opts = SolverOptions{T}(equilibrate=true)
```

Warm starts are supplied in original input coordinates. SDPX maps `x0`, `X0`,
and `Y0` through the selected equilibration and maps `y0` through equality
presolve as needed; callers should not pre-scale them.

## Output, timing, and checkpoints

| Parameter | Default | Meaning |
|---|---:|---|
| `verbosity` | `1` | `0` is silent; values of `1` or higher print iteration information. |
| `timing` | `false` | Records total and phase-level timing, including residual, Schur, KKT, predictor, corrector, line-search, and update phases. |
| `checkpoint_every` | `0` | Save an iterate-level warm-restart checkpoint every N iterations; `0` disables checkpointing. |
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
2. The legacy wrapper derives its restart limit from `maxOmega/OmegaStep`.
   Their defaults produce `max_restarts=10`, while directly constructing
   `SolverOptions` defaults to `5`.

## Recommended settings for sparse CSDR problems

Use automatic selection for the optimized many-`2x2`-block family:

```julia
parameter_policy=:auto
parameter_strategy=:fixed
predictor=:sdpb
max_restarts=10
refine_steps=1
sparse=:auto
equilibrate=false
```

With `parameter_policy=:auto`, the arrow profile chooses `Ωp=Ωd` from the
problem scale (at least 10 and otherwise approximately the maximum
PSD-block infinity norm). Set `parameter_policy=:fixed` when benchmarking an
explicit `Ωp`/`Ωd`; otherwise the structural profile intentionally overrides
those fields.

The zero-probe policy currently selects:

| Maximum active variables per `2x2` block | `β` | `γ` |
|---:|---:|---:|
| 1 to 6 | `0.1` | `0.85` |
| 7 to 14 | `0.1` | `0.8` |
| 15 or more | `0.01` | `0.85` |

This is an empirical structural policy for the tested sparse block-arrow CSDR
family, not a universal replacement for fixed parameters. Problems outside
that shape retain the supplied `β` and `γ`.

For `BigFloat` accuracy runs on the same problem, `β=0.1, γ=0.75` was more
stable at tolerances from `1e-12` through `1e-30`.

The extended-precision matrix kernels remain opt-in:

```julia
opts = SolverOptions{Float64x4}(
    extended_precision_blas=:auto,
    extended_precision_memory_fraction=0.10,
)
```

Automatic mode accounts for arithmetic type, packed dimensions, coefficient
and active density, expected Schur density, Julia thread count, and the memory
budget. It retains the sparse outer-product route when packing is not
predicted to amortize. Exact-arrow models containing only `2x2` PSD blocks use
the fused direct kernel instead for both Float64x4 and BigFloat, regardless of
this packing selector. In that case diagnostics report
`gram_kernel=:fused_arrow_2x2` and
`gram_kernel_reason=:fused_arrow_specialized`, and no transformed panels or
pair buffers are allocated.

Dense, non-arrow high-precision problems can separately opt into guarded
mixed-precision KKT factorization:

```julia
opts = SolverOptions{Float64x4}(
    mixed_precision_kkt=:auto,
)
```

`:auto` requires at least 256 Schur variables; `:on` removes only that size
crossover. Both modes retain all memory, finiteness, rank, condition, and
accuracy guards. The Float64 factor is accepted only when target-precision
residual refinement is predicted to fit the configured step cap. A relative
predictor residual above `1e-8` or stalled refinement recomputes the direction
with the native factorization. Dynamic failures use a two-iteration cooldown
and disable mixed precision after two failures. Static factor/condition
rejections use the same cooldown and disable it after three repeated
rejections. The feature remains off by default pending large full-solve
validation. `result.termination.mixed_precision_kkt` records whether the path
was available and active, its final reason, condition/step estimates, attempt
counts, cooldown, and dynamic/static fallback counts.
