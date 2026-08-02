# SDPX Solver Parameters

These defaults come from `SolverOptions{T}`. Differences in the legacy
`sdp(...)` wrapper are listed separately.

## Interior-point method and initialization

| Parameter | Default | Meaning |
|---|---:|---|
| `β` | `0.1` | Fixed SDP centering/complementarity reduction target and the safe fallback for adaptive `sigma`. Smaller values are usually more aggressive. |
| `γ` | `0.9` | Fixed backtracking reduction factor and exact fraction-to-boundary safety. Adaptive mode separates these roles. |
| `Ωp` | `1` | Initial primal PSD matrices: `X_l=Ωp*I`. |
| `Ωd` | `1` | Initial dual PSD matrices: `Y_l=Ωd*I`. |
| `predictor` | `:classic` | Predictor rule: `:classic` or `:sdpb`. |
| `refine_steps` | `1` | Number of iterative-refinement passes for the KKT predictor/corrector solutions. |
| `step_rule` | `:auto` | `:backtrack`, exact `2x2`-optimized `:fraction_to_boundary`, or `:auto` (fraction-to-boundary for blocks up to `2x2`, backtracking otherwise). |
| `parameter_policy` | `:auto` | Cold-start structural policy. `:auto` may select calibrated initial `β`, `γ`, `Ωp`, and `Ωd`; `:fixed` preserves the supplied values. |
| `parameter_strategy` | `:adaptive` | Guarded per-iteration Mehrotra policy: bounded `sigma`, independent primal/dual fractions, adaptive backtracking, refinement, minimum-step and restart scales. It automatically uses the fixed fallback when cold-start or stability diagnostics are unreliable. |
| `adaptive_sigma_max` | `0` | Expert adaptive-centering cap. Zero selects by structure: 0.20 for the calibrated `large_lattice_dense_schur` profile and the generic 0.50 bound otherwise. A positive override is never allowed below the fixed fallback `β`. |
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
| `working_precision_policy` | `:auto` | BigFloat policy. It may start at a conservatively selected lower precision and retries at `precision_bits` unless the first result passes original-coordinate certification. |
| `minimum_working_precision_bits` | `192` | Lower bound for the staged BigFloat selector. The requested `precision_bits` remains the upper bound and fallback. |
| `convert_inputs` | `false` | Normalize independent `BigFloat` storage to `precision_bits`. This cannot recover digits already lost when the source was created. |
| `equilibrate` | `false` | Expert compatibility flag for the core. Public `scaling=:auto` takes precedence and applies the selected pipeline scaling. |
| `scaling` | `:auto` | LP geometric scaling for the dedicated LP path and adaptive-pass Ruiz congruence/variable scaling for SDP. The calibrated `large_lattice_dense_schur` profile keeps original coordinates when paired with `parameter_strategy=:fixed`; its historical fixed parameters stall under Ruiz. Explicit `:none` and `:equilibrate` remain expert overrides. |
| `sparse` | `:auto` | Storage selection used during ingestion. `:auto` distinguishes sparse coefficient storage from aggregate PSD and Schur density; `true`/`:sparse` and `false`/`:dense` force a path. |
| `extended_precision_blas` | type-dependent | Extended-precision Schur backend: conservative `:auto` for `Float64x4` and `BigFloat`, `:off` for other arithmetic types, or diagnostic `:on`. Float32/Float64 always retain their existing BLAS route. Native BigFloat parallelism is limited to ownership-safe exact reduced-arrow panels and Schur tiles. |
| `extended_precision_memory_fraction` | `0.10` | Maximum fraction of currently available memory that the crossover may reserve for packed extended-precision panels. The cap respects host free memory, cgroups, and `SDPX_MEMORY_LIMIT_BYTES`, and conservatively keeps half of reported free memory outside the packing budget. |
| `mixed_precision_kkt` | type-dependent | `:auto` for `BigFloat` and fixed-width extended arithmetic, `:off` for Float64. Dense problems may use a guarded Float64 factorization; exact singleton-arrow BigFloat problems may use a guarded Float64x4 reduced factorization when MultiFloats is loaded. Every rejection falls back to native arithmetic. |
| `equality_solver` | `:auto` | Uses normal equations while they are stable and switches eligible dense equality systems to rank-revealing QR when factor diagnostics fail. Exactly block-diagonal sparse systems transform `B` with local factors and automatically use blocked triangular extended-precision SYRK when worthwhile. `:normal_equations` and `:qr` are expert overrides. Large systems are protected by dimension and memory crossovers. |
| `mixed_precision_condition_limit` | type-dependent | Maximum conservative Float64 condition estimate accepted for mixed KKT refinement: `1e14` for `Float64x4`, `1e8` otherwise. The predicted correction budget and measured target-precision contraction remain authoritative guards. |
| `mixed_precision_refine_max_steps` | `32` | Maximum target-precision correction solves before native extended-precision fallback. |
| `mixed_precision_memory_fraction` | `0.10` | Maximum fraction of reliably available memory used for persistent Float64 factors and conversion scratch. |
| `force_gc` | `false` | Run a full collection after each accepted iteration; on glibc Linux, also return free allocator pages to the OS. This can reduce retained allocator RSS for very large sparse factorizations and multi-RHS solves, but it cannot reduce the live factorization-local high-water mark and adds synchronization and collection time. |

Unless `refine_tol` is explicitly positive, dense mixed-precision refinement
uses `max(64 * eps(T), min(ϵ_gap, ϵ_primal, ϵ_dual)^2)`. This remains much
tighter than the requested certificate without requiring every unused bit of
the target arithmetic. Predictor solves have a separate `1e-8` relative
residual guard and may perform bounded corrections before falling back.

Native KKT refinement normally targets `64 * eps(T)`. The large regularized
sparse Float64 SDP route instead uses
`max(64 * eps(Float64), min(ϵ_gap, ϵ_primal, ϵ_dual) / 100)`, retaining two
guard digits beyond the requested certificate. This avoids repeated
multi-gigabyte sparse residual products for digits that cannot affect the
requested result. A positive user-supplied `refine_tol` always takes
precedence.

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

The full Newton-method audit, exact diagnostic fields, controller bounds,
fallback rules, arithmetic behavior, and fixed-versus-adaptive results are in
[Adaptive Interior-Point Parameter Policy](adaptive-parameter-policy.md).

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
problem scale. The default large-problem rule is at least 10 and otherwise
`10·max_l ||C_l||_inf`. A wide arrow with at most 256 active variables per
block and `max_l ||C_l||_inf <= 10` instead uses the validated
`5·floor(max_l ||C_l||_inf)` rule, with a floor of 10. The grid quantization
avoids an observed unstable interval in the medium CSDR case and selects its
faster validated point. Set `parameter_policy=:fixed` when
benchmarking an explicit `Ωp`/`Ωd`; otherwise the structural profile
intentionally overrides those fields.

The zero-probe policy currently selects:

| Maximum active variables per `2x2` block | `β` | `γ` |
|---:|---:|---:|
| 1 to 6 | `0.1` | `0.85` |
| 7 to 14 | `0.1` | `0.8` |
| 15 to 256 | `0.1` | `0.85` |
| 257 or more | `0.01` | `0.85` |

This is an empirical structural policy for the tested sparse block-arrow CSDR
family, not a universal replacement for fixed parameters. Problems outside
that shape retain the supplied `β` and `γ`.

For `BigFloat` accuracy runs below a `1e-10` tolerance, the automatic policy
keeps `β=0.1` and caps `γ` at `0.75`; at `1e-10`, the structural profile's
validated `γ=0.85` is retained.

Extended-precision matrix kernels use a conservative automatic crossover by
default for `Float64x4`, other wider immutable arithmetic, and `BigFloat`:

```julia
opts = SolverOptions{Float64x4}(
    extended_precision_blas=:auto,
    extended_precision_memory_fraction=0.10,
)
```

Automatic mode accounts for arithmetic type, packed dimensions, coefficient
and active density, expected Schur density, Julia thread count, and the memory
budget. It retains the sparse outer-product route when packing is not
predicted to amortize. Exact singleton-local `2x2` Float64x4 arrows use a
dedicated crossover: at least 32 shared columns, at least `2e5`
two-row-panel pair operations, at least 0.20 expected shared-Schur density, at
least 0.10 shared active density, at least 1.18 predicted speedup, and a panel
within the configured memory budget. A host calibration may adjust the
column, work, shared-Schur-density, and speedup thresholds. Rejected cases
retain the fused direct kernel and allocate neither transformed panels nor
pair buffers. Selected diagnostics report
`gram_kernel=:reduced_arrow_syrk`,
`:reduced_arrow_threaded_syrk`, or `:fused_arrow_2x2`.

Dense, non-arrow high-precision problems can separately override the guarded
mixed-precision KKT factorization:

```julia
opts = SolverOptions{Float64x4}(
    mixed_precision_kkt=:auto,
)
```

`:auto` requires at least 256 Schur variables and retains the memory,
finiteness, rank, condition, predicted-correction-count, and accuracy guards.
For fixed-width extended arithmetic, explicit `:on` removes the size cutoff
and makes the conservative condition and predicted-step estimates diagnostic:
the measured target-precision residual decrease is the acceptance test.
BigFloat keeps the static condition cutoff in both modes. A relative predictor
residual above `1e-8` that cannot be reduced monotonically, or stalled
corrector refinement, recomputes the direction with the native factorization.
Dynamic failures use a two-iteration cooldown and disable mixed precision
after two failures. Static factor/condition rejections use the same cooldown
and disable it after three repeated rejections. The feature remains off by
default pending broader full-solve validation.
`result.termination.mixed_precision_kkt` records whether the path was
available and active, its final reason, condition/step estimates, measured
predictor corrections, attempt counts, cooldown, and dynamic/static fallback
counts.

For an exact singleton-local BigFloat arrow, loading `MultiFloats` and setting
`mixed_precision_kkt=:on` instead selects
`backend=:float64x4_reduced_arrow`. Exact BigFloat coefficient metrics,
residuals, and refinement are retained; only the reduced shared panel and
factor use Float64x4. Diagnostics record the panel worker count and native
fallback reason. This path remains off by default because the medium CSDR
validation improved the KKT subphase without a clear total-runtime gain.
