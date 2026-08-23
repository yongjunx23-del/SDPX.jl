# Precision

SDPX solves in the arithmetic owned by its typed `Model{T}`. Construct the
model with `Model(Float64)`, `Model(BigFloat; precision_bits=...)`, or a loaded
fixed-width type such as `Model(Float64x4)`. There is no process-global
arithmetic selector and no implicit conversion from an untyped model.

## Choosing a type

| Type | Significand | ≈ decimal digits | Notes |
|---|---|---|---|
| `Float64` | 53 bits | ~16 | fastest; fine when you do not need more than double precision |
| `Float64x2` (MultiFloats.jl) | 105 bits | ~31 | bitstype, zero GC pressure, threads normally |
| `Float64x4` (MultiFloats.jl) | 209 bits | ~62 | sweet spot for many EFT/modular-bootstrap runs |
| `BigFloat` | model `precision_bits` | arbitrary | arbitrary precision; `Model(BigFloat)` defaults to 256 bits |

`MultiFloats.jl` types are enabled automatically once you `using MultiFloats`
in your session, with no other change needed.

## BigFloat precision plumbing

`Model(BigFloat; precision_bits=bits)` owns the model and solve precision.
Public modeling operations copy coefficients, right-hand sides, starts, and
objective data inside that precision scope. Existing `BigFloat` values still
carry the precision at which they were originally created: copying a rounded
value into a wider model cannot recover missing digits. Build source data
inside `setprecision(BigFloat, bits) do ... end` whenever those digits matter.

## Staged working precision

BigFloat callers select the staged working-precision policy through typed
public settings:

```julia
model = Model(BigFloat; precision_bits=256)
settings = Settings{BigFloat}(working_precision_policy=:auto)
result = optimize!(model; settings=settings)
```

The selector combines the smallest requested tolerance, a 96-bit numerical
guard, and a dimension term, rounds upward to 32 bits, and clamps the result to
the predeclared ladder. A lower-precision result is accepted only if normal
original-coordinate certification succeeds. Precision exhaustion, stagnation,
an almost-optimal status, or a numerical failure advances to the requested
model precision when the predeclared policy and remaining wall time allow it.

The policy defaults to `:auto`; `:fixed` remains the expert override. See the
[native BigFloat report](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/evidence/bench/opt2026/BIGFLOAT_NATIVE_OPTIMIZATION_2026-07-26.md)
for measured staged-precision behavior.

## Guarded mixed-precision KKT solves

`mixed_precision_kkt` defaults to `:auto` for `BigFloat` and fixed-width types
wider than `Float64`, and to `:off` for `Float64`. When enabled, SDPX may
attempt a lower-precision factor only if crossover and conditioning guards
permit it, then computes residuals and accumulates corrections in the
requested arithmetic.

This is a guarded accelerator rather than a change in the numerical contract:
loss of positive definiteness or rank during conversion, a conservative
condition estimate above the configured limit, an excessive predicted
correction count, non-finite conversion, or failed target-precision
refinement switches back to the native factorization and recomputes the
direction. Explicit `:on` for fixed-width arithmetic is a measured expert
mode. BigFloat and `:auto` retain static cutoffs. Repeated rejection is
cooled down and eventually disabled for that solve. Exact singleton-local
`2×2` BigFloat arrows have a separate guarded path when `MultiFloats` is
loaded. See
[`bench/mixed_precision_kkt/RESULTS.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/evidence/bench/mixed_precision_kkt/RESULTS.md)
for the current promotion evidence and exact thresholds.

## Tolerance vs. precision

`ϵ_gap` below roughly `100·eps(T)` is usually unreachable: `solve!` warns up
front when this looks like the case. If progress reaches the arithmetic floor,
the solver returns a non-optimal status with structured precision-floor or
stagnation diagnostics instead of silently spinning. As a rule of thumb,
reaching a duality gap of `10^-d` needs roughly `d·log2(10) ≈ 3.32·d`
significand bits of headroom beyond the noise floor of the linear algebra
itself.

## Dynamic range

`MultiFloats.jl` types inherit `Float64`'s exponent range (~10±308). Raw
high-degree-polynomial bootstrap sample data can exceed
this. `solve!` detects a non-finite iterate and reports `NumericalBreakdown`
with a message pointing at `scaling=:equilibrate` (which routinely cuts several
orders of magnitude off the dynamic range of badly scaled data) or falling
back to `BigFloat`.

BigFloat values are mutable, so general native BigFloat kernels are serial;
ownership-safe independent blocks, exact local arrow phases, and disjoint
Schur tiles may use requested workers. Fixed-width types can use Julia threads
more broadly.
