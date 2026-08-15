# Precision

SDPX solves at whatever element type `T` your input arrays (`A`, `C`, `B`,
`b`, `c`) carry. There is no process-global arithmetic mode in the typed API.
The legacy `setArithmeticType` still works for all-`Int`/`Rational` inputs,
which have no type of their own to infer from.

## Choosing a type

| Type | Significand | ≈ decimal digits | Notes |
|---|---|---|---|
| `Float64` | 53 bits | ~16 | fastest; fine when you do not need more than double precision |
| `Float64x2` (MultiFloats.jl) | 105 bits | ~31 | bitstype, zero GC pressure, threads normally |
| `Float64x4` (MultiFloats.jl) | 209 bits | ~62 | sweet spot for many EFT/modular-bootstrap runs |
| `BigFloat` | `precision_bits` option | arbitrary | arbitrary precision; the convenience `solve` API defaults to 256 bits, while `SolverOptions` and the legacy API default to 997 bits (about 300 decimal digits) |

`MultiFloats.jl` types are enabled automatically once you `using MultiFloats`
in your session, with no other change needed.

## BigFloat precision plumbing

`prec` (legacy, base-10 digits) and `precision_bits` (new API, bits) control
the *working* precision of the solve. The one-call and legacy interfaces also
convert exact `Int`/`Rational` inputs inside that precision scope. Existing
`BigFloat` input data still carries the precision at which it was originally
created: solving 256-bit data at 997 bits cannot recover the missing digits.
`SolverOptions{BigFloat}(convert_inputs=true)` can normalize every stored
scalar to the working precision, but it does not create information. To gain
accuracy, rebuild the source data inside
`setprecision(BigFloat, precision_bits) do ... end`.

## Staged working precision

BigFloat callers may opt into a conservative first-attempt precision:

```julia
options = SolverOptions{BigFloat}(
    precision_bits=256,
    working_precision_policy=:auto,
    minimum_working_precision_bits=192,
)
result = solve!(problem, options)
```

The selector combines the smallest requested tolerance, a 96-bit numerical
guard, and a dimension term, rounds upward to 32 bits, and clamps the result
between the configured floor and requested precision. A lower-precision
result is accepted only if normal original-coordinate certification succeeds.
Precision exhaustion, stagnation, `AlmostOptimal`, or a numerical failure
causes a retry at `precision_bits` when time remains. Checkpoint resume
bypasses staging and uses the requested precision.

The policy defaults to `:auto`; `:fixed` remains the expert override. See the
[native BigFloat report](https://github.com/yongjunx23-del/SDPX.jl/blob/main/bench/opt2026/BIGFLOAT_NATIVE_OPTIMIZATION_2026-07-26.md)
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
[`bench/mixed_precision_kkt/RESULTS.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/bench/mixed_precision_kkt/RESULTS.md)
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
