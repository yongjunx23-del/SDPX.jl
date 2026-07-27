# Precision guide

SDPX solves at whatever element type `T` your input arrays (`A`, `C`, `B`, `b`, `c`) carry — build them at the type you want, no global switch required (the legacy `setArithmeticType` still works for all-`Int`/`Rational` inputs, which have no type of their own to infer from).

## Choosing a type

| Type | Significand | ≈ decimal digits | Notes |
|---|---|---|---|
| `Float64` | 53 bits | ~16 | fastest; fine when you don't need more than double precision |
| `Float64x2` (MultiFloats.jl) | 105 bits | ~31 | bitstype, zero GC pressure, threads normally |
| `Float64x4` (MultiFloats.jl) | 209 bits | ~62 | sweet spot for many EFT/modular-bootstrap runs |
| `Double64` (DoubleFloats.jl) | ~106 bits | ~32 | alternative to Float64x2; has `exp`/`log` if ever needed |
| `Float64x{6}`/`Float64x{8}` (MultiFloats.jl) | 313/417 bits | ~94/~125 | near SDPB's common 448-bit band |
| `BigFloat` | `precision_bits` option | arbitrary | arbitrary precision; the convenience `solve` API defaults to 256 bits, while `SolverOptions` and the legacy API default to 997 bits (about 300 decimal digits); exact singleton-local `2x2` arrows can use an ownership-safe threaded native reduced Schur path |

`MultiFloats.jl`/`DoubleFloats.jl` types are enabled automatically once you `using MultiFloats` / `using DoubleFloats` in your session (package extensions) — no other change needed.

## `BigFloat` precision plumbing

`prec` (legacy, base-10 digits) and `precision_bits` (new API, bits) control the
*working* precision of the solve. The one-call and legacy interfaces also
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
between the configured floor and requested precision. A lower-precision result
is accepted only if normal original-coordinate certification succeeds.
Precision exhaustion, stagnation, `AlmostOptimal`, or a numerical failure
causes a retry at `precision_bits` when time remains. The failed iterate is
released before allocating the fallback workspace. Checkpoint resume bypasses
staging and uses the requested precision.

The policy defaults to `:fixed`. On the medium CSDR model, a fixed 192-bit
input/run reduced runtime from 87.168 to 80.703 seconds with the same 41
iterations and certified result. The actual staged run, retaining the
original 256-bit input objects, took 83.933 seconds and selected 192-bit
workspace arithmetic without a retry. That single-model 1.04x staged gain is
useful but is not broad enough evidence to reduce precision automatically for
all users. See the
[native BigFloat report](../bench/opt2026/BIGFLOAT_NATIVE_OPTIMIZATION_2026-07-26.md).

## Guarded mixed-precision KKT solves

Dense, non-arrow `BigFloat` and fixed-width extended-precision problems may
opt into `mixed_precision_kkt=:auto`. SDPX then factors the Schur and equality
complements in Float64 but computes residuals and accumulates corrections in
the requested arithmetic. Ordinary Float64 is never redirected.

This is a guarded accelerator rather than a change in the numerical contract:
loss of positive definiteness or rank during Float64 conversion, a conservative
condition estimate above `1e8` (`1e14` for Float64x4 automatic mode), an
excessive predicted correction count, non-finite conversion, or failed
target-precision refinement switches back to the native factorization and
recomputes the direction. Explicit `:on` for fixed-width arithmetic is a
measured expert mode: condition and predicted-step estimates remain visible,
but monotone target-precision residual correction makes the acceptance
decision. For `Float64x4`, a failed Float64 correction may promote the
preconditioner to `Float64x2` before paying for native `Float64x4`
factorization. That rung uses a blocked lower-triangular Cholesky, disjoint
output tiles, parallel `L^-1 B`, and a one-triangle equality SYRK. The factor
is rebuilt from the current Schur matrix rather than reused across outer
iterations; the measured stale-factor retry added work and passed none of its
Task_Low08 residual guards. Each promoted solve is checked in `Float64x4`
before acceptance, and failure falls back to native arithmetic. Its workspace
is allocated lazily and must pass the configured memory gate. BigFloat and
`:auto` retain the static cutoffs. Repeated rejection is cooled down and
eventually disabled for that solve. The default remains `:off`; see the
[mixed-precision KKT benchmark](../bench/mixed_precision_kkt/RESULTS.md) for
the current promotion evidence and exact thresholds.

Exact singleton-local `2x2` BigFloat arrows have a separate guarded path when
`MultiFloats` is loaded. With `mixed_precision_kkt=:on`, SDPX constructs the
three-dimensional coefficient metric, local diagonals, couplings, residuals,
and refinement corrections in BigFloat. It packs and factors only the reduced
shared system in Float64x4 and can parallelize independent block preparation
plus the blocked SYRK. Any panel, factorization, or refinement failure first
tries the exact native BigFloat reduced panel, then reconstructs the legacy
pairwise shared Schur matrix only if that exact representation is unavailable.
The backend is reported as
`backend=:float64x4_reduced_arrow` in termination diagnostics. It remains off
by default because the validated medium CSDR run improved KKT time but not
total BigFloat runtime.

## Tolerance vs. precision

`ϵ_gap` below roughly `100·eps(T)` is usually unreachable — `solve!` warns up
front when this looks like the case. If progress reaches the arithmetic floor,
the solver returns a non-optimal status with structured precision-floor or
stagnation diagnostics instead of silently spinning. As a rule of thumb,
reaching a duality gap of `10^-d` needs roughly
`d·log2(10) ≈ 3.32·d` significand bits of headroom beyond the noise floor of
the linear algebra itself (which is usually a few bits, not the dominant term
for well-conditioned problems).

## Dynamic range

`MultiFloats.jl` and `DoubleFloats.jl` types inherit `Float64`'s exponent range (~10±308). Raw high-degree-polynomial bootstrap sample data can exceed this. `solve!` detects a non-finite iterate and reports `NumericalBreakdown` with a message pointing at `equilibrate=true` (which routinely cuts several orders of magnitude off the dynamic range of badly-scaled data) or falling back to `BigFloat`.
