# Precision guide

SDPX solves at whatever element type `T` your input arrays (`A`, `C`, `B`, `b`, `c`) carry — build them at the type you want, no global switch required (the legacy `setArithmeticType` still works for all-`Int`/`Rational` inputs, which have no type of their own to infer from).

## Choosing a type

| Type | Significand | ≈ decimal digits | Notes |
|---|---|---|---|
| `Float64` | 53 bits | ~16 | fastest; fine when you don't need more than double precision |
| `Float64x2` (MultiFloats.jl) | ~106 bits | ~32 | bitstype, zero GC pressure, threads normally |
| `Float64x4` (MultiFloats.jl) | ~212 bits | ~64 | sweet spot for many EFT/modular-bootstrap runs |
| `Double64` (DoubleFloats.jl) | ~106 bits | ~32 | alternative to Float64x2; has `exp`/`log` if ever needed |
| `Float64x{6}`/`Float64x{8}` (MultiFloats.jl) | ~318/~424 bits | ~96/~127 | near SDPB's common 448-bit band |
| `BigFloat` | `precision_bits` option (default 997 ≈ 300 decimal digits) | arbitrary | arbitrary precision, but **not thread-safe across OS threads** (see below) and allocates per scalar operation |

`MultiFloats.jl`/`DoubleFloats.jl` types are enabled automatically once you `using MultiFloats` / `using DoubleFloats` in your session (package extensions) — no other change needed.

## `BigFloat` precision plumbing

`prec` (legacy, base-10 digits) and `precision_bits` (new API, bits) both control the *working* precision of the solve, via `setprecision`. This is independent of whatever precision your *input data* happens to carry — if you construct `A`/`C`/... at 256 bits and then solve with `precision_bits=997`, the extra 741 bits carry no real information; `solve!` warns about this automatically. Pass `convert_inputs=true` (or `sdp(...; equilibrate=..., convert_inputs=true)`) to re-round the input data to the working precision explicitly.

## Tolerance vs. precision

`ϵ_gap` below roughly `100·eps(T)` is usually unreachable — `solve!` warns up front when this looks like the case, and the failure mode is an honest `MaxRestartsExceeded`/`IterLimit` status rather than a silent spin. As a rule of thumb, reaching a duality gap of `10^-d` needs roughly `d·log2(10) ≈ 3.32·d` significand bits of headroom beyond the noise floor of the linear algebra itself (which is usually a few bits, not the dominant term for well-conditioned problems).

## Dynamic range

`MultiFloats.jl` and `DoubleFloats.jl` types inherit `Float64`'s exponent range (~10±308). Raw high-degree-polynomial bootstrap sample data can exceed this. `solve!` detects a non-finite iterate and reports `NumericalBreakdown` with a message pointing at `equilibrate=true` (which routinely cuts several orders of magnitude off the dynamic range of badly-scaled data) or falling back to `BigFloat`.
