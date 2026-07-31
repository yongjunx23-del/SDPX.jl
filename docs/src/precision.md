# Precision

SDPX follows the element type of the input data. There is no process-global
arithmetic mode in the typed API.

| Type | Typical use |
| --- | --- |
| `Float64` | fastest baseline and ordinary well-scaled models |
| `MultiFloats.Float64x4` | fixed-width extended precision with SIMD-friendly, ownership-safe threaded kernels |
| `BigFloat` | arbitrary precision and models that exceed the exponent range or accuracy of fixed-width arithmetic |

Create BigFloat model data inside the desired precision scope:

```julia
setprecision(BigFloat, 256) do
    c = BigFloat[1, 2]
    # Construct A, C, B, and b here as BigFloat values too.
end
```

Increasing `precision_bits` after coefficients were rounded cannot recover
lost digits. `convert_inputs=true` normalizes MPFR storage precision but does
not recreate the original data.

BigFloat uses conservative staged working precision by default and retries at
the requested width when original-coordinate certification fails. Mixed
precision KKT factorization is guarded by condition, memory, and refinement
tests; rejection falls back to native arithmetic.

Float64x4 supports broader threaded blocked kernels. Native BigFloat
parallelism is limited to disjoint output tiles or complete independently
owned blocks because MPFR scalar objects are mutable.

See the
[full precision guide](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/precision.md)
for crossover rules, mixed-precision fallback details, and measured results.
