# Experimental n=2 SPD-relative eigensolver route (R0-S diagnostic)

Path: `src/cones/symmetric/eigen.jl` (`_relative2_offdiag_gate`,
`_relative2_jacobi_eigen!`), `src/cones/symmetric/psd.jl`
(`_psd_eigen_route!` route `:experimental_relative2`),
`src/cones/symmetric/types.jl` (constructor selection).
Test: `validation/scientific_core/test_psd_relative2.jl`. 45 assertions pass.

## Question

The production cyclic-Jacobi PSD eigensolver uses an ABSOLUTE rotation
threshold (`eps(T)*scale*max(1,n)*10/off_count`). For a matrix whose
off-diagonal entry is absolutely tiny but relatively large,
e.g. M = [1 δ; δ 2δ²] with δ = 2^-50 (|M12| = 8.9e-16 < 2.2e-15), the
rotation is skipped although the relative correlation
|M12|/√(M11·M22) = 1/√2 is O(1). Does a relative-accuracy local solver
change the downstream PSD NT scaling?

## What the experiment does

- `_relative2_offdiag_gate`: range-safe (exponent-separated, frexp-based,
  interval-bounded) proof of |b|/√(a·c) ≤ τ_off (τ_off = 10·n·eps),
  returning `:pass`/`:fail`/`:unresolved` (unresolved and nonfinite /
  non-positive inputs refuse).
- `_relative2_jacobi_eigen!`: dimension-two Float64-only rotate-or-accept
  using the reviewed bounded rotation
  (g = max(a,c,|b|), d = (c/g−a/g)/2, β = b/g,
  t = β/(d + copysign(hypot(d,β), d)), equal diagonals → t = 1,
  c₁ = 1/√(1+t²), s = t·c₁). Refuses nonfinite/nonpositive rotated
  diagonals. No fallback is permitted.
- Explicit route selection: `PSDNTScaling(2; eigen_route=:experimental_relative2)`
  (only Float64, only n=2; anything else refuses at use time).

## Results (Float64, both recorded and some gated)

| quantity | production absolute route | experimental relative route |
|---|---|---|
| rotation at δ=2^-50 dyadic | skipped (V = I) | performed |
| small eigenvalue w2 | 2δ² (2× off) | δ² (relative accuracy) |
| ‖D0·M·D0 − I‖∞, dyadic | **0.7071** (= ρ; production leaves the full relative contraction) | 7.9e-31 (near-exact; dyadic data is power-of-two representable) |
| ‖D0·M·D0 − I‖∞, δ=1.3·2^-50 | — | 1.1e-16 (eps-level rounding barrier) |

The relative-route D0 attains the relative contraction bound ρ = 1/√2 < 1
at the operator level, whereas the absolute route leaves the full ρ as a
residual. For non-power-of-two small eigenvalues the Float64 root assembly
leaves an eps-level residual in D0·M·D0 − I (the rounding barrier the
design predicted), which is recorded, not gated.

## Status

All three P1 defects found in review (signed-b gate, truncating-div on
negative odd exponent sums, silent unrepresentable-rotation zeroing) are
fixed with regression tests (negative off-diagonal eigensolve, negative odd
exponent-sum gate, unrepresentable-rotation refusal, extreme-separation
pass).

This is an opt-in research route (explicit selection only, Float64 n=2).
Production dispatch is unchanged; the missing production-scale evidence is
exactly the downstream NT-scaling behaviour for general near-boundary PSD
blocks under a relative-accuracy eigendecomposition, which remains an open
experimental question (R0-S continuing). No tolerance widening, hidden
precision, or fallback is involved.
