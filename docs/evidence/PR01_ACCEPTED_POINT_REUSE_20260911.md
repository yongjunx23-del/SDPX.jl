# PR-01 evidence: accepted-point residual lifecycle token

Date: 2026-09-11. Baseline: PR-05.

## The claim being tested

`product_hsd_step!` computed the accepted-point residual twice: once at the end
of a step (`product_cone_hsd.jl`, trailing) and again at the start of the next
step (entry). Between them the accepted iterate does not move on the common
path, so the second computation is redundant.

## Why the obvious implementation is unsound

A static map of every residual and scaling writer (see the section below) found
that the residual is **not** always what the canonical kernel would produce.
`_cert_residual!` (`src/certificates/certificates.jl`) also writes `rP`/`rD`,
reached from `verify_optimal!` during the per-step certificate check, and it uses
a **different accumulation association**:

| | canonical (`hsd_residual!`) | certificate (`_cert_residual!`) |
|---|---|---|
| `rP` | `mul!(A*x)` then add `s - b*τ` | seed with `s - b*τ`, then accumulate `A*x` |
| `rD` | accumulate `A'y` then add `c*τ` | seed with `c[j]*τ`, then accumulate `A'y` |

Mathematically equal; **not bitwise equal**. The Newton direction build consumes
`rP`/`rD`, so treating a certificate-flavoured residual as canonical would
change the trajectory. A "skip if nothing changed the point" flag alone is
therefore wrong.

Measured disagreement (Float64, `kkt_derived_start!` point):

| problem | max abs diff `rP` | max abs diff `rD` |
|---|---|---|
| SOC k=3/8/32 | 0.0 (agree) | 0.0 (agree) |
| LP n=20, m=5 | 4.4e-16 | 1.3e-15 |
| LP n=50, m=10 | 2.6e-15 | 5.3e-15 |
| LP n=200, m=40 | 2.2e-14 | 9.1e-14 |
| LP n=500, m=100 | 1.0e-13 | 2.3e-13 |

Note the SOC cases agree bitwise — which is exactly why a test on an SOC
instance alone would have declared the naive approach safe. The LP cases show
it is not.

## The implementation

A lifecycle token on `HSDState` (not `ProductConeHSDState` — `_cert_residual!`
receives the former, so a token it cannot clear would be unsound by
construction):

- `point_epoch` — bumped by every write to `x`/`y`/`s`/`tau`/`kappa`;
- `residual_epoch` — the `point_epoch` the cached residual was computed from;
- `residual_canonical` — which kernel produced it.

`_hsd_residual_is_fresh` requires **both** that the point has not moved **and**
that the last writer was canonical. `_cert_residual!` clears the canonical mark.
The entry residual is computed only when the token is not fresh.

Instrumented write sites (each `_product_hsd_bump_point_epoch!`):
`linesearch.jl` acceptance commit; `product_cone_solve.jl` cold refinement
(write and restore), terminal trial (commit and restore), rank-ray;
`product_cone_hsd.jl` cold start. The canonical marks are set by
`hsd_residual!`, `_fixed_trace_hsd_residual!` and `_product_hsd_residual!`.

## What was verified

`test/accepted_point_reuse.jl`, 55 assertions:

1. a fresh state never claims freshness (the first entry residual always runs);
2. the canonical kernel marks fresh, a bump clears it, re-running restores it;
3. **negative control**: `_cert_residual!` clears the mark and really does
   change `rP`/`rD` bitwise on an LP (the justification for the field);
4. the token survives a real end-to-end solve;
5. **the invariant itself**: driving `product_hsd_step!` by hand, whenever the
   token claims freshness the canonical residual is recomputed from scratch and
   required to match `rP`, `rD` and `mu` **bitwise**. The test also asserts the
   invariant was actually exercised (`checked > 0`), so it cannot pass
   vacuously on a token that never claims freshness.

## Not claimed

- No wall-clock or allocation improvement is claimed here. The redundant
  computation is removed; its cost was not measured, and on small problems it is
  a fraction of a step.
- The line-search-side scaling duplication (`try_update_scaling!` on the
  accepted point) is **not** addressed. The analysis found it is only
  bit-identical for orthant/SOC/PSD blocks; for Exp/Power the call is not an
  identity because the conjugate path warm-starts from the previous accepted
  state. That needs its own equivalence evidence and is left for a follow-up.
- Bit-identity of the overall trajectory is asserted only through the token
  invariant and the existing suite, not by comparing full iterate sequences
  against the pre-PR-01 build.
