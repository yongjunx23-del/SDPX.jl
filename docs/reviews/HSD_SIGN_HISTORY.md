# HSD Sign History — why the third equation was corrected

**Status:** Historical review note (non-authoritative).  The authoritative
frozen spec is `docs/design/CANONICAL_FORM.md` §2; the derivation companion is
`docs/design/HSD_FORMULATION.md`.  This document records *why* the sign was
changed, so the reasoning survives as context and is never re-litigated as an
open question in production code.

## The frozen (first) convention

The first frozen HSD third equation was written as

```
(G)   c'x + b'y + κ = 0
```

with the homogeneous variables `τ, κ ≥ 0`.  Rearranged this gives

```
κ = −(c'x + b'y).
```

## Why that is inconsistent with an interior point

Weak duality (§1.2 of `HSD_FORMULATION.md`) gives, for any feasible primal
`(x, s)` and any feasible dual `y`,

```
c'x + b'y = s'y ≥ 0     (because s ∈ K, y ∈ K*, and Ax + s = b).
```

Therefore, at any *feasible* point, the first convention forces

```
κ = −(c'x + b'y) = −s'y ≤ 0.
```

But the homogeneous embedding also requires `κ ≥ 0`.  Stacking the two gives
`κ ≤ 0` and `κ ≥ 0`, so `κ = 0` at every feasible point — there is no strictly
interior (all variables strictly positive) central point.  In particular a
central-path point with `s > 0`, `y > 0`, `τ > 0`, `κ > 0` cannot exist while
`c'x + b'y = s'y > 0`.  The interior of the scalar cone `(τ, κ) ∈ R_+²` is
empty under feasible points in this convention.

## The correction

The standard production HSD (Nesterov–Todd style, as used by MOSEK and the
homogeneous Algorithm of Andersen–Andersen) resolves this by negating the third
equation:

```
(G)   −c'x − b'y + κ = 0
```

so that

```
κ = c'x + b'y = s'y ≥ 0,
```

which is consistent with `κ ≥ 0` and admits a genuinely strictly-interior
central point.  This is the convention frozen in `CANONICAL_FORM.md` and
implemented in `src/hsd/hsd.jl` (`hsd_gap_residual` returns
`−c'x − b'y + κ`).

## The skew-operator consequence

With the corrected sign, the skew operator `Q (x; y; τ) = (0; s; −κ)` (note the
`−κ` in the third component), and the complementarity derived from
`(x,y,τ)·Q(x,y,τ) = 0` becomes exactly

```
s'y + τκ = 0
```

at a solution, which with `s ∈ K`, `y ∈ K*`, `τ, κ ≥ 0` forces the clean
complementarity conditions `s'y = 0`, `τκ = 0`.

## The forbidden pairings

No production code anywhere in the HSD core may use `dot(s, x)`, `b's`, or
`(x, τ) ∈ K*`.  These were the root errors of a previous draft: `s ∈ R^m` and
`x ∈ R^n` cannot be paired when `m ≠ n`, and `x` is free (no conjugate cone
partner).  The correct pairings are the conjugate `s'y` (slack × dual) and the
scalar `τκ`.

## How the fixtures were aligned

`test/hsd_equations.jl` (self-contained, no SDPX dependency) now checks the
corrected sign: `−c'x − b'y + κ = 0`, `Q·(x;y;τ) = (0; s; −κ)`, `κ = s'y ≥ 0`
at feasible points, and strictly-interior points with `κ > 0`.  The earlier
worked fixtures that constructed a negative `κ` "interior" point were
deliberately superseded: such a point satisfies the stale algebra but is not a
valid interior point of the frozen cone, and its `κ < 0` violates the `κ ≥ 0`
requirement.
