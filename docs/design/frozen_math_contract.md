# Frozen mathematical contract (PR-00)

**Frozen: 2026-09-11. Package version 0.6.1. HEAD at freeze: `e49f88b`.**

This file is the PR-00 (Clarabel-borrowing plan) freeze of the executable
mathematical contract. Every later PR must be measured against *this* text,
not against an older design document. Source of truth is the running code; the
line references below are the executable definition.

## 1. Five HSD Newton equations (authoritative)

`src/kkt/system.jl` (`HSDNewtonRHS`, and the residual assembly in
`newton_residual!`):

```text
A*dx  + ds  - b*dτ   = r_p      (primal affine)
A'*dy + c*dτ         = r_d      (dual affine)
-c'*dx - b'*dy + dκ  = r_g      (homogeneous gap)
ds + H*dy            = h        (cone complementarity)
κ*dτ + τ*dκ          = r_t      (scalar tau/kappa)
```

### Correction to the plan's Section 3.1 (F08 class)

The plan's Section 3.1 states the sign convention is

```text
cᵀ dx + bᵀ dy + dκ    = r_g
```

That is **not** what this source does, and the plan attributes the claim to
"the current `HSDNewtonRHS`[S09]". The executable source uses the **negative**
signs:

```text
-c'*dx - b'*dy + dκ   = r_g
```

confirming both the `HSDNewtonRHS` docstring and the arithmetic at
`src/kkt/system.jl` (`gap = dkappa - r_g; gap -= c[j]*dx[j]; gap -= b[i]*dy[i]`).

The plan's own Section 6 therefore applies to itself: a new test must build its
reference from the current executable equations, not from a plan document. This
note is the correction of record. No production sign was changed; only the
documentation claim is refused.

## 2. The KKT operator the elimination is stated over

The plan's `K = [0 A'; A -H]` and its scalar closure are consistent with the
signs above once `r_g` is read with the minus signs; the closure identity itself
was not independently re-derived here and is not asserted by PR-00.

## 3. Canonical form and cone order

```text
min c'x   s.t.  A x + s = b,  s ∈ K = K_1 × ... × K_q
```

* `CanonicalConicProgram` is the canonical IR; block order is the
  `ConeProductLayout` order, and `layout_blocks` is the only iteration order.
* `in_canonical_cone` is the membership authority (see
  `test/certificate_layout_storage.jl`).

## 4. Original-coordinate certificate

* Relative gap (restored, frozen): `|p - d| / max(1, (|p| + |d|)/2)`.
  Pinned by `test/gap_normalization.jl`; the weaker `1 + |p| + |d|`
  denominator was rejected.
* Certificate validity is fail-closed: an invalid original-coordinate
  certificate downgrades an otherwise optimal core result to
  `NumericalFailure` / `:original_coordinate_certificate_failed`.

## 5. PSD / svec / SOC normalization

Not re-derived in PR-00. Recorded as **open** so no later PR can claim they were
frozen here. `_pack_svec!` in `src/cones/symmetric/SymmetricCones.jl` remains
the executable reference.

## 6. What PR-00 did not establish

This freeze is a documentation and counting change. It did not run the full
package suite, did not establish multi-precision equivalence, and does not
assert the plan's Section 3.1 closure identity. Those remain PR-00 follow-ups
and later-PR gates.
