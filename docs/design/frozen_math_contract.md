# Frozen mathematical contract (PR-00)

**Frozen: 2026-09-11. Package version 0.6.1.**

**Rebased 2026-09-11 onto `origin/main` @ `4b46cda`** (see
`docs/evidence/BASELINE_CORRECTION_20260911.md` for why the original freeze
point was on a stale base).

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

### RETRACTED: the sign "correction" that was published here

An earlier version of this file claimed the plan's Section 3.1 sign convention
was wrong and that the source used `-c'*dx - b'*dy + dκ`. **That claim is
retracted.**

At the plan's own baseline (`2ab596f`, on `origin/main`) and at the current
`origin/main` (`4b46cda`) the executable source reads:

```text
c'*dx + b'*dy + dκ     = homogeneous_gap
```

with `gap = dkappa - rhs; gap += c[j]*dx[j]; gap += b[i]*dy[i]`.

The minus signs the earlier round observed were real, but only at the stale local
base `db42fd2`. The sign was changed upstream by
`3392e24 kkt: define semantic five-equation Newton system`, which landed after
`db42fd2` and before the plan's baseline. **The plan was right; the reviewer was
reading old code.**

The equations as frozen at the rebased base:

```text
A*dx  + ds  - b*dτ   = r_p      (primal affine)
A'*dy + c*dτ         = r_d      (dual affine)
c'*dx + b'*dy + dκ   = r_g      (homogeneous gap)
ds + H*dy            = h        (cone complementarity)
κ*dτ + τ*dκ          = r_t      (scalar tau/kappa)
```

See `docs/evidence/BASELINE_CORRECTION_20260911.md`.

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
