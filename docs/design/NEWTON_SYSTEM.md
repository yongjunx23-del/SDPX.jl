# Semantic HSD Newton System

This document freezes the sign and ownership contract implemented at source SHA
`50dff5681f1e89cb7ac52f84288cb50d986a5396`. Numerical routes may eliminate
variables or regularize a factorization, but an accepted direction is a solution
of the five **unregularized** equations below.

## Contents

1. [Coordinates and residuals](#coordinates-and-residuals)
2. [The five equations](#the-five-equations)
3. [`HSDNewtonRHS` semantics](#hsdnewtonrhs-semantics)
4. [Predictor and corrector right-hand sides](#predictor-and-corrector-right-hand-sides)
5. [Scalar border and `dκ` recovery](#scalar-border-and-dκ-recovery)
6. [Route obligations](#route-obligations)
7. [Independent oracles](#independent-oracles)
8. [Known integration boundary](#known-integration-boundary)

## Coordinates and residuals

For canonical data `A ∈ ℝ^(m×n)`, `b ∈ ℝ^m`, and `c ∈ ℝ^n`, the HSD iterate is
`(x,s,y,τ,κ)`, and a Newton direction is `(dx,ds,dy,dτ,dκ)`. The current affine
residuals are

```text
rP = A*x + s - b*τ
rD = A'*y + c*τ
rG = -c'*x - b'*y + κ.
```

These definitions are the production definitions at `src/hsd/hsd.jl:438-449`,
`src/hsd/hsd.jl:452-476`, and `src/hsd/hsd.jl:479-486`. The semantic direction
field order is fixed by `NewtonDirection` at `src/kkt/system.jl:205-215`.

The cone layer freezes a self-adjoint local map `H` and a shift `h`. The map
contract is `ds + H*dy = h` (`src/kkt/system.jl:11-20`); product assembly requires
complete, non-overlapping canonical row coverage and a symmetric finite operator
(`src/kkt/system.jl:68-101`). In the product HSD implementation `H` is the
materialized `Theta` map (`src/hsd/product_cone_hsd.jl:2769-2807`).

## The five equations

The authoritative list is declared at `src/kkt/system.jl:119-139`, and the
unregularized verifier evaluates it componentwise at
`src/kkt/system.jl:235-293`.

### 1. Primal affine equation

```text
A*dx + ds - b*dτ = primal_affine = -rP.
```

The signs are evaluated at `src/kkt/system.jl:259-264`; the product-HSD direct
verifier independently evaluates `A*dx + ds - b*dτ + rP` at
`src/hsd/product_cone_hsd.jl:1563-1599` and states the equation explicitly at
`src/hsd/product_cone_hsd.jl:1759-1765`.

### 2. Dual affine equation

```text
A'*dy + c*dτ = dual_affine = -rD.
```

The signs are evaluated at `src/kkt/system.jl:266-270`; the product-HSD direct
verifier evaluates the same expression at
`src/hsd/product_cone_hsd.jl:1602-1633` and states the equation at
`src/hsd/product_cone_hsd.jl:1767-1775`.

### 3. Homogeneous gap equation

```text
-c'*dx - b'*dy + dκ = homogeneous_gap = -rG.
```

The signs are evaluated at `src/kkt/system.jl:272-279`; the product-HSD direct
verifier states and evaluates the same equation at
`src/hsd/product_cone_hsd.jl:1777-1790`.

### 4. Cone complementarity/scaling equation

```text
ds + H*dy = cone_corrector = h.
```

The cone action and residual subtraction are evaluated at
`src/kkt/system.jl:281-288`. Product-HSD recovery uses the algebraically
stable primal formula `ds = -A*dx-rP+b*dτ` and applies `G`/`Theta` to recover and
check the cone equation at `src/hsd/product_cone_hsd.jl:1273-1322`; the final
five-equation authority states `ds + Theta*dy = h` at
`src/hsd/product_cone_hsd.jl:1792-1793`.

`h` is already in the scaled semantic coordinates. A KKT route must not
reinterpret it as an unscaled Jordan-product residual.

### 5. Scalar `τκ` equation

```text
κ*dτ + τ*dκ = tau_kappa = scalar_rhs.
```

The signs are evaluated at `src/kkt/system.jl:290-293`; the product-HSD direct
verifier states and evaluates the equation at
`src/hsd/product_cone_hsd.jl:1795-1806`.

## `HSDNewtonRHS` semantics

`HSDNewtonRHS` has exactly five fields (`src/kkt/system.jl:119-139`):

| field | dimension | semantic value |
|---|---:|---|
| `primal_affine` | `m` | right side of equation 1, normally `-rP` |
| `dual_affine` | `n` | right side of equation 2, normally `-rD` |
| `homogeneous_gap` | scalar | right side of equation 3, normally `-rG` |
| `cone_corrector` | `m` | frozen scaled cone shift `h` |
| `tau_kappa` | scalar | scalar complementarity shift |

`residual_newton_rhs` is the only residual-to-RHS sign conversion: it negates
`rP`, `rD`, and `rG`, and preserves the already-formed cone and scalar shifts
(`src/kkt/system.jl:141-156`). `NewtonSystem` owns `A,b,c,H,τ,κ` plus this RHS,
checks all dimensions and finite data, and requires strict scalar interiority
`τ>0, κ>0` (`src/kkt/system.jl:158-203`). Product-HSD constructs the expanded
system through this converter at `src/hsd/product_cone_hsd.jl:2810-2819`.

## Predictor and corrector right-hand sides

Both directions use the same current affine residual RHS:

```text
predictor: (-rP, -rD, -rG, h_aff, -τκ)
corrector: (-rP, -rD, -rG, h_corr,
            σμ - τκ - dτ_aff*dκ_aff).
```

For the predictor, production forms `h_aff` with `affine_shift!` and uses
`scalar_rhs=-τκ` at `src/hsd/product_cone_hsd.jl:2892-2904` (the coupled route
uses the same composition at `src/hsd/product_cone_hsd.jl:2655-2666`).

For the combined Mehrotra corrector, production computes
`σ=min(1,(μ_aff/μ)^3)`, builds `h_corr`, and uses
`scalar_rhs=σμ-τκ-dτ_aff*dκ_aff` at
`src/hsd/product_cone_hsd.jl:2906-2918`. The symmetric-cone corrector uses the
scaled NT/Jordan algebra; its SOC target is `2σμe`, while orthant and PSD/svec
targets are `σμe` (`src/hsd/product_cone_hsd.jl:2718-2766`). Exp/Power blocks
are delegated to their higher-order corrector through the same `h` interface
(`src/hsd/product_cone_hsd.jl:2737-2746`).

Changing from predictor to corrector therefore changes only
`cone_corrector` and `tau_kappa`; it does **not** change the feasibility
homotopy represented by the first three fields.

## Scalar border and `dκ` recovery

The homogeneous-gap and scalar equations share `dκ`. In the full coupled
border, the gap row has coefficient `+1` on `dκ`, while the scalar row has
coefficients `(κ,τ)` on `(dτ,dκ)`
(`src/hsd/product_cone_hsd.jl:1006-1091`). The semantic equations, not the
regularized or condensed border, remain the acceptance authority.

P1.5 F5 made bordered recovery condition-aware. It forms two candidates:

```text
gap candidate:    dκ_G = -rG + c'*dx + b'*dy
scalar candidate: dκ_S = (scalar_rhs - κ*dτ)/τ.
```

It scores each candidate against **both** original scalar equations and keeps
the lower normalized error (`src/hsd/product_cone_hsd.jl:2177-2241`). Division
by `τ` is permitted only when
`abs(τ) > sqrt(eps(T))*max(1,abs(κ),abs(dτ),abs(scalar_rhs))`; if `τ` is
unresolved, the gap candidate must also certify the scalar equation or recovery
fails closed (`src/hsd/product_cone_hsd.jl:2221-2238`).

The expanded route currently reconstructs `dκ` from the scalar equation and
then submits the direction to the full five-equation verifier
(`src/kkt/expanded_quasidefinite.jl:505-523`). Thus its candidate policy differs,
but its accepted semantic equation does not.

## Route obligations

A route may condense or regularize the matrix, but must:

1. consume the fields of `NewtonSystem` rather than own another sign convention;
2. recover all five semantic direction variables;
3. validate against the unregularized equations with `newton_residual!`;
4. reject a direction that only solves a regularized equation.

The expanded route's exact condensation and RHS mapping are at
`src/kkt/expanded_quasidefinite.jl:302-355`; its recovery is at
`src/kkt/expanded_quasidefinite.jl:505-523`. `NewtonResidual` stores the five
unregularized residual groups (`src/kkt/system.jl:217-233`), and
`max_newton_residual` reduces those groups only (`src/kkt/system.jl:296-313`).

## Independent oracles

- `validation/newton_system_reference.jl` constructs LP, SOC, and PSD/svec fixtures
  directly, solves them through the production expanded route, recomputes every
  equation independently, checks affine residual homotopy, verifies shared
  predictor/corrector affine RHS, and repeats residual evaluation in BigFloat.
- `docs/design/newton_system_oracle.py` is a standard-library-only Python script
  that independently assembles the five-equation Jacobian and checks a direction.
  It documents the equations but is intentionally not part of Julia CI.

## Symmetric augmented-core oracle (Clarabel style)

`validation/symmetric_core_reference.jl` is a self-contained Julia-Base oracle
that proves the Clarabel-style symmetric elimination reproduces the unique
direction of the frozen five-equation system.  In rank-reduced coordinates
`x = V*xr` it forms the symmetric augmented core

```text
K = [ 0    Ar' ]
    [ Ar  -Theta ]
```

and solves, per scaling epoch, the variable and homogeneous right-hand sides

```text
K*w = [ dr; p - h ],        K*u = [ -cr; b ],
```

where `p = rhs.primal_affine`, `d = V'*rhs.dual_affine`, `h = cone_corrector`,
`cr = V'*c`, and `dr = V'*dual_affine`.  The one-dimensional homogeneous
elimination then closes

```text
dτ  = (t - τ*(g + cr'*wx + b'*wy)) / (κ + τ*(cr'*ux + b'*uy))
dxr = wx + dτ*ux,        dx = V*dxr
dy  = wy + dτ*uy
ds  = p - A*dx + b*dτ
dκ  = g + c'*dx + b'*dy.
```

The oracle compares this against a direct dense solve of the full five-equation
Jacobian (and, for the rank-reduced fixture, a direct five-equation solve in
reduced coordinates), evaluates all five residual groups independently, asserts
the symmetry of `K`, and asserts that the current `(nr+1)`-square scalar-bordered
operator is generally **non-symmetric** (`q ≠ r`).  Consequently the augmented
core is the LDL-eligible object; the old bordered and reduced-Schur operators
remain LU-only.

## Symmetric augmented-core CSC pattern contract

`src/kkt/symmetric_core.jl` owns a setup-built `SymmetricCorePattern{T}`:
the frozen lower-triangle CSC structure of

```text
K = [ 0    Ar' ]
    [ Ar  -Theta ]
```

in rank-reduced coordinates `x = V*xr`.  The `x` diagonal (columns
`1..nr`) is structurally present as numerical zeros; `Ar` occupies rows
`nr+1..nr+m` of the x columns; `-Theta` occupies the lower triangle of the
y block with a dense lower triangle per ordered cone block range.  The
pattern signature depends only on Ar `colptr`/`rowval`, the dimensions, the
ordered cone block ranges, and the declared per-block shape (`:dense_lower`
only).  Numeric refills (`refill!`) write only `nzval` through frozen
`ar_slots`, `theta_slots`, and `x_diag_slots`; they never change
`colptr`/`rowval`/`signature` and never allocate a new global matrix.
Refill rejects dimension/range mismatch, asymmetric or non-finite `Theta`,
pattern drift, and a numeric `Ar` whose CSC structure differs.  No
factorization, regularization, or solve lives in this file.

## Float64 CHOLMOD symmetric-core lifecycle

`src/factor_cache/routes/sparse_symbolic_numeric.jl` provides
`SparseSymbolicNumericCache{Float64}` — the single sparse cache for the
symmetric augmented core.  It consumes the frozen lower-triangle CSC from
`SymmetricCorePattern` plus a signed diagonal descriptor (`+1` for reduced-x
rows, `-1` for y rows) and a static regularization magnitude.

- `prepare!` owns a factor-view `SparseMatrixCSC` with exactly the frozen
  `colptr`/`rowval`, an independently retained copy of the original `K`
  values, and the fixed `symbolic_epoch` + pattern signature.  No CHOLMOD
  object exists yet.
- the first `factorize!` calls public `ldlt(Symmetric(K, :L); check=false)`
  and records `symbolic_count == 1`, `numeric_count == 1`.
- every later same-pattern `factorize!` calls public
  `ldlt!(factor, Symmetric(K, :L))` on the same CHOLMOD factor object and
  increments only `numeric_count`.
- signed static regularization is written only to the factor view diagonal
  (`K[diag] += sign*δ`); the retained original `K` values are never
  modified.  After a successful factor the view is restored to the original
  values so it mirrors the unmodified operator.
- `solve!`/`refine_once!` use the public `factor \ rhs`, which allocates a
  result object for CHOLMOD and is then copied into the caller-owned
  destination; diagnostics report `solve_allocation_policy =
  :allocating_factor_backslash_copy`.  No allocation-free claim is made.
- the cache is operationally **Float64-only**: construction, `prepare!`, and
  `factorize!` reject any other element type (Float32/BigFloat) before any
  CHOLMOD factorization.  High-precision arithmetic uses MFLA/BFLA dense
  symmetric LDL instead; this sparse CHOLMOD cache never narrows a
  higher-precision matrix.
- `SparseSymbolicRequirements` enforces square/dimension, `dsigns`
  length/values, and finite nonnegative regularization in its inner
  constructor, and `prepare!` re-validates defensively.  The cache signature
  includes the D-sign vector and the regularization magnitude, so opposite
  signs or a different shift never share an identity.
- pattern drift, non-finite data, a `dsigns` mismatch, a failed refactor,
  and any stale factor revoke the usable state; `solve!` from any state
  other than `Fresh` is rejected, and `Invalid` requires a re-`prepare!`
  before any factorization.  A failed numeric factor detaches the CHOLMOD
  object (`factor === nothing`) and restores the factor view to the original
  values, so `Failed` can never solve stale data; recovery re-runs the sole
  symbolic analysis on the next same-pattern attempt and is not claimed as
  symbolic reuse across a failure.  CHOLMOD `Factor` `success` is a provider
  fact, not a mathematical certificate; the direction is accepted only by
  the unregularized five-equation residual and, at termination, the
  original-coordinate certificate.

## Symmetric augmented-core direction recovery with original-core refinement

`SymmetricCoreWorkspace` (`src/kkt/symmetric_core.jl`) is a test-only,
type-stable direction-recovery layer over `K = [0 Ar'; Ar -Theta]`.  It
consumes a `SymmetricCorePattern`, a concrete `AbstractFactorCache` (the
Float64 CHOLMOD lifecycle), an orthonormal `V`, and the frozen
`NewtonSystem`.  Per factor epoch:

```text
factor Kε once                  (nonzero signed static shift δ)
solve homogeneous core once:     Kε*u = [ -V'c ; b ]
for each sequential variable RHS (predictor, then dependent corrector):
    Kε*w = [ V'*dual_affine ; primal_affine - cone_corrector ]
    refine w and u against the retained original K (same Kε factor)
    dτ = (tau_kappa - τ*(g + cr'wx + b'wy))
           / (κ + τ*(cr'ux + b'uy))
    dxr = wx + dτ*ux ; dx = V*dxr
    dy = wy + dτ*uy
    ds = p - A*dx + b*dτ
    dκ = g + c'*dx + b'*dy
    frozen five-equation residual gate
```

CHOLMOD's nonpivoting LDL cannot factor the structurally-zero primal
diagonal block of `K`, so the caller factors the signed static-shifted
`Kε = K + δ*diag(+1_x, -1_y)` with a nonzero δ.  Every core solve is
refined against the **retained original `K`** (never the regularized view)
with the same `Kε` factor: at most two correction solves per RHS, each
accepted only on strict normalized original-core residual contraction
(`η_current < η_previous`); a non-contracting or non-finite correction
fails closed.  The recovered HSD direction is accepted only through the
frozen `newton_residual!`; the denominator must be finite, nonzero, and
above `sqrt(eps(T))` times a type scale, and the cache factor epoch must
match the workspace epoch.  One factor, one homogeneous solve, and two
sequential variable solves (predictor then corrector) — no refactor in
between.

## Known integration boundary

`src/kkt/system.jl:1-6` declares the semantic layer as the sign authority, but
at this SHA only the `:expanded` product-HSD route constructs a `NewtonSystem`
(`src/hsd/product_cone_hsd.jl:2810-2819`). The default bordered route still
assembles/recomputes equivalent signs directly. This is an implementation
migration gap, not a second approved mathematical convention; later HSD-loop
migration must make every route consume this semantic layer.

## C5: MFLA/BFLA dense symmetric-core factor seam

`SymmetricCoreWorkspace` is arithmetic-agnostic.  For MultiFloat and BigFloat
the dense pivoted-LDL factor of the same symmetric augmented core is built by
`build_symmetric_core_ldlt_cache`:

- a conservative dense byte estimate (`symmetric_core_dense_bytes`) and the
  existing `conservative_memory_upper_bound_eligibility` gate run before any
  dense allocation or factorization;
- the MFLA extension returns an `MFLDLTFactorCache` (fixed-width MultiFloat);
- the BFLA extension returns a `BFLALDLTFactorCache` at an explicit
  `precision_bits` equal to the ambient BigFloat precision;
- a missing provider or a failed memory gate fails closed before allocation.

The exact operator, homogeneous solve, sequential predictor/corrector solves,
original-core refinement, and frozen five-equation residual remain unchanged;
provider LDLT factors only the same `K`.  No new backend or dependency is
introduced.

## C6a semantic/provider shadow parity

`src/kkt/symmetric_core.jl` adds a cold/test-only bridge for the same
symmetric augmented core:

- `_validate_core_preconditions(system, V)` proves V dimensions/isometry,
  every row of `A` in `range(V)`, and `c in range(V)` **before** any Theta,
  `A*V`, pattern, or dense allocation;
- `symmetric_core_provider_available(T, precision_bits)` is the provider
  availability/precision gate overridden by the MFLA/BFLA extensions; the
  base implementation fails closed so an absent provider is rejected before
  any dense allocation;
- `symmetric_core_block_ranges(cone)` extracts the ordered block ranges from
  a `ProductConeLinearization` or `BlockProductConeLinearization` and fails
  closed on any other linearization;
- `symmetric_core_theta(cone)` materializes the exact block-diagonal `Theta`
  operator with owned per-block copies (no BigFloat aliasing);
- `symmetric_core_pattern_from_system(system, V)` validates the documented
  preconditions, forms `Ar = A*V`, freezes the CSC pattern, and refills it
  with the exact `Theta`;
- `build_symmetric_core_workspace(system, V, ...)` runs the precondition,
  provider-availability, and dense-memory eligibility gates before any
  materialization, then prepares and factors the provider cache (Float64
  CHOLMOD with the signed static shift, or the dense MFLA/BFLA LDL),
  synchronizes the workspace, and solves the homogeneous core once.

`validation/newton_system_reference.jl` compares Float64 LP/SOC/PSD symmetric
core predictor and changed corrector directions against the expanded exact
route and asserts that invalid rank bases, out-of-range `A`/`c`, and
unsupported arithmetic fail closed before pattern/provider work.
`validation/providers/provider_smoke.jl` drives the full workspace through the
real MFLA (`Float64x2`/`Float64x4`) and BFLA (`BigFloat256`) providers, and
compares **both** predictor and changed-corrector directions field-by-field
against an arithmetic-generic direct five-equation solve in the provider's own
scalar type (no Float64 downcast).  It also exercises a real
`BlockProductConeLinearization` multi-block case with exact block-diagonal
Theta materialization and all five residual groups.  This is shadow validation
only: no production route dispatches the core yet.
