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

- `test/newton_system_reference.jl` constructs LP, SOC, and PSD/svec fixtures
  directly, solves them through the production expanded route, recomputes every
  equation independently, checks affine residual homotopy, verifies shared
  predictor/corrector affine RHS, and repeats residual evaluation in BigFloat.
- `docs/design/newton_system_oracle.py` is a standard-library-only Python script
  that independently assembles the five-equation Jacobian and checks a direction.
  It documents the equations but is intentionally not part of Julia CI.

## Known integration boundary

`src/kkt/system.jl:1-6` declares the semantic layer as the sign authority, but
at this SHA only the `:expanded` product-HSD route constructs a `NewtonSystem`
(`src/hsd/product_cone_hsd.jl:2810-2819`). The default bordered route still
assembles/recomputes equivalent signs directly. This is an implementation
migration gap, not a second approved mathematical convention; later HSD-loop
migration must make every route consume this semantic layer.
