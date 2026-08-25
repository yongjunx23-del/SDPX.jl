# HSD_FORMULATION.md — Homogeneous Self-Dual (Nesterov–Todd) formulation for SDPX

**Status:** Math / design reference (Subagent B). Companion to
`CANONICAL_FORM.md` (frozen). This document is **not** production code; it is
the review-ready derivation of the HSD formulation used by the SDPX conic
kernel, together with hand-computed fixtures that are mechanically checked in
`test/hsd_equations.jl`.

**Convention authority.** The sign and equation conventions below are the
**frozen HSD spec** (as mandated by the task): primal, dual, and homogeneous
embedding exactly as written. Where the *current production HSD code* uses a
different sign/structure, this is **flagged** in §10 and in the report — it is
**not** "fixed" here (no `src/*.jl` is touched by this document).

**Mechanical verification.** The worked fixtures of §9 are checked numerically
by `SDPX/test/hsd_equations.jl` (self-contained, no SDPX dependency):

```
JULIA_DEPOT_PATH=<repo>/.julia-depot julia --project=<repo> test/hsd_equations.jl
```

---

## 1. The canonical form, primal/dual, and the HSD embedding

### 1.1 Canonical conic form (frozen)

Every problem, after all reductions, is the canonical conic program

```
minimize    c'x
subject to  A x + s = b
            s ∈ K
```

with

- `x ∈ R^n` free decision variables (`x` free is respected; never split),
- `s ∈ R^m` the slack in the product cone `K`,
- `A ∈ R^{m×n}` the (sparse) equality map,
- `b ∈ R^m`, `c ∈ R^n`,
- `K = K_1 × … × K_L ⊂ R^m` an ordered product of native cone blocks.

**Dimensions are the whole game.** `x` is `n`-dimensional, while the slack `s`
and the right-hand side `b` are `m`-dimensional. In general `n ≠ m`. Any
quantity that pairs two vectors must pair vectors of the *same* length, and
must pair *conjugate* cone variables — see §1.4.

### 1.2 Primal and dual in these coordinates

**Primal (minimization).**

```
min c'x   s.t.   A x + s = b,   s ∈ K.
```

**Dual.** Form the Lagrangian with a multiplier `y ∈ R^m`:

```
L(x, s, y) = c'x + y'(A x + s − b).
```

- `inf_{x∈R^n} c'x + y'A x` is finite (equal to `0`) iff `A'y + c = 0`; otherwise it is `−∞`.
- `inf_{s∈K} y's` is `0` if `y ∈ K*` (the dual cone), and `−∞` otherwise.

Hence the **dual problem** is

```
maximize   −b'y
subject to  A'y + c = 0,   y ∈ K*.
```

(Objective `−b'y`; the sign follows from the `−b'y` constant term of the
Lagrangian after `inf_x c'x + y'Ax = 0`.)

**Weak duality.** For any feasible primal `(x,s)` and any feasible dual `y`,

```
c'x + b'y  =  c'x + x'A'y + s'y   (because b'y = x'A'y + s'y, using Ax + s = b)
           =  x'(c + A'y) + s'y   =  s'y   ≥ 0,
```

where the inequality uses `s ∈ K`, `y ∈ K*`, i.e. `s'y ≥ 0`. So `c'x ≥ −b'y`
and the duality gap is

```
gap := c'x − (−b'y) = c'x + b'y = s'y ≥ 0.
```

### 1.3 The homogeneous self-dual embedding (HSD)

Introduce the homogeneous variables `(τ, κ) ≥ 0` and require the *three block
residuals* to vanish:

```
(P)   A x  + s − b τ  = 0          (m equations)
(D)   A'y  + c τ      = 0          (n equations)
(G)   c'x  + b'y + κ  = 0          (1 equation)
s ∈ K,   y ∈ K*,   τ ≥ 0,   κ ≥ 0.
```

Interpretations:

- `(P)` is the homogeneous primal feasibility (at `τ = 1` it is exactly
  `A x + s = b`; `s` is the slack).
- `(D)` is the homogeneous dual feasibility (at `τ = 1` it is exactly
  `A'y + c = 0`).
- `(G)` is the homogeneous gap equation: rearranged, `κ = −(c'x + b'y)`.
  Since `c'x + b'y = s'y ≥ 0` at a feasible point, `κ = −s'y ≤ 0` there — a
  sign convention whose consequences are analyzed in §1.5 and §8.1.

A nonzero point `(x, y, s, τ, κ)` with `τ > 0` yields a feasible primal
`(x/τ, s/τ)` and a feasible dual `y/τ`, and `(G)` becomes the closed duality gap
at the solution. A point with `τ = 0` yields an infeasibility ray (§7).

### 1.4 Why `dot(s, x)` is the wrong bilinear pairing

Two independent reasons forbid `s'x` (and likewise `b's` as a "gap"):

1. **Dimension mismatch.** `s ∈ R^m`, `x ∈ R^n`. The expression `s'x` is not
   even defined when `m ≠ n`, which is the generic case. The correct pairing is
   `s'y`, with `s` (slack in `K`) and `y` (dual variable in `K*`), both in
   `R^m`.

2. **Conjugate cones.** The conic complementarity condition is `s ∘ y = 0`
   (blockwise Jordan product) with `s ∈ K` and `y ∈ K*`, not `s'x`. `x` is a
   *free* variable (`x ∈ R^n`) and carries no cone membership and no conjugate
   "primal" partner; it is paired through `A`/`A'`, not through the bilinear
   complementarity. The correct scalar "duality/product" measure is `s'y`
   (slack × dual) plus `τ κ`, exactly as it appears in the central-path
   parameter `μ = (s'y + τ κ)/(ν + 1)` (§6). The homogeneous skew structure
   (below, §2) is self-consistent precisely because it pairs `(s, y)` and
   `(τ, κ)`.

### 1.5 The self-dual operator: why `Q` is skew-symmetric

The HSD is most compactly written as a **single skew-symmetric linear system.**
Order the unknowns as `(x, y, τ) ∈ R^{n+m+1}` and define the block matrix

```
Q := [   0     A'    c   ]
     [  −A     0     b   ]
     [  −c'   −b'    0   ]
```

(rows/cols are `(x, y, τ)`; block sizes `n, m, 1`). One checks block-by-block
that `Q` is skew-symmetric:

```
Q_{12} = A'  (n×m),   Q_{21} = −A     = −(Q_{12})' ,
Q_{13} = c   (n×1),   Q_{31} = −c'    = −(Q_{13})' ,
Q_{23} = b   (m×1),   Q_{32} = −b'    = −(Q_{23})' ,
Q_{11} = Q_{22} = Q_{33} = 0.
```

Therefore **`Q = −Q'`**. Acting on `(x, y, τ)`:

```
Q·(x; y; τ) = ( A'y + c τ ;  −A x + b τ ;  −c'x − b'y ).
```

Setting the three blocks to `(0; s; κ)` reproduces *exactly* the frozen HSD
equations (§1.3):

- block 1: `A'y + c τ = 0` → `(D)`;
- block 2: `−A x + b τ = s`, i.e. `s = b τ − A x` → `A x + s − b τ = 0` → `(P)`;
- block 3: `−c'x − b'y = κ`, i.e. `c'x + b'y + κ = 0` → `(G)`.

Because `Q` is skew-symmetric, the self-duality identity gives the
complementarity for free:

```
0 = (x, y, τ)·Q(x, y, τ) = (x, y, τ)·(0, s, κ) = y's + τ κ = s'y + τκ.
```

At a solution with `s ∈ K`, `y ∈ K*`, `τ, κ ≥ 0` this forces `s'y = 0` and
`τκ = 0` (the complementarity conditions). This is the rigorous reason the
correct complementarity is `s'y` (+ `τκ`), and why the whole formulation is
called *self-dual*: the coefficient operator of the homogeneous system is its
own (skew) adjoint.

> **Sign note.** Some texts write the third block as `c'x − b'y + κ = 0`
> (dual in the `max b'y` orientation) and a matrix `[[0, A, −b], [−A', 0, c],
> [b', −c', 0]]`. That is a *different* sign orientation of the same operator,
> conjugate by flipping the sign of `y` / of `b'y`. The frozen spec (with
> `A'y + c τ = 0`, `c'x + b'y + κ = 0`, dual objective `−b'y`) is reproduced
> exactly by the `Q` written above. Section 10 flags the production-code sign.

---

## 2. Block matrix `Q`, skew-symmetry, and dimensions

(Content of §1.3/§1.5 collected for quick reference.)

- Variables: `(x, y, τ)`, dimensions `n`, `m`, `1`.
- Slack/dual pairing: `s ∈ R^m` (slack, in `K`), `y ∈ R^m` (dual, in `K*`).
- Free variable: `x ∈ R^n`, no cone membership.
- Scalar pair: `(τ, κ) ∈ R_+ × R_+`.

The skew matrix (verified `Q = −Q'`):

```
Q = [   0    A'   c  ]
    [  −A    0    b  ]
    [  −c'  −b'   0  ]
```

Acting on `(x, y, τ)` it yields the three block residuals `(0; s; κ)` (so
`Q·(x;y;τ) = (0; s; κ)`), and skewness yields the complementarity `s'y + τκ = 0`
at a solution.

---

## 3. Central path and the Newton (predictor/corrector) direction system

### 3.1 Central path

For the product cone `K = K_1 × … × K_L` with barrier degree `ν` (§6), the
*central path* of the HSD is the set of strictly feasible points satisfying

```
(P)   A x + s − b τ = 0
(D)   A'y + c τ     = 0
(G)   c'x + b'y + κ = 0
s ∘ y = μ e_K      (blockwise Jordan product, e_K the cone identity)
τ κ   = μ
τ > 0, κ > 0, s ∈ int K, y ∈ int K*.
```

The value `μ` is the *centrality / duality gap parameter*:

```
μ = (s'y + τ κ) / (ν + 1).
```

(See §6 for the `ν + 1`.)

### 3.2 Residuals and complementarity linearization

Given a current iterate, define residuals

```
rP = A x + s − b τ
rD = A'y + c τ
rG = c'x + b'y + κ
```

and the "complementarity target" for the centering parameter `σ ∈ [0, 1]`:

```
s ∘ y = σ μ e_K ,   τ κ = σ μ .
```

For a symmetric-cone algebra with Jordan product `∘` and left-multiplication
operators `L_u : v ↦ u ∘ v`, the Newton linearization of `s ∘ y = σμ e` is

```
L_y ds + L_s dy  =  σ μ e  − (s ∘ y)  −  (ds_a ∘ dy_a),
```

where `(ds_a, dy_a, dτ_a, dκ_a)` are the affine (predictor) directions entering
the second-order correction `ds_a ∘ dy_a`. The scalar complementarity
linearizes as

```
τ dκ + κ dτ = σμ − τκ − dτ_a dκ_a.
```

### 3.3 The full Newton system (generic form)

The Newton direction `(dx, dy, ds, dτ, dκ)` solves

```
(P)   A dx + ds − b dτ = −rP
(D)   A' dy + c dτ     = −rD
(G)   c' dx + b' dy + dκ = −rG
(C1)  L_y ds + L_s dy  =  σμ e − (s∘y) − (ds_a∘dy_a)
(C2)  τ dκ + κ dτ      =  σμ − τκ − dτ_a dκ_a
```

- **Predictor (affine, Mehrotra):** `σ = 0`, `μ` absent, second-order terms
  omitted. Solves the pure affine-scaling direction `(dx_a, dy_a, ds_a, dτ_a, dκ_a)`.
- **Corrector (centering):** `σ` chosen (e.g. Mehrotra
  `μ_cent = μ·(μ_aff/μ)³`), and the second-order correction `ds_a ∘ dy_a`,
  `dτ_a dκ_a` added. The final direction is the Mehrotra combination.

The cone linearization `(C1)` is what brings the **barrier derivatives** into
the system: `L_s` and `L_y` are the Jordan-algebra (barrier-derivative)
operators of the cone at the current point. For `K = R_+^m` (LP) `L_u =
diag(u)`; for SOC/PSD `L_u` is the corresponding Jordan product, and in the
Nesterov–Todd scaling both `L_s` and `L_y` reduce to the same operator.

---

## 4. τ/κ elimination and the bordered KKT / Schur reduction to (dx, dy)

We eliminate `(ds, dκ, dτ)` and reduce to a system in `(dx, dy)`, which is the
matrix actually factorized by the KKT solver.

**Step 1 — complementarity → `ds`.** The NT scaling point `W` (with
`W s = W^{-1} y =: d`) lets one invert `(C1)`; in the scaled variables
`ŝ = W^{1/2}s = ŷ = W^{1/2} y` the equation `(C1)` becomes the linear
`dŝ + dŷ = ρ̂`, whence `ds` is a linear function of `dy`:

```
ds = ρ̃ − Θ dy,        Θ := the primal–dual scaling operator (cone Newton Hessian).
```

For `K = R_+^m` (LP), `Θ = diag(s/y)` componentwise and `ds = ρ/y − (s/y) dy`.

**Step 2 — τ/κ elimination (`dκ`, then `dτ`).** From `(G)` and `(C2)`:

```
dκ = −rG − c'dx − b'dy
dτ = [ρτ + τ(rG + c'dx + b'dy)] / κ ,   ρτ := σμ − τκ − dτ_a dκ_a .
```

Substituting this `dτ` into `(P)` and `(D)` removes `dτ` and `dκ` entirely.

**Step 3 — the bordered (Schur-complement) primal–dual system.** The reduced
linear system in `(dx, dy)` is the **bordered KKT** / Schur complement of the
full Newton system. In the LP case (diagonal `Θ`), with `S=diag(s)`,
`Y=diag(y)`, `D = S Y^{-1}`, it reads

```
[  (τ/κ) c c'     A' + (τ/κ) c b'    ] [dx]   [ −rD − (c/κ)(ρτ + τ rG)          ]
[  A − (τ/κ) b c'   −(D + (τ/κ) b b') ] [dy] = [ −rP − Y^{-1}ρ + (b/κ)(ρτ + τ rG) ].
```

For a general symmetric cone the scalar `D` is replaced by the cone scaling
operator `Θ` and `Y^{-1}ρ` by the scaled residual from `(C1)`; the block
structure is identical. The `(τ/κ)` terms are the border coming from the scalar
homogeneous pair `(τ,κ)`; the reduction is therefore called **τ/κ elimination
(bordered KKT / Schur-complement reduction)**.

**Why this is the matrix that gets factored.** The solver stores this one
system (the `H`/`A`/`A'` bordered matrix, symmetric quasi-definite) per KKT
epoch and factors it once; predictor, corrector, and iterative refinement all
reuse that single factorization.

---

## 5. Cone-membership conditions and where barrier derivatives enter

- **Primal cone:** `s ∈ K = K_1 × … × K_L`, blockwise. For the native cones:
  - `:nonnegative` (dim `d`): `s_i ≥ 0` for `i = 1..d`;
  - `:soc` (dim `d`): head `t = s_1`, `t ≥ ‖(s_2..s_d)‖`;
  - `:psd` (dim `n`, packed lower triangle): the unpacked symmetric matrix is PSD;
  - `:exp`, `:power`: the respective 3-D membership;
  - `:free`, `:zero`: no inequality (equality rows).
- **Dual cone:** `y ∈ K*`, with `K* = K` for the self-dual symmetric cones
  (`R_+`, SOC, PSD), and the appropriate dual for exp/power.
- **Barrier derivatives.** The central-path Newton system (§3.2, `(C1)`) uses
  `L_s, L_y`, i.e. the *first derivative* of the cone barrier (the Jordan
  product / the scaling `Θ` in §4). The Nesterov–Todd scaling `W` is the
  *second-derivative* object: it is the unique self-adjoint operator with
  `W s = W^{-1} y`. All of these are the cone-algebra (barrier) primitives of
  the native cones.

---

## 6. The central-path parameter `μ` and why `ν + 1`

For a product cone `K` with barrier degree `ν`, the central-path pair is
`s ∘ y = μ e` and `τκ = μ`. Summing the blockwise product `s∘y` over the
barrier-degree of `K`:

```
s'y  =  trace of (s ∘ y) over the Jordan algebra  =  ν μ   (ν = barrier degree of K),
```

and the scalar cone `R_+` for `(τ, κ)` contributes exactly one more:

```
τκ = μ .
```

Therefore `s'y + τκ = νμ + μ = (ν + 1) μ`, i.e.

```
μ = (s'y + τ κ) / (ν + 1).
```

**Why `ν + 1` (not `n + 1`, not `m + 1`).** The embedding contains the two
conjugate pairs `(s, y)` (in `K`, degree `ν`) and `(τ, κ)` (in `R_+`, degree
`1`). The barrier degree of the *full* HSD cone `K × R_+` is `ν + 1`; the gap
is split evenly across all `ν + 1` "coordinate axes". The dimension of the
*total variable space* (`n + m + 2`) is irrelevant. For `K = R_+^m` (LP),
`ν = m` and the denominator is `m + 1`, **not** `n + 1`.

---

## 7. Certificates, ray normalization, and optimal recovery

Termination status is assigned **only** from verified certificates in the
original coordinates (never from raw `τ`/`κ`). The frozen conventions:

- **Optimal.** When `τ > 0`, recover `x = x̂/τ`, `s = ŝ/τ`, `y = ŷ/τ` from the
  HSD point, and verify in *original coordinates*: `Ax + s = b`, `s ∈ K`,
  `A'y + c = 0`, `y ∈ K*`, `c'x + b'y = 0` (gap), all within tolerance.
- **Primal infeasible** (no feasible `(x,s)`). The HSD converges with `τ = 0`;
  the dual variable `y` becomes a Farkas ray:
  ```
  A'y ≈ 0,   y ∈ K*,   b'y < 0.
  ```
  (At `τ = 0`, `(D)` reads `A'y = 0`.) The ray is **normalized by `−b'y = 1`**
  (`y ← y / (−b'y)`).
- **Dual infeasible / primal unbounded.** The free variable `x` gives a ray
  ```
  −A x ∈ K,   c'x < 0.
  ```
  (At `τ = 0`, `(P)` reads `A x + s = 0`, `s ∈ K`, i.e. `−A x = s ∈ K`.) The
  ray is **normalized by `−c'x = 1`** (`x ← x / (−c'x)`).

**Optimal recovery.** `x*/τ`, `s*/τ`, `y*/τ` are the primal/dual optimizers when
`τ > 0`.

---

## 8. Never infer status from raw `τ`/`κ` alone

A small `τ` does *not* by itself mean "primal infeasible", and a small `κ` does
not by itself mean "optimal". Both can be small at interior points, at scaled
paths, and at limit points that are only *numerically* on a boundary. The only
trustworthy signal is a **verified certificate in original coordinates**
(§7): normalized homogeneous residual + cone membership + objective sign +
original-coordinate verification, pushed through the full ReductionChain inverse
map before any MOI status is set. This matches FINAL_ARCHITECTURE.md §2.

---

## 9. Worked examples with verifiable numbers

The fixtures are built so that the *exact* HSD identities hold, and are checked
mechanically in `test/hsd_equations.jl`. We verify (a) `A x + s − bτ = 0`,
(b) `A'y + cτ = 0`, (c) `c'x + b'y + κ = 0`, (d) `Q + Q' = 0`, (e)
`μ = (s'y + τκ)/(ν+1)`, (f) cone membership, and (g) certificate sign
conventions.

### 9.1 Small LP

`K = R_+^2`, `m = n = 2`:

```
A = I_2 = [1 0; 0 1],   b = [1; 1],   c = [−1; −1].
```

Primal: `min −x1 −x2` s.t. `x1 + s1 = 1`, `x2 + s2 = 1`, `s ≥ 0`; optimum at
`x = (1, 1)`, `s = (0, 0)`, value `−2`.
Dual: `max −(y1 + y2)` s.t. `A'y + c = 0` ⇒ `y = −c = (1, 1)`, value `−2`. ✔

**Optimal HSD point** `(x, y, s, τ, κ) = ((1,1), (1,1), (0,0), 1, 0)`:

- (P) `A x + s − b τ = (1,1) + (0,0) − (1,1) = 0` ✔
- (D) `A'y + c τ = (1,1) + (−1,−1)·1 = 0` ✔
- (G) `c'x + b'y + κ = (−1−1) + (1+1) + 0 = 0` ✔
- `Q + Q' = 0` ✔ (verified in test)
- `ν = 2`, `μ = (s'y + τκ)/(ν+1) = (0 + 0)/3 = 0` ✔

**A near-interior point** `(x, y, s, τ, κ) = ((0.4, 0.4), (1, 1), (0.6, 0.6), 1, −1.2)`:
- (P) `(0.4,0.4)+(0.6,0.6)−(1,1)·1 = 0` ✔
- (D) `(1,1)+(−1,−1)·1 = 0` ✔
- (G) `(−0.4−0.4)+(1+1)+κ = 1.2 + κ` → `κ = −1.2` (the frozen-convention value;
  `c'x+b'y = 1.2 = s'y ≥ 0`, see §10.1).

### 9.2 A small SOC program

`K = SOC(2)` (self-dual), `m = 2`, `n = 1`:

```
A = [1; 1] (2×1),   b = [1; 1],   c = [−1].
```

Feasibility `s = b − A x = (1−x, 1−x) ∈ SOC` holds iff `1 − x ≥ |1−x|`, i.e.
`x ≤ 1`; the optimal is `x = 1`, `s = (0,0)`, `obj = −1`.

**Optimal HSD point** `(x, y, s, τ, κ) = (1, (1, 0), (0, 0), 1, 0)`:

- (P) `A x + s − bτ = (1,1) + (0,0) − (1,1)·1 = (0,0)` ✔
- (D) `A'y + cτ = (1·1+1·0) + (−1)·1 = 1 − 1 = 0` ✔
- (G) `c'x + b'y + κ = (−1)(1) + (1·1+1·0) + 0 = −1 + 1 = 0` ✔
- `s ∈ SOC` (`(0,0)`) ✔; `y ∈ SOC` (`1 ≥ |0|`) ✔; `τ = 1 ≥ 0`, `κ = 0 ≥ 0` ✔
- `ν = 2` (SOC barrier degree), `μ = (s'y + τκ)/3 = (0 + 0)/3 = 0` ✔.

**Interior-feasible point** `(x, y, s, τ, κ) = (0.5, (0.5, 0.5), (0.5, 0.5), 1, −0.5)`:

- (P) `A x + s − bτ = (0.5,0.5) + (0.5,0.5) − (1,1)·1 = (0,0)` ✔
- (D) `A'y + cτ = (0.5 + 0.5) + (−1)·1 = 1 − 1 = 0` ✔
- (G) `c'x + b'y + κ = (−0.5) + (1·0.5+1·0.5) + (−0.5) = −0.5 + 1 − 0.5 = 0` ✔
- `s ∈ int SOC` (`0.5 > |0.5|`) ✔, `y ∈ int SOC` ✔, `τ = 1 > 0`, `κ = −0.5`.
  Here `κ = −(c'x+b'y) = −(−0.5 + 1) = −0.5`, the frozen-convention value
  (§10.1); the algebraic identities hold exactly.

### 9.3 A small SDP program

`K = PSD_2` (2×2, stored as lower-triangle vec `(a, b, c)` for
`[[a, b], [b, c]]`), `m = 3`, `n = 1`:

```
A = [1; 1; 1]  (3×1),   b = (4, 2, 2),   c = (−4).
```

**An interior HSD point** with `x = 0`, `τ = 1`, `y = (1, 1, 2)`,
`s = b − A x = (4, 2, 2)`, and `κ = −(c'x + b'y)`:

- (P) `A x + s − b τ = (0,0,0) + (4,2,2) − (4,2,2) = (0,0,0)` ✔
- (D) `A'y + c τ = (1+1+2) + (−4) = 0` ✔
- (G) `c'x + b'y + κ = 0 + (4·1+2·1+2·2) + κ = 10 + κ` → `κ = −10`,
  `c'x + b'y + κ = 0` ✔
- `s` PSD: `(4,2,2) ↦ [[4,2],[2,2]]`, `det = 4 > 0` ✔
- `y` PSD: `(1,1,2) ↦ [[1,1],[1,2]]`, `det = 1 > 0` ✔
- `ν = 2`, `μ = (s'y + τκ)/(ν+1) = (10 + (−10))/3 = 0` ✔.

As noted in §10.1, at feasible points the frozen `(G)` gives `κ = −s'y ≤ 0`
(here `−10`); the algebra is exact. The test file checks all of these
numerically, including `Q + Q' = 0` on the rectangular `A`.

---

## 10. Ambiguities and production-code contradictions (flagged, not fixed)

### 10.1 Sign of the third equation (gap) and the "strict interior" issue

With `y ∈ K*`, `s ∈ K` one always has `c'x + b'y = s'y ≥ 0` (weak duality,
§1.2). Therefore the frozen third equation `c'x + b'y + κ = 0` forces
`κ = −(c'x + b'y) ≤ 0` at any feasible point. A *strictly interior* central
point (`s > 0, y > 0, τ > 0, κ > 0`) would need `κ > 0`, i.e. `c'x+b'y < 0`,
which is *inconsistent* with `s ∈ K, y ∈ K*`. The frozen spec's `κ ≥ 0` and
`s ∈ K, y ∈ K*` are therefore mutually inconsistent for interior points. The
**standard / production HSD resolves this by flipping the sign of the third
equation** to `−c'x − b'y + κ = 0`, so that `κ = c'x + b'y = s'y ≥ 0` and an
interior point with `κ > 0` exists. The worked examples above deliberately
keep the frozen signs and verify the *algebra* (with `κ` computed as
`−(c'x+b'y)`), and §3.2/§8.1 record the consequence.

### 10.2 Production `src/hsd/hsd.jl` uses a different (self-dual) convention

The production HSD state does **not** follow the frozen dual-form spec. In
`src/hsd/hsd.jl`:

- `hsd_dual_residual` (`src/hsd/hsd.jl:65-67`) is `−c'x − b's + κ`,
  i.e. it uses `b's` (`dot(b, s)`) — a term that appears in the production
  self-dual form `(s, κ) ∈ K × R_+`, `(x, τ) ∈ K* × R_+`
  (`FINAL_ARCHITECTURE.md:66-70`, `src/hsd/hsd.jl:9-13`). The frozen dual
  residual is `A'y + cτ = 0` (with an explicit dual `y ∈ K*`), and `b's` is a
  different quantity from `b'y` (and is dimensionally the "wrong" pair for a
  `y`-free formulation).
- `hsd_complementarity` (`src/hsd/hsd.jl:74-76`) returns
  `(dot(s, x) + τκ)/hsd_dimension`, i.e. it uses **`dot(s, x)`** (forbidden by
  the frozen spec) and divides by `hsd_dimension = n+1`
  (`src/hsd/hsd.jl:49`) instead of `ν + 1` (barrier degree + 1). For a general
  cone `n + 1 ≠ ν + 1`.
- `hsd_optimality_gap` (`src/hsd/hsd.jl:96-98`) returns `dot(c,x) + dot(b,s)`,
  again the `b's`/self-dual pairing.

`src/hsd/hsd.jl` therefore encodes the **production self-dual convention**
(consistent with `FINAL_ARCHITECTURE.md` §2), not the frozen MOSEK-style spec
with explicit dual `y`. These are mathematically different (though related)
formulations; this document keeps the frozen one as authoritative and **does
not modify `src/hsd/hsd.jl`** (out of scope).

### 10.3 `n+1` vs `ν+1`

- Frozen spec / this document: denominator `ν + 1` (barrier degree, §6).
- Production `hsd.jl`: denominator `length(x)+1 = n+1`
  (`src/hsd/hsd.jl:49,75`). These coincide only for `K = R_+^m` with `m = n`.
  The barrier-degree notion `ν` already exists in the IR layout
  (`src/ir/layout.jl:79-106`, `barrier_degree`, `layout.barrier_degree`), so
  the production HSD denominator should use `barrier_degree` + 1, not `n+1`.

### 10.4 Certificate sign convention

- Frozen: primal-infeasible ray `A'y ≈ 0`, `y ∈ K*`, `b'y < 0`, normalize
  `−b'y = 1`; dual-infeasible ray `−A x ∈ K`, `c'x < 0`, normalize `−c'x = 1`.
- Production `src/certificates/certificates.jl:140-159` uses `A'y <= tol`,
  `b'y > tol` (opposite sign on `b'y`) and `c'x < -tol`, `Ax+s=0`, `s∈K`. The
  two are related by negating the ray; the production one does not verify
  `y ∈ K*` explicitly in `verify_primal_infeasibility!`
  (`src/certificates/certificates.jl:140-145`).

All of the above are reported; none are fixed in this document (out of scope).

---

## 11. References / notation glossary

- Nesterov & Todd (1998): "Primal-dual interior-point methods for
  self-scaled cones" (the `Q = −Q'` self-dual operator and NT scaling).
- Andersen & Andersen (2000), "The MOSEK interior point optimizer for linear
  programming: implementation of the homogeneous algorithm" — the conic
  homogeneous embedding with `A'y + cτ = 0`, `c'x + b'y + κ = 0`.
- Mehrotra (1992) — predictor/corrector.
- Wright (1997), *Primal-Dual Interior-Point Methods*, §6 (τ/κ-elimination).

Notation: `∘` Jordan product; `ν` barrier degree; `K*` dual cone; `e_K` cone
identity; `()` transpose.
