# HSD_FORMULATION.md — Homogeneous Self-Dual (Nesterov–Todd) formulation for SDPX

**Status:** Math / design reference (Subagent B). Companion to
`CANONICAL_FORM.md` (frozen). This document is **not** production code; it is
the review-ready derivation of the HSD formulation used by the SDPX conic
kernel, together with hand-computed fixtures that the independent tests must
check in `test/hsd_equations.jl`.

**Convention authority.** The sign and equation conventions below follow the
**frozen HSD spec** of `docs/design/CANONICAL_FORM.md` §2 (the corrected gap
sign `−c'x − b'y + κ = 0` with `κ = c'x+b'y` and, at exact
feasibility, `s'y=τκ`).  This is the
authoritative implementation target for `src/hsd/hsd.jl` and
`src/hsd/nonnegative_hsd.jl`, and the target checked by `test/hsd_equations.jl`.
The historical reasoning behind the sign correction (and why the *earlier*
`c'x + b'y + κ = 0` convention was rejected) lives in
`docs/reviews/HSD_SIGN_HISTORY.md` and is **not** an open question here.

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
(G)   −c'x − b'y + κ  = 0          (1 equation)
s ∈ K,   y ∈ K*,   τ ≥ 0,   κ ≥ 0.
```

Interpretations:

- `(P)` is the homogeneous primal feasibility (at `τ = 1` it is exactly
  `A x + s = b`; `s` is the slack).
- `(D)` is the homogeneous dual feasibility (at `τ = 1` it is exactly
  `A'y + c = 0`).
- `(G)` is the homogeneous gap equation: at an exact homogeneous-feasible
  point, rearranging and using `(P)`/`(D)` gives `κ = c'x+b'y = s'y/τ ≥ 0`
  for `τ>0`.  Thus `κ ≥ 0` is consistent.  This is the **corrected** sign; the earlier
  `c'x + b'y + κ = 0` forced `κ = −s'y ≤ 0` and is rejected
  (see `docs/reviews/HSD_SIGN_HISTORY.md`).

A zero-residual point `(x, y, s, τ, κ)` with `τ > 0` yields a feasible primal
`(x/τ, s/τ)` and a feasible dual `y/τ`, and `(G)` becomes the closed duality gap
at the solution. A point with `τ = 0` yields an infeasibility ray (§7).
Production iterates are instead strictly interior **infeasible-start** points;
their three linear residuals need not vanish.

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
   (slack × dual) plus `τ κ`, exactly as it appears in the normalized-gap
   observable `μ̄ = (s'y + τ κ)/(ν + 1)` (§6). The homogeneous skew structure
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

Setting the three blocks to `(0; s; −κ)` reproduces *exactly* the frozen HSD
equations (§1.3):

- block 1: `A'y + c τ = 0` → `(D)`;
- block 2: `−A x + b τ = s`, i.e. `s = b τ − A x` → `A x + s − b τ = 0` → `(P)`;
- block 3: `−c'x − b'y = −κ`, i.e. `−c'x − b'y + κ = 0` → `(G)`
  (equivalently `κ = c'x+b'y`; together with `(P,D)`, `s'y=τκ`).

Because `Q` is skew-symmetric, the self-duality identity gives, for any
zero-residual point,

```
0 = (x, y, τ)·Q(x, y, τ) = (x, y, τ)·(0, s, −κ) = y's − τ κ = s'y − τκ,
```

i.e. `s'y = τκ` at a feasible HSD point.  Combined with `s ∈ K`, `y ∈ K*`,
`τ, κ ≥ 0` (so `s'y ≥ 0` and `τκ ≥ 0`) and driving the observable
`μ̄ = (s'y + τκ)/(ν+1) → 0`, this forces the complementarity conditions
`s'y → 0` and `τκ → 0` at the solution. This is the rigorous reason the
correct complementarity is `s'y` (+ `τκ`), and why the whole formulation is
called *self-dual*: the coefficient operator of the homogeneous system is its
own (skew) adjoint.  For an arbitrary infeasible-start iterate, the stronger
diagnostic identity is

```
s'y − τκ = −x'rD + y'rP − τ rG,
```

obtained by substituting
`Qz = (rD; s−rP; rG−κ)`.  Every residual implementation and fixture must
satisfy this identity to roundoff.

> **Sign note.** Some texts write the third block as `c'x − b'y + κ = 0`
> (dual in the `max b'y` orientation) and a matrix `[[0, A, −b], [−A', 0, c],
> [b', −c', 0]]`. That is a *different* sign orientation of the same operator,
> conjugate by flipping the sign of `y` / of `b'y`. The frozen spec (with
> `A'y + c τ = 0`, `−c'x − b'y + κ = 0`, dual objective `−b'y`) is reproduced
> exactly by the `Q` written above. Section 10 records the sign alignment.

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

At a zero-residual point, acting on `(x, y, τ)` yields `(0; s; −κ)` (so
`Q·(x;y;τ) = (0; s; −κ)`), and skewness yields `s'y − τκ = 0`, i.e. `s'y = τκ`
at a feasible point (with `s ∈ K`, `y ∈ K*`, `τ, κ ≥ 0` this closes the gap as
`μ̄ = (s'y + τκ)/(ν+1) → 0`).

---

## 3. Infeasible-start residual homotopy and Newton directions

### 3.1 Observable gap versus local centering target

For a strictly interior iterate define the observed normalized gap

```
μ̄ = (s'y + τκ) / (ν + 1).
```

This is always a measurement, not an assertion that the iterate lies on an
exact central path.  For `ν != 1`, imposing simultaneously zero linear
residuals, `s∘y=μ e`, and `τκ=μ` would imply both `s'y=νμ` and
`s'y=τκ=μ`, a contradiction for `μ>0`.  Therefore production uses:

- strict cone/scalar interiority;
- affine Newton equations that contract `(rP,rD,rG)`;
- `σμ̄` only as a **local complementarity target** in the corrector.

No document or test may call an arbitrary interior infeasible-start point an
"exact central-path point".  Exact zero-residual fixtures test the embedding;
interior residual-bearing fixtures test initialization and the Newton
homotopy.  These are different fixtures.

### 3.2 Residuals and complementarity linearization

Given a current iterate, define residuals

```
rP = A x + s − b τ
rD = A'y + c τ
rG = −c'x − b'y + κ
```

and the local complementarity target for `σ ∈ [0, 1]`:

```
s ∘ y = σ μ̄ e_K ,   τ κ = σ μ̄ .
```

For a symmetric-cone algebra with Jordan product `∘` and left-multiplication
operators `L_u : v ↦ u ∘ v`, the Newton linearization toward
`s ∘ y = σμ̄ e` is

```
L_y ds + L_s dy  =  σ μ̄ e  − (s ∘ y)  −  (ds_a ∘ dy_a),
```

where `(ds_a, dy_a, dτ_a, dκ_a)` are the affine (predictor) directions entering
the second-order correction `ds_a ∘ dy_a`. The scalar complementarity
linearizes as

```
τ dκ + κ dτ = σμ̄ − τκ − dτ_a dκ_a.
```

### 3.3 The full Newton system (generic form)

The Newton direction `(dx, dy, ds, dτ, dκ)` solves

```
(P)   A dx + ds − b dτ = −rP
(D)   A' dy + c dτ     = −rD
(G)   −c'dx − b'dy + dκ = −rG
(C1)  L_y ds + L_s dy  =  σμ̄ e − (s∘y) − (ds_a∘dy_a)
(C2)  τ dκ + κ dτ      =  σμ̄ − τκ − dτ_a dκ_a
```

- **Predictor (affine, Mehrotra):** `σ = 0`, `μ` absent, second-order terms
  omitted. Solves the pure affine-scaling direction `(dx_a, dy_a, ds_a, dτ_a, dκ_a)`.
- **Corrector (centering):** `σ` chosen (e.g. Mehrotra
  `μ_cent = μ̄·(μ̄_aff/μ̄)³`), and the second-order correction `ds_a ∘ dy_a`,
  `dτ_a dκ_a` added. The final direction is the Mehrotra combination.

The cone linearization `(C1)` is what brings the **barrier derivatives** into
the system: `L_s` and `L_y` are the Jordan-algebra (barrier-derivative)
operators of the cone at the current point. For `K = R_+^m` (LP) `L_u =
diag(u)`; for SOC/PSD `L_u` is the corresponding Jordan product, and in the
Nesterov–Todd scaling both terms are expressed in one scaled frame.

### 3.4 The single-step residual invariant

The corrector contributes **zero** to the `(P,D,G)` right-hand sides.  Thus the
combined affine-plus-corrector direction always satisfies

```
drP = −rP,   drD = −rD,   drG = −rG.
```

All five variables `(x,y,s,τ,κ)` are updated with one common accepted step
length `α`.  Since the three residual maps are linear,

```
(rP,rD,rG)(z + α dz) = (1−α) (rP,rD,rG)(z)
```

up to the documented arithmetic tolerance.  This is a per-step hard invariant.
Separate primal and dual step lengths are not an allowed production option;
they break the homotopy, especially `rG`, and invalidate the invariant.

---

## 4. Frozen NT orientation and bordered reduction

### 4.1 NT orientation

The orientation is fixed by

```
Θ(y) = s,       G := Θ^{-1},       G(s) = y.
```

Let `R` be the self-adjoint NT square-root automorphism: `R²=Θ` and
`λ=R(y)=R^{-1}(s)`.  Scaled directions are
`ds_hat=R^{-1}(ds)` and `dy_hat=R(dy)`.  If

```
L_λ(ds_hat + dy_hat) = rc,
```

then, with `h = R L_λ^{-1}(rc)`, the unscaled relation is exactly

```
ds + Θ dy = h.
```

For an orthant `Θ=Diag(s/y)`.  For a Euclidean Jordan algebra
`Θ=Q_w` with `Q_w(y)=s`.  For PSD matrices, if `P Y P=S`, then
`Θ[Z]=P Z P`; `P` is the congruence matrix realizing `Θ`, whereas `R` is
the square-root **operator**.  These names are not interchangeable.

### 4.2 Reduction to `(dx,dτ)`

Starting only from the frozen Newton equations and
`ds+Θdy=h`, define

```
G = Θ^{-1}
H = A' G A
q = c − A' G b
p = −rD − A' G (h+rP)
r' = τ (c' + b' G A)
d = κ − τ b' G b
g = hτ + τ rG − τ b' G (h+rP),
```

where `hτ` is the right-hand side of `(C2)`.  Direct substitution gives

```
[ H   q ] [dx] = [p]
[ r'  d ] [dτ]   [g].
```

When `H` is nonsingular, factor it once and compute

```
H u = q,       H w = p
dτ = (g − r'w) / (d − r'u)
dx = w − u dτ.
```

Recovery is

```
dy = G(A dx + h + rP − b dτ)
ds = h − Θ dy
dκ = −rG + c'dx + b'dy.
```

The signs `q=c−A'Gb`, `d=κ−τb'Gb`, and the plus sign in
`r'=τ(c'+b'GA)` are frozen.  The former `(dx,dy)` display in this document
had incompatible signs and is rejected.

The Schur denominator `d−r'u` is a numerical gate.  A non-finite or
type-scaled-near-zero value must route to the independent full-Newton
diagnostic or fail closed; it must never be clamped, silently regularized, or
reported as a successful direction.

### 4.3 Independent full-Newton oracle

For a small LP, an oracle assembles the full matrix directly in the unknown
order `(dx,dy,ds,dτ,dκ)`:

```
[  A     0     I    −b   0 ]   # P
[  0     A'    0     c   0 ]   # D
[ −c'   −b'    0     0   1 ]   # G
[  0     S     Y     0   0 ]   # C1, S=Diag(s), Y=Diag(y)
[  0     0     0     κ   τ ]   # C2
```

It forms every residual and right-hand side from the equations above and
solves in higher precision using a pivoted dense QR (or a comparably stable
general full-matrix solve).  It must not call the bordered assembly/recovery,
`Theta/G` helpers, production residual helpers, or the production factor
cache.  Both affine and combined corrector directions are checked.  Direction
components are compared only when the oracle reports a unique,
well-conditioned solution; the five direct equation residuals are always
checked.  An oracle failure rejects the fixture and cannot make the production
gate pass.

### 4.4 Rank-deficient setup route

`H=A'GA` is positive definite only after eliminating the numerical nullspace
of `A`.  Setup therefore performs a type- and scale-aware column-pivoted RRQR
of canonical `A` (or an equivalent rank-revealing orthogonal decomposition):

1. classify the numerical rank and reject an ambiguous rank threshold as
   `InsufficientPrecision`;
2. form a stable row-space basis `V_r` and the reduced full-column-rank map
   `A_r=A V_r`;
3. inspect `c_N=(I−V_r V_r')c`;
4. if `c_N` is significant, use `−c_N` only as a **candidate** nullspace
   dual-infeasibility ray and require the original-coordinate certificate gate;
5. if `c_N` is negligible, solve in reduced coordinates and reconstruct the
   minimum-norm representative `x=V_r x_r`.

RRQR and basis construction are setup/cold-path work and may allocate.  The
rank decision and reconstruction maps are frozen before the hot loop.  Adding
a diagonal to singular `H`, silently switching to a least-squares answer, or
accepting a rank decision without an original-coordinate certificate is
forbidden.

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
- **Symmetric-cone linearization.** In `(C1)`, `L_s` and `L_y` are Jordan
  left-multiplication operators, not barrier gradients.  The frozen NT
  orientation is `Θ(y)=s`, `G(s)=y`, with square-root automorphism
  `R(y)=R^{-1}(s)=λ` as in §4.1.
- **Nonsymmetric cones.** Exp/Power use explicit primal/dual barrier gradients
  and Hessians.  Their full-Hessian reference route is separate and must not be
  described as a Jordan `L_s/L_y` operation.

---

## 6. The observed normalized gap `μ̄` and why `ν + 1`

For a product cone `K` with barrier degree `ν`, a perfectly centered
**local complementarity target** at scalar level `t` has `s∘y=t e` and
`τκ=t`.  Taking the trace functional over the Jordan blocks gives

```
s'y  =  trace of (s ∘ y) over the Jordan algebra  =  ν t,
```

and the scalar cone `R_+` for `(τ, κ)` contributes exactly one more:

```
τκ = t .
```

Therefore `s'y + τκ = (ν+1)t`.  This motivates the observable

```
μ̄ = (s'y + τ κ) / (ν + 1).
```

**Why `ν + 1` (not `n + 1`, not `m + 1`).** The complementarity model contains the two
conjugate pairs `(s, y)` (in `K`, degree `ν`) and `(τ, κ)` (in `R_+`, degree
`1`). The barrier degree of the *full* HSD cone `K × R_+` is `ν + 1`; the gap
is normalized over all `ν + 1` barrier degrees. The dimension of the
*total variable space* (`n + m + 2`) is irrelevant. For `K = R_+^m` (LP),
`ν = m` and the denominator is `m + 1`, **not** `n + 1`.
This normalization does not assert simultaneous exact linear feasibility and
positive exact centrality; §3.1 explains why that assertion is forbidden.

---

## 7. Certificates, ray normalization, and optimal recovery

Termination status is assigned **only** from verified certificates in the
original coordinates (never from raw `τ`/`κ`). The frozen conventions:

- **Optimal.** When `τ > 0`, recover `x = x̂/τ`, `s = ŝ/τ`, `y = ŷ/τ` from the
  HSD point, and verify in *original coordinates*: `Ax + s = b`, `s ∈ K`,
  `A'y + c = 0`, `y ∈ K*`, `c'x + b'y = s'y = 0` (gap), all within tolerance.
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

The fixtures deliberately separate exact zero-residual embedding points from
strictly interior infeasible-start points. Tests verify the three residuals,
`Q+Q'=0`, the arbitrary-iterate skew diagnostic of §1.5, `μ̄`, strict
membership where claimed, and certificate signs. No interior-start fixture is
called centered merely because all cone variables are positive.

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
- (G) `−c'x − b'y + κ = (1+1) − (1+1) + 0 = 0` ✔   (κ = c'x+b'y = 0)
- `Q + Q' = 0` ✔ (verified in test)
- `ν = 2`, `μ = (s'y + τκ)/(ν+1) = (0 + 0)/3 = 0` ✔

**A strictly interior, exact embedding point (not a centered point)**
`(x, y, s, τ, κ) = ((0.4, 0.4), (1, 1), (0.6, 0.6), 1, 1.2)`:
- (P) `(0.4,0.4)+(0.6,0.6)−(1,1)·1 = 0` ✔
- (D) `(1,1)+(−1,−1)·1 = 0` ✔
- (G) `−( −0.4−0.4) − (1+1) + κ = 0` → `κ = c'x + b'y = 1.2 = s'y ≥ 0` ✔
- `κ = 1.2 > 0`, `τ = 1 > 0`, so `(τ, κ)` is strictly interior ✔
- `μ̄ = (s'y + τκ)/(ν+1) = (1.2 + 1.2)/3 = 0.8` ✔
- it is not centered: `s⊙y=(0.6,0.6) != 0.8e` and `τκ=1.2 != 0.8` ✔

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
- (G) `−c'x − b'y + κ = (1) − (1·1+1·0) + 0 = 1 − 1 = 0` ✔
- `s ∈ SOC` (`(0,0)`) ✔; `y ∈ SOC` (`1 ≥ |0|`) ✔; `τ = 1 ≥ 0`, `κ = 0 ≥ 0` ✔
- `ν = 2` (SOC barrier degree), `μ = (s'y + τκ)/3 = (0 + 0)/3 = 0` ✔.

**Strictly interior infeasible-start fixture**
`(x,y,s,τ,κ)=(0,(1,0.2),(2,0.25),1,1)`:

- `s ∈ int SOC` because `2>|0.25|`; `y ∈ int SOC` because `1>|0.2|` ✔
- `τ=1>0`, `κ=1>0` ✔
- `rP=(1,−0.75)`, `rD=0.2`, `rG=−0.2` (nonzero by design) ✔
- `s'y−τκ=1.05` and
  `−x'rD+y'rP−τrG=0+(1−0.15)+0.2=1.05` ✔
- `μ̄=(s'y+τκ)/3=(2.05+1)/3=1.016666…` ✔

### 9.3 A small SDP program

`K = PSD_2`, `m=3`, `n=1`. Execution coordinates are
`svec([[a,b],[b,c]])=(a,√2 b,c)`:

```
A = (1,0,1),   b = (4,√2,2),   c = −3.
```

Let `S=[[4,1],[1,2]]`, `Y=[[1,0.5],[0.5,2]]`, so
`s=svec(S)=(4,√2,2)`, `y=svec(Y)=(1,√2/2,2)`. With `x=0`, `τ=1`,
and `κ=9`:

- (P) `Ax+s−bτ=0`; (D) `A'y+cτ=(1+2)−3=0` ✔
- `dot(s,y)=4+(√2)(√2/2)+4=9=tr(SY)` ✔
- (G) `−c'x−b'y+κ=−9+9=0`; hence `s'y=τκ=9` ✔
- `det(S)=7>0`, `det(Y)=1.75>0`, so both matrices are strictly interior ✔
- `μ̄=(9+9)/3=6`; the point is exact and interior but not centered ✔

Using raw packed dot would give `4·1+1·0.5+2·2=8.5`, which is wrong.
This fixture must fail if execution complementarity bypasses `svec`.

---

## 10. Sign convention: frozen vs. historical, and production alignment

### 10.1 The gap sign (resolved)

The authoritative sign is `−c'x−b'y+κ=0` with `κ=c'x+b'y` and,
at exact feasibility, `s'y=τκ`. This is consistent with `κ≥0` and admits
strictly interior embedding points; it does not create the contradictory
positive exact central-path manifold rejected in §3.1.
The earlier `c'x + b'y + κ = 0` is **rejected** because it forces
`κ = −s'y ≤ 0` at every feasible point, contradicting `κ ≥ 0`. The full
reasoning and the skew-operator consequence are recorded in
`docs/reviews/HSD_SIGN_HISTORY.md`.

### 10.2 Production HSD state follows the corrected frozen sign

`src/hsd/hsd.jl` and `src/hsd/nonnegative_hsd.jl` must implement the corrected
frozen convention: `hsd_gap_residual` returns `−c'x − b'y + κ`, the
complementarity is `s'y + τκ`, and the implementation field `mu` is the observable
`μ̄ = (s'y + τκ)/(ν+1)`, with `ν` the canonical
barrier degree (from `ConeProductLayout`).  The forbidden pairings `dot(s, x)`,
`b's`, and `(x, τ) ∈ K*` appear nowhere in the HSD core.

### 10.3 `ν+1` vs `n+1`

- Frozen spec: denominator `ν + 1` (canonical barrier degree, §6).
- The production HSD uses the canonical `ConeProductLayout.barrier_degree`, so
  its observable is `μ̄ = (s'y + τκ)/(ν+1)` with `ν` the sum of the per-block barrier degrees
  (e.g. `m` for a nonnegative block of dim `m`, `2` for SOC, `n` for PSD, `3`
  for Exp/Power).  For a pure LP with `m = n`, this coincides with the naive
  `n+1`; for mixed/asymmetric cones it is the correct `ν+1`.  This
  normalization does not assert simultaneous exact linear feasibility and
  positive exact centrality; §3.1 explains why that assertion is forbidden.

### 10.4 Certificate sign convention (aligned)

- Frozen & production: primal-infeasible ray `A'y ≈ 0`, `y ∈ K*`, `b'y < 0`,
  normalize `−b'y = 1`; dual-infeasible ray `−A x ∈ K`, `c'x < 0`, normalize
  `−c'x = 1`.  `src/certificates/certificates.jl` verifies both rays in
  **original coordinates** through the reconstruction chain, including explicit
  `y ∈ K*` and the ray's own cone-membership `s_r = −A x ∈ K`.

## 11. References / notation glossary

- Nesterov & Todd (1998): "Primal-dual interior-point methods for
  self-scaled cones" (the `Q = −Q'` self-dual operator and NT scaling).
- Andersen & Andersen (2000), "The MOSEK interior point optimizer for linear
  programming: implementation of the homogeneous algorithm" — the conic
  homogeneous embedding with `A'y + cτ = 0`, `−c'x − b'y + κ = 0`.
- Mehrotra (1992) — predictor/corrector.
- Wright (1997), *Primal-Dual Interior-Point Methods*, §6 (τ/κ-elimination).

Notation: `∘` Jordan product; `ν` barrier degree; `K*` dual cone; `e_K` cone
identity; `()` transpose.
