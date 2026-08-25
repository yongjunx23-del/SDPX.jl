# Frozen Canonical Conic Form & HSD Convention (v2)

**Status: FROZEN.** This document fixes the mathematical conventions that all
SDPX and PMP2SDP work must follow. Do not change it without a Lead review.
Authoritative for: canonical conic form, HSD equations, certificates, and the
`nu` complementarity denominator.

## 1. Canonical conic form

**Primal:**

    minimize      c' x
    subject to    A x + s = b
                  s in K

- `A`: m × n
- `x`: n-dim free variable
- `s`: m-dim product-cone slack
- `K = K_1 × K_2 × … × K_q` (product of native cones)

**Dual:**

    maximize      -b' y
    subject to    A' y + c = 0
                  y in K*

- `y : m`-dim dual cone variable, `y in K*`

Dimensions are always consistent: `x:n`, `y:m`, `s:m`.

## 2. Production HSD residual map (Nesterov–Todd style)

    A x + s - b tau = 0
    A' y + c tau = 0
    -c' x - b' y + kappa = 0

    s in K
    y in K*
    tau >= 0
    kappa >= 0

The three left-hand sides define `rP`, `rD`, and `rG`, respectively.  A solver
iterate is required to be strictly interior, but is **not** required to make
these residuals zero.  Zero residuals describe the exact homogeneous
embedding; the production algorithm is an infeasible-start residual homotopy.

**Sign note (Lead correction of the initial freeze).** The third equation must
carry **negative** coefficients on `c'x` and `b'y`. Reason: weak duality with
`s in K`, `y in K*` gives `tau*(c'x + b'y) = s'y >= 0` (verified from
`A'y + c tau = 0` and `A x + s = b tau`), so for `tau>0` the gap equation
`-c'x - b'y + kappa = 0` yields `kappa = c'x + b'y = s'y/tau >= 0`,
consistent with `kappa >= 0`. The sign as first
frozen (`c'x + b'y + kappa = 0`) forced `kappa <= 0` at feasible points; it is
corrected here. This matches the skew system `Q (x;y;tau) = (0; s; -kappa)` with
`Q = [[0, A', c]; [-A, 0, b]; [-c', -b', 0]]`, `Q = -Q'`.
See `docs/design/HSD_FORMULATION.md` §10.1 for the full analysis.

For an arbitrary iterate, skewness gives the exact diagnostic identity

    s'y - tau*kappa = -x'rD + y'rP - tau*rG.

In particular, exact embedding feasibility implies `s'y = tau*kappa`.  This
identity is a gate on residual/sign implementations; it is not a license to
call every strictly interior point "central".

**Complementarity observable and centering target:**

    mu = (s'y + τ·κ) / (nu + 1)

where `nu` is the barrier/cone rank degree. For a product cone,
`nu = sum(barrier_degree(block_i))`.  This `mu` is the observed normalized gap.
The targets `s ∘ y = sigma*mu*e` (or their NT-scaled equivalent) and
`tau*kappa = sigma*mu` are local Newton centering targets.  They do not define
an additional exact zero-residual central-path manifold: for `nu != 1`, that
would contradict the skew identity above.

The affine direction uses right-hand sides `(-rP,-rD,-rG)`; a corrector adds
zero to those three feasibility right-hand sides.  The accepted combined
direction must use one common step length `alpha` for `(x,y,s,tau,kappa)`, hence

    (rP, rD, rG)_new = (1 - alpha) * (rP, rD, rG)_old

up to roundoff.  Independent primal/dual step lengths are forbidden in the HSD
core because they destroy this residual-homotopy invariant.

**NT orientation:** `Theta` always maps dual-cone directions to primal-cone
directions and satisfies `Theta(y) = s`; `G = inv(Theta)` satisfies `G(s) = y`.
If `R` is the NT square-root automorphism, then `R^2 = Theta`,
`lambda = R(y) = inv(R)(s)`, and the scaled directions are
`ds_hat = inv(R)(ds)`, `dy_hat = R(dy)`.  No implementation may reuse `R`,
`Theta`, or `G` with the opposite orientation.

**Forbidden** anywhere in the HSD core:

    dot(s, x)
    b' s
    (x, τ) in K*

These were the root errors of the previous draft HSDState.

## 2.1 PSD execution coordinates

Every PSD block is represented in the HSD core by `svec`, with diagonal entries
unchanged and off-diagonal entries multiplied by `sqrt(2)`.  Therefore

    dot(svec(S), svec(Y)) = tr(S*Y).

Plain `dot` on unscaled lower-packed matrix entries is forbidden.  Row scaling,
dual adjoint pullback, and matrix reconstruction are distinct maps and must be
tested separately; setup freezes these maps and the hot step performs no
coordinate conversion or allocation.

## 3. Certificates (sign-convention fixed)

**Primal infeasible:** Farkas ray `y`:

    A' y ≈ 0
    y in K*
    b' y < 0

**Dual infeasible / primal unbounded:** ray `x`:

    -A x in K      (i.e. A x ∈ -K, or equivalently s_r = A x with -s_r ∈ K)
    c' x < 0

**Ray normalization** (choose one, global):

    -b' y = 1     (primal infeasible ray)
    -c' x = 1     (dual infeasible ray)

**Optimal recovery** (always from original coordinates):

    x/tau,  s/tau,  y/tau

Status (`Optimal` / `PrimalInfeasible` / `DualInfeasible`) is reported **only**
after a verified original-coordinate certificate, never from raw τ/κ alone.

## 4. Route / fast-path discipline

- Route labels (lp/soc/sdp/mixed) are **fast-path hints only**; correctness must
  not depend on them. Mixed families are first-class executable programs.
- Cone layout is built from the **canonical slack rows** (`s`, `y`), not from
  original variable blocks.
- PowerCone parameters must preserve source type/precision through the
  canonical program; conversion to target arithmetic happens only at the
  ExecutionPlan stage. No forcing to Float64.

## 5. Allocation / zero-gate

"0 Julia bytes warm step" is measured over 10 consecutive warm samples; **all**
samples must equal 0 (no `minimum`, no tolerance). BigFloat/native MPFR memory
is tracked separately from Julia allocation.
