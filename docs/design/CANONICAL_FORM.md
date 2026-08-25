# Frozen Canonical Conic Form & HSD Convention (Lead Agent)

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

## 2. Production HSD (Nesterov–Todd style)

    A x + s - b tau = 0
    A' y + c tau = 0
    c' x + b' y + kappa = 0

    s in K
    y in K*
    tau >= 0
    kappa >= 0

**Central path:**

    mu = (s'y + τ·κ) / (nu + 1)

where `nu` is the barrier/cone rank degree. For a product cone,
`nu = sum(barrier_degree(block_i))`. **Forbidden** anywhere in the HSD core:

    dot(s, x)
    b' s
    (x, τ) in K*

These were the root errors of the previous draft HSDState.

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
