# PR-02 evidence: SOC rank-2 expansion algebra, independently verified

Date: 2026-09-11. Baseline HEAD `7092cc2` (after PR-00 and PR-04A).

## What was run

```
julia --startup-file=no --project=. validation/clarabel_borrowing/soc_rank2_gate.jl
```

**Result: 375 pass, 0 fail, 1.0 s.**

The gate is deliberately self-contained: it does not call SDPX's SOC scaling
builder, its `theta_apply!`, any rank-2 adapter, or any production assembly
routine. It builds `Theta` from the definition `Theta = Q_w = 2ww' - J`,
derives `(D, u, v)` from Clarabel's formulas, and solves the extended system
with dense `\`. A shared implementation cannot make these four claims agree.

## Results

| Gate | Claim | Worst observed |
|---|---|---|
| A | `Theta == D + u*u' - v*v'` | metric residual `7.28e-12` |
| B | eliminating the two auxiliaries reproduces `-Theta` exactly | `3.64e-12` absolute, `2.77e-16` relative |
| C | un-expanded and extended solves give the same direction | `5.23e-12` |
| C | auxiliary recovery `z_v = -v'y`, `z_u = +u'y` | `4.89e-14` |

Covered: SOC dimensions 3/8/16/32/128 crossed with scaling spreads
0.05/1.0/8.0, four random normalized scaling points each (each satisfying
`w1^2 - ||w_tail||^2 = 1`, which is the condition SDPX enforces).

## Two corrections to the plan

1. **The storage crossover is k = 6, not "small".** The plan says the expanded
   form needs about `3k+2` slots against `k(k+1)/2` for packed dense, and
   instructs keeping small SOC dense, without giving the threshold. Computed
   exactly:

   | k | packed | expanded | winner |
   |---|---|---|---|
   | 4 | 10 | 14 | dense |
   | 5 | 15 | 17 | dense |
   | 6 | 21 | 20 | expanded |

   So any `dense_small` cutoff must be **below 6**. This is now asserted in the
   gate rather than left to prose.

2. **The plan's byte figures are numerically right but worth stating exactly.**
   `8390656 * 4 * 8 = 268,500,992 B` = 256.06 MiB and
   `12290 * 4 * 8 = 393,280 B` = 384.06 KiB for Float64x4. Both recomputed and
   asserted (payload only, excluding indices, copies, ordering and fill).

## What this does and does not establish

Establishes: the expansion is an **exact representation change** of the linear
system for the stated `Theta`, and the auxiliary variables are recoverable.

Does **not** establish: that SDPX's `apply_Theta!` equals `Q_w` in
`Theta = D + uu' - vv'` coordinates (that mapping is still a PR-02 implementation
task); anything about factorization fill, memory in situ, or speed; and nothing
about PSD, Exp or Power. Exponent `eta` was taken as 1 (SDPX's normalized
scaling point); the `eta^2` scaling of the general Clarabel form is verified
algebraically by inspection only, not by this gate.
