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

---

# Addendum: the production mapping is NOT Clarabel's formula

Date: 2026-09-11. Follow-up after implementing `soc_rank2_parameters`.

## What the plan warned about, and what actually happens

Plan Section 3.2: *"实际移植必须先建立 SDPX `apply_Theta!` 与该 H 的坐标/缩放对应；
不要从 SDPX NT state 取一个名称相似的 w 就直接套公式。"*

Taking that warning literally and measuring, Clarabel's `(D, u, v)` formulas —
which assume `Theta = Q_w = 2*w*w' - J` with `w0^2 - ||w_tail||^2 == 1` — are
**wrong for SDPX by ~1e-1**, not by rounding.

Measured from the executable kernel `SymmetricCones.quadratic_apply!` (which is
what `theta_apply!` calls), SDPX's operator is

```
alpha = w0^2 + ww,   beta = w0^2 - ww,   ww = ||w_tail||^2

Theta = [ alpha          2*w0*w_tail'              ]
        [ 2*w0*w_tail    beta*I + 2*w_tail*w_tail' ]
```

Clarabel's `Q_w` has `2*w0^2 - 1` in the (1,1) entry; SDPX has
`w0^2 + ww`. They differ only there, by `2*ww`. SDPX's `nt_scaling!` does not
produce Clarabel's normalized point — `w0^2 - ww` is not 1 (measured values
1.06, 1.13, 1.38 for k = 3, 8, 16), and it is not a function of `mu` either
(it is invariant to scaling `(s,y)` together, as it must be).

## The mapping that does hold

Writing `Theta = D + u*u' - v*v'` with `u = [u0; u1*w_tail]`,
`v = [0; v1*w_tail]`, `D = [d0; beta*ones]`, matching the first column
(`u0*u1 = 2*w0`) and the tail block (`u1^2 - v1^2 = 2`) gives an exact identity:
`u0 = sqrt(alpha/2)`, `u1 = 2*w0/u0`, `v1 = sqrt(u1^2 - 2)`, `d0 = alpha - u0^2`.

**Verified to 2.2e-16** against `theta_apply!` for k in 2/3/5/8/16/32/64/128/512
across three seeds, in both assembled-matrix and action form
(`test/soc_rank2_mapping.jl`, 217 assertions).

## Consequence for the plan

The plan's Section 3.2 treats the coordinate correspondence as a preliminary
step. It is not preliminary — it changes the formula. Any implementation that
copied Clarabel's `update_scaling!` rank-2 block onto `SOCNTScaling.w` would
have produced a silently wrong KKT operator that no amount of downstream
verification would have been likely to catch, because the *dense* path would
still have been correct. This is recorded as the sharpest argument in favour of
the plan's own "build an independent reference first" rule.

## Status

`src/cones/symmetric/soc_rank2.jl` provides the mapping and the storage
predicate. It is **not wired into any route**: no KKT pattern, no assembly, no
solve path calls it. Wiring it in is the remaining PR-02 work and must keep the
dense path for k < 6.

`test/soc_rank2_mapping.jl` includes a negative control asserting that
Clarabel's parameters do *not* fit SDPX's metric, so a future "fix" back to the
published formula fails loudly instead of silently.
