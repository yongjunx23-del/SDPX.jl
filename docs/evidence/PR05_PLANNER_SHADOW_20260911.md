# PR-05 evidence: structure-aware core-route planner (shadow mode)

Date: 2026-09-11. Baseline: PR-04A. Deliverable is **advisory by default**.

## Why this is not wired as the default

Plan PR-05 asks to replace the fixed `full > 4*compact` rule with a planner over
`n,m,rank`, cone shape, `nnz(A)`, predicted fill, PSD packed dimension,
precision, provider kernel facts and peak live memory. It also states the
acceptance bar: *"相同数据和算术上下文产生可重现计划"* plus per-structure tests.

An honest attempt at the cost model (below) **disagrees with the legacy rule on
a large sparse small-block case**, scoring the full core as cheaper when the
compact Schur is the correct choice. Rather than tune the coefficients until the
output matched the incumbent — which would be fitting the model to the answer,
not measuring — the model ships in shadow mode:

| `SDPX_CORE_ROUTE_PLANNER` | Behaviour |
|---|---|
| unset / `legacy` | **Default.** Executes the previous `full > 4*compact` rule unchanged. The planner still computes and its reasons are available. |
| `model` | Executes the planner's decision. For evidence runs only. |

This keeps the default policy bit-identical while making the planner's
decisions observable, which is what the plan's receipt discipline requires
before any default change.

## The model

Work in floating-point operation units, both candidates scored with the same
term kinds:

```
full_factor_work   = 3.0 * triangular_nnz + Σ_{k>=6} k^3/3
full_solve_work    = 2 * rhs_count * full_dimension
compact_factor_work= compact_dimension^3 / 3
construction_work  = 0.5 * rank^2 * (1 + cone_dim)
compact_solve_work = 2 * rhs_count * compact_dimension
```

`3.0` is a deliberately pessimistic stated prior for sparse fill (so the model
does not under-price the full core and over-select the compact route). `k >= 6`
is the dense/expanded crossover measured by the PR-02 gate.

## Decisions observed

| Structure | Planner | Legacy rule | Agree? |
|---|---|---|---|
| one dense SOC `k=4000`, dims 4000/800 | `compact_schur` | compact | yes |
| many 3×3 blocks, dims 5000/900 | `full_core` | compact | **no — planner is right** |
| dims 20/3 (below floor) | `full_core` | compact | **no — planner is right** |
| fixed-trace Q3 plan | `full_core` (plan owns it) | n/a | by design |
| non-`:bordered` requested route | `full_core` (route owns it) | n/a | by design |
| 20 000 blocks of size 2, dims 40000/2000 | `full_core` | compact | **no — planner is wrong** |

## Known model defect (recorded, not hidden)

On the last row the `construction_work` term dominates: with `rank=2000`,
`cone_dim=38000` it evaluates to `0.5*2000^2*38001 ≈ 7.7e10`, against a modelled
full-core cost of `6.5e5`. A real Schur-complement build touches each cone row
once per rank column, i.e. `rank * nnz(Ar)`-ish, not `rank^2 * cone_dim`. The
first-order term is mispriced by roughly a factor of `rank`.

Calibrating it against real paired receipts is the next step and is what would
let the default be reconsidered. Until then the shadow default is the correct
engineering position.

## What was verified

- 78 assertions in `test/core_route_planner.jl`, including determinism across
  repeated calls (identical route, scores and reasons), explicit-ownership
  overrides, the below-floor case, and the shadow-mode contract asserted through
  a **real solve** rather than only the planner's return value.
- The planner performs no factorization and no problem-scaling allocation.

## Not established

No timing, no memory, no fill measurement, no default-policy improvement.
This PR delivers the mechanism and its evidence trail, not a speedup.
