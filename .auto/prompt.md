# Autoresearch: CSDR α3 等式预处理与求解加速

## Objective
Improve the CSDR α3 (J40 a15 mu200 x1, twice subtraction) 4-thread solve time on the
frozen input `/tmp/csdr-alpha9-twice/solve-alpha3.bin` (8400 spectral variables,
42 homogeneous sum-rule equalities, 4200 Lorentz cones, Float64x4).

Baseline (4 Julia threads + 4 BLAS threads, warmup + median of 3): **19.46s / 101 iters**.
Frozen benchmark reference: `benchmark/autoresearch/csdr_alpha3_x4.jl` (17.70s on the
original machine; this machine measures 19.46s for the same code).

## Metrics
- **Primary**: `solver_seconds` (lower is better) — median of 3 solves after warmup
- **Secondary**: `allocation_bytes`, `iterations` (must stay 101 unless certified better)

## How to Run
```bash
JULIA_NUM_THREADS=4 PRECOND_VARIANT=baseline ./.auto/measure.sh
```
`PRECOND_VARIANT`: baseline | colnorm | elim | ruiz
`PRECOND_SCALING`: auto | none | equilibrate

## Files in Scope
- `benchmark/autoresearch/csdr_alpha3_precond.jl` — the benchmark (model builders for
  each preprocessing variant + solve + certificate checks)
- `src/hsd/native_hsd_public.jl` — fixed-trace Ruiz plan rebuild (bug fix, committed)
- `src/ingest.jl`, `src/program/equilibrate.jl` — Ruiz equilibration internals
- `src/kkt/specializations/fixed_trace_q3.jl` — fixed-trace Q3 plan

## Off Limits
- `benchmark/autoresearch/csdr_alpha3_x4.jl` — the frozen benchmark (trajectory SHA
  must stay intact)
- Tolerances, certificate semantics, objective value
- No benchmark-specific constants

## Constraints
- Objective must equal `-31.672155970636578` (atol 1e-10) — the certified value
- `certificate.valid == true`, status `:optimal`
- Deterministic: 3 runs must agree on iterations and objective
- No tolerance widening, no uncertified rank reduction

## What's Been Tried
- **baseline** (raw B): 19.46s / 101 iters / 2.32GB — reference
- **colnorm** (equality columns scaled to unit norm): 18.28s / 101 iters / 2.28GB
  (-6.1%, same iterations, certified). Only confirmed positive preprocessing.
- **elim** (RRQR-pivot analytic elimination of 42 variables → 8358 free vars):
  FAILED with `insufficient_precision`. Root cause: reduced objective has 42 orders
  of magnitude dynamic range (min |c| = 1.5e-42); the elimination itself is accurate
  (BigFloat256 diff 6.8e-12) but the reduced problem is intrinsically ill-conditioned.
  cond(P) = 1.8e5 for RRQR pivots — worse than cond(B) = 6.5e4.
- **ruiz** (Settings equilibration=:ruiz on fixed-trace route): CRASHED at iteration 0
  with `numerical_breakdown` before the fix. Root cause: `fixed_trace_plan` was built
  from the UNSCALED canonical while the state was built from the scaled program.
  FIXED in `src/hsd/native_hsd_public.jl` (rebuild plan from `solve_reduced`).
  After fix: optimal + cert=true but 141.8s / 82.8GB — `equilibrate` materializes a
  dense 12642×8400 Float64x4 work copy (~3.4GB) and iterations stay 101, so Ruiz
  has no net benefit on this problem. The fix is still correct and worth keeping.
- **rownorm** (scaling B rows): INVALID — B rows are variables, not equalities;
  scaling them changes the problem (objective drifted to -539.7). Discarded.
- **Profile** (4-thread baseline): hotspots are `_structured_mulacc` (8214 samples),
  `mulacc_x4` (5644), `syrk` (5002), `gemv` (3107), `_product_hsd_newton_residual_ok`
  (3191). Time is dominated by linear-algebra kernels, NOT by iteration count or
  equality conditioning. Preprocessing that does not reduce iterations has
  bounded upside (~6%).

## Key Insight
Iterations are invariant (101) under every conditioning change tried so far. The
bottleneck is per-iteration kernel cost. Directions with real upside:
1. Reduce iterations: initial point (omega), step rule, parameter strategy — these
   live in the legacy `SolverOptions` API (predictor/step_rule/parameter_strategy/
   beta/gamma), not in `Settings`. May require switching the benchmark to
   `SDPX.solve_socp` + `SolverOptions`.
2. Speed kernels: mulacc/syrk/gemv SIMD or structural exploitation.
3. `Settings.scaling` (:auto|:none|:equilibrate) — untested on this problem.
