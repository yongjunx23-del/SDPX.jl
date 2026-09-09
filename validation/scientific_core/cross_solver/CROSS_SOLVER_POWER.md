# Cross-solver diagnosis of the failing Power-cone case (T0)

Status: **first bounded differential completed. The case is not ill-defined.**
Evidence: `local-archives/high-precision-ecosystem-20260908/cross-solver-power/`.
Scripts: `validation/scientific_core/cross_solver/{mosek_power.py,clarabel_power.jl,sdpx_power.jl}`.

## The problem is well-posed

Canonical data (only `A/b/c` of `fixtures/factor_affine_trial_17.toml`; captured
iteration state is not problem data):

    min  t1 + t2 + t3
    s.t. t_i >= 0
         (t_i, 1, a_i) in POW3^{0.5,0.5}      i = 1..3
    a = (0.626678964309454, 0.3230223181314613, -0.7919401216799509)

- Closed form: `sqrt(t_i*1) >= |a_i|`, so `t_i* = a_i^2`.
- Exact optimum: `91209111564668556464635313382413 /
  81129638414606681695789005144064 = 1.12423909864544834210379659915689...`
- Strict primal interior: `t_i = a_i^2 + 1` gives every power determinant exactly 1.
- Strict dual interior: orthant duals `1/2`, power duals `(1/2, (a_i^2+1)/2, -a_i)`
  give exact stationarity and `4uv - w^2 = 1`.
- `A'A = 2I`; no redundant rows.

The optimum is on the cone boundary, so the barrier Hessian becomes ill-conditioned
as `mu -> 0`. That is normal interior-point behavior on a well-posed problem; it is
not an ill-posed model (MOSEK defines ill-posedness by near-infeasibility, dependent
constraints, or non-attained optima).

## Results on identical data (trial 17 and 19 fixtures encode the same problem)

| Solver | Arithmetic | Tolerance | Status | Objective error vs exact | Complementarity |
| --- | --- | --- | --- | --- | --- |
| MOSEK 11.1.3 | Float64 | default | optimal | 3.93e-17 | 7.87e-17 |
| MOSEK 11.1.3 | Float64 | tightened 1e-12 | optimal | 3.93e-17 | 7.87e-17 |
| MOSEK 11.1.3 | Float64 rotated-SOC | default/1e-12 | optimal | 3.93e-17 | 7.87e-17 |
| Clarabel 0.11.1 | Float64 | default 1e-8 | SOLVED | 2.98e-09 | 8.47e-09 |
| Clarabel 0.11.1 | Float64 | tightened 1e-12 | SOLVED | 3.04e-14 | 2.22e-12 |
| Clarabel 0.11.1 | BigFloat256 | tightened 1e-30 | SOLVED | 7.29e-31 | 3.11e-30 |
| Clarabel 0.11.1 | BigFloat512 | tightened 1e-30 | SOLVED | 7.29e-31 (tolerance-limited) | 3.11e-30 |
| Clarabel 0.11.1 | Float64 SOC | default | SOLVED | 5.75e-09 | 1.32e-08 |
| Clarabel 0.11.1 | BigFloat256 SOC | tightened | ALMOST_SOLVED | 8.67e-22 | ~0 |
| **SDPX (dev `113869c`)** | **Float64** | default | **numerical_breakdown** | **1.12 (obj 0)** | n/a |
| SDPX (dev `adcacf8`) | BigFloat256 | default | optimal | 1.48e-26 (rel 1.32e-26) | primal 1.88e-26 / dual 4.07e-26 |
| SDPX (dev `adcacf8`) | BigFloat512 | default | optimal | 3.74e-54 (rel 3.33e-54) | primal 2.31e-52 / dual 4.69e-52 |

Independent audit (not solver residual code): each returned value is converted
exactly to `Rational{BigInt}`, then the original-coordinate primal residual, dual
residual, cone/dual-cone membership, complementarity and objective error are
computed. All reported `SOLVED`/`optimal` points are feasible and near-optimal.

Audit corrections (first T0 draft had defects, now fixed):
- The Clarabel cone-block audit now uses the actual rows: prefix orthant rows
  1-3, Power blocks rows 4-6, 7-9, 10-12. The first draft treated rows 1-3 as a
  Power block and omitted rows 10-12.
- The audit rebuilds `A/b` in the run's own precision `T`, not Float64.
- The SOC arm uses a rounded `sqrt(2)` map (Clarabel 0.11.1 has no rotated-SOC
  cone type), so it is a near-equivalent cross-check; the native Power arm is
  the authority for the solver's power-cone path.
- The SDPX BigFloat arm now passes `Model(BigFloat; precision_bits=...)`
  explicitly; the first draft's "512" run was silently reset to 256 by
  `optimize!`. The corrected 512-bit run reaches 3.74e-54 objective error.

## Where SDPX Float64 fails (parent debug capture)

`SDPX_DEBUG_LINE_SEARCH=1` on the Float64 run shows the final exhausted search
at an already near-converged iterate (`current_merit=2.684e-7`, `p=1.0408e-7`,
`d=1.1119e-7`, `gap=2.6841e-7`). Trials with `alpha` above the progress floor
fail the **nonsymmetric scaling** gate with typed reasons:

- `NS_SCALING_CONJUGATE_FAILED` + `NS_SCALING_NO_FALLBACK` +
  `NS_CONJUGATE_ITERATION_LIMIT` (root solve does not converge at Float64);
- `NS_CONJUGATE_HESSIAN_NOT_SPD`;
- `NS_SCALING_FALLBACK_DENOMINATOR` / `NS_SCALING_FALLBACK_NOT_SPD` with
  `NS_SCALING_GRAM_NONSYMMETRIC`, `NS_SCALING_BFGS_DENOMINATOR`,
  `NS_SCALING_FORCED_DUAL_HESSIAN`.

Trials that do satisfy the scaling neighborhood land at `alpha ~ 2.3e-23`, i.e.
far below the useful-progress floor `2cbrt(eps) ~ 1.21e-5`, and fail only
`progress=false`. So the Float64 failure is a **scaling-construction failure at
near-boundary points**, not a merit/homotopy/termination failure. Note
`base.record.step_size`/`backtracking` are only written on success, so the
reported `step_size=1.921e-4` is the previous accepted step, not the rejected
alpha.

This is exactly what the reviewed repair trajectory (compensated half-Power
factor + full-gap polynomial root + native certified pair) targets: the legacy
log-based factor/root path cannot construct the scaling at these points, while
the compensated factor and polynomial root are independently certified there.

## Diagnosis

1. **The model is well-posed and solvable.** Two independent solvers, two cone
   representations and three arithmetic precisions all solve it; the exact
   optimum and strict primal/dual interiors are known analytically.
2. **The outlier is SDPX's Float64 Power-cone path.** SDPX returns
   `numerical_breakdown` at Float64 while the *same source* at BigFloat256/512
   reaches `optimal` with ~1e-26 residuals. This is a precision-sensitive
   algorithm/implementation failure, not a property of the problem.
3. **"Ill-defined" is not the right explanation.** The failure must be fixed in
   the Float64 nonsymmetric scaling/root/corrector path. External solvers and
   SDPX's own BigFloat path are now available as independent references for the
   actual iteration state, not just the final optimum.
4. **The rotated-SOC form is a legitimate cross-check, not a fix.** For alpha=1/2
   the power cone is exactly a rescaled rotated SOC. Both forms pass externally.
   A solver that only passes the SOC form has not demonstrated native power-cone
   correctness.

## Mechanisms other solvers use (source-read, actionable for R0-P)

- Clarabel 0.11.1 `coneops_powcone.jl`: one-sided Newton for the primal root with
  `eps(T)`-based stopping (`_newton_raphson_onesided`, `nonsymmetric_common.jl:170`),
  Mosek-style BFGS primal-dual scaling with explicit safety guards
  (`|de1|>sqrt(eps)`, `|de2|>eps`, `dot_sz>0`, `dot_delta_sz>0`) that fall back to
  dual scaling near the boundary, an explicit unrolled 3x3 Cholesky for the
  higher correction returning zero on non-positive pivots, and per-cone
  backtracking feasibility checks. `Settings{BigFloat}` does **not** auto-tighten
  tolerances; high-precision runs must set them and
  `static_regularization_proportional = eps(T)^2` explicitly.
- MOSEK solves both forms to machine precision at Float64 with default
  tolerances; it does not need a higher precision request here. MOSEK's own
  guidance separates requested tolerance from arithmetic precision and warns the
  power cone needs "more advanced and less efficient algorithms" than quadratic
  cones.
- SDPX's BigFloat path already solves this case; the practical next step is to
  locate the first Float64 stage whose error the BigFloat reference does not
  share (root interval, scaling/BFGS guard, third contraction, or projection),
  rather than adding more local certificates to a frozen failing state.

## Reproduction

    python3 validation/scientific_core/cross_solver/mosek_power.py \
        validation/scientific_core/fixtures/factor_affine_trial_17.toml <out>
    julia --project=<clarabel-env> validation/scientific_core/cross_solver/clarabel_power.jl \
        validation/scientific_core/fixtures/factor_affine_trial_17.toml <out>
    julia --project=<sdpx-provider-env> validation/scientific_core/cross_solver/sdpx_power.jl \
        validation/scientific_core/fixtures/factor_affine_trial_17.toml <out>

Environments: Clarabel.jl 0.11.1 (`ChSpM`) from the local depot; SDPX development
checkout with BFLA/MFLA/MultiFloats providers; MOSEK 11.1.3 Python package with the
local license. No SDPX production default or gate was changed by this diagnostic.
