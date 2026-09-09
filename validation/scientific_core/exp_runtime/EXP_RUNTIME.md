# R0-E frozen failing exponential-cone cases (exp_entropy_small, exp_logsumexp_small)

Frozen at dev HEAD (0.6.1, development/scientific-core-20260907).

## Well-posedness (external authority)

Both cases are formulated in `benchmark/general/exp.jl` (Mosek Modeling
Cookbook formulations) and are feasible, bounded and solvable:

- `exp_entropy_small` (n=3, seed 0x0e0001): min sum(r) s.t. p >= 0,
  sum(p) = 1, (-r_i, p_i, 1) in K_exp, i=1..3. Known optimum **-log(3)**.
  Feasible point e.g. p_i = 1/3, r_i = -(1/3)log(1/3).
- `exp_logsumexp_small` (n=3, seed 0x0e0002): min t s.t. z >= 0,
  (c_i - t, 1, z_i) in K_exp (sum z <= 1), c = 0.4*randn(Xoshiro(0x0e0002), 3).
  Known optimum **logsumexp(c) = 1.4040073747450363**.

Clarabel 0.11.1 (native canonical form `Ax + s = b`, presolve/equilibrate
disabled, no SDPX state used; script `clarabel_exp.jl`):

| case | Clarabel status | Clarabel obj | known |
|---|---|---|---|
| exp_entropy_small | SOLVED | -1.0986122892423997 | -1.0986122886681098 |
| exp_logsumexp_small | SOLVED | 1.4040073736800012 | 1.4040073747450363 |

The constructions were validated hierarchically (n=1,2,3 give -log(1), -log(2),
-log(3)), and the sign convention (`s = b - Ax`, orthant rows use A = -I) was
verified against a unit case whose optimum is exactly 1.

## SDPX dev failure (public path `GenericConicBenchmark.run_one`, Float64)

| case | status | iterations | cert |
|---|---|---|---|
| exp_entropy_small | numerical_breakdown | 24 | invalid |
| exp_logsumexp_small | numerical_breakdown | 42 | invalid |

Deep diagnostics (exp_entropy_small, raw result):
- reason = **:line_search_breakdown**, stage = :native_hsd,
  product_status = :ProductHSDBreakdown, last_step = :HSDStepBreakdown
- iterations = 24, factorizations = 25, backtracking = 15,
  **terminal_alpha = 0.0, step_size = 1.4258519510530428e-5**
- p_residual = 4.21e-6, d_residual = 4.65e-9, gap_residual = 2.52e-5,
  normalized_residual = 6.08e-6 (merit stalled ~6e-6, nowhere near 1e-8)
- tau = 6.15, kappa = 2.37e-6, mu = 1.43e-5
- fallback_chain = (), route :bordered, factor :cholmod_symmetric_ldl,
  metric :native_product_theta, no equilibration

## Localization direction (next steps)

The failure is a mid-convergence line-search breakdown at merit ~6e-6 with a
tiny last accepted step (1.4e-5) and exhausted backtracking — the same
symptom family as the Float64 Power failure that the factor-pair/whole-epoch
repair addresses. R0-E continues by: (1) reproducing the step-level failure
with the experimental loop machinery for the Exp cone (if the same
scaling/root defect family is confirmed, extend the factor-pair backend to
Exp); (2) localizing the first stage where the Exp conjugate/theta path
diverges from the (correct) SOC/power behavior; (3) precision ladder
(BigFloat256) through the same public path to confirm the breakdown
disappears with precision (as it did for Power).

Explicitly NOT claimed: this freeze does not establish that the failure is in
any particular kernel; it fixes the well-posed target, the external baseline,
and the observable breakdown quantities.
