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

## Localization (scout, HEAD 21484f3)

Bounded instrumented probe (isolated worktree, `SDPX_DEBUG_LINE_SEARCH=1`,
no src edits) reproduced the breakdown exactly (24 iters, merit 6.08e-6).
First-failing stage: **conjugate scaling construction**
(`try_update_nonsymmetric_scaling!` -> `try_update_scaling!`, the neighborhood
gate of the line search), in two phases:

1. First rejection (iter ~12, accepted merit 7.7e-4): trials at alpha 0.575/0.288
   fail `NS_SCALING_SHADOW_IDENTITY_FAILED` with `NS_CONJUGATE_CONVERGED` at
   exp-block offset 7 (trial_mu=1.127e-4, tau_t=3.712, kappa_t=3.256e-5) - the
   double-secant Gram identity fails while the Fenchel shadow solves.
2. Terminal (~20 rejected trials, offsets 4/10, offset 7 once): 
   `NS_SCALING_CONJUGATE_FAILED / NS_CONJUGATE_BARRIER_FAILED` inside the
   Exp-specific `exp_logarithmic_conjugate!` rho-equation kernel
   (conjugate3.jl:1455-1464), tau_t in [6.35,7.17], kappa_t in [1.27e-6,2.18e-6],
   trial_mu in [8.9e-6,1.36e-5]. Terminal micro-trial alpha=5.06e-20 passes
   neighborhood/homotopy/merit but fails progress -> :line_search_breakdown.

Every rejected trial is strictly primal/dual interior (`strict=true`) - a
near-boundary INTERIOR scaling failure at large tau (~6-7) and tiny kappa
(~1e-6), i.e. deep in the HSD central-path neighborhood where the exp shadow
coordinates blow up like the reciprocal gap.

**Ruled out (all pass):** theta/product metric (never reached), predictor
direction and KKT factorization (25 factorizations, fallback_chain=(), factor
cholmod_symmetric_ldl, regularization 4.07e-8 healthy), corrector (no
NS_CORRECTOR_* failure), line-search homotopy and merit gates.

**Power-family comparison:** same stage family (nonsymmetric conjugate/scaling
construction failing at near-boundary interior points in Float64, mid-merit
stall with exhausted backtracking) but Exp-specific manifestation: upstream of
any metric, first a Gram shadow-identity mismatch with converged conjugate,
then outright barrier failure in the Exp-only rho-equation path. The Power
factor-pair repair does not transfer verbatim; the Exp rho-equation residual
and its `psi = z - x`-scale roundoff need their own analysis (the analytic Exp
Cholesky `L11 = 1/psi` terms in `_ns_structural_hessian_factor!` are the
known-unstable-at-small-psi operations).

**Next steps (bounded):** (1) capture the failing per-block (s,y) triples at
offsets 4/7/10 via an in-memory observer (no src edits); (2) precision ladder -
BigFloat256 through the same public path; if the breakdown disappears (as for
Power), the defect is Float64 rounding in the Exp rho-equation/shadow-identity
and the repair follows the factor-pair pattern adapted to the Exp psi-scale
(compensated evaluation / re-derived L11=1/psi terms), NOT tolerance widening
or an extra fallback.

## Precision ladder (parent verification, HEAD 26ab94a)

exp_entropy_small through the identical public path at BigFloat (ambient
precision, ~256 bits; BFLA/MFLA extensions loaded):
- **status=optimal, certificate_valid=true,
  obj = -1.098612288668109691395...** (matches -log(3) =
  -1.098612288668109691395... to 5e-17), iterations=47.
- The Float64 breakdown (24 iter, merit 6e-6) **disappears with precision**,
  confirming the defect is Float64 rounding in the Exp conjugate
  rho-equation / shadow-identity path (the same precision-ladder closure the
  Power case exhibited).
- `expectation_met=false` at BigFloat is a harness artifact of the
  legacy `:known_solver_finding` contract (`validate_result` requires
  `!certificate_valid` for those cases): the BigFloat solve now CERTIFIES,
  which is strictly stronger than the finding the harness expects.

R0-E stage conclusion: the Exp Float64 defect is a near-boundary interior
nonsymmetric conjugate/scaling construction failure caused by Float64
rounding in the Exp rho-equation residual and `psi = z - x`-scale shadow
operations. The repair route (bounded next step) is the factor-pair pattern
adapted to the Exp psi-scale with compensated evaluation of the analytic
Exp structural Cholesky (`L11 = 1/psi` terms), NOT tolerance widening or an
extra fallback; production dispatch unchanged.
