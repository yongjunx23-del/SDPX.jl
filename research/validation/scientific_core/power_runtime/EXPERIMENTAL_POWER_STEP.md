# Experimental factor-pair half-Power step context (R0-P)

Opt-in research loop only. No production route/default change; no root-tolerance
widening, extra fallback, precision promotion, or rank/analytic-root tricks.

## Result (Float64, exact canonical Power problem from factor_affine_trial_17)

Cold start -> certified terminal in 27 accepted steps (source-faithful
boundary; the internal R0-P4 adapter stabilizes that boundary and reaches a
certified terminal in 23 steps with the same unchanged acceptance gates). Acceptance uses the
UNCHANGED progress gate: early/mid steps take genuine steps (alpha >= 0.026,
far above the floor 2cbrt(eps) ~ 1.21e-5); late steps inside the arithmetic
merit neighborhood may be accepted below the floor with real represented
progress (observed alphas 4.9e-6, 3.1e-7, 1.6e-1 at iterations 24-27), exactly
as the source progress predicate permits (linesearch.jl). Each trial carries a
FRESH full-gap polynomial root,
stored-shadow reconstruction, compensated half-Power factor, true-geometry/
decrement/BFGS certificates, and atomic commit of (point, factor pair, warm
lineage, epoch). Accepted pairs are certified with their retained warm probe;
in this run 25/27 accepted pairs are ALSO certifiable from a true cold rebuild
(warm=nothing, fresh owner), recorded as an observed diagnostic, not a
guarantee. Homotopy/merit/progress
gates replicated unchanged from src/hsd. Predictor centering follows the
reviewed source policy including the DUAL boundary (primal and dual cone
bounds + tau/kappa), not primal only. Terminal original-coordinate audit is
computed independently (fresh dense residual evaluation from recovered
coordinates, not the loop residual routine or cached vectors):
- merit (homogeneous residuals) 8.39e-9 <= 1e-8 target
- recovered primal residual 7.00e-10, dual residual 7.00e-10
- complementarity s'y/tau^2 = 6.71e-9 ; cone membership true (cert_tol = 1e-6 predicates)
- recovered homogeneous gap c'x*+b'y*+kappa/tau = 2.80e-9, kappa/tau = 5.58e-10
- THE TERMINAL POINT PASSES THE ORDINARY CERTIFICATE INEQUALITIES of
  src/certificates/certificates.jl:438-502 at the source-default Float64
  tolerance 1e-6, evaluated on RECOVERED coordinates with no tau redivision:
  normalized homogeneous residual m/(||A||+||b||+||c||+1)
  = 7.00e-10 <= 1e-6 (m = 2.80e-9, data_norm = 4); recovered primal
  = 7.00e-10 <= 1e-6*primal_scale and dual = 7.00e-10 <= 1e-6*dual_scale;
  st in K and yt in K* via the source power_membership /
  power_dual_membership predicates at tolerance 1e-6 (= cert_tol);
  recovered gap |c'xt + b'yt| = 2.24e-9 = |rG - kappa/tau|
  = |2.80e-9 - 5.58e-10| <= 1e-6*gap_scale; cone complementarity
  |st'yt| = 6.7e-9 <= gap_lim; kappa/tau = 5.58e-10 <= gap_lim; invariant
  mu/tau^2 = 5.59e-10 <= 1e-6*(1+nu); tau > 1e-6; all scalars finite. A homogeneous-rescaling regression (rho=2) verifies the
  acceptance quantities are scaling-invariant. Independently re-evaluated
  from stored recovered coordinates by a separate dense implementation.
- experimental merit target 1e-8: merit = 2.80e-9 <= 1e-8
- objective 1.1242390972345995 vs exact 1.1242390986454483421 -> err 1.41e-9

Precision-ladder context (identical public path): Float64x4/MFLA solves the
same canonical problem optimal+cert_valid (obj_err 5.6e-23, 75 iters),
BigFloat256/512 1.48e-26/3.74e-54, MOSEK 3.9e-17, Clarabel-BF256 7.3e-31; the
Float64 production defect is specific to Float64 near-boundary interior
scaling/root construction (see local-archives/.../power-precision-ladder/).

227 driver assertions pass at the frozen worktree HEAD; per-step next-epoch
five-equation certification (2^-17), NC affine certification, ordinary terminal
certificate inequalities (complementarity, recovered gap, kappa/tau, normalized
mu, finite positive scalars), same-owner previous-generation stale-token
refusal, and rejected-trial rollback (anchor/generation unchanged) are
asserted.

## How it works

- Predictor centering identical to src/hsd policy: alpha_aff=0.995*boundary
  (min of primal cone bound, dual cone bound, tau, kappa),
  mu_aff=(dot(s+?ds,y+?dy)+(tau+?dtau)(kappa+?dkappa))/(nu+1),
  sigma=min(1,(mu_aff/mu)^3), sigma_mu=sigma*mu.
- Combined RHS/corrector/solve reused from FactorCombinedEpoch; affine solve and
  raw recovery from FactorPreservingAffine; pair/trial construction from
  NativeHalfPair (anchored lineage, generation, warm tokens).
- Trial acceptance: fresh certified pair + strict interior + unchanged
  componentwise residual homotopy + merit envelope + useful-progress gate.
  Rejection backtracks alpha*=0.5 (<=64) with the anchor untouched; acceptance
  advances generation and issues tokens only after ALL gates pass.

## Known boundaries

- Strict double-secant H(s) scaling is NOT qualified by this experiment (the
  explicit dual-Hessian one-secant factor is the declared policy).
- Dense Theta=S*S' materialization at Float64 is numerically hopeless for these
  near-boundary blocks (G*Theta-I~7, Cholesky failure, cond~1e14) - the factor
  pair must be carried through the Newton epoch, as done here.
- BigFloat256/512 and x4 arms, the real power_epigraph_small benchmark, and
  production default integration remain future acceptance levels.
