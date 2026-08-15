# Adaptive Interior-Point Parameter Policy

This document records the implementation audit, controller design, safety
rules, and benchmark gate for SDPX's iteration-level parameter selection. The
guarded adaptive policy is the public default:

```julia
options = SolverOptions{Float64}(
    parameter_strategy=:adaptive,
)
```

Use `parameter_strategy=:fixed` only for reproducibility with the historical
trajectory. Automatic cold-start selection through `parameter_policy=:auto`
is a separate policy that runs one generic automatic Mehrotra controller after
scaling and before the iteration-level policy. The public resolver
`recommended_parameters` reports `profile=:generic_mehrotra`; the immutable
execution plan records the neutral, deferred identity
`:automatic_mehrotra`; and executed diagnostics record the post-scaling
resolution as `:post_scaling_mehrotra`. An explicit fixed policy is recorded
as `:user_fixed`. The adaptive controller itself returns fixed parameters
while the normalized cold-start merit is at least one, and it permanently
falls back after repeated instability.

## Audit of the existing Newton method

For each PSD block, SDPX solves

```text
A_l(x) - C_l = X_l >= 0
c - A*(Y) - B*y = 0
B'*x = b
X_l*Y_l = 0
```

At the current iterate it forms the primal matrix residual
`P_l=A_l(x)-C_l-X_l`, the equality residual `p=b-B'*x`, and the dual
residual `d=c-A*(Y)-B*y`. Linearizing complementarity gives

```text
dX_l = A_l(dx) + P_l
X_l*dY_l + dX_l*Y_l = R_l
```

Eliminating `dY_l` produces the Schur terms
`<A_i, X_l^-1*A_j*Y_l>`. SDPX factors the resulting Schur/equality
system once per outer iteration and reuses that factor for predictor and
corrector right-hand sides. Block directions are recovered after each solve.
This is the HRVW/KSH/M-type direction used by the original code, implemented
with symmetric-square or structure-specific Schur kernels.

The historical fixed SDP path is a predictor/corrector path, but it is not
canonical Mehrotra in every phase:

- `predictor=:classic` includes the current block target `mu_l*I` in the
  predictor.
- `predictor=:sdpb` switches to a pure affine target `-X_l*Y_l` only after
  primal and dual residuals are already below tolerance.
- the historical corrector includes `-dX_aff*dY_aff`, but chooses a
  block-local centering multiplier from a unit trial and the fixed `beta`.

The dedicated LP path was already a conventional Mehrotra predictor-corrector:
it computes affine primal and dual boundary steps, `mu_aff`, a second-order
corrector, and independent accepted primal and dual steps.

The adaptive SDP path now uses the same canonical sequence:

1. build a pure affine predictor;
2. compute exact or bisected PSD boundary steps;
3. compute global average complementarity
   `mu=sum_l <X_l,Y_l>/sum_l k_l`;
4. evaluate `mu_aff` at the independent affine boundary steps;
5. select bounded iteration parameters;
6. build
   `R_l=sigma*mu*I-X_l*Y_l-dX_aff_l*dY_aff_l`;
7. solve the corrector with the existing factor;
8. choose independent primal and dual feasible steps.

## Existing parameters and defaults

| Quantity | Default | Fixed behavior | Adaptive behavior |
|---|---:|---|---|
| `beta` (`β`) | `0.1` | Block complementarity reduction/centering target | Fallback value; `result.parameter_history.beta` aliases the selected `sigma` |
| `gamma` (`γ`) | `0.9` | Backtracking contraction; also the exact fraction-to-boundary safety | Fallback value; split into two fractions and one backtracking factor |
| `sigma` (`σ`) | no separate option | Equal to `beta` | Selected from `mu_aff/mu`, centrality, and recent progress |
| `mu` (`μ`) | derived | Per-block target is `beta*<X_l,Y_l>/k_l` | Global average complementarity is used by the canonical corrector |
| step rule | `:auto` | Exact fraction-to-boundary for blocks no larger than `2x2`; otherwise PSD backtracking | Same structural rule, with separately selected primal and dual safety values |
| refinement | `:auto`, cap `8` | Residual-driven unless a narrow exact-factorization skip is safe | Selector may reduce the cap on a clean factor and restores it on a difficult factor |
| tolerances | `1e-10` | Relative gap, primal residual, and dual residual | Unchanged |
| iteration limit | `200` | Hard outer-iteration limit | Unchanged |

`parameter_policy=:auto` runs the generic automatic Mehrotra controller before
the iteration-level policy. It selects no benchmark-, size-, cone-, or
precision-specific parameter profile: `beta`, `gamma`, `predictor`, and
`parameter_strategy` keep the `SolverOptions` defaults or user choices. After
presolve and scaling, LP, SDP, and NativeSOC use their existing planned KKT
route to solve one primal and one dual affine right-hand side, followed by
typed cone-interior shifts and deterministic complementarity mass balancing.
The automatic path does not use `OmegaP` or `OmegaD`. The public resolver reports
`profile=:generic_mehrotra`; the immutable execution plan records only the
deferred `:automatic_mehrotra` identity; and the executed solve records
`:post_scaling_mehrotra`. `parameter_policy=:fixed` uses the supplied values
exactly and records `:user_fixed`.

This document concerns iteration parameters. The storage and structure
classifications reported by `StructureAnalysis.profile` (for example
`:sparse_coefficients_dense_psd_dense_schur`) describe data layout and kernel
selection, not iteration parameters; they are unaffected by the generic
controller.

After initialization, NativeSOC retains its compact Lorentz-specific
`mu_aff / mu` centering rule and exact cone boundary kernel. The new affine KKT
cold start only replaces the old automatic identity heads; it does not replace
that iteration controller. FixedTraceQ3 uses the same Lorentz margin and
mass-balancing policy through its verified tail-map representation.

## Existing numerical safeguards

- Dense and reduced-arrow Schur factors first try an unregularized Cholesky.
  Failure triggers relative diagonal shifts beginning at `sqrt(eps(T))` and
  increasing by a factor of ten, with six bounded attempts.
- The LP KKT path starts at `max(sqrt(eps(T)), 1e-12)`, decays a previously
  successful shift by ten, and escalates a failing factor by ten for at most
  eight attempts.
- Equality Gram factors use rank-revealing pivoted Cholesky when the ordinary
  factor is numerically unacceptable.
- Iterative refinement uses a target-type residual and stops on achieved
  residual reduction. Mixed-precision KKT paths have separate condition,
  correction-budget, monotonicity, and native-fallback guards.
- Termination uses normalized primal residual, normalized dual residual, and
  relative gap, followed by certification in the original coordinates.
  `AlmostOptimal` distinguishes a relaxed certificate, and
  `InsufficientPrecision` distinguishes an arithmetic floor from algorithmic
  failure.
- A moving-window stagnation detector monitors tolerance-normalized residual,
  gap, and complementarity progress.
- BigFloat `working_precision_policy=:auto` is an outer staged solve: it may
  retry at the requested precision after failed certification. It does not
  mutate MPFR precision in the middle of a Newton iteration.

## Policy component

The implementation is isolated in `src/adaptive_parameters.jl`:

```julia
abstract type AbstractParameterPolicy end

struct FixedParameterPolicy{T} <: AbstractParameterPolicy
    # Exact historical values
end

struct AdaptiveParameterPolicy{T} <: AbstractParameterPolicy
    # Bounds, fallback, and controller settings
end

select_parameters(policy, diagnostics, history)::IterationParameters
```

`select_parameters` is pure and does not mutate the solver. The mutable
`AdaptiveIPMController` only owns history and the one-way fallback state.
`IterationDiagnostics` records:

- normalized primal and dual residuals;
- relative gap, `mu`, and `mu_aff`;
- affine and previous accepted primal/dual steps;
- backtracking and refinement counts;
- relative regularization and a Cholesky-diagonal quality proxy;
- predicted and observed feasibility reduction;
- primal and dual block Cholesky-margin proxies;
- whether the current metrics are at the arithmetic precision floor.

`IterationParameters` returns `sigma`, independent primal and dual
fraction-to-boundary values, the backtracking factor, regularization hints,
the refinement target/cap, the centrality target, and fallback provenance.
Every accepted iteration is copied into `result.parameter_history`.

## Conservative selection rules

All constants are constructed in the solver arithmetic; BigFloat and
MultiFloat diagnostics are never narrowed through Float64.

- `sigma` is based on `(mu_aff/mu)^p`, with `p=3` after a good affine step and
  `p=2` after a short step.
- The candidate is blended as 80% current estimate and 20% previous `sigma`.
- `sigma` is normally bounded by `[0.02, 0.50]`, expanded only as necessary to
  include the configured fixed fallback. The generic automatic controller
  uses the 0.50 upper bound; the expert `adaptive_sigma_max` option can
  reproduce or override this selection.
- Short affine steps, residual growth, frequent backtracking, or poor factors
  impose centrality floors of `0.08` or `0.20`.
- The primal and dual fraction-to-boundary values are selected independently
  in `[0.80, 0.99]`, again expanded only to retain the fallback. Small recent
  steps and repeated backtracking make them more conservative.
- The backtracking contraction is selected in `[0.55, 0.90]`, with the
  configured fallback always representable.
- Clean factors can use at most two refinement passes. A poor factor or a
  previous correction restores the configured cap.
- Non-finite diagnostics, an explicit rank-revealing equality factorization,
  or repeated unstable outcomes switch permanently to the complete fixed path
  for the remainder of the solve. Severe complementarity or feasibility
  growth triggers that recovery without waiting for a second event.

The fallback restores the fixed predictor construction as well as fixed
`beta`, `gamma`, corrector, step, and refinement behavior. This distinction is
important: changing only the scalar values while leaving the affine predictor
enabled is not a true rollback.

Regularization and precision decisions remain deliberately layered. The
policy observes and records their state, while the factorization kernels keep
their failure-driven regularization and the outer BigFloat driver keeps its
certification-driven precision retry. This avoids refactoring a current
factor from parameters selected only after its affine predictor has been
computed.

## Arithmetic behavior

- Float64 uses BLAS-backed factors and the same typed controller calculations.
- MultiFloats, including Float64x4, use exact-type ratios and bounds. Existing
  owned blocked kernels and mixed-precision fallbacks are unchanged.
- BigFloat selection executes at the active MPFR precision. Native BigFloat
  remains ownership-safe; policy selection does not create shared mutable
  matrix entries or add threaded MPFR writes.

The formulas are the same across arithmetic types. Only `eps(T)`, factor
quality, refinement thresholds, precision-floor detection, and existing
backend selection differ.

## Validation policy

Automatic-controller changes are accepted only after the same implementation
passes LP, SDP, NativeSOC, dense/sparse/equality, and arithmetic-type
regression gates. Every gate compares status, iterations, residuals, relative
gap, original-coordinate certification, executed provider/formulation
provenance, and parameter history. Performance claims require repeated
same-node runs; an improvement on one benchmark never creates a new automatic
branch.

Historical benchmark-specific sweeps remain in dated benchmark reports rather
than this maintained policy document. They can be reproduced with explicit
`beta`, `gamma`, `OmegaP`, `OmegaD`, `predictor`, `scaling`, and
`adaptive_sigma_max` options, but they do not influence automatic selection.

## Remaining limitations

- The SDP line search certifies cone interiority. It does not recompute the
  complete residual or merit at every trial because that would repeat the
  dominant contractions; observed reduction is checked at the next Newton
  residual evaluation.
- PSD and factorization margins are inexpensive Cholesky-diagonal proxies, not
  per-iteration minimum eigenvalue computations. Exact eigenvalue checks
  remain part of final validation.
- The returned SDP regularization fields are controller hints and diagnostics;
  current-factor regularization remains failure-driven inside the KKT backend.
- BigFloat precision escalation is between complete solve attempts rather than
  within one factorization.

## Public algorithm references

The design uses published behavior and public documentation, not undocumented
commercial heuristics:

- S. Mehrotra, “On the implementation of a primal-dual interior point
  method,” *SIAM Journal on Optimization* 2(4), 1992.
- R. D. C. Monteiro and Y. Zhang, “A unified analysis for a class of
  path-following primal-dual interior-point algorithms for semidefinite
  programming,” *Mathematical Programming* 81, 1998.
- K. C. Toh, M. J. Todd, and R. H. Tütüncü,
  [SDPT3 implementation paper](https://people.orie.cornell.edu/miketodd/cacsd.pdf).
- [SDPA documentation and source](https://sdpa.sourceforge.net/download.html).
- [SeDuMi reference guide](https://sedumi.ie.lehigh.edu/sedumi/files/sedumi-downloads/SeDuMi_Guide_11.pdf).
- [SDPB source and papers](https://github.com/davidsd/sdpb).
- [Clarabel solver settings](https://clarabel.org/stable/api_settings/) and
  [implementation paper](https://arxiv.org/abs/2405.12762).
- [MOSEK conic termination criteria](https://docs.mosek.com/latest/toolbox/solving-conic.html)
  and [public parameter reference](https://docs.mosek.com/latest/pythonapi/parameters.html).
