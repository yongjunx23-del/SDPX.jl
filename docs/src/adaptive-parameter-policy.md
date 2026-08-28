# Adaptive predictor-corrector policy

Adaptive parameters operate inside the single product-cone HSD state machine.
They do not select LP, SOC, SDP, or other family-specific solvers.

## Policy modes

- `parameter_policy=:auto` builds a typed cold start after presolve and scaling,
  then uses the shared bounded predictor-corrector controller.
- `parameter_policy=:fixed` preserves explicitly supplied expert values.

The resolved policy is recorded in diagnostics. Unsupported or nonfinite
settings fail before they can affect a terminal certificate.

## Cold start

The automatic start:

1. solves the planned affine initialization system;
2. applies cone-native strict-interior shifts;
3. raises aggregate identity mass only when required by the typed cone-vertex
   envelope;
4. cross-centers primal and dual complementarity; and
5. verifies affine residuals and cone margins.

The initialization factor belongs to its operator epoch and may be reused only
under the same factor-receipt rules as later Newton solves.

## Predictor and corrector

At each accepted iterate, the controller:

1. solves the affine predictor;
2. computes the affine complementarity estimate;
3. chooses a bounded centering value from complementarity ratio, centrality,
   residual progress, and factor-quality evidence;
4. builds the Mehrotra corrector including second-order terms;
5. selects primal/dual boundary safeguards; and
6. passes the combined direction to the unified neighborhood/progress line
   search.

All cone blocks contribute to one product complementarity observable. A local
Q3, PSD panel, exponential, or power specialization may accelerate block
operations but cannot own a separate controller.

## Fallback policy

Nonfinite diagnostics, unacceptable exact-operator residuals, factor failure,
or loss of progress trigger a typed failure or an authorized same-iterate KKT
route fallback. The controller does not retry through a legacy engine, a PSD
lift, or lower precision.

A fixed safe parameter set may be used as an iteration-policy fallback only
when explicitly allowed by the plan; this changes controller values, not the
Newton equations or certificate standard.

## Recorded evidence

When iteration history is retained, diagnostics may include:

- `mu` and affine `mu`;
- centering parameter;
- primal and dual step limits;
- neighborhood/progress decisions;
- regularization and refinement;
- operator/factor generations;
- attempted KKT routes; and
- accepted residual progress.

These fields explain trajectory decisions. They do not establish optimality or
infeasibility.

## Arithmetic and determinism

Parameter calculations use the active arithmetic. MultiFloat and BigFloat
values are not narrowed to Float64 to choose a trajectory. Parallel reductions
use the deterministic reduction policy selected by `ThreadBudget`.

## Terminal authority

The controller may recommend continuing, stopping for exhaustion, or presenting
a terminal candidate. Only the reconstructed original-coordinate verifier may
promote the candidate to `:optimal`, `:primal_infeasible`, or
`:dual_infeasible`.
