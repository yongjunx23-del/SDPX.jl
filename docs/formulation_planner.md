# Static formulation planner

`formulation=:auto` now makes a real pre-execution choice between the two
implemented general dense mathematical formulations:

- `DenseNormalEquations`, followed by Cholesky;
- `DenseAugmentedKKT`, followed by pivoted symmetric LDLT.

The planner is deliberately small, conservative, and deterministic. It does
not add a third formulation and never changes formulation after numerical
execution begins.

## Policy boundary

The formulation planner consumes mathematical input facts and equality-basis
evidence. The backend planner separately checks whether the selected
arithmetic has the required Cholesky or LDLT capabilities. Precision policy
remains separate from both. There are no MFLA-, BFLA-, or arithmetic-name
branches in the formulation heuristic.

The first policy uses only two numerical-risk indicators:

1. the scale spread of retained equality rows; and
2. the relative diagonal quality from the normalized RRQR that equality
   presolve already performs for correctness.

It does not compute a condition number, SVD, eigendecomposition, or a second
RRQR. If a verified retained equality basis is unavailable, auto planning
conservatively keeps dense normal equations.

The rounded initial thresholds are:

```text
equality row scale spread >= 1e8
or
retained normalized RRQR diagonal quality <= 1e-8
```

Either condition marks strong normal-equation risk. Augmented KKT is selected
only when that risk is present and the candidate passes structural, backend,
and memory feasibility. Otherwise dense normal equations remain the default.
Round 3 did not provide a credible size/timing crossover, so the first policy
does not invent one; it records both system dimensions and memory estimates
and uses memory only as a hard feasibility gate.

## Explicit requests

- `formulation=:normal_equations` fixes dense normal equations and disables
  the native-Q3 shortcut; it fails closed for dedicated LP, whose Newton
  system is not an SDP normal-equations formulation.
- `formulation=:augmented` fixes dense augmented KKT or fails during planning
  when LDLT, equality-solver, structure, or memory requirements are unmet.
- historical `formulation=:primal` preserves primal orientation and does not
  disable sparse or block-arrow structural routes.
- `formulation=:dual` remains unsupported and fails before backend planning.

An explicit request is never overridden. Auto candidate filtering is not a
runtime fallback: when augmented is mathematically preferred but infeasible,
the decision records both the preference and why normal equations were
selected. Cholesky or LDLT failure never triggers the other formulation.

## Inspection and diagnostics

`SDPX.Experimental.plan_formulation` is a pure inspection API returning a
`FormulationDecision`. The execution plan and result diagnostics expose its
summary through `parameters.formulation_decision` and
`selected_algorithms.formulation_decision`. It records:

- requested, preferred, and selected formulation;
- a stable Symbol reason;
- both candidates and required capabilities;
- structural/backend/memory feasibility;
- equality scale, RRQR evidence, dimensions, and memory estimates;
- the exact risk indicators and thresholds used.

Typical reasons include `:default_dense_normal_equations`,
`:poor_equality_quality`, `:large_equality_scale_spread`,
`:augmented_backend_capability_unavailable`,
`:augmented_memory_unavailable`, `:user_forced_normal`, and
`:user_forced_augmented`.

## Current limitations

The heuristic is calibrated from a small Mac-local Round 3/4 stress set, not
a broad public corpus. It does not predict Schur conditioning unrelated to the
equality basis, model runtime, adapt across iterations, or choose sparse,
block-arrow, LP, Q3, or future native-SOC formulations. Those routes retain
their existing structural planners. Future tuning should add public evidence
before adding rules; benchmark identities or provider names must never enter
the production planner.
