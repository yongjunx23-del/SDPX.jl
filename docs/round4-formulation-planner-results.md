# Round 4: automatic formulation planner

This report records the first static, correctness-first choice between dense
normal equations (NE) and dense equality-augmented KKT. All measurements are
Mac-local; no cluster or Heavy suite was used.

## 1. Round 3 evidence summary

NE was clearly preferable for Schur-coordinate scale `1e8` (13 versus 20
iterations) and remains the mature, smaller default on ordinary dense and
well-conditioned equality systems. Augmented KKT had its clearest numerical
advantage for equality-row scale `1e8`: the recorded raw primal residual was
about `3.4e-47` instead of `1.5e-2`, while both original-coordinate
certificates were valid. On the near-dependent equality ladder, augmented
preserved smaller raw residuals from roughly `1e8` conditioning onward, but
both routes certified and had essentially the same iterations and warm time.
The ordinary control, equality-heavy control, and most ladder cases were
therefore `both_acceptable`. There was no credible matrix-size timing
crossover because all Round 3 systems were tiny.

The two cheap facts correlated with equality-induced NE risk were retained
equality row-scale spread and the normalized equality RRQR diagonal quality.
Post-factor Cholesky quality/failure, LDLT pivots/inertia/growth, backward
error, refinement count, and time remain valuable diagnostics but cannot
participate in the first static choice.

## 2. Features considered

Dimensions, equality count/density, equality aspect ratio, row and column
scale proxies, verified equality rank/quality, cone/block structure, and
normal/augmented dimensions and memory were reviewed. Existing presolve RRQR
evidence was preferred over any new numerical analysis.

## 3. Features rejected

Global coefficient spread, Schur-coordinate spread, equality count or aspect
ratio alone, cone counts, arithmetic type, provider identity, problem name,
runtime timing, factor diagnostics, and exact condition estimation were not
predictive enough or violated the planning boundary. No `cond`, SVD, spectrum
estimate, second RRQR, lookup table, or benchmark-aware rule was added.

## 4. Final ProblemFeatures additions

`DenseFormulationFeatures` records variables, retained equalities, equality
density, equality row-scale spread, NE dimension, augmented dimension, and
augmented square ratio. `EqualityPlanningEvidence` separately records whether
the already-required presolve basis was verified, its before/after rank, and
its relative RRQR quality. No provider, fallback, precision, or selected
formulation is stored in `ProblemFeatures`.

## 5. Final planner algorithm

The pure planner creates NE and augmented candidates, computes two risk flags,
then freezes one feasible formulation before provider instantiation. NE is the
default. Augmented is preferred only for verified equality evidence with a
large equality scale spread or poor RRQR quality. Backend and memory filters
may select the next feasible candidate and preserve the mathematical
preference in diagnostics. Execution never replans.

## 6. Final thresholds and rules

```text
strong risk = verified retained equality basis
              and (row scale spread >= 1e8
                   or relative RRQR quality <= 1e-8)

auto = augmented when strong risk and augmented is feasible
       otherwise normal equations when feasible
```

## 7. Threshold evidence

Both thresholds are rounded order-of-magnitude boundaries from the Round 3
controlled ladder: the equality-scale `1e8` case showed the only strong route
quality separation, and normalized RRQR quality crossed approximately `1e-8`
on the `1e8` near-dependence case. `1e2` through `1e6` remained with NE. No
size threshold was invented; size is recorded and memory is a hard gate.

## 8. FormulationDecision structure

`FormulationDecision` records requested, mathematical preference, final
selection, Symbol reason, both `FormulationCandidate`s, supporting features,
equality evidence, and exact risk indicators/thresholds. Each candidate
records structure/backend/memory feasibility, required capabilities, system
dimension, memory estimate, and reason.

## 9. Auto planning flow

The production order is: problem dimensions/equalities → cheap dense
formulation features → equality presolve and its existing RRQR evidence → pure formulation candidates
and preference → backend/memory feasibility → frozen typed `FormulationPlan` →
LA provider plan → Workspace and numerical execution.

## 10. Explicit planning behavior

`:normal_equations` fixes dense NE and fails on dedicated LP. `:augmented`
fixes dense augmented KKT or fails during planning. Historical `:primal` preserves sparse and
block-arrow structural decisions. `:dual` still fails closed. Explicit
requests are never overridden.

## 11. Backend feasibility

Candidates name semantic capabilities, not providers. Augmented requires
pivoted symmetric LDLT, factor solve, and multiple RHS. If those are missing,
auto records `preferred=:dense_augmented_kkt` but selects feasible NE with an
explicit capability reason; explicit augmented fails. MFLA/BFLA names do not
appear in the heuristic.

## 12. Memory feasibility

The planner compares dimension-only conservative workspace estimates against
available memory. They assume dense block activity, apply the existing 1.5
safety margin, and avoid a full coefficient walk. The augmented estimate
includes the ordinary dense Workspace plus the two
`(m+n)^2` augmented matrices and three augmented vectors that Round 3 added.
An infeasible augmented candidate is rejected before allocation; there is no
OOM-driven formulation retry.

## 13. Diagnostics and reasons

ExecutionPlan parameters and result diagnostics expose the decision summary.
Stable reasons include `:default_dense_normal_equations`,
`:poor_equality_quality`, `:large_equality_scale_spread`,
`:equality_quality_unavailable`,
`:augmented_backend_capability_unavailable`,
`:augmented_memory_unavailable`, `:user_forced_normal`, and
`:user_forced_augmented`. Planned and executed formulation remain separate.

## 14. Planner Micro results

The Round 2 Micro suite completed 8/8 Optimal with no semantic failure. Pure
feature/decision tests cover easy, threshold, evidence-unavailable,
backend-unavailable, memory-unavailable, and both explicit requests. The final
Quick profile completed 1,121 passes with one pre-existing broken marker in
about 49 seconds.

## 15. Representative scoreboard

The final Float64x3/MFLA scoreboard ran 14 cases × three policies. All 42
solves were Optimal, original-coordinate certificate-valid,
reference-objective-valid, planned/executed-consistent, finite, and
fallback-free.

| Classification | Cases | Auto behavior |
|---|---:|---|
| NE preferred | 2 | NE on equality-scale `1e4` and Schur-scale `1e8` |
| Augmented preferred | 1 | Augmented on equality-scale `1e8` |
| Both acceptable | 11 | conservative NE except three `>=1e8` RRQR-risk cases |

Auto selected NE on 10 cases and augmented on 4. All 14 selections were
acceptable. Timing was recorded for observation but deliberately excluded from
single-sample winner classification.

## 16. False positives

There were zero cases where auto chose augmented while the valid comparison
clearly preferred NE. In particular, the Schur-scale `1e8` control stayed NE.

## 17. False negatives

There were zero severe false negatives: auto never produced an invalid result
when either explicit normal equations or explicit augmented KKT was valid. The
equality-scale `1e8` stress switched to augmented as intended.

## 18. Ambiguous cases

Eleven cases remained `both_acceptable`, including ordinary/equality-heavy
controls and most near-dependent ladder points. The planner intentionally does
not manufacture a performance winner for them.

## 19. Local Full results

Representative completed 18 runnable rows with 10 structured skips and zero
errors in the root environment; the provider-enabled environment completed 20
runnable rows with 8 skips and zero errors. Local Full auto-only completed 41
runnable rows with 44 documented external/provider skips and zero errors. All
runnable rows passed their registry semantics; expected stress statuses
(`NumericalBreakdown`/`Stalled`) remained expected rather than regressions.
An additional BFLA256 equality-scale `1e8` solve selected and executed
augmented KKT, was Optimal and certificate-valid, and used no fallback.

## 20. Remaining limitations

The evidence set is small and mostly synthetic; public external loaders remain
deferred. The heuristic does not recognize Schur conditioning unrelated to
equalities, predict runtime, use precision-dependent risk, or make adaptive
iteration-time changes. Sparse, block-arrow, LP, Q3, and native SOC candidates
remain outside this dense two-candidate policy.

## 21. Recommended Round 5 preparation

Keep the candidate protocol and add Native SOC only after its representation,
structural applicability, capabilities, execution, original-coordinate
certificate, and Mac benchmark evidence exist. Do not reuse the equality-risk
thresholds as SOC heuristics and do not change the PSD-lift route implicitly.

## Acceptance answers A–G

**A.** Yes. `formulation=:auto` now uses `ProblemFeatures` plus verified
presolve equality evidence to choose a mathematical formulation.

**B.** Yes. `FormulationPlanner`, `BackendPlanner`, and `PrecisionPolicy`
remain separate and execute in that order.

**C.** No. There is no MFLA/BFLA-specific formulation heuristic.

**D.** No. The planner adds only linear scans and reuses the existing equality
RRQR; it performs no expensive condition analysis or duplicate factorization.

**E.** The final 14-case scoreboard has 0/14 false positives and 0/14 severe
false negatives under its correctness-first classification.

**F.** The 11 `both_acceptable` cases, unobserved public instances, and risk
arising outside the equality basis remain intentionally unresolved.

**G.** Architecturally yes: the candidate/feasibility/decision seam can accept
future Native SOC. Numerically no Native SOC claim is made in this round; it
still needs its own implementation and evidence.
