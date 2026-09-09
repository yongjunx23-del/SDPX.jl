# PLAN_NEXT_STEPS — solver engineering milestones

**Baseline:** clean `2052094f71c19f65b416fea0bfd6d10603f1777f`.

**Requested repository destination:** `docs/design/PLAN_NEXT_STEPS.md`  
**Delivered artifact:** `/Users/xuyongjun/.pi/agent/sessions/--Users-xuyongjun--/subagent-artifacts/outputs/a3b06191-1f76-4dec-90c3-59ebe6ad8912/reports/pool-review.md`

The worktree remains unchanged; this plan is delivered through the authoritative report destination.

## Summary

Move effort from scheduler optimization to default numerical reliability, then qualify precision/provider boundaries and resource ownership before expanding sparse or structured execution. Start with permanent default-Float64 Power/Exp counterexamples and independently checked whole-epoch repair candidates—not an immediate default-policy switch. In parallel, build the cross-version provider matrix and R1-D adversarial certificate tests. Establish a scoped retained-live memory bound before enabling any sparse memory admission. Large-PSD/chordal and N14/SDPB work should begin with reference harnesses and explicit reconstruction contracts, not speculative production dispatch.

## 1. Baseline decisions and evidence corrections

Treat the focused parallel campaign as complete **according to the published ledger**:

- Job 211771: 12/12 completed receipts; approximately **1.539× pool-1→8 speedup**.
- New scheduler versus baseline: performance-neutral at the measured sizes. Do not attribute the existing parallel speedup to it.
- Persistent pool: **3.0471× local finite-batch throughput gate**, not a long-run retention or production-scheduler guarantee.
- Residual fusion and chunk-64 remain rejected. Do not reopen LP `random_large` inner-thread redesign.

Before recording new completion claims, reconcile these stale descriptions:

1. Roadmap and parallel acceptance sections still describe job 211771 as pending, although §5.3 records completion.
2. R2-C’s older prose says sequential-only, but `test/test_r2_full_qualification.jl:180–222` now contains spawned-task tests.
3. `factor_pair_admission.jl` has an obsolete “not implemented” header; its actual admission function returns an admitted experimental route.
4. The roadmap’s historical version-closure tuple is not current HEAD. Record actual source/provider identities per campaign rather than silently substituting old pins.

**Effort:** ½–1 engineer-day. Documentation only; do not change qualification boundaries.

## 2. Ranked delivery sequence

Estimates are focused engineer-days, excluding external solver access, cluster queue time, and mathematical review. Research-dependent completion is explicitly separated from buildable deliverables.

| Rank | First buildable delivery | Estimated effort | Dependency |
|---|---|---:|---|
| 1 | Default Power/Exp failure suite and whole-epoch repair gates | 3–5 days; candidate work below | None |
| 2 | R1 provider/version matrix and extreme-certificate suite | 7–12 days | None for harness |
| 3 | R2-D owned-capacity inventory and bounded retention campaign | 8–12 days | R1 ABI facts for native storage |
| 4 | R3 bounded scalar-LP sparse qualification/admission ledger | 8–15 days | R1-C and relevant R2 resource contracts |
| 5 | Large-PSD reference ladder, then chordal reconstruction prototype | 15–25 days | Reliable provider/ownership tests |
| 6 | N14 BF512 source-correct driver and SDPB comparator | 8–13 days plus runs | Source acquisition; R1 BF512 qualification |

Ranks 2–3 can advance while R0 mathematics is reviewed. Later harness construction need not wait for every earlier milestone, but production admission must respect dependencies.

---

## 3. Rank 1 — close default Float64 Power/Exp failures

### What exists

- Default public failures are localized to nonsymmetric conjugate/scaling construction.
- Half-Power has an explicit Float64 orthant-plus-half-Power factor-pair backend, accepted-step machinery, public qualification tests, and typed refusals.
- Exp has frozen failures, compensated evaluator/conjugate research, and independent replay tests.
- These are **not** blanket default Power/Exp qualification. Half-Power does not cover arbitrary exponents or mixed cones.

### Exact gap

A research or opt-in success does not establish that the **default public route**, at the requested tolerance and original coordinates, succeeds without unsupported representation crossings or precision changes.

### Implementation steps

**R0.1 — make default failure closure executable.**

Touch:

- `benchmark/general/power.jl`, `benchmark/general/exp.jl`: use existing fixtures and references.
- New `test/r0_default_failure_closure.jl`.
- `validation/scientific_core/cross_solver/`.
- Existing `factor_pair_public_qualification.jl`, Exp replay tests.

Create a case manifest recording source, cone composition/exponent, initial seed, expected mathematical outcome, current default outcome, requested tolerances, and first failing predicate. Include entropy/logsumexp and the documented Power failures. Keep current failures explicitly marked until repaired; never turn them into passing Unknown cases.

**R0.2 — complete half-Power candidate qualification before proposing default promotion.**

Touch only where failures identify a gap:

- `src/hsd/factor_pair/factor_pair_hsd.jl`: `cold_start`, `step!`, `solve!`, `cert_quantities`.
- `native_half_pair.jl`, `factor_combined_epoch.jl`, `native_factor_affine_certificate.jl`.
- `src/hsd/factor_pair_admission.jl`: `factor_pair_admission`.
- `src/hsd/native_hsd_public.jl`: existing factor-pair execution/reconstruction fork.

Build cold-seed and boundary tests around actual stored points, pair construction, affine/combined directions, trial refusal, rollback, and accepted-state publication. Verify all five Newton equations independently and terminal coordinates through the original problem.

Do **not** switch defaults in this step. First produce a reviewed capability table: exact half-Power scope, unsupported exponents/products, representation, precision, and memory conditions.

**R0.3 — build an Exp whole-epoch research adapter.**

First implementation stays under `validation/scientific_core/exp_runtime/`:

1. Reuse `compensated_exp_reference.jl` and replay enclosures.
2. Add an Exp-specific pair/trial adapter with explicit root and psi-scale error ledgers.
3. Feed its factor actions through affine, corrector, line-search, rollback, and terminal verification.
4. Inspect production seams in `src/cones/exp_logarithmic.jl`, `nonsymmetric/conjugate3.jl`, `scaling3.jl`, and `cones/runtime/nonsymmetric_api.jl`; migrate only after the complete experimental epoch passes.

An accurate evaluator alone cannot repair a genuine defect in stored shadow coordinates. Preserve that distinction.

**R0.4 — review production policy as a separate change.**

Only after successful scope qualification, propose an explicit default-routing/representation policy. Do not silently broaden the existing half-Power admission to Exp, arbitrary Power exponents, or mixed products.

### Acceptance

- Named default failures become default-route certified solutions at unchanged requested tolerances.
- Independent original-coordinate feasibility, objectives, gap, and relevant rays agree.
- Float64/BF256/BF512/x4 cold-seed and boundary controls remain valid within their declared supported scopes.
- No dense-Theta consumer receives factor-only authority.
- No model-name dispatch, silent precision promotion, tolerance widening, or prohibited candidate integration.

**Completion is falsified by:** opt-in-only success, capture-only replay, tiny-step stagnation, bad rollback, unsupported mixed products silently dispatched, or regression of existing precision/control cases.

**Effort:** 3–5 days for the closure suite; 5–8 days for the half-Power qualification slice; 8–15 days for an Exp research epoch. Full default closure remains mathematics-dependent and should not receive a guaranteed date.

---

## 4. Rank 2 — R1 cross-version ABI and R1-D extremes

### What exists

`src/accuracy_contract.jl`, owned-result tests, provider adapters, original-coordinate certificate summaries, and typed failure symbols exist.

The provider workflow is manual, fixes Julia to **1.10**, and uses historical MFLA pins. It does not establish Julia 1.10/1.11/1.12 compatibility or optional-extension closure.

### Implementation steps

**R1.1 — make the compatibility matrix explicit.**

Touch:

- `.github/workflows/provider-matrix.yml`
- `scripts/provider_smoke.sh`
- `validation/providers/provider_smoke.jl`
- `validation/providers/multifloat_linear_algebra_integration.jl`
- New provider-extension compatibility test.

Use isolated environments for each supported Julia/provider pin combination. Add QDLDL and LinearSolve extension-present/absent legs, including relevant dependency load orders. Record actual package paths, commits/tree hashes, Julia, MPFR/GMP, manifests, and loaded extensions.

Test factorization, solves, original-operator residuals, precision ownership, and typed refusal—not merely successful imports. If an old pin is incompatible, either repair the adapter or explicitly revise the supported-version declaration with approval.

**R1.2 — complete the extreme-result matrix.**

Touch:

- `test/test_r1_full_qualification.jl`
- `test/test_r1b_owned_object_matrix.jl`
- New `test/r1d_extreme_certificates.jl`
- Relevant defects only in `src/accuracy_contract.jl`, `src/validation.jl`, `src/certificates/certificates.jl`, and native-HSD recovery/certificate functions.

Add:

- Very small positive tau and safe refusal when normalization overflows.
- Scaled primal/dual infeasibility rays across safe exponent ranges.
- Zero/ambiguous rays and deliberate Unknown results.
- Actual mutable-backing mutation of source, views, outputs, and retained results.
- Precision/rounding changes that must invalidate incompatible factors.
- Independently recomputed original-coordinate equations.

### Acceptance and falsifiers

Every declared supported matrix cell executes its intended extension tests. Unsupported cells explicitly refuse; missing providers cannot count as passed tests.

**Falsified by:** skipped BF tests reported as success, version-dependent stale factors, backing aliases, false infeasibility certification, nonfinite normalized results accepted, or Unknown promoted to a successful status.

**Effort:** 3–5 days for matrix plumbing; 4–7 days for extreme cases and bounded repairs.

---

## 5. Rank 3 — R2-D long-run retained-memory bound

### What exists

Session symbolic leases, invalidation transactions, symbolic reuse counters, and narrow task-isolation tests exist. R2-D currently measures allocation variation over ten solves.

### Exact gap

Allocated bytes, a flat short RSS trace, and finite-batch pool samples do not establish retained-live storage or peak-memory admission.

### Implementation steps

**R2.1 — define a bounded ownership contract.**

Specify supported problem shape, precision, worker count, cache policy, and retained-result policy. Bound solver-owned state separately from caller-retained outputs.

Touch:

- `src/prepared.jl`: `SolveState`, `_solve_prepared!`.
- `src/factor_cache/session_symbolic_lease.jl`.
- `src/pipeline/workspace_estimate.jl`.
- Provider-owned workspace/factor accounting seams, where available.

Add a read-only ownership inventory: arrays and actual backing capacities, factor/symbolic storage, MPFR limbs, worker scratch, retained leases, and replacement overlap. Deduplicate shared owners. Mark unavailable quantities as unknown rather than applying a guessed multiplier.

**R2.2 — add a long-run falsification harness.**

Add `benchmark/lifecycle/retention_bound.jl` and permanent bounded regression tests.

Start with one supported Float64 session; then BF256/512 and x4 where supported. Warm first, run at least 1,000 same-shape updates, inject failed updates periodically, and separately exercise structure/precision replacement. Test both discarded results and a fixed-size retained-result ring.

Record owner counts/capacities, live versus peak RSS, GC checkpoints, symbolic counts, and stage overlap. Run ordinary-GC and diagnostic full-GC modes separately.

**R2.3 — publish a scoped bound, not a universal RSS promise.**

Define a bound in terms of shape, precision, workers, cache slots, and retained-output count. List every excluded or unavailable native allocation. Repair leaks or uncontrolled caches before claiming closure.

### Acceptance and falsifiers

- Retained solver-owned capacities obey the declared bound independently of solve count.
- Failure/replacement paths release obsolete authority and storage.
- Native scratch and overlap assumptions are documented and supported.
- RSS observations stay within a separately declared operational envelope.

**Falsified by:** solve-count-proportional ownership growth, missing native capacity data, unbounded caches, stale leases, or declaring a bound solely from forced-GC/RSS plateaus.

**Effort:** 8–12 days for inventory, first scoped bound, and harness; unavailable provider accounting may require a separate provider contract before full closure.

---

## 6. Rank 4 — R3 sparse admission

### What exists

BigFloat scalar-LP research execution, private CSC construction, frozen maps/signs/shifts, QDLDL wrapping, and original-system direction checks exist. The memory-admitting entry deliberately refuses; research scope is small and unadmitted.

### Implementation steps

1. Extend existing `experimental_sparse_core_*` tests across the supported provider/version matrix.
2. Keep the first admission scope scalar LP and current bounded dimensions; do not begin with general cone expansion.
3. In `src/kkt/symmetric_core.jl`, augment `experimental_sparse_core_memory_inventory` with the complete phase-live ledger from R2.
4. Account for symbolic fill, permutations, factor capacity, lower/upper copies, snapshots, MPFR storage, refinement scratch, and replacement overlap.
5. Verify `_research_private_lp_pattern`, `factor_experimental_sparse_core_epoch!`, `experimental_sparse_core_accept`, and `solve_experimental_sparse_core_direction!` against independently assembled original systems.
6. Only after the bound is established, revise `prepare_experimental_sparse_core_state` for that exact scope.

Related files:

- `src/factor_cache/routes/experimental_sparse_core.jl`
- `src/factor_cache/routes/qdldl_sparse.jl`
- `validation/scientific_core/PRIVATE_SPARSE_PATTERN.md`

### Acceptance and falsifiers

Acceptance requires genuine sparse execution, truthful ordering/factor identity, frozen explicit regularization, original-unshifted-system direction qualification, and admission before uncontrolled allocations.

**Falsified by:** dense fallback, shifted-system-only residual checks, underestimated fill/capacity, stale map reuse, or merely relabeling the existing diagnostic inventory as a proven bound.

**Effort:** 8–15 days after required ABI/resource facts exist. Otherwise deliver better accounting and retain `unavailable`.

---

## 7. Rank 5 — large PSD and chordal recovery

### What exists

PSD panel/congruence primitives, reference tiles, finite-gate tests, spectral infrastructure, and chordal graph analysis exist. `src/chordal.jl` explicitly remains detection-only and outside automatic solve dispatch.

### Implementation steps

**R4.1 — build a large-PSD reference ladder.**

Touch:

- `src/kkt/psd_panels.jl`: `update_psd_panel_congruence!`, `psd_panel_schur_tile!`, `reference_psd_panel_schur_tile!`, `psd_panel_apply_inverse!`.
- `src/cones/symmetric/psd.jl`, `eigen.jl`.
- `test/test_r4a_pairing_matrix.jl` and new bounded large-PSD tests.

Cover small through k=100 where resources permit, odd tails, views/transposes, cancellation, near-singular SPD cases, and declared precisions. Measure actual operator storage before choosing one optimization.

Do not revive the rejected inverse-check removal or promote the MFLA panel candidate that was unreachable in the measured KKT path.

**R4.2 — implement chordal transformation in validation first.**

Reuse `aggregate_sparsity` and `analyze_chordal_structure` in `src/chordal.jl`. Add a validation-only transform/recovery module with explicit clique/separator maps and primal/dual pullbacks.

First settle which mathematical representation is being transformed: sparse PSD decomposition versus PSD completion. Principal-clique positivity alone is not a generic replacement for a fully specified sparse PSD constraint.

Test paths, overlapping cliques, nonchordal cycles, and dense controls against independent original-matrix checks. Production wiring is a later reviewed step.

### Acceptance and falsifiers

Original-coordinate equivalence, separator consistency, dual stationarity, objective preservation, finite/PSD checks, and honest memory accounting must pass before performance credit.

**Falsified by:** missing overlap constraints, incorrect dual pullback, discarded entries/rank, weakened spectral checks, or kernel speedup on an unreachable route.

**Effort:** 5–8 days for the PSD ladder; 10–17 days for the first chordal transform/recovery prototype.

---

## 8. Rank 6 — N14 BigFloat512 and SDPB comparison

### What exists

`validation/n14_campaign/` contains a bounded original/transformed campaign, model builders, basis preparation, and original-coordinate acceptance. However:

- `solve_n14.jl` hardcodes Float64x4.
- `basis.jl::sdpx_basis` rejects mutable target scalars.
- Raising precision on already-rounded x4 rows cannot create a BF512 scientific input.
- No SDPB comparison artifact was found in the inspected layout.

### Implementation steps

1. Locate and hash the complete high-precision N14 source and generator. If unavailable, stop source-qualification claims.
2. Parameterize `solve_n14.jl`, `prepare_basis.jl`, `basis.jl`, and `fast_model.jl` for explicit arithmetic/model precision.
3. Build BF512 coefficients directly from adequate-precision source, with owned BigFloat storage; do not upcast x4 input.
4. First run original-coordinate N6 controls, then original N14. Admit a transformed BF512 arm only after owned-storage, invertibility, and pullback tests pass.
5. Add `validation/n14_campaign/export_sdpb.jl` and `compare_sdpb.jl` as new harnesses.
6. Validate the SOC/PSD encoding, objective sign, normalization, sampling, and dual pullback on small independently checkable cases before N14.
7. Compare the same finite sampled problem and declared accuracy target, recording preprocessing, solve, verification, memory, solver versions, and actual arithmetic.

### Acceptance and falsifiers

Both outputs must satisfy an independent verifier of the same original finite model. Preserve all 65 variables and 9,300 constraints unless a separately certified transformation proves equivalence.

**Falsified by:** rounded-source padding, old-basis reuse, different SDPB sampling/normalization, transformed-only certification, status-only comparison, or continuum/physical claims inferred from finite samples.

A timeout or Unknown is an honest campaign result, not successful scientific qualification.

**Effort:** 3–5 days for the BF512/source-correct driver; 5–8 days for export/comparison validation; large runs require separately authorized resources.

## 9. Completion discipline

For every milestone, commit implementation and tests separately from qualification evidence. Evidence must record immutable source/provider/input identities, requested and actual precision, extension coverage, resource limits, exits, failures/skips, and independent verification scope.

The next immediate deliverables are **R0.1, R1.1, and R2.1**. They are concrete engineering work with no unresolved numerical-policy promotion. Default R0 promotion, sparse admission, chordal dispatch, and N14 scientific acceptance remain conditional gates—not scheduling promises.