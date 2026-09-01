# Exact-duplicate equality reduction diagnosis

**Status:** diagnosis/design only; no solver implementation in this worktree.
**Worktree:** `/tmp/sdpx-route-guard`, branch `fix/bordered-route-fallback`
(`ab107b1` includes the strict one-shot restart fix).

## Executive finding

The exact-duplicate case is **not a literal factorization-singularity finding**.
The canonical factorization succeeds (bordered: 2 factorizations; expanded: 20
factorizations). The failure is an upstream **rank-deficient equality plus
objective-null degeneracy** exposed by equality elimination:

- the two equality rows are identical, so the equality panel has rank 1;
- the objective `x₁+x₂` is exactly in the equality row span, hence its reduced
  nullspace projection is mathematically zero;
- the current Float64 QR reconstruction leaves the projected objective as
  `-1.1102230246251565e-16`;
- the reduced problem is therefore a one-dimensional nonnegative feasibility
  problem with a zero/near-zero objective and two opposite affine rows;
- bordered rejects its first predictor direction at the full Newton residual
  gate, while expanded repeatedly reaches tau-collapse recovery exhaustion.

Thus the route guard correctly leaves the case fail-closed. It changes the
route, not the degenerate reduced geometry.

## Minimal standalone reproduction

`benchmark/robustness/repro_duplicate_row.jl` constructs the smallest useful
LP: two nonnegative variables, two exactly duplicated ZeroCone rows, and an
objective equal to their common left-hand side. It prints the canonical matrix,
RRQR result, reduced system, and both direct route outcomes. Run it with:

```bash
export JULIA_DEPOT_PATH="/tmp/sdpx-autoresearch/.auto/depot:$HOME/Desktop/project/SDPX/.julia-depot:$HOME/.julia"
JULIA_NUM_THREADS=4 julia --gcthreads=1 --startup-file=no --project=. -e \
  'using Pkg; Pkg.activate(; temp=true); Pkg.develop(; path=pwd()); include("benchmark/robustness/repro_duplicate_row.jl")'
```

Observed on Julia 1.12 / Float64 (the exact values are deterministic for this
small case):

```text
perturb=0.0 canonical=(n=2,m=4) reduction=(status=HSDEqualityReady,rank=1,independent=[1],dependent=[2],rank_tol=4.440892098500626e-16)
  x_particular=[0.4999999999999998, 0.4999999999999999] null_size=(2, 1) reduced=(n=1,m=2,A=[0.7071067811865475; -0.7071067811865476;;],b=[0.4999999999999998,0.4999999999999999],c=[-1.1102230246251565e-16])
  route=bordered reduction_rank=1 status=NumericalBreakdown reason=fixed_trace_predictor_residual_failed iterations=1 factorizations=2 product_status=ProductHSDBreakdown
  route=expanded reduction_rank=1 status=InsufficientPrecision reason=tau_collapse_recovery_exhausted iterations=20 factorizations=20 product_status=ProductHSDInsufficientPrecision
perturb=1.0e-8 ... rank=2 ... reduced=(n=0,m=2) ...
```

The public `kkt_route=:bordered` invocation takes the approved one-shot
fallback and finishes with `(:bordered, :expanded)` and terminal
`:tau_collapse_recovery_exhausted`; it does not claim a repair.

A useful control is a single (non-duplicate) equality with the same objective:
it also fails (`bordered`: `symmetric_core_dispatch_exception`, `expanded`:
`:tau_collapse_recovery_exhausted`). Changing that objective to `x₁` solves on
both routes. Therefore **duplicate-row deletion alone is insufficient**: the
repair must address equality-induced objective-null degeneracy as well.

The `1e-8` perturbation changes the equality rank to 2 and leaves zero reduced
variables (`n=0`), and the public solve is accepted. This is a diagnostic
contrast, not a proposed tolerance widening or a benchmark-specific exception.

## Trace through the implementation

### 1. Canonical data and reduction

For the exact repro, canonicalization produces

```text
A = [-1  0; 0 -1; -1 -1; -1 -1]
b = [0, 0, -1, -1]
c = [-1, -1]
```

Rows 3 and 4 are ZeroCone equality rows. In
`src/hsd/native_hsd_public.jl:992-996`, the public native path canonicalizes
and calls `hsd_equality_reduce` (unless a fixed-trace Q3 plan is present); this
path does not call the higher-level `preprocess` cleanup first. The public
entry point in `src/public/optimize.jl:384-393` confirms the direct
compile/canonical/native sequence.

`src/hsd/equality_reduction.jl:388-422` partitions ZeroCone rows and sets the
rank scale/cutoff. The ordinary dense branch uses column-pivoted QR at
`:429-466`; exact duplicate rows produce rank 1, not `HSDEqualityRankAmbiguous`.
The result is `independent=[1]`, `dependent=[2]`, `upper=[1.4142135623730951]`,
and a unit transfer coefficient (within one ulp). Dependent-RHS consistency is
checked at `:535-580`, and the reconstructed particular solution/nullspace is
checked at `:584-610`; both pass, so the reduction returns `HSDEqualityReady`.

The reduced objective is formed as `transpose(null_basis) * canonical.c` in
`src/hsd/equality_reduction.jl:192-212`. Since `c` is in the equality row
span, this is zero in exact arithmetic; the recorded `-1.11e-16` is QR
roundoff. There is no subsequent objective-null/degenerate-feasibility guard
before `_hsd_eq_build_reduced` returns the reduced model at `:612-625`.

### 2. Existing duplicate cleanup is not the executed fix

The older/staged preprocessing machinery *does* have collision-checked exact
cleanup: `src/preprocessing.jl:1097-1155` fingerprints equality columns,
checks `_equal_equality_columns`, and removes exact duplicates while preserving
a multiplier map. But the direct public native sequence above bypasses that
`preprocess` stage. More importantly, the one-equality control proves that
merging duplicate rows alone leaves the same objective-null geometry and cannot
be the complete fix.

### 3. First bordered failure

After factorization and solution scatter, the predictor path invokes
`_product_hsd_newton_residual_ok` at `src/hsd/predictor_corrector.jl:897-917`.
The returned reason `:fixed_trace_predictor_residual_failed` is assigned only
at `:915-917`, after the candidate is finite (`:910-913`). The residual helper
checks, in order, primal (`src/hsd/product_cone_hsd.jl:2236-2242`), dual
(`:2244-2252`), gap (`:2254-2267`), cone (`:2269-2270`), and scalar
(`:2272-2283`) equations. The current receipt does not persist which of these
subchecks first failed, so the evidence supports a **full residual-gate
failure after a successful factorization**, but not a claim about one specific
component.

### 4. Expanded failure and route independence

Expanded uses the same canonical model and the same rank-1 reduction output;
only its KKT execution path differs. The standalone run shows 20 successful
factorization epochs followed by `ProductHSDInsufficientPrecision` and
`:tau_collapse_recovery_exhausted`, not a factor pivot exception. The reduced
one-dimensional cone feasibility geometry has no objective progress direction
(the projected objective is zero apart from roundoff), so tau recovery cannot
make the homogeneous progress certificate advance. The current data support
**upstream reduced-geometry degeneracy** as the common cause, with route-specific
failure manifestation:

| Route | First observed terminal failure | Evidence |
|---|---|---|
| bordered | `fixed_trace_predictor_residual_failed` at iteration 1 | factor count 2; predictor residual gate after scatter |
| expanded | `tau_collapse_recovery_exhausted` at iteration 20 | factor count 20; repeated recovery, no factorization failure |

This is deliberately weaker than calling the KKT matrix mathematically
singular: no singular pivot was observed. The correct severity is **P1
robustness/design blocker** because the public exact-degenerate LP is not
certified on either route and currently depends on fail-closed termination.

## Minimal fix design (no implementation here)

### Recommended design: explicit rank-deficient/objective-null handling

1. Keep the existing target-arithmetic RRQR and original-coordinate residual
   certification. Do not lower the cutoff or silently promote a near-zero
   projected objective to exact zero.
2. Extend the reduction record with an objective-null diagnostic based on a
   target-arithmetic relation residual (and, where available, exact model
   provenance), alongside rank/nullspace dimensions. This is a classification
   aid, not a tolerance relaxation.
3. For a verified objective-in-equality-span case, route the reduced LP to a
   dedicated **feasibility solve** with a numerically meaningful phase-I
   certificate, or retain the rank-reduced coordinates while using a
   degeneracy-safe homogeneous initialization/recovery. The path must return a
   normal optimal result only after original-coordinate primal feasibility,
   dual feasibility, objective, and complementarity certificates pass.
4. If the relation/objective-null condition cannot be certified in the target
   arithmetic, do not guess. Return a structured `InsufficientPrecision` /
   `NumericalFailure` reason before KKT execution (or retain the ordinary path)
   and preserve fail-closed semantics.
5. Add deterministic reconstruction of duplicate/dependent equality duals
   after the solve; duplicate merging may reduce receipt dimension, but must
   preserve the full original equality-dual map.

An exact duplicate detector/merge remains useful as a **preconditioning and
receipt simplification** step (either in the direct canonical path or by
reusing the existing collision-checked logic), but it is not sufficient by
itself. The one-equality control must remain in the regression suite to prevent
an incomplete “merge-only” fix from being accepted.

### Alternative design requiring stronger proof

Retain a full-rank equality quotient but add a formally certified phase-I
objective for the reduced cone-feasibility problem. This can avoid the current
zero-objective tau collapse, but it changes the HSD initialization and requires
new proof/certificate fields. It should not be implemented as a hidden
objective perturbation.

## Frozen-trajectory / bit-safety assessment

**Risk: medium-to-high until gated; potentially low for the frozen path if
strictly isolated.**

- A direct canonical duplicate/objective-null classifier touches public native
  setup and could alter route selection, setup status, or floating-point
  operation order for every model. It must not be enabled globally without
  proving the frozen CSDR α3 trajectory unchanged.
- The safest first implementation boundary is a cold-path branch entered only
  after target-arithmetic rank/relation evidence identifies dependent equalities
  and a certified objective-null relation. Independent full-rank CSDR α3
  equality panels should take byte-identical existing code; this requires a
  regression guard, not an assumption.
- Reusing the existing exact cleanup can change equality dimensions and dual
  reconstruction, so it also requires original-coordinate certificate tests
  and frozen SHA validation. Do not change equality ordering or QR arithmetic
  on the default full-rank path.
- Never fix this by widening residual/tau tolerances, adding a random/objective
  perturbation, or using Float64 rank decisions for extended arithmetic.
- Acceptance must include the frozen CSDR SHA
  `25ef57d499cb9fdaa45600bd11c7e6948df23ab063434eff126765545e529ca7` and the
  standard `ALL_REGRESSION_GATES_OK` guard.

## Test plan

1. Run `benchmark/robustness/repro_duplicate_row.jl` and assert exact values:
   rank 1, one dependent row, reduced `n=1`, projected objective near zero,
   bordered residual-gate failure, expanded tau-collapse failure.
2. Public-route regression: exact duplicate remains fail-closed unless the new
   certified path proves a valid result; if repaired, require original
   objective/feasibility/dual/complementarity certificates and route receipts.
3. Controls: one equality with objective in its span; one equality with
   objective transverse to its span (`x₁`); duplicate with `1e-8` perturbation;
   inconsistent duplicate RHS (must produce a valid infeasibility certificate,
   not a feasibility result).
4. Check duplicate order permutations and 2+ duplicate groups; verify original
   equality-dual reconstruction and deterministic fingerprints.
5. Exercise Float64, Float64x2/x4, and BigFloat where providers are available;
   rank decisions and relation residuals must stay in declared arithmetic.
6. Run the route-guard 25-test set, full `Pkg.test`, diff audit, and frozen CSDR
   regression guard. Require no tolerance changes and no default-path SHA drift.

## Residual risks / open questions

- The current public receipt does not expose the first failing residual
  subcomponent; a follow-up diagnostic-only telemetry field would improve
  attribution, but must not alter arithmetic or qualification semantics.
- A certified phase-I/degeneracy-safe HSD path is not yet designed to the level
  needed for implementation review; this document intentionally does not claim
  that duplicate merging is a complete solution.
- The exact duplicate and one-equality controls show the issue is broader than
  duplicate storage: equality-induced constant objectives and degenerate LP
  feasibility need a common, certificate-backed policy.
