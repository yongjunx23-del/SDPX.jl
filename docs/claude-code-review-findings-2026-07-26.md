# SDPX Code Review Findings and Required Follow-up

Date: 2026-07-26

Repository: the SDPX.jl repository root

## Purpose

This document records the findings from a read-only review of the current SDPX codebase. It is intended as an implementation handoff for Claude Code.

The review included focused correctness tests and several small numerical probes. The solver source was not modified during the review.

## Important Worktree Safety Notice

The repository was changing while the review was in progress. At the end of the review, the observed state was:

```text
## main...origin/main [ahead 1]
 M bench/baselines/gates.json
 M bench/gates.jl
 M src/kkt_sparse_backend.jl
 M test/kkt_regressions.jl
 M test/pipeline.jl
```

Treat all existing modified and untracked files as user-owned work. Before changing anything:

```bash
git status --short --branch
git diff --stat
git diff
```

Do not overwrite, revert, reset, or silently replace unrelated changes. Reconcile each proposed fix with the current contents because some issues may already be partially addressed by work performed after this review.

Do not commit or push changes unless the user separately requests it.

## Review Summary

The focused tests passed, and the ordinary solve paths appear functional. The most important unresolved risks are:

1. unsafe null-space memory estimates for rank-deficient equality systems;
2. sparse equality-constrained LP regularization and refinement inconsistencies;
3. diagnostics that report planned algorithms instead of the algorithms actually executed;
4. invalid calibration profiles being accepted;
5. process-global BLAS thread changes that are unsafe for concurrent solves;
6. several result fields whose names or values do not match their documented meaning;
7. missing BigFloat precision hygiene in the dedicated LP path.

The new null-space reduction code should not be enabled automatically until its memory gate and test coverage are corrected.

## Priority 0: Establish a Stable Baseline

Before implementation:

1. Inspect the current diff and identify which findings are already being addressed.
2. Run focused tests.
3. Record the Julia version, BLAS implementation, BLAS thread count, and Julia thread count.
4. Save baseline runtimes and allocations for the affected LP, SDP, null-space, and BigFloat tests.

Suggested commands:

```bash
julia --version
julia --startup-file=no --project=. -e 'using LinearAlgebra; println(BLAS.get_config()); println("BLAS threads: ", BLAS.get_num_threads()); println("Julia threads: ", Threads.nthreads())'
julia --startup-file=no --project=. -e 'using Test, SDPX; include("test/lp_sparse.jl"); include("test/nullspace_reduction.jl"); include("test/moi_regressions.jl"); include("test/result_certificate.jl"); include("test/bigfloat_ownership_regressions.jl")'
```

After focused work is complete, run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
git diff --check
```

## Priority 1: Correctness and Memory Safety

### 1. Null-space memory estimation is unsafe

Relevant code:

- `src/nullspace.jl`, especially `nullspace_memory_bytes` near lines 53-57;
- `src/nullspace.jl`, `nullspace_reduce` near line 234.

The current estimate is effectively based on:

```text
m * (m - number_of_equalities)
```

However, the actual null-space basis has dimensions:

```text
m * (m - rank(B))
```

For rank-deficient equality systems, this can underestimate memory by a large factor. The documentation currently describes full column rank as producing the largest `Z`, but for a fixed number of equality constraints it produces the smallest `Z`.

Observed example:

```text
m = 100
number_of_equalities = 80
rank(B) = 1
reported estimate = 16,000 bytes
actual Z storage is approximately 79,200 bytes
Base.summarysize(Z) = 79,248 bytes
```

With a 20,000-byte budget, the current selector incorrectly approves null-space reduction.

Unchecked integer multiplication can also overflow. A large synthetic input returned a negative byte count, after which the memory gate returned `true`.

Required changes:

- Use checked or saturating integer arithmetic for all memory estimates.
- Return `typemax(Int)` or a structured “estimate overflowed” result when the exact value cannot be represented.
- Do not use the number of equality columns as a substitute for numerical rank.
- Before rank is known, use a safe upper bound, normally `m * m * sizeof(T)`, plus QR workspace and transformed-problem storage.
- Prefer a staged implementation:
  1. compute a compact or rank-revealing factorization;
  2. determine numerical rank;
  3. calculate the exact projected-basis and transformed-data budget;
  4. materialize the full basis only if the budget permits it.
- Make `nullspace_reduce` enforce the memory gate internally. It must not rely only on the caller.
- Include transformed objective, cone data, workspace, and temporary factorization storage in the estimate.
- Return a clear fallback reason in diagnostics when reduction is rejected.

Required tests:

- full-rank equality matrix;
- rank-deficient equality matrix;
- nearly rank-deficient equality matrix with tolerance-dependent rank;
- zero equalities and zero-dimensional null space;
- sparse input matrix;
- `Float64`, `Float64x4`, and `BigFloat`;
- deliberately tiny memory budgets;
- integer-overflow-sized synthetic dimensions without allocating those matrices;
- verification that rejection does not partially mutate the problem.

Do not enable automatic null-space reduction until these tests pass.

### 2. Sparse equality LP KKT formulation needs consistency and refinement

Relevant code:

- `src/lp_sparse.jl`, around lines 100-105;
- `src/kkt_sparse_backend.jl`, `refine_solution!` near line 468;
- dense KKT construction in the dedicated LP solver.

After accounting for the sparse formulation's multiplier sign convention, the equality-block regularization sign does not match the dense KKT formulation. The discrepancy is of order `delta`, so it can become significant when regularization is increased for a difficult or nearly singular problem.

The iterative-refinement helper exists, but it is not called by the solver. A direct numerical probe showed:

```text
problem size: n = 80, m = 20
regularization delta: 1e-8
direct residual: 1.9615033955422412e-7
residual after two refinement passes: 2.3869795029440866e-14
```

The current helper forms residuals using the already regularized system matrix. Therefore it improves the factorization solve but cannot remove bias relative to the unregularized Newton equations, despite documentation referring to the “original KKT equations.”

Required changes:

- Write down one canonical dense KKT equation and derive the sparse sign conversion explicitly in a code comment and test.
- Make the sparse and dense regularization conventions algebraically equivalent.
- Decide whether refinement targets:
  - the regularized system solved by the factorization; or
  - the original unregularized Newton equations.
- If unregularized refinement is desired, implement an explicit residual operator for the original equations and use the regularized factorization as the correction preconditioner.
- Integrate refinement into predictor and corrector solves only when a residual-based trigger indicates it is useful.
- Preallocate residual and correction buffers.
- Limit the number of passes and stop when the residual stops improving.
- Record the number of refinement passes, initial residual, final residual, and failure/fallback reason in diagnostics.

Required tests:

- equality-constrained sparse LP with known solution;
- ill-conditioned and nearly dependent equality rows;
- equivalence of dense and sparse Newton directions at the same regularization;
- residual monotonicity or safe early termination;
- no-equality sparse-normal path;
- end-to-end objective, primal residual, dual residual, and status comparison between dense and sparse backends.

A well-conditioned test with 220 variables, box constraints, and `sum(x) = 1` selected `:sparse_ldl`; dense and sparse solves were both optimal and differed in objective by approximately `5.44e-15`. This is useful as a smoke test but is not sufficient to exercise the regularization issue.

### 3. Reject invalid calibration profiles

Relevant code:

- `src/kernels/extended_precision_blas/calibration.jl`, around lines 104-123;
- crossover selection and workspace initialization.

The cache loader currently accepts invalid finite values and `NaN`. A synthetic profile with negative thresholds and a `NaN` density was accepted, and the selector enabled a packed kernel for a two-column case with reason `:predicted_speedup`.

Required validation:

- minimum columns must be at least 2;
- work thresholds must be finite and nonnegative;
- predicted speedup must be finite and at least 1;
- density thresholds must be finite and lie in `[0, 1]`;
- unknown versions or incomplete profiles must fall back entirely to conservative defaults;
- invalid profiles should optionally produce a diagnostic warning without breaking a solve.

The calibration signature should also include all settings that materially affect the crossover:

- Julia version;
- arithmetic type;
- BigFloat precision;
- Julia thread count;
- BLAS implementation;
- BLAS thread count;
- relevant kernel version or source revision;
- architecture information already used by the cache.

`load_profile` appears reachable once per block through crossover selection. Cache the validated profile in memory per configuration or load it once into a solver/workspace object.

Required tests:

- negative values;
- `NaN` and `Inf`;
- missing fields;
- incompatible schema versions;
- thread-count mismatch;
- BigFloat-precision mismatch;
- valid profile round trip;
- conservative fallback after corruption.

## Priority 2: Truthful Results and Diagnostics

### 4. Report algorithms actually used, not only the pre-solve plan

Relevant code:

- `src/pipeline.jl`, plan construction near lines 444-459;
- `src/lp_solver.jl`, runtime sparse selection near lines 1127-1129;
- `src/pipeline.jl`, `_attach_diagnostics` near lines 918-927.

The execution plan is built before presolve and scaling complete. For an equality-constrained LP it can record:

```text
kkt = :dense_lu
gram = :blas_syrk
```

At runtime, the solver may instead select:

```text
kkt = :sparse_ldl
```

The final result currently copies the plan into `selected_algorithms`, so it can report a backend that never ran and a Gram kernel that was skipped.

Required changes:

- Distinguish `planned_algorithms` from `actual_algorithms`; or
- update the final execution record whenever a backend or kernel is selected, attempted, rejected, retried, or replaced.
- Include fallback history where practical.
- Do not report `:blas_syrk` if no Gram matrix was constructed.

Required tests:

- force sparse LDL and assert that the result says `:sparse_ldl`;
- force dense LU and assert that the result says `:dense_lu`;
- trigger a fallback and verify both attempted and final backends;
- verify the reported kernel corresponds to an actually executed kernel.

### 5. `minimum_psd_eigenvalue` does not contain an eigenvalue

Relevant code:

- `src/validation.jl`, around lines 791-824.

The current value is the negative of a required diagonal PSD shift, not the minimum eigenvalue. For `diag(2, 3, 4, 5, 6)`, the reported proxy was `-0.0`, while the true minimum eigenvalue is `2.0`.

For larger blocks, the validator can perform only a pass/fail test at the allowed shift, so even the required shift may not be fully resolved.

Required changes:

- Rename the current metric to something such as `negative_required_psd_shift` or `psd_shift_violation`; and
- compute the true minimum eigenvalue only when optional diagnostics or spectrum extraction requests it.
- If compatibility requires preserving `minimum_psd_eigenvalue`, document and deprecate it, then add an accurately named field.
- Clearly distinguish exact eigendecomposition, estimated eigenvalue, and threshold-only validation.

Required tests:

- positive definite diagonal matrix;
- semidefinite matrix with zero eigenvalue;
- matrix with a known negative eigenvalue;
- large block that uses the threshold-only path;
- optional spectrum extraction enabled and disabled.

### 6. Initial parameter diagnostics may differ from actual parameters

Relevant code:

- `src/solve.jl`, SDP pipeline and equilibration around lines 1438-1442;
- `src/solve.jl`, parameter recommendation around lines 625-634;
- `src/pipeline.jl`, diagnostic attachment near line 926;
- LP scaling and setup around `src/lp_solver.jl:1103` and `src/solve.jl:1543-1555`.

The plan can contain parameters selected before equilibration, while the core recomputes parameters from the equilibrated SDP. Diagnostics then report the plan values rather than those actually used.

For LPs, automatic parameters are selected on the unscaled problem and then reused after scaling. The current LP scale indicator is row-scale invariant but is not generally invariant to variable scaling.

Required changes:

- Record the exact initial parameters passed into the iteration loop.
- Record adaptive values separately by iteration.
- Select LP profiles from the scaled problem unless invariance has been established and tested.
- Preserve the distinction between user-requested, automatically recommended, and actually used values.

### 7. BigFloat LP bypasses precision-consistency handling

Relevant code:

- `src/solve.jl`, `_solve_sdp_core!` near lines 614-617;
- dedicated LP dispatch near lines 1534-1561.

The SDP path calls `check_precision_consistency` and can reround converted inputs. The dedicated LP path bypasses this logic.

Required changes:

- Move precision-consistency checking to a shared stage before LP/SDP dispatch.
- Perform it exactly once.
- Preserve the user's input objects.
- Respect the solve time budget while rerounding.
- Report whether input rerounding occurred.

Required tests:

- construct BigFloat LP coefficients at a lower precision;
- request a higher solve precision;
- verify the warning or diagnostic;
- verify rerounding behavior when enabled;
- verify original input values and precision are not mutated;
- compare objective and residuals with a problem constructed natively at the requested precision.

### 8. Status and provenance semantics are inconsistent

Relevant code:

- `src/types.jl`, `NumericalFailure` documentation near lines 48-51;
- `src/validation.jl`, `certify_final_result` near lines 702-719;
- `src/validation.jl`, `solve_provenance` near lines 757-770.

Issues:

- `NumericalFailure` is documented as the status for independent-validation failure, but validation currently downgrades the result to `Stalled`.
- `mixed_precision_used` reflects only the final active state. If mixed precision was used successfully and later disabled, the result can incorrectly report `false`.
- `refinement_steps` remains empty even when refinement mechanisms exist.

Required changes:

- Decide and document authoritative status semantics.
- If independent validation invalidates an apparently successful result, return `NumericalFailure` and map it consistently to MOI's numerical-error status; otherwise revise the enum documentation and tests.
- Track `mixed_precision_ever_used` separately from `mixed_precision_active_at_exit`.
- Count successful mixed-precision factorizations.
- Aggregate actual refinement passes and report them.

## Priority 3: Performance and Concurrency

### 9. First sparse LP factorization is duplicated

Relevant code:

- `src/lp_sparse.jl`, around lines 125-135.

When the sparse system has not yet been analyzed, `analyze!` appears to perform numeric factorization and increments the factorization counter. The caller then immediately invokes `factorize!` again on the same matrix.

Required changes:

- On the first call, return the successful factorization produced by `analyze!`.
- Set `analyzed = true` only after successful analysis/factorization.
- On later calls with the same sparsity pattern, perform only numeric refactorization.
- Confirm whether the `factorization` field in `LPSparseSystem` is redundant and remove it only if all uses and compatibility concerns have been checked.

Required test:

```text
after the first factor call:
analyses == 1
factorizations == 1
```

Also verify that numeric refactorizations reuse the symbolic analysis when the sparsity pattern is unchanged.

### 10. Sparse solve compatibility fix introduces avoidable allocations

Relevant current worktree code:

- `src/kkt_sparse_backend.jl`, around lines 157-160;
- `src/lp_sparse.jl`, solve wrapper around lines 149-152.

The current Julia 1.10 compatibility change uses:

```julia
factorization \ rhs
```

which allocates an output vector. The wrapper also allocates `similar(rhs)`, so predictor and corrector solves can allocate two vectors each.

Correctness and Julia 1.10 compatibility take priority over allocation reduction.

After correctness is established:

- reuse solution storage in `LPSparseSystem`;
- investigate a version- or capability-based `ldiv!` path;
- consider a supported matrix right-hand-side interface if vector `ldiv!` is unavailable;
- benchmark factor solve allocations separately from matrix assembly;
- test both Julia 1.10 and Julia 1.12.

The review machine had Julia 1.12.6 only, so Julia 1.10 behavior remains unverified here.

### 11. BLAS thread management is process-global and concurrency-unsafe

Relevant code:

- `src/step.jl`, around lines 110-118;
- thread tests in `test/threads.jl`.

`BLAS.set_num_threads` mutates process-global state. Two concurrent solves can interleave:

1. read previous value;
2. set requested value;
3. execute work;
4. restore a stale previous value.

This can cause oversubscription, nondeterministic performance, and an incorrect final BLAS thread setting.

Possible designs:

- protect the complete BLAS-thread-scoped region with a process-wide `ReentrantLock`;
- set BLAS thread count once at process startup and avoid per-solve mutation;
- use backend-specific thread controls if available.

Do not use a lock only around `set_num_threads`; the protected region must include the computation whose thread setting is being controlled.

Required tests:

- run concurrent solves requesting different BLAS widths;
- verify each solve completes correctly;
- verify the original BLAS thread count is restored;
- verify exceptions also restore the prior value;
- measure whether locking removes parallel benefit and document the policy.

### 12. Other memory estimators can overflow

Relevant code:

- `src/pipeline.jl`, around lines 832-841;
- `estimate_sdp_workspace_bytes` near line 844 and related estimators.

Large synthetic dimensions can overflow `Int`, producing plausible but incorrect values or negative values.

Required changes:

- use shared checked/saturating multiplication and addition helpers;
- make every estimator return a conservative value after overflow;
- include overflow/fallback information in diagnostics where useful;
- test near `typemax(Int)` without allocating matrices.

## Priority 4: Null-space Integration and Documentation

The current null-space commit adds useful helper functions, but it is not yet integrated into the automatic pipeline. Keep it opt-in until the safety work above is complete.

Coverage still needed:

- consistent rank-deficient problems;
- sparse source matrices;
- `Float64x4`;
- BigFloat at multiple precisions;
- memory-budget refusal;
- solve-result reconstruction;
- equality multiplier recovery;
- final certificate validation after reconstruction;
- comparison with the unreduced solve.

The docstring for `recover_equality_multiplier` near `src/nullspace.jl:327` describes a signature containing `X_blocks`, `Y_blocks`, and `c`, while the function accepts a different argument list. Correct the documentation after confirming the intended API.

## Legacy SDPJSolver Content

No active legacy `SDPJSolver.jl` implementation was found. Remaining references appear to be attribution, license, documentation, or historical comments.

Do not delete attribution or license references. They are important for MIT-license provenance and for explaining that SDPX was derived from and improved upon an earlier open-source project.

## Recommended Implementation Order

Work in small, reviewable patches:

### Phase A: Safety and numerical correctness

1. Null-space memory arithmetic, rank handling, and internal budget enforcement.
2. Sparse LP KKT sign/regularization equivalence.
3. Residual-triggered sparse LP refinement.
4. Calibration-profile validation and conservative fallback.

### Phase B: Result truthfulness and precision robustness

1. Planned versus actual backend and kernel reporting.
2. Accurate parameter reporting after scaling/equilibration.
3. BigFloat LP precision-consistency handling.
4. PSD metric naming or true eigenvalue computation.
5. Status, mixed-precision, and refinement provenance.

### Phase C: Performance and concurrency

1. Remove the duplicate first sparse factorization.
2. Cache validated calibration profiles in memory.
3. Reduce sparse solve allocations while retaining Julia 1.10 compatibility.
4. Make BLAS thread control concurrency-safe.
5. Harden all remaining memory estimators.

### Phase D: Integration and benchmarking

1. Expand null-space and sparse equality LP tests.
2. Run the entire test suite on Julia 1.10 and Julia 1.12.
3. Benchmark before and after each performance change.
4. Enable new automatic behavior only after accuracy and stability gates pass.

Do not combine all phases into one large patch. Each patch should have a focused test that fails before the fix and passes afterward.

## Numerical Acceptance Criteria

For every solver-path change:

- final status must remain correct;
- objective value must agree with the baseline within the requested precision-dependent tolerance;
- primal and dual residuals must not regress materially;
- PSD validation must not regress;
- refinement must reduce the intended residual or exit safely without changing the accepted solution;
- sparse and dense equality LP paths must agree on well-conditioned cases;
- difficult cases must either improve or produce a clear, correct fallback/status;
- BigFloat tests must use tolerances derived from working precision rather than hard-coded Float64 tolerances.

For automatic selection:

- diagnostics must report the backend and kernel actually executed;
- every rejected algorithm must have a stable reason code;
- invalid calibration data must never enable an aggressive kernel;
- memory estimates must never wrap around;
- automatic null-space reduction must remain disabled until its worst-case budget is safe.

## Performance Acceptance Criteria

Measure wall time, allocations, and peak memory separately for:

- presolve and scaling;
- KKT/Schur assembly;
- symbolic analysis;
- numeric factorization;
- triangular/factor solves;
- iterative refinement;
- reconstruction and validation.

For multithreaded changes, test 1, 2, 4, and 8 Julia threads where hardware permits. Report BLAS threads separately.

For BigFloat:

- use one thread for MPFR scalar kernels unless a tested higher-level parallel decomposition is selected;
- record working precision;
- exclude compilation and first-run calibration from steady-state timings;
- verify that performance improvements do not come from lowering effective precision.

Do not enable a new default solely from a microbenchmark. It must improve or preserve end-to-end performance on representative LP and SDP problems without degrading accuracy or robustness.

## Test Evidence Collected During This Review

Focused test results:

| Test file | Result |
|---|---:|
| `test/lp_sparse.jl` | 12 / 12 passed |
| `test/nullspace_reduction.jl` | 16 / 16 passed |
| `test/moi_regressions.jl` | 87 / 87 passed |
| `test/result_certificate.jl` | 62 / 62 passed |
| `test/bigfloat_ownership_regressions.jl` | 24 / 24 passed |
| **Total** | **201 / 201 passed** |

Additional probes:

- Sparse LDL direct residual: approximately `1.96e-7`.
- Residual after two refinement passes: approximately `2.39e-14`.
- Sparse versus dense equality LP objective difference: approximately `5.44e-15`.
- BigFloat null-space smoke test at 192 bits:
  - computed rank: 2;
  - orthogonality error: approximately `7.97e-58`;
  - equality feasibility error: approximately `1.27e-57`.
- Positive-definite PSD diagnostic probe:
  - reported proxy: `-0.0`;
  - true minimum eigenvalue: `2.0`.
- Rank-deficient null-space memory probe:
  - estimate: `16,000` bytes;
  - actual basis storage: approximately `79,200` bytes.

The full package test suite was not run during this read-only review. Julia 1.10 was not available on the review machine.

## Definition of Done

The follow-up is complete only when:

- all Priority 1 correctness and safety items have regression tests;
- null-space reduction cannot exceed its configured memory budget through rank deficiency or integer overflow;
- sparse equality LP regularization is algebraically consistent with the dense formulation;
- refinement is integrated safely and its diagnostics are populated;
- result diagnostics match the algorithms and parameters actually used;
- BigFloat LPs receive the same precision-consistency protection as SDPs;
- corrupted calibration caches fall back safely;
- concurrent solves cannot corrupt BLAS thread state;
- the complete test suite passes on supported Julia versions, including Julia 1.10 and Julia 1.12;
- performance benchmarks show no unexplained regression;
- `git diff --check` passes;
- documentation and API comments match the implemented behavior.
