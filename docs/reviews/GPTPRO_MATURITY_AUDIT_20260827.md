# SDPX.jl follow-up maturity audit

I accept the reported local and cluster outcomes as authoritative: the A1 changes resolved the mixed free/equality/PSD origin stall; Phase 0 was frozen with a fast gate; P1 added typed Nonpositive and RSOC transforms but has not yet rewired the production lowerers; RSOC, Exp, and Power have native working cases. 

This remains a **source-only review**. I did not run Julia. I inspected the complete supplied `src/` tree: approximately **90,646 Julia source lines in 137 files**, plus roughly **53,800 test lines**. The archive does not contain the package extension modules declared in `Project.toml`, so I have not credited the weak-dependency provider integrations as fully audited.

## Bottom line

The A1 work is a genuine numerical improvement, not a cosmetic patch. In particular:

* the runtime PSD NT congruence construction is algebraically sound;
* terminal verification before direction breakdown repairs a real false-negative termination path;
* the original-coordinate certificate boundary remains strong;
* the new transform work is headed in the correct direction.

However, I would **pause Phase 2 briefly for a 3–5 day stabilization patch**. The exact supplied tree contains one apparent package-load blocker and several integration defects that should not be carried into the new `NewtonSystem`.

**Overall production-maturity score: 6.0/10.**

That means **advanced research solver with unusually strong certificate discipline**, but not yet a production general-purpose conic solver. It is substantially closer than in the previous review.

---

# A. Maturity scorecard

| Area                                  |      Score | Evidence that earns the score                                                                                                                                                                    | Evidence that caps the score                                                                                                                                                                                                                                                 |
| ------------------------------------- | ---------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correctness and certificate authority | **7.2/10** | Independent original-coordinate optimality gate; native HSD also independently gates primal- and dual-infeasibility rays; successful statuses are downgraded when public verification fails.     | Supplied tree likely fails to load; nonsymmetric trial scaling uses the wrong μ; direct Exp/Power optimal verification is skipped unless ad hoc refinement succeeds; P1 is not production-wired; trivial dual infeasibility remains unclassified.                            |
| Numerical robustness                  | **5.8/10** | Cholesky-congruence PSD NT, factorized orientation checks, Cholesky-frame boundary handling, terminal candidate checks, strict fail-closed cone kernels, working mixed-cone cases.               | Native HSD still relies on dense normal-equation condensation plus an unregularized bordered LU; no equilibration, robust KKT regularization, inertia control, iterative-refinement ladder, or route recovery; rank-one SOC/PSD and redundant-bound cases remain known gaps. |
| Performance and scale                 | **4.2/10** | Strong reusable ideas already exist in factor caches, sparse LA, deterministic block threading, fixed-width extended-precision kernels, and mixed-precision planning.                            | The actual product HSD route is explicitly dense and serial, performs two numerical reductions, retains several dense copies, and does not use the mature sparse/threaded/factor-cache machinery.                                                                            |
| API completeness and MOI conformance  | **6.3/10** | Substantial direct `Model` API; one-shot `MOI.copy_to` adapter; LP/SOC/RSOC/PSD/Zero/Reals/Nonpositive surface; statuses, objectives, primal/dual getters, starts, and result attributes.        | Exp/Power are deliberately absent from MOI; only a targeted `MOI.Test` subset runs; standard rank-one PSD tests are excluded; native HSD rejects starts and many public settings; variable-attribute introspection is incorrect.                                             |
| Testing culture                       | **8.0/10** | About 1,063 testsets and 9,523 tests; property/reference tests, arbitrary-precision tests, allocation tests, certificate tests, MOI tests, and honest `@test_broken` gaps.                       | The supplied source and test narrative are not synchronized; the duplicate include is inconsistent with a green load/test run; P1 tests do not test RSOC through the stack; some tests depend on include order; the fast gate omits several claimed cone families.           |
| Feature completeness                  | **6.7/10** | Direct support exists for LP/SOC/RSOC/PSD/Exp/Power plus free, zero, and signed orthants; HSD optimal and ray certificates; multiple arithmetic abstractions and solver-planning infrastructure. | The single native path lacks sparse execution, equilibration, ordinary presolve, warm starts, mature infeasibility classification, robust boundary termination, and a unified high-precision KKT route.                                                                      |
| Maintainability                       | **3.8/10** | Many local contracts are careful; numerical invariants are documented; types and failure enums are generally preferable to string-based control flow.                                            | Approximately 90.6k source lines, multiple full solver engines, several implementations of the same cone and KKT mathematics, large monolithic files, duplicate include, two incompatible transform hierarchies, and production reconstruction metadata using `Any`.         |

A nonincremental MOI adapter is not itself a defect: MOI explicitly supports optimizers populated through `copy_to`, commonly behind a `CachingOptimizer`, and does not require every solver to implement the full interface. The real API maturity gaps are incomplete declared cone coverage, limited conformance testing, and inconsistent introspection. ([JuMP][1])

---

## 1. Correctness and certificate authority — 7.2/10

The strongest part of SDPX remains its result boundary.

### Strong evidence

`src/public/optimize.jl:_public_original_certificate` at approximately lines 578–703 independently checks:

* original variable-block primal cone membership;
* original affine-row cone membership;
* dual cone membership;
* original stationarity

  $$
  c-A^\top y-s=0;
  $$
* scaled primal residual;
* scaled dual residual;
* relative objective gap;
* finiteness.

`src/public/optimize.jl:_public_result_from_core` downgrades a core `Optimal` result to `NumericalFailure` if this check fails.

For the direct HSD route, `src/hsd/native_hsd_public.jl:_public_result_from_native_hsd` is stronger still: it independently verifies all three promotable outcomes:

* `Optimal`;
* `PrimalInfeasible`;
* `DualInfeasible`.

The canonical verifiers in `src/certificates/certificates.jl` also normalize and recheck rays before reconstruction. That is the correct authority model.

### Score caps

The exact archived tree has a likely load blocker at `src/SDPX.jl:25,35`, and there are current false-negative correctness paths:

* wrong trial μ for Exp/Power;
* direct nonsymmetric optimal verification disabled;
* verified terminal result discarded if mutable scaling restoration fails;
* dual-infeasible elementary case still returns numerical breakdown;
* P1 transforms are not yet the production reconstruction authority.

These defects are not known false-positive certificate escapes, which is why the score remains above 7. They do prevent a production-correctness score above about 7.5.

---

## 2. Numerical robustness — 5.8/10

A1 changes the assessment materially.

### Strong evidence

The production-shaped PSD runtime implementation in `src/cones/runtime/symmetric_api.jl:_runtime_try_nt!(::PSDRuntimeBlock)` now constructs

$$
Y=LL^\top,\qquad M=L^\top S L,\qquad
P=L^{-\top}M^{1/2}L^{-1}.
$$

This is algebraically correct because

$$
PYP
=
L^{-\top}M^{1/2}L^{-1}
LL^\top
L^{-\top}M^{1/2}L^{-1}
=
L^{-\top}ML^{-1}
=
S.
$$

The implementation then checks the construction in the Cholesky frame rather than relying solely on the ill-conditioned triple product \(PYP\). I found no sign or orientation error in:

* `_runtime_psd_cholesky!`;
* `_runtime_chol_congruence_solve!`;
* the `core^2=M` check;
* the `L'PL=core` check;
* the root/inverse-root checks;
* construction of the scaled `Lambda`.

The terminal/current verification logic in `src/hsd/product_cone_solve.jl` is also an important improvement. A failed next direction no longer automatically invalidates the current accepted iterate.

### Score caps

The Newton-system architecture is still the limiting factor:

* `src/hsd/equality_reduction.jl` performs dense equality reduction;
* `src/hsd/hsd.jl:_hsd_rowspace_reduction` performs another dense RRQR-based reduction;
* `src/hsd/product_cone_hsd.jl:_product_hsd_form_schur_border!` forms normal equations;
* `_product_hsd_assemble_bordered!` creates a generally nonsymmetric scalar border and performs one-sided row scaling;
* `_product_hsd_factor_bordered!` performs one generic LU without a mature regularization/recovery ladder.

This is not yet comparable with standard production practice. Clarabel, for example, enables data equilibration, presolve, static and dynamic KKT regularization, and iterative refinement by default. ([Oxford Control][2])

SDPT3 similarly emphasizes data-scaled initial iterates, detection of nearly dependent constraints, and density/conditioning-sensitive choices of sparse or dense factorizations. 

---

## 3. Performance and scale — 4.2/10

The repository contains good performance technology, but the production candidate HSD engine does not yet use most of it.

### Good infrastructure already present

* `src/factor_cache/`: symbolic/numeric ownership, epochs, factor reuse.
* `src/kkt_route.jl`: one-factor/multiple-RHS design.
* `src/kernels/threaded.jl`: deterministic cost-weighted block scheduling, unique output ownership, BLAS/thread coordination.
* `src/kernels/mixed_precision_kkt.jl`: condition-based route and refinement estimates.
* `src/sparse_la.jl`: sparse execution concepts.
* `src/kernels/extended_precision_blas/`: fixed-width high-precision kernels.
* `src/soc_native.jl`: local Q3 elimination and inertia-aware augmented factorization.

### Current native-HSD limitations

`src/hsd/native_hsd_public.jl:_public_validate_native_hsd_policy` explicitly states that native HSD:

* does not equilibrate;
* does not run ordinary presolve;
* does not execute sparse;
* supports only the built-in serial provider;
* does not expose BLAS thread control.

`HSDState` retains:

* sparse \(A\);
* sparse \(A^\top\);
* dense \(A\);
* reduced \(A_r\);
* sparse \(A_r^\top\);
* dense rank basis;
* dense Schur matrix;
* several full-sized residual and direction buffers.

In addition, `_product_bordered_factor_certificate!` reconstructs the LU product with three nested loops. That is useful as a test oracle but can approach factorization cost if run every epoch.

The principal speed task is therefore not micro-optimizing cone loops. It is connecting HSD to a sparse, reusable, regularized KKT layer and deleting duplicate state.

---

## 4. API and MOI conformance — 6.3/10

The direct modeling API is useful and substantially implemented. The MOI adapter is also more complete than a typical experimental wrapper.

### Strengths

`src/moi_wrapper.jl` supports:

* nonincremental `copy_to`;
* scalar affine equalities and bounds;
* intervals;
* vector affine and variable forms;
* Nonnegative and Nonpositive cones;
* Zero and Reals;
* SOC and RSOC;
* PSD triangle and scaled PSD affine forms;
* starts;
* termination/primal/dual statuses;
* objectives and relative gap;
* variable and constraint primal/dual values;
* iteration and solve-time attributes.

### Concrete gaps

1. **Exp/Power absent from MOI.**
   `MOIVectorConicSet` deliberately excludes them at `src/moi_wrapper.jl:276–286`.

2. **Targeted rather than complete supported-surface testing.**
   `test/moi_conformance.jl:75–114` runs a manually selected subset.

3. **Rank-one PSD tests excluded.**
   The test comments at approximately lines 81–87 acknowledge the current boundary failure.

4. **Variable attribute introspection is incorrect.**
   `src/moi_wrapper.jl:1985` always returns an empty `ListOfVariableAttributesSet`, even though `VariablePrimalStart` is supported and copied.

5. **Native starts are rejected.**
   The direct native route rejects both a supplied `warm_start` and starts stored in the model.

6. **Public settings exceed native capabilities.**
   The user can express scaling, sparse, provider, equality-solver, history, trace, and thread settings that native HSD then rejects.

For v1, keep the one-shot interface. Do not spend time implementing incremental mutation. Correct the declared support surface and run all applicable `MOI.Test` cases for that surface.

---

## 5. Testing culture — 8.0/10

This is one of the strongest parts of the project.

The supplied tree has approximately:

* 139 Julia test files;
* 53,800 test lines;
* 1,063 `@testset` occurrences;
* 9,523 `@test` occurrences;
* 9 `@test_broken`;
* 6 skipped tests.

The tests include:

* algebraic cone identities;
* arbitrary-precision routes;
* independent reference kernels;
* allocation checks;
* public result reconstruction;
* optimal and infeasibility certificates;
* MOI conformance;
* factor-cache epochs;
* sparse and threaded kernels;
* regression tests for known numerical failures.

Honest `@test_broken` coverage is preferable to hiding a failure.

### Why it is not yet 9/10

The exact archive and claimed run evidence are inconsistent in several places:

* `src/SDPX.jl` includes `program/transforms.jl` twice;
* `test/quick_gate.jl:10–14` still calls mixed free/PSD a known failure;
* `test/kernel_failure_regressions.jl:206–211` still marks a mixed free/equality/PSD case broken;
* the same file records elementary dual infeasibility as expected numerical breakdown;
* the P1 RSOC tests test a standalone `AbstractConeTransform`, not composition through `ReconstructionStack`;
* the Nonpositive executable smoke test uses a zero objective, so it is weak evidence for stationarity and objective-sign reconstruction.

This is primarily a **test provenance and manifest problem**. Each release gate should state the exact source SHA, test manifest SHA, arithmetic type, and enabled known-gap flags.

---

## 6. Feature completeness — 6.7/10

At the mathematical-cone level, the direct API has unusually broad scope:

* free coordinates;
* equality rows;
* nonnegative/nonpositive orthants;
* SOC;
* RSOC;
* PSD triangle;
* exponential cone;
* power cone.

The repository also contains substantial multi-precision, sparse, threaded, MOI, chordal, presolve, and factor-cache machinery.

The score remains below 7 because many of those capabilities are not available simultaneously through the intended native HSD engine. A production feature is not merely code existing somewhere in the repository; it must be reachable through one supported route with coherent reconstruction, status, and test contracts.

---

## 7. Maintainability — 3.8/10

Individual code sections are frequently careful. The global architecture is not yet maintainable.

Largest duplicated solver files include:

* `src/lp_solver.jl`: approximately 4,521 lines;
* `src/kkt.jl`: approximately 4,300;
* `src/solver/interior_point.jl`: approximately 4,117;
* `src/schur.jl`: approximately 3,947;
* `src/soc_native.jl`: approximately 3,188;
* `src/hsd/product_cone_hsd.jl`: approximately 2,916.

The repository currently has separate or partially separate implementations of:

* orthant HSD;
* product-cone HSD;
* LP Mehrotra;
* general SDP Mehrotra;
* SOC-native iteration;
* several Schur/KKT assemblers;
* multiple cold starts;
* multiple cone-algebra layers;
* multiple factor/workspace hierarchies.

The P1 code itself currently adds a second transformation hierarchy rather than replacing the first. That is exactly the kind of additive transition that must be resolved quickly rather than allowed to persist for several phases.

---

# C. Review of the new A1/P1 code

## What is correct and should be preserved

### C1. Runtime Cholesky-congruence PSD NT

**Assessment: approve.**

The construction in `src/cones/runtime/symmetric_api.jl:_runtime_try_nt!` is mathematically correct and better conditioned than the previous explicit \(Y^{\pm1/2}\) construction.

Specific good choices:

* unpivoted Cholesky is used as a strict-interior predicate;
* the congruence is implemented through triangular solves;
* no inverse of \(Y\) is explicitly formed;
* orientation is checked in the Cholesky frame;
* square root and inverse square root use a shared frozen eigensystem;
* the runtime API returns `false` on expected numerical rejection rather than allocating exceptions during backtracking.

I found no formula change that alters the underlying NT scaling.

### C2. Terminal/current certificate verification

**Assessment: approve with changes.**

`src/hsd/product_cone_solve.jl:385–432` and the `HSDStepDirectionFailed` handling at approximately lines 520–538 correctly distinguish:

* inability to compute the next direction;
* validity of the current accepted iterate;
* validity of a finite terminal Newton trial.

This is the correct conceptual ordering.

### C3. Nonpositive transform

**Assessment: algebraically correct but not integrated.**

`NonpositiveToNonnegative` correctly uses

$$
\hat s=-s,\qquad \hat y=-y,\qquad
\hat A=-A,\qquad \hat b=-b.
$$

The dual map is the inverse adjoint of \(-I\), which is again \(-I\), so the pairing and stationarity checks are correct.

### C4. RSOC transform

**Assessment: mathematical convention correct.**

The orthogonal transform

$$
(u,v,w)\mapsto
\left(
\frac{u+v}{\sqrt2},
\frac{u-v}{\sqrt2},
w
\right)
$$

is the cleanest convention. The primal inverse and dual inverse-adjoint coincide because the map is symmetric, orthogonal, and involutory.

---

## Blocking findings before Phase 2

### B0. The supplied tree appears not to load

`src/SDPX.jl` contains:

```julia
include("program/transforms.jl")  # line 25
...
include("program/transforms.jl")  # line 35
```

The file defines:

```julia
abstract type AbstractProgramTransform{T} end
```

Including it twice in the same module normally attempts to redefine the type constant and is a hard package-load error.

This creates two possibilities:

1. the archived tree is not the exact tree used for the green test run; or
2. the duplicate was introduced during archive assembly.

Either way, treat it as a release-process blocker.

**Required fix**

* Remove the second include.
* Add an include-graph uniqueness test.
* Record the exact Git tree SHA in every quick-gate artifact.
* Generate a source manifest from the same tree that Julia loads.
* Reconcile stale known-gap comments against that SHA.

---

### B1. Nonsymmetric trial scaling uses the previous iterate’s μ

`src/hsd/product_cone_hsd.jl:_product_hsd_trial_scaling!` currently does:

```julia
try_update_scaling!(state.runtime, base.st, base.yt, base.mu)
```

But the line search has already formed:

```julia
base.tau_t
base.kappa_t
base.st
base.yt
```

For product HSD, the trial complementarity observable is

$$
\mu_t=
\frac{\langle s_t,y_t\rangle+\tau_t\kappa_t}{\nu+1}.
$$

The nonsymmetric scaling contract explicitly states that Exp/Power production code must use the single global embedding μ. The runtime also requires `scaling.mu == expected_mu`.

Therefore the trial scaling is being built at the **old μ**, not at the trial point’s μ. After acceptance, `state.runtime.last_mu` is overwritten with the newly recomputed `base.mu`, but the block metrics themselves were constructed with the old value.

This can cause:

* incorrect line-search acceptance or rejection for Exp/Power;
* runtime metadata claiming a μ different from that used to construct the metric;
* an accepted point failing immediately when scaling is recomputed at the real μ;
* path-dependent behavior hidden by rebuilding on the next iteration.

**Required fix**

Compute before the scaling call:

```julia
mu_t = (dot(base.st, base.yt) + base.tau_t * base.kappa_t) /
       T(base.nu + 1)
```

and pass `mu_t`.

On acceptance, require:

```julia
state.runtime.last_mu == base.mu
```

and, for each nonsymmetric block:

```julia
block.scaling.mu == base.mu
```

Add a regression in which \(\mu_t/\mu\) is materially different from one, so the test cannot pass accidentally.

This is the most important mathematical bug to fix before Phase 2.

---

### B2. P1 currently has two incompatible transform systems

`src/program/transforms.jl` defines:

```julia
abstract type AbstractProgramTransform{T} end
```

and `ReconstructionStack{T}` stores:

```julia
Vector{AbstractProgramTransform{T}}
```

Later in the same file it defines a second hierarchy:

```julia
abstract type AbstractConeTransform{T<:AbstractFloat} end
```

`RotatedSOCToSOC` is a subtype of `AbstractConeTransform`, not `AbstractProgramTransform`.

Consequently:

* `RotatedSOCToSOC` cannot be pushed into `ReconstructionStack`;
* the pairing-invariant method signatures differ;
* the stationarity-invariant method signatures differ;
* P1 does not yet provide a single reconstruction contract.

The source comments acknowledge this as provisional, but Phase 2 should not be built on top of a provisional split interface.

#### Deeper design issue: the stack is dimension-preserving only

`_stack_forward!` and `_stack_backward!` allocate both intermediate buffers using:

```julia
similar(dest)
```

That works for Nonpositive and RSOC, but not for future transformations that change dimensions:

* equality elimination;
* free-variable splitting;
* singleton substitution;
* duplicate-row removal;
* chordal decomposition;
* fixed-variable elimination.

The previous architecture plan expected the same stack eventually to own those transforms.

**Required fix**

Choose one of two clean designs:

**Preferred for a solo developer**

* `AbstractCoordinateTransform`: dimension-preserving local maps such as Nonpositive and RSOC.
* `AbstractProgramReduction`: dimension-changing presolve/equality transformations with explicit source and target dimensions.

Both implement the same reconstruction semantics, but do not pretend to have identical scratch requirements.

Alternatively, make every transform declare:

```julia
source_primal_dimension
target_primal_dimension
source_dual_dimension
target_dual_dimension
scratch_requirements
```

and let the stack create a setup-time scratch plan.

Also add stack-level objective-constant composition. `objective_shift(transform)` exists, but no `objective_shift(::ReconstructionStack)` currently composes it.

---

### B3. A verified terminal result can be discarded during state restoration

In `_product_hsd_terminal_verified_result!`:

1. the trial point is installed;
2. `_product_hsd_candidate_result!` may create a verified result;
3. `_product_hsd_make_result` copies the original and HSD trial coordinates;
4. the mutable state is restored;
5. `try_update_scaling!` is called on the restored accepted point;
6. if that restoration fails, the verified output arrays are zeroed and the function returns `nothing`.

Once `_product_hsd_make_result` has copied a verified result, failure to restore a mutable workspace must not invalidate it. The solver is about to return; the result no longer depends on the reusable runtime.

**Required fix**

* Return the already-created immutable result.
* Mark `state.runtime.valid=false` if restoration fails.
* Record a diagnostic such as `:post_result_state_restore_failed`.
* Prefer explicit runtime checkpoints so restoration normally cannot fail.

This is a false-negative availability bug, not a false-certificate bug.

---

### B4. Direct optimal verification is skipped for Exp/Power

`_product_hsd_candidate_result!` currently:

1. runs `_product_hsd_refined_optimal_result!`;
2. if that fails, calls `_product_hsd_verified_result` with

   ```julia
   check_optimal = !_product_hsd_has_nonsymmetric(state)
   ```

Thus, with Exp or Power blocks, a directly valid accepted HSD iterate is **never checked for optimality** unless the ad hoc dense affine refinement succeeds first.

But `verify_optimal!` already supports Exp and Power through `in_canonical_cone`.

**Required ordering**

1. Run the unchanged direct verifier for all cone families.
2. If it fails and the point is sufficiently close, attempt refinement.
3. Re-run the same verifier after refinement.
4. Then test infeasibility rays as appropriate.

The current ordering cannot create a false `Optimal`, but it can miss valid nonsymmetric solutions.

---

## Important findings to address during or immediately before Phase 2

### I1. `dkappa` recovery is not always well conditioned

`_product_hsd_recover_dkappa!` divides by `base.tau`:

```julia
base.dkappa =
    (scalar_rhs - base.kappa * base.dtau) / base.tau
```

Strict positivity of \(\tau\) does not imply good conditioning. Near an infeasibility certificate, \(\tau\) can be positive but extremely small.

Do not describe the equation as categorically “well-conditioned.”

**Phase 2 fix**

Include \(\Delta\tau\) and \(\Delta\kappa\) in the unified scalar border, or solve the two scalar equations with scaled pivoting. If post-recovery remains:

* compute candidates from both scalar and gap equations;
* evaluate both in the unregularized five-equation residual;
* choose the lower-residual candidate;
* fail into regularization or precision escalation when the denominator is unresolved.

---

### I2. Family-gated backtracking encodes termination behavior indirectly

The line search chooses:

* 16 trials for LP/SOC;
* 64 for PSD/Exp/Power.

The accompanying comment says that increasing the pure-SOC budget caused iteration-limit behavior where the shorter budget had converged via terminal verification.

Increasing only the maximum number of trials cannot change any decision within the first 16 trials. Therefore the behavioral difference means:

* at 16 trials, line-search exhaustion triggers terminal verification;
* at 64 trials, a much smaller step is eventually accepted;
* those tiny accepted steps then prevent or delay termination.

This is a control-flow smell. Backtracking count is functioning as an implicit stall detector.

**Replace with**

* one conservative hard backtracking limit;
* minimum useful step/progress criterion;
* explicit neighborhood criterion;
* terminal certificate check when progress becomes negligible;
* centering restoration;
* route or precision recovery.

The current family gate is acceptable as a temporary regression patch, but should not survive the new HSD loop.

---

### I3. RSOC application is not alias-safe

`_rsoc_transform_apply!` writes one row at a time while reading `source`. If:

```julia
destination === source
```

the first output overwrites `source[1]` before the second row reads it.

Either:

* explicitly reject aliasing; or preferably
* save `u=source[1]` and `v=source[2]` before writing.

Also replace the four dense \(n\times n\) matrices with:

```julia
dimension
inv_sqrt_two
```

and an \(O(n)\) transform. The current implementation uses \(O(n^2)\) work and four \(O(n^2)\) allocations for a map with only four nontrivial coefficients.

Additional RSOC cleanup:

* `_rsoc_transform_precision_bits` returns 53 for every non-BigFloat type; that is wrong metadata for Float32 and fixed-width multi-floats.
* `pairing_scale` should be exactly one whenever the dual map is the true inverse adjoint. A nonunit scale would indicate that the dual transform contract is wrong.

---

### I4. PSD scaling logic remains duplicated

The Cholesky-congruence construction now exists separately in:

* `src/cones/runtime/symmetric_api.jl`;
* `src/cones/symmetric/psd.jl:nt_scaling!`.

They differ in:

* whether explicit \(L^{-1}\) is formed;
* which orientation checks are used;
* exception versus Boolean behavior;
* failure diagnostics.

The new `NewtonSystem` must not accidentally call the older throwing implementation and reintroduce the rejected numerical path.

**Required refactor**

Create one internal nonthrowing numerical kernel:

```julia
try_psd_nt_scaling!(state, s, y) -> PSDNTResult
```

where `PSDNTResult` contains a structured reason such as:

* `CholeskyFailed`;
* `CoreSqrtFailed`;
* `FactorizedOrientationFailed`;
* `RootInverseFailed`;
* `LambdaEigenFailed`.

Then:

* runtime HSD consumes the result directly;
* a public/testing wrapper may turn it into a `DomainError`;
* legacy code is removed later.

`_runtime_psd_nt_fail(::Symbol,::Int)=false` currently discards exactly the diagnostic information Phase 2 recovery will need.

---

### I5. Avoid catching every exception around the PSD eigensolver

The runtime code uses a broad `try/catch` around `_psd_eigen_route!` and converts every exception to `false`.

That can hide:

* bounds errors;
* method errors;
* workspace dimension bugs;
* invalid mutations introduced during refactoring.

Only expected numerical nonconvergence should become a fail-closed numerical result. Programmer errors should propagate.

---

### I6. Ray normalization ignores the caller’s explicit tolerance

In `src/certificates/certificates.jl`:

```julia
dot(b, state.yt) ≈ -one(T)
dot(c, state.xt) ≈ -one(T)
```

uses Julia’s default `isapprox`, not the caller’s certificate tolerance.

Replace both with explicit tolerance checks tied to `tol`. Certificate gates should have one visible tolerance authority.

---

### I7. Production transform migration must replace `CanonicalBlockMap`

The current production reconstruction still uses `src/ir/layout.jl:CanonicalBlockMap` with:

```julia
linear::Any
linear_adjoint::Any
coordinate_map::Any
```

and `src/ir/canonical.jl` directly constructs RSOC matrices and sign metadata.

P1 should replace this path, not coexist indefinitely beside it. Otherwise there will be:

* one transform used in tests;
* another transform used by production lowering;
* a third reconstruction path used by certificates.

---

## Pre-Phase-2 acceptance patch

Before merging substantial `NewtonSystem` work, require all of the following:

* [ ] `using SDPX` succeeds from the exact archived source tree.
* [ ] `program/transforms.jl` is included exactly once.
* [ ] One abstract transform hierarchy remains.
* [ ] RSOC can be pushed through the production reconstruction stack.
* [ ] Transform composition has explicit dimensions and objective-shift behavior.
* [ ] Trial Exp/Power scaling uses \(\mu_t\), with block-level μ invariants tested.
* [ ] A directly valid Exp/Power optimum is tested before refinement.
* [ ] A verified terminal result survives restoration failure.
* [ ] Ray normalization uses the requested certificate tolerance.
* [ ] Source SHA, test-manifest SHA, and known-gap flags appear in the quick-gate report.
* [ ] Stale mixed-free/PSD known-gap declarations are reconciled with the authoritative test result.

---

# B. Priority gaps to a mature production solver

Effort below means **focused solo-developer effort**, with AI agents producing drafts, tests, and independent reviews. Numerical architecture integration and acceptance remain serial and usually dominate.

| Priority | Work package                                                        | Main files                                                                                                   |                          Focused effort |           Required for v1           |
| -------: | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------: | :---------------------------------: |
|        0 | Stabilization and provenance                                        | `src/SDPX.jl`, `program/transforms*`, `product_cone_hsd.jl`, `product_cone_solve.jl`, certificate tests      |                            **3–5 days** |                 Yes                 |
|        1 | Production transform integration                                    | `program/transforms.jl`, `ir/canonical.jl`, `ir/layout.jl`, `ir/reconstruction.jl`, lowerers                 |                       **1.5–2.5 weeks** |                 Yes                 |
|        2 | Unified Newton equations and dense expanded KKT                     | new `kkt/system.jl`, `expanded_quasidefinite.jl`, `regularization.jl`, `refinement.jl`; factor-cache/LA APIs |                           **3–5 weeks** |                 Yes                 |
|        3 | Equilibration, KKT-derived start, equality policy, minimal presolve | `cold_start.jl`, `hsd/equality_reduction.jl`, new program passes                                             |                           **2–3 weeks** |                 Yes                 |
|        4 | Migrate product HSD to new KKT/recovery loop                        | split `product_cone_hsd.jl`, `product_cone_solve.jl`; unify nonsymmetric route                               |                           **2–3 weeks** |                 Yes                 |
|        5 | One production sparse high-precision KKT route                      | `sparse_la.jl`, factor cache, one selected backend, symbolic reuse                                           |                           **4–7 weeks** | Yes for bootstrap-scale credibility |
|        6 | MOI and release hardening                                           | `moi_wrapper.jl`, conformance tests, public settings/statuses                                                |                           **1–2 weeks** |                 Yes                 |
|        7 | Retire legacy engines and remove duplicate math                     | `lp_solver.jl`, `solver/interior_point.jl`, `soc_native.jl`, old KKT/Schur/state files                       |                           **1–2 weeks** |         Yes for HSD-only v1         |
|        8 | Mature-solver extras                                                | chordal, warm updates, automatic precision ladder, advanced presolve, multiple providers                     | **20–35 additional weeks collectively** |                  No                 |

## Priority 0 — stabilization

This should happen immediately.

Deliverables:

* remove duplicate include;
* unify P1 interfaces;
* fix trial μ;
* fix terminal-result ownership;
* correct nonsymmetric certification ordering;
* explicit ray normalization tolerance;
* exact test/source provenance.

Rollback boundary: one small commit before the Phase 2 branch is rebased.

---

## Priority 1 — production transformation chain

The transform layer must become the only owner of:

* source-to-canonical primal maps;
* inverse primal reconstruction;
* dual inverse-adjoint maps;
* primal-infeasibility-ray reconstruction;
* dual-infeasibility-ray reconstruction;
* objective constants;
* dimension changes.

Do not retain `CanonicalBlockMap` as a second long-term authority.

Acceptance gates:

* pairing invariance;
* stationarity invariance;
* objective invariance;
* primal and dual ray invariance;
* dimension-changing composition;
* complete round-trip through actual model compilation;
* original-coordinate public certificate.

---

## Priority 2 — `NewtonSystem` plus expanded quasidefinite KKT

This remains the single highest-value architectural package.

It should introduce:

1. One semantic definition of the HSD Newton equations.
2. One `NewtonRHS` representation for predictor, corrector, and restoration directions.
3. An expanded symmetric-indefinite or quasidefinite KKT route.
4. Signed static regularization.
5. Dynamic pivot regularization.
6. Expected inertia checks.
7. One factorization for multiple RHS.
8. Iterative refinement against the **unregularized** equations.
9. Typed failures and route recovery.

Do not make the reduced normal-equation route the default for free/equality/PSD mixtures.

Clarabel’s documented defaults illustrate the minimum production pattern: equilibration, static and dynamic KKT regularization, iterative refinement, presolve, and selectable direct KKT solvers are normal solver infrastructure rather than optional sophistication. ([Oxford Control][2])

---

## Priority 3 — equilibration, initialization, equalities, presolve

Implement:

* cone-preserving Ruiz-style equilibration;
* KKT-derived primal/dual affine starts;
* strict-interior cone shifts;
* the existing PSD continuation repair;
* identity-mass floor;
* primal/dual cross-centering;
* a single numerical rank decision after scaling;
* retained equalities by default for mixed PSD systems;
* minimal reversible presolve.

The current unconditional identity start should remain only as an emergency diagnostic option.

SDPT3’s documented initialization scales identity/unit starts using problem data precisely because arbitrary unscaled identity starts are sensitive to data magnitude. It also explicitly detects nearly dependent constraints rather than blindly condensing them. 

---

## Priority 4 — unified HSD recovery and termination

Replace the current collection of special termination paths with a clear recovery ladder:

1. Verify current accepted point.
2. Verify finite terminal trial.
3. Increase centering/restoration.
4. Increase KKT regularization.
5. Switch reduced/expanded KKT route.
6. Refine the direction.
7. Promote arithmetic precision if enabled.
8. Return a typed unverified failure.

Carry separate tolerances for:

* primal feasibility;
* dual feasibility;
* absolute gap;
* relative gap;
* ray residuals;
* \(\kappa/\tau\);
* Newton residual;
* cone margin.

`_native_hsd_tol` currently collapses primal, dual, and gap tolerances to their minimum. That should disappear.

---

## Priority 5 — one sparse high-precision KKT route

For a general conic solver, dense-only can be called a useful reference implementation. For a solver aimed at numerical bootstrap, dense-only cannot credibly be called v1.0.

Choose exactly one sparse direct route initially:

* one ordering strategy;
* one symbolic-analysis cache;
* one numeric factorization interface;
* one regularization contract;
* one iterative-refinement implementation;
* one high-precision arithmetic backend.

Do not attempt to support every declared provider at v1.

The route planner can initially be simple:

* dense below a setup-estimated dimension/fill threshold;
* sparse otherwise;
* fixed-trace Q3 specialization disabled until after release.

SDPT3 likewise makes sparse/dense factorization choices based on system density and explicitly warns that reduced systems become severely ill-conditioned in difficult cases. 

---

## Priority 6 — MOI and release hardening

For v1:

* retain nonincremental `copy_to`;
* fix attribute introspection;
* run all applicable `MOI.Test` cases for every declared supported function/set pair;
* remove declarations for any route not green;
* expose Exp/Power through MOI only after their full status/result tests pass;
* unify public settings with actual HSD capabilities;
* document unsupported starts and updates explicitly.

Do not implement incremental model mutation merely to improve an abstract completeness score.

---

## Priority 7 — legacy deletion

Delete legacy execution engines only after the new route passes the same certificate gates.

The deletion gate should require:

* one reachable HSD solve loop;
* one semantic Newton equation implementation;
* one certificate authority;
* no automatic legacy fallback;
* no legacy engine included or compiled;
* no production reference kernel included merely for testing;
* quick gate under 60 seconds;
* all claimed cone/status cases green.

Move `src/cones/nonsymmetric/full_newton_reference.jl` and any independent reference cone algebra into `test/reference/`, unless production code genuinely depends on them.

---

## Gaps the previous Phases 1–8 plan did not state explicitly enough

| Newly visible requirement                                                              | Insert into       |
| -------------------------------------------------------------------------------------- | ----------------- |
| Exact-tree/source-SHA and include-uniqueness gate                                      | Phase 0           |
| Separation of local dimension-preserving transforms from dimension-changing reductions | Phase 1           |
| Stack-level objective-constant composition                                             | Phase 1           |
| Removal of production `CanonicalBlockMap`/`Any` reconstruction authority               | Phase 1           |
| Trial-point global μ consistency for Exp/Power                                         | Phase 0 / Phase 4 |
| Transactional cone-scaling updates and structured block failure reasons                | Phase 2 / Phase 4 |
| Immutable verified-result ownership independent of mutable runtime restoration         | Phase 4           |
| Tiny-\(\tau\) conditioning policy for \(\Delta\kappa\)                                 | Phase 2           |
| Direct nonsymmetric certificate before ad hoc refinement                               | Phase 0           |
| Explicit sparse-backend and licensing/portability decision                             | Phase 2 / Phase 5 |
| Audit of extension modules declared in `Project.toml`                                  | Phase 5 / release |
| Thread-safe BigFloat precision ownership and task-local scratch                        | Phase 5 / Phase 6 |
| Full feasible/primal-infeasible/dual-infeasible status matrix per claimed cone family  | Phase 4 / Phase 6 |
| MOI attribute-list and support-matrix correctness                                      | Phase 6           |
| Removal of test-reference implementations from production includes                     | Phase 7           |
| Replacement of family-based line-search limits with progress/neighborhood logic        | Phase 4           |
| Separate primal, dual, gap, ray, and \(\kappa/\tau\) tolerances                        | Phase 4           |
| Test manifest synchronization; no stale `@test_broken` for a claimed fixed case        | Every phase       |

---

# D. Fastest credible v1.0 for a solo developer

The fastest credible v1 is **not** the full mature-solver target. It should be a deliberately narrow, reliable product.

## Minimal v1.0 support contract

### Problem class

Linear-objective affine conic programs:

$$
\min c^\top x+c_0
\quad\text{subject to}\quad
Ax+s=b,\qquad s\in K.
$$

Do not add quadratic objectives for v1.

### Guaranteed cone surface

* `Reals`;
* `ZeroCone`;
* `Nonnegative`;
* `Nonpositive` through an exact transform;
* SOC;
* RSOC through an exact orthogonal transform;
* PSD triangle.

For Exp and Power:

* keep direct `Model` support only after the trial-μ bug is fixed and their release status matrix is green;
* otherwise label them experimental for the first release candidate;
* defer MOI exposure until standard MOI tests and infeasibility/result queries pass.

### Required numerical infrastructure

* one product-cone HSD engine;
* dense expanded KKT;
* one sparse expanded KKT;
* cone-preserving equilibration;
* KKT-derived cold start;
* signed regularization;
* iterative refinement against unregularized equations;
* strict original-coordinate optimal and ray certificates;
* terminal verification near rank-one optima;
* minimal reversible presolve;
* user-selected arithmetic and precision.

### Arithmetic support

Because arbitrary precision is central to the package identity:

* `Float64` must be fully supported;
* `BigFloat` must remain a correctness-supported arithmetic;
* exactly one sparse high-precision production route should be guaranteed;
* MultiFloat variants beyond the selected production type may remain experimental;
* automatic precision escalation is not required for v1.

A solver that supports BigFloat only through a dense reference path may be useful, but should state its scale limit explicitly rather than imply production-scale arbitrary-precision sparsity.

### API support

* direct `Model` API;
* one-shot MOI `copy_to`;
* complete MOI tests for the claimed surface;
* deterministic status and certificate queries;
* explicit unsupported-feature errors.

No incremental MOI model editing is needed.

---

## What to cut or defer

### Cut from the v1 critical path

1. **Fixed-trace Q3 local elimination.**
   Preserve the code and tests, but do not integrate it before the generic KKT path is stable.

2. **Automatic precision escalation.**
   Let the user choose `Float64`, selected MultiFloat, or BigFloat.

3. **Chordal decomposition.**

4. **Facial reduction.**

5. **Warm starts and persistent data updates.**

6. **Multiple sparse providers.**
   Support one.

7. **Elaborate adaptive route planning.**
   Use a deterministic setup-time dense/sparse rule.

8. **Custom block threading across every cone.**
   Initially rely on the chosen factorization backend’s threads. Port the deterministic block scheduler after v1.

9. **Gondzio multi-corrector loops.**
   One predictor, one corrector, and at most one restoration RHS are enough initially.

10. **Separate primal and dual step lengths.**

11. **Full iteration history and performance tracing.**

12. **MOI Exp/Power exposure unless fully green.**

13. **Incremental direct-model editing and deletion.**

14. **QP objectives.**

15. **GPU, distributed, and matrix-free routes.**

16. **Advanced presolve such as general facial reduction or 2×2 PSD-to-RSOC conversion.**

### Do not cut

* typed primal/dual/ray reconstruction;
* original-coordinate certificate authority;
* expanded KKT;
* KKT regularization and refinement;
* equilibration;
* data-dependent initialization;
* sparse execution;
* rank-one boundary termination;
* both infeasibility ray types;
* strict failure statuses;
* exact source/test provenance;
* no-tolerance-loosening policy.

### Relax the zero-allocation goal carefully

For v1:

* require zero heap allocation in the steady-state fixed-width Float64/high-precision hot path after setup;
* for BigFloat, require bounded reusable workspaces and no structural allocations proportional to iteration count;
* do not delay release merely to eliminate every MPFR-internal scalar allocation.

Correctness and sparse KKT robustness are worth much more than a perfect allocation score in the first release.

---

## Minimal v1 release gates

A v1 release candidate should require:

* [ ] Exact source SHA and test-manifest SHA are recorded.
* [ ] Package-load and precompile smoke tests pass.
* [ ] Only one HSD engine is reachable.
* [ ] Every claimed cone family has feasible and near-boundary optimal regressions.
* [ ] Every claimed status surface has primal-infeasible and dual-infeasible regressions where applicable.
* [ ] No `@test_broken` remains inside the claimed support matrix.
* [ ] Every successful status passes original-coordinate verification.
* [ ] No tolerance is loosened relative to the requested settings.
* [ ] The `<60 s` kernel quick gate remains green.
* [ ] One medium sparse scale gate stays within a declared memory budget.
* [ ] All applicable MOI tests for the declared surface pass.
* [ ] Float64 and the chosen high-precision production arithmetic pass the same semantic regression set.
* [ ] Legacy Mehrotra engines are neither compiled nor used as hidden fallbacks.
* [ ] Unsupported features are listed explicitly as v1 non-goals.

---

# Recommended immediate sequence

1. **Create a small “P1.5 stabilization” commit** fixing:

   * duplicate include;
   * exact-tree provenance;
   * trial μ;
   * transform hierarchy;
   * dimension/scratch contract;
   * terminal-result restoration semantics;
   * direct Exp/Power verification ordering;
   * explicit ray-normalization tolerance.

2. **Re-run only the package-load, A1, P1, certificate, known-failure, and quick-gate suites.**

3. **Freeze the `NewtonSystem` semantic equations before implementing any factorization route.**

4. **Implement dense expanded KKT first**, with regularization, inertia checking, multi-RHS reuse, and unregularized refinement.

5. **Use that same KKT layer for the cold start** before migrating the iteration loop.

6. **Add one sparse high-precision route**, then retire legacy engines.

A credible narrow v1 is approximately **10–14 focused developer-weeks** if the existing factor-cache and high-precision sparse backend integrations can be reused without redesign; it is more realistically **14–20 weeks** if the sparse arbitrary-precision factorization contract has to be rebuilt. The full mature-solver target—automatic precision escalation, chordal decomposition, warm updates, multiple providers, advanced presolve, and specialized assembly—adds roughly **20–35 focused weeks** beyond that.

**Final verdict: proceed toward Phase 2, but only after the P1.5 stabilization patch.** The mathematical direction is now good; the largest near-term risk is no longer PSD NT scaling, but allowing transitional interfaces and control-flow patches to become permanent architecture.

[1]: https://jump.dev/MathOptInterface.jl/stable/tutorials/implementing/?utm_source=chatgpt.com "Implementing a solver interface · MathOptInterface - JuMP-dev"
[2]: https://oxfordcontrol.github.io/ClarabelDocs/stable/api_settings/?utm_source=chatgpt.com "Solver Settings · Clarabel jl/rs"
