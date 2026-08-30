# SDPX.jl roadmap to 1.0

**Status:** active implementation plan

**Updated:** 2026-08-29
**Authority:** this file describes current work. Frozen mathematical contracts
live in `docs/design/`; historical plans remain available in Git history and in
the local project archive.

## 1. Completion definition

SDPX 1.0 is complete when all of the following hold:

1. one product-cone homogeneous self-dual (HSD) state machine is the only
   production solver engine;
2. one five-equation `NewtonSystem` defines every KKT route;
3. LP, SOC, RSOC, PSD, exponential, power, free, zero, nonnegative, and
   nonpositive blocks share the same canonical program and certificate path;
4. every terminal status is justified by an original-coordinate optimality or
   infeasibility certificate;
5. Float64, MultiFloat, and BigFloat execution never silently narrows values;
6. KKT/linear-algebra providers are replaceable without changing the Newton
   equations;
7. the public API, MOI adapter, CLI, general benchmark, and physics/bootstrap
   benchmark all call the same production engine;
8. no hidden legacy solver or PSD-lift fallback remains; and
9. the black-box E2E, provider checks, benchmark campaigns, cluster runs, and
   final independent reviews pass on one frozen source SHA.

No test may be deleted or weakened merely to obtain a green result. A provider
factorization receipt is implementation evidence, not a mathematical
certificate.

## 2. Frozen architecture

### Canonical problem

All frontends lower to

```text
minimize    c'x
subject to  A*x + s = b
            s in K
```

where `K` is an ordered product of native cone blocks. Exact transforms retain
an inverse reconstruction map. Approximation or relaxation transforms must be
labelled and may not produce an unconditional certificate for the original
problem.

### Product-cone HSD

The production state is `(x,s,y,tau,kappa)`. Predictor, corrector, recovery,
termination, and certificate generation use the sign conventions frozen in
`docs/design/HSD_FORMULATION.md`.

Raw `tau`, `kappa`, iteration limits, factorization success, or provider status
never promote an `Optimal`, `PrimalInfeasible`, or `DualInfeasible` result.
Only the original-coordinate verifier may do so.

### Newton system and KKT routes

`docs/design/NEWTON_SYSTEM.md` defines the five equations. Implementations currently expose:

- `:bordered` — conservative public default, still using the proven legacy
  full-border execution on the public path;
- `:expanded` — exact nonsymmetric expanded solve;
- `:sparse_schur` — reduced sparse Schur solve with same-iterate fallback.

The integration branch also contains an internal, opt-in Clarabel-style
symmetric augmented core

```text
K = [ 0   Ar'
      Ar -Theta ]
```

with one factor epoch, one homogeneous solve, sequential predictor/corrector
RHS solves, scalar `dτ` recovery, original-K refinement, and the frozen
five-equation gate. It is not yet the public default: the attempted public
switch was reverted pending final E2E and diagnostics/allocation cleanup.
The exact expanded and legacy bordered operators remain nonsymmetric and must
never be passed to LDL.

### Provider ownership

- SuiteSparse/UMFPACK: Float64 sparse exact solves;
- MultiFloatLinearAlgebra and BigFloatLinearAlgebra: high-precision dense and
  local-block
  factor/solve providers;
- SDPX: canonicalization, cone algebra, assembly, route policy, refinement,
  fallback, reconstruction, and certification.

MFLA/BFLA kernels are not copied into SDPX. LinearSolve/SciMLBase are retired.

## 3. Completed foundations

The local integration line currently contains:

- typed canonical storage and reconstruction transforms;
- product-cone runtime for all claimed cone families;
- PSD congruence/NT scaling, boundary steps, and panel kernels;
- frozen `NewtonSystem`, bordered and expanded sessions, refinement, and
  backward-error gates;
- equality reduction, presolve maps, optional Ruiz equilibration, and KKT cold
  starts;
- unified predictor/corrector, line search, recovery, and termination;
- sparse reduced-Schur session, pattern/epoch ownership, block incidence maps,
  fallback receipts, and phase timings;
- fail-closed nonfinite/tolerance/certificate checks;
- MFLA/BFLA adapters with provider-owned factor and solve state;
- public native-only API and one-shot MOI path;
- separate `benchmark/general/` and `benchmark/bootstrap/physics/` trees;
- removal of Mathematica/WSTP integration and the standalone
  `nonnegative_hsd.jl` solver.

The local symmetric-core integration branch additionally contains:

- an independent full-five-equation augmented-core oracle;
- frozen block-aware CSC pattern/numeric refill;
- Float64 CHOLMOD symbolic reuse and signed numeric refactor;
- MFLA Float64x2/x4 and BFLA BigFloat256 dense pivoted-LDL factories;
- state-owned, epoch-refactorable core workspaces and truthful receipts;
- same-iterate LP/SOC/PSD/Exp/Power shadow parity against expanded reference;
- internal prepared-core production steps with raw core `dy` ownership;
- an opt-in forced Power dual-Hessian scaling experiment.

This list records implementation presence, not final release qualification.
The user will run the sole black-box E2E only after the remaining development
and final review are complete.

## 4. Current implementation work

### A. Legacy source retirement — complete

The standalone nonnegative HSD, LP, sparse-LP, SDP interior-point, legacy
Newton-step, and standalone NativeSOC engines have been deleted. Still-required
generic helpers were moved to native owners; public `engine=:auto` and
qualified SDP/Conic entrypoints now execute product HSD. No legacy solver file
is included or compiled.

### B. Symmetric-core public integration — complete

Completed milestones (R1–R6):

- Public `:bordered` routes to the Clarabel-style symmetric augmented core (`K = [0 Ar'; Ar -Theta]`) with pre-allocation execution planning;
- Arithmetic-specific descriptors: Float64 (sparse, CHOLMOD LDL), BigFloat (dense, BFLA LDL), MultiFloat (dense, MFLA LDL);
- Preserves full rank/ray authority on row-space reduction before memory preflight or core allocation;
- Pruned legacy workspace allocation on prepared symmetric core states;
- Dual-Hessian retry policy for Power cones with iterate checkpointing;
- Truthful execution diagnostics published in `ExecutionPlan` and `NativeHSDDiagnostics`.

### C. Dependency and design pruning — complete

- Pruned `AppleAccelerate`, `GenericLinearAlgebra`, and `JLD2` extensions and weakdeps;
- Kept stdlib `SuiteSparse`/`CHOLMOD`, `MultiFloats`, `MultiFloatLinearAlgebra`, and `BigFloatLinearAlgebra`;
- Maintained fail-closed precision and memory policies without hidden fallbacks.

### D. Runtime and performance wiring — complete

- Deterministic serial execution budget (`executed_threads = 1`);
- Residual and RHS terms reused across predictor/corrector solves without reallocation;
- Zero tolerance relaxation and zero hidden route fallbacks.

### E. Validation and handoff

- All focused fast suites (<60s) pass 100% (`validation/symmetric_core_reference.jl`, `validation/newton_system_reference.jl`, `validation/product_hsd_symmetric_shadow.jl`, `validation/product_hsd_symmetric_state.jl`, `validation/power_core_conditioning.jl`, `validation/power_core_dual_hessian_experiment.jl`);
- Repository clean and frozen for final user-owned E2E (`Pkg.test()`).

## 5. Verification layers

### Black-box E2E

`Pkg.test()` runs the sole `test/runtests.jl` regression suite. It selects
deterministic cases from `benchmark/general/` and checks:

```text
general case -> public optimize! -> terminal result -> original certificate
```

The suite covers LP optimal/primal-infeasible/dual-infeasible, SOCP, SDP,
exponential, and power examples in Float64. It does not inspect KKT routes,
providers, allocations, receipts, RSS, or thread scheduling.

### Specialist validation

Provider and independent Newton checks live under `validation/`; allocation,
route, memory, and physics checks live under `benchmark/`. They are manual or
release validation, not additional package-test suites and not part of the E2E
definition.

### Fixed-trace Q3 integration status

The local 2×2 factor kernel and the generic equality Schur
`B*H^-1*B'`/RHS/recovery oracle are implemented and validated for Float64 and
BigFloat.  CSDR now has an exact post-Wilson angular/energy operator whose
forward, adjoint, reconstruction, objective, and materialized reduced panel
pass parity tests.  Medium/full F3L probes retain 168 scaled equality
coordinates while reducing the prospective dense factor dimension from
5,208/19,608 to 168.

This is still internal.  Public HSD first eliminates ZeroCone rows through a
global nullspace basis, destroying the disjoint Q3 tail coordinates.  A
production switch therefore requires one equality-aware five-equation route:
retain ZeroCone rows with barrier degree zero, use the Q3 Schur factor for the
core homogeneous/predictor/corrector RHS, recover all local primal/dual rows,
and pass `newton_residual!` plus original-coordinate certification.  The
current generic bordered probe stops at iteration-zero line-search breakdown;
expanded fails KKT initialization.  Neither may be reported as CSDR progress.

### Benchmark and cluster validation

- `benchmark/general/`: solver-oriented LP/SOCP/SDP/Exp/Power corpora;
- `benchmark/bootstrap/physics/`: provenance-backed physical applications;
- local small/medium campaigns before cluster submission;
- medium/large general and physics/bootstrap campaigns on the UCAS PBS cluster;
- local and cluster artifacts must name the same frozen SHA.

## 6. Final integration sequence

1. complete Power scaling/retry policy and prepared-core public planning;
2. qualify arithmetic-specific diagnostics and remove unused legacy ownership;
3. prune optional dependencies and dead execution scaffolding;
4. finish bounded terminal residual/thread/fixq3 wiring;
5. run individual short family/provider/Newton validations;
6. obtain one final independent code/math/performance review and fix blockers;
7. freeze a clean handoff SHA;
8. the user runs the sole black-box E2E;
9. repair any E2E failure without weakening certificates or tolerances;
10. run local/cluster campaigns on the same frozen SHA;
11. push the complete batch to GitHub once; and
12. continue optimization before declaring a formal 1.0 release.

## 7. Non-negotiable policies

- `:bordered` remains the default until another route wins a complete evidence
  matrix.
- Equilibration remains opt-in until regression-free on physical probes.
- Nonfinite values and invalid tolerances fail closed.
- Original-coordinate certificates are the sole terminal authority.
- No silent Float64 downgrade is allowed.
- No family-specific hidden fallback or legacy public selector is allowed.
- Historical reviews are evidence, not current architecture specifications.

## 8. Phase 1 execution status (2026-08-30)

Phase 1 (C1-C6) is complete on this branch:

| ID | Fix commit | Exit evidence |
| --- | --- | --- |
| C1 | 0f47d37 | `validation/scalar_closure.jl` (classification fixtures, public min x >= 0 -> Optimal with original-coordinate certification, scaled variants, 1x1 PSD edge); symmetric-core and fixed-trace share one classifier |
| C2 | eb6645c | `validation/power_conjugate_root.jl` (captured-root representability fixture, 256-bit oracle sanity, fail-closed checks); power_epigraph_small default 1e-8 -> Optimal, certificate valid, objective within tolerance |
| C3 | 18ac53b | QDLDL refine_once! via the provider seam; refine lifecycle validation (10 tests) with original-operator residual contraction |
| C4 | 18ac53b | FactorRequirements <: AbstractFactorRequirements; frozen-shape ownership enforced and tested |
| C5 | 48d951f | bin/test/runtests.jl CLI option validation (6 tests) |
| C6 | 48d951f | equality COO length/duplicate validation (4 tests) |

Additional state on this branch beyond the reviewed plan:

- CSDR fixed-trace Q3 production solve: alpha3 Float64x4 21.7s and alpha9
  76.8s (bit-identical trajectories), BigFloat256 alpha3 optimal/certified
  (127s) — the BigFloat path required fixing the shared-MPFR aliasing traps
  (fill!/zeros shared slots + in-place provider mutation) via
  `_owned_setindex!` and a trial-residual fallback.
- MFLA mulacc_x4 fused kernel (1,000,005-case adversarial bitwise validation)
  and SIMD tail lanes for syrk/gemv remainders (bit-identical).
- benchmark/general inventory expanded: 20 external specs (SDPLIB 14,
  Netlib 5, CBLIB 1) with `sdpa_model` conversion and vendored data.

The Phase 2 (ownership simplification) and Phase 3 (evidence-driven
performance) work of the reviewed plan remain open on top of this branch.
