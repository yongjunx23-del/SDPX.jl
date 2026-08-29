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

### B. Finish symmetric-core public integration

Current frozen facts:

- public `:bordered` remains the old proven route;
- prepared-core execution is internal/opt-in and passes focused LP/SOC/PSD/Exp
  probes plus Power with raw core `dy`;
- forced Power dual-Hessian scaling is implemented only as an internal
  experiment and reaches `ProductHSDOptimal` in the bounded trajectory test;
- QDLDL/PureKLU remain removed; high-precision sparse is unsupported.

Remaining work:

1. decide and implement the final Power policy: primal-dual core, explicit
   dual-Hessian retry/checkpoint, and only a truthful safety fallback if strict
   public tolerances still require it;
2. rebuild public planning before state allocation, preserving rank/ray
   authority;
3. publish arithmetic-specific planned/executed provider, factor, kernel,
   precision, regularization, memory, and attempt facts;
4. remove unused legacy workspace allocation from prepared-core states;
5. switch public `:bordered` only after all short family probes pass; the user
   then runs the sole E2E.

### C. Reduce dependency and design surface

After the public switch is qualified, remove production-unreachable:

- full-border/coupled/reduced duplicate sessions and `ProviderLPLUCache`;
- internal high-precision generic LU/LDL fallback;
- stale calibration/fixq3 scaffolding without a production caller;
- GenericLinearAlgebra, AppleAccelerate, JLD2, and `LegacyLABackend` only where
  the final call graph proves they are unnecessary.

MFLA, BFLA, MultiFloats, and stdlib SuiteSparse/CHOLMOD remain.

### D. Finish bounded runtime/performance wiring

- reuse existing fused direction terms in terminal certification only where
  mathematically identical;
- enforce one deterministic ThreadBudget owner and truthful
  requested/effective diagnostics, with serial default;
- retain fixq3 only as a zero-new-route Newton contribution, otherwise delete
  it;
- remove duplicate factor-proof/workspace work without weakening original-K,
  five-equation, or original-coordinate certificate gates.

### E. Close general benchmark findings

Resolve the remaining SDP, exponential, and power numerical findings without
loosening certificates or hiding failures. General benchmark expectations must
remain explicit and independently checkable. The user, not the development
loop, runs the final sole E2E.

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
