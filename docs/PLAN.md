# SDPX.jl roadmap to 1.0

**Status:** active implementation plan

**Updated:** 2026-08-28
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

`docs/design/NEWTON_SYSTEM.md` defines the five equations. Implementations may
choose:

- `:bordered` — conservative default;
- `:expanded` — exact nonsymmetric expanded solve;
- `:sparse_schur` — reduced sparse Schur solve with same-iterate fallback.

The exact expanded operator is nonsymmetric. A symmetric quasidefinite
companion may provide inertia evidence only when its contract is applicable;
it does not replace the exact solve or residual check.

### Provider ownership

- SuiteSparse/UMFPACK: Float64 sparse exact solves;
- PureKLU: exact nonsymmetric high-precision sparse LU;
- QDLDL: symmetric-companion inertia and regularization evidence;
- MultiFloatLinearAlgebra and BigFloatLinearAlgebra: dense and local-block
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
- MFLA/BFLA adapters and reviewed PureKLU/QDLDL prototypes;
- public native-only API and one-shot MOI path;
- separate `benchmark/general/` and `benchmark/bootstrap/physics/` trees;
- removal of Mathematica/WSTP integration and the standalone
  `nonnegative_hsd.jl` solver.

This list records implementation presence, not final release qualification.
The unified verification campaign has not yet been run on the final SHA.

## 4. Current implementation work

### A. Retire remaining legacy code

The standalone nonnegative HSD, LP, sparse-LP, SDP interior-point, and legacy
Newton-step engines have been deleted after their still-required generic helpers
were moved to native owners. Public `engine=:auto` and qualified SDP entrypoints
now execute product HSD.

The remaining retirement target is:

```text
src/soc_native.jl
```

Its exact source callers are tracked in `docs/LEGACY_ENGINE_REFERENCES.md`.

### B. Complete production provider dispatch

Connect PureKLU and QDLDL to the production HSD route with:

- exact operator and factor-generation receipts;
- memory preflight;
- no arithmetic narrowing;
- original-operator residual checks for every right-hand side; and
- same-iterate fallback to accepted dense routes.

### C. Finish performance wiring

- extend the now-wired fused predictor/corrector residual evaluation to
  terminal certificate inputs;
- apply deterministic `ThreadBudget` at pipeline entry;
- move the remaining fixq3 ownership out of `soc_native.jl`; Exp/Power 3x3
  contributions are wired for expanded and sparse-Schur assembly;
- verify compact workspace and PSD panel use in actual hot paths;
- calibrate route selection only from measured evidence.

### D. Close general benchmark findings

Resolve the remaining SDP, exponential, and power numerical findings without
loosening certificates or hiding failures. General benchmark expectations must
remain explicit and independently checkable.

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

1. finish and review all active implementation branches;
2. merge to one clean local integration SHA;
3. run package load and black-box E2E;
4. run targeted provider, precision, MOI, certificate, allocation, and route
   validation;
5. repair failures without weakening the E2E contract or relaxing tolerances;
6. run local general and physics/bootstrap campaigns;
7. freeze and deploy the same SHA to the cluster;
8. run medium/large and high-precision campaigns;
9. obtain a fresh independent code/math/performance review;
10. fix every release blocker;
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
