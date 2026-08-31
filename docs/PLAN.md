# SDPX.jl roadmap to 1.0

**Status:** active correctness remediation; `v0.6.0` is not release-qualified

**Updated:** 2026-08-30

**Reviewed baseline:** `bad117f7d8b8cc49b0fef08f30e22505f140360a`
(`main`, `v0.6.0`)

**Authority:** this file describes current work. Frozen mathematical contracts
live in `docs/design/`; completed historical plans and reviews remain in Git
history and `docs/reviews/`.

## 1. Completion definition

SDPX 1.0 is complete only when all of the following hold on one frozen source
SHA:

1. one product-cone homogeneous self-dual (HSD) state machine is the only
   production solver engine;
2. one five-equation `NewtonSystem` defines every KKT route;
3. LP, SOC, RSOC, PSD, exponential, power, free, zero, nonnegative, and
   nonpositive blocks share the same canonical program and certificate path;
4. every `Optimal`, `PrimalInfeasible`, and `DualInfeasible` result has a
   valid original-coordinate certificate;
5. Float64, MultiFloat, and BigFloat execution never silently narrows values;
6. KKT and linear-algebra providers are replaceable without changing the
   Newton equations;
7. the typed API, MOI adapter, CLI, general benchmarks, and physics/bootstrap
   benchmarks call the same production engine;
8. no hidden legacy solver or PSD-lift fallback is reachable;
9. every supported Julia/OS job, documentation build, provider contract,
   package test, CLI test, and release benchmark gate is green; and
10. local and cluster campaigns reproduce on the same frozen SHA.

Passing a narrow smoke suite is not release qualification. No test may be
deleted, weakened, or reclassified as an expected numerical failure merely to
obtain a green result. Provider receipts are implementation evidence, not
mathematical certificates.

## 2. Current verified state

### Implemented foundation

- The public path uses the native product-cone HSD engine.
- The symmetric augmented core is the production `:bordered` implementation;
  the former standalone solver/KKT engines have been retired.
- Canonicalization retains reconstruction data and the final public result is
  checked in original coordinates.
- Invalid public certificates downgrade otherwise terminal core statuses.
- Reference tests for the frozen five-equation system, symmetric core, state
  ownership, and factor epochs are extensive and currently pass locally.
- Typed modeling, MOI, CLI, general benchmark, provider-extension, and
  physics/bootstrap surfaces exist, but are not yet covered by one coherent
  automated release gate.

### Review baseline evidence

The 2026-08-30 review of `bad117f` established:

- local Julia 1.12.6 `Pkg.test()` passes 21/21 assertions, covering only seven
  selected Float64 E2E cases;
- `benchmark/general/test_small.jl` runs 17 small cases but produces only 11
  valid certificates and explicitly accepts six invalid certificates;
- Julia 1.10.11 reproduces a hard LAPACK method error in the production LU
  factor-cache route;
- the documented scalar lower-bound LP, elementary unbounded models, and a
  bounded rank-deficient model do not return the required certified results;
- three small SDP, two exponential-cone, and one power-cone analytical cases
  remain unresolved;
- the release commit's main CI run has seven failed jobs out of ten, and the
  tag CI run also fails;
- documentation, benchmark-environment, and minimum-version provider gates do
  not complete successfully; and
- no false positive terminal certificate was observed: confirmed defects fail
  closed as numerical/precision failures.

Therefore `v0.6.0` is a failed release candidate, not evidence that the 1.0
completion definition has been met.

## 3. Frozen engineering invariants

All remediation must preserve these invariants:

- The canonical problem remains `min c'x` subject to `A*x + s = b`, `s in K`.
- The production state remains `(x, s, y, tau, kappa)` with the signs and five
  equations frozen in `docs/design/HSD_FORMULATION.md` and
  `docs/design/NEWTON_SYSTEM.md`.
- Only original-coordinate verification may publish a terminal mathematical
  status.
- Nonfinite values, invalid tolerances, failed factorization, and uncertain
  cone or rank classifications fail closed.
- Repairs must not weaken certificate tolerances, cone identities, root
  certification, or third-derivative symmetry checks without an independent
  higher-precision error proof.
- No silent route, arithmetic, provider, or solver-engine fallback is allowed.
- Every code change follows
  `reproduce -> freeze regression -> surgical change -> verify`.

## 4. Release-blocker register

### B0 — Restore the declared Julia 1.10 contract

**Observed:** `src/factor_cache/routes/common.jl` calls
`LinearAlgebra.LAPACK.getrf!(F, ipiv)`. Julia 1.10 exposes only the
single-matrix signature, so the v2 acceptance gate throws a `MethodError`.

**Work:**

1. add a factor-cache implementation compatible with every declared Julia
   version, retaining owned pivot storage and warm-path allocation guarantees;
2. add a direct Julia 1.10 regression for prepare/factorize/solve/refine; and
3. test both the minimum Julia version and latest Julia in package and v2
   gates.

**Exit:** Julia 1.10 and latest Julia solve the LP factor-cache fixture and the
full public E2E suite without compatibility branches changing numerical
semantics.

### B1 — Close basic LP, ray, and rank-reduction behavior

Freeze public-API regressions for all four failures before changing recovery:

1. the documented `min x` with `x >= 0` and `x - 1 >= 0` must return
   `Optimal`, `x = 1`, and a valid certificate;
2. `min x` for free `x` must return `DualInfeasible` with a valid ray;
3. `max x` for `x >= 0` and `min x` for `x <= 0` must return
   `DualInfeasible` with valid rays; and
4. a bounded problem with an irrelevant free variable must be reduced and
   solved, not rejected at iteration zero.

Audit canonical recovery, sparse-rank authority, nullspace reconstruction,
and exception preservation. Do not convert these fixtures into accepted
numerical failures.

**Exit:** every fixture returns the mathematically correct certified status
through the typed API and MOI path on all supported platforms.

### B2 — Make the public certificate the stopping authority

**Observed:** `exp_entropy_small` reaches internal `Optimal` under a global
canonical residual scale, then fails public certification because reconstructed
stationarity uses a different scale (`1.5480679849e-8 > 1e-8`).

**Work:**

1. define one authoritative source-model residual map and normalization;
2. either evaluate it at accepted-iterate stopping/refinement or prove a
   conservative amplification bound for every lowering/reconstruction map;
3. retain canonical five-equation verification as an independent internal
   gate; and
4. add a captured entropy trajectory regression at the requested tolerance.

**Exit:** core terminal status and public certification cannot disagree merely
because they normalize equivalent residuals differently.

### B3 — Add a zero-dimensional reduced feasibility path

**Observed:** `sdp_theta_k4`, `sdp_rank1_boundary`, and `sdp_random_small`
reduce to `variables = 0` and terminate with
`tau_collapse_recovery_exhausted` instead of certifying fixed-cone solutions.

**Work:**

1. freeze exact reduced-width-zero PSD fixtures;
2. directly verify fixed slack feasibility and construct/verify a compatible
   dual without running an unnecessary projective HSD trajectory; and
3. prove reconstruction through equality and cone maps.

**Exit:** all three small SDP cases return valid certificates and the general
path remains unchanged for positive reduced dimension.

### B4 — Repair nonsymmetric Exp/Power trajectory integration

**Observed:**

- `exp_logsumexp_small` fails with
  `NS_CORRECTOR_THIRD_SYMMETRY_MISMATCH`, currently collapsed to the generic
  `symmetric_core_dispatch_exception` reason;
- `power_geomean_small` fails with
  `NS_CONJUGATE_ROOT_RESOLUTION_LIMIT`; and
- the required `power_epigraph_small` passes on macOS but fails in current
  Linux and Windows package jobs.

**Work:**

1. capture the last accepted nonsymmetric states and affine directions as
   deterministic fixtures;
2. compare Float64 third contractions, scaling, root brackets, and step
   acceptance against a 256-bit oracle;
3. preserve typed `NonsymmetricRuntimeResult` reasons through symmetric-core
   dispatch; and
4. remove platform dependence without weakening fail-closed checks.

**Exit:** every checked-in small Exp and Power case is certificate-valid on
Julia 1.10/latest across Linux, macOS, and Windows.

### B5 — Repair optional-provider version and execution contracts

**Observed:** declared compatibility permits MultiFloatLinearAlgebra 0.1 and
BigFloatLinearAlgebra 0.1, while the extensions import cache APIs introduced in
MFLA 0.4 and BFLA 0.2. The pinned-minimum CI environment therefore fails
precompilation. The workflow also runs both providers in one process even
though `scripts/provider_smoke.sh` deliberately isolates them.

**Work:**

1. either raise compatible lower bounds or add explicit version adapters;
2. test declared minimum and latest supported versions independently;
3. make CI call the maintained provider smoke entrypoint; and
4. restore full factor lifecycle, direction parity, precision ownership, and
   public high-precision solve checks.

**Exit:** minimum/latest MFLA and BFLA matrices load, solve, refine, report
truthful receipts, and pass public certificate tests without arithmetic
narrowing.

### B6 — Reconcile the public API with executable documentation

**Observed:** `Settings` accepts and documentation advertises formulation,
provider, presolve, sparse, equality-solver, and BLAS-thread values that the
only public solver route categorically rejects. Checked-in examples still pass
removed `algorithm=:lp`, `:socp`, and `:sdp` selectors. The documented time
limit is described as end-to-end although the solver timer starts after setup.

**Work:**

1. remove unsupported public choices or implement them with truthful execution
   receipts;
2. update every example to the native-only policy and run examples in CI;
3. decide and test whether `Limits.time` includes canonicalization, planning,
   solve, recovery, and certification; and
4. test every public setting for accepted execution or an intentional,
   documented construction-time rejection.

**Exit:** all checked-in examples execute, and no documented accepted setting
is unconditionally unusable on the sole production route.

### B7 — Make documentation, CLI, and benchmark gates runnable

**Observed:**

- Documenter fails `checkdocs=:exports` with 66 missing exported docstrings;
- the benchmark environment contains unregistered `PMP2SDP` but CI does not
  develop a source for it; and
- CLI tests require a setup side effect and are not run by CI.

**Work:** document or deliberately unexport the missing API, make benchmark
dependencies reproducible from a clean checkout, and give the CLI an isolated
test environment invoked by CI.

**Exit:** docs, CLI, and micro-benchmark jobs run from a fresh checkout with no
unrecorded local state.

## 5. Test and release-gate redesign

The release gate must test behavior, not merely repository layout.

### Required package coverage

- Replace the seven-case selection with certificate-backed coverage of all 17
  checked-in small analytical cases.
- Add the B1 scalar/ray/rank fixtures permanently.
- Add MOI conformance tests for every supported objective, variable domain,
  constraint type, status, result accessor, and start-value contract.
- Exercise CLI parsing and file I/O, JLD2 persistence if retained, examples,
  and public non-Float64 paths.
- Remove the quick-check rule that `test/` may contain only two files; require
  instead that every test file is reachable from `runtests.jl` or is named on a
  reviewed manual-validation allowlist.

### Known-finding policy

Development reports may retain a quarantined known-finding label, but:

- an invalid certificate cannot satisfy a release-quality test;
- the release job must visibly fail while a bounded analytical case remains
  quarantined; and
- status, typed reason, iteration, residuals, and certificate limits must be
  retained as artifacts for every failure.

### Required platform matrix

Run at minimum:

- Julia 1.10, Linux, one thread;
- latest Julia, Linux, four threads;
- latest Julia, macOS, four threads; and
- latest Julia, Windows, four threads.

Provider minimum/latest jobs and high-precision public solves are additional
gates, not replacements for the standard matrix.

## 6. Execution sequence

### Phase A — Freeze regressions and restore scalar correctness

1. add failing fixtures for B0 and B1 without changing solver behavior;
2. repair Julia 1.10 factorization compatibility;
3. repair scalar lower-bound reconstruction, unbounded rays, and redundant
   free-variable reduction; and
4. run package, MOI, and scalar validation on the full platform matrix.

### Phase B — Close cone-family correctness

1. unify stopping and public certificate authority (B2);
2. implement the reduced-width-zero SDP certificate path (B3);
3. repair Exp third-correction and Power root/scaling integration (B4); and
4. require 17/17 valid certificates in the small analytical suite.

### Phase C — Restore extension and public-contract integrity

1. repair provider compatibility and minimum/latest CI (B5);
2. reconcile settings, examples, limits, and documentation (B6);
3. make docs, CLI, and benchmark environments clean-checkout reproducible
   (B7); and
4. run all specialist validations from CI or a required release workflow.

### Phase D — Performance and application qualification

Only after Phases A-C are green:

1. benchmark cold/warm allocations and factor reuse without weakening checks;
2. qualify fixed-trace Q3 and physics/bootstrap paths through the same public
   HSD and certificate authority;
3. run medium/large general and physics campaigns locally, then on UCAS PBS;
4. compare runtime, memory, residuals, and certificate outcomes against the
   frozen baseline; and
5. optimize only regressions supported by retained benchmark artifacts.

## 7. Evidence matrix

| Gate at `bad117f` | Current evidence | Required state |
| --- | --- | --- |
| Local Julia 1.12 `Pkg.test()` | Pass, 21/21 narrow assertions | Pass with expanded public coverage |
| General small tier | 11/17 valid certificates | 17/17 valid certificates |
| Julia 1.10 compatibility | Fail, `getrf!` `MethodError` | Pass package and v2 gates |
| Linux latest package job | Fail | Pass |
| Windows latest package job | Fail | Pass |
| macOS latest package job | Pass narrow suite | Pass expanded suite |
| Fixed Newton/core references | Pass locally | Pass in required CI/release workflow |
| MFLA/BFLA provider smoke | Fail declared minimum versions | Pass minimum and latest versions |
| Documentation build | Fail missing exports | Pass `checkdocs=:exports` |
| Benchmark smoke environment | Fail unregistered dependency | Instantiate and run cleanly |
| CLI tests | Not in CI; setup-dependent | Clean-checkout CI pass |
| MOI conformance | Not present | Required supported-surface pass |
| Worktree/source identity | Clean at reviewed SHA | One clean frozen candidate SHA |

Update this matrix with command output or CI artifact links; do not change a
cell to “Pass” based on implementation presence or an unexecuted plan.

## 8. Standard verification commands

Run from the repository root in clean environments:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project=. benchmark/general/test_small.jl
julia --startup-file=no --project=. validation/newton_system_reference.jl
julia --startup-file=no --project=. validation/symmetric_core_reference.jl
julia --startup-file=no --project=. validation/product_hsd_symmetric_state.jl
```

Also run:

- `scripts/provider_smoke.sh` against declared minimum and latest provider
  checkouts in separate fresh environments;
- the clean-checkout CLI suite;
- `julia --project=docs docs/make.jl` after developing SDPX into the docs
  environment; and
- the benchmark micro runner after developing every unregistered dependency
  explicitly.

For each phase record command, Julia version, OS, thread count, commit SHA,
exit code, assertion count, and retained failure artifact.

## 9. Deferred work and historical evidence

The following work remains valuable but cannot substitute for closing the
release blockers:

- fixed-trace Q3 equality-aware production integration;
- Float64/MultiFloat/BigFloat performance tuning and allocation reduction;
- external SDPLIB, Netlib, CBLIB, and physics/bootstrap campaigns;
- cluster throughput and memory tuning; and
- alternative KKT-route performance comparisons.

Completed source-retirement, QDLDL lifecycle, factor-ownership, scalar/root
oracle, fixed-trace, provider-kernel, and external-inventory work remains in
Git history and validation artifacts. Historical “complete” labels describe
their reviewed scope only; they do not override the current evidence matrix.

## 10. Release protocol

1. Close B0-B7 with regression-first commits.
2. Make every row of the evidence matrix pass on one clean candidate SHA.
3. Run local medium/large campaigns and inspect retained artifacts.
4. Run the UCAS cluster campaign on the identical SHA.
5. Obtain a final independent numerical, implementation, API, and packaging
   review.
6. Repair every release blocker without weakening certificates or tests.
7. Re-run the entire matrix and campaigns on the new frozen SHA.
8. Only then create a new version/tag and publish release notes.

Do not use the existing `v0.6.0` tag as a qualified baseline, and do not create
another release tag while any required job or analytical certificate is red.
