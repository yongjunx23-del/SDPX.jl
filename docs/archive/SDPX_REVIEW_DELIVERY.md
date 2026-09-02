# SDPX.jl solver-speed review and continuation plan

## Scope and safety rule

This delivery treats numerical correctness as a hard constraint. A speed change is acceptable only when it preserves the five HSD residual/equation checks, status/certificate semantics, and the requested arithmetic backend. Tolerance relaxation, silent fallback to a lower precision, or changing a problem formulation is not counted as a performance improvement.

The repository-specific automated report, test log, patch, and push/PR status are appended or linked in the delivery bundle. The status file is the source of truth for whether a direct code change was made, tested, committed, or pushed.

## Executive recommendation

The next major speed gain should not come from more local `@inbounds` edits. The dominant program should be:

1. make phase-level timing and allocation data trustworthy;
2. cache the KKT pattern, ordering, symbolic factorization, and workspaces across Newton steps and repeated solves;
3. reduce numeric-factorization cost through formulation selection, scaling, and mixed-precision refinement;
4. make cone kernels allocation-free and block-parallel, with PSD handled separately from scalar/SOC/EXP/POWER cones;
5. add precision escalation so expensive MFLA/BFLA work is used only where residual evidence requires it;
6. add larger structural changes—chordal decomposition, matrix-free Schur or hybrid KKT—only after the preceding gates are stable.

## Required telemetry before further optimization

Every solve should report, without changing the public result contract:

- canonicalization/setup and compilation-excluded wall time;
- KKT structural build, symbolic analysis/order, numeric assembly, factorization, triangular solves, iterative refinement;
- cone scaling/Hessian/inverse-Hessian work by cone family and by PSD block size;
- residual and certificate evaluation, line search, finalization;
- allocations and peak workspace bytes by phase;
- factorization backend requested, planned, executed, fallback reason, arithmetic used for factor and residuals;
- Newton iterations, rejected steps, regularization changes, refinement iterations, and refactorizations;
- first solve versus same-structure repeat solve.

Acceptance gate: phase totals reconcile with wall time to within a small accounting overhead, and repeated runs on a fixed problem produce stable medians.

## Ordered implementation plan

### Phase 0 — Freeze correctness and benchmark contracts

Build a manifest of small, medium, and large LP, SOCP, SDP, EXP, POWER, and mixed-cone problems. Include feasible optimum, primal infeasible, dual infeasible, ill-conditioned, nearly degenerate, and high-accuracy instances. Save structural fingerprints, expected status, objective interval, residual limits, certificate checks, and formulation size.

Run each in Float64, Float64x2/x3/x4 where supported, and BigFloat 256/512. Record cold-start and warm/repeated-solve results separately. Compilation must not be mixed into steady-state solve time.

Gate: no optimization branch proceeds unless it reproduces status, certificate validity, objective interval, and the five-equation checks on the baseline corpus.

### Phase 1 — Measurement and type-stability audit

Add a single internal timing/counter object with zero/near-zero overhead when disabled. Profile representative solves with Julia sampling profiles, allocation tracking, and type-instability tools. Audit abstract fields, `Vector{AbstractCone}`, captured closures, accidental `Any`, views that allocate, sparse format conversions, and precision conversions inside Newton loops.

Prioritize findings by inclusive time × call count × avoidability. Do not optimize compilation latency and steady-state kernels in the same patch.

Gate: top hot paths are identified for each cone family and precision backend; at least 90–95% of steady-state time is assigned to named phases.

### Phase 2 — Reuse KKT structure and workspaces

Introduce a structure key based on dimensions, cone partition, sparsity pattern, KKT formulation, ordering, and factor backend. Reuse:

- CSC column pointers/row indices;
- permutation/order and symbolic factorization;
- scatter maps from model blocks to KKT numeric values;
- block-local cone workspaces;
- RHS/solution/refinement buffers;
- equilibration vectors when the structure and scaling policy permit.

Only numeric values should be refreshed per Newton step. Separate `analyze!`, `factorize!`, `solve!`, and `refine!` ownership. Ensure `clear!` invalidates all derived state and that planned/executed backend metadata cannot become stale.

Gate: after the first Newton step, symbolic-analysis count remains one unless regularization/formulation changes the pattern; same-structure repeated solve performs zero new symbolic analyses and near-zero setup allocations.

### Phase 3 — KKT formulation and sparse factorization policy

Benchmark augmented/quasi-definite KKT versus primal or dual Schur forms on structural features rather than using one global default. Candidate features: `m/n`, estimated fill, PSD block spectrum, density of `A`, cone Hessian block density, arithmetic backend, and available sparse factor provider.

For each option, track fill ratio, factor bytes, factor time, solve time, pivot/regularization behavior, and refinement count. Cache AMD/COLAMD/METIS-like ordering results where available. Avoid materializing dense global cone matrices; retain block-local operators and scatter into sparse numeric storage.

For high precision, keep the provider contract explicit: unsupported sparse factorization must fail clearly or choose a declared alternative—never silently densify a large problem or silently lower precision.

Gate: planner choices beat or match the best fixed policy on the benchmark training set and do not regress the held-out set beyond the agreed budget.

### Phase 4 — Mixed-precision factor/solve/refine

Implement a ladder rather than starting every solve at maximum precision:

- scale/equilibrate and form residuals in the requested truth precision;
- factor in the cheapest backend predicted to be safe;
- solve and perform residual-based iterative refinement;
- escalate factor precision or switch formulation when contraction stalls, correction grows, or backward error fails;
- recompute certificates and final residuals in truth precision.

For MFLA, specialize dot, axpy, norm, triangular solve, and sparse accumulation paths to avoid scalar boxing and repeated conversion. For BFLA, minimize BigFloat allocation and precision-context churn; use persistent buffers and explicit rounding/precision boundaries.

Gate: every accepted mixed-precision result passes truth-precision residual/certificate checks; escalation is deterministic and fully recorded. Compare end-to-end time, not factor time alone.

### Phase 5 — Cone-local kernel optimization

Use in-place APIs and preallocated block-local storage for scaling points, gradients, Hessians, inverse Hessians, Jordan products, and step-bound calculations. Fuse loops where it reduces memory traffic without obscuring formulas. Avoid generic dispatch in inner loops by grouping cone blocks by concrete type and precision provider.

For SOC/RSOC, avoid temporary vectors and repeated norm computation; centralize numerically stable step-length and boundary logic. For EXP/POWER, reduce transcendental calls where exact reuse is possible, but retain guarded domains and non-finite checks. Do not replace robust formulas with faster fragile approximations.

Gate: zero steady-state allocations in scalar/nonnegative/SOC/RSOC block kernels; EXP/POWER allocations are bounded and explained; kernel microbenchmarks translate into whole-solver gains.

### Phase 6 — PSD-specific path

Treat PSD as a separate performance program:

- store svec/smat index maps once;
- avoid materializing Kronecker products or full global Hessians;
- use BLAS-3 congruence transformations and symmetric rank-k operations;
- batch blocks by size and parallelize independent blocks;
- select small-block stack/static kernels only below an evidence-based threshold;
- reuse eigendecomposition/factor workspaces;
- consider low-rank or partial spectral work only where the barrier formula and correctness permit;
- add chordal decomposition for structurally sparse large PSD cones, with reconstruction/certificate tests.

Gate: report performance by PSD block-size distribution, not only total cone dimension. Chordal conversion must preserve objective and residual/certificate semantics on decomposable test cases.

### Phase 7 — Iteration-count reductions

Once linear algebra is trustworthy, tune predictor-corrector/centering, neighborhood control, step safeguarding, regularization, scaling, and linear-solve tolerance. Use residual trends to avoid oversolving early Newton systems and tighten near convergence. Keep status logic independent of heuristic progress metrics.

Gate: median Newton iterations and rejected steps fall without increasing failure, false-optimal, or inaccurate-certificate rates.

### Phase 8 — Parallelism and repeated-solve API

Parallelize only independent coarse work: cone blocks, PSD blocks, residual reductions, and selected assembly ranges. Prevent BLAS × Julia thread oversubscription and make deterministic single-thread mode available. Add a reusable solver/session object for same-structure parameter sweeps and warm starts.

Gate: scaling is measured at 1, 2, 4, 8, … threads with fixed BLAS settings; small problems do not regress due to thread overhead; repeated-solve APIs show setup amortization explicitly.

### Phase 9 — Advanced alternatives

Evaluate matrix-free Schur/Krylov only for regimes where factor fill or memory dominates, with a cone-aware preconditioner and strict stopping rules tied to HSD residual progress. Keep the direct sparse path as a reference and fallback. Consider out-of-core or distributed PSD work only after single-node memory traffic is measured.

Gate: advanced paths win on a declared structural region and have deterministic fallback when convergence or preconditioning fails.

## Benchmark matrix

Use at least these axes:

- cone family: LP, SOC/RSOC, PSD triangle, EXP, POWER, mixed;
- structure: dense, sparse random, block angular, network-like, moment/Toeplitz, chordal PSD, bootstrap-like;
- size: subsecond smoke, seconds, minutes, and large stress tiers;
- conditioning: nominal, scaled, nearly dependent rows, boundary optimum, ill-conditioned KKT;
- outcome: optimal, primal infeasible, dual infeasible, iteration/precision escalation;
- arithmetic: Float64, Float64x2/x3/x4, BigFloat 256/512;
- lifecycle: cold compile, first solve after compile, same-pattern new values, warm start;
- threads and BLAS threads;
- backend/formulation requested versus executed.

Primary metrics: wall time, setup/solve split, factor time, allocations, factor memory/fill, Newton iterations, refinement iterations, final residuals, objective/certificate accuracy, and success rate.

## Performance regression policy

Use robust medians and enough repetitions for subsecond cases. A patch fails if it wins one microbenchmark but causes any of the following without an explicit approved tradeoff:

- changed mathematical formulation or relaxed tolerance;
- changed status/certificate validity;
- hidden precision downgrade or silent dense fallback;
- substantial held-out benchmark regression;
- higher peak memory that blocks intended scale;
- nondeterministic factor-provider selection;
- materially worse compilation/code-size behavior for common paths.

## High-priority bug classes to keep under dedicated regression tests

- stale factor/cache state after `clear!`, dimension change, precision change, or fallback;
- planned backend reported as executed backend;
- aliasing between RHS, correction, and refinement buffers;
- non-finite/negative step lengths accepted at cone boundaries;
- zero-dimensional or empty-cone edge cases;
- RSOC transform/sign/scaling inconsistencies;
- PSD triangle index or off-diagonal `sqrt(2)` scaling mismatch;
- EXP/POWER domain guards at nearly-boundary points;
- residual/certificate computed in factor precision instead of truth precision;
- silent sparse-to-dense or high-to-low-precision fallback;
- reuse of symbolic factorization after a sparsity-pattern change;
- BigFloat precision context or conversion performed inside hot loops;
- incorrect result finalization when termination occurs before a full Newton step.

## Recommended first implementation slice

The first mergeable performance slice should contain only:

1. phase counters/timers and exact provider/formulation metadata;
2. a structure-keyed cache for KKT sparsity, ordering, symbolic factorization, scatter maps, and workspaces;
3. cold/first/repeat benchmark cases proving reuse;
4. cache invalidation tests for dimension, cone partition, sparsity pattern, precision, backend, and formulation;
5. no algorithm/tolerance change.

This slice makes every later optimization attributable and gives immediate gains for Newton iterations and parameter sweeps with low mathematical risk.

---

# Active execution plan and status (updated 2026-09-01)

This section supersedes the obsolete `CLONE_FAILED` automated report. The
repository was recovered, tested, optimized, and merged through PRs #27 and
#30. The ordered plan below is now the source of truth.

## Safety invariants

No benchmark or performance candidate may:

- widen solver or certificate tolerances;
- change the physical/mathematical problem while claiming a speedup;
- introduce benchmark-specific constants into `src/`;
- lower arithmetic precision silently;
- drop equality rows/rank without a reviewed exact-equivalence proof;
- accept a status/objective/certificate/trajectory mismatch;
- publish a build-only physics artifact as a solved paper bound.

Julia 1.12 + MultiFloats production runs use `--gcthreads=1`. Every cluster
case uses immutable source/environment/input hashes and private output shards.

## Completed and merged foundation

The following work is on GitHub `main` and its CI is green:

- phase-level native-HSD telemetry and truthful requested/planned/executed
  receipts;
- structure-keyed symmetric-core cache and invalidation tests;
- planner-vs-fixed-policy gate evidence;
- fixed-trace barrier-free columns and frozen CSDR trajectory protection;
- blockwise certificate predicate parallelism;
- additive `Settings.iteration_knobs` with bit-identical defaults;
- certified PSD small-tier cases;
- unified lifecycle regression guard;
- E2E, docs and quality workflows, with non-E2E provider drift checks manual.

Last reviewed remote main at this update: `820485074ebb...`.

## Pre-integration branches

| Workstream | Local head | Review state | Publication state |
|---|---:|---|---|
| General Benchmark V2 / Stage-A integration | main `25ac853` (PR #31) | **MERGED / STAGE A COMPLETE**: final independent reviewer MERGE-READY after all P1 closure; 19 certified typed cases (17 optimal + Farkas/ray) across all 8 catalog families; strict precision/source/model/original-coordinate gates; external parity-pending rows ineligible; schema observed/missing fields honest. Precision receipts: x2 17/17, x4 16/17 (Chebyshev process SIGSEGV 3/3, never solver status), BF256 16/17 (EXP breakdown). Integrated route guard, optloop bridge, Modular build-only and 4D spec-only. Local Pkg.test/Documenter/catalog/schema/frozen SHA pass; PR #31 CI passes Julia1.10 + latest Linux/macOS/Windows, docs, quality, structural, Devin | merged to origin/main |

| Bordered route fallback | main via `25ac853` | **MERGED**: exact trigger, strict one-shot bordered->expanded terminal restart, child-composed attempts/coherent aliases; boundary certifies; duplicate remains platform-honest fail-closed (expanded terminal on macOS/Windows, earlier bordered dispatch failure on Linux); frozen CSDR SHA unchanged | main |
| Product-HSD allocation gate (Stage C lane-3) | `20be893` | **PASS**: knob lookups hoisted to typed state fields, per-step allocations 3.9 KiB -> 288 B residual (bordered dispatch, documented audit ceiling), frozen CSDR SHA bit-identical, full Pkg.test green | branch `perf/product-hsd-alloc-gate` pushed (off origin/main) |
| MFLA Phase 5 kernels (Stage C lane 2) | `50e6e0b` | **COMPLETE for this program**: bit-safe weighted-panel threading (`5885060`, ~4.2% panel bench, SDPX frozen SHA gate bit-identical) + triangular-solve parallelism ceiling documented (`50e6e0b`: single-RHS TRSV inherently serial, TRSM already column-parallel; solve bucket has no bit-safe parallel headroom) | branch `perf/phase5-kernels` pushed (MultiFloatLinearAlgebra.jl repo) |
| Hellerman modular PMP | main via `25ac853` (`7803a51`) | **MERGED AS BUILD-ONLY**; literal basis/provenance, standalone 121 tests, no certified witness/bound claimed | unregistered on main |
| 4D S-matrix spec | main via `25ac853` (`bb14f88`) | **MERGED AS SPEC-ONLY**; 149 CI assertions, sampled/unregistered, paper-equivalence false, dual fail-closed | unregistered on main |
| Dependent optimization workflow | main via `25ac853` (`20758d8` + integration fixes) | **LOCAL PRECONDITION PROVEN, READY FLAG DISABLED**: world-age-safe standalone bridge, schema-v9 target [9,9,9] and deterministic objective/cert/semantic/live validator; catalog gate 61 cases. Stage-B fresh-process/peak-RSS evidence now active; flag remains locked | main |
| CSDR scientific local pipeline | `b9d6d2d` (2026-09-01 session) | **TRANSFERRED to a dedicated window (2026-09-02) - this session no longer schedules CSDR work.** State at transfer: review loop CLOSED (4 rounds, all findings fixed); pair transaction reworked (atomic staging + validator-owned private snapshots + final live rechecks), wire format hard-fails closed (schema v2 + `decimal-bigfloat-v1`), BigFloat grid determinism fixed (serial order + canonical decimal encoding; 5+2 fresh-process byte-identical receipts), crash-residue policy documented, ABA/race/cleanup tests real (B-live-through-parsing), synthetic transaction-only l_max=0 zero-residual pair fixture PROVEN (held-out residuals 1.27e-16/1.04e-13, published pair hash `41d6638...`, race single-publication) and isolated from calibration/release/ladder. **Phase 2 / calibration / ladder remain KEEP LOCKED until a real accepted zero/twice pair exists.** Tiny zero/twice still intentionally fail held-out-a (543.9265.../427.5102...). Full handoff with branch lineage, guard commands, env rules, and immediate work items: `/Users/xuyongjun/Desktop/project/SDPX/CSDR/HANDOFF_CSDR_PROGRAM_20260902.md` | transferred to user-owned window |

## Required execution order

### Stage A — finish and merge the benchmark system

1. Complete General Benchmark V2 rather than wrapping V1 metadata only.
2. Require exact typed reference contracts, original-coordinate certificate
   gates, fixed-endian fingerprints, actual setup/core/recovery telemetry, and
   arithmetic-preserving objective/residual storage.
3. Integrate the reviewed modular-PMP and 4D S-matrix spec-only branches.
4. Run package tests, Documenter, benchmark catalog build, schema-v9 receipt
   checks, frozen CSDR trajectory, and GitHub CI on one integration branch.
5. Merge only after every benchmark is classified as either:
   - `solve_eligible` with an independent status/objective/certificate oracle;
   - `build_only` with construction/fingerprint/witness or explicit blocker;
   - `xfail` with a tracked, non-passing failure contract.

The dependent optimizer readiness flag remains disabled until Stage A produces
at least one complete solve-eligible schema-v9 target.

### Stage B — freeze the performance baseline

For every eligible target, freeze:

- source commit/tree and benchmark/catalog/input fingerprints;
- Project/Manifest/provider hashes and Julia/BLAS/GC/thread settings;
- resolved tolerances and requested/planned/executed route, formulation,
  backend, provider, kernel and reuse receipts;
- status, certificate kind/failures, objective interval, iterations and
  declared trajectory semantics;
- one excluded warmup and three fresh/warm measured samples;
- setup, core, recovery, factor, solve, allocation and peak-RSS evidence.

Build-only cases are never selected as solver-optimization targets.

### Stage C — multi-round solver optimization

Each round may generate isolated candidates in three lanes:

1. formulation/planner/cache;
2. KKT factorization/triangular-solve/kernel;
3. allocation/memory/threading.

A candidate is accepted only if it preserves every Stage-B identity and
correctness receipt and delivers at least a stable 2% median improvement on the
target metric. Allocation-only wins do not qualify.

Stopping conditions:

- maximum six accepted/rejected rounds per target;
- two consecutive rounds below 2% or rejected;
- explicit wall-clock/cluster budget exhaustion;
- any correctness, memory, nondeterminism or provider-drift blocker.

The parent/main agent retains final merge authority; automated children may not
push or merge.

### Stage D — local versus cluster validation

#### Local Mac

Use local fresh processes for:

- unit/package/docs checks;
- small/medium LP, SOC, RSOC, SDP, EXP, POWER and mixed cases;
- lifecycle/cache/planner invalidation;
- Modular/4D construction and spec-only tests;
- frozen CSDR alpha3 trajectory and other bounded certified cases;
- quick candidate profiling and falsifying tests.

#### UCAS PBS cluster

Use the cluster only after a candidate passes all local gates:

- large/stress general benchmark cases;
- high-precision and large PSD cases;
- later CSDR fixed-ratio/Nmu convergence cases;
- paired base/candidate fresh-process comparisons on identical hardware.

Cluster rules:

1. inspect live queues/nodes before every wave;
2. deploy one immutable commit-specific source/environment;
3. start with one `ppn=8` production-precision pilot;
4. measure per-process peak RSS and useful concurrency;
5. request an explicit aggregate core budget before scaling;
6. for large independent sweeps, use measured 50–60 core single-node packs
   only when the memory rule permits;
7. keep Julia/BLAS/OMP concurrency within reserved cores and use
   `--gcthreads=1` for Float64x4;
8. use private shards, atomic ledgers, finite retries and artifact validators;
9. never use `ppn=3/4`, overwrite old results, or equate `qsub` success with
   scientific completion.

### Stage E — complete SDPX review, cleanup and documentation

After Stage C/D optimization converges:

1. freeze the best integrated main candidate;
2. perform parallel mathematical, API, numerical, performance, benchmark,
   provider, CI/release and documentation reviews;
3. synthesize a per-file `keep / migrate / delete` inventory;
4. delete only files proven unreferenced by code, tests, workflows and docs;
5. remove stale benchmark frameworks, dead runners, generated manifests,
   obsolete workflows and superseded review prose;
6. rewrite README, API, benchmark, physics-to-convex, precision, threading,
   cluster, optimization and troubleshooting documentation;
7. run full tests, docs, catalog gates, frozen trajectories and main CI;
8. merge cleanup/docs only after review evidence is attached.

### Stage F — long-horizon CSDR convergence — TRANSFERRED (2026-09-02)

**This stage now runs in a dedicated user-owned window. See
`/Users/xuyongjun/Desktop/project/SDPX/CSDR/HANDOFF_CSDR_PROGRAM_20260902.md`
for full state (branch lineage, locked gates, guard commands, env rules).**

CSDR is independent of the Guerrieri S-matrix references. The final solver is
used for paired `g0=c_0_0` zero/twice-subtraction campaigns with a fixed rational
`J/N_a` ladder. Energy discretization (`N_mu`, grid family/map/endpoint policy)
is calibrated separately before the ratio ladder is scaled.

Current scientific evidence:

- tiny zero sampled solve is rejected by held-out-a residual near `543.93`;
- tiny twice sampled solve is rejected near `427.51`;
- these are finite-discretization failures, not accepted bounds;
- no new PBS job has been submitted by this plan.

Phase-2 calibration release, ladder unlock and scheduler remain unavailable
until the local artifact-backed case/finalizer/pair pipeline receives an
independent PASS and a real accepted zero/twice pair exists.

Before production submission, the user/main agent must set:

- explicit `g0_atol`/`g0_rtol` convergence thresholds;
- aggregate cluster core budget after the `ppn=8` memory pilot.

### Stage G — pole-augmented massless-EFT bootstrap benchmark and optimization (added 2026-09-02)

User-selected target:
`/Users/xuyongjun/Desktop/project/Primal bootstap/massless_eft/SDPX_MODELING_AND_OPTIMIZATION_GUIDE.md`
(SHA-256
`aafd4ccccfe00a05717d92061f097521508bdaa4430b889a3aad213c4ea3c372`).
This work is independent of the transferred CSDR zero/twice campaign.

The benchmark must model the finite pole-augmented 4D massless scalar problem:
real coefficients `(alpha_pole, alpha_ab)` and one 3D Lorentz cone for every
even-spin/energy-grid pair. The source PMP block
`[2-Im(T) Re(T); Re(T) Im(T)] >= 0` is exactly equivalent to
`(1, Re(T), 1-Im(T)) in Q3`, hence to `|1+iT| <= 1` for the generated rows.
The guide/report's prose `|1+4iT| <= 1` is not equivalent unless an additional
rescaling is declared; physical normalization remains a blocking decision and
must not be silently chosen.

The exact `g0 = 3*alpha_00 - 3*alpha_pole` objective is consistent across the
PMP, auditor, SDPX and Clarabel paths. For the generator's displayed pair basis
and `sigma2=(s^2+t^2+u^2)/2`, direct Taylor expansion instead gives
`g2 = alpha_10/2 + alpha_20/4 - alpha_11/32 - 3*alpha_pole/8`.
The guide/report/auditor use `-alpha_11/16`, and the current SDPX prototype uses
a second, substantially different wrong expression. No `g2` or directional
benchmark may be certified until that coefficient convention is resolved and
receipts are regenerated. The N14 `g0` extrema below are unaffected.

Independent finite-problem oracle:
`massless_eft/results/pole_ansatz_summary_results.json` SHA-256
`03399fed09bcf535fce61084536ff8c18d9aa260243cd657e89f76f7b69525ba`.
Its audited 1024-bit SDPB rows for `Nmax=14, Lmax=60, ngrid=300, Q=2048` give
`min_g0 = -6.103003970118916970904496194881379...` and
`max_g0 = 3.314872949902211924680613852881385...`, with accepted local grid
audits and scaled gaps below `2.4e-23`. Generator/auditor hashes are
`72e0e5f16c1b0f5f3e671a9f0599d06425977d672437a265fbab1b5f07827fcd`
and `407c4ff154c69989e24a28ec764bcb17dbf7c71d17c3f35858588c318231fbba`.
These receipts certify only that exact finite grid, not the continuum or a
smaller smoke configuration.

Required delivery order:

1. Add `benchmark/bootstrap/physics/massless_eft/` with a deterministic typed
   row generator, exact pair-basis and `g0` map, both normalization alternatives
   recorded fail-closed, source/oracle manifest, original-grid and denser
   held-out unitarity audits, and explicit smoke/train/production resource
   tiers. Keep `g2` diagnostic-only until its source convention is repaired.
2. Start smoke/train rows as build-only unless they own an independent oracle;
   promote only the exact N14 production row after coefficient/fingerprint
   parity with the 1024-bit source and original-coordinate certificate plus
   held-out audit gates pass.
3. Compare the existing symbolic `Model` construction with a reviewed direct
   sparse/native bridge. Do not assume an unreviewed internal IR is public, do
   not parse decimal matrices during timed solves, and do not narrow BigFloat
   rows through Float64.
4. Freeze one excluded warmup process plus three fresh measured processes with
   schema-v9 identities, route/provider/kernel receipts, setup/core/recovery,
   allocations, peak RSS, objective/certificate metrics, and input hashes.
5. Optimize only against that benchmark. Candidate priorities are frontend
   affine-row construction, source-owned sparse SOC blocks, blockwise SOC
   Schur assembly, and MFLA dense factor/solve. Every accepted candidate must
   improve median runtime by at least 2%, preserve the exact finite problem and
   original-coordinate/held-out gates, and pass the broader conic regression
   matrix plus frozen CSDR guard in its separate window.
6. The guide's quoted `1-8 s` solve times are hypotheses, not benchmark
   results. Process crashes remain provider/runtime evidence, and continuum
   extrapolations remain outside solver acceptance.

## Current immediate work

1. Finish the CSDR local-pipeline pair transaction/temp-cleanup review.
   **DONE (2026-09-01)**: 4-round adversarial review loop closed at `b9d6d2d`;
   all P1/P2 findings fixed with receipts; determinism fixed; synthetic
   transaction-only pair fixture proven and isolated; Phase-2/calibration/
   ladder stay KEEP LOCKED pending a real accepted pair.
2. Close dependent-optimizer receipt and runtime workflow findings.
   **PREP COMPLETE**: closure dossier banked (component map, exact
   solve-eligible schema-v9 input contract, fail-closed audit list,
   post-Stage-A checklist). Execution awaits the first solve-eligible V2
   schema-v9 target from Stage A.
3. Rebuild General Benchmark V2 to the reviewed migration contract.
   **IN PROGRESS**: gates + typed V1 results + typed artifacts + 6 certified
   typed LP cases landed on `fix/general-benchmark-v2-review` (pushed).
   Remaining ladder: Chebyshev, ill-conditioned, SOCP, RSOC/SDP/EXP/Power/
   mixed families; fresh-process/peak-RSS/schema-v9 wiring; external
   holdout pinning; provider-backed precision qualification.
4. Integrate approved 4D and Modular branches with the corrected benchmark
   framework. **PREP COMPLETE**: integration dossiers banked for both
   branches (file inventories, entry-point maps, V2 disposition = build_only,
   conflict predictions, P2 notes, ordered checklists). Execution after the
   V2 framework slice lands.
5. Execute Stages B–F continuously, stopping only at a documented safety,
   scientific, resource or user-decision gate.

Session 2026-09-01 parallel-lane ledger (all children commit locally; parent
retains merge/push authority per Stage C):

- MFLA `perf/phase5-kernels` pushed: `5885060` panel threading (bit-identical,
  frozen SHA gate), `50e6e0b` solve-bucket ceiling documentation.
- SDPX `fix/general-benchmark-v2-review` pushed: `4829fcc`/`c88d373`/`50f95d7`
  gates, `ad2bca1` typed V1 results, `2d38fc4` tranche boundary,
  `27e398b` typed artifacts, `85cb289`/`2d7da69`/`2d38b4c`/`ccc6adc`/
  `d804f2b` LP cases + ray contracts, `1c446e2`/`ba0dafa` ill-conditioned,
  `adaf0f8` SOCP, merges `28a8e32` (RSOC/SDP `47db463`) +
  `1d4e6b5` (EXP/Power `a6e3e9c`) + holdout merge `8f9d24d` (`b1f0ae0`/
  `c8f5964`); full Pkg.test green at every merge.
- SDPX `perf/product-hsd-alloc-gate` pushed: `20be893` knob-hoist
  (3.9 KiB/step -> 288 B audit ceiling, frozen SHA bit-identical, Pkg.test
  green).
- Robustness findings banked at `/tmp/robustness-probe/FINDINGS.md`:
  bordered/fixed-trace route sensitivity (P1 - same near-boundary case
  solves in 9 iters on expanded/sparse; 1e-8-perturbed near-rank solves in
  8 bordered iters, so failure is geometry+reduction+route dependent, not
  exact-dependence alone), equality-reduction compatibility (P1),
  residual diagnostics Bool-only (P2), tau_collapse recovery (P2),
  V1-vs-V2 EXP/Power contrast flagged for separate investigation.
- CSDR `feat/g0-convergence-cluster-campaign` local: `78658da`/`804dcb8`/
  `5a0f979`/`c5f0b9d`/`a712dfc`/`088f819`/`b9d6d2d` (pair transaction +
  determinism + final closure, 4 review rounds, receipts recorded above).
- origin/main `8204850` re-verified bit-identical via the unified regression
  guard in a detached worktree.
