# SDPX.jl correctness-first simplification and solver-efficiency plan

**Status:** reviewer plan; implementation has not started

**Updated:** 2026-08-30

**Reviewed source:** 04398a6fa6ac45c8dd0ebbb94a21f30beb6f8edb

**Authority:** this file is the ordered work plan. The mathematical contracts in
docs/design/ remain frozen. This review changes no Julia source, test, project
metadata, solver behavior, or tolerance. Earlier claims in this file that the
public symmetric core, dependency pruning, and final validation were complete
are reset to unverified because the reviewed source and reproduced failures
contradict those claims.

## 1. Required order and stop rules

The work has three sequential implementation phases:

~~~text
Phase 1: fix and certify all confirmed bugs
    |
    | correctness gate
    v
Phase 2: remove duplicate and unreachable structure
    |
    | simplification gate
    v
Phase 3: optimize measured solver bottlenecks
    |
    | performance and release gate
    v
candidate release SHA
~~~

The phases must not overlap:

1. No deletion or architectural consolidation begins while a confirmed bug is
   open.
2. No speed optimization begins while two production paths still own the same
   mathematical operation.
3. No benchmark claim uses the reviewed SHA as a performance baseline because
   that SHA is not correctness-clean.
4. Every solver change follows:

   ~~~text
   reproduce -> add a failing regression -> make one surgical change
   -> run focused verification -> run the complete phase gate
   ~~~

5. Tests, requested tolerances, certificate thresholds, supported cone
   semantics, and expected terminal statuses may not be weakened to make a
   phase pass.
6. A failed gate stops later-phase work. It does not justify a fallback to
   unchecked output, a hidden tolerance change, or a second solver engine.

## 2. Reviewed architecture and current evidence

### 2.1 Live production path

The intended live path is:

~~~text
Model / MOI / CLI / compatibility adapters
                    |
                    v
       compile_product_cone_model
                    |
                    v
 NativeConeProgram: minimize c'x, A*x+s=b, s in K
                    |
                    v
 equality reduction + reconstruction maps
                    |
                    v
         _public_native_hsd_core
                    |
                    v
 ProductConeHSDState + cone runtime
                    |
                    v
 frozen five-equation NewtonSystem
      +-------------+-------------+
      |             |             |
      v             v             v
 symmetric core  expanded KKT  sparse Schur
      |             |             |
      +------ FactorCache/provider+
                    |
                    v
 unregularized five-equation residual gate
                    |
                    v
 line search -> recovery -> original-coordinate certificate
                    |
                    v
                  Result
~~~

The public Model path reaches _public_optimize_native_hsd from
src/public/optimize.jl. However, the repository still includes substantial
compatibility, legacy numerical, duplicate lowering, planning, reference, and
provider scaffolding around that path. Phase 2 must prove reachability before
removing any of it.

### 2.2 Static planning baseline

At the reviewed SHA:

| Item | Observed value |
| --- | ---: |
| Julia files under src/ | 153 |
| Lines in those files | 91,645 |
| Direct includes in src/SDPX.jl | 127 |
| Production-included independent reference | src/cones/nonsymmetric/full_newton_reference.jl |
| Production-included legacy backend | src/la_backends/legacy.jl |
| Optional packages still listed despite prior pruning claim | AppleAccelerate, GenericLinearAlgebra, JLD2 |

These counts are an inventory, not a target to game. The structural objective
is one owner for each responsibility. A 20–30% reduction in production source
and direct includes is a planning estimate only; required code must not be
deleted to hit a percentage.

### 2.3 Confirmed bug register

All locations below refer to the reviewed SHA.

| ID | Priority | Confirmed failure | Primary location | Exit evidence |
| --- | --- | --- | --- | --- |
| C1 | P1 | Scalar bounded LP min x, x >= 0 returns numerical_breakdown; default symmetric core divides by an exact-zero scalar closure and alternate routes see the same rank deficiency | src/kkt/symmetric_core.jl:1679–1696; src/hsd/product_cone_hsd.jl:3101–3113 | Model and MOI scalar regressions are optimal with valid requested-tolerance certificates; compatible and incompatible singular systems are distinguished |
| C2 | P1 | Checked-in power_epigraph_small fails at the default Float64 1e-8 tolerance near convergence; setting all tolerances to 1e-6 masks the failure | src/cones/nonsymmetric/conjugate3.jl:1144–1283; src/cones/nonsymmetric/scaling3.jl:1323–1330 | The unchanged default case and captured kernel fixture pass at 1e-8, with an original-coordinate certificate |
| C3 | P2 | QDLDL refine_once! calls solve! on the provider object instead of the declared provider seam, producing MethodError | src/factor_cache/routes/qdldl_sparse.jl:226–234 | Full prepare/factor/solve/refine lifecycle passes and refinement reduces the original-operator residual |
| C4 | P2 | QDLDL prepare! accepts AbstractFactorRequirements, but standard FactorRequirements is not its subtype; shape fields can also be mutated without rebuilding the frozen provider | src/factor_cache/requirements.jl:21–35; src/factor_cache/routes/qdldl_sparse.jl:140–152 | Standard requirements dispatch works; epoch and frozen-shape invariants are explicit and tested |
| C5 | P2 | Unknown CLI option --bogus=value is silently ignored because the equals branch returns before the option whitelist | bin/sdpx.jl:93–99 | Unknown options fail identically in equals and separated forms before a solve starts |
| C6 | P2 | Equality COO arrays of unequal lengths silently truncate through zip | bin/sdpx_solve.jl:138–142 | All three mismatch directions return a schema error before allocation or solve |

The package E2E run at review time had 18 passing assertions and three failing
assertions, all belonging to the Power example. The scalar issue was reproduced
through the public Model API and MOI. The QDLDL and input issues were reproduced
through their direct public or protocol surfaces.

Lower-priority ingestion findings are tracked behind C6:

- duplicate COO coordinates currently overwrite rather than following an
  explicit schema rule;
- repeated coefficient entries for the same variable overwrite even though the
  bridge schema describes one entry per variable.

They should be fixed as input hardening after the six confirmed regressions,
not mixed into an unreviewed semantic change.

## 3. Non-negotiable mathematical and software invariants

Every phase preserves the following:

1. Canonical form remains A*x+s=b with the sign conventions frozen in
   docs/design/CANONICAL_FORM.md and docs/design/HSD_FORMULATION.md.
2. Every direction satisfies all five unregularized equations in
   docs/design/NEWTON_SYSTEM.md. A compatible singular system may choose a
   deterministic gauge, but it may not skip an equation.
3. Predictor and corrector share one valid factor epoch. A stale receipt,
   invalidated factor, or factor from another pattern is never reused.
4. Route fallback is same-iterate, explicit, recorded, and truthful about the
   attempted and executed routes.
5. Only original-coordinate verification may promote Optimal,
   PrimalInfeasible, or DualInfeasible. Raw tau/kappa, factor success, iteration
   count, or provider status is not a certificate.
6. Float64, MultiFloat, and BigFloat work is performed in the declared
   arithmetic. No hidden Float64 narrowing is permitted.
7. Nonfinite values, invalid dimensions, malformed input, unsupported provider
   capability, and unresolved numerical conditions fail closed.
8. The common step length, cone-interior tests, Power membership, and public
   requested tolerances are unchanged unless a separate mathematical design
   review explicitly changes their contract.
9. Independent reference implementations remain independent evidence. If
   removed from production loading, they move to validation rather than being
   deleted.
10. No regression test is deleted, skipped, relabelled, or made less strict
    during code reduction or optimization.

## 4. Phase 0 — freeze evidence before implementation

Phase 0 is review and measurement setup, not a solver change.

### 4.1 Record the environment

For the bug-fix branch, record:

- source SHA and clean/dirty status;
- Julia version and startup flags;
- operating system, CPU, and available memory;
- Julia, BLAS, provider, and process thread counts;
- SuiteSparse/CHOLMOD, UMFPACK, QDLDL, MFLA, and BFLA versions when present;
- arithmetic type and precision;
- complete resolved Settings and execution route;
- model fingerprint and reconstruction-map fingerprint.

Artifacts should be stored under a SHA-named review/benchmark directory so a
later candidate cannot overwrite the defect evidence.

### 4.2 Freeze one minimal reproduction per bug

Before each fix, add a regression that fails for the reviewed reason:

- C1: public Model, MOI GreaterThan, and direct symmetric-core compatible
  singular fixtures;
- C2: checked-in power_epigraph_small plus the exact failed root tuple;
- C3: a valid QDLDL factorization followed by refine_once!;
- C4: prepare!(cache, FactorRequirements(...)) and a shape-change attempt;
- C5: unknown option in both CLI spellings;
- C6: each equality-triplet array shorter than the other two.

Each reproduction records terminal status, typed reason, route attempts,
iteration, factor/solve/refinement counters, and certificate residuals where
applicable. Exception text alone is insufficient evidence.

### 4.3 Baseline gate

Phase 1 starts only when:

- each regression fails for its expected reviewed reason;
- the existing failures have not been hidden by a dependency or environment
  change;
- no unrelated source edits are in the bug-fix branch; and
- the seven checked-in public E2E cases and the current small-case inventory
  have recorded expected outcomes.

## 5. Phase 1 — fix correctness bugs

Fix C1 through C6 as separate, reviewable patches. C1 and C2 block all other
solver development because they affect public correctness. C3 and C4 should be
reviewed together at the phase gate but remain separate commits. C5 and C6 are
schema-boundary fixes and must not invoke the solver on rejected input.

### 5.1 C1 — handle the compatible singular scalar closure

#### Cause

The reduced scalar equation in the symmetric core is:

~~~text
D * d_tau = N

D = kappa + tau * (dot(c_r, u_x) + dot(b, u_y))
N = rhs.tau_kappa
    - tau * (rhs.homogeneous_gap + dot(c_r, w_x) + dot(b, w_y))
~~~

The current code treats an exactly zero D as an unconditional exception. For
the scalar bounded LP, D=0 and N=0 can describe a compatible rank-zero scalar
closure: d_tau is a gauge coordinate and the five-equation system has a family
of valid directions. Expanded and sparse routes encounter the same structural
rank deficiency, so replacing one factorization alone will not solve it.

The existing validation intentionally fails closed on a zero denominator. It
must be refined to distinguish compatible rank-zero and incompatible
rank-deficient closures; the safety check must not simply be removed.

#### Required design

Create one scalar-closure classifier shared by the general symmetric core and
the fixed-trace Q3 specialization:

~~~text
classify D and N with independent accumulated-work error bounds

if D is resolvably nonzero
    d_tau = N / D
    classification = regular
elseif D is unresolved at zero and N is independently compatible with zero
    d_tau = 0
    classification = compatible_singular_gauge
else
    reject as incompatible or insufficient-precision scalar closure
end

recover d_x, d_y, d_s, d_kappa
accept only if all five unregularized Newton residual groups pass
~~~

Implementation requirements:

1. Compute separate actual-work/roundoff bounds for D and N. Do not use the
   public optimization tolerance, a hard-coded epsilon multiplier detached from
   work, or max(1, scale).
2. Use a deterministic gauge. d_tau=0 is the simplest exact rank-zero gauge;
   compare it against an independent full-Jacobian QR/SVD oracle before making
   it production policy.
3. Reconstruct the complete direction before accepting it.
4. Require the existing five-equation residual verifier using the original
   unregularized operator.
5. Preserve a typed distinction among regular, compatible singular,
   incompatible singular, and insufficient precision.
6. Do not let the dispatcher convert the expected compatible condition into
   symmetric_core_dispatch_exception.
7. Use the same closure helper in
   src/kkt/specializations/fixed_trace_q3.jl so the denominator policy cannot
   diverge.
8. For explicit expanded and sparse_schur requests, either implement the same
   gauge or make a recorded same-iterate fallback to the corrected symmetric
   core. The result must identify both attempted and executed routes.
9. Keep factor-epoch and predictor/corrector reuse rules unchanged.

#### Required tests

- Public Float64 Model for min x subject to x>=0 returns Optimal, x is near
  zero, and the original-coordinate certificate meets the requested tolerance.
- The equivalent MOI scalar GreaterThan model has the same outcome.
- Direct exact compatible, exact incompatible, near-compatible, and
  near-incompatible scalar-closure fixtures exercise all classifications.
- An independent full five-equation oracle confirms that the selected gauge is
  in the solution set.
- Scaling b and c across powers of two does not change certificate validity or
  classification solely because of units.
- Minimal Nonnegative, SOC, and 1x1 PSD cases cover the same structural edge.
- Default, expanded, and sparse routes either succeed directly or report their
  same-iterate fallback truthfully.
- A failed residual gate remains a failure; it is not converted to a usable
  direction.

#### C1 exit gate

C1 is complete only when the public and MOI regressions pass, the compatible
and incompatible oracle fixtures pass, all five-equation validations remain
green, and no tolerance or residual threshold changed.

### 5.2 C2 — make the Power conjugate root representability-aware

#### Cause

_ns_conjugate_gap_root maintains a certified bracket, but the current
next==current path reports root_resolution_limited and fails unconditionally.
At the default outer tolerance, power_epigraph_small reaches a near-boundary
dual point where Float64 has no representable point between the current
candidate and the requested update. The 1e-6 run stops earlier and therefore
does not prove that 1e-6 is the correct public default.

try_update_nonsymmetric_scaling! fails immediately when the conjugate update
fails. The later dual-Hessian retry policy does not cover this failure. The
repair must therefore decide whether the best representable root is
mathematically certifiable, not merely retry another unchecked scaling mode.

#### Required investigation

Capture the first failed Power update with:

- cone exponent, primal and dual block values;
- lower and upper bracket endpoints;
- current and proposed root;
- function value, derivative, and monotonicity sign;
- seed source and iteration/bisection counts;
- nextfloat/prevfloat neighbors;
- accumulated roundoff and attainable residual floor;
- Cartesian reconstruction result;
- Hessian factor/inverse, shadow, metric, and secant residuals;
- outer five-equation residual and line-search result.

Turn this tuple into a deterministic local kernel fixture before editing the
root solver. Compare it with a 256-bit or higher BigFloat oracle.

#### Required design

Keep the certified bracket and safeguarded Newton/bisection logic. Add a
representability-aware terminal branch:

~~~text
update and verify the certified bracket

if the ordinary root gates pass
    accept current

next = safeguarded Newton or midpoint candidate

if next == current
    candidates = finite representable members among:
                 current, lower endpoint, upper endpoint,
                 prevfloat/current, nextfloat/current
    candidate = member with the smallest certified residual

    if the bracket width and residual are at the arithmetic attainability floor
       and Cartesian reconstruction is certified
       and shadow/Fenchel identities are certified
       and Hessian factor, inverse, metric, and secant checks pass
        accept candidate with resolution_limited_but_certified diagnostics
    else
        fail closed as insufficient precision
    end
end
~~~

Implementation requirements:

1. Derive the attainable residual from local spacing, the derivative, and
   accumulated work. Do not substitute the outer solver tolerance for the root
   certificate.
2. Preserve bracket ordering and cone-domain membership for every candidate.
3. Accept a resolution-limited candidate only after all existing
   reconstruction and differential certificates pass.
4. Pass the resulting direction through the unchanged five-equation residual
   gate.
5. Promote the final result only if the original-coordinate certificate meets
   the unchanged public 1e-8 request.
6. Map true arithmetic exhaustion to a typed InsufficientPrecision-style
   reason rather than generic numerical_breakdown.
7. Check whether the current outer iterate is already terminally certifiable
   before starting an unnecessary new scaling epoch. This check may avoid the
   failing update only when the full original-coordinate certificate already
   passes.
8. Do not globally relax auto_tolerance, Power membership, line-search,
   Hessian, or certificate thresholds.
9. Do not silently report a 1e-6 certificate for a requested 1e-8 solve. If a
   model is genuinely unattainable in Float64 after the local repair, return a
   truthful precision failure or require higher precision.

#### Required tests

- power_epigraph_small passes with default Float64 settings and its current
  objective tolerance.
- The final original-coordinate certificate satisfies the requested 1e-8
  limits.
- The explicit 1e-6 case remains valid but is not used as the default.
- The captured failing root tuple agrees with the high-precision oracle.
- Exponent sweeps include values around 0.35 and 0.5 and values nearer 0 and 1.
- Data are scaled over powers of two; both signs relevant to the third
  coordinate and prevfloat/nextfloat boundary points are covered.
- Non-interior, nonfinite, and truly unresolved cases still fail closed.
- Float64, supported MultiFloat types, and BigFloat preserve their arithmetic.
- Existing power_geomean and simple Power cases do not regress.
- No medium or large case is relabelled Optimal unless its certificate passes.

#### C2 exit gate

All seven checked-in E2E cases must pass with their existing requested
tolerances and expected statuses. The focused Power conditioning and captured
root validations must pass before C3–C6 are allowed into the phase-complete
candidate.

### 5.3 C3 — call the QDLDL provider refinement seam

#### Cause

refine_once! calls solve!(cache.provider, ...), but QDLDL providers are
abstracted by _qdldl_provider_solve!. The direct provider call has no matching
method and produces MethodError after an otherwise valid factorization.

#### Required change

The future patch should:

1. require a Fresh factor state and matching epoch;
2. validate residual and correction dimensions;
3. reject nonfinite residual input;
4. call _qdldl_provider_solve!(cache.provider, correction, residual);
5. reject a nonfinite correction;
6. update refinement counters only after success;
7. preserve the factor receipt and Fresh authority; and
8. define whether a provider refinement increments only refine_count or both
   refine_count and solve_count, then apply the same convention to every cache
   route.

The refinement test must form the residual with the original unregularized
operator and show residual contraction against a dense reference solution.

#### Required tests

- Factor, solve, compute an original-operator residual, refine once, and compare
  the correction with a dense reference.
- Exercise Float64 and installed Float64x2/Float64x4 providers.
- Cover unprepared, stale, failed, invalidated, wrong-dimension, and nonfinite
  states.
- Assert factor, solve, and refinement counter semantics.

### 5.4 C4 — repair FactorRequirements dispatch and frozen-shape ownership

#### Cause

QDLDL prepare! accepts AbstractFactorRequirements, while the standard
FactorRequirements does not subtype it. In addition, prepare! can mutate cache
dimension/shape metadata even though the provider and sparse pattern were
constructed for a frozen shape.

#### Required change

Make the documented type hierarchy real:

~~~text
abstract type AbstractFactorRequirements end
struct FactorRequirements <: AbstractFactorRequirements
    ...
end
~~~

Then make QDLDL preparation enforce one explicit ownership policy:

- same frozen dimension and pattern: update the symbolic/factor epoch and reset
  factor state as required;
- changed dimension or pattern: reject with a typed shape/pattern error, or
  construct a new cache through an explicit rebuild operation;
- never mutate cache.n or prepared_shape while retaining a provider built for
  the previous shape.

Do not add a QDLDL-only union signature that leaves the common protocol
internally inconsistent.

#### Required tests

- prepare!(cache, FactorRequirements(n, epoch)) dispatches successfully.
- A same-shape new epoch has defined state-reset behavior.
- A dimension or sparsity-pattern change cannot leave a seemingly Fresh stale
  provider.
- The complete lifecycle passes:

  ~~~text
  construct -> prepare -> factor -> solve -> refine -> invalidate
  -> stale solve rejection
  ~~~

QDLDL remains an optional validation route after C3/C4. It is not promoted to a
performance route until its quasi-definite diagonal, regularization, inertia,
and original-operator refinement preconditions are proven for SDPX's KKT
matrix.

### 5.5 C5 — validate CLI option names before value syntax

#### Cause

The equals-form branch returns before the shared option whitelist, so unknown
keys in --key=value form are discarded while the separated form is rejected.

#### Required change

Use one token-normalization and validation path:

~~~text
normalize token into key and optional inline value
classify key as value option, flag option, or unknown
reject unknown
validate whether this key permits an inline/separate value
consume and convert the value
~~~

Keep flag grammar explicit: flags such as --help and --quiet should accept only
their documented form. A parser error must occur before model loading, solve
execution, or result-file creation.

#### Required tests

- --typo=value and --typo value fail with the same unknown-option class.
- Known value options such as --precision=840 and --precision 840 are
  equivalent.
- Missing values, extra values on flags, invalid conversion, --help, and
  --quiet follow documented behavior.
- Parser rejection proves that no solver call and no output write occurred.

Add a focused bin/test/runtests.jl suite rather than relying on manual CLI
execution.

### 5.6 C6 — validate equality COO arrays before iteration

#### Cause

The block-matrix helper checks triplet lengths, but equality ingestion uses zip
directly. zip stops at the shortest input and silently drops data.

#### Required change

Use one shared checked-triplet routine for block matrices, constants, and
equalities:

1. read rows, columns, and values;
2. require all three lengths to be equal before zip, allocation fill, index
   conversion, or solve;
3. require integer indices in bounds;
4. require finite numeric values in the declared arithmetic;
5. apply one documented duplicate-coordinate policy; and
6. return a structured bridge schema error without invoking the solver.

The recommended duplicate policy at the external schema boundary is
fail-closed rejection. If additive COO semantics are desired, that must be a
separate schema decision with deterministic accumulation and documentation; it
must not emerge accidentally from the bug fix. Repeated variable entries
within one coefficient block should likewise be rejected if the one-entry
schema is retained.

#### Required tests

- rows short, columns short, and values short each fail independently.
- Empty but consistently sized arrays follow the documented empty-matrix rule.
- Out-of-range, noninteger, nonfinite, duplicate-coordinate, and repeated
  variable cases fail with specific schema reasons.
- Valid equality and block triplets produce identical canonical matrices before
  and after the refactor.
- An injected solve spy confirms no solver call on malformed input.

### 5.7 Phase 1 correctness gate

Phase 1 is complete only when all of the following are true on one clean SHA:

- all six minimal regressions are green;
- all seven existing E2E cases pass with unchanged expectations;
- the scalar Model and MOI cases are included in black-box coverage;
- the current benchmark/general small inventory (17 cases at review time)
  produces its expected statuses and valid certificates;
- every claimed optimal or infeasible result passes original-coordinate
  verification;
- route diagnostics, fallbacks, epochs, and factor counters are truthful;
- no tolerance, certificate gate, or expected result was loosened;
- malformed CLI/bridge input cannot reach model construction or the solver; and
- all focused commands in Section 8 pass.

Only after this SHA is frozen may Phase 2 begin.

## 6. Phase 2 — reduce code and simplify ownership

Phase 2 is behavior-preserving deletion and consolidation. It is not a rewrite.
Every batch must be independently reviewable, net-negative in production code,
and followed by the complete Phase 1 gate.

### 6.1 Build a liveness and ownership map

Before deleting code, generate a symbol/include map rooted at:

- exports in src/SDPX.jl;
- Model optimize!, result accessors, and Settings;
- the MOI Optimizer;
- bin/sdpx.jl and bin/sdpx_solve.jl;
- benchmark/general and validation entrypoints;
- documented qualified SDPProblem and ConicProblem compatibility calls;
- downstream CSDR usage.

CSDR currently uses qualified SDPX.ingest, SDPX.SDPProblem, SDPX.prepare,
SDPX.solve!, SDPX.solve_socp, and fixed-trace Q3 helpers. Non-exported does not
mean unused. Before removing any such surface, choose and verify one of:

1. migrate CSDR to Model/NativeConeProgram first; or
2. retain a thin compatibility adapter that constructs Model/Settings and calls
   the single product-HSD engine.

The liveness report must classify each candidate as:

- live production owner;
- compatibility adapter;
- validation/reference oracle;
- unreachable implementation;
- optional extension;
- downstream-only qualified API; or
- unknown, requiring more evidence.

Unknown code is not deleted.

### 6.2 Remove reference code from production loading, not from evidence

First low-risk batch:

- move src/cones/nonsymmetric/full_newton_reference.jl to validation ownership;
- move any retained Cartesian/Newton oracle embedded in conjugate3.jl to a
  focused validation file if production does not call it;
- remove their production includes;
- keep oracle tests independent of production helpers.

Gate:

- using SDPX, package precompilation, Power validation, all E2E cases, and
  original-coordinate certificates are unchanged;
- validation invokes the moved oracle directly; and
- cold-load changes are recorded but not yet advertised as a speed result.

Audit the actual Project.toml weakdeps and ext/ implementations for
AppleAccelerate, GenericLinearAlgebra, and JLD2. The old plan incorrectly said
they were already pruned. Remove an extension only if:

- no public capability or documented workflow requires it;
- no downstream package uses it;
- absence behavior is explicit;
- package loading and provider matrices pass; and
- corresponding documentation is updated in the same batch.

### 6.3 Remove the unreachable duplicate public-lowering path

The reviewed Model path calls _public_optimize_native_hsd directly. Static
review found no live Model caller for the older family-lowering/result path.
Candidates include:

- src/ir/lower_lp.jl;
- src/ir/lower_sdp.jl;
- src/ir/lower_soc.jl;
- src/ir/lift_psd.jl;
- dead family-start, lowering, and result adapters in
  src/public/optimize.jl.

Before deletion:

1. prove no export, MOI path, CLI path, benchmark, example, or qualified
   compatibility entrypoint calls the candidate;
2. distinguish canonicalization/reconstruction helpers used by the live native
   path from dead family dispatch;
3. add public smoke coverage for LP, SOC, PSD, Exp, and Power;
4. add a PSD model that would reveal an accidental return of the PSD-lift path.

After deletion, every frontend must reach the same NativeConeProgram,
reconstruction, and certificate path.

### 6.4 Make one bordered core own the public route

After C1 is proven, make the prepared symmetric augmented core the sole
implementation of public :bordered. Remove the second full-border numerical
implementation and duplicate ownership only after route parity.

Candidates for consolidation include:

- SymmetricBorderedWorkspace and its duplicate assembly/factor/recovery flow;
- duplicate scalar d_tau recovery;
- duplicate factor-receipt and refinement bookkeeping;
- broad try/catch-to-Bool dispatch that erases typed failure reasons;
- legacy bordered fallback state in ProductConeHSDState.

Target route policy:

~~~text
:bordered     -> symmetric augmented core
:expanded     -> exact expanded route
                 -> recorded same-iterate symmetric-core fallback
:sparse_schur -> sparse reduced route
                 -> expanded route
                 -> recorded same-iterate symmetric-core fallback
~~~

Keep expanded and sparse routes as public expert routes and independent
correctness evidence while they remain documented. Do not remove them merely
because the default route works.

Replace ambiguous Boolean direction results with one typed result carrying:

- requested, attempted, and executed route;
- failure stage and typed reason;
- pattern/factor epoch;
- provider and arithmetic;
- whether a compatible singular gauge was selected;
- fallback chain; and
- residual-gate outcome.

Route parity must cover LP, SOC, PSD, Exp, Power, Zero, and mixed products.

### 6.5 Collapse settings and planning to one chain

Settings and policy are spread across frontend/solve_options.jl,
midend/resolve_options.jl, public/settings.jl, pipeline/, types/plans.jl, and
private native-HSD planning.

Target ownership:

~~~text
public Settings
      |
      v
ResolvedSettings        # all defaults and capability checks exactly once
      |
      v
ExecutionPlan           # immutable route/provider/precision/thread decisions
      |
      v
runtime workspaces      # execute; do not reinterpret policy
~~~

Required consolidation steps:

1. inventory every default, alias, environment override, and compatibility
   translation;
2. define one canonical name and validation point for each option;
3. make arithmetic/provider capability resolution explicit;
4. remove downstream re-resolution and private shadow defaults;
5. ensure diagnostics publish the immutable resolved values;
6. delete old option/planner types only after serialized settings, CLI, MOI,
   Model, and compatibility parity.

Do not introduce a generic framework or macro layer merely to reduce file
count. The result should be fewer policy owners and less dispatch, not hidden
indirection.

### 6.6 Consolidate the FactorCache/KKT protocol

After C3/C4, all factor routes should implement one explicit lifecycle:

~~~text
prepare! -> factorize! -> solve! / solve_many! -> refine_once!
         -> invalidate! -> stale-use rejection
~~~

Shared invariants:

- frozen dimension and structural pattern;
- monotonically owned symbolic/factor epoch;
- provider-owned factor and solve storage;
- dimension and finiteness checks at the protocol boundary;
- original-operator refinement residual;
- truthful counters and receipts;
- no solve after failed factorization or invalidation.

Use a common conformance harness for CHOLMOD, dense high-precision, QDLDL, and
other installed routes. Share validation helpers where this removes repeated
code without adding runtime dynamic dispatch. Avoid a macro-heavy provider
rewrite.

Only after the protocol map is complete should the old KKTBackend/la_backend
and Workspace numerical layers be considered for removal.

### 6.7 Isolate compatibility adapters and retire legacy numerics in batches

Potential legacy numerical clusters include:

- src/kkt.jl;
- src/schur.jl;
- src/workspace.jl;
- src/kernels/threaded.jl;
- src/kernels/mixed_precision_kkt.jl;
- src/kkt_backend.jl;
- old pipeline/midend preprocessing execution;
- src/prepared.jl;
- src/la_backends/legacy.jl.

Do not delete this list as one patch. For each cluster:

1. identify its data types, ingestion functions, preprocessing, numerical
   kernels, and result adapters separately;
2. retain or migrate data/ingestion pieces needed by CSDR;
3. prove whether solve!/prepare currently converts to Model and calls product
   HSD;
4. compare canonical A, b, c, cone order, reconstruction maps, and result
   certificates before and after adapter simplification;
5. remove only the unreachable numerical owner;
6. run a downstream CSDR compatibility smoke before the next cluster.

The prepared compatibility path currently risks doing legacy preprocessing and
then repeating canonical reduction in product HSD. Phase 2 should choose one
canonicalization owner and remove the duplicate pass while preserving exact
model and reconstruction fingerprints. The runtime benefit is measured only in
Phase 3.

### 6.8 Simplify state, diagnostics, and validation ownership

HSDState forwards many fields through Base.getproperty to a bordered workspace.
Migrate callers to explicit owners and remove transition forwarding once call
sites are covered.

Retain the useful conceptual split:

~~~text
HSDState            = mathematical iterate, residuals, tau/kappa
ProductConeHSDState = cone runtime, route workspace, execution receipts
~~~

Do not merge these large types unless profiling later proves a concrete cost.
Explicit field ownership is the Phase 2 objective.

Consolidate:

- one original-coordinate result/certificate adapter;
- one five-equation residual verifier used by all routes;
- one route/fallback diagnostic schema;
- one factor receipt schema.

Move validation-only state and independent oracles out of production includes.
Do not couple the production verifier to its reference oracle.

### 6.9 Phase 2 simplification gate

Every deletion batch must satisfy:

- the complete Phase 1 correctness gate;
- package load/precompile and public API smoke;
- MOI, CLI, bridge, SDPProblem, ConicProblem, and CSDR compatibility smoke;
- identical expected statuses and certificate validity;
- no new alternate solver engine, speculative abstraction, or hidden fallback;
- net-negative production LOC and include count for the batch;
- documentation and Project.toml/ext metadata match the source;
- independent oracles remain runnable under validation/; and
- git diff shows one bounded ownership change.

Phase 2 is complete when the qualitative targets hold:

1. one production product-HSD engine;
2. one Settings -> ResolvedSettings -> ExecutionPlan chain;
3. one five-equation residual authority;
4. one FactorCache lifecycle;
5. one public canonicalization/reconstruction/certificate path;
6. thin, tested compatibility adapters rather than a second numerical stack;
7. reference mathematics is not loaded in production; and
8. no duplicate bordered implementation owns the public default.

Freeze this clean SHA and its source/include inventory. Only then begin
performance work.

## 7. Phase 3 — improve solver speed from evidence

### 7.1 Freeze the post-reduction baseline

Record a new baseline from the clean Phase 2 SHA. Before/after runs must use the
same:

- Julia build and startup configuration;
- machine, power mode, and CPU affinity where controllable;
- Julia, BLAS, and provider thread counts;
- package environment and provider versions;
- arithmetic and precision;
- requested tolerances and iteration limits;
- route, fallback policy, and model fingerprint;
- warm-up policy and sample count.

Separate:

- cold process: package load, precompile/cache state, first model build, first
  solve;
- warm solve: already-loaded package with compiled methods;
- repeated solve: same structure with new numeric data, where supported.

Use at least five fresh-process cold samples and ten hot samples after one
unrecorded warm-up. Report median and IQR/MAD, not the best run. Run paired
baseline/candidate samples on the same host and alternate their order when
practical.

### 7.2 Benchmark matrix

The release matrix should include:

- C1 scalar regression;
- all seven black-box E2E cases;
- the full benchmark/general small inventory;
- representative medium/large LP, SOCP, SDP, Exp, and Power cases;
- power_epigraph_small and at least one medium Power case;
- small/medium PSD MaxCut or equivalent PSD panel cases;
- sparse Schur fixtures;
- QDLDL factor/solve/refine microbenchmarks, still optional;
- CSDR fixed-trace Q3 small and medium workloads;
- Float64, supported Float64x2/Float64x4, and BigFloat cases.

For cluster campaigns, run local small/medium correctness and resource probes
first. Local and cluster artifacts must identify the same clean SHA.

### 7.3 Metrics and attribution

For every sample record:

- terminal status, objective, primal/dual gap, and certificate residuals;
- total, setup, equality/rank, symbolic, numeric assembly, factorization,
  predictor solve, corrector solve, refinement, line search, state update, and
  certificate time;
- iteration, factorization, solve, refinement, fallback, and root-iteration
  counts;
- Julia allocations, allocated bytes, GC time, and peak RSS;
- factor pattern and epoch identifiers;
- route/provider/arithmetic/thread decisions;
- cold load and time-to-first-result separately.

No optimization is selected from total wall time alone. The phase trace must
identify a dominant, repeatable component first.

### 7.4 Performance work package P1 — cold load and first solve

Hypothesis: production loading and inference still pay for reference files,
dead compatibility paths, and optional extensions.

Candidate actions:

- verify that Phase 2 removed reference and dead-engine includes;
- inspect invalidations and excessively broad method signatures;
- precompile only stable public entrypoints and representative cone/runtime
  types;
- delay optional provider initialization until selected;
- avoid loading checkpoint/benchmark-only dependencies in the solver path.

Acceptance:

- at least a 20% project target reduction in median time-to-first-result on a
  fresh depot/cache protocol, if cold load is a measured user bottleneck;
- no hot-solve regression over 5%;
- package load and every optional-provider absence/presence case remain
  correct.

### 7.5 Performance work package P2 — canonicalization and prepared reuse

Hypothesis: compatibility prepare/solve and repeated solves redo equality,
presolve, reconstruction, and symbolic planning work.

Candidate actions:

- cache immutable canonical structure, cone layout, equality/rank map,
  reconstruction map, and structural fingerprint;
- separate structural preparation from numeric b/c/block updates;
- ensure CSDR compatibility adapters do not preprocess and then invoke a second
  canonical reduction;
- invalidate the cache on any structural change and prove the fingerprint catches
  it.

Acceptance:

- canonical A, b, c, cone order, and reconstruction fingerprints match the
  uncached path;
- repeated numeric solves skip only structure-invariant work;
- no stale map can certify a changed model;
- setup-time and allocation improvement is repeatable on prepared CSDR and
  repeated general cases.

### 7.6 Performance work package P3 — KKT assembly and factor reuse

Hypothesis: numeric refill, global Theta materialization, or duplicate
factorization dominates medium cases.

Candidate actions:

- retain one frozen CSC/panel pattern per structural epoch;
- write cone-block numeric contributions directly into owned slots;
- avoid a global dense Theta when block-to-core assembly is sufficient;
- factor exactly once per HSD predictor/corrector epoch;
- reuse that factor for homogeneous, predictor, and corrector right-hand sides;
- batch multiple RHS solves only when the provider exposes a supported,
  allocation-safe interface;
- keep refinement against the original unregularized operator.

Acceptance:

- one factorization per intended epoch, with truthful counters;
- no stale factor use after pattern/numeric invalidation;
- unchanged five-equation residuals and certificates;
- measurable factor/assembly reduction on the cases where profiling identified
  the bottleneck;
- no provider-internal API reach-through.

### 7.7 Performance work package P4 — fused direction and residual work

Hypothesis: A*dx, transpose products, cone actions, and scalar reductions are
recomputed by direction construction, acceptance, refinement, and line search.

Candidate actions:

- activate or complete a state-owned fused direction workspace;
- compute each matrix/cone action once per candidate direction;
- reuse results in all residual groups without aliasing;
- use fixed-size/static scratch only where dimensions are truly fixed;
- hoist route selection outside the hot iteration loop if profiling shows
  dispatch cost and same-iterate fallback remains possible.

Acceptance:

- term-by-term parity with the independent residual oracle;
- no borrowed scratch survives the owning epoch;
- allocation reduction is visible in
  benchmark/general/performance/hsd_allocation.jl;
- wall-time gain is repeatable, not merely fewer allocations in a
  non-dominant path.

### 7.8 Performance work package P5 — Exp/Power cone kernels

This package starts only after C2 is complete.

Hypothesis: repeated scalar root evaluations, seed reconstruction, and
small-vector allocations dominate nonsymmetric scaling/line search.

Candidate actions:

- retain certified bracket/seed state when mathematical state and epoch match;
- cache repeated scalar evaluations within one root solve;
- use allocation-free fixed-size arithmetic for three-dimensional cone blocks;
- avoid recomputing Hessian/shadow quantities already certified for the same
  point;
- preserve the representability and fail-closed logic from C2.

Acceptance:

- identical membership, root, Hessian, secant, and certificate outcomes;
- captured boundary fixtures remain certified or fail with the same truthful
  reason;
- fewer root evaluations or allocations correspond to a repeatable Exp/Power
  solve-time gain;
- no relaxed root or outer tolerance.

### 7.9 Performance work package P6 — PSD panels and sparse incidence

Hypothesis: symmetric cone action, congruence transforms, and sparse-to-panel
gather/scatter dominate SDP workloads.

Candidate actions:

- operate on the owned triangle and mirror only at defined boundaries;
- batch BLAS-compatible panel operations at the declared arithmetic;
- reuse eigen/factor scratch by block size and epoch;
- precompute sparse incidence maps and numeric slot ownership;
- choose dense versus sparse panel treatment from measured fill and dimensions,
  not a broad heuristic copied across arithmetic types.

Acceptance:

- symmetry, PSD interior, scaling identities, and five-equation residuals
  remain within the same limits;
- Float64 and high-precision results agree with their reference tolerances;
- medium PSD cases show the expected phase-specific gain;
- memory/RSS does not grow without bound across repeated solves.

### 7.10 Performance work package P7 — high-precision ownership

Hypothesis: temporary creation, alias repair, and precision conversion cause
allocation and RSS growth in MultiFloat/BigFloat runs.

Candidate actions:

- use provider-owned factors and solve buffers;
- pre-size state-owned scratch at the declared precision;
- eliminate accidental Float64 literals/conversions in hot kernels;
- avoid repeated BigFloat construction and precision resets;
- test aliasing explicitly for RHS, correction, and factor buffers.

Acceptance:

- no numerical narrowing;
- stable precision metadata and provider selection;
- bounded RSS across repeated solves;
- unchanged high-precision certificates;
- a measured reduction in allocations or phase time on representative
  high-precision cases.

### 7.11 Performance work package P8 — parallelism

Parallelism is last, after the serial dataflow is allocation- and
ownership-clean.

Candidate actions:

- parallelize independent cone blocks or PSD panels only when profiles show
  sufficient granularity;
- permit one active parallel layer at a time: Julia tasks or BLAS/provider
  threads;
- use deterministic reductions for residual and certificate quantities;
- avoid task creation inside tiny three-dimensional cone kernels.

Acceptance:

- serial reference results and certificates remain the authority;
- repeated parallel runs are deterministic within declared arithmetic bounds;
- speedup is reported against the same total core allocation;
- no oversubscription, deadlock, or increased peak memory that erases the gain.

### 7.12 Performance work package P9 — provider and route promotion

QDLDL or another provider is promoted only after the common conformance harness
and KKT precondition proof pass.

Required evidence:

- structural diagonal/quasi-definiteness assumptions match the assembled KKT;
- regularization and inertia policy are explicit;
- factor and solve residuals are checked against the original operator;
- refinement and stale-epoch behavior pass;
- route fallback remains truthful;
- the provider wins representative sparse cases after setup cost, not only a
  microbenchmark.

No provider becomes the default solely because its MethodError was fixed.

### 7.13 Phase 3 performance and release gate

Every optimization is one hypothesis and one bounded patch. It is retained only
when:

- the complete Phase 1 and Phase 2 gates remain green;
- statuses, expected objectives, and certificates are unchanged or stronger;
- requested tolerances and arithmetic are identical before and after;
- a claimed speed improvement is repeatable, normally at least 10% in median
  with a materially shifted IQR/MAD on the targeted workload;
- the project aims for at least 20% on the dominant representative medium/large
  bottleneck, but correctness is never traded to meet that target;
- no release workload regresses more than 5% in median time, allocations, or
  peak RSS without an explicit reviewed tradeoff;
- factorization/fallback/root counters explain the gain;
- fixed-width warm-step benchmarks that claim allocation-free execution show
  zero Julia allocations across ten samples;
- BigFloat/MultiFloat repeated-run RSS is stable; and
- benchmark artifacts identify the clean candidate SHA and environment.

An optimization that fails its focused benchmark, worsens fallback frequency,
or weakens numerical evidence is reverted independently.

## 8. Verification matrix and commands

Commands are run from the SDPX.jl package root unless noted.

### 8.1 Black-box public behavior

~~~sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. benchmark/general/test_small.jl
~~~

The tests must assert expected status and original-coordinate certificate, not
only absence of an exception.

### 8.2 Newton and symmetric-core evidence

~~~sh
julia --project=. validation/newton_system_reference.jl
julia --project=. validation/symmetric_core_reference.jl
julia --project=. validation/product_hsd_symmetric_state.jl
julia --project=. validation/product_hsd_symmetric_shadow.jl
~~~

Add the C1 compatible/incompatible closure fixtures to this layer and the
public scalar regression to Pkg.test().

### 8.3 Nonsymmetric cone evidence

~~~sh
julia --project=. validation/power_core_conditioning.jl
julia --project=. validation/power_core_dual_hessian_experiment.jl
~~~

Add a focused captured-root validation that checks Float64 against a
high-precision oracle. The experiment file is evidence only; it must not become
a hidden production fallback.

### 8.4 Factor/provider evidence

~~~sh
julia --project=. validation/qdldl_sparse_provider.jl
./scripts/provider_smoke.sh
~~~

The QDLDL validation must be extended from factor/solve coverage to the full
prepare/factor/solve/refine/invalidate lifecycle.

### 8.5 CLI and bridge evidence

Create bin/test/runtests.jl and run:

~~~sh
julia --project=bin bin/test/runtests.jl
~~~

It must cover parser grammar, equality/block schema validation, structured
errors, and proof that rejected input does not start a solve or write output.

### 8.6 Allocation and performance evidence

~~~sh
julia --project=. benchmark/general/performance/hsd_allocation.jl --check
~~~

Performance campaigns must emit machine-readable status/certificate and phase
metrics beside timing data. A fast invalid result is a failed sample.

### 8.7 Downstream compatibility

Before Phase 2 deletion, add a small CSDR smoke that exercises:

- ingest -> SDPProblem;
- prepare -> solve!;
- direct solve!;
- solve_socp where still supported;
- fixed-trace helper loading where still supported.

Run it in the CSDR environment against the local SDPX checkout. Record the exact
command in the Phase 2 baseline artifact rather than depending on a developer's
global environment.

## 9. Review and commit sequence

Recommended implementation sequence:

| Commit group | Content | Required review evidence |
| --- | --- | --- |
| T0 | Six failing regressions and captured Power root fixture | Fail for reviewed reasons on baseline SHA |
| C1 | Compatible singular scalar closure and typed route result | Direct oracle, five equations, public Model/MOI |
| C2 | Representability-aware Power root termination | High-precision oracle, unchanged 1e-8 E2E certificate |
| C3 | QDLDL provider refinement seam | Residual contraction and counter checks |
| C4 | Requirements hierarchy and frozen-shape policy | Complete cache lifecycle |
| C5 | Shared CLI option validation | Equals/separated grammar parity |
| C6 | Shared checked COO ingestion | All mismatch and no-solve tests |
| G1 | Complete Phase 1 gate and frozen correctness SHA | Full command matrix |
| R1…Rn | One structural ownership/deletion batch per commit | Net-negative diff plus full correctness/downstream gate |
| G2 | Frozen simplified SHA and new baseline | Source map, API map, benchmark protocol |
| P1…Pn | One measured performance hypothesis per commit | Paired profile, full correctness, benchmark diff |
| G3 | Candidate release SHA | Local/cluster artifacts on identical SHA |

Do not combine a correctness fix with deletion or a performance refactor. That
would make regression attribution and rollback unreliable.

## 10. Risks and required mitigations

| Risk | Failure mode | Mitigation |
| --- | --- | --- |
| Scalar gauge is mathematically compatible but poor for line search | Valid Newton family member causes unstable progress | Compare deterministic gauge with full-Jacobian oracle; require all five equations and public certificate; profile alternatives only after correctness |
| Near-zero classifier accepts noise | False direction or certificate | Independent D/N work bounds, incompatible fixtures, unregularized residual gate, fail as insufficient precision when unresolved |
| Power fix accepts an uncertified representable point | False optimality near the cone boundary | Certified bracket, reconstruction/Hessian/shadow/secant gates, five equations, unchanged original certificate |
| Power bug is hidden by looser tolerances | Apparent green E2E with weaker answer | Prohibit global tolerance changes; assert resolved 1e-8 settings in regression |
| Factor hierarchy change alters dispatch elsewhere | Provider regressions or ambiguity | Common lifecycle harness across every installed cache/provider |
| QDLDL assumptions do not match SDPX KKT | Incorrect inertia/factor result | Keep optional until diagonal, regularization, inertia, and original-residual proof |
| CLI/bridge consumers relied on silent truncation | New user-visible input rejection | Treat as documented schema correction; emit specific actionable errors |
| Dead-code deletion breaks qualified CSDR usage | Downstream solve failure | Liveness roots include CSDR; migrate first or keep a thin adapter; run smoke after every batch |
| Removing reference code removes evidence | Production and oracle share one bug | Move reference code to validation and keep it independent |
| Settings consolidation changes defaults | Silent route/tolerance/provider change | Snapshot ResolvedSettings and diagnostics through every frontend |
| Prepared cache becomes stale | Wrong model solved or falsely certified | Structural fingerprint, explicit epochs, invalidation tests, canonical parity |
| Benchmark noise or JIT dominates result | False speed claim | Separate cold/hot, paired samples, median plus dispersion, fixed environment |
| Thread oversubscription or nondeterminism | Slower or nonreproducible solve | One active parallel layer, deterministic reductions, serial reference |
| BigFloat buffer growth/aliasing | RSS leak or corrupted RHS | Provider ownership, alias tests, repeated-run RSS gate |

## 11. Deferred until the three phases are complete

The following must not distract from C1–C6, ownership reduction, and measured
general solver performance:

- new cone families or new public solver engines;
- public API expansion unrelated to compatibility preservation;
- promotion of QDLDL or another provider without its proof matrix;
- global equilibration/default-route changes;
- equality-aware fixed-trace Q3 production integration;
- large cluster campaigns before local correctness and resource gates;
- publication of 1.0 or performance claims from different source SHAs.

Fixed-trace Q3 remains valuable for CSDR, but its production integration is a
separate numerical milestone: retain ZeroCone equality authority, solve the
same five-equation system, recover all local variables, and pass the
original-coordinate certificate before measuring speed.

## 12. Final definition of done

The plan is complete only when one clean source SHA provides all of:

1. C1–C6 regressions and the full correctness matrix are green without weaker
   tolerances or certificates.
2. One product-cone HSD production engine, one settings/planning chain, one KKT
   cache protocol, one bordered core, and one result/certificate path remain.
3. Public Model, MOI, CLI, bridge, SDPProblem/ConicProblem compatibility, and
   CSDR smoke all reach that engine.
4. Reference mathematics remains available outside production loading.
5. Source/include reduction is documented as an outcome of proven liveness,
   not an arbitrary quota.
6. Performance changes are tied to measured phase bottlenecks and repeatable
   paired results.
7. Every retained optimization passes the same correctness, precision,
   fallback, allocation, and RSS gates.
8. Local and cluster artifacts, documentation, dependency metadata, and the
   release candidate all name the same SHA.

Until then, no section should be marked complete merely because code exists or
a focused internal experiment passes.
