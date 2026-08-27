# GPT Pro kernel restructure review — 2026-08-27
> [!NOTE]
> SUPERSEDED as planning authority by docs/PLAN.md; retained as the original Pro plan with merge-timeline notes.


Source: ChatGPT conversation `6a90013f-8410-83ec-9d2f-3f7125dd798d` (GPT-5.6 Sol Pro,
pro_extended, worked 76m 22s). Oracle job `6f5c9ceb-81d8-4a96-9db1-6895de056a65`
(worker heartbeat died before capture; content recovered from the browser conversation).
Archive: `context-6f5c9ceb-81d8-4a96-9db1-6895de056a65.tar.zst` (wave/v4-a0 kernel
snapshot ≈59k lines; predates part of the A1 work).

Status: **AUTHORITATIVE restructure plan** for the HSD-only kernel wave.
Verdict: **NEEDS_RESTRUCTURE — but the HSD-only direction is correct.**

> The target should be **one product-cone HSD state machine with several algebraically
> equivalent KKT realizations**, not one HSD implementation permanently tied to the
> current dense bordered normal-equation LU.

Three blocking defects:

1. The native HSD path has no robust linear-system recovery ladder (normal equations +
   nonsymmetric homogeneous border + one-sided row scaling + one unregularized LU,
   failures terminal).
2. The initial-point and reduction pipeline is inappropriate for equality-heavy mixed
   models (unscaled identity start + mandatory dense equality null-space reduction +
   second dense row-space reduction).
3. Canonical cone transformations are not first-class objects (public routing advertises
   Nonpositive/RSOC; runtime supports only normalized families; no explicit
   primal/dual/reconstruction contract).

Sequence: freeze A1 → typed transforms → robust expanded KKT route → port cold start →
migrate HSD engine → remove old engines.

---

# A. Target architecture

## A1. Core design principle

One internal normalized program: min cᵀx+c₀ s.t. Ax+s=b, s∈K with
K = K₊ × K_SOC × K_PSD × K_exp × K_pow; internal x is free. Model-level variable cones
are lowered through exact reversible transformations. These are NOT separate numerical
cones:

- `Nonpositive`: exact sign map to Nonnegative.
- `RSOC`: exact isometry to SOC.
- `ZeroCone`: affine equality subsystem, retained in expanded KKT or eliminated by a
  reversible equality transform.
- `Reals`: free coordinates in x, not entries of the product-cone scaling operator.

## A2. File/module layout

```
src/
├── SDPX.jl
├── public/{model,settings,optimize,result,diagnostics,moi}.jl
├── program/{types,compile,canonicalize,transforms,presolve,equilibrate,equalities,route_plan,reconstruction}.jl
├── cones/{interface,product,layout}.jl
│   ├── symmetric/{nonnegative,soc,psd_triangle,eigen,nt_scaling,boundary}.jl
│   └── nonsymmetric/{exponential,power,conjugate,scaling,corrector,initialization,boundary}.jl
├── hsd/{embedding,state,residuals,initialize,rhs,direction,neighborhood,linesearch,recovery,termination,solve}.jl
├── kkt/{system,assembly,expanded_quasidefinite,reduced_schur,regularization,refinement,route,symbolic}.jl
│   └── specializations/fixed_trace_q3.jl
├── la/{api,capabilities,factorization_session,dense,sparse,generic_precision,mixed_precision,extended_precision_blas,threading}.jl
└── certificates/{canonical,rays,reconstruction,original}.jl
```

Key distinction: **one HSD algorithm, several KKT routes.** `expanded_quasidefinite.jl`,
`reduced_schur.jl`, `fixed_trace_q3.jl` all solve the same `NewtonSystem` — eliminators
and assemblers, not separate solvers.

## A3. Typed program and transformation stack

```julia
struct ConicProgram{T,Ti}
    A::SparseMatrixCSC{T,Ti}; b::Vector{T}; c::Vector{T}
    objective_constant::T
    cones::ProductConeLayout
    transforms::ReconstructionStack{T}
    equilibration::EquilibrationMap{T}; presolve::PresolveMap{T}
    fingerprint::ProgramFingerprint
end
```

Every coordinate-changing compilation op pushes a typed transform onto
`ReconstructionStack`. Each transform implements: `forward_primal!`, `backward_primal!`,
`forward_dual!`, `backward_dual!`, `backward_primal_ray!`, `backward_dual_ray!`,
`objective_shift`, `verify_pairing_invariant`, `verify_stationarity_invariant`.

- **Nonpositive**: T=−I; ŝ=−s, Â=−A, b̂=−b; dual uses inverse adjoint ŷ=−y; preserves
  pairing ⟨s,y⟩=⟨ŝ,ŷ⟩. One object owns row signs, dual reconstruction, ray
  reconstruction, original-coordinate checking.
- **RSOC**: clean internal map is the orthogonal isometry
  (u,v,w)↦((u+v)/√2,(u−v)/√2,w); the current public map (u+v,u−v,√2w) is a uniform √2
  scaling and cone-correct but its inverse adjoint/pairing scale must be carried
  explicitly (`src/public/optimize.jl:495–509`). Untyped mixture is not valid.
- **PSD triangle**: make the metric explicit (svec with √2 weights, or packed-lower +
  `TriangleMetric` used by every dot/adjoint/KKT/reconstruction). Never let primal
  packing, dual packing, Jordan products, and Euclidean adjoints independently decide
  off-diagonal factors.

## A4. Ownership rules

- `ConicProgram`: owns normalized immutable A,b,c, cone layout, presolve/equilibration
  maps, reconstruction chain, source-precision metadata. Never iterates/KKT factors/
  mutable scaling.
- `HSDState`: sole owner of x,s,y,τ,κ; directions; residuals; trial iterate; centrality
  diagnostics; accepted-point checkpoint; counters. No duplicate legacy
  Workspace/HotStepState/LP/SDP/product-HSD states of the same quantities.
- `ProductConeRuntime`: immutable block descriptors; per-block NT/nonsymmetric scaling;
  boundary/eigen/Cholesky scratch; cone-quality diagnostics. Never A, KKT, status, or
  certificate policy.
- `KKTWorkspace`: one symbolic pattern, one numeric matrix, one factorization session,
  one multi-RHS panel, regularization vectors, refinement buffers, route diagnostics.
  Never owns the HSD iterate or decides Optimal.
- `FactorizationSession`: symbolic ordering, numeric factor, factor epoch, expected and
  observed inertia, pivot/regularization diagnostics, solve/refinement counters.
  Predictor, corrector, centrality corrections, and refinement share the factor.
- **Certificate layer**: receives a read-only candidate, reconstructs through every
  transform in reverse order, evaluates the original model. ONLY this layer may promote
  `Optimal` / `PrimalInfeasible` / `DualInfeasible`.

## A5. KKT architecture

One semantic `NewtonSystem` defining the five HSD Newton equations once (primal affine;
dual affine; homogeneous gap; cone complementarity/scaling; τκ complementarity). Cone
kernels contribute a self-adjoint local linearization + corrector RHS; KKT routes may
eliminate variables but may not rederive signs.

- **Robust route: expanded symmetric-quasidefinite KKT** (default for mixed,
  equality-heavy, ill-conditioned): saddle-point form
  K = [[Rx, Aᵀ, c], [A, −(H+Ry), −b], [cᵀ, −bᵀ, −ρ]] with signed static regularization
  blocks; pivoted LDLᵀ with expected-inertia check. Avoids cond-number squaring through
  AᵀGA; naturally accommodates free variables and equality rows.
- **Fast route: reduced Schur** only when setup predicts advantage (reliable rank,
  acceptable cone-scaling condition, bounded fill, memory, benign geometry). Solution
  must be checked against the unregularized five-equation Newton residual; on failure
  switch to expanded within the same HSD state.
- **Structural route: fixed-trace Q3** as a `KKTContribution`/local eliminator — never
  its own iteration loop, termination, cold start, or result type.
- **Regularization/refinement ladder**: attempt planned factorization → on failure retry
  with norm-scaled signed static regularization → dynamic pivot regularization preserving
  target inertia → solve all RHS with the same factor → refine against the UNREGULARIZED
  Newton equations → on stagnation increase regularization or switch route → escalate
  precision → only then a typed numerical failure. A regularized direction is acceptable
  only if its unregularized residual passes the strict backward-error contract.

## A6. Cold start

`product_hsd_cold_start!` (product_cone_hsd.jl:252–259) sets x=0, cone identities,
τ=κ=1 without using A,b,c — must become the emergency fallback, not the default. Target
pipeline: cone-preserving equilibration → affine least-residual KKT solves (primal then
dual residual; reuse symbolic factor) → minimal strict-interior shift per cone block →
legacy identity-mass floor → primal/dual cross-centering → PSD continuation repair in
Cholesky form → nonsymmetric cones at validated central unit points → τ=κ=1 → record
initial residual/complementarity/min cone margin/KKT quality. (Clarabel does the same.)

## A7. Equality handling

Do NOT default to the current dense equality null-space reduction
(`hsd/equality_reduction.jl`) followed by the second dense row-space reduction
(`_hsd_rowspace_reduction`, hsd.jl:201–325). Policy: small dense → pivoted RRQR; large
sparse with major reduction → sparse QR subject to fill estimate; mixed
free/equality/PSD → normally retain equalities in the expanded KKT; at most ONE
numerical rank transform; rank decisions after equilibration, structural exact
eliminations before.

## A8. Current-file disposition (summary)

- Survive/reorganize: `modeling/*`, `ir/types|storage|layout`, `public/*`,
  `moi_wrapper.jl` → `public/`, `program/`.
- Survive with typed transforms: `ir/canonical.jl`, `ir/reconstruction.jl`, lowerers.
- Survive (A1 baseline): `cones/symmetric/*`.
- Merge: `cones/runtime/{types,product,symmetric_api}` → `cones/product.jl`,
  `cones/layout.jl`; `cones/nonsymmetric/*` survive and merge;
  `full_newton_reference.jl` → test-only oracle.
- Split: `hsd/hsd.jl` (state/residual algebra kept, mandatory row-space reduction
  removed); `hsd/product_cone_hsd.jl` → rhs/direction/linesearch/recovery;
  `product_cone_solve.jl` → solve/termination.
- Retain as optional reversible transform: `hsd/equality_reduction.jl` →
  `program/equalities.jl`.
- Survive: `certificates/*`, `_public_original_certificate`;
  `cold_start.jl` almost entirely → `hsd/initialize.jl`; `soc_presolve.jl` generalized
  → `program/presolve.jl`; `step_hot.jl` hot-path contract kept;
  `kkt_route.jl`+`factor_cache/*` → `kkt/route.jl`+`la/factorization_session.jl`;
  `la_backend.jl`+`sparse_la.jl` redesigned around indefinite KKT;
  `kernels/mixed_precision_kkt.jl` policy port; `kernels/threaded.jl` scheduler port;
  `kernels/extended_precision_blas/*` survive.
- Extract then delete: `soc_native.jl` (Q3, LDL retry, residual checks);
  `kkt.jl`/`schur.jl` tested kernels; `cone_algebra.jl`/`soc_lorentz_kernels.jl`
  (formulas kept only as tests).
- Delete after parity: `lp_solver.jl`, `lp_sparse.jl`, `solver/interior_point.jl`,
  `step.jl`, `hsd/nonnegative_hsd.jl`;
  merge `hsd/nonsymmetric_coupled.jl`/`nonsymmetric_schur3.jl` into generic cone KKT
  contribution.
- Quarantine: `chordal.jl` (later as reversible program transform).
- Collapse: `pipeline/*`, planners → `program/route_plan.jl` (program/KKT/LA route
  planning, not engine selection).

---

# B. Legacy keep-list (port these; the old engines are not keepers)

| Legacy strategy | Why | New location |
|---|---|---|
| `_fixed_trace_q3_active_variables`/`_fixed_trace_q3_reduction` (soc_native.jl:28–54,243–292) | exact Q3 fixed-head pattern detection, invertibility/ownership gates | `kkt/specializations/fixed_trace_q3.jl` |
| `:fixed_trace_q3_local_elimination` route | local 2×2 tail elimination; valuable for many small identical cones | `KKTContribution`, not a solver |
| Inertia-aware augmented LDL retry (`_native_soc_assemble_factor!`, soc_native.jl:1136+) | unregularized→type-aware regularization→inertia check; much stronger than bordered LU | `kkt/regularization.jl` |
| Unregularized direction residual checks (soc_native.jl) | regularization must not silently change Newton equations | `kkt/refinement.jl` |
| Guarded singleton elimination (soc_presolve.jl:330–510) | pivot quality, relation width, finiteness, deterministic tie-breaks, fill/work estimates | `program/presolve.jl` |
| `NativeSOCPresolveMap` reconstruction | restores primal + eliminated equality duals | generic `PresolveMap` |
| `_continuation_psd_repair!` (cold_start.jl:328–524) | minimal strict PSD identity shift via Cholesky bracketing | `cones/symmetric/boundary.jl` or `hsd/initialize.jl` |
| `_cold_start_identity_mass_shifts` (cold_start.jl:553–605) | keeps affine start away from cone vertex | `hsd/initialize.jl` |
| `_cold_start_centering_shifts` (cold_start.jl:621–670) | cross-centers primal/dual identity masses | `hsd/initialize.jl` |
| KKT-derived LP/SDP starts (`_lp_phase2_cold_start!`, `_kkt_cold_start_initialization`) | data-dependent starts; directly addresses the origin stall | `hsd/initialize.jl` |
| `step_hot.jl` allocation contract | preallocated buffers, isbits status, one factor/epoch, no factorization in line search | whole HSD hot path |
| `kkt_route.jl` one-factor/multi-RHS contract | predictor/corrector/refinement share one factorization | `la/factorization_session.jl` |
| Sparse symbolic/numeric separation (sparse_la.jl) | ordering/allocation reuse | `la/sparse.jl` |
| Precision prediction/escalation (mixed_precision_kkt.jl) | condition estimates + gates instead of unconditional BigFloat | `la/mixed_precision.jl` |
| Cost-weighted deterministic LPT scheduling (threaded.jl) | block ownership, no atomics, BLAS/Julia coordination | `la/threading.jl` |
| Nonsymmetric scaling checkpoints/rejected-trial restoration | Exp/Power line-search failures must not corrupt scaling state | `hsd/linesearch.jl` |
| Stable primal recovery `_product_hsd_recover!` (product_cone_hsd.jl:1253–1303) | ds=−A dx−rP+b dτ directly; avoids ΘG round-trip | `hsd/recovery.jl` |
| `_public_original_certificate` (public/optimize.jl:578–674) | original-coordinate verification | `certificates/original.jl` |

---

# C. Root-cause review

## C1. Mixed Reals + PSD + ZeroCone models

`_product_hsd_apply_symmetric_G!` correctly applies cone scaling only to runtime cone
rows (Reals∈x; ZeroCone∈equality). Missing is a setup invariant PROVING coverage
(every cone row covered once; no Zero/Reals/Nonpositive/RSOC reaches runtime; free
columns survive reductions; equality reconstruction maps valid).

Confirmed problematic sequence (native public policy disables equilibration, presolve,
sparse, alt providers — native_hsd_public.jl:112–144):
1. Dense equality null-space reduction (equality_reduction.jl, pivoted QR, dense null
   basis, active_A*null_basis).
2. Second dense row-space reduction (`_hsd_rowspace_reduction`, hsd.jl:201–325).
3. Data-independent identity start (product_hsd_cold_start!).
4. Normal-equation condensation H=Ar'GAr (product_cone_hsd.jl:616–674) — squares
   conditioning.
5. Nonsymmetric bordered system (752–840) where q≠r generally, only row scaling.
6. One unregularized generic LU (893–931) — no signed regularization, no inertia, no
   iterative refinement, no route fallback.

Required changes: (1) cone-preserving equilibration before rank decisions; (2) structural
presolve separate from numerical rank reduction; (3) at most one numerical equality/rank
transform; (4) default mixed free/equality/PSD to expanded quasidefinite KKT; (5)
KKT-derived start replacing identity start; (6) signed regularization + inertia +
refinement; (7) reduced-Schur→expanded fallback before precision escalation; (8) check a
terminal original-coordinate candidate before returning any direction failure; (9) record
the real reason (rank ambiguity / factor failure / wrong inertia / refinement stagnation /
cone scaling unresolved / line-search neighborhood / insufficient precision).

## C2. Nonpositive failures

Public gate lists `:nonpositive` and `:rsoc` as supported (native_hsd_public.jl:156–175)
but `ProductConeRuntime` accepts only nonnegative/soc/psd/exp/power and fails closed
otherwise (cones/runtime/product.jl:210–242,422–428). Correct ONLY IF canonicalization
performs exact transformations first — which is not visible in the runtime type or
ownership contract. Conclusion: **the support claim is broader than the auditable
normalized-program contract.** Implement `NonpositiveToNonnegative{T}` and
`RotatedSOCToSOC{T}` transforms owning row/col sign, rhs sign, dual inverse-adjoint, ray
reconstruction; setup assertions that runtime layout contains only canonical families;
property tests (pairing ⟨s,y⟩=⟨Ts,T⁻ᵀy⟩; stationarity; ray reconstruction; objective
equality); fail at compilation if any invariant is unprovable. The redundant capped LP
additionally needs duplicate-row/simple-bound presolve + robust KKT route.

## C3. Tiny rank-one SOC/PSD optima

At rank-1/extreme-ray optima: cone eigenvalue → 0, NT scaling conditioning rises, normal
equations worsen, predictor directions go tangential. This is normal. The SOC gates
(`_soc_q_condition_reliable` soc.jl:165–196, `_soc_spectral_gap_reliable`:198–212,
`_product_hsd_cone_newton_stats` product_cone_hsd.jl:1371–1381) are defensible fail-closed
arithmetic gates, but the solver treats their failure as TERMINAL instead of as a request
for a more centered direction / alternate route / higher precision / termination from an
already valid candidate.

Line search (`_product_hsd_line_search!` 2704–2781) requires the NEXT NT scaling to be
constructible before accepting the trial — wrong ordering near boundary optima. Also
`product_hsd_solve!` (product_cone_solve.jl:521–525) returns immediately on
`HSDStepDirectionFailed` WITHOUT first running `_product_hsd_terminal_verified_result!` —
a failed next step can conceal a valid current iterate.

Required behavior on scaling/factor/direction failure: (1) verify current accepted iterate
in original coordinates; (2) verify a rejected finite trial before discarding; (3) then σ
centrality restoration, bounded Gondzio-style corrections on the same factor, increased
signed regularization, alternate route, precision escalation; (4) only then return
`StalledUnverified`/`InsufficientPrecision`/`NumericalFailure`. Keep
σ=(μ_aff/μ)³ predictor-corrector but add per-cone neighborhood tracking (complementarity
spread, min scaled eigenvalue, Jordan ratio, Exp/Power Fenchel residual, τκ/μ) with a σ
floor + restoration RHS when out of neighborhood. Common step length is conservative and
NOT the primary defect; separate primal/dual steps later.

## C4. Mature-solver practice

- Clarabel: equilibration → data-dependent starts from symmetric KKT solves → one
  factorization for affine/centering RHS → statically+dynamically regularized symmetric
  quasidefinite system; iterative refinement and presolve on by default.
- SDPT3: initial points performance-sensitive, scaled to data; detects nearly dependent
  constraints; Schur system ill-conditioning as complementarity→0; recommends iterative
  refinement.
- SDPA: Mehrotra predictor-corrector; sparse/dense-aware assembly formulas; primal/dual
  PSD boundary steps with damping; parallel + multiple-precision variants. Lesson:
  preserve block-aware Schur assembly, boundary-step logic, precision specialization
  behind the HSD equations — not the legacy state machine.
- SeDuMi: near complementary solutions → ill-conditioned NT scaling and cancellation;
  maintain Cholesky/v-space factor information rather than repeatedly reconstructing
  near-singular matrices from unstable eigensystems. A1's congruence scaling + Cholesky
  frame is consistent; factor-form iterates a reasonable later enhancement.
- MOSEK: HSD conic optimizer; separate feasibility/gap tolerances; central-path controls;
  fraction-to-boundary; step-stall detection. Adopt separation of diagnostics; do NOT
  adopt relaxed "near optimal" multipliers (original-coordinate certificates stay
  authoritative).
- COSMO: operator-splitting; take only modular linear-system backends, chordal
  preprocessing, block threading, BLAS/Julia oversubscription avoidance.

---

# D. Speed review

D1 Redundant implementations (more damaging than line count): nonnegative HSD vs product
HSD vs legacy LP vs legacy SDP vs soc_native; KKT assembly in schur.jl/kkt.jl/soc_native/
product_cone_hsd/nonsymmetric_schur3/nonsymmetric_coupled; cone algebra in four places;
multiple cold starts; multiple factor-cache/workspace/state hierarchies.

D2 Memory: `SymmetricBorderedWorkspace` owns matrix + scaled copy + full factor-error
matrix; HSDState owns sparse A + dense A + reduced Ar + bases + dense H + residuals.
Target: one canonical sparse A; optional route-specific panel; one numeric KKT; one
factor; one multi-RHS panel; small residual/refinement workspace. factor_error and
independent matrix/factor_matrix copies only for diagnostic builds.

D3 Factor-certificate overhead: `_product_bordered_factor_certificate!` (861–890) is
O(n³) LU-product reconstruction every epoch — keep as dev oracle/periodic canary/test
invariant only; production acceptance = pivot diagnostics + inertia + backward error +
refinement + unregularized five-equation residual.

D4 Schur assembly: per-column scatter+G-apply+dot with poor locality. Target panelized
GA_J, syrk!/Gram updates, block-specific sparse/dense SDP formulas, local contributions
from small SOC/Exp/Power blocks, unique tile ownership, SDPA-style sparse/dense kernel
selection per PSD block.

D5 Factor reuse: product state constructs a base Schur driver AND a separate bordered
LPLU driver; the base driver is unused by the product step — remove duplicate ownership.

D6 Sparse/dense routing: setup-estimated (dimension, density, fill, frontal memory,
contribution cost, #RHS, arithmetic, memory). Policy: small/dense → dense pivoted LDL;
sparse bounded fill → sparse LDL; reduced Schur only when materially smaller + favorable;
Q3 when exact predicate holds; no silent dense fallback over memory budget; thresholds as
benchmark-calibrated configuration, not embedded in cone code.

D7 Threading: preserve threaded.jl policy (deterministic LPT, block ownership, no
atomics/locks, fixed reduction order, Julia×BLAS coordination, serial MPFR unless private
scratch, MultiFloat parallel OK). Move policy to setup; hot loop receives a concrete
schedule. Sparse factorization backend owns its thread count; no oversubscription.

---

# E. Robustness review

E1 Reclassify gates: (a) FATAL setup invariants (dimensions, overlapping/uncovered cone
blocks, unsupported family, nonfinite data, bad PSD shape, unprovable transform,
inconsistent exact zero row) — reject immediately; (b) RECOVERABLE numerical-quality
failures (spectral gap, NT condition, pivot threshold, wrong inertia, refinement
stagnation, line-search scaling failure, tiny step, reduced-Schur residual) — invoke the
recovery ladder; (c) AUTHORITATIVE result gate — original-coordinate verification only.

E2 Backtracking: keep A1 policy (0.9 damping, 64 trials, checkpoint restore); add
cone-interior + residual-homotopy test, neighborhood test, current/trial certificate
check on scaling failure, centering restoration with the existing factor, route/precision
switch, record first+final rejection reasons. Do not halve steps for directions whose
unregularized residual is already unacceptable.

E3 Presolve: conservative reversible passes (zero rows/cols, singleton equality
substitution, duplicate/proportional rows, fixed variables, bound folding, redundant
bound detection, unconstrained objective directions, structurally dependent equalities,
1×1 PSD→Nonnegative, optional exact 2×2 PSD→RSOC, fixed-trace Q3 detection). Every pass
produces reconstruction + ray maps + objective shift + proof category (exact structural /
numerically guarded / rejected). Facial reduction is NOT a prerequisite.

E4 Equilibration: native public path currently does NOT equilibrate. Add iterative
Ruiz-style scaling subject to cone invariance: free columns independent positive;
Nonnegative rows independent positive; SOC/RSOC one scalar per block; PSD one scalar per
block (later optional diagonal congruence with full primal/dual map); Exp/Power one
scalar per block; equality rows independent positive. Store all scalings; reconstruct
primal+dual before certification. Rank tests/pivot thresholds/regularization magnitudes
operate on the equilibrated system.

E5 Precision escalation: Float64 → Float64x2 → Float64x4 → BigFloat(p). Triggers:
repeated wrong inertia; KKT backward error above target after refinement; refinement
ratio stalls; cone-scaling condition unresolved; step repeatedly below stall threshold;
route-switch failure; expected digits below certificate requirement. Rebuild from
highest-precision SOURCE coefficients (not from rounded Float64); preserve symbolic
structure/route; convert+re-center only if strict interiority survives else cold-restart
high-precision; scoped BigFloat precision owner + preallocated scratch; never loosen
tolerances; distinguish `InsufficientPrecision` from invalid/infeasible model.

E6 Separate termination tolerances: `_native_hsd_tol` (native_hsd_public.jl:179–186)
collapses primal/dual/gap to their minimum — carry separately: primal feasibility, dual
feasibility, relative gap, absolute gap, ray residual, κ/τ, cone margin, Newton residual.

E7 Status taxonomy: replace generic `direction_breakdown` with Optimal / PrimalInfeasible
/ DualInfeasible / MaxIterations / TimeLimit / StalledUnverified / InsufficientPrecision
/ NumericalFailure / RankAmbiguous / InvalidModel, plus last internal reason
(e.g. ReducedSchurWrongInertia, SOCScalingUnresolved).

---

# F. Phased plan

- **Phase 0 — Freeze numerical baseline**: confirm A1 (congruence NT, Cholesky boundary,
  Jacobi threshold, 0.9/64) is baseline; add `test/kernel_failure_regressions.jl` with
  synthetic tests: tiny rank-1 SOC boundary, 2×2 rank-1 PSD boundary, mixed
  free/equality/PSD, Nonpositive, redundant capped LP, RSOC transform, Exp/Power smoke,
  primal+dual infeasibility rays. Known failures visible in CI (expected-failure report).
  Parallel: highly (PSD/SOC/transform/failure-status tests independent). Rollback: one
  commit.
- **Phase 1 — Typed canonical transformations** (`program/transforms.jl`; Nonpositive,
  RSOC, ZeroCone extraction, free-variable substitution, PSD packing metric;
  ProductConeRuntime constructible only from NormalizedConeLayout; coverage assertions;
  all transforms reconstruct optima + both ray types). Acceptance: property tests
  (pairing/stationarity/rays/objective); runtime sees only canonical blocks;
  unsupported transform fails at compilation. ~1 eng-week; Nonpositive/RSOC/PSD metric
  parallelizable after interface freeze.
- **Phase 2 — Unified KKT + factorization layer**: `NewtonSystem` semantic definition;
  expanded quasidefinite route; pivoted LDL (LA capability layer currently reports it
  unavailable — la_backend.jl:116–141); generic-precision fallback; expected inertia;
  signed static + dynamic pivot regularization; multi-RHS; unregularized iterative
  refinement; one factor per epoch; sparse symbolic/numeric separation. Largest
  foundation package (~2–3 eng-weeks); 4 parallel workstreams (dense LDL, sparse,
  assembly, refinement) after FactorizationSession agreement.
- **Phase 3 — Equilibration, presolve, equality policy, cold start**: structural
  presolve; cone-preserving equilibration; cost-based equality strategy
  (retain/sparse-QR/dense-RRQR); disable second row-space reduction by default;
  KKT-derived primal+dual starts; port PSD continuation repair + identity-mass floor +
  cross-centering; Exp/Power validated central points. Cures the origin stall before
  rewiring the loop (~1–2 eng-weeks; equilibration/presolve/repair parallel).
- **Phase 4 — Migrate product HSD engine**: one HSD state; KKTSession replaces
  bordered-LU; expanded KKT robust default; reduced Schur optional; one
  `ConeLinearization` interface (Mehrotra symmetric, 3rd-order Exp/Power);
  neighborhood restoration; terminal current/trial certificate checks; route fallback;
  precision escalation callback; typed failure reasons; preserve A1 changes. Mostly
  serial (~2 eng-weeks).
- **Phase 5 — Structural specialization + performance**: port fixq3 as
  `kkt/specializations/fixed_trace_q3.jl`; panelized/block assembly; sparse/dense kernel
  selection; symbolic reuse; remove every-epoch LU reconstruction; LPT scheduling;
  memory-budget route planning; remove duplicate dense copies. Acceptance: warm
  fixed-width step zero heap allocation; exactly one factorization/epoch; no silent
  densification; deterministic across threads; ≤10% median kernel regression threshold.
  Strongly parallel (~2–3 eng-weeks).
- **Phase 6 — Solver-wide precision escalation**: type/precision ladder; estimate bits
  from KKT condition/refinement/cone condition/certificate tolerance; rebuild from source
  coefficients; re-center or cold-restart; record transitions; distinguish
  InsufficientPrecision. (~1–2 eng-weeks; estimator + backend parallel.)
- **Phase 7 — Delete legacy engines**: lp_solver, lp_sparse, solver/interior_point,
  step, nonnegative_hsd, soc_native, old production portions of kkt/schur/cone_algebra/
  soc_lorentz_kernels/nonsymmetric_coupled/nonsymmetric_schur3. One HSD solve loop + one
  Newton-system implementation reachable; legacy preserved in history/tag only;
  pre-deletion signed tag; deletion is one isolated revertible merge. (~1 eng-week.)
- **Phase 8 — Optional later**: chordal decomposition, facial reduction, warm starts,
  matrix-free route, distributed PSD assembly, v-space factor-form iterates. Each enters
  as reversible transform or alternative route — never another solver state machine.

**First milestone**: freeze A1 + synthetic failure suite, then build `NewtonSystem` +
expanded quasidefinite KKT route + KKT-derived cold start. Do NOT start by deleting
legacy files or tuning the bordered LU. Scope: 10–14 engineer-weeks ≈ 6–8 calendar weeks
with 3 parallel streams; transforms/LA/cone-tests/presolve/Q3/sparse/threading
parallelize well; one owner for the unified HSD state machine.