# R0-P4: opt-in half-Power factor-pair runtime boundary

**Design review only; no code changes or tests.**  
Inspected checkout: `/tmp/sdpx-scientific-core-20260907`, HEAD `1269938`, SDPX 0.6.1.

## Inherited decisions

- Preserve the existing public defaults, precision, root tolerances, rank policy, certificate tolerances and line-search inequalities.
- The factor representation must survive the **entire Newton epoch**. A successful factor-pair certificate must never authorize an old dense-Theta consumer.
- The reviewed construction is explicitly **dual-Hessian one-secant**. It does not qualify strict double-secant scaling.
- Root, stored-shadow geometry, factor, metric, Newton direction, accepted step and public solution certificates remain separate authorities.
- Parent implementation owns critical numerical changes.

**Scope confirmed with the supervisor:** first admission is Float64 products containing only orthant and exactly half-Power blocks, with at least one Power block. Reuse the existing canonicalization/equality-reduction pipeline. Reject mixed Exp/SOC/PSD products and unsupported KKT routes before iteration. A mixed-cone factor/dense hybrid is out of scope.

## Diagnosis

### The necessary boundary is larger than `apply_Theta!`

The current production path assumes a dense nonsymmetric metric in several independent places:

- `runtime/nonsymmetric_api.jl:151–207` validates dense `g`, `theta` and their factorization before accepting live state.
- Its checkpoint and restore paths copy dense metric storage (`223–371`).
- `runtime/product.jl:175–207` provides the dense-Theta-based nonsymmetric G application.
- `hsd/predictor_corrector.jl:141–284` constructs expanded/sparse linearizations.
- `_product_hsd_symmetric_core_linearization!`, at `819–875`, populates stored block operators; nonsymmetric blocks use `nonsymmetric_scaling_contribution3!`.
- `_product_hsd_core_scatter!`, at `654–700`, applies Theta and performs G/Theta diagnostic recovery.
- `hsd/product_cone_hsd.jl:658–681,824–915,978+` contains dense-metric round-trip/copy machinery.

Changing only the scaling constructor or adding a factor-backed Theta action therefore leaves incompatible consumers reachable.

### The experimental KKT executor is not the production symmetric core

`FactorPreservingAffine._assemble_epoch`, at `factor_preserving_affine.jl:217–235`, constructs a dense, generally **nonsymmetric** `(n+m+2)` core and factors it with `lu`.

It is not the current symmetric augmented LDL/CHOLMOD executor. The `c`, `bhat`, tau and kappa border entries make that distinction substantive.

Initial admission must disclose:

- dense transformed core;
- LU with its actual pivoting;
- one factorization reused for affine and combined solves;
- no CHOLMOD, symmetric-LDL or sparse-core claim.

Dense **KKT** storage is allowed here; dense **Theta** materialization is not.

### The reviewed prototype already supplies most numerical interfaces

- `NP.build`: pair construction and certification (`native_half_pair.jl:159–220`).
- `NP.epoch`: bounded, ownership-isolated, typed-refusal epoch adapter (`221–251`).
- `FA.affine_rhs` / `FA.solve`: affine system and recovery (`236–264`).
- `NC.certify`: independent affine metric/coefficient/five-equation certification (`native_factor_affine_certificate.jl:94–185`).
- `FC.build` / `FC.solve` / `FC.certify`: current-point correction, combined RHS, factor reuse and certification (`factor_combined_epoch.jl:50–185`).
- `NP.trial`: rounded trial point and fresh pair construction (`252–268`).

However, `ExperimentalPowerStep.build_epoch` calls `FA._assemble_epoch` directly (`experimental_power_step.jl:213–224`), bypassing `NP.epoch`’s admission checks and typed numerical refusal. The production adapter should use the latter contract.

## Drift / contradiction check

1. **The alpha claim is overstated.**  
   `EXPERIMENTAL_POWER_STEP.md:8–10` says every accepted alpha lies in `[0.026,0.90]`. Inherited run logs include later accepted alphas below `2cbrt(eps)`. Source permits this inside the arithmetic neighborhood: `linesearch.jl:96–117`.

   Preserve the actual policy. Require a decisive above-floor repair step, not an invented universal minimum-alpha rule.

2. **Prototype terminal certification is not production-target qualification.**  
   The document distinguishes experimental merit target `1e-8` from certificate tolerance `1e-6` (`EXPERIMENTAL_POWER_STEP.md:21–38`). Public qualification must use the actual requested certificate tolerance and reconstruction pipeline. Do not silently use the prototype’s `1e-6`.

3. **`NP.certify` is retained-probe replay, not cold rebuilding.**  
   It calls `root_call(..., r.probe)` (`native_half_pair.jl:116–132`). Warm-seeded certification of the actual stored pair is valid evidence; a fresh cold rebuild is a different construction and remains diagnostic.

4. **The current certificate implementation also needs a representation audit.**  
   `NC._certify_equations` allocates global interval `Theta` and `Wmatrix` arrays (`native_factor_affine_certificate.jl:122–135`). These are verifier enclosures, not the unstable Float64 solver metric, but the literal no-dense-Theta production boundary should eliminate that global matrix construction too. Stream block polynomial actions and work bounds; preserve the same inequalities.

5. **A harness-target reconstruction is not public dispatch integration.**  
   The inherited benchmark arm establishes a useful data link, but R0-P4 must actually execute the compiled canonical problem through its reconstruction maps. It must not recognize a benchmark name or rebuild its reduced equations manually.

6. **Broader roadmap stages remain open.**  
   Precision-ladder success supports precision sensitivity. It does not establish a particular Exp arithmetic operation as the unique cause, nor close R0-E, R0-Q or R1–R6.

## Recommendation

### A. Exact backend boundary

Introduce a **separate internal factor-pair HSD state**, rather than attaching factor receipts to the existing dense-metric runtime.

Proposed types:

- `HalfPowerFactorPairHSDState`
- `HalfPowerFactorEpoch`
- `HalfPowerAcceptedStepReceipt`
- `HalfPowerFactorRefusal`

These are design names, not existing symbols.

The new state owns:

- canonical/reduced problem and reconstruction lineage;
- accepted `x,s,y,tau,kappa,mu`;
- NP owner, pair and warm-token generation;
- factor-defined cone;
- current epoch and LU factor;
- affine and combined direction receipts;
- separate trial scratch;
- phase timings and typed terminal diagnostics.

It must **not** contain a usable legacy `NonsymmetricScalingWorkspace` whose `theta`, `g` or `valid` fields can be consulted accidentally.

#### Dispatch placement

Two admission checks are necessary:

1. **Plan-time capability check:** classify the original cone product and requested settings before expensive numerical setup.
2. **Post-reduction admission:** validate the actual reduced dimensions, layout, coefficient storage, reconstruction maps and memory requirement.

The execution fork belongs in `_public_native_hsd_core`:

- retain its canonical/equality-reduction work;
- branch after the admitted reduced problem is available;
- branch **before** `_product_cone_hsd_state` constructs the prepared symmetric core and `ProductConeRuntime`.

Relevant source:

- `_public_native_hsd_core`: `native_hsd_public.jl:1385+`;
- reduced/equilibrated program: `1475–1482`;
- row-space reduction: `1568–1584`;
- legacy memory/core setup and state construction: `1682–1810`;
- solve and public recovery: `1817–1836+`;
- `_product_cone_hsd_state`: `product_cone_hsd.jl:225–278`.

Do not run legacy runtime initialization first and switch after it fails.

#### Consumer map

| Responsibility | Opted-in factor-pair route | Unselected legacy route |
|---|---|---|
| Product construction | Orthant scales plus NP half-Power receipts | Existing `ProductConeRuntime` |
| Pair update/neighborhood | Fresh `NP.build` / `NP.trial` | Existing `try_update_scaling!` |
| Theta action | `S(St(v))` | Existing dense/block operators |
| G action | `Wt(W(v))` | Existing dense-Theta solve |
| Whitening | `W`, `Wt`, `S`, `St` triangular actions | Existing route machinery |
| Affine RHS/direction | FA contract plus NC certificate | Existing predictor/KKT methods |
| Corrector RHS | FC/HC current-point contract | Existing `corrector_shift!` |
| Combined solve | Same epoch LU plus FC certificate | Existing prepared-core solve |
| Direction recovery | Raw factor-coordinate recovery | Existing scatter |
| Trial rollback | Discard private trial scratch | Existing dense checkpoints |
| Commit | Atomic accepted-state/epoch/generation swap | Existing update |
| Terminal recovery | Shared canonical/original-coordinate certificate adapter | Existing public authority |

The opted-in route must never call:

- dense metric live/checkpoint preflights;
- `force_power_dual_hessian_scaling!` as a rescue;
- legacy expanded/sparse/block-matrix linearization;
- legacy core scatter or dense G/Theta round trips;
- a terminal rescue that rebuilds legacy scaling.

`set_power_dual_hessian_mode!` and `force_power_dual_hessian_scaling!` (`nonsymmetric_api.jl:1059–1088`) still execute the old dense backend. They are not the proposed opt-in selector.

**Which consumers keep dense Theta?** Only the existing, unselected legacy routes. Their defaults and tests remain unchanged. Within one opted-in solve, no cone consumer keeps dense Theta—even the orthant uses scalar factors. Mixed-cone hybrids are explicitly refused.

### B. Epoch adapter contract

#### 1. Admission and coordinate lineage

Initial numerical bounds retain the reviewed NP limits:

- `1 <= m <= 32`;
- `1 <= n <= 16`;
- at least one Power block;
- every Power exponent has the exact Float64 half representation;
- contiguous orthant-prefix/Power-block layout.

These follow `NP.build:159–165`, `NP.Layout:26–32`, and `NP.epoch:221–238`.

If the canonical layout needs reordering, use an explicit reversible row permutation recorded in the reconstruction chain. Never infer block positions from model names.

Reuse existing certified equality/rank reduction. Preserve all basis maps and original objectives. Ambiguous or incompatible rank results retain their existing fail-closed/certificate handling. Do not create a second rank algorithm.

For first admission, reject equilibration and nondefault iteration policies rather than silently approximating unsupported semantics.

#### 2. Pair

Build from the actual stored Float64 `s,y,mu`, never a reference-generated replacement.

The receipt binds:

- owner and generation;
- exact point/factor words;
- layout and settings;
- explicit one-secant policy;
- root, reconstruction, stored-geometry and BFGS reports;
- verified arithmetic context.

The runtime guard must enforce the inherited Julia/platform/math/rounding/FTZ/FMA restrictions before certified EFT operations are used.

#### 3. Epoch

Use the `NP.epoch` contract, including:

- owned copies of `A,b,c,x,s,y`;
- valid sorted CSC structure;
- scalar finiteness and positivity;
- fresh pair certification;
- factor and problem fingerprints;
- typed singular/domain/nonfinite refusal.

Extend the epoch fingerprint to bind **public reconstruction lineage**, backend policy and owner generation. Its current numeric/structural fingerprints are useful but are not a complete public-program identity.

#### 4. Factor-defined transformed system

For each Power block retain

\[
S=\sqrt{\mu}\,L^{-T}R,\qquad
W=R^{-1}L^T/\sqrt{\mu}.
\]

`FA.transform` implements these and their transposes (`factor_preserving_affine.jl:85–92`). The product implementation includes orthant scalar factors (`110–128`).

Form only

\[
\widehat A=WA,\qquad \widehat b=Wb.
\]

The FA core is

\[
K=
\begin{pmatrix}
0&\widehat A^T&c&0\\
\widehat A&-I&-\widehat b&0\\
c^T&\widehat b^T&0&1\\
0&0&\kappa&\tau
\end{pmatrix}.
\]

Unknowns are \((dx,\widehat{dy},d\tau,d\kappa)\), where

\[
dy=W^T\widehat{dy},\qquad
ds=S(\widehat h-\widehat{dy}),\qquad \widehat h=Wh.
\]

This preserves `ds + Theta*dy = h` without constructing Theta.

Reject unsupported size/memory requirements before allocating the core. Budget live epoch, LU, certificate scratch and provisional next epoch simultaneously. Do not use the existing development RSS override as admission evidence.

#### 5. Affine direction

Use `FA.affine_rhs` and `FA.solve`, followed by `NC.certify`.

Keep the signed physical equations:

\[
A\,dx+ds-b\,d\tau=r_p,\quad
A^Tdy+c\,d\tau=r_d,
\]

\[
c^Tdx+b^Tdy+d\kappa=r_g,\quad
ds+\Theta dy=h,\quad
\kappa d\tau+\tau d\kappa=r_{\tau\kappa}.
\]

Certification must bind the actual RHS and raw direction, not a corrected or reference-substituted vector.

#### 6. Predictor and combined RHS

Preserve the source predictor:

- both primal and dual cone boundaries;
- tau and kappa boundaries;
- `.995` affine boundary factor;
- actual affine `mu_aff`;
- `sigma=min(1,(mu_aff/mu)^3)`.

Source: `predictor_corrector.jl:4–50`; experimental implementation: `experimental_power_step.jl:83–119`.

Port boundary formulas only after sign, cancellation, finite-range and parity tests. A numerically different but algebraically equivalent implementation must be reviewed as such—not described as bit-identical automatically.

`FC.build` must consume the **certified affine receipt**. It forms the orthant correction, current-point Power correction and scalar product term `-dtau_aff*dkappa_aff` (`factor_combined_epoch.jl:50–76`).

`FC.solve` reuses the same LU (`79–91`). `FC.certify` verifies its actual transformed and physical direction.

#### 7. Trial and commit

Compute each trial point once in owned scratch. The residual evaluator and NP trial builder must certify those same stored words.

Preserve, without widening:

- strict primal/dual interior;
- fresh pair certification;
- componentwise residual homotopy;
- raw max-inf merit and existing scale;
- useful-progress predicate;
- `.9` default damping, `.5` contraction and 64 backtracks.

Source authority: `linesearch.jl:60–117,132–242`.

Before publishing an accepted update, prepare the next pair/epoch transaction:

1. validate all existing acceptance gates;
2. form the next epoch from the trial pair;
3. certify its affine direction;
4. prepare new owner tokens and receipts;
5. atomically publish point, pair, epoch and generation.

This adds a clearly recorded **backend-readiness requirement**, not a modified progress tolerance. If readiness fails, retain the old anchor and report the typed stage; any permitted backtracking stays within the same fixed budget.

The prebuilt next epoch should become the next iteration’s epoch, avoiding a second factorization of identical data.

#### 8. Terminal/public authority

Reuse the ordinary candidate, recovery and original-coordinate certification requirements. Do not use `merit <= 1e-8` alone as `Optimal`.

Existing terminal helpers have `ProductConeHSDState` dependencies and some legacy runtime restoration paths (`product_cone_solve.jl:395–550`). Extract route-neutral candidate/certificate logic or add explicit factor-state methods. Do not fabricate a legacy runtime to satisfy those signatures.

Require the actual requested tolerance, tau recovery condition, recovered affine residuals, cone membership, objective gap, complementarity, kappa/tau and mu/tau². Preserve reconstruction to the original user model, including objective sign/constant and equality maps.

### C. Opt-in setting and refusal

Recommended public field:

**`nonsymmetric_backend::NonsymmetricBackendChoice`**

Proposed enum values:

- `NativeNonsymmetricBackend` — default; exactly existing behavior.
- `ExperimentalHalfPowerFactorPairBackend` — explicit R0-P4 opt-in.

The enum selects a whole-epoch backend, not a cone metric preference. Lower the experimental choice internally to the fixed policy tag `ExperimentalDualHessianOneSecant`, corresponding to `NP.POLICY`.

Do not overload existing `engine`, `scaling`, `provider` or `kkt_route` names with changed meanings. `Settings` currently has these independent fields (`public/settings.jl:182–203`).

Initial compatible settings:

- native HSD engine;
- Float64;
- `kkt_route=:bordered`;
- automatic formulation/provider selection;
- no forced sparse storage;
- no equilibration;
- classic/default iteration knobs;
- single-thread execution;
- unchanged, supported requested accuracy.

The experimental execution plan explicitly resolves these to `:factor_pair_augmented_lu`, dense storage and its actual LU provider. Explicit conflicting provider/formulation/storage requests refuse rather than being ignored.

Do not silently relax an accuracy request below the currently qualified range. Extending that range is a separate qualification gate.

#### Refusal semantics

Use a structured refusal carrying:

- stage and reason enum;
- block offset, when applicable;
- accepted generation;
- attempted alpha and backtrack index;
- requested/executed backend and policy;
- numerical subreceipt.

Suggested stages: admission, root, stored geometry, metric, epoch factor, affine certificate, current-point corrector, combined certificate, trial, next epoch and public recovery.

- Invalid API syntax: normal `ArgumentError`.
- Unsupported valid request: typed admission refusal, surfaced as a dedicated unsupported-backend exception/report before iteration.
- Expected numerical failure during execution: typed numerical refusal; terminate without `Optimal` unless the unchanged accepted point independently passes the ordinary certificate.
- Programming errors such as bounds/method/ownership misuse: propagate, not swallowed as numerical failure.

**Default behavior:** no experimental selection means no attempt to admit this backend. A default-path failure does not trigger it. An explicit experimental request that fails admission does not fall back to the default path.

### D. One-secant and strict double-secant split

#### Admitted one-secant policy

The retained shadow factor supports the declared BFGS map

\[
\Theta_*=
B-\frac{Byy^TB}{y^TBy}+\frac{ss^T}{s^Ty},
\qquad B=\mu H(\widetilde s)^{-1}.
\]

It must satisfy its stored-geometry and true-BFGS metric certificate, including the first secant. It does **not** promise the second secant.

Keep the reviewed bounds and their operation budgets; in particular the existing `2^-22` geometry/decrement/metric and `64eps` transformed-coefficient / `2^-17` physical forcing gates.

#### Current-point corrector remains mandatory

One-secant **scaling** does not remove the corrector’s current-primal obligation.

`FC.build:63–68` calls `HC.compute` on `epoch.s`, not the conjugate shadow. `HC.current_factor`, at `half_power_native_corrector.jl:118–141`, independently verifies the selected current-point factor.

Retain current-point Hessian solve, third contraction, symmetry, Euler/projection and posterior checks.

#### Strict scaling is a separate, initially unavailable capability

Strict double-secant must use

\[
\widetilde y=-\nabla F(s),\quad
\Theta y=s,\quad
\Theta\widetilde y=\widetilde s,
\]

with the consistent dual identities.

The current implementation explicitly uses `workspace.primal_hessian` in its strict reference metric (`scaling3.jl:1084–1123`) and checks degree-three shadow Gram identities (`999–1034`).

For half-Power, at the actual stored current point \(s=(x,y,z)\),

\[
d=xy-z^2,\quad q=(y,x,-2z)^T,\quad
J=\begin{pmatrix}0&1&0\\1&0&0\\0&0&-2\end{pmatrix},
\]

\[
H(s)=qq^T/d^2-J/d+
\operatorname{diag}\!\left(\frac1{2x^2},\frac1{2y^2},0\right).
\]

Before strict metric construction, require a separately identified current-point factor \(L_s\) with independently enclosed

\[
\|L_s^{-1}H(s)L_s^{-T}-I\|_2\le2^{-22},
\]

and the existing Hessian-factor backward requirement. Certify the stored gradient/dual shadow consistently and preserve strict Gram symmetry, degree identities, secant-plane degeneracy tests, complement-space metric, SPD/inverse-action and both secants.

These are **necessary, not sufficient** conditions. A current-point corrector certificate alone does not qualify strict scaling, and `L(shadow)` cannot substitute for `L_s`.

First R0-P4 has no strict factor-pair implementation. A strict-factor request must refuse; it must not downgrade to one-secant.

### E. Ordered implementation plan and review gates

1. **Correct the evidence ledger.** Separate observed alpha history, requested certificate tolerance, cold-rebuild diagnostics and public-route claims.
2. **Freeze scope/admission specification.** Enumerate types, cones, dimensions, runtime context, settings and accuracy support.
3. **Add typed selector and plan metadata.** Default behavior unchanged; selector tests before numerical code.
4. **Port reviewed arithmetic into an internal package namespace.** No runtime inclusion from `validation/`; keep research wrappers and independent oracle implementation separate.
5. **Audit all coefficient/metric consumers.** Establish a denylist of dense-Theta calls unreachable from the new state.
6. **Implement canonical/reduction adapter.** Preserve row permutations, basis maps and original reconstruction. Review signs and coordinate ownership.
7. **Implement bounded memory admission and state allocation.** Include simultaneous accepted/trial/next-epoch lifetimes.
8. **Implement NP pair/context admission.** Preserve exact settings, arithmetic guards and domain/budget refusals.
9. **Implement the `NP.epoch`-based LU adapter.** Review matrix signs, full border, pivots, ownership and fingerprints.
10. **Port affine solve and NC certification.** Stream verifier coefficient bounds rather than global interval Theta. Revalidate against unchanged independent equations.
11. **Integrate predictor policy.** Test primal/dual boundaries, tau/kappa bounds and sigma using actual affine directions.
12. **Port current-point corrector and FC combined solve.** Replace generic numerical `error(...)` sites with narrowly typed refusal; programmer errors still propagate.
13. **Integrate trial scratch and existing acceptance predicates.** Test exact stored-word agreement and every rejection path.
14. **Implement next-epoch preparation and atomic commit.** Test generation/token replay, failed commits and terminal-step coverage.
15. **Integrate ordinary termination/recovery/public results.** No metric rebuild through the legacy backend; no relaxed certificate.
16. **Run pinned qualification and default-regression matrix.** Commit before tests, assert loaded roots/versions/precision in-process, hash before/after, enforce bounded processes. Independent review must close before admitting the opt-in public route.

Each numerical gate closes independently; passing step 10 does not authorize steps 13–15.

### F. Acceptance criteria and permitted claims

#### “Opt-in public Float64 half-Power route qualified”

All of the following are required:

- Actual `Model → compile/canonicalize → reduce → new backend → reconstruct → Result` execution, selected explicitly.
- Positive admission tests and pre-iteration refusal tests for every excluded combination.
- No hidden precision, dense Theta, fallback route or factor substitution.
- Complete cold-to-terminal canonical and actual benchmark runs, with original-coordinate certification at the declared requested tolerance.
- A genuine above-floor accepted repair step, unchanged acceptance predicates, and certified next epoch.
- Per-epoch affine/combined certificates and current-point corrector evidence.
- Negative controls for stored point/factor/RHS corruption, refingerprinting, stale tokens, rejected-token reuse, factor-role exchange, sign errors, zero-change trials, scalar-product omission, overflow/domain refusal and rollback failure.
- Correct reconstruction under equalities, row permutations and objective transforms.
- Float64 baseline failure retained; BF256/BF512/x4 controls retain their own arithmetic and exact input provenance.
- Default route, plans and numerical behavior remain unchanged.
- Independent design/implementation/evidence review is closed.

For unsupported infeasibility/unbounded/non-Slater regimes, safe non-success is acceptable within a clearly declared experimental scope. Claiming certificate production in those regimes requires separate original-coordinate ray qualification.

#### “Production/default Float64 Power dispatch repaired”

**Not authorized by this design.**

That stronger claim additionally requires explicit approval to change selection policy, coverage of the intended Power exponents/types/mixed products, qualification of the chosen scaling policy—including strict obligations if advertised—and the production correctness/resource matrix.

An opt-in one-secant half-Power result cannot silently become a default strict/general-Power result.

## Risks

- The reviewed helper caps are much smaller than typical production problems.
- Dense LU plus independent certification can dominate runtime and memory; this is not R2 sparse-KKT completion.
- Factor accuracy and corrector availability can still fail at otherwise interior points.
- Canonical and reduced coordinate lineages are more complex than the standalone fixture.
- Current generic exception sites can hide numerical versus programming failures unless carefully separated.
- Experimental success does not establish general convergence or authorize Exp migration.

## Need from main agent

No unresolved design decision. The supervisor confirmed the narrow initial admission scope.

## Suggested execution prompt

**No worker implementation handoff is warranted.** The parent should implement the critical numerical boundary in ordered slices above. This report authorizes neither default-dispatch changes nor broader cone/precision admission.