# R0-E: compensated Exp evaluator and conjugate — implementation design

**Artifact name for runtime persistence:** `r0e-compensated-exp-design.md`  
**Read-only delivery:** no files edited, no commits, no Julia solver validation performed.

**Source inspected:** SDPX 0.6.1, `bc38f727d4c9ef885e6d408b50b2fa96cfc83d6c`. The relevant Exp/scaling sources remained unchanged during inspection; concurrent parent additions under `src/hsd/factor_pair/` were not treated as reviewed source.

## Inherited decisions

- The frozen problems are well-posed. Their current Float64 failure is first observed in conjugate/scaling construction.
- Preserve root tolerance, iteration budget, scaling predicates, line-search acceptance, original-coordinate certification, and explicit fallback policy.
- Actual stored points, gradients, factors, and directions—not ideal mathematical substitutes—remain the objects being certified.
- Compensation is an explicitly declared arithmetic network built from Float64 operations. It must not secretly invoke BigFloat, MultiFloat promotion, or a fallback solver.
- Parent owns critical numerical implementation.

**Scope confirmed with the supervisor during this consultation:**

> First deliverable is a **validation-only compensated Exp evaluator/conjugate with independent stored-word replay**. No public/default dispatch change, no new public Exp backend, and no expansion of R0-P4 half-Power admission. Whole-epoch Exp migration remains conditional on downstream evidence.

## Diagnosis

### 1. Failure B is an allowance failure, not evidence of an unrepresentable root

At `src/cones/exp_logarithmic.jl:146–147`, the actual failing predicate is:

\[
0<|\psi|,\qquad E_\psi\le|\psi|/2.
\]

The exception is `"unresolved Fenchel margin error"`.

Its error ledger begins at lines 102–114 with

\[
E_\rho=
\frac{|R|+\gamma_{64}\,\mathrm{root\_work}}
     {1+1/(1+\rho)},
\qquad
\gamma_{64}=\frac{64\epsilon}{1-64\epsilon},
\]

then propagates through reconstructed \(y,z,l,x,\psi\). Here `root_work` includes several order-one quantities even when \(D,\rho\) are order \(10^{-6}\). The subsequent coordinate bounds are amplified by \(1/\rho\), and the margin ledger treats correlated reconstruction terms independently.

The capture report’s sentence “the rho root itself is unrepresentable in Float64” is therefore unsupported. The observed exception says the **current bound is unresolved**, not that no adequately accurate Float64 root exists.

A second distinction matters: this helper evaluates the **conjugate shadow’s** margin. In fixture B, that margin is approximately \(0.140918\), not the original trial point’s \(10^{-6}\)-scale margin. The two points must not share an error ledger.

### 2. Failure A combines evaluation errors and genuine stored-coordinate defects

The strict scaling path uses:

- `_ns_scaling_gauged_dot3`, `scaling3.jl:165–177`;
- `_ns_scaling_double_secant!`, `scaling3.jl:999–1034`;
- `_ns_scaling_relative_difference`, `scaling3.jl:211–226`.

The gauged dot uses ordinary products and additions after power-of-two normalization. Normalization prevents range failure but does not cure cancellation.

The actual comparison is equivalent, within its supported gauge, to

\[
|a-b|\le t(|a|+|b|),\qquad t=8192\epsilon.
\]

Near \(a=b=3\), the absolute budget is approximately

\[
6t=1.0913936421275\times10^{-11}.
\]

It is **not** an absolute `8192eps` test.

Compensated dot evaluation can remove arithmetic error in this predicate. It cannot remove a real defect in the stored vectors.

### 3. Independent diagnostic: accurate evaluation alone does not guarantee strict acceptance

I performed read-only numerical analysis using:

- exact conversion of the supplied Float64 inputs;
- mpmath calculations at 256 and 384 bits;
- a numerical monotone-root reference;
- explicit Float64 rounding of \(\rho,y,z,x\);
- exact rational dot products of the resulting Float64 words.

The rounded outputs agreed between both reference precisions. This is diagnostic evidence, not a directed transcendental certificate or validation of the proposed implementation.

Candidate construction was:

\[
r=\operatorname{Float64}(\rho_*),\quad
Y=\operatorname{Float64}\!\left(-\frac1{ur}\right),\quad
Z=\operatorname{Float64}\!\left(\frac{1+r}{rw}\right),
\]
\[
X=\operatorname{Float64}\!\left(Y\log(Z/Y)+1/u\right).
\]

For the second column below, the mathematical primal gradient was evaluated independently and each component rounded to Float64.

| Fixture | Exact \(\langle d,(X,Y,Z)\rangle-3\) | Exact \(\langle-\operatorname{Float64}(\nabla F(s)),s\rangle-3\) |
|---|---:|---:|
| A, block 7 | \(-1.1583611\times10^{-13}\) | \(-6.2736536\times10^{-11}\) |
| B, block 4 | \(+5.4984549\times10^{-11}\) | \(+3.0556294\times10^{-11}\) |
| B, block 7 | \(+3.6994610\times10^{-11}\) | \(+1.1058668\times10^{-10}\) |
| B, block 10 | \(+1.7404303\times10^{-11}\) | \(-5.2757772\times10^{-10}\) |

The **old captured A7 shadow** has exact dot defect

\[
-4.880169398448517\times10^{-11},
\]

so its failure is not merely an inaccurate three-term summation.

Consequences:

1. Accurate reconstruction can substantially improve A7’s primal shadow.
2. Rounding the true current-primal gradient into three Float64 values can still violate the unchanged Euler/shadow identity.
3. B’s accurately reconstructed Float64 shadows can still violate the strict cross identity.
4. These examples do not prove that every alternative representation is impossible. They do falsify a blanket promise that compensating scalar formulas alone repairs the existing strict runtime.

**Expected first-deliverable outcome:** much tighter, truthful arithmetic and reconstruction certificates, with explicit stored-geometry refusals where necessary—not a promised Float64 `Optimal`.

## Drift / contradiction check

Correct these interpretations in subsequent evidence:

- `psi = z - x` is not this barrier’s margin. It is
  \[
  \psi=y\log(z/y)-x.
  \]
- A converged \(\rho\) solve does not certify stored-shadow geometry.
- A successful gradient-replay allowance does not certify the strict cross identities.
- A failure before metric construction does not qualify the downstream metric, corrector, or Newton path. They were not the first observed failing stage.
- The Exp branch at `conjugate3.jl:1453–1460` calls the scalar conjugate without passing a warm root. Historical warm-state metadata is not evidence that this scalar Exp solve used warm initialization.
- The existing log-work helper explicitly disclaims a rigorous transcendental enclosure. Its work counters cannot become a DD error certificate merely by multiplying them by a smaller epsilon.
- `HSD_FORMULATION.md` is marked historical and contains legacy sign conventions. This repair must not port or change HSD equations from that document.

## Recommendation

### A. Implementation locations and boundaries

Create proposed research files:

- `validation/scientific_core/exp_runtime/compensated_exp_reference.jl`
- `validation/scientific_core/exp_runtime/test_compensated_exp_reference.jl`
- `validation/scientific_core/exp_runtime/run_compensated_exp_replay.jl`

The names above are proposed, not existing implemented interfaces.

| Source reference at inspected pin | Research implementation action |
|---|---|
| `exponential.jl:64–94`, `_nonsymmetric_positive_log_ratio*` | Add a separate guarded compensated log-ratio helper. Retain the legacy helper for baseline comparisons. |
| `exp_logarithmic.jl:6–19`, `_exp_logarithmic_terms` | Research equivalent computes a compensated \(l\) and \(\psi\), retaining their error bounds. |
| `exp_logarithmic.jl:29–35`, `_exp_logarithmic_gradient_values` | Research gradient consumes the same compensated terms. Record both expansion-valued and rounded Float64 results. |
| `exp_logarithmic.jl:183–184`, `exp_logarithmic_conjugate!` | Research equivalent constructs compensated \(l_0\), \(v/u\), and \(D\). |
| `exp_logarithmic.jl:186–197` | Research safeguarded root loop evaluates a compensated residual with a certified bound; retains the existing tolerance and iteration cap. |
| `exp_logarithmic.jl:199–209` | Reconstruct and round \(y,z\) first; recompute the same compensated log-ratio from those stored values; then form and round \(x\). Independently replay the stored triple. |
| `exp_logarithmic.jl:92–170` | New, separately derived reconstruction ledger for the new arithmetic network. Keep legacy allowances observable; do not shrink their constants while retaining old arithmetic. |
| `scaling3.jl:165–177, 1010–1034` | Research-only exact/compensated pairing audit alongside the unchanged production pairing and predicate. |
| `conjugate3.jl:134–160, 223–245` | Later factor-validation experiment only. Do not change factor construction in the first slice. |
| `conjugate3.jl:1453–1528` | Observe conjugate, factor, and Cartesian-check stage boundaries. No first-slice production replacement. |

**No source-method override is needed for the first experiment.** Run explicit research functions on captured inputs. This avoids accidentally changing production membership, barrier, corrector, and certificate behavior through a shared helper.

### B. Arithmetic contract and reusable helpers

Use a research value of the form

\[
\mathcal B=(h,\ell,E),\qquad
|v-(h+\ell)|\le E,
\]

where every field is Float64 and \(E\ge0\) is rounded outward.

Record explicitly:

- storage inputs/outputs: Float64;
- intermediate representation: compensated two-component Float64 plus error radius;
- independent verifier: separate BigFloat/MPFR calculation;
- no reference value fed back into candidate construction.

#### Existing EFT helpers

At the inspected pin, there is no top-level `SDPX._two_prod` or `SDPX._two_sum` implementation to assume.

The reusable research transforms are in:

`validation/scientific_core/power_half_phi_reference.jl`

- `PowerHalfPhiReference._two_prod`, lines 16–20:
  \[
  p=\mathrm{RN}(ab),\quad e=\operatorname{fma}(a,b,-p).
  \]
  Under its arithmetic/range preconditions, \(p+e=ab\) exactly.
- `_two_sum`, lines 22–32: general TwoSum, without a magnitude-order assumption; \(s+e=a+b\) exactly under its preconditions.
- `_grow`, lines 33–45: retains every component and represents the exact sum of the supplied Float64 terms.
- `_runtime_ok`, lines 101–104: pinned Julia/platform/math/rounding/FTZ/FMA context.

Reuse their transforms through a private research namespace. Do not reuse the half-Power-specific polynomial or log-series domain proof for Exp.

**Important:** `_enclose_sum` collapses an expansion into ordinary Float64 interval endpoints. Doing that to an order-one logarithm before its cancellation can lose the precision being recovered. Keep the two components through cancellation; bound only the discarded remainder.

#### Range and context guards

Initially inherit the verified Julia 1.12.6/aarch64 Darwin context, nearest rounding, default math, FTZ disabled, explicit FMA. Reject unsupported contexts; do not mutate them to obtain admission.

Every EFT invocation must have a range proof, not merely finite outputs. A conservative first experiment can require every nonzero EFT operand exponent to lie in `[-450,450]`, with checks on intermediate results. This leaves normal-range room for exact product residuals. Refuse nonzero subnormal inputs/intermediates or exhausted expansion capacity.

This guard is a bounded research choice, not general Exp range support.

#### Error propagation

For centers \(A=h_A+\ell_A\), \(B=h_B+\ell_B\):

- **Addition:** sum components with TwoSum; if compression discards an exact expansion remainder of magnitude at most \(\delta\),
  \[
  E_{A+B}\le E_A+E_B+\delta.
  \]
- **Multiplication:** form all four center products using TwoProd, then accumulate:
  \[
  E_{AB}\le |A|E_B+|B|E_A+E_AE_B+\delta.
  \]
- **Division:** obtain a two-component quotient candidate \(Q\), and evaluate the center residual \(r=A-QB\) through EFT. If \(|B|>E_B\),
  \[
  E_{A/B}\le
  \frac{|r|+E_A+|Q|E_B}{|B|-E_B}.
  \]

All bounds use outward arithmetic and denominator lower bounds. Compression must bound its actual discarded components. It must not assume that an arbitrary “DD” operator has a universal \(u^2\) error.

Operation accounting:

- center addition: four input components;
- center multiplication: four TwoProd calls, at most eight product components before accumulation;
- scalar three-term dot: three TwoProd calls, six components before accumulation;
- division: two quotient estimates followed by an independently enclosed center residual.

Record actual TwoSum/TwoProd counts and discarded-tail bounds. Use finite implementation caps; exhausting them is a typed refusal.

### C. Compensation order and formulas

#### C1. Compensated logarithms \(l\) and \(l_0\)

An FMA applied after a rounded libm logarithm does not recover that logarithm’s lost bits.

For the first certified candidate, avoid assuming a new libm accuracy contract. Use bounded range reduction and a rational atanh series.

For positive exact Float64 inputs \(a,b\), extract exact binary exponents:

\[
a=m_a2^{e_a},\quad b=m_b2^{e_b},\quad
m_a,m_b\in[1/2,1).
\]

Set

\[
t=\frac{m_a-m_b}{m_a+m_b},\qquad |t|<1/3,
\]

evaluating numerator, denominator, and division with the compensated arithmetic above. Then

\[
\log(a/b)
=(e_a-e_b)\log2+
2\sum_{j=0}^{39}\frac{t^{2j+1}}{2j+1}+\mathcal R.
\]

Use the same series at \(t=1/3\) to enclose \(\log2\). Generate rational coefficients through enclosed division, not an unqualified table of rounded constants.

For an enclosing \(q\ge|t|\), \(q<1\),

\[
|\mathcal R|
\le
\frac{2q^{81}}{81(1-q^2)}.
\]

At \(q=1/3\), this is approximately \(6.26435\times10^{-41}\). Use the actual outward \(q\), not the ideal value, for the bound.

A Horner implementation requires 39 recurrence multiplications, 39 additions, the initial square, and the final multiplication by \(t\), plus range-reduction/coefficient work.

**Research helper acceptance target:**

\[
E_{\log(a/b)}\le 2^{-90}(1+|e_a-e_b|).
\]

This is a target checked from the returned bound, not an assumed theorem that every input will pass. If it fails, return `:log_enclosure_unresolved`.

Use this same helper for:

\[
l=\log(z/y),\qquad l_0=\log(w/(-u)).
\]

Retain legacy log values and branch information for A/B attribution. The new network is intentionally not bit-identical to the old one.

For \(\log(1+\rho)\), do **not** first round `1+rho`. Use

\[
t_\rho=\frac{\rho}{2+\rho}
\]

and the same atanh-series machinery on the compensated denominator.

#### C2. Margins \(\psi\) and \(D\)

Compute

\[
\psi=y\,l-x
\]

by multiplying both log components by \(y\) and retaining the subtraction residual.

For exact stored \(x,y\),

\[
E_\psi\le |y|E_l+\delta_{\mathrm{mul/sub}}.
\]

Do not round \(l\) back to one Float64 before the multiplication.

Compute

\[
V=v/u,\qquad C=1-V,\qquad D=C+l_0
\]

with compensated division and sums:

\[
E_D\le E_V+E_{l_0}+\delta_C+\delta_D.
\]

Require \(D_{\mathrm{lo}}>0\). An interval straddling zero is unresolved, not strictly dual interior.

**Do not replace legacy `psi_work = |y*l|+|x|+|psi|` by `|psi|`.** That would discard legitimate work without proof. The valid change is to derive the new network’s operation-specific error using retained EFT residuals.

#### C3. Root residual and safeguarded solve

The equation remains

\[
f(\rho)=\rho+\log(1+\rho)-D=0.
\]

With \(D\in[D_{\mathrm{lo}},D_{\mathrm{hi}}]\), initialize an enclosing bracket

\[
[\rho_{\mathrm{lo}},\rho_{\mathrm{hi}}]
=[D_{\mathrm{lo}}/2,D_{\mathrm{hi}}].
\]

Evaluate \(f(r)\) as a compensated sum at the actual stored Float64 iterate \(r\). Let \(R_{\max}\) enclose \(|f(r)|\).

Keep the source threshold:

\[
R_{\max}\le16\epsilon\,D_{\mathrm{lo}}.
\]

This certifies the unchanged relative tolerance against the real margin; it does not widen it.

Preserve safeguarded Newton and the 64-iteration default cap. Update a bracket endpoint only when the residual sign is certified. A straddling sign that also fails the residual test is unresolved; do not invent a sign or relax stopping.

For a root-location error bound, use the minimum derivative over the enclosing bracket:

\[
E_\rho\le
\frac{R_{\max}}
{1+1/(1+\rho_{\mathrm{hi}})}.
\]

Using the derivative only at the candidate is not generally a rigorous finite-error lower bound.

For the captured cases, source root thresholds are approximately:

- A: \(2.0\times10^{-19}\);
- B4/B7: \(5.7\times10^{-21}\);
- B10: \(2.7\times10^{-21}\).

A successfully certified \(10^{-27}\)-scale compensated log/margin error is comfortably below those thresholds. It does not eliminate the final Float64 root rounding, which must remain in the residual certificate.

#### C4. Reconstruction from actual stored words

For the accepted scalar root word \(r\), define

\[
Y_0=-\frac1{ur},\qquad
Z_0=\frac{1+r}{rw},\qquad
p_*=-1/u.
\]

1. Compute enclosed \(Y_0,Z_0\).
2. Round to stored Float64 \(Y,Z\), recording \(e_Y=Y-Y_0\), \(e_Z=Z-Z_0\) bounds.
3. Recompute the compensated \(L=\log(Z/Y)\) from **stored** \(Y,Z\).
4. Compute an enclosed target \(YL+1/u\).
5. Round to stored Float64 \(X\), recording its rounding error.
6. Independently replay
   \[
   P=Y\log(Z/Y)-X
   \]
   from the stored triple.

The last rounding of \(X\) cannot be removed from the ledger. If \(X\) is order \(10^5\), its rounding error can be order \(10^{-11}\), despite a much more accurate log kernel.

Never substitute \(p_*\) for the actual stored-point margin \(P\).

### D. Correlation-aware replay bound

The old forward chain loses a useful exact identity.

For ideal \(Y_0,Z_0\) at any positive \(r\), let

\[
L_0=\log(Z_0/Y_0)=-l_0+\log(1+r).
\]

Using \(p_*=-1/u\), the ideal gradient satisfies

\[
g_1+u=0,\qquad
g_2+v=u\,f(r),\qquad
g_3+w=0.
\]

Thus root error enters the second gradient equation directly through \(u f(r)\). It need not be pessimistically propagated through two large, correlated coordinates and then added repeatedly to the reconstructed margin.

Let certified positive lower bounds be \(P_{\min},Y_{\min},Z_{\min}\), and let

\[
E_P\ge|P-p_*|,\quad E_Y\ge|Y-Y_0|,\quad E_Z\ge|Z-Z_0|.
\]

Under the unchanged half-coordinate guards,

\[
E_L\ge|\log(Z/Y)-L_0|
\]

can be obtained from independent evaluation or bounded by

\[
2(E_Y/Y_0+E_Z/Z_0).
\]

Define

\[
B_1=\frac{E_P}{P_{\min}p_*},
\]

\[
B_2=|u|R_{\max}
+\frac{E_L}{P_{\min}}
+|L_0-1|B_1
+\frac{E_Y}{Y_{\min}Y_0},
\]

\[
B_3=
\frac{E_{Y/Z}}{P_{\min}}
+\left|\frac{Y_0}{Z_0}\right|B_1
+\frac{E_Z}{Z_{\min}Z_0},
\]

where a valid ratio bound is

\[
E_{Y/Z}\le
\frac{E_Y}{Z_{\min}}
+\frac{Y_0E_Z}{Z_{\min}Z_0}.
\]

Add the compensated gradient evaluator’s own error and the final gradient-component rounding error separately.

These bounds:

- retain degree-\(-1\) gradient units;
- retain actual \(X,Y,Z\) rounding;
- preserve half-coordinate and half-margin safeguards;
- avoid pretending that a smaller margin expression itself proves smaller error.

Keep these distinct:

1. root residual bound;
2. distance to the root-defined reconstruction;
3. actual stored-shadow margin;
4. actual stored-gradient residual;
5. strict pairing/geometry predicates.

A replay bound explaining a rounding error is not permission to ignore a failed strict pairing gate.

### E. Independent pairing and geometry audit

For every relevant pairing, compute two results:

1. the unchanged production gauged dot and predicate;
2. an exact-expansion dot of the actual stored Float64 words.

Use three TwoProd calls and an exact six-term expansion sum. Compare against the **same** sum-work inequality:

\[
|m-3|\le t(|m|+3),
\]
\[
|m_{12}-m_{21}|\le t(|m_{12}|+|m_{21}|).
\]

Do not replace these with a product-work-relative dot certificate: that would enlarge the permitted cancellation-dependent defect.

Classify separately:

- `:pairing_evaluation_error`: exact stored-word predicate passes; production evaluation fails;
- `:stored_shadow_identity_failure`: exact stored-word predicate fails;
- `:stored_gradient_identity_failure`: rounded current-primal gradient fails;
- `:predicate_unresolved`: enclosure straddles the unchanged decision boundary.

Do not overwrite a pairing with the theoretical value `3`. Do not rescale or adjust the shadow/gradient to force the identity.

For the fixtures, the target arithmetic error can be far below the approximately \(1.09\times10^{-11}\) gate. **The total stored-object defect cannot be promised below it**, as the diagnostic table demonstrates.

### F. Result and refusal contract

The research receipt should bind:

- exact input words;
- arithmetic context and implementation/version identity;
- owner/generation if reused across calls;
- compensated log/margin values and radii;
- root bracket, iterations, residual enclosure;
- stored output words and reconstruction errors;
- independently replayed gradient and pairing results;
- operation counts and refusal stage.

Use separate statuses such as:

- `root_certified`;
- `conjugate_replay_certified`;
- `stored_geometry_certified`;
- `unsupported`;
- `numerical_refusal`.

A candidate may be retained as **diagnostic data** after refusal, but accepted output buffers must remain unchanged.

Initially, factor certification is a separate, unimplemented capability. A conjugate receipt must not claim factor, scaling, Newton, or accepted-step authority.

## Gates and policies that remain unchanged

1. Primal/dual finite-domain requirements and strict interior requirements.
2. Root equation, `16eps` relative residual threshold, and default maximum 64 iterations.
3. Half-coordinate and half-margin safeguards.
4. Gradient-unit dimensional consistency.
5. Strict shadow identities, Gram symmetry, and actual sum-work tolerance.
6. `validation_tolerance = 8192eps`.
7. `degeneracy_tolerance = 128eps`, and all subsequent factor/secant/inverse tests.
8. Strict versus explicitly permitted dual-Hessian fallback policy.
9. Common global HSD \(\mu\).
10. Predictor, corrector, homotopy, merit, progress, damping, and backtracking policies.
11. Requested public solution/certificate tolerances and original-coordinate recovery.
12. Default Exp dispatch and R0-P4’s half-Power-only admission.

The baseline already exhibits explicitly reported **cone-scaling** fallbacks. This is distinct from the empty **KKT-route** fallback chain. Add no new fallback and do not conflate the two.

## Test vectors and negative controls

### Frozen exact fixtures

Load Float64 words from the capture’s hexadecimal fields, not shortened decimal reconstructions.

**A7:**

- \(s=(1.3592969807740543,\ 1.2381483457908147,\ 3.7116126129914613)\)
- \(d=(-3.7115577834073776,\ 0.37488694027023745,\ 1.2342973253738969)\)
- iteration 11, search 12, backtrack 0, block 7.

**B4:**

- \(s=(2.5987120767380656,\ 2.3655624340181229,\ 7.0963207400760488)\)
- \(d=(-7.096318086434569,\ 0.6996695192106015,\ 2.365481347097429)\)

**B7:**

- \(s=(2.5987114070521948,\ 2.3655556108413434,\ 7.0963207400760488)\)
- \(d=(-7.0963180864345965,\ 0.69966951918709142,\ 2.3654813490516893)\)

**B10:**

- \(s=(2.5986781856659027,\ 2.3652053318971311,\ 7.0963207400760488)\)
- \(d=(-7.0963180864466064,\ 0.69966950300778219,\ 2.3654793769942568)\)

B is iteration 16, search 17, backtrack 0. Add all available captured records from iterations 16–24, not just the first B point.

Preserve the distinction between a record’s active failure and a later fresh replay. Sibling workspace fields may be historical when the runtime short-circuits at an earlier block.

### Required assertions

- Directed independent reference containment for \(l,l_0,D,\rho,P,g\).
- Exact rational agreement for all stored-word pairings.
- A7 old shadow is rejected by the exact unchanged identity predicate.
- Improved A7 shadow may pass its cross identity while the rounded current-primal gradient still fails; the receipt must report both.
- B no longer gets mislabeled “root unrepresentable” when the actual failing authority is the replay bound or stored geometry.
- A smaller bound must enclose the independently measured error.
- Fixed positive reference points from `test/exp_logarithmic_conjugate.jl` remain useful controls.
- Power-of-two inverse primal/dual scaling preserves dimensionless decisions within the declared range.

### Negative controls

- NaN/Inf, nonpositive \(y,z,w\), invalid \(u\), zero/negative/unresolved \(D\).
- Subnormal/overflow intermediates, unsupported rounding/math context, exhausted operation budget.
- A root perturbation large enough to fail the unchanged `16eps` threshold.
- Perturbed log low component or underreported radius: independent enclosure check must fail.
- Corrupt stored \(X\) after construction while retaining the old root receipt.
- Substitute ideal \(p_*\) for actual \(P\): independent stored-point replay must detect it.
- Replace an actual failed dot by `3`, or normalize a shadow to enforce it: rejection.
- Corrupt output gradients or exchange current-primal and conjugate-shadow roles.
- Stale owner/generation, modified input words, rejected receipt reuse, and output aliasing.
- Failure after partial reconstruction must not publish partially changed accepted outputs.
- If later connected to factors: corrupt factor words or substitute an SPD but incorrect factor; SPD alone must not pass.

## Expected outcomes and falsification

### First research slice

No public Float64 status change is expected: production dispatch remains untouched.

The compensated evaluator should either:

- produce independently valid bounds substantially tighter than the old allowance ledger; or
- refuse with the correct arithmetic, range, root, replay, or stored-geometry reason.

Existing strict geometry may still refuse. That is a useful, truthful result.

### Later end-to-end qualification

Only after separately reviewed integration would successful repair mean:

- `exp_entropy_small`: certified `optimal`, objective consistent with
  \[
  -\log3=-1.098612288668109691\ldots;
  \]
- `exp_logsumexp_small`: certified `optimal`, objective consistent with the frozen binary-input target approximately `1.4040073747450363`;
- all residuals, complementarity quantities, membership tests, and recovered-objective tests pass their actual requested tolerances;
- no hidden fallback or relaxation.

There is no defensible predicted iteration count or exact final residual from the available evidence.

### Falsification

The scalar-only repair hypothesis is falsified if:

- improved root/margin evaluation still leaves exact stored pairings outside the unchanged gate;
- tighter reported bounds fail independent containment;
- a failure merely moves from replay allowance to factor/metric/corrector construction;
- progress still requires the terminal near-zero step;
- success requires larger tolerances, changed fallback policy, or substituted ideal geometry.

The diagnostic table already demonstrates the first limitation for the straightforward Float64 reconstruction.

## Minimum bounded experiment

Run **one fixture-replay experiment first**, not repeated full solves.

Configuration:

- committed isolated checkout;
- `--startup-file=no --threads=1 --gcthreads=1 --heap-size-hint=2G`;
- BLAS/OMP/MKL threads 1;
- owned-process-group watchdog, **180-second wall cap**;
- in-process source root, HEAD, Julia version, arithmetic-context, and before/after hashes;
- no reference computation inside the candidate constructor.

Within that process:

1. Freeze the four exact fixtures above.
2. Replay the unchanged scalar conjugate, capturing the precise exception.
3. Evaluate the research compensated candidate.
4. Independently verify logs, margins, root residuals, and stored gradients.
5. Evaluate exact rational stored-word pairings.
6. Print the first differing authority and all relevant bounds.

Compare four explicit variants:

- unchanged baseline;
- compensated evaluator/reconstruction;
- compensated pairing evaluation only;
- both.

This distinguishes log/margin error, conservative error propagation, pairing-evaluation error, and genuine stored-object defects.

If this closes the arithmetic slice, run the complete 172-record replay next in its own bounded process. Full entropy/logsumexp solves belong to later separately bounded, explicitly instrumented integration experiments.

## Gate for a later whole-epoch Exp backend

This report does not authorize that backend. Evidence needed to justify its design and implementation includes:

1. Independently verified compensated arithmetic and reconstruction bounds.
2. A declared representation that can satisfy its chosen stored-shadow/scaling obligations without altering inputs or enforcing identities artificially.
3. Fresh certified monotone-\(\rho\) root and true stored-point geometry at each candidate—not Power’s polynomial “full-gap” machinery transplanted by name.
4. Separate current-primal factor and corrector certificates.
5. If dense metric materialization fails, direct evidence identifying that failure and a factor-preserving consumer map.
6. Explicit strict versus one-secant policy approval. R0-P4’s Power approval does not extend automatically to Exp.
7. Affine/combined Newton certificates, unchanged step acceptance, rollback, and next-epoch readiness.
8. Actual canonicalization/recovery and independent original-coordinate terminal certification for both frozen models.

Items 1–5 can justify proceeding with a new backend design. Items 6–8 are required before claiming that backend qualified; terminal certification cannot be presumed in advance.

## Risks

- Cartesian Float64 shadow/gradient storage can remain the limiting representation after scalar arithmetic improves.
- A certified finite-series log requires complete coefficient, truncation, range, and rounding accounting; existing libm work counters do not supply it.
- Correlation-aware reconstruction formulas must be checked independently. Reusing the constructor’s ideal identities as the verifier would recreate circular certification.
- Exp factor conditioning and dense-metric publication may introduce the next failure.
- This analysis did not reconstruct all 172 records or run a Julia candidate.
- The mpmath diagnostic is not a proof of correctly rounded transcendental values. Its stored-word dot calculations are exact; future validation should use directed reference enclosures.
- A blanket prohibition on all existing fallback observations would misdescribe the baseline. The requirement is no **new or hidden** fallback.

## Need from main agent

No unresolved scope decision: the supervisor approved the validation-only first slice.

During implementation:

- keep candidate construction and independent verification separate;
- retain truthful stored-geometry refusals;
- correct the “unrepresentable rho” interpretation;
- do not promise public `Optimal` from this scalar repair;
- obtain a separate decision before changing Exp representation or scaling policy.

## Suggested execution prompt

**No worker implementation handoff is warranted.** The parent should implement the critical numerical research slice above.

The immediate deliverable is a bounded, independently checked evaluator/conjugate receipt that explains A and B accurately—not a public Exp backend or a declaration that R0-E is closed.