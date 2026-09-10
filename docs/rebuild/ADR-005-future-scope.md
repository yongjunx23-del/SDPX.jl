# ADR-005: Future scope — QP, large PSD, indirect KKT

**Status: ACCEPTED (D01, 2026-09-11).** P2, non-blocking. This ADR scopes three
candidate capabilities and commits to **implementing none of them in this
round**. It exists so the decision is recorded with evidence rather than
rediscovered later.

Baseline: `docs/rebuild/baseline.md`.

---

## 1. Ranking, and the evidence it rests on

| Candidate | Verdict | Why |
|---|---|---|
| **Large PSD** | **first to investigate** | It is the only one with a measured structural cost already in the tree, an existing opt-in module, and a falsifiable hypothesis (below). |
| **QP objective** | **second** | Highest user-visible reach, but it forces a change to the *canonical contract* and to the original-coordinate certificate — a P0-grade change, not a P2 one. |
| **Indirect KKT** | **last, and probably never for this solver** | It requires an inexact-Newton acceptance theory that this codebase's strict five-equation certificate is specifically designed to avoid. |

The ranking is deliberately *not* by perceived user demand. It is by "what can
be investigated without disturbing the certification layer", because ADR-003 §1
makes the certification layer non-negotiable.

## 2. Large PSD — chordal

### What already exists (verified, not assumed)

`src/chordal.jl` is **307 lines** and its header states plainly:

> `EXPERIMENTAL / OPT-IN — not reachable from `solve`.`

Call sites outside itself are limited to `include("chordal.jl")` in `src/SDPX.jl`
and a `chordal` **cost estimate field** in `src/preprocessing.jl`
(`ChordalCostEstimate`, `_chordal_cost`). So detection exists, the transformation
does not, and the production path does not use either.

`src/preprocessing.jl:1424` already refuses when "aggregate PSD-block density is
too high for profitable chordal" — i.e. the "no benefit" answer is already
implemented, which is the right default.

### The distinction the card demands: completion vs sparse PSD

These are **different transformations** and must not be conflated:

- **Sparse PSD (clique splitting).** Replace one `k×k` constraint `X ⪰ 0` by one
  smaller PSD constraint per maximal clique. This is exact **only if** the
  aggregate sparsity pattern is chordal *and* the PSD matrix is required to have
  the same sparsity pattern — i.e. only for **sparse PSD**, where `X`'s nonzeros
  are constrained to the pattern.
- **Chordal completion.** For a **dense** `X` whose *data* pattern is sparse, one
  must first choose a chordal extension and introduce **fill variables**. The
  resulting problem is a relaxation unless the completion is proven exact for the
  given structure.

The card requires that any chordal proposal "必须证明原问题类型、填充变量、重叠
一致性和对偶恢复正确". Both variants therefore owe four proofs: problem-class
identification, fill-variable accounting, clique-overlap consistency, and dual
recovery.

### Falsifiable hypothesis and its oracle

> **H1.** For a PSD block of side `p` whose aggregate pattern has clique number
> `ω ≪ p`, splitting into maximal cliques reduces total factorization work from
> `O(p³)` toward `O(Σ cᵢ³)` and reduces peak PSD metric storage.
>
> **Falsified if** on any constructed instance with `ω/p < 1/4` the measured
> `Σ cᵢ³` is not smaller than `p³`, or dual recovery fails the original-coordinate
> certificate.

**Independent oracle.** The existing A01 oracle family already builds PSD `svec`
maps and reconstruction maps *without* calling production code
(`test/rebuild/reference_oracles.jl`, 0 SDPX calls). The chordal oracle must
follow the same rule: construct the clique-split problem and its dual recovery
from the definition, then compare **the recovered original-coordinate
certificate** against the un-split solve. Comparing objectives alone would be the
"same wrong kernel validates itself" failure the packet forbids.

Explicitly **not** claimed: Clarabel's `skron!`-style explicit `q×q` metric is
`O(p⁴)` storage for packed `q = p(p+1)/2` and is **not** a general endpoint for
large SDP. It may be useful as a comparison implementation, nothing more.

## 3. QP objective

### Why this is P0-grade despite being scoped P2

`src/ir/canonical.jl` carries a **linear** objective only (`c`,
`objective_sign`, `objective_constant`). `src/moi_wrapper.jl` contains **zero**
occurrences of `Quadratic`. Adding a quadratic objective is therefore not an
extension — it changes the canonical contract that the whole solver is written
against.

Every one of these must be addressed, and the card requires the list be complete:

| Area | Impact |
|---|---|
| **Embedding** | The five-equation HSD system is derived for `min c'x`. A quadratic term changes the dual residual and the stationarity equation; the derivation must be redone, not patched. |
| **Residual** | `rD = A'y + cτ` becomes `A'y + (c + Px)τ`. The scaling of `P` relative to `A`,`b`,`c` must be defined before equilibration can be trusted. |
| **Dual ray** | Dual infeasibility currently means `A'y + cτ = 0` with a negative objective direction. With `P ⪰ 0` the ray condition becomes `A'y + (c + Px)τ = 0` and must hold **along the ray**; the existing ray certificate does not cover this and cannot be reused unchecked. |
| **`P = 0`** | Must reduce **exactly** to today's behaviour, bit-for-bit where the arithmetic order permits, or the change is not additive. |
| **MOI** | `ScalarQuadraticFunction` / `VectorQuadraticFunction` support, plus the MOI conic→quadratic bridge and its own feasibility semantics. |

### Falsifiable hypothesis and its oracle

> **H2.** With `P = 0`, a QP-capable build reproduces the current LP/SOC/SDP
> results **and certificates** exactly; with `P ≻ 0` on a problem with a known
> analytic optimum it terminates with a valid original-coordinate certificate.
>
> **Falsified if** any `P = 0` case differs, or a `P ≻ 0` case fails the
> certificate that the same problem passes when reformulated as an SOC
> epigraph.

The oracle is that last point: a QP can be rewritten as an SOC problem, and the
existing SOC path already has certified behaviour. That gives an **independent**
route to the same answer rather than a self-consistency check.

## 4. Indirect KKT

### Why the answer is likely "no"

The packet's own instruction is that an indirect proposal "必须提出preconditioner
与inexact-Newton验收，不搬SCS误差阈值". The second half is the binding constraint:
SCS is a first-order method whose inexact-solve tolerance is admissible *because*
it is not claiming an exact Newton step. SDPX's acceptance layer is a strict
five-equation residual plus an original-coordinate certificate (ADR-003 §1 L2/L3).
An inexact KKT solve changes `rP`/`rD` in a way that is indistinguishable from a
modeling error at the acceptance gate.

A defensible proposal would therefore need **all** of:

1. A preconditioner with a stated condition-number or spectral bound on the
   relevant operator class.
2. An **inexact-Newton acceptance condition** tied to the *current* residual
   (e.g. forcing sequence `‖η_k‖ ≤ κ‖r_k‖`), with a proof that the strict
   five-equation gate still holds at acceptance.
3. A demonstration that the acceptance gate is not merely loosened: the same
   instances must still pass the L3 original-coordinate certificate.

### Falsifiable hypothesis

> **H3.** For a sparse operator class where a direct factor exceeds a stated
> memory bound, an inexact solve meeting a forcing-sequence criterion reaches the
> same original-coordinate certificate as the direct route.
>
> **Falsified if** the inexact route needs a relaxed L3 tolerance to certify, or
> its iteration count grows without bound as the forcing sequence tightens.

The oracle is the **direct route's** certificate on the same instance — which is
only a legitimate oracle if the direct route can actually run within the memory
budget, so the hypothesis is scoped to instance sizes where it can.

## 5. What this round will NOT do, and why

- **No second algorithm in the main loop.** The card forbids it and ADR-001
  forbids a second HSD loop. All three candidates are scoped as *future* work.
- **No QP work**, because it changes the canonical contract and the certificate,
  which are P0 freeze items in this round.
- **No chordal wiring**, even though `src/chordal.jl` exists. Detection is not
  the transformation, and the "no benefit" branch must keep firing.
- **No indirect KKT prototyping on the production path.**
- **No new third-party dependency** for any of the three. P01 owns the sparse
  third-party boundary and no proposal here may pre-empt it.
- **No performance number is claimed for any candidate.** Nothing was measured
  for this ADR; the ranking rests on structural evidence in the tree, and §1 says
  so.

## 6. Entry conditions for promoting any candidate to P0

1. The A00/I01/I02 integration sequence has landed and the certification layer is
   unchanged (ADR-003 §1).
2. A falsifiable hypothesis above has an **independent** oracle that does not
   share code with the implementation under test.
3. A measured baseline exists for the instance class the candidate targets — the
   Q01 harness (`benchmark/rebuild/measure.jl`) can produce it, but its
   `prepared_solve` phase is not yet real (see Q01-F1).
4. The candidate does not require relaxing a tolerance, a certificate gate, or a
   precision to show a benefit.
