# ADR-002: Provider contract

**Status: ACCEPTED (A00, 2026-09-11).** Binding on M01–M03, B01–B04, P01–P03,
S05, and on I01/I02.

Scope: the minimum adapter contract between SDPX and the numerical providers
(MFLA, BFLA, stdlib). The packet's §3.5 is explicit that this round must **not**
create a fourth large LA framework: SDPX defines a minimal contract in-tree;
MFLA/BFLA keep their own public APIs; semantics are aligned by a shared,
solver-independent test set.

---

## 1. Environment reality this ADR must respect

From `docs/rebuild/baseline.md` §2:

- The default SDPX environment **does not contain** `MultiFloats`,
  `MultiFloatLinearAlgebra`, `BigFloatLinearAlgebra`, or `QDLDL`. They are
  `[weakdeps]`.
- Therefore **every MF/BF capability claim is unverified by `Pkg.test()`**.
- `.github/workflows/provider-matrix.yml` is **manual-only**, with a recorded
  policy that "the MFLA legs of the matrix fail on every CI". There is
  currently **no automated MF/BF coverage**.
- MFLA local HEAD (`50e6e0b`) is **ahead of** the packet's frozen `b38dea1`
  (v0.4.0). Unresolved; see §6.

A contract that cannot be exercised is a claim, not a contract. Every acceptor
of this ADR must say which environment it was verified in.

## 2. Operation set

Names are proposals; the *semantics* are the decision.

| Operation | Semantics | Must NOT do |
|---|---|---|
| `capabilities(request)` | report exact capability facts only | perform a trial factorization or a benchmark |
| `prepare_factor!(handle, spec)` | allocate capacity; establish symbolic/shape/precision | solve |
| `refactor_numeric!(handle, values)` | numeric factorization; return a generation + report | change the symbolic structure |
| `solve_into!(dest, handle, rhs, op)` | vector/matrix, N/T explicitly named | allocate a fresh factor; silently transpose |
| `factor_summary(handle)` | **O(1) small report** | allocate a matrix, recompute inertia, or copy the factor |
| `copy_operator_snapshot(handle)` | explicit, expensive copy | be used on a hot path |
| `inspect_factor(handle, audit_spec)` | explicit deep check | substitute for the routine numerical residual |
| `invalidate_numeric!(handle)` | revoke the numeric factor only | revoke the symbolic lease |

## 3. Capability facts that must be described

Per packet §3.5, these boolean labels are **insufficient on their own**:
`sparse=true`, `threading=true`, `multi_rhs=true`. Each is a claim that has
already been misread once; the packet names the specific traps:

- `multi_rhs=true` does **not** mean batch throughput — it may mean per-column
  calls in a loop. The contract must say which.
- `threading=true` does **not** mean *that* factorization is parallel.
- `sparse` must not be permitted to fall back to dense and exhaust memory.

Required description fields: arithmetic and bit width; rectangular vs square;
symmetric vs general; triangle convention; index width; in-place destination
ownership; transpose solve support; symbolic reuse; failure semantics;
concurrency limits; whether batch RHS is genuinely batched; and whether the
numeric kernel can implicitly convert precision.

## 4. Failure semantics (the load-bearing part)

Packet §3.6 names the concrete hazard: a provider may legitimately retain its
**physical** factor when an input pre-check fails (a strong exception guarantee).
That is the provider's right. But:

> SDPX 的新 Newton 请求不能因此继续拿旧 factor 解新 RHS。adapter 必须立即撤销
> 旧逻辑 lease；物理内容是否保留可由 provider 自己决定。

**Decision:** on any failed `refactor_numeric!`, the adapter must revoke the
logical lease *before* returning, regardless of whether the provider kept the
physical factor. A subsequent solve must fail closed rather than reuse.

Corollary: SDPX must never infer freshness from "the factor object still exists".

## 5. What each side owns

Reaffirming ADR-001 §2 for the provider boundary:

- **Provider owns**: physical factor, pivots, block grammar, factor scratch, the
  `provider_generation` counter.
- **SDPX owns**: the current Newton operator, the authorized factor input, and
  the logical lease binding `matrix_epoch → provider_generation`.

SDPX must not interpret provider factor storage. The provider must not decide
SDPX's numeric policy.

## 6. Unresolved at A00 — MFLA SHA drift

The packet freezes MFLA at `b38dea1` (v0.4.0). The local checkout is at
`50e6e0b`. No task in this packet has yet declared which revision it measures
against, and a result obtained at `50e6e0b` may not be attributed to `b38dea1`.

**Decision:** this ADR does not resolve the drift. It records it as a blocking
prerequisite for M01–M03 and P02: the first MFLA task to start must state the
revision it measured and, if it differs from `b38dea1`, either (a) justify the
newer revision with its diff, or (b) measure at the frozen revision. Silently
measuring the newer one and reporting it as `b38dea1` is forbidden.

> **RESOLVED — see §10 (added 2026-09-11).** M01, the first MFLA task, stated its
> revision, and §10 records the diffs that satisfy option (a) for both providers.
> Both checkouts have also moved again since §8/§9 were written, because packet
> commits landed on top of them, so the revision labels in those two sections are
> historical rather than current.

## 7. Consequences

- A capability claim without an environment + revision + test is not accepted.
- `factor_summary` being O(1) is a hard requirement: a summary that recomputes
  inertia is a numeric operation and would need its own gate.
- Provider-gated tests must **skip with a reason** when the provider env is
  absent, never pass silently. `test/sparse_qdldl_provider.jl` is the reference
  pattern: a skip is visible in the summary; a missing test is not.

---

## 8. Verification of §4 against the actual BFLA code (added 2026-09-11)

§4 asserted the "retained physical factor" hazard from the packet's prose. It has
now been checked against the real provider source at BFLA `f95d3e6`
(`src/caches.jl`), and the packet's description is **accurate**.

All four BFLA `factorize!` methods — `BFLACholeskyCache` (:376), `BFLALUCache`
(:749), `BFLALDLTCache` (:990), `BFLARRQRCache` (:1205) — use a **two-phase**
design:

```julia
# Preflight: shape/precision checks that do not mutate factor storage. A
# preflight error preserves the previous (possibly successful) factor.
_cache_require_prepared(cache, "factorize!")
_require_cache_matrix(cache, A, "factorize!")
# Commit phase: invalidate the old success first, then factorize. Any
# exception leaves the status non-success (never a stale :success).
cache.status = FactorStatus(:unprepared, nothing)
```

Two consequences that sharpen §4 rather than merely restating it:

1. **BFLA's `:success` cannot go stale within the commit phase** — the status is
   cleared *before* the work. That is a genuine and documented guarantee.
2. **It does not extend to the preflight phase.** Preflight deliberately does not
   touch `status` or storage, so after a *preflight* failure the cache still
   reports its previous `:success` and still holds the previous physical factor.
   An SDPX adapter that inspects `cache.status` after a failed call would
   therefore conclude the old factor is valid **for the new request**.

That is exactly the failure §4 legislates against. The mitigation is unchanged
and is now grounded in the provider's actual structure: SDPX must revoke the
**logical** lease on any failed `refactor_numeric!`, before inspecting any
provider status, because the provider is entitled to keep both its storage and
its previous success flag across a preflight rejection.

Recorded here as evidence, not as a request for BFLA to change: the preflight
behaviour is a reasonable strong-exception guarantee. The obligation is
SDPX-side.

---

## 9. The §4 hazard is provider-independent, not a BFLA quirk

§8 checked BFLA. This section checks MFLA — and the result strengthens §4 from a
provider-specific observation into a structural property of both providers.

**A near-miss worth recording.** The first pass grepped MFLA for `preflight` and
`:unprepared` and found **neither**, which would have been recorded as "MFLA does
not share BFLA's two-phase design". That was wrong: MFLA implements the same
structure under a different name (`invalidate!(cache)`), and the grep was too
literal. Provider parity must be established by reading the ordering of the
operations, not by matching vocabulary.

All four MFLA `factorize!` methods (`src/factor_caches.jl` :39, :138, :245, :488)
have this shape:

```
_check_config_frozen(cache, config)          # throws
n == size(A, 2) || throw(DimensionMismatch)  # throws  (square caches only)
_check_supported(MF)                         # throws
_check_prepared(cache, (n, n))               # throws
invalidate!(cache)                           # <-- commit phase starts HERE
copyto!(cache.factors, A)
status = _..._factorize_core!(...)           # numerical work
cache.status = status
```

So the comparison is:

| | BFLA `f95d3e6` | MFLA `50e6e0b` |
|---|---|---|
| Commit-phase marker | `status = FactorStatus(:unprepared, nothing)` | `invalidate!(cache)` |
| Throwing checks before it | yes | yes |
| Consequence of a **preflight** throw | previous factor **and** previous `:success` retained | previous factor **and** previous status retained |
| Consequence of a **commit-phase** failure | status non-success, never stale `:success` | `invalidate!` already ran, so no stale success |

Both providers therefore offer a **strong exception guarantee on preflight
rejection and no stale success on commit failure**. That is a defensible design,
and it is *identical in effect* across the two libraries.

**Consequence for §4.** The rule is unchanged but is now established as
provider-independent: after ANY failed `refactor_numeric!`, regardless of
provider and regardless of which phase rejected, SDPX must revoke the logical
lease before reading any provider status. An adapter cannot distinguish "the
call failed and the old factor is still valid" from "the call failed and the old
factor is still valid *for the old request*" by inspecting the provider alone.

This is exactly the packet's §3.6 point — "失败语义需要特别对齐" — and it turns
out the alignment requirement is not that the providers differ, but that **both
providers' guarantees are about physical retention, while SDPX's need is about
logical validity.**

## 10. RESOLVED — provider revisions as actually measured (added 2026-09-11)

§6 left the MFLA drift to "the first MFLA task to start", requiring that task to
state the revision it measured and either justify the newer revision with its
diff, or measure at the frozen one. That has now happened, and this section
settles the drift for both providers. It does **not** rewrite §8 or §9, whose
revision labels were correct when written; read them with this section applied.

Both checkouts have since moved again, because packet tasks were committed on top
of them. The revisions named in §8 (`BFLA f95d3e6`) and §9 (`MFLA 50e6e0b`) are
therefore historical.

### The chain, with the diffs that justify it

| Provider | ADR's label | Measured at | Diff from the label | Verdict |
|---|---|---|---|---|
| MFLA | `50e6e0b` (and `b38dea1` before it) | **`3ddf8ed`** (v0.4.0+4) | 4 files, **1690 insertions, 0 deletions**; `src/factorizations/` empty; entry point byte-identical | load-code equivalent to `50e6e0b` |
| BFLA | `f95d3e6` | **`9d9683c`** (v0.3.0+1) | 5 files, **1679 insertions, 0 deletions**; entry point byte-identical; none of the 5 in the include graph | load-code equivalent to `f95d3e6` |

    # MFLA
    git -C MultiFloatLinearAlgebra.jl diff --stat 50e6e0b 3ddf8ed
    git -C MultiFloatLinearAlgebra.jl diff 50e6e0b 3ddf8ed -- src/factorizations/
    git -C MultiFloatLinearAlgebra.jl diff 50e6e0b 3ddf8ed -- src/MultiFloatLinearAlgebra.jl
    # BFLA
    git -C BigFloatLinearAlgebra.jl diff --stat f95d3e6 9d9683c
    git -C BigFloatLinearAlgebra.jl diff f95d3e6 9d9683c -- src/BigFloatLinearAlgebra.jl

Zero deletions + an unchanged module entry point + no new file in the include graph
means `using <Provider>` executes identical code at either revision. This is the
justification §6(a) asks for.

For MFLA the remaining step back to the packet's frozen `b38dea1` is already
settled in `docs/rebuild/baseline.md` §1.1: three commits, a single hunk in
`_factor_ldlt_panel!`, BK pivot grammar unchanged, version not bumped — threading
only.

### The rule, restated because it is the part that keeps being violated

**Name the revision actually measured.** A result taken at `3ddf8ed` is a
`3ddf8ed` result and may not be reported as `50e6e0b`, `b38dea1` or "the frozen
revision". The load-code equivalence may be *stated alongside* a result; it may
not replace naming the revision. The same applies to BFLA and `9d9683c`.

Threading and performance claims must additionally name the revision, because for
MFLA that is precisely where the pre-existing drift lives.

### What each task actually measured

| Task | MFLA | BFLA |
|---|---|---|
| M01 | `50e6e0b` (stated in its report) | — |
| B01 | — | `f95d3e6` |
| S05 | `3ddf8ed` | `9d9683c` |
| A01b | `3ddf8ed` | `9d9683c` |
| Q01 | `3ddf8ed` | `9d9683c` |

No task reported a result under a revision it did not measure, and three
independently flagged the drift rather than papering over it. §6 is now closed.

### The equivalence EXPIRES

Every statement above holds only while the packet's new files stay **un-wired**.
The moment I01/I02 adds an `include` for `src/contracts/*.jl` (MFLA) or
`src/contracts/*.jl` + `src/mpfr_context.jl` (BFLA), the entry point changes, the
equivalence lapses, and every provider result must be re-attributed to the new
revision. I01/I02 must re-check this and I03 must not freeze a release on a stale
equivalence.

### One capability question this does NOT settle

`S05-F1`: MFLA's own `capabilities(MF)` still reports `multi_rhs::Bool` — the
exact generalization §3 of this ADR rejects. S05 refused to promote it and the
live leg conservatively reports `MultiRHSPerColumn`. Whether batching is genuinely
supported is therefore a **decision**, not a measurement, until M01–M03 answers
it. ADR-002 §3 is not satisfied for MFLA on this point.

## 11. The §10 equivalence has EXPIRED — I01 wired both providers (added 2026-09-11)

§10 closed the §6 SHA drift by proving that `using <Provider>` executed identical
code at the ADR's label and at the revision a task actually measured. It then said
the equivalence would lapse the moment I01/I02 added an `include` for the new
files, and required I01/I02 to re-check it and I03 not to freeze on a stale one.

**Both providers are now wired, so it has lapsed.** This section exists so no later
task cites §10 as if it still held.

| Provider | §10's equivalence | Wired at | Diff that ended it | Status |
|---|---|---|---|---|
| MFLA | `50e6e0b` (via `3ddf8ed`) | `2294ada` | 3 `include`s added after `factor_cache_requirements.jl:36` | **void** |
| BFLA | `f95d3e6` (via `9d9683c`) | `db06034` | `mpfr_context.jl` + 3 `include`s added | **void** |

### Why the inference no longer holds

§10's justification was: zero deletions, plus an unchanged module entry point, plus
no new file in the include graph, therefore the loaded code is identical. I01
changed the **entry point** of both packages. The other two premises still hold —
the wiring commits delete nothing from the packages' kernels — but the inference
needs all three, and the load graph is now strictly larger. `using
MultiFloatLinearAlgebra` and `using BigFloatLinearAlgebra` each execute top-level
code that did not execute before.

Note what did *not* change, because it bounds the blast radius: for BFLA the wiring
is additive at the name surface. Measured at `9d9683c` and at `db06034`, the
exported name set is byte-identical (115 names) and of the `names(M; all=true)`
entries, 272 real (non-gensym) names became 350 with **zero removals**. So the
equivalence is void as a *proof technique*, not because provider behaviour is
suspected to have changed. BFLA's own suite is 10864/10864 at both revisions and
B01's driver measures identical values in sandbox and wired modes.

### The rule, for the third time

**Name the revision actually measured.** Results taken at `3ddf8ed`/`9d9683c`
remain valid *as results at those revisions* and may not be re-attributed to
`2294ada`/`db06034`. Results taken at the wired revisions must say so. The I01
rollback path is recorded in `docs/rebuild/I01_integration_record.md` §3; reverting
the wiring does not revive this equivalence for any result already reported, it only
makes the old revision current again.

### What §11 does not settle

§10's open capability question (`S05-F1`: MFLA's `capabilities(MF)` still reporting
`multi_rhs::Bool`, the generalization §3 rejects) is untouched. It stays a decision,
not a measurement, until M01–M03 answers it, and ADR-002 §3 remains unsatisfied for
MFLA on that point.
