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

## 7. Consequences

- A capability claim without an environment + revision + test is not accepted.
- `factor_summary` being O(1) is a hard requirement: a summary that recomputes
  inertia is a numeric operation and would need its own gate.
- Provider-gated tests must **skip with a reason** when the provider env is
  absent, never pass silently. `test/sparse_qdldl_provider.jl` is the reference
  pattern: a skip is visible in the summary; a missing test is not.
