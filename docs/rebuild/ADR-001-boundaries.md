# ADR-001: Responsibility boundaries and ownership

**Status: ACCEPTED (A00, 2026-09-11).** This ADR is binding on I01/I02/I03 and on
every task card. It freezes the ownership half of the packet's architecture
proposal (`ENGINEERING_REBUILD.md` §3.1–3.7) and answers A00's second acceptance
item: "ADR说明每个物理数组、factor和logical epoch的唯一拥有者".

Baseline: see `docs/rebuild/baseline.md` for the frozen SHAs.

---

## 1. The rule this ADR exists to enforce

> 一份物理 factor，一个逻辑 Newton 身份。
> (`ENGINEERING_REBUILD.md` §3.6)

The failure this prevents: SDPX's cache, its `HotRouteCache`, a provider's opaque
handle, and the provider's own cache each independently believing they hold a
valid factor, so a *fresh* Newton request is answered with a *stale* factor. The
packet names the concrete case (§3.6): a BFLA dense cache whose input pre-check
fails retains the old physical factor — a legitimate strong exception guarantee
for the provider, but SDPX must not then solve new RHS with it.

## 2. Ownership table

Every physical array, factor, and epoch has exactly one owner. "Owner" means the
sole authority to mutate, invalidate, or publish freshness.

| Artifact | Owner | Not the owner | Notes |
|---|---|---|---|
| Raw user `Model` / MOI input snapshot | Frontend (compiled problem) | solver session | kept for final verification |
| Canonical `A`, `b`, `c` and cone layout | Compiled problem | KKT session | read-only to the solve |
| `TransformStack` (elimination, RSOC, scaling, fixed vars) | Compiled problem | public API | owns forward *and* inverse maps plus objective offset; the public layer must not patch signs |
| Iterate `x, y, s, τ, κ` | Solver session | KKT session, cones | exactly one current point |
| Residual arrays + residual epoch | Solver session | KKT session | see §4 |
| Cone pair-wise scaling state (`ProductConeRuntime`) | Solver session (cone runtime) | KKT session | pair-dependent, rebuilt per accepted point |
| Direction buffers (affine / combined / trial) | Solver session | KKT session | exclusively held |
| `matrix_epoch` (which Newton operator) | Solver session | provider | incremented at the named transition boundary |
| `K_original` (unregularized math operator, may be an action) | KKT session | provider | |
| `K_factor_input` (after authorized scaling/ordering/signed shift) | KKT session | provider | the *only* legitimate input to a factor |
| Physical factor, pivots, block grammar, factor scratch | **Provider (MFLA/BFLA/stdlib)** | SDPX | SDPX must not interpret its storage |
| `provider_generation` (numeric factor identity) | **Provider** | SDPX | SDPX records it, does not mint it |
| Logical lease binding `matrix_epoch → provider_generation` | Solver session | provider | thin wrapper only; see §3 |
| Symbolic structure + its lease | `SessionSymbolicLease` holder | factor caches | see §5 |
| Telemetry counters and timings | Solver session | any correctness gate | must never decide a numeric outcome |

## 3. What the lease wrapper MAY and MAY NOT do

Allowed: record `matrix_epoch → provider_generation`, and revoke the logical
lease.

Forbidden (packet §3.6):
- copying a large factor a second time;
- re-factorizing "to be safe";
- declaring freshness from unverified metadata rather than the provider's own
  generation.

## 4. Residual and scaling epochs (carried over, not reinvented)

The rebased work already established, and this ADR adopts rather than re-derives:

- `HSDState.point_epoch` — bumped by every write to `x/y/s/τ/κ`.
- `HSDState.residual_epoch` — the `point_epoch` the cached residual came from.
- `HSDState.residual_canonical` — **which kernel** produced it.

The third field is load-bearing and must not be dropped during the S02/S03
extraction. `_cert_residual!` writes `rP`/`rD` with a **different accumulation
association** than `hsd_residual!` (it pre-seeds `rD` with `c[j]*tau` instead of
accumulating `A'y` first). The values agree mathematically but not bitwise, and
the direction build consumes them. Measured disagreement: 0.0 on SOC k=3/8/32,
but 2.2e-14 at LP n=200 and 2.3e-13 at LP n=500. A "point has not moved" flag
alone is therefore unsound. The invariant is asserted bitwise by
`test/accepted_point_reuse.jl`.

Packet §3.2 requires: "已经在上一轮实现的 scaling/residual freshness 需要随此
state 迁移，而不是重新发明." This ADR makes that explicit — S02/S03 **move** this
state, they do not redesign it.

## 5. Session symbolic lease: exists, unwired

`src/factor_cache/session_symbolic_lease.jl` exists and its header states
"Not wired into Prepared execution yet". `SessionSymbolicKey` currently admits
**only** `arithmetic === :float64 && precision_bits === 53 && provider === :cholmod
&& route === :bordered && core_owner === :generic`, and refuses anything else.

Decision: this narrow admission is **correct and must be preserved** during I01.
Widening it is a numeric-capability change and belongs to I02, per the packet's
rule that I01 switches responsibilities only.

## 6. Concurrency

Packet §3.7: the thread budget is a **session resource**, not a synonym for
`Threads.nthreads()`. Three distinct consumers — coarse cone batch, LA
factor/panel, BLAS — must not have their limits multiplied together. Two
concurrent solves must own independent mutable workspace.

The plan's 16/64-thread tiers are **not measurable on this host**
(`Sys.CPU_THREADS == 4`); any task requiring them must record them as
`unsupported`, which per `AGENTS.md` may satisfy an explicit-refusal negative
test but **cannot** satisfy a required release capability.

## 7. Consequences

- S02/S03/S05 extractions must relocate state, not redesign it. A redesign
  disguised as an extraction is a violation of this ADR.
- Any change to the ownership table requires a superseding ADR, not an edit here.
- Deleting or renaming a source file remains I01/I03's authority alone.
