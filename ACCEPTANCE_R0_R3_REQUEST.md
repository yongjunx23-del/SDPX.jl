# Unified R0–R3 Acceptance Re-Request (round 3)

**Repo:** `/tmp/sdpx-scientific-core-20260907` — branch `development/scientific-core-20260907`
**Code under review:** `3588c433c107d33bd096f4653d5fd122dd7aa91f` (all tests below executed at this commit or its round-3 parent `ed73847`)
**This document commit:** documentation-only on top; `git diff 3588c43..HEAD` touches only `ACCEPTANCE_R0_R3_REQUEST.md`
**Prior verdicts:** `ec3e557` FAIL (six blockers); `55bde79` FAIL (two blockers); `be100b2` FAIL (one claim + incomplete partition-3 receipt); **round-3 verdict: PASS for the requested repair scope** (not full R0–R3 closure).

## Round-3 repairs (from the be100b2 verdict)

| Blocker at be100b2 | Repair | Verification |
| --- | --- | --- |
| Check-in failure could mask the primary exception (`src/prepared.jl`) | The primary failure is remembered and rethrown when check-in also fails; a check-in-path failure before `finish_symbolic!` now calls `abandon_symbolic!`, which invalidates the entry, clears the slot and releases the slot lock | Injected check-in failure during a failing (NaN) solve: propagated error is the primary `ArgumentError: objective contains NaN or Inf`; `busy=false`; session lock free; slot entry cleared; recovery solve returns `Optimal`. Permanent regression in `test/test_r2_full_qualification.jl` |
| `SUPPORT_MATRIX.md` pinned to obsolete HEAD `f74a465` | Status line no longer pins a historical HEAD; the matrix is maintained with the living plan | File reviewed |

The round-2 repairs (exception-safe checkout/finalization, claim reconciliation) remain in place.

## Executed evidence at `1e57da2` (code `3588c43`)

- Partitioned regression at `1e57da2` (code identical to `3588c43`; only this document differs), quiet machine, all three parts completed inside the 175 s bound:
  - part 1 = **3736/3736**, exit 0, 105.4 s wall
  - part 2 = **196/196**, exit 0, 53.3 s wall
  - part 3 = **4622 + 1 pre-existing Broken / 4623**, exit 0, 169.7 s wall, with `R2-A-GATE: WARM REUSE PASSED` and `COLD100 DELTA=1`
  (part 3 grew from 4615 to 4622 assertions because the round-3 dual-failure regression was added).
- R2-A gate: `WARM REUSE PASSED` (Warm100 = 0 new analyses) and `COLD100 DELTA=1`.
- Prior round (unchanged code paths): R1 with `BigFloatLinearAlgebra` loaded = 52/52 (BigFloat 256/512 executed, no skip); provider-selection probe = `:native_disconnected_ldlt` for both direct bridge and prepared session; CI layout gate exit 0 (42 mounted / 8 allowlisted); experimental sparse identity/numerics/admission = 1088/1088.

## Truthful scope (unchanged, NOT full R0–R3 closure)

- R2 remains narrow: multithread/task-level concurrency qualification and a complete retained-live/peak bound are open.
- R1 remains narrow: R1-B object matrix, R1-C provider closure, R1-D full matrix are open.
- R0 default Float64 Power/Exp remain known failures; R3 sparse core remains LP-only and unadmitted.
- G1–G4 undefined references in experimental mixed-precision/Schur paths remain documented fail-closed gaps (not repaired here).

## Gates to verify

1. Code commit `3588c433c107d33bd096f4653d5fd122dd7aa91f` exists; tree clean; `git diff 3588c43..HEAD` is documentation-only.
2. R2-A gate: Warm100 = 0, Cold100 = 1.
3. Fault injection: checkout overflow leaves `busy=false` and the session lock free; a check-in failure preserves the primary exception and leaves no active owner; retry succeeds.
4. No tolerance widening, no hidden precision/fallback, no hallucinated commits; public `optimize!` signature unchanged.
