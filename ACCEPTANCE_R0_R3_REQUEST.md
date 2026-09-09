# Unified R0–R3 Acceptance Re-Request (round 2)

**Repo:** `/tmp/sdpx-scientific-core-20260907` — branch `development/scientific-core-20260907`
**Code under review:** `175e31dcfb567441eaf8bb42e36316a7854567a8` (all tests below executed at this commit)
**This document commit:** documentation-only on top; `git diff 175e31d..HEAD` touches only `ACCEPTANCE_R0_R3_REQUEST.md`
**Prior verdicts:** `ec3e557` FAIL (six blockers); `55bde79` FAIL (two blockers)

## Round-2 blocker repairs

| Blocker at 55bde79 | Repair | Verification |
| --- | --- | --- |
| Checkout/finalization exceptions strand the prepared session (`src/prepared.jl:911-922,964-973`) | Outer `try/finally` now covers `checkout_symbolic!`; lease check-in is nested inside it, so a `finish_symbolic!` throw cannot prevent `busy=false`/unlock or mask the primary exception | Fault injection `slot.attempt = typemax(UInt64)`: `OverflowError`, `busy=false`, session lock free; after restoring the counter the retry returns `Optimal` |
| Remaining claim contradictions (`SUPPORT_MATRIX.md:18-19,41,61`; `test_r2_full_qualification.jl:1-5`; R1 ray comment) | Support matrix no longer marks R1/R2 "done" and no longer claims symbolic reuse unimplemented; R2 test header states narrow scope; R1 infeasibility comment states it checks solver-reported certificate facts, not an independent ray recomputation | Files reviewed; claims now match the narrow evidence |

## Executed evidence at `175e31d`

- Partitioned regression `validation/scientific_core/run_partitioned_regression.jl`: **part 1 = 3736/3736, part 2 = 196/196, part 3 = 4615 + 1 pre-existing Broken / 4616** — all exit 0. Part 3 took 2m31s under concurrent load (1 thread, bounded launcher); no test failed.
- R2-A gate: `WARM REUSE PASSED` (Warm100 = 0 new analyses) and `COLD100 DELTA=1`.
- Prior round (unchanged code paths): R1 with `BigFloatLinearAlgebra` loaded = 52/52 (BigFloat 256/512 executed, no skip); provider-selection probe = `:native_disconnected_ldlt` for both direct bridge and prepared session; CI layout gate exit 0 (42 mounted / 8 allowlisted); experimental sparse identity/numerics/admission = 1088/1088.

## Truthful scope (unchanged, NOT full R0–R3 closure)

- R2 remains narrow: multithread/task-level concurrency qualification and a complete retained-live/peak bound are open.
- R1 remains narrow: R1-B object matrix, R1-C provider closure, R1-D full matrix are open.
- R0 default Float64 Power/Exp remain known failures; R3 sparse core remains LP-only and unadmitted.
- G1–G4 undefined references in experimental mixed-precision/Schur paths remain documented fail-closed gaps (not repaired here).

## Gates to verify

1. Code commit `175e31dcfb567441eaf8bb42e36316a7854567a8` exists; tree clean; `git diff 175e31d..HEAD` is documentation-only.
2. R2-A gate: Warm100 = 0, Cold100 = 1.
3. Fault injection: checkout overflow leaves `busy=false` and the session lock free; retry succeeds.
4. No tolerance widening, no hidden precision/fallback, no hallucinated commits; public `optimize!` signature unchanged.
