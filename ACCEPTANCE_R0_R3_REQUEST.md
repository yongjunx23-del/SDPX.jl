# Unified R0–R3 Acceptance Re-Request (post-blocker repair)

**Repo:** `/tmp/sdpx-scientific-core-20260907` — branch `development/scientific-core-20260907`
**Code under review:** `6994c89227f9dfe626bbedd9f39a05027f208a33` (every test below was executed at this commit)
**Request document:** this commit, documentation-only; `git diff 6994c89..HEAD` touches only `ACCEPTANCE_R0_R3_REQUEST.md`
**Previous reviewed HEAD:** `ec3e557` (oracle verdict: FAIL, six blockers)
**Baseline:** `d4438c2`

## What this re-request contains

All six blockers from the ec3e557 review are repaired and re-verified:

| # | Blocker (ec3e557) | Repair | Verification |
| --- | --- | --- | --- |
| 1 | R1 BigFloat branch skipped; `ResultFieldNotRetained` when BFLA imported | Both BigFloat `optimize!` calls now retain requested outputs; model declares `precision_bits` per iteration | Executed with `using BigFloatLinearAlgebra`: 256/512 branch runs and passes (exit 0) |
| 2 | Lease path skipped ordinary provider selection (`symmetric_core.jl:2148-2154`) | `DisconnectedLDLTCache` selection now runs first and unconditionally; a lease only authorizes reuse of a compatible CHOLMOD entry | R2-A gate still `WARM REUSE PASSED` / `COLD100 DELTA=1` |
| 3 | Validation ran before checkout; failed update preserved the retained entry (`prepared.jl`) | Checkout now precedes validation; failed update revokes via `finish_symbolic!(certified_optimal=false)` and advances the attempt | R2-B invalidation tests pass; R2-A gate unchanged |
| 4 | `optimize!` exposed `execution_context` publicly | Public `optimize!` signature restored; bridge calls internal `_optimize_impl` seam | All suites pass; no public-API use of the context |
| 5 | Docs/tests overclaimed full R1/R2 closure | Roadmap + `R2_FULL_QUALIFICATION.md` now separate narrow implementation from open matrix/resource gates; R2-C/D labels state sequential/allocation-variation scope; R1 mutation test uses a real pre-mutation snapshot; infeasibility test validates the original-coordinate ray certificate | `test/test_r1_full_qualification.jl`, `test/test_r2_full_qualification.jl` |
| 6 | CI `quick-checks.yml` layout gate contradicted the real `test/` organization | Gate now requires every `test/*.jl` to be mounted in `runtests.jl` or on an explicit manual-only allowlist | Gate evaluated locally: 42 mounted, 8 allowlisted |

## Executed evidence at this HEAD (parent-run, bounded single-thread processes)

- Partitioned regression `validation/scientific_core/run_partitioned_regression.jl`: **parts 1, 2, 3 all exit 0** (provenance asserts ROOT/HEAD/clean held).
- R5-A `test/test_r5a_precision_controller.jl`: **58/58 pass**; R5-B `test/test_r5b_thread_budget.jl`: **64/64 pass** (now wired into `runtests.jl`).
- R2-A gate inside the partitioned suite: Warm100 = 0 new analyses, Cold100 = 1.
- R1 `test/test_r1_full_qualification.jl` passes both without BFLA (BigFloat skipped) and with `using BigFloatLinearAlgebra` (BigFloat 256/512 executed).

## Truthful scope (NOT a claim of full R0–R3 closure)

- R2 full closure still requires multithread/task-level concurrency qualification and a complete retained-live-object/peak bound; R2-C/D remain narrow.
- R1 full closure still requires the R1-B BigFloat object matrix, R1-C provider closure (Julia 1.10/1.11/1.12, LinearSolve/QDLDL), and R1-D full matrix.
- R0 default Float64 Power/Exp remain known failures; R3 sparse core remains LP-only and UNADMITTED.
- R4/R5/R6 quick items landed as narrow qualification (R5-A/B tests, R4-A standalone pairing matrix, R6-A support matrix update, R6-D reproducibility scaffolding).
- Known source gaps G1–G4 (undefined references in the experimental mixed-precision/Schur paths) are documented fail-closed findings, not repaired here.

## Gates to verify

1. Code commit `6994c89227f9dfe626bbedd9f39a05027f208a33` exists; `git status --porcelain` empty; `git diff 6994c89..HEAD` is documentation-only.
2. R2-A gate: Warm100 = 0, Cold100 = 1.
3. R1 with BFLA imported: BigFloat 256/512 executes and passes.
4. No tolerance widening, no hidden precision/fallback, no hallucinated commits, public `optimize!` signature unchanged.
