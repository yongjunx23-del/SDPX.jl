# Unified R0–R3 Scientific-Core Acceptance Request (Astra Oracle Review)

**Repo:** `/tmp/sdpx-scientific-core-20260907` — branch `development/scientific-core-20260907`
**Submitted HEAD:** `5eeda2d861c0f0f3d24260ddc725a7fdc900e2b5`
**Baseline:** `d4438c2` (pre-R2-wiring)

## Review scope

Independent review of the genuine committed HEAD for the R0–R3 closure of the
scientific-core roadmap (`docs/design/SCIENTIFIC_CORE_ROADMAP.md`). All claims
below are backed by real commits and executed test runs; the oracle is asked to
verify the exact git worktree, run the shipped tests, and flag any missing
commits/files.

## Commit chain since baseline

| Commit | Content |
| --- | --- |
| `a274011` | R2 lifecycle & session-local symbolic reuse (Cold100=1 / Warm100=0 gate) |
| `7920982` | R1 full qualification (AccuracyContract, owned-result isolation, truthful certificates) |
| `a4715a4` | Cleanup: remove 2 orphaned scripts, add factor-pair mirror drift guard, simplify roadmap |
| `3bfa948` | Delegated qualification fixes: wire R1 into runtests.jl; BFLA import in BigFloat owned-results test |
| `5eeda2d` | Merge of `3bfa948` (this HEAD) |

## What was verified (executed by worker on `3bfa948`, clean tree)

- `test/session_symbolic_lease.jl`: **881/881**
- `test/accuracy_contract.jl`: **135/135**
- `test/test_r1_full_qualification.jl`: PASS (R1-B BigFloat = pre-existing expected-Broken, exit 0)
- `test/test_r2_full_qualification.jl`: PASS (R2-B/C/D)
- `test/experimental_sparse_core.jl` / `_identity` / `_numerics` / `_research_support`: PASS
- `test/test_r2a_symbolic_numeric_separation.jl`: **PASS — `R2-A-GATE: WARM REUSE PASSED`, `COLD100 DELTA=1`**
- `test/test_r2_lifecycle_qualification.jl`: PASS (9 + 1411 + 55)
- `test/test_prepared_bigfloat_owned_results.jl`: PASS 20/20 standalone after BFLA import fix
- Partitioned regression (`run_partitioned_regression.jl`): **part 1 = 3736/3736, part 2 = 196/196, part 3 = 4480 + 1 pre-existing Broken / 4481** — all exit 0, all ≤180 s, provenance asserts (ROOT/HEAD/clean) held.

## Known environment-blocked items (NOT code defects, fail-closed)

1. `test/experimental_sparse_core_ordering.jl` — requires a BFLA pin with natural-ordering support; locked pin `f95d3e6` defines none, so it fails closed (companion `test/qdldl_legacy_ordering.jl` passes 9/9 with fail-closed expectations for this pin). Environment/pin decision, not numeric.
2. `run_private_sparse_pattern.jl` and `run_constraint_contractions.jl` — provenance-pinned drivers blocked because the shared BFLA/MFLA checkouts (`f95d3e6`/`50e6e0b`) don't match the pins the drivers assert (`5fce2e68`/`aaa71f33`/`5399c0cc`). Infrastructure, outside the worker's scope.

## Gates to verify

1. `git -C /tmp/sdpx-scientific-core-20260907 rev-parse HEAD` == `5eeda2d861c0f0f3d24260ddc725a7fdc900e2b5`; worktree clean (only untracked `ACCEPTANCE_R0_R3_REQUEST.md`).
2. R2-A gate: Warm100 = 0 new symbolic analyses; Cold100 = 1.
3. No tolerance widening, no hidden precision/fallback, no hallucinated commits.
4. R1/R2/R3 test files wired into `test/runtests.jl` and passing in the partitioned regression.
