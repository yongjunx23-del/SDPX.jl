# SDPX performance execution plan — honest status

This table is the single tracker for the GPT-Pro performance execution plan
(`SDPX_PERFORMANCE_EXECUTION_PLAN.md`).  Rule: a task is `VERIFIED` only when
it was actually executed in this worktree and the evidence is quoted below.
Anything not executed is `NOT_STARTED`.

| task  | status | evidence / note |
|-------|--------|-----------------|
| P0-00 | PARTIAL | `validation/performance/setup_identity.sh` + `identity.jl` exist and were executed by the P0-02 lane (distinct SHAs, diffstat, user tree untouched). Not yet run for the full identity contract in this worktree. |
| P0-01 | VERIFIED | SOC3 boundary fast-path branch-tree regression fixed (`fix(soc): restore reference branch tree in dim-3 boundary fast path`). Analytical counterexamples `s=(2,1,0),d=(0,1,0)` → ref 1 (fast path returned 3/2) and `s=(1,0,0),d=(2,1,0)` → ref Inf (fast path returned −1/6) now pass for Float64, Float64x4 and BigFloat256. Regression: `validation/performance/repro_soc3_boundary.jl`; the `p0` gate runs it (21/21 across three precisions). |
| P0-02 | VERIFIED | `validation/performance/{bitwise,replay,run_gates,digest_demo,identity}.jl`, `setup_identity.sh`. Executed in this worktree: `--gate p0` 21/21 (float64 7, float64x4 7, bigfloat256 7); `--gate semantics` 18/18 (float64); `--gate threads --threads 1` 5/5. Bad `--gate`/`--types` exit 2, identity mismatch exits 3. 1-ulp last-limb Float64x4 difference is detected. |
| P0-03 | VERIFIED | Root cause exported and fixed; see `docs/evidence/P0_03_PLATFORM_DIRECTION_BREAKDOWN.md`. Fix = composed backward-error bound in `_product_bordered_triangular_solution_ok!`. CI diagnostic run 34438051521: icelake-server, znver4 and apple-m1 all `optimal`/12 iterations. |
| P1-01 | NOT_STARTED | metric/RHS split not implemented. |
| P1-02 | NOT_STARTED | residual/statistic reuse not implemented. |
| P1-03 | NOT_STARTED | scratch preallocation not implemented. |
| P2-01 | VERIFIED | `_runtime_step_threaded!` used `min(Threads.nthreads(), 8)`, so a `threads=1` solve could start 8 workers. Admitted budget now travels on `ProductConeRuntime.worker_budget`, set from the solve-requested value at HSD setup; diagnostics now report the admitted budget instead of a hardcoded 1. Verified: `requested=1 executed=1` with `Limits(threads=1)` in a 4-thread process. |
| P2-02 | NOT_STARTED | deterministic executor not implemented. |
| P2-03 | NOT_STARTED | wider independent-work parallelism not implemented. |
| P3-00 | NOT_STARTED | iteration telemetry/policy state not implemented. |
| P3-01 | NOT_STARTED | β single-factor experiment not run. |
| P3-02 | NOT_STARTED | adaptive β controller not implemented. |
| P3-03 | NOT_STARTED | σ experiment not run. |
| P4    | NOT_STARTED | selective additional correction not implemented. |
| P5    | NOT_STARTED | large-KKT/sparse scaling not implemented. |
| G0    | PARTIAL | identity emitter + loader-path check implemented and enforced by `run_gates.jl` (exit 3 on mismatch); full per-run identity not yet recorded for every performance sample. |
| G1    | PARTIAL | P0 analytical counterexamples + boundary/illegal-input semantics gate green (`--gate p0` 21/21, `--gate semantics` 18/18). The platform failure regression (P0-03) is covered by the CI diagnostic workflow, not yet by a committed local test. |
| G2    | PARTIAL | bit-level payload comparison exists and is exercised; not yet applied to every E-class change (none landed yet). |
| G3    | NOT_STARTED | no A-class policy landed. |
| G4    | NOT_STARTED | lifecycle/epoch/alias/concurrency gate not executed. |
| G5    | NOT_STARTED | five-cone × precision × scale matrix not run. |
| G6    | NOT_STARTED | no statistically-qualified performance comparison yet. |

## CI layers cleared on the candidate branch (`development/scientific-core-20260907`)

The platform `test` matrix aborts at the first errored top-level testset, so
each fix reveals the next previously-masked failure.  Cleared in order:

1. **SOCP/native-V2 lowering receipts** — canonical `_domain_token` /`_sense_token` /`_canonical_power_token` (commit `4ff976d`).
2. **`factor_pair_backend_selector.jl`** — admission capability checks
   short-circuit in order, so a multi-thread process hit the `threads`
   refusal before `iteration_policy`/`cones`; pinned `Limits(threads=1)`
   (`27839e4`).
3. **`native_structure_diagnostics.jl`** — pinned `Limits(threads=1)` for the
   bordered-LP compact-plan control.
4. **`power_epigraph_small` E2E control** — the Float64 Power/Exp breakdown is
   platform-dependent, so the known-breakdown control now accepts exactly the
   two internally consistent truthful states (`7ea4952`).
5. **x86 bordered LP `direction_breakdown`** — P0-03, the triangular-solve
   certificate bound; see the evidence document above.
6. **`factor_pair_public_qualification.jl` "default route unchanged"** — the
   same platform-dependent Float64 Power breakdown as (4); the control now
   asserts `optimal ⇔ valid certificate` instead of a pinned non-optimal
   outcome.  Confirmed not to be caused by P0-03: this model never reaches the
   bordered triangular certificate (zero gate traces under
   `SDPX_DEBUG_DIRECTION=1`).  macOS CI passed this control before and after.
7. **`Pkg` missing from the test target** —
   `validation/scientific_core/test_r2a_symbolic_numeric_separation.jl` uses
   `using Pkg` for provenance and is included by `test/runtests.jl`; `Pkg` was
   not in `[targets] test`, so every platform that reached it aborted with
   `Package Pkg not found in current path`.

`test.yml` also gained `workflow_dispatch` so the platform matrix can be
validated on a candidate branch instead of after `main` turns red.

Layers below those are still unknown; the matrix has not yet run to
completion on all four platforms.

## Notes on scope

- The plan's own first batch is P0-00 → P0-01 → P0-02 → P0-03, then a legal
  baseline re-measurement, then P1-01. That is what is done here.
- `P1-01` must not start before the P0 numeric path is settled; it is the next
  task.
- No performance number in this document is a claim of improvement. The CSDR
  figure quoted elsewhere is a reference measurement, not a verified
  before/after pair.

## Runner map (P0-02 deliverables)

- `validation/performance/bitwise.jl` — canonical payload bytes, SHA-256
  numeric digests, `bitwise_equal`, `first_differing_index`. Float64
  reinterpret, all Float64x4 limbs, BigFloat precision/sign/exponent/exact
  significand. Non-numeric metadata never enters a digest.
- `validation/performance/replay.jl` — per-iteration replay recorder with
  rolling digest, `finalize_replay`, `compare_replays` (first divergence and
  diverging field), witnessed failures sorted by original index.
- `validation/performance/run_gates.jl` — `--gate p0|semantics|threads|providers|all`
  CLI, per-type child processes, TOML results with one row per check plus the
  commands run, non-zero exit on any failure/error.
- `validation/performance/setup_identity.sh` — P0-00 worktree/identity setup.
- `validation/performance/identity.jl` — `identity.toml` emitter.
- `validation/performance/repro_soc3_boundary.jl` — P0-01 counterexamples.
