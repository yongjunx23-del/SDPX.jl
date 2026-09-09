# CSDR type-stability allocation reduction

Date: 2026-09-06. Implemented directly by parent, no subagents. Uncommitted on SDPX main based at db42fd281d17004808c93b88cc8c8b01e553ea1e plus accepted public-gap restoration. MFLA unchanged.

## Accepted result: allocation reduction, not speed credit

Two type-stability changes reduce cumulative allocated bytes on frozen CSDR α3 from approximately 2.32–2.33 GB per warmed solve to 0.89–0.91 GB (about 61%). These figures are Julia cumulative allocation counts, not live memory or a leak estimate. The unchanged full guard passes after main integration: optimal, valid certificate, 101 iterations, objective -31.672155970636577, exact terminal-certificate digest 25ef57d499cb9fdaa45600bd11c7e6948df23ab063434eff126765545e529ca7, ALL_REGRESSION_GATES_OK.

No stable ≥2% runtime improvement is claimed. Warm wall times vary materially during this desktop session. The candidate is retained for its consistent large allocation reduction and unchanged acceptance behavior, not as a Stage-C speed win. Peak RSS and JIT latency are not claimed improved.

## Code changes

- `src/kkt/specializations/fixed_trace_q3.jl`: carry `typeof(plan)` as an additional workspace parameter and declare `plan::P`, instead of an `Any` field. Existing partial-type dispatch remains valid; the sole explicit constructor supplies the added parameter. No operations, iteration policy, factorization order, or mutable buffer ownership change.
- `src/certificates/certificates.jl`: dispatch once to a noinline `_in_canonical_blocks` helper using the actual layout storage type. The existing serial membership loop, block order, tolerance, view ranges, finite gate, and early exits are unchanged. This avoids dynamic dispatch/boxing per block caused by the intentionally type-erased layout field in `CanonicalConicProgram`.
- `test/certificate_layout_storage.jl`: vector/tuple layout paths in Float64 and BigFloat; interior/boundary/outside, primal/dual, NaN, invalid tolerance, and dimension checks. Included in `test/runtests.jl`.

No problem-name branches, numerical parameter tuning, tolerance changes, guard edits, or BigFloat ownership changes.

## Measurement sequence and all candidate stages

All processes used Julia 1.12.6, four Julia threads, --startup-file=no, --gcthreads=1, explicit unset SDPX_MEMORY_RSS_OVERRIDE_MB/SDPX_CSDR_PROFILE, pinned environments, one warmup followed by three timed solves. The same harness checks certification and exact terminal digest on every solve. Original input hash is asserted by that harness.

Retained artifacts: `/tmp/csdr-typed-plan/`. Baseline source is `/tmp/csdr-product-reuse-review/baseline-src` including accepted gap normalization. Candidate source was copied from it, then changed only as recorded below.

| Stage | Batch medians (seconds) | Median allocated bytes per batch |
|---|---|---|
| Baseline, first two batches | 17.931620, 17.911731 | 2,317,466,896; 2,327,362,832 |
| Typed plan only | 17.522818, 17.702847 | 1,105,800,976; 1,107,554,064 |
| Typed plan + certificate barrier | 19.072677, 19.433581 | 908,529,424; 892,112,656 |
| Contemporaneous baseline between barrier batches | 19.500760 | 2,332,966,160 |

Typed-plan-only aggregate time improvement was 1.83%, not enough for speed credit. Barrier batches bracket the later baseline: one is ~2.2% faster, the other ~0.34%, so stable speed credit is again unsupported. Do not compare the later 19-second batches against early 17-second baselines as if environment load were controlled identically.

Earlier local type assertions are retained separately in `/tmp/csdr-metric-type-experiment/`; they are superseded, not integrated. The typed plan addresses the underlying erased field rather than asserting types separately at each caller.

## Validation

- Integrated main: `test/certificate_layout_storage.jl` 48/48 and gap normalization 34/34, exit 0.
- Isolated candidate: fixed-trace Newton reference 28/28; equality Schur 22/22; ordinary conic E2E subset 18/18 plus 4/4 spot checks, exit 0.
- Unchanged full guard: candidate and integrated main both exit 0 with ALL_REGRESSION_GATES_OK. These checks cover the terminal digest, not a recorded full iterate history. Phase accounting/cache counters are printed by the guard, not asserted; observed cache counters are zero.
- `git diff --check` passes.
- Broader `benchmark/general/test_small.jl` fails before solves on four outdated inventory assertions (22 vs expected17 small cases, expanded family set, 7 vs expected5 medium/large). Baseline reproduces all four failures; parser checks pass. This is recorded, not repaired or called an all-suite pass.
- Full test suite and broader BigFloat solver matrix were not run; BigFloat layout semantics were tested, but no general BigFloat allocation benefit is claimed.

## Receipts

Under `/tmp/csdr-typed-plan/`:
- `baseline{1,2,3}.log`, `candidate{1,2}.log`, `barrier{1,2}.log` and `.exit`: full raw samples for every batch above.
- `measure.jl`: identical measurement script, per-sample printing, frozen checksum/certificate checks.
- `focused-tests.log`, `layout-tests.log`, `ordinary-cert.log`: candidate focused results.
- `general-small.log`, `general-small-baseline.log`: reproduced unrelated failures.
- `final-guard.log`, `integrated-guard.log` and `.exit`: full unchanged guards.
- `integrated-tests.log`, `integrated-identities.txt`: main checks, exact source/input/env/script hashes and dirty status.

Integrated guard command:
`env -u SDPX_MEMORY_RSS_OVERRIDE_MB JULIA_NUM_THREADS=4 julia --startup-file=no --gcthreads=1 --project=/tmp/csdr-integrated-profile/guard-env-main /Users/xuyongjun/Desktop/project/SDPX/SDPX.jl/benchmark/lifecycle/regression_guard.jl`
