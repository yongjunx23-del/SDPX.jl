# Persistent pool: local finite-batch gate

2026-09-09. **Local gate PASS; opt-in benchmark tool only.** Not cluster,
long-run retention, independent certificate or α3 latency qualification.

## Frozen execution

- Source: clean `464750b5076a2357c721a869ba8f39be6e3fd88b`,
  `src/ext` identical to development `5eb7d2b`.
- Harness SHA256: `cf7808057895078a16b94631ea0339b5df7792408d1089f7f51ddc55bc840eab`.
- Run ID: `0b8e114a-47fb-4c14-9ea1-cc422d4ba9ca`, protocol 2.
- Mac: 10 cores, 16 GiB. Configured concurrency 2, not CPU reservation/affinity.
  Each Julia 1.12.6 process: 1 Julia thread, 1 GC thread, BLAS/OMP/MKL=1,
  startup disabled, private source-matched environment.
- Same 8 planted-LP items in each arm; seed `0x004c5003 + i`, rebuilt per item.
  Persistent warmup excluded from item count but included in batch wall.
- Outer 900 s process-group watchdog: exit 0, no timeout, 388.8 s total.

## Measured batches

| Order | Arm | Batch wall (s) | Reported-valid solves/h | Median solver time (s) |
|---|---|---:|---:|---:|
| 1 | fresh | 143.7 | 200.42 | 17.35 |
| 2 | persistent | 47.9 | 601.28 | 2.45 |
| 3 | persistent | 46.8 | 615.13 | 2.41 |
| 4 | fresh | 144.9 | 198.78 | 17.40 |

Arm throughput medians: **199.6003 → 608.2017 solves/h**, ratio **3.047098**.
Fresh solver timing includes first-call JIT; this is startup amortization,
not a numerical-kernel speedup. The comparison includes startup/collection.

All 32 item receipts report optimal status, valid certificate and satisfied
analytic-objective expectation. A separate read-only verification checked
identities, references, fingerprints, 4 warmups and 4 worker summaries;
claims covered every item exactly once. No failed/missing items or children.
This validates recorded facts and the analytic objective, not original x/y/s
certificate equations independently.

Worker peak/pre-exit samples: **1.893–1.930 GiB**, all available and below the
configured 3 GiB limit. Workers exit at batch end. This finite sample does not
prove bounded retained memory for indefinitely running workers.

## Test and artifact provenance

Solver-free contract tests: 130 passing assertions across separate groups.
Lifecycle/receipt/worker/memory/exclusivity groups were run before final
launcher-mock correction; race14/gate23/CLI8 were rerun on clean `5aab5be`.
Production validation was not weakened. First group outputs survive in the
worker transcript only; the latter three have persisted logs under
`/tmp/sdpx-pool-retry-20260909T222728/`.

Measurements and read-only verification:
`/tmp/sdpx-pool-qualified-20260909T223540/{compare.log,verify.log,report.md,run/}`.
Full comparison summary is `run/comparison_summary.toml`; per-item provenance
and environment hashes are retained in the batch artifacts.

The earlier 2.85× untracked-harness experiment remains historical, not promoted
retroactively. Current evidence supports using the repaired pool explicitly
for this finite local workload; repeat qualification before cluster campaign
use or claims about other cones/precisions/long-lived services.
