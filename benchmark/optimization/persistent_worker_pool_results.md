# Persistent warmed worker pool vs fresh-process launcher: measurement verdict

Date: 2026-09-09. Worktree branch `perf/persistent-pool-*`. Driver:
`benchmark/optimization/persistent_worker_pool.jl` (execution management
only; no solver numerics changed).

## Workload (recorded exactly)

Generic LP target `lp_random_large` (`GenericConicBenchmark`, kind=:planted,
m=400, n=1200). Item queue of N=8 independent items built by varying ONLY
the planted seed:

    seed_i = 0x004c5003 + UInt32(i),  i in 0..7

Same model structure per item (n=1200 box LP); each item carries its own
independently recomputed known objective (`_planted_lp_objective`) and the
standard tolerance 2e-5. Every item is REBUILT per solve (fresh `Model`;
rebuild is milliseconds vs multi-second solves) so the comparison isolates
process startup/JIT amortization, not session reuse. Solver path is
identical in both arms: `run_one(spec, Float64; threads=1)` with
`certification=true`. Each arm reserves the same W=2 cores (`--threads=1`,
BLAS/OMP/MKL=1 per process).

Raw artifacts for this run live outside the worktree (`/tmp/pool_compare`,
`rep{1,2}_{fresh,persistent}/results/*.toml` + `batch_summary.toml` +
`comparison_summary.toml`); rerun with `--mode=compare --items=8
--workers=2 --outdir=<OUTSIDE_WORKTREE> --reps=2`.

## Measured comparison (interleaved fresh, persistent, fresh, persistent)

| rep | mode | certified | wall incl. startup+collect | certified solves/h | per-item median solve | per-worker peak RSS | retained RSS after batch | failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fresh (1 proc/item, 2 concurrent) | 8/8 | 167.4 s | 172.1 | 21.05 s | 1.93 GiB | 0 (processes exit) | 0 |
| 1 | persistent (2 warmed workers) | 8/8 | 57.1 s | 504.3 | 3.78 s | 2.02 GiB max | 2.02 GiB max | 0 |
| 2 | fresh | 8/8 | 161.3 s | 178.5 | 20.12 s | 1.91 GiB | 0 | 0 |
| 2 | persistent | 8/8 | 58.1 s | 495.5 | 4.18 s | 1.96 GiB max | 1.96 GiB max | 0 |

Medians: fresh 175.3 solves/h, persistent 499.9 solves/h.
**Gate ratio (persistent/fresh) = 2.85 ≥ 1.02 → PASS** (~185% improvement,
not a borderline 2%).

Why: a fresh process pays Julia load + full JIT on every item (item wall
~36 s, of which ~20 s is the timed solve including JIT); a warmed worker
pays one excluded warmup (~36 s, counted in wall time, not in certified
count) and then solves each further item in ~3–4 s. Per-item objectives are
bit-identical across fresh/persistent and across reps (e.g. item 0:
887.0975136347557, 10 iterations everywhere); state-isolation check passes,
no certification regression (32/32 certified, 0 failures, warmups also
optimal+valid).

## Verdict

**Accept gate PASSED on this workload.** The persistent warmed one-thread
process pool is a genuine, large throughput win on the proven
process-throughput axis and is kept as an opt-in tool
(`--mode=persistent`; the fresh launcher remains the default path).

## Honest limitations

- Single machine (10-core Mac, 16 GiB), one workload family (planted box
  LP, n=1200), one batch shape (8 items, 2 workers), 2 whole-batch reps.
  Cluster numbers will differ in magnitude; the mechanism (JIT/startup
  amortization) transfers, the ratio must be re-measured there.
- Small-N effect: with only 4 items/worker the excluded warmup (~36 s) is
  a large share of persistent wall time; larger queues amortize it further
  (ratio grows), smaller queues shrink the win. The gate comparison counts
  warmup against the pool honestly.
- Retained memory is bounded but real: ~2 GiB/worker stays resident after
  the batch (== peak, no growth across items). Memory budgeting for larger
  pools must reserve peak-per-worker, not per-item.
- No claim about inner-thread latency or other cones/precisions; solver
  source untouched.
