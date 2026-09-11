# Persistent warmed worker pool: historical exploratory measurements

**Not an acceptance receipt.** Astra audit of `6a17505` found timeout,
queue-race, validation and provenance blockers. The measurements below used
tracked solver HEAD `f1c5df4` with an untracked, unhashed harness; they do not
qualify the subsequently committed/repaired driver. Protocol v2 requires a
new output directory and complete correctness/memory gates. Rerun pending.

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
`certification=true`. Each arm uses the same W=2 process-concurrency limit
(`--threads=1`, BLAS/OMP/MKL=1); this does not reserve cores or pin affinity.

Raw artifacts for this run live outside the worktree (`/tmp/pool_compare`,
`rep{1,2}_{fresh,persistent}/results/*.toml` + `batch_summary.toml` +
`comparison_summary.toml`); rerun with `--mode=compare --items=8
--workers=2 --outdir=<OUTSIDE_WORKTREE> --reps=2`.

## Measured comparison (interleaved fresh, persistent, fresh, persistent)

| rep | mode | reported valid | wall incl. startup+collect | reported-valid solves/h | per-item median solve | per-worker peak RSS | historical RSS column¹ | failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fresh (1 proc/item, 2 concurrent) | 8/8 | 167.4 s | 172.1 | 21.05 s | 1.93 GiB | 0 (processes exit) | 0 |
| 1 | persistent (2 warmed workers) | 8/8 | 57.1 s | 504.3 | 3.78 s | 2.02 GiB max | 2.02 GiB max | 0 |
| 2 | fresh | 8/8 | 161.3 s | 178.5 | 20.12 s | 1.91 GiB | 0 | 0 |
| 2 | persistent | 8/8 | 58.1 s | 495.5 | 4.18 s | 1.96 GiB max | 1.96 GiB max | 0 |

Medians: fresh 175.3 solves/h, persistent 499.9 solves/h.
Observed throughput ratio = **2.85** (~185% improvement). This exceeds the
performance threshold alone; the original gate omitted correctness failures
and cannot establish acceptance.

Why: a fresh process pays Julia load + full JIT on every item (item wall
~36 s, of which ~20 s is the timed solve including JIT); a warmed worker
pays one excluded warmup (~36 s, counted in wall time, not in certified
count) and then solves each further item in ~3–4 s. Per-item objectives are
bit-identical across fresh/persistent and across reps (e.g. item 0:
887.0975136347557, 10 iterations everywhere); state-isolation check passes,
all 32 historical receipts report validity, no recorded failures, and
warmups also report optimal+valid. This is not independent certification.

## Verdict

Keep as an **opt-in experimental tool**, pending source-matched tests and a
new comparison using the repaired gates. These receipts support startup/JIT
amortization on this batch, not production qualification or α3 latency gains.

## Honest limitations

- Single machine (10-core Mac, 16 GiB), one workload family (planted box
  LP, n=1200), one batch shape (8 items, 2 workers), 2 whole-batch reps.
  Cluster numbers will differ in magnitude; the mechanism (JIT/startup
  amortization) transfers, the ratio must be re-measured there.
- Small-N effect: with only 4 items/worker the excluded warmup (~36 s) is
  a large share of persistent wall time; larger queues amortize it further
  (ratio grows), smaller queues shrink the win. The gate comparison counts
  warmup against the pool honestly.
- ¹The historical table mixed fresh post-exit zero with persistent pre-exit
  samples/peaks. Both arms terminate workers at batch end. Sampled peaks did
  grow across items (about 23–35 MB for rep 1 workers); four items per worker
  establish neither a long-run retention bound nor absence of growth.
  Protocol v2 labels end-of-work/pre-exit RSS separately and checks a configured
  per-worker limit; that finite-run check is not a proof of bounded retention.
- No claim about inner-thread latency or other cones/precisions; solver
  source untouched.
