# Opt-in persistent LP benchmark

`persistent_worker_pool.jl` compares fresh Julia processes with warmed workers
on a fixed planted-LP item stream. It is **not a general scan API**. Every item
is rebuilt; no prepared-session warm start or shared numerical state is used.

## Run

Use a clean committed checkout and a private Julia environment whose SDPX path
points to that checkout. Use an explicit Julia executable; set BLAS/OMP/MKL to
one. Output must be a **new directory outside the checkout**; existing evidence
is refused, never deleted.

```sh
export JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$JULIA" --startup-file=no --threads=1 --gcthreads=1 --heap-size-hint=2G \
  --project="$ENVIRONMENT" benchmark/optimization/persistent_worker_pool.jl \
  --mode=compare --items=8 --workers=2 --reps=2 \
  --rss-limit-bytes=3221225472 --outdir="$NEW_OUTPUT_DIRECTORY"
```

Modes: `fresh`, `persistent`, `compare`. Two comparison repetitions run
fresh/persistent/persistent/fresh. `workers` limits concurrent processes; it
does not reserve CPUs or pin affinity. Size concurrency for aggregate memory,
not just core count. Current workload: `lp_random_large`, seed `0x004c5003+i`,
8 items by default; all model/solver settings match between arms.

Use a bounded external process-group watchdog for the entire comparison
(the local validated N=8/W=2 run used 900 seconds). The internal parent has a
1700-second batch deadline and kills/reaps its direct children on failure.
The cooperative worker timeout is not a hard per-solve timeout. This tool is
not a general descendant-process supervisor or fault-tolerant queue service.

## Results and limits

Per-item, warmup, worker and batch TOML receipts record source/run identity,
reported certificate facts, objective-reference agreement, timings and RSS.
`comparison_summary.toml` passes only with complete valid batches, matching
numerical fingerprints, available memory samples within the configured limit
and ≥2% throughput improvement. Invalid comparisons exit nonzero. Single-arm
modes validate batch results but do not apply the comparison's memory/speed gate.

Startup, persistent warmup and collection count in batch wall time. Warmups do
not count as completed items. Fresh first-call solve timings include JIT, so
per-item time differences are not isolated numerical-kernel speedups.

Local repaired-driver evidence: **3.0471× throughput**, 32/32 valid reported
results, finite-run RSS below 3 GiB/worker. See
`../../docs/evidence/PERSISTENT_POOL_LOCAL_GATE.md`. Pre-exit RSS is not proof
of long-run retained-memory bounds; receipt validation is not independent
original-coordinate certificate recomputation. Requalify other workloads,
precisions and cluster configurations before drawing broader conclusions.

## Solver-free contract tests

Run `test_persistent_worker_pool.jl --group=<group>` in separate processes
under an external watchdog. Groups: `lifecycle`, `receipts`, `workers`,
`memory`, `exclusivity`, `race`, `gate`, `cli`. Tests stub the benchmark solver
and selected launch methods; they do not benchmark numerical performance.
Do not include this standalone mock harness in the package's test process.
