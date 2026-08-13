# v0.5 Round-1 local benchmark harness

Deterministic local benchmark for the Round-1 performance-trace projection.
The default run exercises the complete 13-case Float64 matrix:

- LP: dense, sparse, equality-heavy, and near-rank-deficient;
- SOCP: Q3, Q8, Q32, and mixed scalar/SOC cones;
- SDP: ordinary dense, many small PSD blocks, sparse, equality-heavy, and
  block-arrow.

Every solve is projected through `SDPX.performance_trace`, so the output is a
stable projection of what the solver actually recorded.  For LP/SDP rows the
harness independently recomputes the original-coordinate certificate with
`SDPX.result_certificate` and records `certificate_valid` separately from the
projection's `certificate_available`.  Native SOCP rows retain the lifted
solver's certificate-availability fact but leave independent validity empty
until an original-coordinate `ConicResult` certificate API exists.

## Run

From the repository root:

```sh
julia --project=. bench/v050_round1/run_local.jl
```

Optional extended-precision cases (`--extended`) append a 128-bit `BigFloat`
SDP solve and, when `MultiFloats` is loadable, a `Float64x2` SDP solve:

```sh
julia --project=. bench/v050_round1/run_local.jl --extended
```

Output is written to `bench/v050_round1/out/rows.tsv` and
`bench/v050_round1/out/rows.toml`.  The two files carry the same rows:
`status`, objectives and residuals, gap, certificate availability/validity,
iteration counters, routing/provider/fallback facts, phase timings,
workspace/RSS bytes, and explicit availability flags for the timings, memory,
and routing projections.

Each row includes a variant label and the source commit.  Run the harness in
separate baseline and candidate worktrees, then pair the artifacts offline:

```sh
SDPX_BENCH_VARIANT=baseline SDPX_BENCH_OUTDIR=baseline \
  julia --project=. bench/v050_round1/run_local.jl
SDPX_BENCH_VARIANT=candidate SDPX_BENCH_OUTDIR=candidate \
  julia --project=. bench/v050_round1/run_local.jl
julia --project=. bench/v050_round1/compare_local.jl \
  baseline/rows.toml candidate/rows.toml comparison.tsv
```

Each case is warmed once by default so JIT compilation is kept out of the
recorded solve.  The comparison table reports status/certificate/route/provider/fallback
agreement, objective deltas, and candidate/baseline timing, allocation, and
memory ratios.  It never loads two SDPX versions into one Julia process.

Missing values are written as an empty field in the TSV and an empty string in
the TOML; a solve whose optional arithmetic package is absent is recorded as an
`unavailable` row rather than failing the run.

High-precision objective/residual/gap fields are stored as decimal strings so
they are not narrowed to Float64 by TOML. `solve_allocated_bytes` is the Julia
allocation count for the complete post-warmup solve call, including the result
and diagnostics objects. `process_peak_rss_bytes` is the process-wide high-water
mark reported by the runtime, not a case-local RSS delta. Availability flags
mean that the corresponding primary trace fact exists; individual optional
sub-phase fields may still be empty.

The TOML manifest carries the same complete family catalogue.  Timings and
allocations are one post-warmup sample; release A/B work should run the whole
harness repeatedly on paired nodes and aggregate one ratio per node/case.
