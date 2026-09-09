# Autoresearch: SDPX solver speed / iterations / memory across precisions

## Objective
Make the SDPX conic solver faster (wall time), fewer iterations, and lower
peak memory across Float64 / MultiFloat (Float64x4) / BigFloat arithmetic.
Primary workload: frozen CSDR alpha3 benchmark (Float64x4, 8400 vars,
42 equalities, 4200 SOC blocks, bordered KKT route) — digest-guarded so any
numerical drift fails the run. Secondary: Float64 LP/SOCP general catalog.

## Metrics
- **Primary**: solver_seconds (s, lower is better) — CSDR alpha3 median of 3
- **Secondary**: iterations, allocation_bytes, peak_rss_bytes, lp_seconds

## How to Run
`./.auto/measure.sh` — outputs `METRIC name=value` lines.
- CSDR: benchmark/autoresearch/csdr_alpha3_x4.jl (frozen input, trajectory
  digest enforced inside the script)
- LP: .auto/lp_bench.jl (Float64 general catalog small tier, threads=1)

## Files in Scope
- src/hsd/** (predictor-corrector, initialization, linesearch, termination)
- src/kkt/** (symmetric core, reduced schur, residual workspace)
- src/cones/** (SOC/PSD kernels, runtime)
- src/factor_cache/** (routes, structure cache)
- src/program/equilibrate.jl
- benchmark/autoresearch/csdr_alpha3_x4.jl (measurement harness only —
  digest constants must never change)

## Off Limits
- test/** expectations, docs/**, .github/**
- Any change that alters the CSDR trajectory digest (objective/iterations/
  residuals/gap must stay bit-identical)
- Tolerance loosening, precision fallbacks, model-name dispatch
- Protected checkouts under /Users/xuyongjun/Desktop/project/SDPX/**

## Constraints
- Correctness gate: .auto/checks.sh (focused test subset) must pass before
  a keep; trajectory digest equality is part of measure.sh itself
- Julia: explicit executable, --startup-file=no, BLAS/OMP/MKL=1, offline
- Local machine: 10-core M-series Mac; heavy parallel validation (16/64
  threads) is deferred to the UCAS cluster, not part of this loop

## What's Been Tried
- (baseline session start)

## Known platform divergence (from CI diagnosis 2026-09-10)
- bordered compact-Schur LP (12 rows, full=14, compact=3) breaks down at
  iteration 6 with :direction_breakdown on x86 (CI ubuntu/windows, Julia 1.10
  and 1.x) while ARM macOS passes. mu=9.6e-6, tau=0.104, kappa=1.8e-4.
- power_epigraph_small E2E: x86 reaches certified optimum, ARM breaks down.
  Both are FP-arithmetic divergence in direction construction / residual
  acceptance gates. Fixing these robustness gaps IS part of the optimization
  goal (a solver that breaks down is not high-performance).

## Session baseline (2026-09-10, commit 27839e4)
- CSDR alpha3 Float64x4: 19.9s median, 105 iterations, 914MB alloc, 5.0GB RSS
- lp_seconds (3 catalog cases): 30.9s (JIT-heavy; treat as secondary)
- Guard: certificate valid + objective within 1e-8 rel + 3-run determinism

## Experiment log
### E1: threaded exact dual Newton stats (KEPT)
- Dual residual gate was 16.8ms/gate x2/iter (3.5s total, 30% of solve).
- Added `_dual_newton_stats_threaded!` in the MFLA ext: fixed contiguous
  column partition, per-task local reduction, ordered merge. Per-column
  muladd chain unchanged; max/and are exact so results are bit-identical.
- CSDR: 19.9s -> 17.37s (1.15x), 105 iters unchanged, objective/residuals
  bit-identical. NOTE: ext module code must live INSIDE the module `end`.
- Remaining hot spots (per iter): Gram SYRK 19ms, gate cone 2ms, solve
  stages ~12ms x2, corr_rhs 9.2ms, schur_assembly 8.3ms, homogeneous 6.2ms.

### E2: direct dim-3 SOC boundary step + threaded max-step (KEPT)
- boundary_alpha spent most time copying blocks into scratch before
  boundary_step!; copy cost > step cost for dim-3 blocks.
- Added copy-free direct boundary computation for dim-3 SOC + threaded
  exact min reduction. CSDR: 17.37 -> 17.35s (small; spawn overhead ate
  most of it), allocations -4MB, RSS -250MB. Bit-identical.
- Learned: MultiFloat lane SIMD only wins on large reductions (syrk/gemm);
  small-block lane versions are SLOWER (gather overhead) and not
  bit-identical (different rounding tree). Do not lane-ify small blocks.

### Current phase breakdown after E1+E2 (15.55s core, 17.35s wall)
per-iteration (~105 iters): Gram SYRK 19ms (hardware-bound, linear scaling
to 4 threads), q3_metric total 27ms, homogeneous solve 6.1ms, predictor
solve 23ms, corrector solve 24ms, corr_rhs 4.2ms, schur_assembly 8.3ms,
line_search 12.4ms, residual 10.8ms.
Next lever is ALGORITHMIC: reduce the 105 iterations (predictor/corrector
policy, step quality) — that is GPT Pro territory per user directive.
