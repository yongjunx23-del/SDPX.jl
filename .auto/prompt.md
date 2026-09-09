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
