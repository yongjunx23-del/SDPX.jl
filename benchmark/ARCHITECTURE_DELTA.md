# Benchmark architecture

The canonical benchmark path is:

- `benchmark/runner.jl` executes suites from
  `benchmark/SDPXBenchmarkRegistry.jl` through one result schema and cache
  convention, writing matching TOML and TSV artifacts with semantic and
  provenance facts. Comparison uses `benchmark/compare.jl`.
- `bench/gates.jl` plus `bench/baselines/gates.json` remain the numerical
  correctness acceptance gate.
- Application/cluster campaigns under `bench/`, the fixed-trace benchmark under
  `bench/soc_fixed_trace/`, and the scoreboards under `benchmark/` stay
  specialized.
- `bench/public_conic_suite/` is retained as provenance/catalogue only:
  manifests, tier configs, pathological generators, downloader, and data
  placeholders. Its former duplicate runner and scripts were removed.

Removed orchestration: the former small-tier benchmark harness and its
generator/environment modules, the old local phase runner, the completed
stage-specific cluster probes, and the legacy development smoke script. CI
runs only `benchmark/runner.jl micro` as the benchmark smoke.
