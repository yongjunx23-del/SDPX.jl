# Benchmark architecture

The canonical benchmark path is:

- `benchmark/runner.jl` executes suites from
  `benchmark/SDPXBenchmarkRegistry.jl` through one result schema and cache
  convention, writing matching TOML and TSV artifacts with semantic and
  provenance facts. Comparison uses `benchmark/compare.jl`.
- `benchmark/loaders/csdr_fixed_trace.jl` is the benchmark-only adapter for the
  pinned neutral Full-unitarity-EFT artifact. It reconstructs a NativeSOC Q3
  model and its physical objective without adding application logic to `src/`.
- `benchmark/gates.jl` plus `benchmark/baselines/gates.json` remain the
  numerical correctness acceptance gate.
- Archived application/cluster campaigns under `docs/evidence/bench/` (the
  former `bench/` tree, including `soc_fixed_trace/`) and the scoreboards under
  `benchmark/` stay specialized.
- The certified Full-unitarity J40 anchor is the deliberate exception: it is
  registered in the canonical `large` suite so it shares the same schema,
  cache, semantic gates and comparator. The older bootstrap scripts remain
  diagnostic/reference drivers rather than a second benchmark registry.
- `docs/evidence/bench/public_conic_suite/` is retained as archived
  provenance/catalogue only:
  manifests, tier configs, pathological generators, downloader, and data
  placeholders. Its former duplicate runner and scripts were removed.

Removed orchestration: the former small-tier benchmark harness and its
generator/environment modules, the old local phase runner, the completed
stage-specific cluster probes, and the legacy development smoke script. CI
runs only `benchmark/runner.jl micro` as the benchmark smoke.
