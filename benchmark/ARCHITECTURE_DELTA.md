# Round 2 architecture delta

- **Reuse:** `bench/v050_round1` supplies 13 deterministic LP/SOCP/SDP
  builders and a stable `performance_trace` projection. `bench/environment.jl`
  retains machine/provenance helpers. `bench/public_conic_suite` supplies
  official-source manifests, an explicit checksum downloader, and pathological
  families. The package quick/full test profiles remain software-contract tests.
- **Duplication:** Round 1, the public pathological campaign, `bench/run.jl`,
  and several cluster probes each had a separate row schema and runner. The new
  layer consolidates only the Mac regression surface; application/cluster
  benchmarks stay specialized.
- **Missing coverage:** no staged Mac registry, no stable `BenchmarkSpec`, no
  deterministic Round-3 conditioning/rank/scaling selection, and no unified
  offline skip result for unsupported MPS/SDPA/CBF inputs.
- **New boundary:** `benchmark/` owns registry, suites, generators, cache,
  runner and comparison. `test/` keeps parser/API/kernel/small correctness.
  `src/` is unchanged: results are projected through existing diagnostics and
  `performance_trace`.
- **Suites:** Micro and Representative execute local synthetic cases and emit
  structured skips for external cases. Local Full is comprehensive but samples
  high precision. Heavy is metadata-only and refuses execution.
- **Loader policy:** NETLIB, SDPLIB, DIMACS and CBLIB retain authoritative URLs,
  filenames, formats, checksums, license notes, sizes and purposes. Full MPS,
  SDPA sparse and CBF parsers are explicitly deferred rather than rushed into
  the solver or ordinary tests.
