# SDPX.jl Public Conic Benchmark Suite (provenance catalogue)

This directory is retained as a provenance and workload catalogue for public
conic benchmarks. It does not contain its own benchmark runner. Execution is
handled by the canonical repository runner:

```bash
julia --project=bench benchmark/runner.jl micro --output=/tmp/sdpx-micro.toml
```

See `benchmark/README.md` for suite selection, result schema, and comparison.

## Retained contents

- `manifests/`: public source manifests for LP, SOCP, and SDP workloads,
  including download URLs and pathological parameter grids.
- `configs/`: tier and development-gate definitions.
- `generators/`: parameterized LP/SOCP/SDP pathological generators.
- `scripts/download_external.jl`: on-demand, checksum-pinned download helper
  for manifest entries.
- `data/external/` and `data/generated/`: ignored placeholders for downloaded
  and generated inputs.
- `Project.toml`: suite dependencies for local tooling.
- `RESULT_SCHEMA.md` and `SOURCES_AND_PROVENANCE.md`: retained field and source
  provenance notes.

External binaries are not embedded. Downloads require an explicit helper call
and are stored under ignored data directories. Mittelmann reference times in
provenance files are orientation data only and must never be used for
cross-machine speedup claims.

The pathological generators are retained, but campaign execution goes through
the canonical registry runner or the retained specialized benchmark campaigns.
There is no standalone suite runner here.
