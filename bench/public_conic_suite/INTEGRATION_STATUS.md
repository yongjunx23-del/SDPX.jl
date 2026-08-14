# Integration status

`bench/public_conic_suite` is retained as a provenance/catalogue layer. The
canonical execution path is `benchmark/runner.jl`; see `benchmark/README.md`.

- Manifests, tier configs, generators, and source provenance are present.
- `scripts/download_external.jl` remains available for on-demand downloads.
- Full SDPA, CBF, and MPS loaders are not implemented in this snapshot; external
  manifest entries are recorded as structured skips until a supported loader
  exists.

No benchmark results are claimed from the catalogue itself.
