# Integration status

`bench/public_conic_suite` is retained as a provenance/catalogue layer. The
canonical execution path is `benchmark/runner.jl`; see `benchmark/README.md`.

- Manifests, tier configs, generators, and source provenance are present.
- The canonical runner implements checksum-verified Netlib compressed MPS,
  historical SDPLIB compact SDPpack, and DIMACS sparse-SDPA loaders.
- Typed, JuMP-free pathological LP/SOCP/SDP cases execute through the same
  registry and result schema.
- CBF is limited to continuous `F/L+/L-/L=/Q` formulations SDPX can
  represent without changing cones. CBLIB `nql30` is the pinned native-SOCP
  anchor; the canonical Large-suite campaign now has a three-process
  certificate-valid Float64 result. Unsupported CBF cone/integer features
  remain explicit skips or errors.
- `scripts/download_external.jl` is legacy catalogue tooling: it records an
  observed digest but does not validate a pinned expected digest and writes to
  a different cache root. Use canonical `benchmark/runner.jl --prepare` for
  executable benchmark inputs.

No benchmark results are claimed from the catalogue itself.
