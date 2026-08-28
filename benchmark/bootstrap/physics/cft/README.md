# CFT polynomial-matrix build benchmark

This catalog preserves the former compact CFT/PMP fixture without retaining the
old `BootstrapBenchmark` registry.

It builds a deterministic half-line polynomial matrix program through
PMP2SDP.jl at four scales. The catalog is deliberately `build_only`: it checks
construction, parameter identity, and fingerprints, but does not claim that the
low-order surrogate reproduces the full published conformal-bootstrap bound.

PMP2SDP is resolved in this order:

1. an installed `PMP2SDP` package;
2. `SDPX_PMP2SDP_ROOT=/path/to/PMP2SDP.jl`;
3. a sibling `PMP2SDP.jl` developer checkout.

No workstation-specific absolute path is embedded.

Example:

```bash
julia --project=benchmark/bootstrap/benchenv \
  benchmark/bootstrap/runner.jl smoke \
  --catalog=benchmark/bootstrap/physics/cft/catalog.jl \
  --execution-mode=build --output=/tmp/sdpx-cft-build.toml
```

See `PROVENANCE.md` for the distinction between the reference target and the
shipped surrogate fixture.
