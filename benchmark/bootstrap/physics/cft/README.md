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

## Catalog contract

- **Physical assumptions/conventions:** a deliberately synthetic matrix-polynomial half-line surrogate for a CFT functional.
- **Primary equations/version:** provenance records arXiv:2411.15300; the full conformal-block source table is not shipped, so no paper reproduction is claimed.
- **Truncation/discretization:** monomial degree 4/8/12/27, block counts 2/4/6/10, and matrix dimensions 1/1/2/2.
- **Convex variables/objective/cones:** one scalar functional variable, coefficient matching, and matrix-SOS PSD Gram blocks; objective is synthetic `max y`.
- **Strict witness:** `y=0` is the intended positive-sector witness; it is validated as construction metadata, not an SDPX optimum certificate.
- **Reference status:** `:build_only`, `paper_equivalent=false`.
- **Excluded claims:** no conformal-block/crossing/OPE bound, published `10.9293` reproduction, or continuum conclusion.
- **Scaling tiers:** tiny/small/medium/stress degree and matrix-block ladder above; build-only.

The shared checklist is [`../PHYSICS_CATALOG_TEMPLATE.md`](../PHYSICS_CATALOG_TEMPLATE.md).
