# Massless EFT pole-augmented bootstrap slice

This reviewed benchmark is a finite, sampled build-only problem.  It exposes
mathematically identical SOCP and 2x2 real-PSD `SDPX.Model` builders from one
coefficient artifact.  It does not specialize or modify a solver route.

## Normalization and frozen algebra

The implemented source variable is `tau`; its physical factor-four
normalization is unresolved and is not inferred.  The only frozen equivalence
is

```
[2-Im(tau)  Re(tau); Re(tau) Im(tau)] >= 0
  <=> [1, Re(tau), 1-Im(tau)] in Q3
  <=> abs(1+i*tau) <= 1.
```

The source rows are generated in a 1024-bit BigFloat guard scope in the
external generator's operation order, checked against a literal reference
implementation, and converted once to the requested target type.  The full
artifact fingerprint is `external_checksum`; catalog loading uses only a
cheap spec-plus-manifest fingerprint and never builds N14.

## Objectives and claim boundary

`build_model` and `build_soc_problem` accept only `:none`, `:min_g0`, and
`:max_g0`.  `g0 = 3 alpha_00 - 3 alpha_pole`.  `g2` retains its disputed
source coefficient as diagnostic metadata only and is not a builder objective.
All rows and tiers are `:sampled_build_only`; held-out grids have
`heldout_ngrid = 2*ngrid-1` and audits are diagnostic only.  The exact N14
external receipt strings remain manifest metadata, not a coefficient-matrix
identity or certificate.  Commit `1541ab4` is recorded only as the SDPX
import base.

The production spec is catalog metadata but is absent from ordinary suites.
Use the explicit guarded `production_driver.jl` only when intentionally
building N14/Q2048; production generation is not a normal test or gate.

Run `julia examples/massless_eft_representation.jl soc` or `sdp` for a small
build/parity probe.  The probe reports build status and a real algebraic parity
certificate, and exits nonzero if either fails.
