# SDPX v0.4.1-dev auto-conic package manifest

This tree is intended as a **direct continuation point** for SDPX development.
It is based on the uploaded SDPX v0.4.0 source archive and includes the
uploaded public LP/SOCP/SDP benchmark starter kit.

## What is ready now

- SDPB-style command line frontend: `bin/sdpx`.
- Small public `SolveOptions`, with every policy defaulting to `:auto`.
- Independent `dualityGapThreshold`, `primalErrorThreshold`, and
  `dualErrorThreshold` controls.
- Integer `--precision=BITS` support for BigFloat CLI ingestion/solve scopes.
- Requested/resolved option provenance and execution-plan reporting.
- Supplied public conic benchmark suite under `bench/public_conic_suite/`.
- Development-gate mapping for LP, SOCP, SDP and pathological/high-precision
  tests.
- P0 fix: explicit equality QR is reflected in sparse-Schur execution planning.
- P0 fix: sparse equality certificate accumulation traverses CSC nonzeros rather
  than all logical matrix entries.

## Intentionally not rewritten yet

- Existing v0.4.0 numerical solver core remains the production core.
- General SOCP still uses the existing PSD-arrow route except for existing
  specialized/native paths; the general native Lorentz backend is the next
  backend milestone.
- LP preprocessing is not yet end-to-end sparse.
- `KKTBackend` is not yet the sole production factor/solve/refine dispatch
  boundary.
- CLI schema v1 is the current JSON bridge; native MPS/CBF/SDPA command-line
  loaders are a follow-up milestone.

These are deliberate migration boundaries so numerical equivalence can be
checked against the bundled benchmark suite before replacing the old paths.

## First verification on your development machine

```bash
./scripts/dev_v041_smoke.sh
```

Then read `START_HERE_v0.4.1.md` and `DEVELOPMENT_v0.4.1.md`.

## Assembly provenance

- Uploaded SDPX source archive SHA256:
  `7fa890daaf4b3f824a9c66ee31458a5a2a6f01345ba7b03e6618fb7fb6f661eb`
- Uploaded benchmark archive SHA256:
  `e763644f5397b7a72ccb69f8d429e9b879562dd876c58f2a4940a60e9e17cd80`
- The assembly environment did not contain Julia, so Julia tests were not
  executed here. Static shell/JSON/TOML checks were performed.
