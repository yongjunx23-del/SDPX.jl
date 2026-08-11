# Snapshot notes

This package was assembled from the two user-supplied archives on 2026-08-11.

## Inputs

```text
SDPX.jl-main (1).zip
SHA-256 7fa890daaf4b3f824a9c66ee31458a5a2a6f01345ba7b03e6618fb7fb6f661eb

SDPX_benchmark_suite_20260811(1).zip
SHA-256 e763644f5397b7a72ccb69f8d429e9b879562dd876c58f2a4940a60e9e17cd80
```

## Runtime-validation status

The assembly environment used to prepare this snapshot does **not** contain a
Julia executable.  The new Julia files were therefore reviewed statically,
shell wrappers were checked with `bash -n`, JSON fixtures were parsed, and
source delimiters were sanity-checked, but the Julia test suite could not be
executed here.

The first action on a development machine should therefore be:

```bash
./scripts/dev_v041_smoke.sh
```

or, for a more controlled start:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia bin/setup_cli.jl
./bin/sdpx --help
```

Any syntax/API issue found by that first Julia run should be fixed as a small
bootstrap commit while preserving the new frontend/midend boundary described
in `DEVELOPMENT_v0.4.1.md`.

## Final auto-conic packaging additions

The packaged development baseline also contains two low-risk P0 code fixes:

- explicit `equality_solver=:qr` is reflected during Schur backend planning so
  memory preflight does not advertise the sparse-Schur route that the current
  workspace will disable;
- sparse equality backward-error/certificate scale accumulation uses CSC
  `nzrange` traversal instead of scanning every logical B entry.

A lazy precision-registry guard was also added to the CLI bridge so direct
`include`/embedding does not rely on package-loader `__init__` behavior.
