# Physics/bootstrap benchmark

This directory contains the provenance-backed physical benchmark system. The
general solver benchmark and black-box E2E cases live in `benchmark/general/`.

## Harness and tools

- `PhysicsBenchmarkHarness.jl` — catalog types, tracked loading, validation,
  schema, runner, and comparator module.
- `runner.jl` / `runner_impl.jl` — in-process build/solve/profile execution.
- `result_schema.jl` — schema-v9 TOML/TSV result contract.
- `compare.jl` / `compare_impl.jl` — strict baseline/candidate pairing.
- `fresh_process_runner.jl` / `fresh_process_campaign.jl` — repeated isolated
  Julia processes for timing and stability evidence.
- `fixtures/smoke_catalog.jl` — harness smoke fixture, not a physics claim.
- `benchenv/Project.toml` — isolated benchmark environment.

## Physical catalogs

`physics/` contains independent application catalogs:

- `cft/` — build-only conformal-bootstrap PMP surrogate;
- `lattice_bootstrap/` — finite-N lattice-bootstrap SDP;
- `matrix_bootstrap/` — matrix-bootstrap SDP;
- `modular_lp/` — finite-grid modular-bootstrap LP;
- `smatrix_soc/` — sampled S-matrix SOCP;
- `thermal_exp/` — Gibbs/KL exponential-cone model;
- `thermal_power/` — Renyi power-cone model.

Each catalog owns its problem builder, provenance, deterministic fingerprint,
reference policy, and semantic validator.

## Build-only semantics

A build-only catalog must agree in three places:

1. `:build_only` tag;
2. build-only reference status; and
3. `solve_settings.build_only == true`.

Build success validates construction only. It is not an optimality or physics
claim. A solve/profile route requires an explicit solve contract and an
independent solve reference.

## Running

```bash
julia --project=benchmark/bootstrap/benchenv \
  benchmark/bootstrap/runner.jl smoke \
  --catalog=benchmark/bootstrap/fixtures/smoke_catalog.jl \
  --samples=1 --output=/tmp/sdpx-bootstrap-smoke.toml
```

Use `fresh_process_runner.jl` for performance evidence and `compare.jl` for
paired comparisons. Generated results belong outside the source tree.
