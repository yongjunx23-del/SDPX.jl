# v0.4.1-dev change map

This is the shortest map of where to start coding.

## New frontend

- `src/frontend/solve_options.jl` — small public all-auto `SolveOptions`.
- `src/frontend/high_level_solve.jl` — bridge from `SolveOptions` to existing
  SDP/LP/SOC solve paths.
- `bin/sdpx.jl` — SDPB-style user CLI.
- `bin/sdpx` — shell entry point.
- `bin/setup_cli.jl` — one-time CLI environment setup.

## New midend boundary

- `src/midend/resolve_options.jl` — resolves the all-auto policy to concrete
  `SolverOptions{T}` while leaving structural backend selection to the existing
  `ExecutionPlan`.

## Backend

No large numerical backend rewrite is hidden in this archive.  Existing LP,
Q3/SOC, SDP, Schur, KKT and validation implementations are preserved so the
next commits can establish equivalence gates before migration.

The exact P0 backend tasks are in `DEVELOPMENT_v0.4.1.md` and
`PROMPT_V041_CONIC_REFACTOR.md`.

## Benchmark

- `bench/public_conic_suite/` — complete supplied benchmark starter kit.
- `bench/public_conic_suite/INTEGRATION_STATUS.md` — what is and is not wired
  into SDPX yet.

## Tests/examples/docs

- `test/frontend_auto_options.jl`
- additional SDPB-style parser assertions in `test/cli_bridge.jl`
- `examples/cli_problem.json`
- `examples/cli_problem_high_precision.json`
- `docs/cli.md`
- `docs/architecture-v041.md`
- `scripts/dev_v041_smoke.sh`

## First command on a machine with Julia

```bash
./scripts/dev_v041_smoke.sh
```

The preparation environment did not have Julia; see `SNAPSHOT_NOTES.md`.
