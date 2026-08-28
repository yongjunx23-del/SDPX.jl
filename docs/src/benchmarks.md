# Benchmarks and evidence

SDPX keeps solver-oriented and physics-oriented evidence separate:

- `benchmark/general/` contains deterministic LP/SOCP/SDP/Exp/Power cases and
  public-data readers. The black-box E2E selects a small subset from here.
- `benchmark/bootstrap/physics/` contains provenance-backed physical catalogs.

The bootstrap harness provides reproducible physical measurements. Physics
decisions live in injected catalogs; the harness does not embed application
registries, external-data loaders, or historical result tables.

The active entry points are:

- `benchmark/bootstrap/runner.jl` for in-process runs and schema-v9 result rows;
- `benchmark/bootstrap/fresh_process_runner.jl` for independent-process repetitions;
- `benchmark/bootstrap/compare.jl` for paired baseline/candidate comparisons;
- `benchmark/bootstrap/PhysicsBenchmarkHarness.jl` for the catalog/build/validate API.

A catalog must provide deterministic problem identities and fingerprints, a
typed builder, reference tolerances, and an independent validation callback.
The runner records both the solver's original-coordinate certificate and the
catalog verdict. Neither timing nor an objective value is reportable as a
successful benchmark when either validation boundary fails.

## Evidence policy

A performance claim must identify the catalog version, problem fingerprint,
arithmetic, precision, tolerances, provider, solver route, source hashes,
hardware, thread configuration, warm-up boundary, repetition count, and robust
timing statistic. Compare only rows that the comparator accepts as paired.

Finite catalog execution is discrete evidence. It does not establish behavior
between sampled parameter values, on untested hardware, or for a continuum of
physics inputs. Any stronger conclusion requires a separately documented
analytic or interval-certified argument.

See `benchmark/README.md` for the catalog contract and exact commands.
