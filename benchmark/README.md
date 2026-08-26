# SDPX physics benchmark harness

This directory contains a problem-agnostic measurement harness. Physics
catalogs own problem selection, coefficient construction, and independent
validation. The harness owns process isolation, solver measurement, schema-v7
serialization, and paired comparison. It does not turn a finite benchmark
catalog into a general correctness or physics claim.

Bundled paper-grounded catalogs currently cover Hellerman modular LP, Paulos
sampled S-matrix SOCP, Lin--Zheng matrix-bootstrap SDP, and Kazakov--Zheng
finite-N lattice-bootstrap SDP under `benchmark/physics/`. Each is explicitly
versioned and carries its own provenance and semantic validator.

## Catalog contract

`PhysicsBenchmarkHarness.jl` defines four public data types:

- `PhysicsBenchmarkSpec` describes identity, provenance, reference tolerances,
  and a mandatory deterministic input fingerprint;
- `PhysicsBenchmarkEntry` selects one problem, arithmetic, and provider;
- `PhysicsBenchmarkReference` states the expected status/objective policy;
- `PhysicsBenchmarkCatalog` supplies suites plus injected `build` and
  `validate` callbacks.

`build(spec, T)` returns a named tuple with at least `problem`, `expected`, and
`kind`. `validate(spec, built, result, metrics)` returns no failures on success
or stable failure labels on rejection. Catalog files passed on the command line
must define `physics_benchmark_catalog()` and return a
`PhysicsBenchmarkCatalog`.

Catalog entries tagged `:build_only` with reference status `:build_only` or
`:sampled_build_only` take a separate construction path. The runner times
problem generation/lowering and validates the artifact fingerprint, but never
loads a linear-algebra provider or calls a solver. This state is not treated as
an optimality claim.

The bundled `fixtures/smoke_catalog.jl` is only a harness smoke test. It is not
a scientific benchmark suite.

## Canonical runner

```sh
julia --project=. benchmark/runner.jl smoke \
  --problem=smoke/lp_box --arithmetic=float64 --provider=auto \
  --samples=1 --output=/tmp/sdpx-smoke.toml
```

Use `--catalog=/absolute/path/catalog.jl` to inject a physics catalog. The CLI
contract is intentionally stable for fresh-process children:

```text
runner.jl SUITE --problem=ID --arithmetic=TYPE --provider=PROVIDER \
  --samples=1 --output=RESULT.toml [--catalog=CATALOG.jl]
```

Each selection produces one schema-v7 result row. Rows record catalog identity,
input fingerprint, source/environment hashes, solver route, certificate,
catalog validation, timing, allocation, and sampling parity. A strict run
writes its artifacts before reporting a semantic failure.

## Fresh-process campaigns

```sh
julia --project=. benchmark/fresh_process_runner.jl smoke \
  --problem=smoke/lp_box --arithmetic=float64 --provider=auto \
  --repetitions=3 --threads=1 --blas-threads=1 \
  --campaign-dir=/tmp/sdpx-fresh
```

Add the same `--catalog=...` option for an injected catalog. Every repetition
starts a new Julia process, performs its own untimed warm-up, and must emit
exactly one result row. Aggregation fails closed unless selection, catalog,
input, environment, route, status, objective, iterations, certificate, and
semantic result pair across all children.

## Paired comparison

```sh
julia --project=. benchmark/compare.jl \
  baseline.toml candidate.toml comparison.tsv
```

The comparator accepts schema-v7 files only for a valid claim. It uses scoped
`BigFloat` parsing for objective differences and rejects mismatched selections,
catalog identity, fingerprints, environments, formulations, or references.
Dirty-tree comparisons require the explicit diagnostic override in the API.

Generated inputs, result files, historical baselines, and evidence are not
stored in this directory. Keep campaign artifacts in an external work area.
