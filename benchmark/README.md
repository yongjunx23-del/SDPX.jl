# SDPX physics benchmark harness

This directory contains a problem-agnostic measurement harness. Physics
catalogs own problem selection, coefficient construction, and independent
validation. The harness owns process isolation, solver measurement, schema-v8
serialization, and paired comparison. It does not turn a finite benchmark
catalog into a general correctness or physics claim.

Every run has an explicit `execution_mode`: `:build` constructs and validates
the catalog artifact only, `:solve` executes one solver route, and `:profile`
executes the same solve route while retaining the phase telemetry exposed by
the solver. Rows also carry `requested_engine` and `executed_engine` plus
immutable campaign/shard/PBS identity. The native HSD engine is currently a
schema extension only; requesting it for `:solve`/`:profile` fails closed
until a public adapter and independent certificate contract are accepted.

Bundled paper-grounded catalogs currently cover Hellerman modular LP, Paulos
sampled S-matrix SOCP, Lin--Zheng matrix-bootstrap SDP, Kazakov--Zheng
finite-N lattice-bootstrap SDP, a finite-level Gibbs/KL exponential-cone
model, and a Giudice maximum-Rényi power-cone model under
`benchmark/physics/`. Each is explicitly versioned and carries its own
provenance and semantic validator. The EXP and Power catalogs are deliberately
build-only until their native solver routes pass the product-HSD gates.

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
`PhysicsBenchmarkCatalog`. Builders, validators, and solve contracts may be
functions or callable functors; the harness invokes injected callables through
Julia's latest-world boundary.

Catalog entries are build-only only when all three declarations agree:
`:build_only` is present in `tags`, `reference.status` is `:build_only` or
`:sampled_build_only`, and `solve_settings.build_only == true`. A mismatch is
an error artifact, not a silently reclassified benchmark. Such entries take a
separate construction path in `:build` mode. The
runner times problem generation/lowering and validates the artifact
fingerprint, but never loads a linear-algebra provider or calls a solver. This
state is not treated as an optimality claim. A build-only catalog is rejected
in `:solve`/`:profile` unless its builder returns a callable `solve_contract`:

```julia
(built, T, provider, execution_mode) -> result
```

The provider and mode arguments may be omitted for a shorter `(built, T,
provider)` or `(built, T)` contract. The same built artifact must declare an
independent `solve_reference` (a `PhysicsBenchmarkReference`, or equivalent
named tuple). The harness derives a release-grade lowercase 64-hex
`contract_fingerprint` from the callable source closure, solve reference,
solve settings, checksum, catalog identity, and route. An optional declared
token is included as an input, never trusted as the sole authority. The
construction reference is never reused as a solve oracle; missing or
unverifiable source evidence fails closed before the timed call.

Engine routing is resolved before execution and the resolved enum is passed
directly into the result row. `auto` selects `catalog_contract` when one is
present and otherwise `sdpx_legacy`; explicit `catalog_contract` requires the
hook; explicit legacy cannot solve a build-only entry; native HSD labels remain
unavailable for solve/profile until their adapter is accepted. A rejected
route records `executed_engine=none` in an error artifact.

The bundled `fixtures/smoke_catalog.jl` is only a harness smoke test. It is not
a scientific benchmark suite.

## Canonical runner

```sh
julia --project=. benchmark/runner.jl smoke \
  --problem=smoke/lp_box --arithmetic=float64 --provider=auto \
  --samples=1 --output=/tmp/sdpx-smoke.toml
```

The default is `--execution-mode=solve --engine=auto`, preserving the legacy
smoke solve. Construction-only validation is explicit:

```sh
julia --project=. benchmark/runner.jl smoke \
  --problem=smoke/lp_box --execution-mode=build --samples=3 \
  --campaign-id=local-build --shard-id=local --output=/tmp/sdpx-build.toml
```

Use `--catalog=/absolute/path/catalog.jl` to inject a physics catalog. The CLI
contract is intentionally stable for fresh-process children:

```text
runner.jl SUITE --problem=ID --arithmetic=TYPE --provider=PROVIDER \
  --execution-mode=build|solve|profile \
  --engine=auto|legacy|sdpx_legacy|catalog_contract|native|native_hsd|product_hsd \
  --campaign-id=ID --shard-id=ID --shard-index=N --shard-count=N \
  --samples=1 --output=RESULT.toml [--catalog=CATALOG.jl]
```

Each selection produces one schema-v8 result row. Rows record catalog identity,
input fingerprint, source/environment hashes, solver route, certificate (only
for solve/profile), catalog validation, scaling/layout, timing, allocation,
RSS, IQR, and sampling parity. A strict run writes its artifacts before
reporting a semantic failure.

For solve/profile, both `--samples=1` and `--samples>=3` use the same timing
boundary: deterministic construction, route resolution, and reference checks
are outside the measurement; only the resolved solve call is timed. Every
sample rebuilds once, and the optional warm-up is separate. Sample parity also
binds the input checksum, catalog and contract content hashes, scaling/layout,
engine route, and campaign/shard identity.

## Fresh-process campaigns

```sh
julia --project=. benchmark/fresh_process_runner.jl smoke \
  --problem=smoke/lp_box --arithmetic=float64 --provider=auto \
  --repetitions=3 --threads=1 --blas-threads=1 \
  --campaign-dir=/tmp/sdpx-fresh
```

Fresh campaigns forward `--execution-mode`, `--engine`, and stable campaign /
shard identity to every child. Use `--execution-mode=build` for build-only
catalog campaigns; aggregation then gates on `catalog_validation_pass` rather
than a solver certificate.

Add the same `--catalog=...` option for an injected catalog. Every repetition
starts a new Julia process, performs its own untimed warm-up, and must emit
exactly one result row. Aggregation fails closed unless selection, catalog,
input, environment, route, immutable campaign/shard identity, status, and
semantic result pair across all children. Solve/profile campaigns additionally
pair objective, iterations, and certificate evidence; build campaigns pair
catalog validation instead. Every child must be a schema-v8 row with
`sample_count=1` and must match the campaign/shard identity requested by the
parent. Required route/source fields are non-empty and all content identity
fields (`project`, manifest, driver, solver, catalog, harness, schema, and
contract) are validated as SHA-256 values.

## Paired comparison

```sh
julia --project=. benchmark/compare.jl \
  baseline.toml candidate.toml comparison.tsv
```

The comparator accepts schema-v8 files for a valid claim. It uses scoped
`BigFloat` parsing for objective differences and rejects mismatched selections,
catalog identity, fingerprints, environments, formulations, references,
execution routes, or shard topology. Campaign IDs and PBS job IDs are retained
in the comparison output but may differ between baseline and candidate jobs.
Duplicate result keys are rejected before indexing, and positive shard numbers
are compared in canonical integer form (`01` and `1` denote the same shard).
For sample counts of three or more, serialized sample arrays are parsed again;
the comparator recomputes timing statistics and semantic parity instead of
trusting the aggregate parity flag. The command writes the requested TSV
diagnostic and returns a nonzero exit status if any row is not
`comparison_valid=true`.
v7 and older files produce an explicit `legacy_schema_version` diagnostic and
can never set `comparison_valid=true`. Dirty-tree comparisons require the
explicit diagnostic override in the API.

Generated inputs, result files, historical baselines, and evidence are not
stored in this directory. Keep campaign artifacts in an external work area.
