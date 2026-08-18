# SDPX canonical benchmark registry and runner

`benchmark/runner.jl` is the canonical benchmark runner. It loads
`benchmark/SDPXBenchmarkRegistry.jl`, which owns the registry, suites,
generators, cache, result schema, and comparison. Every suite writes one
machine-readable result convention and no benchmark hook is added to numerical
code.

## Suites

- `micro`: eight tiny generated LP/SOCP/SDP and rank/conditioning cases.
- `representative`: broader generated coverage plus curated public metadata.
- `local_full`: every small/medium registered case, mostly Float64, with a
  deliberately small Float64x2/x3/x4 and BigFloat-256 sample.
- `large`: executable cluster-only application anchors. It currently contains
  the pinned Full-unitarity-EFT J40/Na15/Nmu200/Nx2/Nalpha2 NativeSOC case in
  Float64x2 and Float64x4. The runner refuses this suite outside PBS unless
  `--allow-large` is explicitly supplied for a diagnostic.
- `heavy`: full NETLIB/SDPLIB/CBLIB, Mittelmann, large sparse, bootstrap and
  precision sweeps. It is register-only and the runner refuses to execute it.
- `analytic_fast`: small closed-form LP/SOCP/SDP correctness and equivalence
  checks.
- `analytic_numerical`: moderate size/pathology with selected MFLA/BFLA rows.
- `analytic_stress`: explicitly gated scale/precision research cases.

## Local commands

```sh
julia --project=bench benchmark/runner.jl micro --output=/tmp/sdpx-micro.toml
julia --project=. benchmark/runner.jl representative --verbose
julia --project=. benchmark/runner.jl local_full
julia --project=. benchmark/runner.jl micro --problem=synthetic/sdp_dense
julia --project=. benchmark/runner.jl micro \
  --problem=synthetic/sdp_dense --arithmetic=bigfloat256 --provider=bfla
```

Results are written as matching TOML and TSV files. Semantic facts (status,
objective, residuals, certificate, iterations, planned/executed route/provider,
and fallback) are primary. A solved row carries `PASS`/`FAIL`/`UNRESOLVED`,
`semantic_pass`, compact individual/group failures, and an explicit
`unexpected_fallback` flag. The runner writes the complete artifact before
failing on a semantic regression. Timing comparisons require passing gates;
use at least three post-warmup samples for median/MAD data.

Current SOCP cases exercise the native Lorentz frontend. Rows record the
executed specialization and whether a PSD lift was present; the Full-unitarity
case requires `fixed_trace_q3`, an original-coordinate Lorentz certificate,
MFLA provenance, and `psd_lift_used=false`. LP and SDP rows are labeled
`lp_native` and `sdp_native`, while the separate planned/executed formulation
columns retain the KKT formulation selected inside the solver.

## Compare

```sh
julia --project=. benchmark/compare.jl baseline.toml candidate.toml comparison.tsv
```

The comparator requires identical selections and input fingerprints before it
reports semantic agreement or timing ratios. It also rejects mismatched Julia,
OS, CPU, thread counts, BLAS threads, precision and conic formulation. Every
row records the source commit and whether the source tree was dirty.

## Public data and provenance

The registry curates NETLIB LP, SDPLIB, DIMACS continuous conic and CBLIB
continuous instances. Data are not vendored. Downloads occur only under an
explicit `--prepare` request, use authoritative URLs, and verify SHA-256:

```sh
julia --project=. benchmark/runner.jl representative \
  --prepare --problem=netlib/afiro
```

Ordinary tests and solves never access the network. Missing data or a missing
MPS/SDPA/CBF loader produces a structured `skipped` result. The full external
parsers remain unsupported.

The Full-unitarity input is the neutral
`csdr_fixed_trace_reduced_v1` payload, not an archived `SDPXProblem`. Copy the
payload into the configured cache root under `csdr/`; its pinned SHA-256 is
verified before deserialization. It is not downloaded automatically and is not
tracked by Git. A PBS template is provided at
`benchmark/cluster/full_unitarity_eft.pbs`; submit Float64x2 and Float64x4 as
separate jobs so their timings remain directly attributable.

## Full-unitarity scaling ladder

The executable anchor is J40/Na15/Nmu200/Nx2/Nalpha2. Heavy metadata also
records the next two source-model rungs:

- scale 2: J80/Na30/Nmu400/Nx4/Nalpha4;
- scale 4: J160/Na60/Nmu800/Nx8/Nalpha8.

Those rungs are register-only until the source generator creates independent
neutral payloads and their checksums are pinned. Doubling means regenerating
the physical model with all five resolution parameters doubled; copying the
J40 matrices or repeating cones is forbidden. The J40 payload is the certified
application holdout. Algorithm tuning uses synthetic fixed-trace proxies and
bounded 1/5/20-iteration diagnostics, not repeated full holdout solves.

`bench/public_conic_suite/` is retained as a provenance/catalogue layer:
manifests, tier configs, pathological generators, the on-demand downloader,
and data placeholders. It has no runner of its own.

## Analytic referee suite

`benchmark/analytic/` supplies six deterministic LP/SOCP/SDP families with
closed-form objectives (and bound-direction checks for the moment hierarchy).
They use the same canonical runner and result schema as every other benchmark;
there is no second timer, provider selector or solver path. Only `PASS` rows
are eligible for timing comparisons, while `FAIL` and `UNRESOLVED` remain
visible in a generated failure map.

```sh
JULIA_DEPOT_PATH=/tmp/sdpx-julia-depot julia --project=. \
  benchmark/runner.jl analytic_fast --allow-semantic-failures \
  --output=/tmp/sdpx-analytic.toml
```

See `benchmark/analytic/README.md` for the exact formulas, registered grids,
group gates and initial baseline observations.

## Precision sampling

Float64 is the default. Local Full adds selected LP/SDP/stress cases in
Float64x2/x3/x4 and BigFloat-256; it never forms a Cartesian product of cases,
providers and arithmetic. Representative and Local Full each contain one
explicit MFLA and one explicit BFLA smoke; they become structured skips when
the corresponding optional package is unavailable. Loading an arithmetic type
alone does not opt `:auto` into an optional provider.

## Specialized campaigns

Application/cluster benchmarks remain under `bench/`. The historical Round
3--5 scoreboards have been removed: their semantic coverage is retained by the
canonical registry suites, unit tests, and the correctness contracts in
`benchmark/contracts/`. `bench/gates.jl` with `bench/baselines/gates.json`
remains the correctness acceptance gate.
