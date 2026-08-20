# SDPX canonical benchmark registry and runner

`benchmark/runner.jl` is the canonical benchmark runner. It loads
`benchmark/SDPXBenchmarkRegistry.jl`, which owns the registry, suites,
generators, cache, result schema, and comparison. Every suite writes one
machine-readable result convention and no benchmark hook is added to numerical
code.

## Suites

- `micro`: eleven tiny generated LP/SOCP/SDP, rank/conditioning, and typed
  pathological cases.
- `representative`: broader generated/pathological coverage plus executable
  checksum-pinned public cases when their local cache is present.
- `local_full`: every small/medium registered case, mostly Float64, with a
  deliberately small Float64x2/x3/x4 and BigFloat-256 sample.
- `large`: executable cluster-only public/application anchors. It contains the
  native-CBF CBLIB `nql30` SOCP in Float64 and the pinned Full-unitarity-EFT
  J40/Na15/Nmu200/Nx2/Nalpha2 NativeSOC case in Float64x2 and Float64x4. The
  runner refuses this suite outside PBS unless `--allow-large` is explicitly
  supplied for a diagnostic.
- `heavy`: full NETLIB/SDPLIB/CBLIB, Mittelmann, large sparse, bootstrap and
  precision sweeps. It is register-only and the runner refuses to execute it.

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
and fallback) are primary. A solved row carries `semantic_pass`, a compact list
of `semantic_failures`, and an explicit `unexpected_fallback` flag. The runner
writes the complete artifact before failing on a semantic regression. Timings
are one post-warmup observation and are never an ordinary CI failure threshold.

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
row records the source commit and whether the source tree was dirty. Schema-v6
rows must carry canonical 64-hex `solver_source_sha256` values on
both sides; the values may differ for an optimization candidate. Missing or
legacy hashes fail closed with comparison evidence instead of producing a
timing claim. For repeated samples, `total_seconds` is the reported aggregate
median (the arithmetic mean of the two middle observations for an even sample
count), so the scalar timing and `sample_median_seconds` are consistent.

For timing claims, use the fresh-process campaign wrapper. Every repetition
starts a separate Julia process, performs an untimed warmup, writes its own raw
TOML/TSV/log, and is aggregated only when status, objective, iterations,
certificate, route, input fingerprint, and environment agree:

```sh
julia --project=. benchmark/fresh_process_runner.jl micro \
  --problem=synthetic/lp_box --arithmetic=float64 --provider=auto \
  --repetitions=3 --threads=1 --blas-threads=1 \
  --campaign-dir=work/baseline/lp_box_float64
```

`process_peak_rss_bytes` is the peak of the complete child Julia process. It
includes the runtime, package loading, compilation caches, and allocator
arenas, so use it for same-environment regression detection rather than as a
solver-workspace measurement. `workspace_bytes` is the separate solver-owned
storage estimate.

## Public data and provenance

The registry curates NETLIB LP, SDPLIB, DIMACS continuous conic and CBLIB
continuous instances. Data are not vendored. Downloads occur only under an
explicit `--prepare` request, use authoritative URLs, and verify SHA-256:

```sh
julia --project=. benchmark/runner.jl representative \
  --prepare --problem=netlib/afiro
```

Ordinary tests and solves never access the network. Missing data or a missing
or unsupported loader produces a structured `skipped` result. The canonical
runner implements Netlib compressed MPS, historical SDPLIB compact SDPpack,
and DIMACS sparse SDPA. CBF support is deliberately limited to cone families
that SDPX can execute without changing the published formulation. CBLIB
`nql30` is the native continuous-SOCP anchor; CBF instances containing integer,
PSD, rotated-quadratic, exponential, or power-cone constructs fail closed or
remain metadata-only.

`nql30` is a completed native-SOCP anchor. On the accepted local one-thread
campaign, three fresh processes all return `Optimal` in 13 iterations with an
original-coordinate certificate, objective `-0.9460283775140597`, and matching
input/route/source identities. The median post-warmup solve time is 14.814 s
and solver workspace is 186,614,568 bytes. These numbers are local evidence,
not a cross-machine reference; reproduce them through the Large suite before
comparing another candidate.

Fresh-process campaigns for an explicitly authorized Large-suite diagnostic
must also pass `--allow-large` to `fresh_process_runner.jl`; the flag is
forwarded unchanged to each canonical child.

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

## Precision sampling

Float64 is the default. Local Full adds selected LP/SDP/stress cases in
Float64x2/x3/x4 and BigFloat-256; it never forms a Cartesian product of cases,
providers and arithmetic. Representative and Local Full each contain one
explicit MFLA and one explicit BFLA smoke; they become structured skips when
the corresponding optional package is unavailable. Loading an arithmetic type
alone does not opt `:auto` into an optional provider.

## Specialized campaigns

Application/cluster benchmarks under `bench/` and the scoreboards under
`benchmark/` (`round3_augmented_ab.jl`, `round4_formulation_scoreboard.jl`,
`round5_soc_scoreboard.jl`) remain specialized and outside the canonical
runner. `bench/gates.jl` with `bench/baselines/gates.json` remains the
correctness acceptance gate.
