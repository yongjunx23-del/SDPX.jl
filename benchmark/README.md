# SDPX comprehensive registry and lightweight Mac regression

Round 2 separates a comprehensive workload catalogue from stage-specific
execution. It reuses the solver's existing `performance_trace`; no benchmark
hook was added to numerical code.

## Suites

- `micro`: eight tiny generated LP/SOCP/SDP and rank/conditioning cases.
- `representative`: broader generated coverage plus curated public metadata.
- `local_full`: every small/medium registered case, mostly Float64, with a
  deliberately small Float64x2/x3/x4 and BigFloat-256 sample.
- `heavy`: full NETLIB/SDPLIB/CBLIB, Mittelmann, large sparse, bootstrap and
  precision sweeps. It is register-only and the runner refuses to execute it.

## Local commands

```sh
julia --project=. benchmark/runner.jl micro
julia --project=. benchmark/runner.jl representative --verbose
julia --project=. benchmark/runner.jl local_full
julia --project=. benchmark/runner.jl micro --problem=synthetic/sdp_dense
julia --project=. benchmark/runner.jl micro \
  --problem=synthetic/sdp_dense --arithmetic=bigfloat256 --provider=bfla
```

Results are written as matching TOML and TSV files. Semantic facts—status,
objective, residuals, certificate, iterations, planned/executed route/provider
and fallback—are primary. A solved row carries `semantic_pass`, a compact list
of `semantic_failures`, and an explicit `unexpected_fallback` flag. The runner
writes the complete artifact before failing on a semantic regression. Timings
are one post-warmup observation and are never an ordinary CI failure threshold.

Current SOCP cases exercise the public SOCP frontend followed by its PSD lift;
rows label this explicitly as `conic_formulation=soc_psd_lift`. They are not
presented as native SOCP measurements, and their certificate policy is recorded
as `original_coordinate_api_unavailable`. LP and SDP rows are labeled `lp_native`
and `sdp_native`, while the separate planned/executed formulation columns retain
the KKT formulation selected inside the solver.

Compare two artifacts offline:

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
MPS/SDPA/CBF loader produces a structured `skipped` result. Current Round 2
deliberately leaves those three full external parsers unsupported.

## Precision sampling

Float64 is the default. Local Full adds selected LP/SDP/stress cases in
Float64x2/x3/x4 and BigFloat-256; it never forms a Cartesian product of cases,
providers and arithmetic. Representative and Local Full each contain one
explicit MFLA and one explicit BFLA smoke; they become structured skips when
the corresponding optional package is unavailable. Loading an arithmetic type
alone does not opt `:auto` into an optional provider.

## Round 3 subset

Start Dense Normal Equations vs Augmented KKT + LDLT work with:

1. `synthetic/lp_eq_full`
2. `synthetic/lp_eq_near_1e2`
3. `synthetic/lp_eq_near_1e4`
4. `synthetic/lp_eq_near_1e8`
5. `synthetic/lp_eq_exact_deficient`
6. `synthetic/lp_scale_1e8`
7. `synthetic/sdp_equality_heavy`
8. `synthetic/sdp_small_eig_1e8`
9. `synthetic/sdp_small_eig_1e12`
10. `synthetic/sdp_block_arrow`

These provide well-conditioned control, equality-heavy, rank-deficient,
conditioning, scaling, near-singular Schur and structured cases without a
large workload.

## Round 4 formulation planner scoreboard

The Round 4 scoreboard is a small correctness-first A/B over the dense cases
that informed the initial static formulation policy:

```sh
SDPX_ROUND4_OUT=/tmp/sdpx-round4.toml \
  julia --project=. benchmark/round4_formulation_scoreboard.jl
```

Each case is solved with `:auto`, explicit `:normal_equations`, and explicit
`:augmented` using the same arithmetic, provider, tolerances, and one warmup.
Original-coordinate certificate validity is the first gate. Residual quality
and iteration count classify only pairs that also match the registry or
analytic reference objective within a scale-aware tolerance. Wall time is
recorded but never selects the winner from this single-sample local run.
Planned and executed formulations must match and any runtime fallback is an
error.
