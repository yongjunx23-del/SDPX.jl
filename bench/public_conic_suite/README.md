# SDPX.jl Public Conic Benchmark Suite — 2026-08-11

A benchmark starter kit for continued SDPX.jl development.

## What is inside

- **30 curated SDP cases** with 2026-04-25 Mittelmann reference metadata.
- **12 core / 18 full SOCP cases** from the 2026-08-10 Mittelmann CBLIB campaign.
- **40 LP core cases**: 20 Netlib regression cases + 20 modern large LP cases.
- **25 large Network-LP stress cases**.
- **8 × 100 infeasible/weak-infeasible SDP family reference counts**.
- Parameterized **high-precision/pathological generators** for LP, SOCP and SDP.
- A download manifest and Julia downloader that records local SHA-256 digests.
- CI/nightly/release tier definitions.
- A complete development prompt in `PROMPT_SDPX_BENCHMARK_INTEGRATION.md`.

## Why public binaries are not embedded

The selected CBLIB SOCP files alone include several files of hundreds of MB.
Committing them would bloat SDPX.jl and make CI brittle. Instead:

1. the archive carries exact public filenames/URLs where a stable direct URL is known;
2. `scripts/download_external.jl` retrieves them into `data/external/`;
3. every downloaded file is hashed into `data/download_lock.toml`;
4. generated pathological cases are defined completely in source and require no binary download.

This is better suited to a long-lived solver repository.

## Suggested repository location

Copy this whole directory to:

```text
SDPX.jl/
└── bench/
    └── public_conic_suite/
```

Keep SDPX's existing benchmark directories. This suite is an additional
**standardized workload layer**, not a replacement for application-specific
CSDR/lattice-bootstrap/mixed-precision benchmarks.

## First setup

From the copied suite directory:

```bash
julia scripts/setup_benchmark_env.jl
```

Then generate a small pathological campaign:

```bash
julia --project=. scripts/run_pathological.jl --precision=float64
julia --project=. scripts/run_pathological.jl --precision=float64x4
julia --project=. scripts/run_pathological.jl --precision=bigfloat256
julia --project=. scripts/run_pathological.jl --precision=bigfloat512
```

Download only the manageable SDP core first:

```bash
julia scripts/download_external.jl --suite=sdp --tier=core
```

CBLIB core is large. Download deliberately, e.g. first two files:

```bash
julia scripts/download_external.jl --suite=socp --tier=core --limit=2
```

## Integration order

1. **Import + certification before speed.**
2. Make `smoke` green on Float64.
3. Add Float64x4/BigFloat precision ladders.
4. Add external solver adapters on the *same hardware*.
5. Establish a pinned baseline JSON/CSV.
6. Only then optimize KKT/Schur/scaling/cone kernels.
7. Every optimization PR must report both performance delta and certification delta.

## Important SDPX-specific note

General MOI SOC constraints currently pass through SDPX's supported conic
frontend and may use a PSD arrow lift, while specialized fixed-trace Q3 models
have a native SOCP path. Do **not** mix those formulations in one timing score.
Record a `formulation` field and benchmark:
- general SOC frontend / arrow-lift behavior;
- native fixed-trace SOC backend;
separately.

Likewise, LP benchmarking should use the dedicated LP frontend when possible so
that the benchmark actually exercises the optimized LP path instead of
manufacturing 1×1 PSD blocks.

## Files to read first

1. `PROMPT_SDPX_BENCHMARK_INTEGRATION.md`
2. `configs/tiers.toml`
3. `RESULT_SCHEMA.md`
4. `manifests/pathological_suites.toml`
5. `references/*.csv`
