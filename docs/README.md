# SDPX documentation map

The shipped documentation is intentionally divided by authority.

## User manual

`docs/src/` is the Documenter source. It describes the current public API,
product-cone HSD pipeline, providers, precision, diagnostics, benchmarks, and
operations.

Build it with:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Active scientific-core implementation plan

[`design/SCIENTIFIC_CORE_ROADMAP.md`](design/SCIENTIFIC_CORE_ROADMAP.md) is the single current R0–R6 execution plan. It records qualified work, unresolved gates, cross-solver/precision diagnostics, implementation packages, and acceptance criteria. No full R0–R6 stage is closed.

The obsolete root and mirrored handover documents have been removed from the active tree; their historical contents remain recoverable from Git. Numerical evidence, failure records, and independent mathematical contracts are preserved.

## Frozen design contracts

`docs/design/` contains mathematical or architectural contracts that production
code must respect:

- `CANONICAL_FORM.md`
- `HSD_FORMULATION.md`
- `NEWTON_SYSTEM.md`
- `NONSYMMETRIC_SCALING.md`
- `GENERAL_BENCHMARK.md`

`newton_system_oracle.py` is the independent small-system oracle associated
with the Newton-system contract.

## Evidence and historical material

Provider spikes, review transcripts, completed wave plans, superseded
architecture reviews, old maturity reports, and resolved bug diaries are not
part of the shipped documentation tree. They are archived in `docs/archive/` or
remain recoverable from Git history. Local convenience copies created during
cleanups are stored outside the repository at:

```text
../local-archives/SDPX.jl-docs-20260828/
../local-archives/SDPX.jl-cleanup-20260828/
```

That local path is not a package dependency and is not referenced by builds or
tests.
