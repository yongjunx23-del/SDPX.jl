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

## Active implementation plan

`docs/PLAN.md` is the current roadmap. It is concise by design and does not
retain completed wave transcripts.

`docs/LEGACY_ENGINE_REFERENCES.md` is temporary operational evidence for Phase
10 source retirement. Remove it only after every listed legacy file and caller
has been retired and verified.

## Frozen design contracts

`docs/design/` contains mathematical or architectural contracts that production
code must respect:

- `CANONICAL_FORM.md`
- `HSD_FORMULATION.md`
- `NEWTON_SYSTEM.md`
- `NONSYMMETRIC_SCALING.md`
- `HIGH_PRECISION_SPARSE_PROVIDERS.md`
- `GENERAL_BENCHMARK.md`

`newton_system_oracle.py` is the independent small-system oracle associated
with the Newton-system contract.

## Evidence and historical material

Provider spikes, review transcripts, completed wave plans, superseded
architecture reviews, old maturity reports, and resolved bug diaries are not
part of the shipped documentation tree. They remain recoverable from Git
history. Local convenience copies created during the 2026-08-28 cleanup are
stored outside the repository at:

```text
../local-archives/SDPX.jl-docs-20260828/
../local-archives/SDPX.jl-cleanup-20260828/
```

That local path is not a package dependency and is not referenced by builds or
tests.
