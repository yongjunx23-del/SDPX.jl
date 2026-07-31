# Development

Instantiate and run the package tests:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test()'
```

Build the manual locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Only the small benchmark tier belongs in routine CI:

```bash
julia --project=bench -e \
  'include("bench/run.jl"); main(tiers=(:small,), types=(Float64,), reps=1)'
```

Performance changes must state the model, arithmetic, tolerances, hardware,
thread configuration, warm-up, timing boundary, numerical result, and repeated
runtime statistic. BigFloat threads may only own disjoint mutable storage.

See
[CONTRIBUTING.md](https://github.com/yongjunx23-del/SDPX.jl/blob/main/CONTRIBUTING.md)
for the review standard and
[THIRD_PARTY_NOTICES.md](https://github.com/yongjunx23-del/SDPX.jl/blob/main/THIRD_PARTY_NOTICES.md)
for provenance requirements.

The current high-risk work is staged in two design records:

- [Optimize-mode infeasibility certificates and HSD migration](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/hsd-infeasibility-design.md)
- [Staged solver-file decomposition](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/solve-refactor-plan.md)
