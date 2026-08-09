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

See [CONTRIBUTING.md](https://github.com/yongjunx23-del/SDPX.jl/blob/main/CONTRIBUTING.md)
for the review standard, [THIRD_PARTY_NOTICES.md](https://github.com/yongjunx23-del/SDPX.jl/blob/main/THIRD_PARTY_NOTICES.md)
for provenance, and [WORKLOG.md](https://github.com/yongjunx23-del/SDPX.jl/blob/main/WORKLOG.md)
for implementation and measurement history.

Operational references are kept as focused documents: the
[bridge schema](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/bridge-schema.md),
[cluster workflow](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/cluster-workflow.md),
[Julia/MOI interface](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/julia-interface.md),
[Convex interface](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/convex-interface.md),
[parameters](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/parameters.md),
[precision](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/precision.md),
[preprocessing](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/preprocessing.md),
[threading](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/threading.md),
and [automatic pipeline](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/automatic-optimization-pipeline.md)
guides.  The [adaptive parameter policy](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-parameter-policy.md)
and [benchmark results](https://github.com/yongjunx23-del/SDPX.jl/blob/main/bench/RESULTS.md)
record current operating evidence.
