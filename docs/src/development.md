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
julia --project=bench benchmark/runner.jl micro --output=/tmp/sdpx-micro.toml
```

Performance changes must state the model, arithmetic, tolerances, hardware,
thread configuration, warm-up, timing boundary, numerical result, and repeated
runtime statistic, and must preserve the original-coordinate certificate
boundary. BigFloat threads may only own disjoint mutable storage.

See [CONTRIBUTING.md](https://github.com/yongjunx23-del/SDPX.jl/blob/main/CONTRIBUTING.md)
for the review standard and
[THIRD_PARTY_NOTICES.md](https://github.com/yongjunx23-del/SDPX.jl/blob/main/THIRD_PARTY_NOTICES.md)
for provenance.

Operational references are kept as focused documents: the
[bridge schema](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/bridge-schema.md),
[cluster workflow](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/cluster-workflow.md),
[adaptive dense/sparse optimization](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/adaptive-dense-sparse-optimization.md),
[adaptive parameter policy](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/adaptive-parameter-policy.md),
and [threading](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/threading.md)
guides. Measured results and provenance live in the benchmark directories and
[`bench/RESULTS.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/bench/RESULTS.md).
