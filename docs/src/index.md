# SDPX.jl

A native Julia semidefinite-programming solver: primal-dual interior-point
method with the HRVW/KSH/M direction and a Mehrotra predictor-corrector, in
arbitrary precision (`Float64`, `MultiFloats.Float64xN`,
`BigFloat`). Built for bootstrap problems whose conditioning exceeds what
`Float64` can carry, with native LP and SOCP paths alongside the PSD engine.

> Status: experimental, pre-1.0. The API may change between minor versions.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

## First solve

The public route is a typed `Model`: declare product-cone variables, add affine
cone constraints, set one objective, then call `optimize!`. This small LP uses
only exported names and returns a typed result whose fields are read through
accessors:

```julia
using SDPX

model = Model(Float64)
x = variable!(model, :x, 2; domain=Nonnegative())
constraint!(model, :mass, x[1] + x[2] - 1, ZeroCone())
objective!(model, Minimize(), 2 * x[1] + 3 * x[2])

settings = Settings(
    model;
    algorithm=:lp,
    limits=Limits(iterations=200, time=60.0, threads=1),
    verbosity=0,
)
outputs = Outputs(:all, :all, :all; objectives=true, certificate=:summary)
result = optimize!(model; settings=settings, outputs=outputs)

status(result)             # :optimal
value(result, x)           # approximately [1, 0]
primal_objective(result)   # approximately 2
certificate(result).valid  # true when the independent check passes
```

The three runnable, self-checking examples are deliberately the only examples
promoted by the v0.5 documentation:

- [`examples/quartic_integral_sdp.jl`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/examples/quartic_integral_sdp.jl)
  — a native PSD moment bound with Float64 and BigFloat modes.
- [`examples/moment_lp.jl`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/examples/moment_lp.jl)
  — a finite-grid moment LP and objective-sense checks.
- [`examples/l2_integral_socp.jl`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/examples/l2_integral_socp.jl)
  — a native Lorentz-cone integral bound.

## Where to go next

- [Quick start](quickstart.md) — the typed `Model` builder and result accessors.
- [SOCP and fixed trace](socp.md) — native Lorentz solves and the Q3
  specialization.
- [Precision](precision.md) — choosing Float64, Float64x4, or BigFloat.
- [JuMP and MOI](jump.md) — JuMP modeling through the SDPX optimizer.
- [Automatic pipeline](pipeline.md) — classification, presolve, scaling, and
  planning.
- [Preprocessing](preprocessing.md) — conservative structural reductions.
- [Sparse execution](sparse-execution.md) — frozen-CSC Schur and LP paths.
- [Parameters](parameters.md) — the common and expert option surface.
- [Diagnostics and certificates](diagnostics.md) — interpreting a result.
- [Architecture](architecture.md) — frontend, midend, planning, and KKT
  formulations.
- [Linear-algebra providers](providers.md) — the replaceable LA seam.
- [Benchmarks and evidence](benchmarks.md) — measured results and policy.
- [API reference](api.md) — qualified implementation and metadata reference;
  the public quickstart is the typed `Model` route above.
- [README](https://github.com/yongjunx23-del/SDPX.jl#readme) — features,
  precision guidance, and known limitations.

The repository's [`docs/`](https://github.com/yongjunx23-del/SDPX.jl/tree/main/docs)
directory keeps the long-term operational and research notes:
[adaptive dense/sparse optimization](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-dense-sparse-optimization.md),
[adaptive parameter policy](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-parameter-policy.md),
[bridge schema](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/bridge-schema.md),
[cluster workflow](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/cluster-workflow.md),
and [threading](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/threading.md).
