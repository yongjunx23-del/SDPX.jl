# SDPX.jl

A native Julia semidefinite-programming solver: primal-dual interior-point
method with the HRVW/KSH/M direction and a Mehrotra predictor-corrector, in
arbitrary precision (`Float64`, `MultiFloats.Float64xN`, `Double64`,
`BigFloat`). Built for bootstrap problems whose conditioning exceeds what
`Float64` can carry, with native LP and SOCP paths alongside the PSD engine.

> Status: experimental, pre-1.0. The API may change between minor versions.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

## First solve

```julia
using SDPX

# minimise cᵀx  subject to  Σᵢ xᵢ Aᵢ − C ⪰ 0
A = zeros(2, 2, 2)
A[1, 1, 1] = 1
A[2, 2, 2] = 1
C = [0.0 1.0; 1.0 0.0]
c = [2.0, 3.0]

result = solve(c, [A], [C], Matrix{Float64}(undef, 2, 0), Float64[]; verbosity=0)
result.status      # Optimal
result.pObj        # 4.898979506633980  (exact optimum 2√6)
```

The repository's [`examples/`](https://github.com/yongjunx23-del/SDPX.jl/tree/main/examples)
directory holds runnable, self-checking versions of this and more: extended
precision, the sparse LP path, native SOCP, independent certificates, JuMP,
and the command-line bridge. They are executed by the test suite, so they do
not go stale.

## Where to go next

- [Quick start](quickstart.md) — native-array and typed problem workflows.
- [SOCP and fixed trace](socp.md) — native Lorentz solves and the Q3
  specialization.
- [Precision](precision.md) — choosing Float64, Float64x4, or BigFloat.
- [JuMP and MOI](jump.md) — JuMP modeling through the SDPX optimizer.
- [Convex.jl](convex.md) — DCP modeling through the SDPX MOI optimizer.
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
- [API reference](api.md) — the stable-intent entry points.
- [README](https://github.com/yongjunx23-del/SDPX.jl#readme) — features,
  precision guidance, and known limitations.

The repository's [`docs/`](https://github.com/yongjunx23-del/SDPX.jl/tree/main/docs)
directory keeps the long-term operational and research notes:
[adaptive dense/sparse optimization](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-dense-sparse-optimization.md),
[adaptive parameter policy](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-parameter-policy.md),
[bridge schema](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/bridge-schema.md),
[cluster workflow](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/cluster-workflow.md),
and [threading](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/threading.md).
