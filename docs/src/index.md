# SDPX.jl

A native Julia semidefinite-programming solver: primal-dual interior-point
method with the HRVW/KSH/M direction and a Mehrotra predictor-corrector, in
arbitrary precision (`Float64`, `MultiFloats.Float64xN`, `Double64`,
`BigFloat`). Built for bootstrap problems whose conditioning exceeds what
`Float64` can carry.

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
directory holds runnable, self-checking versions of this and more — extended
precision, the sparse LP path, independent certificates, JuMP, and the
command-line bridge behind the Mathematica interface. They are executed by the
test suite, so they do not go stale.

## Where to go next

- [Quick start](quickstart.md) — native-array and typed problem workflows.
- [Precision](precision.md) — choosing Float64, Float64x4, or BigFloat.
- [Automatic pipeline](pipeline.md) — presolve, scaling, and kernel selection.
- [Diagnostics and certificates](diagnostics.md) — interpreting a result.
- [API reference](api.md) — the stable-intent entry points.
- [README](https://github.com/yongjunx23-del/SDPX.jl#readme) — features,
  precision guidance, benchmarks policy, and known limitations.
- The repository's
  [`docs/`](https://github.com/yongjunx23-del/SDPX.jl/tree/main/docs)
  directory — full design notes and measured decision records.
