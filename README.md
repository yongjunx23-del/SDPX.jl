# SDPX.jl

[![CI](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SDPX.jl is a native Julia solver for linear programs, second-order cone
programs, and semidefinite programs. It provides a typed modeling API,
extended-precision arithmetic, parallel numerical kernels, and
original-coordinate result certificates.

> SDPX is experimental and pre-1.0. See [CHANGELOG.md](CHANGELOG.md) when
> upgrading between minor versions.

## Installation

SDPX requires Julia 1.10 or newer.

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

## Minimal SDP quick start

This example solves

```math
\begin{aligned}
\max_{W_0,W_2,W_4}\quad & W_2\\
\text{s.t.}\quad
&W_0=1,\\
&W_0-W_2-W_4=0,\\
&
\begin{bmatrix}
W_0&W_2\\
W_2&W_4
\end{bmatrix}\succeq0.
\end{aligned}
```

```julia
using SDPX

model = Model(Float64)
w = variable!(model, :w, 3; domain=Reals()) # [W0, W2, W4]

constraint!(model, :normalization, w[1] - 1, ZeroCone())
constraint!(model, :recurrence, w[1] - w[2] - w[3], ZeroCone())
constraint!(
    model,
    :moment_matrix,
    [w[1] w[2]; w[2] w[3]],
    PSDCone(),
)
objective!(model, Maximize(), w[2])

settings = Settings(
    model;
    algorithm=:sdp,
    limits=Limits(iterations=200, time=60.0, threads=1),
    verbosity=0,
)
outputs = Outputs(
    :all,
    :all,
    :all;
    objectives=true,
    certificate=:summary,
)
result = optimize!(model; settings=settings, outputs=outputs)

status(result)             # :optimal
value(result, w)           # approximately [1, 0.618034, 0.381966]
primal_objective(result)   # approximately 0.618034
certificate(result).valid  # true
```

The mathematical origin of this SDP, its higher-order bootstrap extension,
alternative LP/SOCP formulations, precision choices, providers, and result
certification are developed in the
[worked examples](examples/README.md). See the
[documentation](docs/src/index.md) for the complete user guide.

For extended precision, the worked example recommends
[MultiFloatLinearAlgebra](https://github.com/yongjunx23-del/MultiFloatLinearAlgebra.jl)
for `Float64x2`/`Float64x4` and
[BigFloatLinearAlgebra](https://github.com/yongjunx23-del/BigFloatLinearAlgebra.jl)
for `BigFloat`.
