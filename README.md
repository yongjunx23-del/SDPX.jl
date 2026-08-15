# SDPX.jl

[![CI](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SDPX.jl is a native Julia solver for linear programs (LPs), second-order cone
programs (SOCPs), and semidefinite programs (SDPs). It combines primal-dual
interior-point methods, automatic formulation planning, sparse and structured
linear algebra, and independent original-coordinate result certificates.

The solver follows the arithmetic of its inputs. It supports `Float64`,
fixed-width `MultiFloats` types, and `BigFloat`, making it suitable
for bootstrap and other ill-conditioned conic models that need more precision
than `Float64` provides.

> **Status: experimental, pre-1.0.** Public interfaces may change between
> minor versions. See [CHANGELOG.md](CHANGELOG.md) when upgrading.

## Highlights

- Native LP, Lorentz-cone SOCP, and block-SDP execution.
- NativeSOC primal-dual IPM with a verified fixed-trace Q3 specialization.
- Dense normal equations, dense augmented KKT, sparse LP, and sparse Schur
  formulations selected before numerical execution.
- Optional MFLA and BFLA providers for extended and arbitrary precision.
- JuMP/MathOptInterface, Convex.jl, raw-array, typed-problem, and command-line
  frontends.
- Presolve, scaling, diagnostics, performance traces, and certificates in the
  model's original coordinates.

Unsupported arithmetic, provider, storage, and formulation combinations fail
during planning instead of silently switching algorithms.

## Installation

SDPX is not yet registered in Julia's General registry:

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

For a local checkout, use `Pkg.develop(path=".")`. Optional frontends and
linear-algebra providers load through Julia package extensions when their
packages are installed.

## Quick start

The native SDP model is

\[
\min_x c^T x
\quad\text{subject to}\quad
\sum_i x_i A_i^{(l)}-C^{(l)}\succeq0,
\qquad B^T x=b.
\]

```julia
using SDPX

# Minimise cᵀx subject to Σᵢ xᵢAᵢ − C ⪰ 0.
A = zeros(2, 2, 2)
A[1, 1, 1] = 1
A[2, 2, 2] = 1
C = [0.0 1.0; 1.0 0.0]
c = [2.0, 3.0]

result = solve(
    c,
    [A],
    [C],
    Matrix{Float64}(undef, 2, 0),
    Float64[];
    verbosity=0,
)

result.status       # Optimal
result.pObj         # 4.898979506633980 (exactly 2√6)
result.termination  # stopping reason and executed-plan diagnostics
```

Call `ingest` once and reuse the resulting typed `SDPProblem` or
`PreparedSolver` for repeated solves. Runnable LP, SOCP, SDP, extended-
precision, JuMP, and certificate examples live in [`examples/`](examples/).

## LP and SOCP

Use `linear_program` or `solve_lp` for scalar inequalities without creating
`1×1` PSD blocks:

```julia
problem = linear_program(c, G, h; Aeq=Aeq, beq=beq, sparse=:auto)
result = solve(problem; tolerance=1e-8, verbosity=0)
```

Use `second_order_program` or `solve_socp` for Lorentz cones. NativeSOC solves
general-dimensional cones directly. Eligible fixed-trace `2×2` PSD blocks are
compiled to Q3 coordinates and solved by the same primal-dual IPM, without a
production PSD lift. See [the SOCP guide](docs/src/socp.md).

## Precision and providers

| Arithmetic | Typical use |
| --- | --- |
| `Float64` | Well-scaled baseline problems |
| `MultiFloats.Float64x2` / `Float64x4` | Fast fixed-width extra precision |
| `BigFloat` | Arbitrary precision and difficult exponent ranges |

Construct every `BigFloat` coefficient inside the intended precision scope:

```julia
setprecision(BigFloat, 256) do
    c = BigFloat[1, 2]
    # Construct the remaining model data here too.
end
```

Increasing `precision_bits` after coefficients were rounded cannot recover
lost digits. SDPX validates precision and mutable-scalar ownership at provider
boundaries. See [precision](docs/src/precision.md) and
[linear-algebra providers](docs/src/providers.md).

## Frontends and command line

`SDPX.Optimizer` is a non-incremental MathOptInterface optimizer. It supports
JuMP affine equalities, scalar and vector linear cones, PSD triangle
constraints, standard second-order cones, and rotated second-order cones.
Rotated cones use an exact sparse linear map into NativeSOC; MOI primal and
dual getters map the result back to rotated coordinates. Convex.jl support is
available through the optional `SDPX.solve_convex!` extension.

The SDPB-style command-line frontend uses the same planning and certification
pipeline:

```bash
julia bin/setup_cli.jl
./bin/sdpx problem.json result.json --precision=840
```

See [JuMP/MOI](docs/src/jump.md), [Convex.jl](docs/src/convex.md), and the
[CLI guide](docs/src/cli.md).

## Documentation and development

The [Documenter manual](https://yongjunx23-del.github.io/SDPX.jl/) covers the
automatic pipeline, preprocessing, sparse execution, parameters, diagnostics,
certificates, providers, and benchmark policy. Operational and research notes
remain in [`docs/`](docs/).

Run the package tests with:

```julia
using Pkg
Pkg.test()
```

Provider integrations and expensive numerical campaigns have separate test
profiles; see [CONTRIBUTING.md](CONTRIBUTING.md) and
[benchmark/README.md](benchmark/README.md). Performance claims must identify
the model, arithmetic, tolerance, hardware, thread configuration, timing
boundary, and certificate result.

## Citation and license

If you use SDPX.jl in academic work, cite:

```text
Yongjun Xu, "SDPX.jl: a native Julia solver for high-precision conic
optimization", version 0.5.0-DEV, https://github.com/yongjunx23-del/SDPX.jl
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff). SDPX began as a
fork of [SDPJSolver.jl](https://github.com/FishboneChiang/SDPJSolver.jl);
copyright, derived components, acknowledgements, and third-party notices are
recorded in [LICENSE](LICENSE), [CONTRIBUTORS.md](CONTRIBUTORS.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
