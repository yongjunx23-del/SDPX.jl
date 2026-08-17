# SDPX.jl

[![CI](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SDPX.jl is a native Julia solver for linear programs (LPs), second-order cone
programs (SOCPs), and semidefinite programs (SDPs). It provides typed modeling,
automatic execution planning, extended-precision arithmetic, and independent
certification in the model's original coordinates.

> **Status:** experimental, pre-1.0. Public interfaces may change between
> minor versions; consult [CHANGELOG.md](CHANGELOG.md) when upgrading.

## Features

- One typed public modeling route for LP, native Lorentz/rotated-Lorentz SOCP,
  and block SDP models.
- `Float64`, optional fixed-width `MultiFloats`, and `BigFloat` arithmetic.
- Dense and sparse execution plans selected before numerical execution.
- Original-coordinate primal/dual residuals and result certificates.
- MathOptInterface/JuMP and command-line integration.
- Unsupported mixed cone families and provider/formulation combinations fail
  during planning instead of silently changing algorithms.

## Installation

SDPX is not yet registered in Julia's General registry:

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

For a local checkout:

```julia
using Pkg
Pkg.develop(path=".")
Pkg.instantiate()
```

SDPX requires Julia 1.10 or newer.

## Quick start

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
outputs = Outputs(
    :all,
    :all,
    :all;
    objectives=true,
    certificate=:summary,
)
result = optimize!(model; settings=settings, outputs=outputs)

status(result)             # :optimal
value(result, x)           # approximately [1, 0]
primal_objective(result)   # approximately 2
certificate(result).valid  # true
```

Use one non-free cone family per model. `Nonnegative`/`Nonpositive` select the
LP route, `LorentzCone`/`RotatedLorentzCone` select NativeSOC, and `PSDCone`
selects the SDP route. `Reals` and `ZeroCone` may be used as auxiliary blocks.

## Examples

The tested examples are:

- [Moment LP](examples/moment_lp.jl)
- [L2 integral SOCP](examples/l2_integral_socp.jl)
- [Quartic integral SDP](examples/quartic_integral_sdp.jl)

Run them from the repository root after instantiating the examples environment:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/moment_lp.jl 17
julia --project=examples examples/l2_integral_socp.jl 16
julia --project=examples examples/quartic_integral_sdp.jl --order 8
```

## Precision

Create `BigFloat` data inside the intended precision scope; increasing the
solver precision after coefficients have already been rounded cannot recover
lost digits.

```julia
setprecision(BigFloat, 256) do
    model = Model(BigFloat; precision_bits=256)
    # Construct variables, coefficients, and constants here.
end
```

Optional linear-algebra providers load through Julia package extensions. See
[Precision](docs/src/precision.md) and [Providers](docs/src/providers.md).

## JuMP and command line

`SDPX.Optimizer` is a non-incremental MathOptInterface optimizer:

```julia
using JuMP, SDPX
model = JuMP.Model(() -> SDPX.Optimizer(sparse=:auto, verbosity=0))
```

For the JSON command-line bridge:

```bash
julia bin/setup_cli.jl
./bin/sdpx problem.json result.json
```

See the [JuMP/MOI](docs/src/jump.md), [CLI](docs/src/cli.md), and
[API](docs/src/api.md) documentation.

## Tests and benchmarks

```julia
using Pkg
Pkg.test()
```

The deterministic benchmark smoke suite is:

```bash
julia --project=. benchmark/runner.jl micro --output=/tmp/sdpx-micro.toml
```

Larger campaigns require validated external inputs. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [benchmark/README.md](benchmark/README.md).

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). SDPX derives from
[SDPJSolver.jl](https://github.com/FishboneChiang/SDPJSolver.jl); copyright,
contributors, and derived-component notices are recorded in
[LICENSE](LICENSE), [CONTRIBUTORS.md](CONTRIBUTORS.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
