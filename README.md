# SDPX.jl

[![CI](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SDPX.jl is a high-performance Julia product-cone optimizer for Linear Programming (LP), Second-Order Cone Programming (SOCP), Rotated Second-Order Cone (RSOC), Semidefinite Programming (SDP), Exponential Cone (EXP), and Power Cone models.

A single unified Homogeneous Self-Dual (HSD) predictor-corrector engine powers all formulations with original-coordinate mathematical certificates, extended-precision arithmetic, column-owned parallel Schur assembly, and zero hot-loop allocations.

## Features

- **Unified Conic Engine**: LP, SOCP, RSOC, SDP, EXP, Power, and arbitrary mixed product cones run through a single homogeneous interior-point state machine ($Ax + s - b\tau = 0$, $A^\top y + c\tau = 0$, $-c^\top x - b^\top y + \kappa = 0$).
- **Multi-Precision Support**:
  - `Float64` (standard hardware precision, BLAS/LAPACK accelerated)
  - `Float64x2` (Double64 ~106 bits) and `Float64x4` (QuadFloat ~212 bits) via [MultiFloats.jl](https://github.com/JuliaArrays/MultiFloats.jl) and [MultiFloatLinearAlgebra.jl](https://github.com/yongjunx23-del/MultiFloatLinearAlgebra.jl)
  - `BigFloat` (arbitrary-precision MPFR) via [BigFloatLinearAlgebra.jl](https://github.com/yongjunx23-del/BigFloatLinearAlgebra.jl)
- **High-Performance Architecture**:
  - Compact $(nr+1)$-dimensional Schur border for tall symmetric systems
  - Column-owned thread partition with zero atomic locks and zero per-thread matrix copies
  - Reusable provider-native factor caches for true allocation-free hot-loop iterations
  - Incremental PackageCompiler sysimage eliminating ~120s of Julia JIT latency
- **Strict Verification**:
  - Independent original-coordinate verification (`verify_optimal!`, Farkas primal/dual rays)
  - Full-canonical recovery checks, finite gates, and 5-equation Newton residual replay
  - No silent precision downgrades, uncertified rank reductions, or tolerance widening

## Installation

SDPX requires Julia 1.10 or newer (Julia 1.12 recommended).

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

For multi-precision acceleration, install the complementary linear algebra providers:
```julia
Pkg.add(["MultiFloats", "MultiFloatLinearAlgebra", "BigFloatLinearAlgebra"])
```

## Quick Start: SDP Model

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
    limits=Limits(iterations=200, time=60.0, threads=4),
    verbosity=0,
)
outputs = Outputs(
    :all, :all, :all;
    objectives=true,
    certificate=:summary,
)
result = optimize!(model; settings=settings, outputs=outputs)

println("Status: ", status(result))                      # :optimal
println("Primal objective: ", primal_objective(result))  # ~0.618034
println("Certificate valid: ", certificate(result).valid) # true
```

## Quick Start: Multi-Precision SOCP (`Float64x4`)

```julia
using SDPX, MultiFloats, MultiFloatLinearAlgebra

const T = Float64x4
model = Model(T)
x = variable!(model, :x, 3; domain=Reals())

# Second-order cone: ||[x[2], x[3]]||_2 <= x[1]
constraint!(model, :soc, Any[x[1], x[2], x[3]], LorentzCone())
constraint!(model, :bound, x[1] - one(T), ZeroCone())
objective!(model, Minimize(), -x[2])

settings = Settings(
    model;
    tolerances=Tolerances{T}(primal=T(1e-24), dual=T(1e-24), gap=T(1e-24)),
    limits=Limits(iterations=100, time=60.0, threads=4),
)
result = optimize!(model; settings=settings)
println("Solution in Float64x4: ", value(result, x))
```

## JuMP and MathOptInterface (MOI)

SDPX provides a full MOI wrapper for seamless integration with JuMP:

```julia
using JuMP, SDPX

model = JuMP.Model(SDPX.Optimizer)
set_attribute(model, "threads", 4)
set_attribute(model, "tolerances.primal", 1e-8)

@variable(model, x >= 0)
@variable(model, y >= 0)
@constraint(model, x + 2y == 4)
@objective(model, Maximize, x + y)

optimize!(model)
println("Objective: ", objective_value(model))
```

## Performance & Sysimage Build

To eliminate JIT compilation overhead for high-precision workflows (e.g. large-scale bootstrap problems), build a precompiled performance sysimage:

```bash
julia --project=. scripts/build_performance_sysimage.jl sysimages/sdpx-performance.so
```

Then launch Julia with the sysimage:
```bash
julia -J sysimages/sdpx-performance.so --project=. your_script.jl
```

## Documentation

Detailed guides and references are located in `docs/`:
- [Architecture & HSD Formulation](docs/src/architecture.md)
- [Linear-Algebra Providers](docs/src/providers.md)
- [Multi-Precision Guide](docs/src/precision.md)
- [Cluster / PBS Execution Guide](docs/src/cluster-workflow.md)
- [General Benchmark Suite V2](docs/src/benchmarks.md)

## License

SDPX.jl is released under the MIT License.
