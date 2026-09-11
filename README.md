# SDPX.jl

[![CI](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SDPX.jl** is a native Julia solver for conic optimization — linear programs
(LP), second-order cone programs (SOCP), semidefinite programs (SDP), and
exponential / power cone problems. It solves them with a homogeneous
self-dual interior-point method and returns a **mathematical certificate**
that the answer is genuinely optimal, not just "close enough".

It is designed for problems where **precision matters**: the same model can be
solved in standard `Float64`, in extended-precision `Float64x4` (≈209 bits),
or in arbitrary-precision `BigFloat` — with no change to your code except the
element type.

> **Status.** Qualification is ongoing. See the
> [scientific-core roadmap](docs/design/SCIENTIFIC_CORE_ROADMAP.md) for the
> supported scope and known limitations.

---

## Installation

Supported Julia versions: **1.10 (LTS) and 1.12 (audited)**. Julia 1.13 is
currently unsupported — see the pinned issue for the known `factor_receipt`
regression; CI tracks it on a non-blocking canary leg.

```julia
using Pkg
Pkg.add(url = "https://github.com/yongjunx23-del/SDPX.jl")
```

To use extended precision, also add the linear-algebra providers:

```julia
Pkg.add(["MultiFloats", "MultiFloatLinearAlgebra", "BigFloatLinearAlgebra"])
```

---

## Quick start

The workflow is always the same:

1. **`Model(T)`** — create a model in arithmetic type `T` (`Float64`, `Float64x4`, `BigFloat`).
2. **`variable!`** — declare decision variables.
3. **`constraint!`** — add constraints.
4. **`objective!`** — set the objective.
5. **`optimize!`** — solve it.
6. **`value` / `dual` / `objective_value`** — read the answer.

### A linear program

```julia
using SDPX

model = Model(Float64)
x = variable!(model, :x, 2; domain = Nonnegative())

constraint!(model, :budget, x[1] + 2x[2] - 4.0, ZeroCone())
objective!(model, Maximize(), 3x[1] + x[2])

result = optimize!(model)

status(result)             # :optimal
objective_value(result)    # ≈ 12.0
value(result, x)           # primal solution: x[1] ≈ 4, x[2] ≈ 0
```

Typing the result in the REPL shows a readable summary:

```julia
result
```

```
======================== SDPX Conic Optimizer ========================
Status: optimal
Termination: reason=verified_accepted_step stage=original_coordinate_certification
Iterations: 10
Solve time: 12.122804084 s
Primal objective: 11.999999994104824
Dual objective: 11.999999993383403
Duality gap (relative): 6.01e-11
Primal residual: 4.44e-16
Dual residual: 2.66e-9
Certificate: valid (original_coordinates)
======================================================================
```

### A semidefinite program (matrix variable)

```julia
using SDPX

model = Model(Float64)
X = variable!(model, :X, 2, 2; domain = PSDCone())   # X is a 2×2 PSD matrix

constraint!(model, :diag1, X[1,1] - 1.0, ZeroCone()) # X[1,1] = 1
constraint!(model, :diag2, X[2,2] - 1.0, ZeroCone()) # X[2,2] = 1
objective!(model, Minimize(), X[1,2])                # minimize the correlation

result = optimize!(model)

objective_value(result)   # ≈ -1.0
value(result, X)          # 2×2 matrix: [1.0 -1.0; -1.0 1.0]
```

### Extended precision in one line

Only the element type changes:

```julia
using SDPX, MultiFloats, MultiFloatLinearAlgebra

const T = Float64x4            # ≈209 bits of precision
model = Model(T)
x = variable!(model, :x, 2; domain = Nonnegative())
constraint!(model, :budget, x[1] + 2x[2] - one(T), ZeroCone())
objective!(model, Maximize(), 3x[1] + x[2])

result = optimize!(model)
value(result, x)               # a Float64x4 vector
```

---

## Reading a result

Every solve returns a `Result` with a consistent, solver-like interface:

| Query | Returns |
|---|---|
| `status(result)` | `:optimal`, `:primal_infeasible`, `:dual_infeasible`, `:iteration_limit`, … |
| `objective_value(result)` | optimal primal objective |
| `value(result, x)` | primal value of a variable / block |
| `dual(result, con)` | dual value of a constraint |
| `primal_residual(result)` | KKT primal residual |
| `dual_residual(result)` | KKT dual residual |
| `relative_gap(result)` | relative duality gap |
| `iterations(result)` | number of interior-point iterations |
| `solve_time(result)` | wall time of the solve (seconds) |
| `certificate(result)` | the optimality certificate (`.valid`) |
| `is_optimal(result)` | `true` when optimal **and** certified |

`is_optimal(result) === true` is the check you should use to trust a result.

---

## JuMP / MathOptInterface

SDPX also ships a full MathOptInterface wrapper, so it works with JuMP:

```julia
using JuMP, SDPX

model = JuMP.Model(SDPX.Optimizer)
set_attribute(model, "threads", 4)

@variable(model, x >= 0)
@variable(model, y >= 0)
@constraint(model, x + 2y == 4)
@objective(model, Maximize, x + y)

optimize!(model)
objective_value(model)
```

---

## What makes it different

- **Certified answers.** Results carry an original-coordinate certificate;
  `certificate(result).valid` confirms the primal and dual solutions satisfy
  the KKT conditions in the original problem.
- **One engine, many cones.** LP, SOCP, SDP, exponential and power cones share
  a single homogeneous self-dual interior-point solver, so mixed models "just
  work".
- **Precision is a type.** `Float64`, `Float64x4`, and `BigFloat` are all
  first-class. Raise precision when a result is not accurate enough — the
  model code stays the same.
- **Fast for scans.** A built-in persistent process pool amortizes startup and
  JIT across many independent solves (see
  [`benchmark/optimization/PERSISTENT_POOL.md`](benchmark/optimization/PERSISTENT_POOL.md)).

---

## Documentation

More detail lives in `docs/`:

- [Architecture & HSD formulation](docs/src/architecture.md)
- [Linear-algebra providers](docs/src/providers.md)
- [Multi-precision guide](docs/src/precision.md)
- [Cluster / PBS execution](docs/src/cluster-workflow.md)
- [Scientific-core roadmap](docs/design/SCIENTIFIC_CORE_ROADMAP.md)

## License

SDPX.jl is released under the MIT License.
