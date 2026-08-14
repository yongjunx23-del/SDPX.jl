# SDPX.jl

[![CI](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SDPX.jl is a native Julia solver for linear programs (LPs), second-order cone
programs (SOCPs), and semidefinite programs (SDPs). It uses a primal-dual
interior-point method with the HRVW/KSH/M search direction and a Mehrotra
predictor-corrector. The typed API follows the arithmetic of its inputs,
including `Float64`, fixed-width `MultiFloats` types, `Double64`, and
`BigFloat`.

The package is designed for bootstrap calculations whose conditioning can be
too difficult for `Float64`, while retaining fast paths for ordinary sparse
and dense models. It also includes preprocessing, sparse execution,
diagnostics, independent result certificates, and JuMP/MathOptInterface and
Convex.jl adapters.

> **Status: experimental, pre-1.0.** The public API is usable but may change
> between minor versions. Check [CHANGELOG.md](CHANGELOG.md) before upgrading.

## Installation

SDPX is not yet registered in the Julia General registry. Install the current
repository directly:

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

For a checkout under development, use `Pkg.develop(path=".")`. Optional
frontends and arithmetic backends are loaded as Julia package extensions when
their packages are present.

## First solve

The native model is

\[
\min_x c^T x \quad\text{subject to}\quad
\sum_i x_i A_i^{(l)}-C^{(l)}\succeq0,
\qquad B^T x=b.
\]

`A` is a vector of three-dimensional block arrays, `C` a vector of symmetric
constant blocks, and `B,b` optional affine equalities. A minimal solve is:

```julia
using SDPX

# minimise cᵀx subject to Σᵢ xᵢ Aᵢ − C ⪰ 0
A = zeros(2, 2, 2)
A[1, 1, 1] = 1
A[2, 2, 2] = 1
C = [0.0 1.0; 1.0 0.0]
c = [2.0, 3.0]

result = solve(c, [A], [C], Matrix{Float64}(undef, 2, 0), Float64[];
               verbosity=0)

result.status       # Optimal
result.pObj         # 4.898979506633980 (the exact value is 2√6)
result.termination  # stopping reason and convergence diagnostics
```

The raw-array call infers the element type from the data. For repeated solves,
call `ingest` once and reuse the resulting typed `SDPProblem` or
`PreparedSolver`.

The complete runnable set is in [`examples/`](examples/). It is exercised by
the test suite; setup and commands are in
[`examples/README.md`](examples/README.md).

## LP, SOCP, and SDP support

- **LP:** `linear_program` and `solve_lp` build and solve native LPs, with a
  dedicated scalar Mehrotra path and sparse normal-equation execution for
  eligible equality-free systems.
- **SOCP:** `second_order_program` and `solve_socp` build and solve
  Lorentz-cone models directly, including general-dimensional native SOC.
- **SDP:** the typed block-SDP API and the JuMP/MOI optimizer solve PSD
  problems, including sparse Schur paths for eligible models.

All three cone families share one automatic workflow: classify, presolve,
scale, formulate, plan, solve, and certify in the original coordinates.
Providers and linear-algebra backends are planned before numerical execution;
unsupported provider/formulation pairs fail closed during planning rather
than silently switching routes.

## Native SOCP and fixed-trace Q3

`solve_socp` and pure-SOC MOI models run in Lorentz coordinates. Production
general-dimensional SOC uses a native Nesterov–Todd path; SDPX does not
construct PSD matrices for these solves. The PSD-arrow compilation remains
only as a correctness reference in the test helpers.

`FixedTraceQ3Execution` is a verified specialization for the common bootstrap
case where a `2×2` PSD block has a fixed positive trace. Because

\[
\begin{bmatrix}a&b\\b&c\end{bmatrix}\succeq0,\quad a+c=\tau
\quad\Longleftrightarrow\quad
(\tau,\ a-c,\ 2b)\in\mathcal Q_3,
\]

the block is represented as a three-dimensional Lorentz cone with two local
variables. The compact specialization shares the native SOC residuals,
controller, providers, result type, and certificate with the general path.
Under `specialization=:auto`, it is selected whenever the exact structural
reduction verifies (positive fixed head, two nonsingular local tail
variables, and no shared variables); otherwise the model uses general Lorentz.

`SDPX.Experimental.analyze_fixed_trace(problem)` conservatively detects direct
or equality-implied fixed traces without changing the model. See
[`docs/src/socp.md`](docs/src/socp.md) for the full scope, Q3 direction
options, and executed-backend diagnostics.

## Precision

SDPX follows the element type of the input arrays; there is no process-global
arithmetic mode in the typed API.

| Type | Typical use |
| --- | --- |
| `Float64` | Fast baseline for well-scaled models |
| `MultiFloats.Float64x2`/`Float64x4` | Fixed-width extra precision with threaded kernels |
| `DoubleFloats.Double64` | Fixed-width alternative backend |
| `BigFloat` | Arbitrary precision or difficult exponent ranges |

Construct `BigFloat` data inside the desired precision scope:

```julia
setprecision(BigFloat, 256) do
    c = BigFloat[1, 2]
    # Construct A, C, B, and b here as BigFloat values too.
end
```

Raising `precision_bits` after coefficients have been rounded cannot recreate
lost digits. BigFloat uses conservative staged working precision by default
and retries at the requested width if original-coordinate certification
fails. General native BigFloat kernels are serial because MPFR values are
mutable; ownership-safe independent blocks and exact local arrow phases may
use requested workers. See [`docs/src/precision.md`](docs/src/precision.md).

## Command-line frontend

A small SDPB-style CLI exposes the same automatic policy. After one-time
setup:

```bash
julia bin/setup_cli.jl
./bin/sdpx problem.json
```

A high-precision run can be written in the familiar form:

```bash
./bin/sdpx problem.json result.json \
  --precision=840 \
  --dualityGapThreshold=1e-80 \
  --primalErrorThreshold=1e-80 \
  --dualErrorThreshold=1e-80
```

An integer precision is a BigFloat **bit count**. The result JSON records
both `resolved_options` and the executed automatic `plan`, so `auto` is
inspectable. See [`docs/src/cli.md`](docs/src/cli.md).

## JuMP and Convex.jl

`SDPX.Optimizer` is a non-incremental MathOptInterface optimizer. JuMP models
can use affine equalities, scalar bounds, PSD triangle constraints, and
second-order cones; use `GenericModel{T}` with `SDPX.Optimizer{T}` for
extended coefficient types. Convex.jl support is an optional extension via
`SDPX.solve_convex!`, including typed extended-precision models and a compact
upper-triangle PSD-variable factory. See
[`docs/src/jump.md`](docs/src/jump.md) and
[`docs/src/convex.md`](docs/src/convex.md).

## Documentation

The [Documenter manual](https://yongjunx23-del.github.io/SDPX.jl/) covers the
quick start, CLI, precision, JuMP/MOI, Convex, SOCP, pipeline, preprocessing,
sparse execution, parameters, diagnostics, architecture, providers, and
benchmark policy. Long-term operational and research notes remain readable
directly in [`docs/`](docs/):

- [adaptive dense/sparse optimization](docs/adaptive-dense-sparse-optimization.md)
- [adaptive parameter policy](docs/adaptive-parameter-policy.md)
- [bridge schema](docs/bridge-schema.md)
- [cluster workflow](docs/cluster-workflow.md)
- [threading](docs/threading.md)

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing code or documentation.
Run the package tests and the small benchmark tier in the appropriate
environment; cluster validation is preferred for expensive numerical
campaigns. Performance claims must state the model, arithmetic, tolerances,
hardware, thread configuration, warm-up, timing boundary, and repeated-run
statistics, and the independent certificate boundary must be preserved.

## Citation

If you use SDPX.jl in academic work, cite it as:

```text
Yongjun Xu, "SDPX.jl: an arbitrary-precision primal-dual interior-point SDP
solver in Julia", version 0.5.0-DEV, https://github.com/yongjunx23-del/SDPX.jl
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## Acknowledgements and license

SDPX began as a fork of
[SDPJSolver.jl](https://github.com/FishboneChiang/SDPJSolver.jl); upstream
copyright and derived components are recorded in [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The HRVW/KSH/M direction is
from the SDPA tradition, the KKT block-elimination design draws on SDPB, and
Clarabel.jl is a reference for several equilibration and refinement ideas.
Contributors and acknowledgements are listed in
[CONTRIBUTORS.md](CONTRIBUTORS.md).
