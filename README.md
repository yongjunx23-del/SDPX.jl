# SDPX.jl

[![CI](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SDPX.jl is a native Julia solver for linear programs (LPs), second-order cone
programs (SOCPs), and semidefinite programs (SDPs). It provides a typed
modeling API, `Float64` and extended-precision arithmetic, parallel numerical
kernels, and original-coordinate result certificates.

> **Status:** experimental, pre-1.0. Public interfaces may change between
> minor versions; consult [CHANGELOG.md](CHANGELOG.md) when upgrading.

## Installation

SDPX requires Julia 1.10 or newer. Install the current development version with

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

## SDP quick start: a 2×2 quartic-moment truncation

Consider the normalized zero-dimensional quartic integral with density

```math
\rho(x) \propto \exp\!\left(-\frac{x^2}{2}-\frac{g x^4}{4}\right),
\qquad W_k=\langle x^k\rangle .
```

At `g = 1`, normalization and integration by parts give

```math
W_0=1, \qquad W_0-W_2-W_4=0.
```

The first even moment matrix must be positive semidefinite:

```math
H_1=\begin{bmatrix}W_0&W_2\\W_2&W_4\end{bmatrix}\succeq0.
```

Maximizing `W₂` under these constraints gives a certified upper bound. The
matrix entries and linear equations can be written directly as affine
expressions:

```julia
using SDPX
using LinearAlgebra

model = Model(Float64)
w = variable!(model, :w, 3; domain=Reals())  # [W0, W2, W4]

# Linear equalities are expressions constrained to the zero cone.
constraint!(model, :normalization, w[1] - 1, ZeroCone())
constraint!(model, :quartic_recurrence, w[1] - w[2] - w[3], ZeroCone())

# A symmetric matrix of affine expressions is one PSD constraint.
H = [w[1] w[2];
     w[2] w[3]]
constraint!(model, :moment_matrix, H, PSDCone())

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

moments = value(result, w)
H_value = [moments[1] moments[2]; moments[2] moments[3]]
H_spectrum = eigvals(Symmetric(H_value))
cert = certificate(result)

status(result)             # :optimal
moments                    # approximately [1, 0.618034, 0.381966]
primal_objective(result)   # approximately 0.618034
H_spectrum                 # nonnegative up to solver tolerance
cert.valid                 # true
cert.primal_residual
cert.dual_residual
cert.relative_gap
```

`ZeroCone()` enters a linear equality. `PSDCone()` accepts a square symmetric
matrix whose entries may contain variables, constants, and affine
combinations. After solving, reconstruct the numerical matrix from
`value(result, w)` and use `eigvals(Symmetric(...))` to inspect its spectrum.

## Worked examples

The repository contains three self-checking examples. Set them up once from
the repository root:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

### SDP: quartic integral bounds

[`examples/quartic_integral_sdp.jl`](examples/quartic_integral_sdp.jl) extends
the quick start to moments `W₀,…,W₂ₙ`. It imposes every available recurrence

```math
(2k+1)W_{2k}-W_{2k+2}-gW_{2k+4}=0
```

and the parity-aware Hankel matrix

```math
H_{ij}=\begin{cases}W_{i+j},&i+j\text{ even},\\0,&i+j\text{ odd}.
\end{cases}
```

Separate minimization and maximization models produce lower and upper bounds
on `W₂`. The example reconstructs `H`, computes its eigenvalues, checks the
original-coordinate certificate, and prints the smallest eigenvalue and
primal residual.

```bash
julia --project=examples examples/quartic_integral_sdp.jl \
  --g 1 --order 8 --bound both --arithmetic f64
```

Typical `Float64`, `g=1`, order-8 output brackets the reference value
`W₂ ≈ 0.46791991697`:

```text
lower W2 ≈ 0.46781157267
upper W2 ≈ 0.46794579859
```

### LP: finite-grid moment bounds

[`examples/moment_lp.jl`](examples/moment_lp.jl) puts a probability vector
`p` on a uniform grid `tᵢ∈[0,1]`:

```math
p_i\ge0,\qquad \sum_i p_i=1,\qquad \sum_i t_i p_i=\frac12.
```

It minimizes and maximizes the second moment `Σ tᵢ²pᵢ`. The exact finite-grid
bounds are `1/4` and `1/2`; the script checks both, the mass and mean
residuals, and the certificate.

```julia
model = Model(Float64)
p = variable!(model, :p, n; domain=Nonnegative())
constraint!(model, :mass, sum(p) - 1, ZeroCone())
constraint!(model, :mean, sum(grid[i] * p[i] for i in 1:n) - 0.5, ZeroCone())
objective!(model, Minimize(), sum(grid[i]^2 * p[i] for i in 1:n))
```

Run the full example with

```bash
julia --project=examples examples/moment_lp.jl 17
```

### SOCP: an L2 integral bound

[`examples/l2_integral_socp.jl`](examples/l2_integral_socp.jl) uses a vector
`u` with zero discrete mean and Euclidean norm at most one:

```math
\sum_i u_i=0,\qquad (1,u_1,\ldots,u_n)\in\mathcal Q_{n+1}.
```

The Lorentz constraint is entered as a tuple; no matrix lift is needed:

```julia
model = Model(Float64)
u = variable!(model, :u, n; domain=Reals())
constraint!(model, :mean_zero, (1 / sqrt(n)) * sum(u), ZeroCone())
constraint!(model, :l2_ball, tuple(1.0, u...), LorentzCone())
objective!(model, Maximize(), sum(coefficients[i] * u[i] for i in 1:n))
```

The script compares the result with the analytic finite-grid value, then
prints the equality residual, Lorentz-cone margin, and certificate result:

```bash
julia --project=examples examples/l2_integral_socp.jl 16
```

See [examples/README.md](examples/README.md) for the equations, output fields,
and additional commands.

## Multiple precision

The SDP example supports `Float64`, `Float64x2`, `Float64x4`, and `BigFloat`:

```bash
julia --project=examples examples/quartic_integral_sdp.jl --order 8 --arithmetic f64x4
julia --project=examples examples/quartic_integral_sdp.jl --order 8 --arithmetic bf256
julia --project=examples examples/quartic_integral_sdp.jl --order 8 --arithmetic bf512
```

When constructing a `BigFloat` model yourself, create every coefficient and
constant inside the intended precision scope:

```julia
setprecision(BigFloat, 256) do
    model = Model(BigFloat; precision_bits=256)
    one_big = BigFloat(1)          # created at 256-bit precision
    # Build variables, matrices, equalities, and the objective here.
end
```

Increasing solver precision after coefficients were rounded cannot recover
lost digits. `GenericLinearAlgebra` is included in the examples environment so
`eigvals(Symmetric(bigfloat_matrix))` keeps the matrix arithmetic instead of
projecting it to `Float64`.

## Parallel execution

Start Julia with enough worker threads, then give the solver no more threads
than Julia owns:

```bash
julia -t 8 --project=examples examples/quartic_integral_sdp.jl \
  --order 8 --threads 8 --blas-threads 1
```

In direct API use:

```julia
settings = Settings(
    model;
    algorithm=:sdp,
    limits=Limits(iterations=250, time=120.0, threads=8),
    blas_threads=1,
)
```

`Limits(..., threads=8)` controls SDPX worker parallelism. `blas_threads`
controls the linear-algebra provider separately; keeping it at 1 avoids nested
oversubscription when SDPX already uses several Julia threads. Inspect the
resolved choice with `execution_plan(result)` and collect timing data with
`performance_trace(result)` when the corresponding output was requested.

## JuMP, command line, and tests

`SDPX.Optimizer` is a non-incremental MathOptInterface optimizer:

```julia
using JuMP, SDPX
model = JuMP.Model(() -> SDPX.Optimizer(sparse=:auto, verbosity=0))
```

The JSON command-line bridge is available through

```bash
julia bin/setup_cli.jl
./bin/sdpx problem.json result.json
```

Run the package tests with

```julia
using Pkg
Pkg.test()
```

More detail is available in the [API](docs/src/api.md),
[JuMP/MOI](docs/src/jump.md), [precision](docs/src/precision.md),
[providers](docs/src/providers.md), and [CLI](docs/src/cli.md) documentation.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). SDPX derives from
[SDPJSolver.jl](https://github.com/FishboneChiang/SDPJSolver.jl); copyright,
contributors, and derived-component notices are recorded in
[LICENSE](LICENSE), [CONTRIBUTORS.md](CONTRIBUTORS.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
