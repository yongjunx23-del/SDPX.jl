# Quick start

## Install

Until SDPX is registered, install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

## Solve from arrays

SDPX represents

```math
\min_x c^\mathsf{T}x
\quad\text{subject to}\quad
\sum_i x_i A_i^{(l)}-C_l \succeq 0,\qquad B^\mathsf{T}x=b.
```

```julia
using SDPX

A = zeros(2, 2, 2)
A[1, 1, 1] = 1
A[2, 2, 2] = 1
C = [0.0 1.0; 1.0 0.0]
c = [2.0, 3.0]
B = zeros(2, 0)
b = Float64[]

result = solve(
    c,
    [A],
    [C],
    B,
    b;
    tolerance=1e-8,
    threads=1,
    verbosity=0,
)

@assert result.status == Optimal
summary = solve_summary(
    ingest(c, [A], [C], B, b; verbosity=0),
    result,
)
```

## LP and SOCP

Native LP models avoid manufacturing one `1×1` PSD block per inequality:

```julia
problem = SDPX.linear_program(
    c,
    G,
    h;
    Aeq=Aeq,
    beq=beq,
    sparse=:auto,
)
result = SDPX.solve(problem; tolerance=1e-8, threads=4)
```

This represents `G*x >= h` and `Aeq*x = beq`; `SDPX.solve_lp(c, G, h; ...)` is
the one-call form. Compact Lorentz-cone models use `second_order_program` and
`solve_socp`; see [socp.md](socp.md).

## Reuse an ingested problem

Ingestion analyzes sparsity and creates the internal coefficient layout. Reuse
the returned problem when solving the same model repeatedly:

```julia
problem = ingest(c, [A], [C], B, b; sparse=:auto, verbosity=0)
result = solve(problem; tolerance=1e-8, threads=4, verbosity=0)
```

For expert control, use ASCII aliases or the original Unicode fields:

```julia
options = SolverOptions(
    Float64;
    tolerance=1e-9,
    maximum_iterations=300,
    time_limit=120.0,
    beta=0.1,
    gamma=0.9,
    verbosity=0,
)
result = solve!(problem, options)
```

The examples directory contains self-checking LP, SOCP, SDP, extended-
precision, and JuMP programs and is exercised by the test suite.
