# Convex.jl interface guide

SDPX can solve Convex.jl models through its MathOptInterface optimizer. This
integration does not add Convex as a runtime dependency of SDPX: install and
load Convex only in applications that use the DCP frontend.

## Installation and first solve

```julia
using Pkg
Pkg.add("Convex")
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

Both packages define `solve!`, so import Convex and qualify the call:

```julia
import Convex
import MathOptInterface as MOI
using SDPX

x = Convex.Variable(2)
nonnegative = x >= 0
normalization = sum(x) == 1
problem = Convex.minimize(
    2x[1] + x[2],
    [nonnegative, normalization],
)

solver = MOI.OptimizerWithAttributes(
    SDPX.Optimizer,
    "tolerance" => 1e-8,
    "max_iterations" => 300,
    MOI.TimeLimitSec() => 60.0,
    MOI.NumberOfThreads() => 4,
)
Convex.solve!(problem, solver; silent=true)

@assert Convex.termination_status(problem) == MOI.OPTIMAL
println("objective = ", problem.optval)
println("x = ", Convex.evaluate(x))
println("nonnegative dual = ", nonnegative.dual)
```

Convex sees that SDPX is non-incremental, inserts an MOI cache, canonicalizes
the DCP expressions, applies required cone bridges, and copies the completed
model to SDPX. The numerical solve then uses the same automatic pipeline and
interior-point implementation as the native API.

## Supported model families

The supported Convex subset is determined by the cones produced after DCP
canonicalization:

- affine objectives, equalities, inequalities, intervals, and nonnegativity;
- Euclidean norms and other forms that canonicalize to second-order cones;
- real positive-semidefinite variables and affine PSD constraints;
- minimization, maximization, and feasibility problems built from these forms.

Atoms that require exponential, power, geometric-mean, relative-entropy, or
other unsupported cones are not accepted. Complex Hermitian PSD modeling is
also outside the current real-SDP interface.

Convex represents PSD expressions with `PositiveSemidefiniteConeSquare`. MOI's
standard square bridge enforces missing symmetry equations and passes one
triangle to SDPX. Construct high-precision PSD expressions symbolically
symmetrically when possible; the upstream bridge uses approximate coefficient
tests to avoid redundant symmetry rows. Ordinary `Convex.Semidefinite`
variables are symbolically consistent. A native SDPX upper-triangle model
avoids the full square variable representation and its symmetry/copy overhead.

## Float64x4

The Convex problem numeric type and optimizer type must match. Setting only one
of them produces Float64 functions that cannot be copied into a typed
Float64x4 optimizer.

```julia
import Convex
import MathOptInterface as MOI
using MultiFloats: Float64x4
using SDPX

const T = Float64x4
x = Convex.Variable(2)
problem = Convex.minimize(
    Convex.norm2(x),
    [sum(x) == one(T)];
    numeric_type=T,
)
solver = MOI.OptimizerWithAttributes(
    SDPX.Optimizer{T},
    "tolerance" => T(1e-18),
    MOI.NumberOfThreads() => 4,
)
Convex.solve!(problem, solver; silent=true)
```

## BigFloat

Construct constants and the problem inside the intended precision scope. Pin
the solver working precision when the run is intended to compare identical
MPFR arithmetic rather than exercise adaptive precision selection.

```julia
import Convex
import MathOptInterface as MOI
using SDPX

setprecision(BigFloat, 256) do
    x = Convex.Variable(2)
    X = Convex.Semidefinite(2)
    problem = Convex.minimize(
        Convex.tr(X),
        [X[1, 2] == one(BigFloat)];
        numeric_type=BigFloat,
    )
    solver = MOI.OptimizerWithAttributes(
        SDPX.Optimizer{BigFloat},
        "tolerance" => parse(BigFloat, "1e-24"),
        "precision" => 256,
        "working_precision_policy" => :fixed,
        MOI.NumberOfThreads() => 1,
    )
    Convex.solve!(problem, solver; silent=true)
end
```

General BigFloat models remain serial unless the SDPX structure classifier
selects one of its ownership-safe parallel kernels.

## Results and diagnostics

Use Convex accessors for ordinary application code:

```julia
Convex.termination_status(problem)
Convex.primal_status(problem)
Convex.dual_status(problem)
problem.optval
Convex.evaluate(x)
constraint.dual
```

MOI attributes remain available on `problem.model`:

```julia
MOI.get(problem.model, MOI.SolveTimeSec())
MOI.get(problem.model, MOI.BarrierIterations())
raw = MOI.get(problem.model, MOI.RawSolver())  # SDPX.SDPResult
raw.diagnostics
```

`warmstart=true` is currently ignored because SDPX does not implement
`MOI.VariablePrimalStart`. Modifying and re-solving a Convex problem rebuilds
the completed SDPX representation because the optimizer is non-incremental.

## Native versus Convex modeling

Convex is preferable when readable DCP expressions and rapid model iteration
matter most. Native SDPX modeling is preferable when a large bootstrap code
already owns final block coefficients, when repeated solves reuse structure,
or when model construction and peak memory are significant.

The benchmark in [`bench/convex_frontend`](../bench/convex_frontend/README.md)
compares three identical deterministic problems through both routes. It
reports expression/array construction, Convex canonicalization plus MOI copy,
SDPX core solve, validation, allocations, and peak RSS separately. Compilation
is warmed and reported outside the steady-state samples.
