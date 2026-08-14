# Convex.jl

SDPX can solve Convex.jl models through its MathOptInterface optimizer. Convex
is an optional package extension rather than a mandatory runtime dependency.
Loading both packages enables a compact solver factory, one-call solve helper,
and a native upper-triangle PSD-variable factory.

## Installation and first solve

```julia
using Pkg
Pkg.add("Convex")
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

The high-level SDPX helper avoids the `solve!` name shared by both packages:

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

SDPX.solve_convex!(
    problem;
    tolerance=1e-8,
    maximum_iterations=300,
    time_limit=60.0,
    threads=4,
)

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

## Native triangle PSD variables

Use `SDPX.convex_semidefinite(n)` for new models. It owns only `n(n+1)/2`
scalar variables, constructs the symmetric Convex expression by reusing each
off-diagonal entry, and attaches the PSD constraint automatically:

```julia
X = SDPX.convex_semidefinite(20)  # triangle is the default
problem = Convex.minimize(Convex.tr(X), [X[1, 2] == 1])
SDPX.solve_convex!(problem; tolerance=1e-8, threads=4)
```

Existing `Convex.Semidefinite(n)` models remain supported. The explicit
compatibility spelling is:

```julia
X = SDPX.convex_semidefinite(20; representation=:square)
```

The square representation owns `n^2` variables. MOI's standard bridge then
adds symmetry equalities and converts it to a triangle cone. The native helper
therefore reduces source variables by nearly two and avoids those symmetry
rows without changing the resulting symmetric matrix.

When the PSD dual is needed, request the attached constraint explicitly:

```julia
psd = SDPX.convex_semidefinite(20; return_metadata=true)
X = psd.matrix
# After solving: psd.constraint.dual
```

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
SDPX.solve_convex!(
    problem;
    numeric_type=T,
    tolerance=T(1e-18),
    threads=4,
)
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
    X = SDPX.convex_semidefinite(2)
    problem = Convex.minimize(
        Convex.tr(X),
        [X[1, 2] == one(BigFloat)];
        numeric_type=BigFloat,
    )
    SDPX.solve_convex!(
        problem;
        numeric_type=BigFloat,
        tolerance=parse(BigFloat, "1e-24"),
        precision_bits=256,
        working_precision_policy=:fixed,
        threads=1,
    )
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

The benchmark in [`bench/convex_frontend/README.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/bench/convex_frontend/README.md)
compares identical deterministic problems through both routes and reports
expression/array construction, canonicalization plus MOI copy, SDPX core
solve, validation, allocations, and peak RSS separately.
