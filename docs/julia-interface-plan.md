# Julia Interface Roadmap

## Implementation status

The first non-incremental MathOptInterface/JuMP wrapper is implemented in
`src/moi_wrapper.jl`. It supports affine objectives, equalities, PSD triangle
constraints, scaled PSD coordinates, `Float64`, and typed extended-precision
models such as `Float64x4`. See the
[user guide](julia-interface.md) for executable examples and current
limitations.

The remaining items in this document are longer-term API and ecosystem work:
a native problem builder, repeated-solve sessions, broader MOI forms and
bridges, infeasibility certificates, and a fuller `MOI.Test` matrix.

SDPX should expose two layers:

1. A compact native API for bootstrap codes that already have block SDP data.
2. A MathOptInterface optimizer so the same solver works naturally with JuMP,
   Convex.jl-compatible bridges, and the Julia optimization ecosystem.

Do not create a second modeling language with custom macros. Let JuMP handle
general modeling and keep the native API optimized for block-sparse assembly.

## Native API

### Immutable finalized problem

Keep `SDPProblem{T}` as the solve-time object, but replace positional,
shape-sensitive construction with a builder:

```julia
builder = SDPX.ProblemBuilder{Float64x4}()

g = SDPX.add_variables!(builder, 4; kind=:global)
u = SDPX.add_variables!(builder, 60; kind=:block_local)

SDPX.add_psd_block!(
    builder,
    2;
    constant=C_l,
    coefficients=[
        g[1] => A_g1,
        u[1] => A_u1,
    ],
)

SDPX.add_equalities!(builder, B, b)
SDPX.set_objective!(builder, c; sense=:min)

problem = SDPX.finalize(builder)
result = SDPX.solve(problem; sparse=:auto, tolerance=1e-7)
```

`finalize` should validate dimensions and symmetry, build the incidence graph,
choose packed storage, precompute the KKT pattern, and return an immutable
object.

### Clear configuration

Retain the typed low-level form:

```julia
options = SDPX.SolverOptions{Float64x4}(
    sparse=:auto,
    equilibrate=false,
    parameter_policy=:auto,
    ϵ_gap=Float64x4(1e-8),
)
result = SDPX.solve!(problem, options)
```

Add an ergonomic wrapper that maps a small set of stable names:

```julia
result = SDPX.solve(
    problem;
    tolerance=1e-8,
    time_limit=60.0,
    threads=4,
    linear_solver=:qdldl,
    verbose=true,
)
```

Avoid keeping two independent option systems. The wrapper must translate into
`SolverOptions` and reject unknown keywords.

### Result and diagnostics

The result should contain:

- a machine-readable status enum;
- primal and dual solutions;
- objective values;
- scaled and unscaled residuals;
- PSD minimum-eigenvalue checks;
- iteration, restart, and regularization counts;
- setup, assembly, factorization, solve, and total times;
- linear-solver name, thread count, arithmetic type, and factor statistics.

Use `show` for a concise summary and keep all fields programmatically
accessible.

### Repeated solves

Bootstrap applications often solve a family with an unchanged sparsity
pattern. Support:

```julia
session = SDPX.analyze(problem, options)
SDPX.update_objective!(session, c2)
SDPX.update_rhs!(session, b2)
result = SDPX.solve!(session)
```

The session should reuse:

- the block-incidence graph;
- KKT CSC structure and index maps;
- ordering and symbolic factorization;
- workspace allocations;
- optionally the previous iterate as a warm start.

Make data-update validity explicit. A structural nonzero change should either
trigger re-analysis or throw a clear error.

## MathOptInterface and JuMP

The implemented type is `SDPX.Optimizer{T} <: MOI.AbstractOptimizer`. It is a
non-incremental, copy-in optimizer because SDPX finalizes its structure before
solving.

### Initial supported form

- scalar affine objective;
- `MOI.MIN_SENSE` and `MOI.MAX_SENSE`;
- free scalar variables;
- `MOI.ScalarAffineFunction{T}` in `MOI.EqualTo{T}`;
- `MOI.VectorAffineFunction{T}` in
  `MOI.PositiveSemidefiniteConeTriangle`;
- optionally PSD constrained variables through
  `MOI.VectorOfVariables` in `MOI.PositiveSemidefiniteConeTriangle`.

Use the MOI triangle convention exactly: packed upper triangle and
`sqrt(2)`-scaled off-diagonal coordinates where required by the relevant
primal/dual mapping. Add explicit tests for trace inner products and dual
signs.

### Required result attributes

At minimum implement:

- `MOI.TerminationStatus`;
- `MOI.PrimalStatus` and `MOI.DualStatus`;
- `MOI.RawStatusString`;
- `MOI.ResultCount`;
- `MOI.ObjectiveValue` and `MOI.DualObjectiveValue`;
- `MOI.VariablePrimal`;
- `MOI.ConstraintPrimal` and `MOI.ConstraintDual`;
- `MOI.SolveTimeSec`;
- `MOI.BarrierIterations`;
- `MOI.SolverName`, `MOI.SolverVersion`, and `MOI.Silent`.

Map every SDPX status explicitly. Do not report `MOI.INFEASIBLE` or
`MOI.DUAL_INFEASIBLE` until the solver has a valid certificate.

Expose stable options as typed MOI attributes and keep
`MOI.RawOptimizerAttribute` for advanced fields. A user should be able to
write:

```julia
using JuMP, SDPX, MultiFloats

const T = Float64x4
model = GenericModel{T}(() -> SDPX.Optimizer{T}())
set_optimizer_attribute(model, "sparse", true)
set_optimizer_attribute(model, "linear_solver", :qdldl)
```

The exact constructor syntax should be verified against the supported JuMP
version and included in an executed documentation test.

### Test strategy

Use `MOI.Test` incrementally:

1. Basic model lifecycle and attributes.
2. Linear equality and objective tests.
3. Small `2x2` and `3x3` PSD primal/dual tests.
4. Infeasible and iteration-limit status mapping.
5. Duplicate coefficients and rank-deficient equalities.
6. `Float64`, `BigFloat`, and `Float64x4`.
7. JuMP integration examples as documentation tests.

Run reference comparisons against Clarabel on the same MOI model, but compare
residuals and objectives rather than bitwise solution vectors.

## Package layout

```text
src/
  SDPX.jl
  native/
  linear_solvers/
  moi/
    optimizer.jl
    copy_to.jl
    attributes.jl
    status.jl
ext/
test/
  native/
  moi/
  regression/
docs/
```

MathOptInterface can be a direct dependency if the wrapper ships in the main
package. JuMP should remain a documentation/test dependency, not a runtime
dependency.

## References

- MOI solver-interface guide:
  <https://jump.dev/MathOptInterface.jl/stable/tutorials/implementing/>
- MOI solution and status conventions:
  <https://jump.dev/MathOptInterface.jl/stable/manual/solutions/>
- JuMP arbitrary-precision models:
  <https://jump.dev/JuMP.jl/stable/tutorials/conic/arbitrary_precision/>
- Clarabel PSD triangle convention:
  <https://clarabel.org/stable/api_cone_types/>
