# Convex.jl

SDPX is available as a Convex.jl solver through its non-incremental
MathOptInterface optimizer. Convex performs DCP canonicalization, MOI caches
the completed model, and SDPX converts the supported cones to its native typed
block-SDP representation.

```julia
import Convex
import MathOptInterface as MOI
using SDPX

x = Convex.Variable(2)
problem = Convex.minimize(2x[1] + x[2], [x >= 0, sum(x) == 1])
SDPX.solve_convex!(problem; tolerance=1e-8, threads=4)

@assert Convex.termination_status(problem) == MOI.OPTIMAL
println(problem.optval)
println(Convex.evaluate(x))
```

Affine LP forms, Euclidean-norm/SOC forms, and real PSD forms are supported.
For new PSD models, `SDPX.convex_semidefinite(n)` creates a symmetric Convex
expression backed by only `n(n+1)/2` variables. The original square form remains
available as `SDPX.convex_semidefinite(n; representation=:square)`.

For extended precision, both ends must use the same coefficient type:

```julia
import Convex
using MultiFloats: Float64x4
using SDPX

const T = Float64x4
x = Convex.Variable(2)
problem = Convex.minimize(
    T(2) * x[1] + x[2],
    [x >= zero(T), sum(x) == one(T)];
    numeric_type=T,
)
SDPX.solve_convex!(
    problem;
    numeric_type=T,
    tolerance=T(1e-18),
    threads=4,
)
```

Atoms requiring exponential, power, or other unsupported cones cannot yet be
solved by SDPX. Convex warm starts are also unavailable because the SDPX MOI
wrapper does not currently implement `MOI.VariablePrimalStart`.

See the
[complete Convex guide](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/convex-interface.md)
and the
[matched frontend benchmark](https://github.com/yongjunx23-del/SDPX.jl/tree/main/bench/convex_frontend)
for precision configuration, result access, benchmark boundaries, and current
performance tradeoffs.
