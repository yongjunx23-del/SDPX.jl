import Convex
import MathOptInterface as MOI
using LinearAlgebra
using SDPX

solver = MOI.OptimizerWithAttributes(
    SDPX.Optimizer,
    "tolerance" => 1e-8,
    "max_iterations" => 200,
    MOI.NumberOfThreads() => 1,
)

# Linear program: minimize 2x₁ + x₂ over the unit simplex.
x = Convex.Variable(2)
lp = Convex.minimize(2x[1] + x[2], [x >= 0, sum(x) == 1])
Convex.solve!(lp, solver; silent=true)
@assert Convex.termination_status(lp) == MOI.OPTIMAL
@assert isapprox(lp.optval, 1.0; rtol=1e-7)

# Second-order cone program: minimum Euclidean norm on the same affine slice.
y = Convex.Variable(2)
socp = Convex.minimize(Convex.norm2(y), [sum(y) == 1])
Convex.solve!(socp, solver; silent=true)
@assert Convex.termination_status(socp) == MOI.OPTIMAL
@assert isapprox(socp.optval, inv(sqrt(2.0)); rtol=1e-7)

# Semidefinite program: minimize tr(X) subject to X ⪧0 and X₁₂=1.
X = Convex.Semidefinite(2)
sdp = Convex.minimize(Convex.tr(X), [X[1, 2] == 1])
Convex.solve!(sdp, solver; silent=true)
@assert Convex.termination_status(sdp) == MOI.OPTIMAL
@assert isapprox(sdp.optval, 2.0; rtol=1e-7)
@assert eigmin(Symmetric(Matrix(Convex.evaluate(X)))) >= -1e-7

println("LP objective:   ", lp.optval)
println("SOCP objective: ", socp.optval)
println("SDP objective:  ", sdp.optval)
