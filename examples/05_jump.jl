# The same problem through JuMP.
#
# SDPX implements MathOptInterface, so it can be used as a JuMP optimizer and
# the model can be written in the natural algebraic form rather than assembled
# into coefficient arrays by hand. Compare this against `01_basic_sdp.jl`:
# same problem, same answer, considerably less bookkeeping.
#
# Run with:  julia --project=examples examples/05_jump.jl

using JuMP
using LinearAlgebra          # `Symmetric` lives here, not in JuMP
using Printf
using SDPX

# Solver options go to the Optimizer constructor, not to `optimize!`.
model = Model(() -> SDPX.Optimizer(sparse=:auto, verbosity=0))

@variable(model, x[1:2])
@constraint(model, Symmetric([x[1] -1.0; -1.0 x[2]]) in PSDCone())
@objective(model, Min, 2x[1] + 3x[2])

optimize!(model)

exact = 2 * sqrt(6)
@printf("termination : %s\n", termination_status(model))
@printf("primal      : %s\n", primal_status(model))
@printf("objective   : %.15f\n", objective_value(model))
@printf("exact 2√6   : %.15f\n", exact)
@printf("error       : %.3e\n", abs(objective_value(model) - exact))
@printf("x           : %s\n", value.(x))

termination_status(model) == MOI.OPTIMAL ||
    error("expected OPTIMAL, got $(termination_status(model))")
abs(objective_value(model) - exact) < 1e-7 ||
    error("objective $(objective_value(model)) is too far from $(exact)")
