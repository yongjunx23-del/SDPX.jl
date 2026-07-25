# A first semidefinite program, with an optimum known in closed form.
#
#     minimise    2 x₁ + 3 x₂
#     subject to  [ x₁  -1 ]  ⪰ 0
#                 [ -1  x₂ ]
#
# The constraint is `x₁ x₂ ≥ 1` with `x₁, x₂ ≥ 0`, so the optimum is at
# `x₁ = √(3/2)`, `x₂ = √(2/3)` with value `2√6`. Having the exact answer is
# what makes this worth shipping as an example: every other number printed
# below can be checked against it.
#
# Run with:  julia --project=examples examples/01_basic_sdp.jl

using LinearAlgebra
using Printf
using SDPX

# SDPX takes the constraint as `Σᵢ xᵢ Aᵢ − C ⪰ 0`, with `A[i, :, :]` the
# coefficient matrix of variable `i` in one PSD block.
coefficients = zeros(2, 2, 2)
coefficients[1, 1, 1] = 1.0          # x₁ enters the (1,1) entry
coefficients[2, 2, 2] = 1.0          # x₂ enters the (2,2) entry
constant = [0.0 1.0; 1.0 0.0]        # the off-diagonal -1 entries
objective = [2.0, 3.0]

# No equality constraints here, so `B` has zero columns and `b` is empty.
result = solve(
    objective,
    [coefficients],
    [constant],
    Matrix{Float64}(undef, 2, 0),
    Float64[];
    verbosity=0,
)

exact = 2 * sqrt(6)
@printf("status     : %s\n", result.status)
@printf("objective  : %.15f\n", result.pObj)
@printf("exact 2√6  : %.15f\n", exact)
@printf("error      : %.3e\n", abs(result.pObj - exact))
@printf("iterations : %d\n", result.iterations)
@printf("x          : %s\n", result.x)
@printf("expected   : [%.15f, %.15f]\n", sqrt(3 / 2), sqrt(2 / 3))

# `termination` records why the solve stopped, including the measured
# convergence rate the stagnation detector observed.
@printf("\nstopped because: %s\n", result.termination.reason)

result.status == SDPX.Optimal || error("expected Optimal, got $(result.status)")
abs(result.pObj - exact) < 1e-7 ||
    error("objective is $(result.pObj), too far from $(exact)")
