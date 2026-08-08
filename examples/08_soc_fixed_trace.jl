using LinearAlgebra
using SDPX

# A direct Lorentz model: min t subject to (t, x, y) in Q3, x=3, y=4.
soc = second_order_program(
    [1.0, 0.0, 0.0],
    Matrix{Float64}(I, 3, 3),
    zeros(3);
    Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
    beq=[3.0, 4.0],
)
soc_result = solve_socp(soc; verbosity=0)
@assert soc_result.status == SDPX.Optimal
@assert isapprox(soc_result.pObj, 5.0; atol=1e-7)

# The fixed-trace form of the unit disk,
#
#     [1 + q  r    ] >= 0    <=>    (1, q, r) in Q3,
#     [r      1 - q]
#
# has two local variables and a constant cone head. This is the narrow
# structure for which `algorithm=:socp` selects SDPX's native Q3 backend.
coefficients = zeros(2, 2, 2)
coefficients[1, :, :] = [1.0 0.0; 0.0 -1.0]
coefficients[2, :, :] = [0.0 1.0; 1.0 0.0]
fixed_trace = ingest(
    [-1.0, 0.0],
    [coefficients],
    [[-1.0 0.0; 0.0 -1.0]],
    zeros(2, 0),
    Float64[];
    sparse=true,
    verbosity=0,
)
q3_options = SolverOptions{Float64}(
    algorithm=:socp,
    scaling=:none,
    ϵ_gap=1e-8,
    ϵ_primal=1e-8,
    ϵ_dual=1e-8,
    verbosity=0,
)
q3_result = solve!(fixed_trace, q3_options)
certificate = result_certificate(fixed_trace, q3_result, q3_options)

@assert q3_result.status == SDPX.Optimal
@assert certificate.valid
@assert q3_result.termination.executed.kkt == :q3_block_diagonal_equality
@assert isapprox(q3_result.pObj, -1.0; atol=1e-7)

println("direct SOC optimum = ", soc_result.pObj)
println("fixed-trace Q3 optimum = ", q3_result.pObj)
