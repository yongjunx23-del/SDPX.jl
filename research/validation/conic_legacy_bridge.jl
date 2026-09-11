using Test
using SparseArrays
using SDPX

@testset "ConicProblem SolverOptions compatibility bridge" begin
    T = Float64
    A = sparse([2, 3], [1, 2], T[1, 1], 3, 2)
    problem = SDPX.second_order_program(
        zeros(T, 2),
        [SDPX.SOCConstraint(A, T[1, 0, 0]; T)],
    )
    options = SDPX.SolverOptions(
        T;
        maximum_iterations=100,
        time_limit=30.0,
        diagnostics=true,
        timing=true,
    )
    result = SDPX.solve_socp(problem, options)
    @test result.status === SDPX.Optimal
    @test result.p_res == zero(T)
    @test result.d_res == zero(T)
end
