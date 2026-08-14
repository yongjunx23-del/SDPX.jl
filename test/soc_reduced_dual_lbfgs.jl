using SDPX
using Test
using MultiFloats: Float64x2, Float64x3, Float64x4

@testset "typed deterministic L-BFGS" begin
    schedule = SDPX._reduced_dual_schedule(Float64, :auto, 1e-12)
    @test schedule[end] == 1e-12
    @test all(schedule[index] < schedule[index - 1] for index in 2:length(schedule))
    @test count(value -> isapprox(value, 1e-12; rtol=sqrt(eps()), atol=0),
                schedule) == 1

    function quadratic!(gradient, point, tau)
        gradient[1] = 2 * (point[1] - 2)
        gradient[2] = 4 * (point[2] + 1)
        return (point[1] - 2)^2 + 2 * (point[2] + 1)^2 + tau * zero(tau)
    end
    options = SDPX.ReducedDualLBFGSOptions{Float64}(
        gradient_tolerance=1e-10, maximum_iterations=80,
    )
    first = SDPX._solve_reduced_dual_lbfgs(
        quadratic!, [0.0, 0.0], [1e-2, 1e-4], options,
    )
    second = SDPX._solve_reduced_dual_lbfgs(
        quadratic!, [0.0, 0.0], [1e-2, 1e-4], options,
    )
    @test first.status === :converged
    @test first.point ≈ [2.0, -1.0] atol=1e-9
    @test first.point == second.point
    @test first.counters.continuation_stages == 2
    @test first.counters.accepted_steps == first.iterations

    function no_step!(gradient, point, tau)
        gradient[1] = one(eltype(point))
        return iszero(point[1]) ? zero(eltype(point)) : eltype(point)(Inf)
    end
    failed = SDPX._solve_reduced_dual_lbfgs(
        no_step!, [0.0], [1e-3],
        SDPX.ReducedDualLBFGSOptions{Float64}(
            minimum_step=0.25, maximum_iterations=4,
        ),
    )
    @test failed.reason === :line_search_failed
end

if get(ENV, "SDPX_RUN_MFLA_REDUCED_DUAL", "0") == "1"
    @eval using MultiFloatLinearAlgebra
    @testset "MFLA typed reduced dual" begin
        A = [0.0 0.0; 1 0; 0 1]
        problem = second_order_program(
            zeros(2), [SOCConstraint(A, [1.0, 0.0, 0.0])];
            Aeq=reshape([1.0, 0.0], 1, 2), beq=[0.5],
        )
        for T in (Float64x2, Float64x3, Float64x4)
            first = solve_value(
                problem;
                arithmetic=T,
                tolerance=T(1e-12),
                smoothing=T[T(1e-3), T(1e-12)],
                linear_algebra_backend=:multifloat,
                maximum_iterations=100,
            )
            second = solve_value(
                problem;
                arithmetic=T,
                tolerance=T(1e-12),
                smoothing=T[T(1e-3), T(1e-12)],
                linear_algebra_backend=:multifloat,
                maximum_iterations=100,
            )
            @test first.status === SDPX.Optimal
            @test first.certificate.valid
            @test first.provider === :multifloat_linear_algebra
            @test first.arithmetic === SDPX._la_arithmetic_symbol(T)
            @test first.objective == second.objective
            @test first.y == second.y
            @test first.termination.no_psd_lift
            @test isempty(first.termination.fallback_chain)
        end
    end
end

@testset "answer-only reduced dual certification" begin
    A = [0.0 0.0; 1 0; 0 1]
    problem = second_order_program(
        zeros(2), [SOCConstraint(A, [1.0, 0.0, 0.0])];
        Aeq=reshape([1.0, 0.0], 1, 2), beq=[0.5],
    )
    result = solve_value(
        problem;
        tolerance=1e-6,
        smoothing=[1e-2, 1e-6],
        maximum_iterations=80,
    )
    @test result.status === SDPX.Optimal
    @test result.certificate.valid
    @test result.algorithm === :reduced_dual_lbfgs
    @test result.specialization === :fixed_trace_q3
    @test result.provider === :blas_lapack
    @test result.interval_kind === :numerical_certificate
    @test !result.rigorous_interval
    @test result.lower <= result.objective <= result.upper
    @test result.termination.no_psd_lift
    @test isempty(result.termination.fallback_chain)
    @test result.counters.ipm_polish_iterations == 0
    trace = SDPX.performance_trace(result)
    @test trace.setup.solver === :native_soc_reduced_dual
    @test trace.final.certificate_valid
    @test trace.counters.lbfgs_iterations == result.counters.lbfgs_iterations
    @test trace.counters.iterations == result.counters.lbfgs_iterations
    @test occursin("iterations=", sprint(show, trace))
    reconstructed = reconstruct_fixed_trace_solution(problem, result)
    @test length(reconstructed.x) == 2
    @test reconstructed.slack[1][1] >= hypot(
        reconstructed.slack[1][2], reconstructed.slack[1][3],
    )
    @test_throws ArgumentError solve_value(
        problem; linear_algebra_backend=:multifloat,
    )
    @test_throws ArgumentError solve_value(
        problem;
        arithmetic=Float64x2,
        linear_algebra_backend=:standard,
    )

    limited = solve_value(
        problem;
        tolerance=1e-6,
        smoothing=[1e-2],
        maximum_iterations=80,
        max_time=0.0,
    )
    @test limited.status === SDPX.TimeLimit
    @test limited.termination.optimizer_reason === :time_limit

    polished = solve_value(
        problem;
        tolerance=1e-6,
        smoothing=[1e-2, 1e-6],
        polish=:native_soc_ipm,
        maximum_iterations=80,
    )
    @test polished.status === SDPX.Optimal
    @test polished.certificate.valid
    @test polished.polish === :native_soc_ipm
    @test polished.termination.polish === :native_soc_ipm
    @test polished.reconstruction_token === nothing
    @test polished.counters.ipm_polish_iterations > 0
    @test_throws ArgumentError reconstruct_fixed_trace_solution(problem, polished)

    reference = solve_value(
        problem;
        soc_algorithm=:primal_dual_ipm,
        tolerance=1e-6,
        maximum_iterations=80,
    )
    @test reference.status === SDPX.Optimal
    @test reference.certificate.valid
    @test reference.algorithm === :primal_dual_ipm

    invalid_ipm = ConicResult{Float64}(
        SDPX.Optimal,
        "synthetic invalid Optimal",
        [0.0, 0.0],
        [[1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [0.0],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        nothing,
        nothing,
    )
    invalid_wrapped = SDPX._certified_from_ipm(
        invalid_ipm,
        problem,
        SolverOptions(Float64; tolerance=1e-6),
        0.0,
    )
    @test invalid_wrapped.status === SDPX.NumericalFailure
    @test invalid_wrapped.termination.reason === :final_certificate_failed

    for bits in (128, 256)
        big = solve_value(
            problem;
            arithmetic=BigFloat,
            precision_bits=bits,
            tolerance="1e-20",
            smoothing=("1e-8", "1e-20"),
            maximum_iterations=100,
        )
        @test big.arithmetic === :bigfloat
        @test big.precision_bits == bits
        @test big.certificate.valid
        @test precision(big.objective) == bits
        reconstructed_big = reconstruct_fixed_trace_solution(problem, big)
        @test all(value -> precision(value) == bits, reconstructed_big.x)
    end
end
