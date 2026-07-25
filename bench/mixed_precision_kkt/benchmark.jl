#!/usr/bin/env julia

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using Random
using SDPX

LinearAlgebra.BLAS.set_num_threads(1)

function benchmark_problem(::Type{T}, variables::Int, equalities::Int) where {T}
    coefficients = [zeros(T, variables, 2, 2)]
    @inbounds for variable in 1:variables
        coefficients[1][variable, 1, 1] = one(T)
        coefficients[1][variable, 2, 2] =
            one(T) + T(variable) / T(variables)
    end
    equality_matrix =
        T.(randn(MersenneTwister(2), variables, equalities))
    return SDPX.ingest(
        ones(T, variables),
        coefficients,
        [zeros(T, 2, 2)],
        equality_matrix,
        zeros(T, equalities);
        sparse=false,
        verbosity=0,
    )
end

function minimum_timing(run, repetitions::Int)
    run()
    measurements = [@timed run() for _ in 1:repetitions]
    return measurements[argmin(getproperty.(measurements, :time))]
end

function direct_kkt_case(::Type{T}, variables::Int; repetitions::Int=3) where {T}
    equalities = max(1, variables ÷ 16)
    problem = benchmark_problem(T, variables, equalities)
    rng = MersenneTwister(3)
    random_matrix = T.(randn(rng, variables, variables))
    schur =
        random_matrix * transpose(random_matrix) +
        T(5) * Matrix{T}(I, variables, variables)
    primal_rhs = T.(randn(rng, variables))
    equality_rhs = T.(randn(rng, equalities))

    function make_runner(mode::Symbol)
        options = SDPX.SolverOptions{T}(
            verbosity=0,
            mixed_precision_kkt=mode,
            mixed_precision_memory_fraction=1.0,
            mixed_precision_condition_limit=1.0e10,
            mixed_precision_refine_max_steps=20,
        )
        workspace = SDPX.Workspace(
            problem;
            mixed_precision_kkt=mode,
            mixed_precision_memory_fraction=1.0,
            thread_count=1,
        )
        SDPX.copy_owned!(workspace.S, schur)
        SDPX.copy_owned!(workspace.p, equality_rhs)
        return function run()
            factor = SDPX.factor_kkt!(workspace, problem, options)
            factor.ok || error("KKT factorization failed")
            SDPX.solve_kkt!(
                workspace,
                equalities,
                primal_rhs,
                equality_rhs,
                workspace.dx,
                workspace.dy,
            )
            steps, residual = SDPX.refine_direction!(
                workspace,
                problem,
                options,
                primal_rhs,
            )
            return (; workspace, steps, residual)
        end
    end

    native = minimum_timing(make_runner(:off), repetitions)
    mixed = minimum_timing(make_runner(:on), repetitions)
    workspace = mixed.value.workspace
    primal_error = maximum(
        abs,
        schur * workspace.dx -
        problem.B * workspace.dy -
        primal_rhs,
    )
    equality_error = maximum(
        abs,
        transpose(problem.B) * workspace.dx -
        equality_rhs,
    )
    return (
        arithmetic=string(T),
        variables,
        equalities,
        native_seconds=native.time,
        mixed_seconds=mixed.time,
        speedup=native.time / mixed.time,
        native_bytes=native.bytes,
        mixed_bytes=mixed.bytes,
        refinement_steps=mixed.value.steps,
        residual=Float64(mixed.value.residual),
        primal_error=Float64(primal_error),
        equality_error=Float64(equality_error),
        condition_estimate=workspace.mixed_precision.condition_estimate,
        predicted_steps=workspace.mixed_precision.predicted_refinement_steps,
        active=workspace.mixed_precision.active,
        reason=workspace.mixed_precision.reason,
    )
end

function analytic_problem(::Type{T}) where {T}
    coefficients = zeros(T, 2, 2, 2)
    coefficients[1, 1, 1] = one(T)
    coefficients[2, 2, 2] = one(T)
    constant = T[0 1; 1 0]
    return SDPX.ingest(
        T[2, 3],
        [coefficients],
        [constant],
        zeros(T, 2, 0),
        T[];
        sparse=false,
        verbosity=0,
    )
end

function complete_solve_case(::Type{T}; repetitions::Int=3) where {T}
    problem = analytic_problem(T)
    function run(mode::Symbol)
        options = SDPX.SolverOptions{T}(
            verbosity=0,
            precision_bits=256,
            ϵ_gap=T(1e-30),
            ϵ_primal=T(1e-30),
            ϵ_dual=T(1e-30),
            iter_max=100,
            parameter_policy=:fixed,
            mixed_precision_kkt=mode,
            mixed_precision_memory_fraction=1.0,
            mixed_precision_refine_max_steps=20,
        )
        return SDPX.solve!(problem, options)
    end
    native = minimum_timing(() -> run(:off), repetitions)
    mixed = minimum_timing(() -> run(:on), repetitions)
    return (
        arithmetic=string(T),
        native_seconds=native.time,
        mixed_seconds=mixed.time,
        speedup=native.time / mixed.time,
        native_bytes=native.bytes,
        mixed_bytes=mixed.bytes,
        native_status=native.value.status,
        mixed_status=mixed.value.status,
        native_iterations=native.value.iterations,
        mixed_iterations=mixed.value.iterations,
        native_objective=Float64(native.value.pObj),
        mixed_objective=Float64(mixed.value.pObj),
        native_gap=Float64(native.value.gap_rel),
        mixed_gap=Float64(mixed.value.gap_rel),
        native_primal_residual=Float64(native.value.p_res),
        mixed_primal_residual=Float64(mixed.value.p_res),
        native_dual_residual=Float64(native.value.d_res),
        mixed_dual_residual=Float64(mixed.value.d_res),
    )
end

function condition_estimator_case(dimension::Int)
    rng = MersenneTwister(5512)
    random_matrix = randn(rng, dimension, dimension)
    source =
        random_matrix * transpose(random_matrix) +
        dimension * I
    factor = cholesky!(Symmetric(Matrix(source), :L))
    SDPX._triangular_condition_estimate(factor)
    estimate = @timed SDPX._triangular_condition_estimate(factor)
    factor_scratch = Matrix(source)
    cholesky!(Symmetric(factor_scratch, :L))
    copyto!(factor_scratch, source)
    factor_timing =
        @timed cholesky!(Symmetric(factor_scratch, :L))
    return (
        dimension,
        estimator_seconds=estimate.time,
        factor_seconds=factor_timing.time,
        estimator_over_factor=estimate.time / factor_timing.time,
        estimator_bytes=estimate.bytes,
        estimate=estimate.value,
    )
end

function print_rows()
    println(
        "kind,arithmetic,variables,equalities,native_seconds," *
        "mixed_seconds,speedup,native_bytes,mixed_bytes,steps,residual," *
        "primal_error,equality_error,condition_estimate,predicted_steps," *
        "active,reason",
    )
    setprecision(BigFloat, 256) do
        for T in (BigFloat, Float64x4), variables in (64, 128, 256)
            row = direct_kkt_case(T, variables)
            println(
                join(
                    (
                        "direct_kkt",
                        row.arithmetic,
                        row.variables,
                        row.equalities,
                        row.native_seconds,
                        row.mixed_seconds,
                        row.speedup,
                        row.native_bytes,
                        row.mixed_bytes,
                        row.refinement_steps,
                        row.residual,
                        row.primal_error,
                        row.equality_error,
                        row.condition_estimate,
                        row.predicted_steps,
                        row.active,
                        row.reason,
                    ),
                    ',',
                ),
            )
        end
        for T in (BigFloat, Float64x4)
            row = complete_solve_case(T)
            println(
                "complete_solve,$(row.arithmetic),2,0," *
                "$(row.native_seconds),$(row.mixed_seconds),$(row.speedup)," *
                "$(row.native_bytes),$(row.mixed_bytes),," *
                "$(row.mixed_gap),$(row.mixed_primal_residual)," *
                "$(row.mixed_dual_residual),,,,," *
                "status=$(row.native_status)/$(row.mixed_status);" *
                "iterations=$(row.native_iterations)/$(row.mixed_iterations);" *
                "objective=$(row.native_objective)/$(row.mixed_objective)",
            )
        end
    end
    estimator = condition_estimator_case(4096)
    println(
        "condition_estimator,Float64,$(estimator.dimension),0," *
        "$(estimator.factor_seconds),$(estimator.estimator_seconds)," *
        "$(estimator.estimator_over_factor),0,$(estimator.estimator_bytes)," *
        ",,,,,,,estimate=$(estimator.estimate)",
    )
end

print_rows()
