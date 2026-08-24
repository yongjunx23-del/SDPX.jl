#!/usr/bin/env julia

"""
Benchmark the ownership-safe BigFloat block-diagonal equality KKT path.

The synthetic problem matches the important geometry of the primal CSDR
model: one independent 2x2 PSD cell per two Schur variables and a dense,
moderate-width equality panel.  Model construction and Julia compilation are
outside the reported factor/solve samples.  Every thread point reuses the same
immutable problem and is validated against the original KKT equations.
"""

using LinearAlgebra
using Printf
using SDPX
using SparseArrays

function parse_cli(arguments)
    values = Dict{String,String}()
    for argument in arguments
        startswith(argument, "--") || error("unknown argument: $argument")
        key, value = split(argument[3:end], '='; limit=2)
        values[key] = value
    end
    return (
        variables=parse(Int, get(values, "variables", "3400")),
        equalities=parse(Int, get(values, "equalities", "144")),
        precision_bits=parse(Int, get(values, "precision-bits", "512")),
        repetitions=parse(Int, get(values, "repetitions", "3")),
        thread_points=parse.(
            Int,
            split(get(values, "threads", "1,2,4,8"), ','),
        ),
    )
end

function build_problem(variables::Int, equalities::Int)
    iseven(variables) || error("variables must be even")
    0 < equalities <= variables ||
        error("equalities must lie in 1:variables")
    block_count = div(variables, 2)
    off_diagonal = sparse(
        [1, 2],
        [2, 1],
        BigFloat[1, 1],
        2,
        2,
    )
    diagonal = sparse(
        [1, 2],
        [1, 2],
        BigFloat[1, -1],
        2,
        2,
    )
    blocks = Vector{
        SDPX.ActiveSparseCoefficientVector{BigFloat}
    }(undef, block_count)
    for block in 1:block_count
        first = 2block - 1
        blocks[block] = SDPX.ActiveSparseCoefficientVector(
            BigFloat,
            variables,
            [first, first + 1],
            [off_diagonal, diagonal],
            2,
        )
    end
    constants = fill(BigFloat[0 0; 0 -2], block_count)
    equality = SDPX.alloc_zeros(BigFloat, variables, equalities)
    @inbounds for column in 1:equalities, row in 1:variables
        perturbation = 0.015 * sin(0.013 * row + 0.017 * column)
        value = row == column ? 1.0 + perturbation : perturbation
        SDPX._mpfr_set_float64!(equality[row, column], value)
    end
    return SDPX.ingest(
        ones(BigFloat, variables),
        blocks,
        constants,
        equality,
        zeros(BigFloat, equalities);
        sparse=true,
        validate=true,
        symmetrize=false,
        verbosity=0,
    )
end

function build_iterates(block_count::Int)
    primal = BigFloat[2.0 0.07; 0.07 1.7]
    dual = BigFloat[1.6 0.04; 0.04 2.1]
    return fill(primal, block_count), fill(dual, block_count)
end

function median_sample(operation, repetitions::Int)
    samples = Float64[]
    for _ in 1:max(repetitions, 1)
        push!(samples, @elapsed operation())
    end
    sort!(samples)
    return samples[cld(length(samples), 2)], minimum(samples), maximum(samples)
end

function relative_kkt_residual(
    workspace,
    problem,
    primal_rhs,
    equality_rhs,
    dx,
    dy,
)
    first = SDPX.alloc_zeros(BigFloat, problem.dims.m)
    second = SDPX.alloc_zeros(BigFloat, problem.dims.n)
    SDPX.schur_mul!(
        first,
        workspace,
        dx,
        one(BigFloat),
        zero(BigFloat),
    )
    SDPX.kmul_owned!(first, problem.B, dy, -one(BigFloat), one(BigFloat))
    SDPX.kaxpby_owned!(
        -one(BigFloat),
        primal_rhs,
        one(BigFloat),
        first,
    )
    SDPX.kmul_owned!(second, transpose(problem.B), dx)
    SDPX.kaxpby_owned!(
        -one(BigFloat),
        equality_rhs,
        one(BigFloat),
        second,
    )
    scale = max(
        maximum(abs, primal_rhs),
        maximum(abs, equality_rhs),
        one(BigFloat),
    )
    return max(maximum(abs, first), maximum(abs, second)) / scale
end

function benchmark_point(
    problem,
    primal,
    dual,
    requested_threads::Int,
    repetitions::Int,
)
    threads = min(max(requested_threads, 1), Threads.nthreads())
    options = SDPX.SolverOptions{BigFloat}(
        verbosity=0,
        threads=threads,
        extended_precision_blas=:on,
        equality_solver=:normal_equations,
    )
    workspace = SDPX.Workspace(
        problem;
        thread_count=threads,
        extended_precision_blas=:on,
        equality_solver=:normal_equations,
    )
    SDPX.factor_blocks!(workspace, primal, dual) ||
        error("block factorization failed")
    SDPX.schur_build!(
        workspace,
        problem,
        problem.cons,
        primal,
        dual,
    )
    backend = SDPX.select_backend(workspace)
    warm = SDPX.factorize!(backend, workspace, problem, options)
    warm.ok || error("warm KKT factorization failed")

    factor_operation() = begin
        factor = SDPX.factorize!(backend, workspace, problem, options)
        factor.ok || error("KKT factorization failed")
        factor
    end
    factor_median, factor_minimum, factor_maximum =
        median_sample(factor_operation, repetitions)
    factor_allocated = @allocated factor_operation()
    factor = factor_operation()

    primal_rhs = BigFloat.(range(
        BigFloat("-0.9"),
        BigFloat("1.1");
        length=problem.dims.m,
    ))
    equality_rhs = BigFloat.(range(
        BigFloat("-0.2"),
        BigFloat("0.3");
        length=problem.dims.n,
    ))
    dx = zeros(BigFloat, problem.dims.m)
    dy = zeros(BigFloat, problem.dims.n)
    solve_operation() = SDPX.solve_kkt!(
        workspace,
        problem.dims.n,
        primal_rhs,
        equality_rhs,
        dx,
        dy,
    )
    solve_operation()
    solve_median, solve_minimum, solve_maximum =
        median_sample(solve_operation, repetitions)
    solve_allocated = @allocated solve_operation()
    residual = relative_kkt_residual(
        workspace,
        problem,
        primal_rhs,
        equality_rhs,
        dx,
        dy,
    )
    phase = factor.phase_times
    selected_workers =
        SDPX.ExtendedPrecisionBLAS._syrk_bigfloat_selected_workers(
            workspace.Btil,
            SDPX._equality_gram_crossover(
                workspace.Btil,
                options,
                threads,
            ).config,
            threads,
        )
    return (
        requested_threads=requested_threads,
        threads=threads,
        selected_workers=selected_workers,
        gram_kernel=workspace.equality_gram_kernel,
        factor_median=factor_median,
        factor_minimum=factor_minimum,
        factor_maximum=factor_maximum,
        solve_median=solve_median,
        solve_minimum=solve_minimum,
        solve_maximum=solve_maximum,
        equality_gram=phase.equality_gram,
        constraint_solve=phase.constraint_triangular_solve,
        equality_factor=phase.equality_factorization,
        factor_allocated=factor_allocated,
        solve_allocated=solve_allocated,
        relative_residual=Float64(residual),
    )
end

function main(arguments)
    cli = parse_cli(arguments)
    setprecision(BigFloat, cli.precision_bits) do
        problem = build_problem(cli.variables, cli.equalities)
        primal, dual = build_iterates(problem.dims.L)
        println(
            "precision_bits,variables,equalities,blocks,requested_threads," *
            "threads,selected_workers,gram_kernel,factor_median_seconds," *
            "factor_minimum_seconds,factor_maximum_seconds,solve_median_seconds," *
            "solve_minimum_seconds,solve_maximum_seconds,equality_gram_seconds," *
            "constraint_solve_seconds,equality_factor_seconds," *
            "factor_allocated_bytes,solve_allocated_bytes,relative_kkt_residual",
        )
        for threads in cli.thread_points
            row = benchmark_point(
                problem,
                primal,
                dual,
                threads,
                cli.repetitions,
            )
            @printf(
                "%d,%d,%d,%d,%d,%d,%d,%s,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%d,%.6e\n",
                cli.precision_bits,
                problem.dims.m,
                problem.dims.n,
                problem.dims.L,
                row.requested_threads,
                row.threads,
                row.selected_workers,
                string(row.gram_kernel),
                row.factor_median,
                row.factor_minimum,
                row.factor_maximum,
                row.solve_median,
                row.solve_minimum,
                row.solve_maximum,
                row.equality_gram,
                row.constraint_solve,
                row.equality_factor,
                row.factor_allocated,
                row.solve_allocated,
                row.relative_residual,
            )
        end
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
