#!/usr/bin/env julia

"""Controlled fixed-parameter sweep for the Task_Low08 Float64 solve."""

using LinearAlgebra
using Printf
using SDPX
using SparseArrays

let root = get(
        ENV,
        "SDPX_BENCH_SOURCE",
        normpath(joinpath(@__DIR__, "..", "..")),
    ),
    path = joinpath(
        root,
        "bench",
        "lattice_bootstrap",
        "benchmark_sdpx_float64_solve.jl",
    )
    source = read(path, String)
    include_string(
        @__MODULE__,
        replace(source, r"\nmain\(ARGS\)\s*$" => "\n"),
        path,
    )
end

parse_grid(name::String, fallback::String) = [
    parse(Float64, value)
    for value in split(get(ENV, name, fallback), ',')
]

function options(beta::Float64, gamma::Float64, tolerance::Float64)
    return SDPX.SolverOptions{Float64}(
        β=beta,
        γ=gamma,
        Ωp=100.0,
        Ωd=0.001,
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
        iter_max=50,
        max_time=90.0,
        verbosity=0,
        timing=true,
        restart=true,
        max_restarts=5,
        sparse=:auto,
        parameter_policy=:fixed,
        predictor=:sdpb,
        step_rule=:backtrack,
        refine_steps=1,
    )
end

function main(arguments)
    length(arguments) == 3 ||
        error("usage: sweep_task_low08_parameters.jl INPUT OUTPUT BLAS_THREADS")
    input, output = abspath(arguments[1]), abspath(arguments[2])
    blas_threads = parse(Int, arguments[3])
    tolerance = parse(Float64, get(ENV, "SWEEP_TOL", "1e-6"))
    betas = parse_grid("SWEEP_BETAS", "0.03,0.05,0.075,0.1,0.15")
    gammas = parse_grid("SWEEP_GAMMAS", "0.8,0.85,0.9")

    SDPX.set_blas_threads!(blas_threads)
    data = read_problem(input)
    equality_ids, rank, _, _ = equality_basis(data.B, data.b)
    problem = SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B[:, equality_ids],
        data.b[equality_ids];
        sparse=:auto,
        validate=false,
        symmetrize=false,
        verbosity=0,
    )
    @printf(
        "variables=%d equalities=%d->%d blocks=%d julia_threads=%d blas_threads=%d\n",
        problem.dims.m,
        size(data.B, 2),
        rank,
        problem.dims.L,
        Threads.nthreads(),
        BLAS.get_num_threads(),
    )
    SDPX.solve!(problem, options(0.1, 0.85, tolerance))

    mkpath(dirname(output))
    open(output, "w") do stream
        println(
            stream,
            "beta,gamma,seconds,iterations,status,objective,dual_objective," *
            "relative_gap,primal_residual,dual_residual,schur_seconds," *
            "kkt_seconds,factor_seconds,solve_seconds,allocated_bytes",
        )
        for beta in betas, gamma in gammas
            GC.gc()
            measurement = @timed SDPX.solve!(
                problem,
                options(beta, gamma, tolerance),
            )
            result = measurement.value
            timings = result.timings
            @printf(
                "beta=%.3f gamma=%.3f seconds=%.3f iterations=%d status=%s gap=%.3e p_res=%.3e d_res=%.3e\n",
                beta,
                gamma,
                measurement.time,
                result.iterations,
                result.status,
                result.gap_rel,
                result.p_res,
                result.d_res,
            )
            flush(stdout)
            @printf(
                stream,
                "%.6g,%.6g,%.6f,%d,%s,%.16g,%.16g,%.8e,%.8e,%.8e,%.6f,%.6f,%.6f,%.6f,%d\n",
                beta,
                gamma,
                measurement.time,
                result.iterations,
                result.status,
                result.pObj,
                result.dObj,
                result.gap_rel,
                result.p_res,
                result.d_res,
                timings.schur_assembly,
                timings.kkt_factorization,
                timings.kkt_schur_factorization,
                timings.predictor_linear_solve +
                    timings.corrector_linear_solve,
                measurement.bytes,
            )
            flush(stream)
        end
    end
end

main(ARGS)
