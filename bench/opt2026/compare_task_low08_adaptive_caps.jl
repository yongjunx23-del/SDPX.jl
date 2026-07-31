#!/usr/bin/env julia

"""
Compare Task_Low08 fixed and adaptive centering trajectories in one process.

The model is read, rank-reduced, and ingested once. A one-iteration solve then
warms every major Float64 SDP kernel before the measured configurations run on
the same node. This isolates the effect of `adaptive_sigma_max` from input,
presolve, compilation, and node-to-node variation.
"""

using LinearAlgebra
using Printf
using SDPX

include(
    normpath(
        joinpath(
            @__DIR__,
            "..",
            "lattice_bootstrap",
            "benchmark_sdpx_float64_solve.jl",
        ),
    ),
)

function comparison_options(
    strategy::Symbol,
    sigma_max::Float64,
    tolerance::Float64,
    maximum_iterations::Int,
    maximum_time::Float64,
)
    return SDPX.SolverOptions{Float64}(
        β=0.1,
        γ=0.9,
        Ωp=1.0,
        Ωd=1.0,
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
        iter_max=maximum_iterations,
        max_time=maximum_time,
        verbosity=1,
        timing=true,
        restart=true,
        max_restarts=5,
        sparse=:auto,
        parameter_policy=:auto,
        parameter_strategy=strategy,
        adaptive_sigma_max=sigma_max,
        predictor=:classic,
        step_rule=:auto,
        refine_steps=1,
        threads=Threads.nthreads(),
    )
end

function measured_run(
    label::String,
    problem,
    full_B,
    full_b,
    options,
)
    GC.gc()
    measurement = @timed SDPX.solve!(problem, options)
    result = measurement.value
    validation = validate_solution(problem, result, full_B, full_b)
    certificate = result.diagnostics === nothing ?
                  nothing :
                  result.diagnostics.selected_algorithms.certificate
    maximum_sigma = maximum(
        (row.sigma for row in result.parameter_history);
        init=0.0,
    )
    total_backtracks = sum(
        row.backtracking_count for row in result.parameter_history;
        init=0,
    )
    @printf(
        "%s status=%s iterations=%d seconds=%.3f sigma_max=%.3f backtracks=%d gap=%.3e p_res=%.3e d_res=%.3e min_eig=%.3e\n",
        label,
        result.status,
        result.iterations,
        measurement.time,
        maximum_sigma,
        total_backtracks,
        result.gap_rel,
        result.p_res,
        result.d_res,
        validation.minimum_psd_eigenvalue,
    )
    flush(stdout)
    return Dict{String,Any}(
        "label" => label,
        "status" => string(result.status),
        "message" => result.message,
        "iterations" => result.iterations,
        "solve_seconds" => measurement.time,
        "allocated_bytes" => measurement.bytes,
        "objective" => result.pObj,
        "dual_objective" => result.dObj,
        "relative_gap" => result.gap_rel,
        "primal_residual" => result.p_res,
        "dual_residual" => result.d_res,
        "maximum_selected_sigma" => maximum_sigma,
        "total_backtracking_count" => total_backtracks,
        "requested_adaptive_sigma_max" => options.adaptive_sigma_max,
        "parameter_strategy" => string(options.parameter_strategy),
        "timings" => result.timings,
        "validation" => validation,
        "certificate_valid" =>
            certificate === nothing ? false : certificate.valid,
        "certificate" => certificate,
        "parameter_history" => result.parameter_history,
    )
end

function compare_main(arguments)
    length(arguments) >= 2 || error(
        "usage: compare_task_low08_adaptive_caps.jl INPUT OUTPUT " *
        "[BLAS_THREADS] [TOLERANCE] [MAX_TIME]",
    )
    input_path = abspath(arguments[1])
    output_path = abspath(arguments[2])
    blas_threads = length(arguments) >= 3 ? parse(Int, arguments[3]) : 16
    tolerance = length(arguments) >= 4 ? parse(Float64, arguments[4]) : 1e-6
    maximum_time = length(arguments) >= 5 ? parse(Float64, arguments[5]) : 600.0
    SDPX.set_blas_threads!(blas_threads)

    input_timing = @timed read_problem(input_path)
    data = input_timing.value
    presolve_timing = @timed equality_basis(data.B, data.b)
    equality_ids, equality_rank, rank_tolerance, dependency_residual =
        presolve_timing.value
    ingest_timing = @timed SDPX.ingest(
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
    problem = ingest_timing.value

    @printf(
        "Task_Low08 variables=%d equalities=%d->%d blocks=%d Julia_threads=%d BLAS_threads=%d\n",
        problem.dims.m,
        length(data.b),
        equality_rank,
        problem.dims.L,
        Threads.nthreads(),
        SDPX.blas_threads(),
    )

    # Compile the complete first-iteration path before collecting measurements.
    warmup = comparison_options(:adaptive, 0.15, tolerance, 1, maximum_time)
    SDPX.solve!(problem, warmup)

    configurations = if haskey(ENV, "SDPX_SIGMA_CAPS")
        caps = parse.(Float64, split(ENV["SDPX_SIGMA_CAPS"], ','))
        [
            (
                @sprintf("adaptive_cap_%.3f_run_%d", cap, index),
                :adaptive,
                cap,
            )
            for (index, cap) in enumerate(caps)
        ]
    else
        [
            ("adaptive_cap_0.15", :adaptive, 0.15),
            ("adaptive_cap_0.20", :adaptive, 0.20),
            ("fixed_profile", :fixed, 0.0),
            ("adaptive_uncapped_0.50", :adaptive, 0.50),
        ]
    end
    runs = Any[]
    for (label, strategy, sigma_max) in configurations
        options = comparison_options(
            strategy,
            sigma_max,
            tolerance,
            100,
            maximum_time,
        )
        push!(
            runs,
            measured_run(label, problem, data.B, data.b, options),
        )
    end

    output = Dict{String,Any}(
        "benchmark" => "Task_Low08 adaptive centering cap comparison",
        "sdpx_version" => string(pkgversion(SDPX)),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => SDPX.blas_threads(),
        "input_seconds" => input_timing.time,
        "equality_presolve_seconds" => presolve_timing.time,
        "ingest_seconds" => ingest_timing.time,
        "equality_rank" => equality_rank,
        "equality_rank_tolerance" => rank_tolerance,
        "equality_dependency_residual" => dependency_residual,
        "peak_rss_bytes" => Sys.maxrss(),
        "runs" => runs,
    )
    write_json(output_path, output)
end

(abspath(PROGRAM_FILE) == @__FILE__) && compare_main(ARGS)
