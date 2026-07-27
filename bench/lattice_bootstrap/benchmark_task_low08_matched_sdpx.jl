#!/usr/bin/env julia

"""Run repeated certificate-checked SDPX solves matched to the MOSEK driver."""

using LinearAlgebra
using Printf
using SDPX
using SparseArrays
using Sockets

let path = joinpath(@__DIR__, "benchmark_sdpx_float64_solve.jl")
    source = read(path, String)
    include_string(
        @__MODULE__,
        replace(source, r"\nmain\(ARGS\)\s*$" => "\n"),
        path,
    )
end

function machine_metadata(numa_policy)
    cpu_model = "unknown"
    if Sys.islinux() && isfile("/proc/cpuinfo")
        for line in eachline("/proc/cpuinfo")
            startswith(line, "model name") || continue
            cpu_model = strip(split(line, ":"; limit=2)[2])
            break
        end
    end
    affinity = "unknown"
    if Sys.islinux() && isfile("/proc/self/status")
        for line in eachline("/proc/self/status")
            startswith(line, "Cpus_allowed_list:") || continue
            affinity = strip(split(line, ":"; limit=2)[2])
            break
        end
    end
    return Dict(
        "node" => gethostname(),
        "cpu_model" => cpu_model,
        "affinity" => affinity,
        "numa_policy" => numa_policy,
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => SDPX.blas_threads(),
        "blas_vendor" => string(BLAS.get_config()),
    )
end

function timing_dictionary(timings)
    return Dict(
        string(name) => getproperty(timings, name)
        for name in propertynames(timings)
    )
end

function main(arguments)
    length(arguments) >= 3 || error(
        "usage: benchmark_task_low08_matched_sdpx.jl INPUT OUTPUT BASIS_IDS " *
        "[BLAS_THREADS] [TOLERANCE] [REPETITIONS] [NUMA_POLICY] [BETA] " *
        "[GAMMA] [PARAMETER_POLICY]",
    )
    input_path = abspath(arguments[1])
    output_path = abspath(arguments[2])
    basis_path = abspath(arguments[3])
    blas_threads = length(arguments) >= 4 ? parse(Int, arguments[4]) : 1
    tolerance = length(arguments) >= 5 ? parse(Float64, arguments[5]) : 1e-6
    repetitions = length(arguments) >= 6 ? parse(Int, arguments[6]) : 3
    numa_policy = length(arguments) >= 7 ? arguments[7] : "default"
    beta = length(arguments) >= 8 ? parse(Float64, arguments[8]) : 0.1
    gamma = length(arguments) >= 9 ? parse(Float64, arguments[9]) : 0.85
    parameter_policy =
        length(arguments) >= 10 ? Symbol(arguments[10]) : :fixed
    parameter_policy in (:fixed, :auto) ||
        error("PARAMETER_POLICY must be fixed or auto")
    SDPX.set_blas_threads!(blas_threads)

    process_started = time()
    read_timing = @timed read_problem(input_path)
    data = read_timing.value

    basis_timing = @timed begin
        equality_ids = parse.(Int, readlines(basis_path))
        isempty(equality_ids) && error("The shared equality basis is empty")
        length(unique(equality_ids)) == length(equality_ids) ||
            error("The shared equality basis contains duplicates")
        all(index -> 1 <= index <= length(data.b), equality_ids) ||
            error("The shared equality basis contains an out-of-range index")
        equality_ids
    end
    equality_ids = basis_timing.value
    dependency_check = @timed begin
        _, equality_rank, rank_tolerance, dependency_residual =
            equality_basis(data.B, data.b)
        equality_rank == length(equality_ids) ||
            error("Shared and SDPX equality ranks disagree")
        (equality_rank, rank_tolerance, dependency_residual)
    end
    equality_rank, rank_tolerance, dependency_residual =
        dependency_check.value

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
    options = SDPX.SolverOptions{Float64}(
        β=beta,
        γ=gamma,
        Ωp=100.0,
        Ωd=0.001,
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
        iter_max=100,
        max_time=1200.0,
        verbosity=0,
        timing=true,
        restart=true,
        max_restarts=5,
        sparse=:auto,
        presolve=false,
        parameter_policy=parameter_policy,
        predictor=:sdpb,
        step_rule=:backtrack,
        refine_steps=1,
    )

    # Full untimed warm-up, matching the MOSEK harness.
    warmup = SDPX.solve!(problem, options)
    warmup.status == SDPX.Optimal ||
        error("SDPX warm-up failed with status $(warmup.status)")
    warmup = nothing
    GC.gc()

    runs = Dict{String,Any}[]
    for repetition in 1:repetitions
        GC.gc()
        solve_timing = @timed SDPX.solve!(problem, options)
        result = solve_timing.value
        parameter_profile = result.diagnostics === nothing ?
                            :unavailable :
                            result.diagnostics.plan.parameter_profile
        initial_parameters = result.diagnostics === nothing ?
                             nothing :
                             result.diagnostics.plan.parameters
        validation_timing =
            @timed validate_solution(problem, result, data.B, data.b)
        diagnostics = validation_timing.value
        certificate = result.diagnostics === nothing ?
                      nothing :
                      result.diagnostics.selected_algorithms.certificate
        warnings = result.diagnostics === nothing ?
                   String[] :
                   result.diagnostics.warnings
        refinement_steps = get(
            result.termination,
            :total_refinement_steps,
            0,
        )
        push!(
            runs,
            Dict(
                "repetition" => repetition,
                "status" => string(result.status),
                "solver_seconds" => solve_timing.time,
                "allocated_bytes" => solve_timing.bytes,
                "validation_seconds" => validation_timing.time,
                "end_to_end_from_ingested_seconds" =>
                    solve_timing.time + validation_timing.time,
                "iterations" => result.iterations,
                "objective" => result.pObj,
                "dual_objective" => result.dObj,
                "relative_gap" => result.gap_rel,
                "primal_residual" => result.p_res,
                "dual_residual" => result.d_res,
                "maximum_equality_violation" =>
                    diagnostics.max_absolute_linear_residual,
                "minimum_primal_psd_eigenvalue" =>
                    diagnostics.minimum_psd_eigenvalue,
                "minimum_dual_psd_eigenvalue" =>
                    diagnostics.minimum_dual_psd_eigenvalue,
                "refinement_steps" => refinement_steps,
                "fallback_events" => warnings,
                "parameter_profile" => parameter_profile,
                "initial_parameters" => initial_parameters,
                "certificate" => certificate,
                "solver_phases" => timing_dictionary(result.timings),
            ),
        )
        result = nothing
    end

    solver_times = sort([run["solver_seconds"] for run in runs])
    end_to_end_times =
        sort([run["end_to_end_from_ingested_seconds"] for run in runs])
    median_index = (length(runs) + 1) ÷ 2
    output = Dict{String,Any}(
        "benchmark" => "Task_Low08 matched Float64 solve",
        "solver" => "SDPX",
        "arithmetic" => "Float64",
        "sdpx_version" => string(pkgversion(SDPX)),
        "julia_version" => string(VERSION),
        "tolerance" => tolerance,
        "beta" => beta,
        "gamma" => gamma,
        "parameter_policy" => parameter_policy,
        "repetitions" => repetitions,
        "warmup" => "one complete cold-start solve from ingested problem",
        "variables" => problem.dims.m,
        "equalities" => length(data.b),
        "presolved_equalities" => problem.dims.n,
        "equality_rank" => equality_rank,
        "equality_rank_tolerance" => rank_tolerance,
        "equality_dependency_residual" => dependency_residual,
        "psd_blocks" => problem.dims.L,
        "timing_seconds" => Dict(
            "input_read" => read_timing.time,
            "shared_basis_read" => basis_timing.time,
            "shared_basis_verification" => dependency_check.time,
            "model_ingest" => ingest_timing.time,
            "median_solver" => solver_times[median_index],
            "median_end_to_end_from_ingested" =>
                end_to_end_times[median_index],
            "process_total_including_warmup" => time() - process_started,
        ),
        "peak_rss_bytes" => Sys.maxrss(),
        "machine" => machine_metadata(numa_policy),
        "runs" => runs,
    )
    write_json(output_path, output)
    final_run = runs[end]
    @printf(
        "SDPX statuses=%s median_solve=%.3fs objective=%.12g gap=%.3e eq=%.3e\n",
        join([run["status"] for run in runs], ","),
        output["timing_seconds"]["median_solver"],
        final_run["objective"],
        final_run["relative_gap"],
        final_run["maximum_equality_violation"],
    )
end

main(ARGS)
