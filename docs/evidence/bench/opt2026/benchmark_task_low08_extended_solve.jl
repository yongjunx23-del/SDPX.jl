#!/usr/bin/env julia

"""Memory-gated Task_Low08 solve benchmark for Float64x4 and BigFloat."""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX
using SparseArrays
using Sockets

let root = get(
        ENV,
        "SDPX_BENCH_SOURCE",
        normpath(joinpath(@__DIR__, "..", "..")),
    ),
    path = joinpath(
        root,
        "bench",
        "extended_precision_blas",
        "benchmark_schur.jl",
    )
    source = read(path, String)
    include_string(
        @__MODULE__,
        replace(source, r"\nmain\(ARGS\)\s*$" => "\n"),
        path,
    )
end

function approximate_psd_validation(problem, result, full_B, full_b)
    equality_residual = transpose(full_B) * result.x - full_b
    minimum_primal = Inf
    minimum_dual = Inf
    constraints = problem.cons::SDPX.SparseCons
    for block in 1:problem.dims.L
        matrix = -Float64.(problem.C[block])
        for variable in constraints.active[block]
            coefficient = constraints.Asp[block][variable]
            rows = rowvals(coefficient)
            values = nonzeros(coefficient)
            multiplier = result.x[variable]
            for column in axes(coefficient, 2)
                for index in nzrange(coefficient, column)
                    matrix[rows[index], column] +=
                        Float64(multiplier * values[index])
                end
            end
        end
        minimum_primal = min(
            minimum_primal,
            eigmin(Symmetric(matrix)),
        )
        minimum_dual = min(
            minimum_dual,
            eigmin(Symmetric(Float64.(result.Y[block]))),
        )
    end
    return (
        maximum_equality_violation=Float64(maximum(abs, equality_residual)),
        equality_residual_l2=Float64(norm(equality_residual)),
        minimum_primal_psd_eigenvalue=minimum_primal,
        minimum_dual_psd_eigenvalue=minimum_dual,
    )
end

csv_safe(value) = replace(string(value), ',' => ';', '\n' => ' ')

function append_result(path::String, row)
    mkpath(dirname(path))
    new_file = !isfile(path) || filesize(path) == 0
    names = propertynames(row)
    open(path, "a") do output
        new_file && println(output, join(names, ','))
        println(
            output,
            join(
                (csv_safe(getproperty(row, name)) for name in names),
                ',',
            ),
        )
    end
end

function run_case(
    input::String,
    output::String,
    basis_path::String,
    ::Type{T};
    precision_bits::Int,
    blas_threads::Int,
    tolerance::Float64,
    iterations::Int,
    repetitions::Int,
    beta::Float64,
    gamma::Float64,
    extended_mode::Symbol,
    mixed_mode::Symbol,
    mixed_condition_limit::Float64,
    stall_iterations::Int,
    max_time::Float64,
    verbosity::Int,
) where {T}
    SDPX.set_blas_threads!(blas_threads)
    load_measurement = @timed read_lattice_problem(input, T)
    data = load_measurement.value
    equality_ids = parse.(Int, readlines(basis_path))
    ingest_measurement = @timed SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B[:, equality_ids],
        data.b[equality_ids];
        sparse=:auto,
        validate=false,
        symmetrize=false,
        verbosity=verbosity,
    )
    problem = ingest_measurement.value

    iteration_trace = NamedTuple[]
    trace_callback = state -> begin
        push!(
            iteration_trace,
            (
                iteration=state.iter,
                primal_objective=Float64(state.pObj),
                dual_objective=Float64(state.dObj),
                gap=Float64(state.gap),
                relative_gap=Float64(state.gap_rel),
                primal_residual=Float64(state.p_res),
                dual_residual=Float64(state.d_res),
                complementarity=Float64(state.complementarity),
                termination_merit=Float64(state.termination_merit),
                refinement_steps=state.refine_steps,
                refinement_residual=Float64(state.refine_residual),
            ),
        )
        return false
    end
    options(maximum_iterations; trace::Bool=false) = SDPX.SolverOptions{T}(
        β=T(beta),
        γ=T(gamma),
        Ωp=T(100),
        Ωd=T(0.001),
        ϵ_gap=T(tolerance),
        ϵ_primal=T(tolerance),
        ϵ_dual=T(tolerance),
        iter_max=maximum_iterations,
        max_time=max_time,
        verbosity=verbosity,
        timing=true,
        restart=true,
        max_restarts=5,
        sparse=:auto,
        presolve=false,
        parameter_policy=:fixed,
        predictor=:sdpb,
        step_rule=:backtrack,
        refine_steps=1,
        refine_policy=:auto,
        refine_max_steps=8,
        extended_precision_blas=extended_mode,
        extended_precision_memory_fraction=0.25,
        mixed_precision_kkt=mixed_mode,
        mixed_precision_condition_limit=mixed_condition_limit,
        mixed_precision_memory_fraction=0.25,
        mixed_precision_refine_max_steps=32,
        threads=Threads.nthreads(),
        diagnostics=true,
        stall_iterations=stall_iterations,
        callback=trace ? trace_callback : nothing,
    )

    SDPX.solve!(problem, options(1; trace=false))
    for repetition in 1:repetitions
        GC.gc()
        empty!(iteration_trace)
        solve_measurement = @timed SDPX.solve!(
            problem,
            options(iterations; trace=true),
        )
        result = solve_measurement.value
        validation_measurement = @timed approximate_psd_validation(
            problem,
            result,
            data.B,
            data.b,
        )
        validation = validation_measurement.value
        timings = result.timings
        warnings = result.diagnostics === nothing ?
                   String[] :
                   result.diagnostics.warnings
        selected_algorithms = if result.diagnostics === nothing
            ""
        else
            selected = result.diagnostics.selected_algorithms
            join(
                (
                    "solver=$(selected.solver)",
                    "kkt=$(selected.kkt)",
                    "gram=$(selected.gram)",
                    "scheduling=$(selected.scheduling)",
                    "threads=$(selected.threads)",
                    "parameter_profile=$(selected.parameter_profile)",
                ),
                '|',
            )
        end
        refinement_steps = get(
            result.termination,
            :total_refinement_steps,
            0,
        )
        mixed_kkt = get(
            result.termination,
            :mixed_precision_kkt,
            (available=false,),
        )
        termination_reason = get(result.termination, :reason, :none)
        row = (
            arithmetic=T === BigFloat ? "BigFloat" : "Float64x4",
            precision_bits=T === BigFloat ?
                           precision(BigFloat) :
                           precision(T),
            repetition,
            node=gethostname(),
            julia_threads=Threads.nthreads(),
            blas_threads=BLAS.get_num_threads(),
            extended_mode,
            mixed_mode,
            mixed_condition_limit,
            beta,
            gamma,
            tolerance,
            requested_iterations=iterations,
            stall_iterations,
            maximum_solve_seconds=max_time,
            verbosity,
            iterations=result.iterations,
            status=result.status,
            message=result.message,
            termination_reason,
            termination_merit=Float64(
                get(result.termination, :merit, NaN),
            ),
            termination_rate=Float64(
                get(result.termination, :rate, NaN),
            ),
            termination_projected_iterations=Float64(
                get(result.termination, :projected_iterations, NaN),
            ),
            load_seconds=load_measurement.time,
            ingest_seconds=ingest_measurement.time,
            solve_seconds=solve_measurement.time,
            validation_seconds=validation_measurement.time,
            allocated_bytes=solve_measurement.bytes,
            peak_rss_bytes=Sys.maxrss(),
            setup_validation_seconds=timings.setup_validation,
            precision_preparation_seconds=
                timings.precision_preparation,
            equilibration_seconds=timings.equilibration,
            parameter_selection_seconds=
                timings.parameter_selection,
            workspace_setup_seconds=timings.workspace_setup,
            initialization_seconds=timings.initialization,
            initial_residual_seconds=timings.initial_residual,
            schur_seconds=timings.schur_assembly,
            kkt_seconds=timings.kkt_factorization,
            schur_copy_seconds=timings.kkt_schur_copy,
            factor_seconds=timings.kkt_schur_factorization,
            constraint_triangular_solve_seconds=
                timings.kkt_constraint_triangular_solve,
            equality_gram_seconds=timings.kkt_equality_gram,
            equality_factorization_seconds=
                timings.kkt_equality_factorization,
            kkt_other_seconds=timings.kkt_other,
            predictor_seconds=timings.predictor,
            predictor_rhs_seconds=timings.predictor_rhs,
            predictor_linear_solve_seconds=
                timings.predictor_linear_solve,
            predictor_direction_recovery_seconds=
                timings.predictor_direction_recovery,
            complementarity_analysis_seconds=
                timings.complementarity_analysis,
            corrector_seconds=timings.corrector,
            corrector_rhs_seconds=timings.corrector_rhs,
            corrector_linear_solve_seconds=
                timings.corrector_linear_solve,
            refinement_seconds=timings.refinement,
            corrector_direction_recovery_seconds=
                timings.corrector_direction_recovery,
            residual_seconds=timings.residual_and_block_factor,
            line_search_seconds=timings.line_search,
            update_seconds=timings.update,
            finalization_seconds=timings.finalization,
            other_seconds=timings.other,
            refinement_steps,
            mixed_kkt_backend=get(mixed_kkt, :backend, :unavailable),
            mixed_kkt_condition_estimate=
                get(mixed_kkt, :condition_estimate, NaN),
            mixed_kkt_predicted_refinement_steps=
                get(mixed_kkt, :predicted_refinement_steps, 0),
            mixed_kkt_predictor_refinement_steps=
                get(mixed_kkt, :predictor_refinement_steps, 0),
            mixed_kkt_float64_regularization_attempts=
                get(
                    mixed_kkt,
                    :float64_regularization_attempts,
                    0,
                ),
            mixed_kkt_intermediate_factor_attempts=
                get(mixed_kkt, :intermediate_factor_attempts, 0),
            mixed_kkt_intermediate_refinement_steps=
                get(mixed_kkt, :intermediate_refinement_steps, 0),
            mixed_kkt_intermediate_factor_seconds=
                get(mixed_kkt, :intermediate_factor_seconds, 0.0),
            mixed_kkt_intermediate_solve_seconds=
                get(mixed_kkt, :intermediate_solve_seconds, 0.0),
            mixed_kkt_intermediate_arithmetic=
                get(mixed_kkt, :intermediate_arithmetic, Nothing),
            mixed_kkt_intermediate_storage_bytes=
                get(mixed_kkt, :intermediate_storage_bytes, 0),
            mixed_kkt_factor_attempts=
                get(mixed_kkt, :factor_attempt_count, 0),
            mixed_kkt_dynamic_fallbacks=
                get(mixed_kkt, :dynamic_fallback_count, 0),
            mixed_kkt_static_rejections=
                get(mixed_kkt, :static_rejection_count, 0),
            objective=Float64(result.pObj),
            dual_objective=Float64(result.dObj),
            relative_gap=Float64(result.gap_rel),
            primal_residual=Float64(result.p_res),
            dual_residual=Float64(result.d_res),
            maximum_equality_violation=
                validation.maximum_equality_violation,
            minimum_primal_psd_eigenvalue=
                validation.minimum_primal_psd_eigenvalue,
            minimum_dual_psd_eigenvalue=
                validation.minimum_dual_psd_eigenvalue,
            warnings=join(warnings, '|'),
            selected_algorithms,
        )
        append_result(output, row)
        trace_output = replace(output, r"\.csv$" => ".iterations.csv")
        for trace_row in iteration_trace
            append_result(
                trace_output,
                merge(
                    (
                        arithmetic=row.arithmetic,
                        precision_bits=row.precision_bits,
                        repetition,
                    ),
                    trace_row,
                ),
            )
        end
        @printf(
            "arithmetic=%s threads=%d/%d repetition=%d solve=%.3f status=%s iterations=%d gap=%.3e schur=%.3f kkt=%.3f peak_gib=%.3f\n",
            row.arithmetic,
            row.julia_threads,
            row.blas_threads,
            repetition,
            row.solve_seconds,
            row.status,
            row.iterations,
            row.relative_gap,
            row.schur_seconds,
            row.kkt_seconds,
            row.peak_rss_bytes / 2.0^30,
        )
        flush(stdout)
    end
end

function main(arguments)
    length(arguments) >= 4 || error(
        "usage: benchmark_task_low08_extended_solve.jl INPUT OUTPUT BASIS_IDS " *
        "ARITHMETIC [BLAS_THREADS] [ITERATIONS] [REPETITIONS] [BETA] [GAMMA] " *
        "[EXTENDED_MODE] [MIXED_MODE] [PRECISION_BITS]",
    )
    input = abspath(arguments[1])
    output = abspath(arguments[2])
    basis_path = abspath(arguments[3])
    arithmetic = Symbol(lowercase(arguments[4]))
    arithmetic in (:float64x4, :bigfloat) ||
        error("ARITHMETIC must be float64x4 or bigfloat")
    blas_threads = length(arguments) >= 5 ? parse(Int, arguments[5]) : 1
    iterations = length(arguments) >= 6 ? parse(Int, arguments[6]) : 3
    repetitions = length(arguments) >= 7 ? parse(Int, arguments[7]) : 1
    beta = length(arguments) >= 8 ? parse(Float64, arguments[8]) : 0.075
    gamma = length(arguments) >= 9 ? parse(Float64, arguments[9]) : 0.8
    extended_mode =
        length(arguments) >= 10 ? Symbol(arguments[10]) : :auto
    mixed_mode =
        length(arguments) >= 11 ? Symbol(arguments[11]) : :on
    precision_bits =
        length(arguments) >= 12 ? parse(Int, arguments[12]) : 256
    tolerance = parse(Float64, get(ENV, "EXTENDED_TOLERANCE", "1e-12"))
    mixed_condition_limit = parse(
        Float64,
        get(ENV, "MIXED_CONDITION_LIMIT", "1e8"),
    )
    stall_iterations = parse(
        Int,
        get(ENV, "STALL_ITERATIONS", "15"),
    )
    max_time = parse(
        Float64,
        get(ENV, "EXTENDED_MAX_TIME", "1800"),
    )
    verbosity = parse(
        Int,
        get(ENV, "EXTENDED_VERBOSITY", "0"),
    )

    if arithmetic === :bigfloat
        setprecision(BigFloat, precision_bits) do
            run_case(
                input,
                output,
                basis_path,
                BigFloat;
                precision_bits,
                blas_threads,
                tolerance,
                iterations,
                repetitions,
                beta,
                gamma,
                extended_mode,
                mixed_mode,
                mixed_condition_limit,
                stall_iterations,
                max_time,
                verbosity,
            )
        end
    else
        run_case(
            input,
            output,
            basis_path,
            Float64x4;
            precision_bits,
            blas_threads,
            tolerance,
            iterations,
            repetitions,
            beta,
            gamma,
            extended_mode,
            mixed_mode,
            mixed_condition_limit,
            stall_iterations,
            max_time,
            verbosity,
        )
    end
end

main(ARGS)
