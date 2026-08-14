include(joinpath(@__DIR__, "run.jl"))
using MultiFloats: Float64x2, Float64x3, Float64x4
using Statistics

function _reduced_dual_target_type(name::AbstractString)
    normalized = lowercase(strip(name))
    normalized == "float64" && return Float64
    normalized == "float64x2" && return Float64x2
    normalized == "float64x3" && return Float64x3
    normalized == "float64x4" && return Float64x4
    error("SDPX_BENCH_ARITHMETIC must be float64/float64x2/float64x3/float64x4")
end

_reduced_dual_provider(::Type{Float64}) = :standard
_reduced_dual_provider(::Type) = :multifloat

function _reduced_dual_warmup(::Type{T}, backend_request, threads) where {T}
    A = sparse([2, 3], [1, 2], T[one(T), one(T)], 3, 2)
    problem = SDPX.second_order_program(
        T[zero(T), zero(T)],
        [SDPX.SOCConstraint(A, T[one(T), zero(T), zero(T)]; T)];
        Aeq=reshape(T[one(T), zero(T)], 1, 2),
        beq=T[T(1) / T(2)],
        T,
    )
    SDPX.solve_value(
        problem;
        arithmetic=T,
        tolerance=T(1e-6),
        smoothing=T[T(1e-3), T(1e-6)],
        linear_algebra_backend=backend_request,
        threads,
        maximum_iterations=40,
        timing=false,
        diagnostics=false,
    )
    return nothing
end

function _fg_microbenchmark(problem, backend_request, threads, tau; warmups=2, reps=10)
    T = eltype(problem)
    layout = SDPX._compile_fixed_trace_q3_dual(problem)
    plan = SDPX._plan_reduced_dual(T, backend_request, threads)
    backend = SDPX.instantiate_la_backend(plan.la_config, T, threads)
    workspace = SDPX._fixed_trace_dual_workspace(layout)
    y = SDPX.alloc_zeros(T, length(problem.beq))
    for _ in 1:warmups
        SDPX._fixed_trace_dual_evaluate!(
            layout, backend, y, tau, workspace.u, workspace.x,
            workspace.gradient, workspace.w, workspace.rho, workspace.wnorm,
        )
    end
    samples = Float64[]
    allocations = Int[]
    bytes = Int[]
    for _ in 1:reps
        measured = @timed SDPX._fixed_trace_dual_evaluate!(
            layout, backend, y, tau, workspace.u, workspace.x,
            workspace.gradient, workspace.w, workspace.rho, workspace.wnorm,
        )
        push!(samples, measured.time)
        push!(allocations, measured.gcstats.malloc + measured.gcstats.realloc +
                           measured.gcstats.poolalloc + measured.gcstats.bigalloc)
        push!(bytes, measured.bytes)
    end
    return (
        minimum=minimum(samples), median=median(samples),
        allocations_min=minimum(allocations), bytes_min=minimum(bytes),
        warmups, repetitions=reps,
    )
end

function _write_reduced_dual_report(path, facts)
    open(path, "w") do io
        println(io, "# Reduced-dual L-BFGS FixedTrace CSDR result\n")
        println(io, "```text")
        for key in sort!(collect(keys(facts)))
            println(io, rpad(key, 38), facts[key])
        end
        println(io, "```\n")
        println(io, "The solve used the analytic FixedTraceQ3 reduced dual.")
        println(io, "No PSD lift, Gram matrix, HVP, CG, or hidden IPM fallback was used.")
    end
end

function reduced_dual_main(args=ARGS)
    model_path = isempty(args) ?
        "/tmp/csdr-fixedtrace-reduced-neutral.bin" : abspath(args[1])
    output_dir = length(args) >= 2 ? abspath(args[2]) :
        joinpath(@__DIR__, "..", "..", "out", "bootstrap_full_unitraity_eft_reduced_dual")
    mkpath(output_dir)
    payload = open(deserialize, model_path)
    payload isa NamedTuple && hasproperty(payload, :schema) &&
        payload.schema === :csdr_fixed_trace_reduced_v1 || error(
            "run_reduced_dual.jl requires the neutral reduced payload",
        )
    target_type = _reduced_dual_target_type(
        get(ENV, "SDPX_BENCH_ARITHMETIC", "float64"),
    )
    converted_payload = _convert_neutral_arithmetic(payload, target_type)
    problem, elimination = _native_fixed_trace_problem(converted_payload, nothing)
    T = eltype(problem)
    tolerance = parse(T, get(ENV, "SDPX_BENCH_TOLERANCE", "1e-12"))
    smoothing = if haskey(ENV, "SDPX_BENCH_TAU_SCHEDULE")
        T[parse(T, strip(value)) for value in
          split(ENV["SDPX_BENCH_TAU_SCHEDULE"], ',')]
    else
        SDPX._reduced_dual_schedule(T, :auto, tolerance)
    end
    maximum_iterations = parse(Int, get(ENV, "SDPX_BENCH_MAX_ITERATIONS", "200"))
    max_time = parse(Float64, get(ENV, "SDPX_BENCH_MAX_TIME", "900"))
    threads = parse(Int, get(ENV, "SDPX_BENCH_SOLVER_THREADS", "1"))
    backend_request = _reduced_dual_provider(T)
    require_optimal = lowercase(get(
        ENV, "SDPX_BENCH_REQUIRE_OPTIMAL",
        maximum_iterations >= 200 ? "true" : "false",
    )) in ("1", "true", "yes")

    micro = _fg_microbenchmark(
        problem, backend_request, threads, smoothing[1];
        warmups=parse(Int, get(ENV, "SDPX_BENCH_FG_WARMUPS", "2")),
        reps=parse(Int, get(ENV, "SDPX_BENCH_FG_REPETITIONS", "10")),
    )
    _reduced_dual_warmup(T, backend_request, threads)
    timed = @timed SDPX.solve_value(
        problem;
        arithmetic=T, soc_algorithm=:reduced_dual_lbfgs,
        tolerance, smoothing, polish=:none,
        linear_algebra_backend=backend_request, threads,
        history_size=parse(Int, get(ENV, "SDPX_BENCH_LBFGS_HISTORY", "10")),
        maximum_iterations, max_time, timing=true, diagnostics=true,
    )
    result = timed.value
    reconstruction_started = time()
    reconstructed = result.reconstruction_token === nothing ? nothing :
        SDPX.reconstruct_fixed_trace_solution(problem, result)
    reconstruction_seconds = time() - reconstruction_started
    physical = reconstructed === nothing ? T(NaN) :
        _physical_objective(converted_payload, elimination, reconstructed.x)
    reference = parse(T, "30.4732058529286002611344264526887472")
    reference_upper = parse(T, "30.4732058529379858416561499463130361")
    in_interval = reference <= physical <= reference_upper
    expected_provider = T === Float64 ? :blas_lapack : :multifloat_linear_algebra
    production_invariants =
        result.specialization === :fixed_trace_q3 &&
        result.provider === expected_provider &&
        result.termination.no_psd_lift &&
        isempty(result.termination.fallback_chain) &&
        result.counters.ipm_polish_iterations == 0
    full_gate = result.status === SDPX.Optimal && result.certificate.valid &&
                in_interval && result.relative_gap <= tolerance && production_invariants
    allocation_count = timed.gcstats.malloc + timed.gcstats.realloc +
                       timed.gcstats.poolalloc + timed.gcstats.bigalloc
    facts = Dict{String,Any}(
        "schema_version" => 1,
        "status" => string(result.status),
        "arithmetic" => string(result.arithmetic),
        "precision_bits" => result.precision_bits,
        "provider" => string(result.provider),
        "algorithm" => string(result.algorithm),
        "specialization" => string(result.specialization),
        "solve_seconds" => timed.time,
        "setup_seconds" => result.timings.reduced_dual_setup,
        "fg_min_seconds" => micro.minimum,
        "fg_median_seconds" => micro.median,
        "fg_warmups" => micro.warmups,
        "fg_repetitions" => micro.repetitions,
        "fg_allocations_min" => micro.allocations_min,
        "fg_bytes_min" => micro.bytes_min,
        "lbfgs_iterations" => result.counters.lbfgs_iterations,
        "lbfgs_history_size" => parse(Int, get(
            ENV, "SDPX_BENCH_LBFGS_HISTORY", "10",
        )),
        "termination_reason" => string(result.termination.reason),
        "optimizer_reason" => string(result.termination.optimizer_reason),
        "final_gradient_norm" => string(isempty(result.smoothing_history) ?
            T(Inf) : result.smoothing_history[end].gradient_norm),
        "smoothing_schedule" => string.(smoothing),
        "smoothing_history" => [Dict(
            "tau" => string(stage.tau),
            "iterations" => stage.iterations,
            "objective" => string(stage.objective),
            "gradient_norm" => string(stage.gradient_norm),
            "reason" => string(stage.reason),
        ) for stage in result.smoothing_history],
        "objective_evaluations" => result.counters.objective_evaluations,
        "gradient_evaluations" => result.counters.gradient_evaluations,
        "line_search_trials" => result.counters.line_search_trials,
        "accepted_steps" => result.counters.accepted_steps,
        "seconds_per_accepted_iteration" => timed.time /
            max(result.counters.accepted_steps, 1),
        "transpose_gemv_seconds" => result.timings.equality_transpose_gemv,
        "forward_gemv_seconds" => result.timings.equality_forward_gemv,
        "block_support_seconds" => result.timings.block_support,
        "certificate_seconds" => result.timings.certificate,
        "reconstruction_seconds" => reconstruction_seconds,
        "solve_allocations" => allocation_count,
        "solve_allocated_bytes" => timed.bytes,
        "solve_gc_seconds" => timed.gctime,
        "objective" => string(result.objective),
        "physical_objective" => string(physical),
        "reference_objective" => string(reference),
        "historical_interval_upper" => string(reference_upper),
        "objective_in_historical_interval" => in_interval,
        "certificate_valid" => result.certificate.valid,
        "certificate_failures" => join(string.(result.certificate.failures), ","),
        "certificate_interval_lower" => string(result.lower),
        "certificate_interval_upper" => string(result.upper),
        "interval_kind" => string(result.interval_kind),
        "rigorous_interval" => result.rigorous_interval,
        "relative_gap" => string(result.relative_gap),
        "ipm_polish" => string(result.polish),
        "strict_spectrum_reconstruction" => false,
        "production_invariants_valid" => production_invariants,
        "full_numerical_gate_valid" => full_gate,
        "model_path" => model_path,
        "model_fingerprint" => bytes2hex(collect(result.reconstruction_token.problem_fingerprint)),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "solver_threads" => threads,
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "source_commit" => get(ENV, "SDPX_BENCH_SOURCE", "working_tree"),
    )
    open(joinpath(output_dir, "result.toml"), "w") do io
        TOML.print(io, facts; sorted=true)
    end
    _write_reduced_dual_report(joinpath(output_dir, "RESULTS.md"), facts)
    println("Reduced-dual CSDR result written to $output_dir")
    production_invariants || error("reduced-dual production invariant failed")
    require_optimal && !full_gate && error("reduced-dual full numerical gate failed")
    return facts
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && reduced_dual_main()
