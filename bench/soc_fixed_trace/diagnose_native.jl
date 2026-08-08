#!/usr/bin/env julia

# Run the compact fixed-trace Q3 core without the public PSD2 fallback. This
# is a diagnostic driver, not a benchmark row: it preserves the native status,
# phase timings, iteration history, and original-coordinate certificate needed
# to understand why promotion failed.

include(joinpath(@__DIR__, "benchmark.jl"))

function _native_diagnostic(config::Config, ::Type{T}) where {T}
    config.mode === :socp || error("native diagnostics require --mode=socp")
    config.preflight_only && error("native diagnostics perform one direct solve")
    BLAS.set_num_threads(1)

    model_hash = config.synthetic ? "synthetic" : _sha256_file(config.model)
    raw_model = config.synthetic ? nothing : _load_payload(config.model, config.release)
    problem = config.synthetic ?
              synthetic_problem(T; blocks=config.synthetic_blocks, sparse=config.sparse) :
              _problem_as_type(raw_model, T, config.sparse)
    preflight = _preflight!(problem, config, model_hash)
    options = _solver_options(T, config)

    context_before = _process_context_switches()
    cpu_before = _process_cpu_seconds()
    started = time_ns()
    result = SDPX._solve_fixed_trace_q3_core!(
        problem,
        options;
        deadline=time() + config.time_limit_seconds,
    )
    wall_seconds = (time_ns() - started) / 1.0e9
    cpu_after = _process_cpu_seconds()
    context_after = _process_context_switches()
    cpu_seconds = isfinite(cpu_before) && isfinite(cpu_after) ?
                  max(0.0, cpu_after - cpu_before) : NaN
    certificate = SDPX.result_certificate(problem, result, options)

    row = Dict{String,Any}(
        "status" => string(result.status),
        "message" => result.message,
        "iterations" => result.iterations,
        "wall_seconds" => wall_seconds,
        "cpu_seconds" => cpu_seconds,
        "mean_active_cores" => wall_seconds > 0 ? cpu_seconds / wall_seconds : 0.0,
        "process_peak_rss_bytes_after" => Sys.maxrss(),
        "objective_primal" => string(certificate.primal_objective),
        "objective_dual" => string(certificate.dual_objective),
        "certificate_valid" => certificate.valid,
        "certificate_failures" => string.(certificate.failures),
        "certificate_gap_relative" => string(certificate.gap_relative),
        "certificate_primal_residual" => string(certificate.primal_residual),
        "certificate_dual_residual" => string(certificate.dual_residual),
        "certificate_equality_backward_error" =>
            string(certificate.equality_backward_error),
        "certificate_primal_block_backward_error" =>
            string(certificate.primal_block_backward_error),
        "certificate_dual_backward_error" =>
            string(certificate.dual_backward_error),
        "voluntary_context_switches" =>
            context_before.voluntary >= 0 && context_after.voluntary >= 0 ?
            context_after.voluntary - context_before.voluntary : -1,
        "involuntary_context_switches" =>
            context_before.involuntary >= 0 && context_after.involuntary >= 0 ?
            context_after.involuntary - context_before.involuntary : -1,
    )
    result.timings === nothing || _flatten!(row, result.timings, "phase")
    result.termination === nothing ||
        _flatten!(row, result.termination, "termination")

    history = [
        Dict(string(name) => _toml_scalar(value) for (name, value) in pairs(entry))
        for entry in result.parameter_history
    ]
    metadata = resource_metadata(config)
    payload = Dict{String,Any}(
        "case" => config.case,
        "mode" => "native_q3_diagnostic",
        "arithmetic" => config.arithmetic,
        "model_sha256" => model_hash,
        "geometry" => Dict(
            "blocks" => problem.dims.L,
            "variables" => problem.dims.m,
            "equalities" => problem.dims.n,
        ),
        "preflight" => Dict(
            string(name) => _toml_scalar(value)
            for (name, value) in pairs(preflight)
            if name !== :block_dimensions
        ),
        "resources" => Dict(
            key => _toml_scalar(value) for (key, value) in metadata
        ),
        "result" => Dict(key => _toml_scalar(value) for (key, value) in row),
        "parameter_history" => history,
    )
    mkpath(dirname(abspath(config.output)))
    open(config.output, "w") do io
        TOML.print(io, payload; sorted=true)
    end
    _write_manifest(
        config.manifest,
        config,
        model_hash,
        geometry(problem),
        metadata,
        preflight,
        [row],
    )
    println(
        "native_q3 status=$(result.status) iterations=$(result.iterations) " *
        "wall_seconds=$wall_seconds cpu_seconds=$cpu_seconds " *
        "certificate_valid=$(certificate.valid)",
    )
    println("message=$(result.message)")
    println("certificate_failures=$(certificate.failures)")
    return result.status === SDPX.Optimal && certificate.valid ? 0 : 3
end

function diagnostic_main(args=ARGS)
    config = parse_cli(args)
    isempty(config.output) && error("native diagnostics require --output")
    T = arithmetic_type(config.arithmetic)
    code = _with_precision(
        () -> _native_diagnostic(config, T),
        T,
        config.precision_bits,
    )
    exit(code)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && diagnostic_main()
