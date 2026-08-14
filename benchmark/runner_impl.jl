const RESULT_COLUMNS = (
    :schema_version, :source_commit, :source_dirty, :julia_version, :os,
    :cpu_name, :julia_threads, :blas_threads, :suite, :problem_id, :name, :family,
    :problem_type, :conic_formulation, :source, :purpose, :seed, :arithmetic,
    :precision_bits, :requested_provider, :status, :reference_status,
    :reference_absolute_tolerance, :reference_relative_tolerance,
    :skip_reason, :termination_reason,
    :variables, :equalities, :blocks, :block_sizes, :planned_formulation,
    :executed_formulation, :planned_backend, :executed_backend,
    :planned_provider, :executed_provider, :fallback_reason,
    :la_fallback_reason, :iterations, :objective, :reference_objective,
    :objective_error, :primal_residual, :dual_residual, :relative_gap,
    :certificate_policy, :certificate_available, :certificate_valid,
    :provider_match, :unexpected_fallback,
    :semantic_pass, :semantic_failures, :total_seconds,
    :assembly_seconds, :factor_seconds, :solve_seconds,
    :refinement_seconds, :input_fingerprint, :external_checksum,
)

_cell(value) = value === missing || value === nothing ? "" :
               replace(string(value), '\t' => ' ', '\n' => ' ', '\r' => ' ')

function _source_commit()
    return try
        readchomp(`git -C $REPOSITORY rev-parse HEAD`)
    catch
        "unknown"
    end
end

function _source_dirty()
    return try
        !isempty(readchomp(`git -C $REPOSITORY status --porcelain`))
    catch
        true
    end
end

_cpu_name() = try
    Sys.CPU_NAME
catch
    "unknown"
end

_blas_threads() = try
    LinearAlgebra.BLAS.get_num_threads()
catch
    missing
end

function _arithmetic_label(::Type{T}) where {T}
    T === Float64 && return :float64
    T === BigFloat && return Symbol("bigfloat", precision(BigFloat))
    if SDPX.is_multifloat_arithmetic(T)
        parameters = Base.unwrap_unionall(T).parameters
        return Symbol("float64x", parameters[2])
    end
    return Symbol(lowercase(string(nameof(T))))
end

_precision_bits(::Type{BigFloat}) = precision(BigFloat)
_precision_bits(::Type{T}) where {T} = SDPX.sig_bits(T)

function _load_multifloat_type(arithmetic::Symbol)
    return get(MULTIFLOAT_TYPES, arithmetic, nothing)
end

function _arithmetic_type(arithmetic::Symbol)
    arithmetic === :float64 && return Float64
    startswith(string(arithmetic), "bigfloat") && return BigFloat
    T = _load_multifloat_type(arithmetic)
    T === nothing && throw(ArgumentError(
        "arithmetic $arithmetic is unavailable; install/load MultiFloats for fixed multiword arithmetic",
    ))
    return T
end

function _load_requested_provider(provider::Symbol)
    if provider === :bfla
        @eval import BigFloatLinearAlgebra
    elseif provider === :multifloat
        @eval import MultiFloatLinearAlgebra
    end
    return nothing
end

function _options(provider::Symbol, ::Type{T}; verbose=false) where {T}
    tolerance = T === Float64 ? 1.0e-8 : T(1.0e-20)
    return SDPX.SolveOptions(
        duality_gap_threshold=tolerance,
        primal_error_threshold=tolerance,
        dual_error_threshold=tolerance,
        maximum_iterations=200,
        threads=1,
        verbosity=0,
        linear_algebra_backend=provider,
        diagnostics=true,
        timing=true,
        certification=true,
    )
end

function _solve_built(built, ::Type{T}, provider; verbose=false) where {T}
    options = _options(provider, T; verbose=verbose)
    if built.kind === :socp
        return SDPX.solve_socp(built.problem, options)
    end
    return SDPX.solve(built.problem, options)
end

function _safe_certificate(problem, result, ::Type{T}) where {T}
    try
        return SDPX.result_certificate(
            problem,
            result,
            SDPX.SolverOptions{T}(
                ϵ_gap=T === Float64 ? T(1e-8) : T(1e-20),
                ϵ_primal=T === Float64 ? T(1e-8) : T(1e-20),
                ϵ_dual=T === Float64 ? T(1e-8) : T(1e-20),
            ),
        )
    catch
        return nothing
    end
end

function _problem_facts(problem)
    canonical = SDPX.Experimental.canonicalize(problem)
    features = SDPX.Experimental.extract_problem_features(canonical)
    blocks = length(features.linear_cones) + length(features.lorentz_cones) +
             length(features.psd_cones)
    block_sizes = vcat(
        [cone.dimension for cone in features.linear_cones],
        [cone.dimension for cone in features.lorentz_cones],
        [cone.dimension for cone in features.psd_cones],
    )
    return (
        variables=features.variables,
        equalities=features.equalities.matrix.rows,
        blocks=blocks,
        block_sizes=Tuple(block_sizes),
    )
end

function _problem_fingerprint(spec, built, arithmetic)
    facts = _problem_facts(built.problem)
    payload = join((
        spec.id,
        string(arithmetic),
        string(spec.seed),
        repr(spec.parameters),
        repr(facts),
        bytes2hex(SHA.sha256(read(joinpath(ROOT, "generators", "problems.jl")))),
    ), "|")
    return bytes2hex(SHA.sha256(payload))
end

_trace_value(value) = SDPX.isavailable(value) ? value : missing
_sym(value) = value isa Symbol ? value : missing

_normalized_status(status) = Symbol(lowercase(string(status)))

function _unexpected_fallback(spec, fallback_reason, la_fallback_reason)
    planned_reasons = (
        :none,
        :not_executed,
        :route_not_migrated,
        :requested_legacy,
        :compatibility,
        :unsupported_arithmetic,
    )
    fallback_reason === missing && return true
    la_fallback_reason === missing && return true
    fallback_reason in planned_reasons || return true
    la_fallback_reason in planned_reasons && return false
    :expected_la_factor_failure in spec.tags &&
        return la_fallback_reason !== :la_equality_factor_failed
    if :rank_deficient in spec.tags || :rank_ladder in spec.tags
        return !(la_fallback_reason in (:la_equality_factor_failed, :la_factor_failed))
    end
    return true
end

function _semantic_failures(
    spec,
    status,
    objective,
    expected,
    primal_residual,
    dual_residual,
    relative_gap,
    certificate,
    provider_match,
    unexpected_fallback,
)
    failures = String[]
    _normalized_status(status) === _normalized_status(spec.reference.status) ||
        push!(failures, "status")
    actual_optimal = _normalized_status(status) === :optimal
    if actual_optimal && expected !== nothing
        error = abs(objective - expected)
        allowed = spec.reference.absolute_tolerance +
                  spec.reference.relative_tolerance * max(one(error), abs(expected))
        (!isfinite(error) || error > allowed) && push!(failures, "objective")
    end
    residual_limit = 10 * max(
        spec.reference.absolute_tolerance,
        spec.reference.relative_tolerance,
    )
    if actual_optimal
        for (name, value) in (
            ("primal_residual", primal_residual),
            ("dual_residual", dual_residual),
            ("relative_gap", relative_gap),
        )
            if value === missing
                push!(failures, name * "_missing")
            elseif !isfinite(value) || abs(value) > residual_limit
                push!(failures, name)
            end
        end
    end
    if actual_optimal && spec.family in (:lp, :socp, :sdp)
        (certificate === nothing || !certificate.valid) &&
            push!(failures, "certificate")
    end
    provider_match || push!(failures, "provider_mismatch")
    unexpected_fallback && push!(failures, "unexpected_fallback")
    return failures
end

function _explicit_provider_matches(requested, executed, ::Type{T}) where {T}
    requested === :auto && return true
    executed === missing && return false
    expected = requested === :multifloat ? :multifloat_linear_algebra :
               requested === :bfla ? :bigfloat_linear_algebra :
               requested === :legacy ? :sdpx_legacy_la :
               requested === :standard ?
                   (T <: Union{Float32,Float64} ? :blas_lapack :
                    :generic_linear_algebra) : requested
    return executed === expected
end

function _formulation(trace, result, planned::Bool)
    diagnostics = result isa SDPX.ConicResult ? result.diagnostics : result.diagnostics
    diagnostics === nothing && return missing
    selected = diagnostics.selected_algorithms
    if planned
        return get(selected, :planned_kkt_formulation, missing)
    end
    return get(selected, :executed_kkt_formulation,
               get(selected, :planned_kkt_formulation, missing))
end

function _result_row(spec, suite, arithmetic, provider, built, result, elapsed)
    trace = SDPX.performance_trace(result)
    certificate = _safe_certificate(built.problem, result, eltype(built.problem))
    objective = trace.final.primal_objective
    expected = built.expected === nothing ? spec.reference.objective : built.expected
    objective_error = expected === nothing ? missing : abs(objective - expected)
    primal_residual = trace.final.primal_residual
    dual_residual = trace.final.dual_residual
    relative_gap = trace.final.relative_gap
    fallback_reason = _trace_value(trace.setup.fallback_reason)
    la_fallback_reason = _trace_value(trace.setup.executed_la_fallback_reason)
    executed_provider = _trace_value(trace.setup.executed_la_provider)
    provider_match = _explicit_provider_matches(
        provider, executed_provider, eltype(built.problem),
    )
    unexpected_fallback = _unexpected_fallback(
        spec, fallback_reason, la_fallback_reason,
    )
    failures = _semantic_failures(
        spec,
        trace.final.status,
        objective,
        expected,
        primal_residual,
        dual_residual,
        relative_gap,
        certificate,
        provider_match,
        unexpected_fallback,
    )
    facts = _problem_facts(built.problem)
    solve_seconds = begin
        predictor = _trace_value(trace.iteration.predictor_solve_seconds)
        corrector = _trace_value(trace.iteration.corrector_solve_seconds)
        predictor === missing || corrector === missing ? missing : predictor + corrector
    end
    return (
        schema_version=1,
        source_commit=_source_commit(),
        source_dirty=_source_dirty(),
        julia_version=string(VERSION),
        os=string(Sys.KERNEL),
        cpu_name=_cpu_name(),
        julia_threads=Threads.nthreads(),
        blas_threads=_blas_threads(),
        suite=suite,
        problem_id=spec.id,
        name=spec.name,
        family=spec.family,
        problem_type=spec.problem_type,
        conic_formulation=built.kind === :socp ? :native_lorentz :
                          spec.family === :lp ? :lp_native : :sdp_native,
        source=spec.source,
        purpose=spec.purpose,
        seed=spec.seed,
        arithmetic=arithmetic,
        precision_bits=_precision_bits(eltype(built.problem)),
        requested_provider=provider,
        status=trace.final.status,
        reference_status=spec.reference.status,
        reference_absolute_tolerance=spec.reference.absolute_tolerance,
        reference_relative_tolerance=spec.reference.relative_tolerance,
        skip_reason=missing,
        termination_reason=_trace_value(trace.final.termination_reason),
        variables=facts.variables,
        equalities=facts.equalities,
        blocks=facts.blocks,
        block_sizes=facts.block_sizes,
        planned_formulation=_formulation(trace, result, true),
        executed_formulation=_formulation(trace, result, false),
        planned_backend=_trace_value(trace.setup.planned_backend),
        executed_backend=_trace_value(trace.setup.executed_backend),
        planned_provider=_trace_value(trace.setup.planned_la_provider),
        executed_provider=executed_provider,
        fallback_reason=fallback_reason,
        la_fallback_reason=la_fallback_reason,
        iterations=trace.counters.iterations,
        objective=string(objective),
        reference_objective=expected === nothing ? missing : string(expected),
        objective_error=objective_error === missing ? missing : string(objective_error),
        primal_residual=string(primal_residual),
        dual_residual=string(dual_residual),
        relative_gap=string(relative_gap),
        certificate_policy=:original_coordinate_required,
        certificate_available=_trace_value(trace.final.certificate_available),
        certificate_valid=certificate === nothing ? missing : certificate.valid,
        provider_match=provider_match,
        unexpected_fallback=unexpected_fallback,
        semantic_pass=isempty(failures),
        semantic_failures=join(failures, ","),
        total_seconds=elapsed,
        assembly_seconds=_trace_value(trace.iteration.schur_assembly_seconds),
        factor_seconds=_trace_value(trace.iteration.kkt_factorization_seconds),
        solve_seconds=solve_seconds,
        refinement_seconds=_trace_value(trace.iteration.refinement_seconds),
        input_fingerprint=_problem_fingerprint(spec, built, arithmetic),
        external_checksum=missing,
    )
end

function _skip_row(spec, suite, arithmetic, provider, reason, checksum=missing)
    values = Dict{Symbol,Any}(field => missing for field in RESULT_COLUMNS)
    merge!(values, Dict(
        :schema_version => 1,
        :source_commit => _source_commit(),
        :source_dirty => _source_dirty(),
        :julia_version => string(VERSION),
        :os => string(Sys.KERNEL),
        :cpu_name => _cpu_name(),
        :julia_threads => Threads.nthreads(),
        :blas_threads => _blas_threads(),
        :suite => suite,
        :problem_id => spec.id,
        :name => spec.name,
        :family => spec.family,
        :problem_type => spec.problem_type,
        :conic_formulation => missing,
        :source => spec.source,
        :purpose => spec.purpose,
        :seed => spec.seed,
        :arithmetic => arithmetic,
        :requested_provider => provider,
        :status => :skipped,
        :reference_status => spec.reference.status,
        :reference_absolute_tolerance => spec.reference.absolute_tolerance,
        :reference_relative_tolerance => spec.reference.relative_tolerance,
        :certificate_policy => :original_coordinate_required,
        :skip_reason => reason,
        :external_checksum => checksum === missing && spec.external !== nothing ?
                              something(spec.external.sha256, missing) : checksum,
        :input_fingerprint => spec.external === nothing ? missing :
                              bytes2hex(SHA.sha256(join((
                                  spec.id,
                                  spec.external.authoritative_url,
                                  string(spec.external.sha256),
                                  string(spec.external.format),
                              ), "|"))),
    ))
    return NamedTuple{RESULT_COLUMNS}(Tuple(values[field] for field in RESULT_COLUMNS))
end

function _error_row(spec, suite, arithmetic, provider, exception)
    skipped = _skip_row(
        spec, suite, arithmetic, provider,
        "execution_error: " * sprint(showerror, exception),
    )
    values = Dict{Symbol,Any}(
        field => getproperty(skipped, field) for field in RESULT_COLUMNS
    )
    values[:status] = :error
    values[:semantic_pass] = false
    values[:semantic_failures] = "execution_error"
    return NamedTuple{RESULT_COLUMNS}(
        Tuple(values[field] for field in RESULT_COLUMNS),
    )
end

function _write_tsv(path, rows)
    open(path, "w") do io
        println(io, join(string.(RESULT_COLUMNS), '\t'))
        for row in rows
            println(io, join((_cell(getproperty(row, field)) for field in RESULT_COLUMNS), '\t'))
        end
    end
end

function _toml_value(value)
    if value === missing || value === nothing
        return ""
    end
    value isa Symbol && return string(value)
    value isa Tuple && return join(string.(value), ",")
    return value
end

function write_results(path::AbstractString, rows)
    root, extension = splitext(path)
    tsv = extension == ".tsv" ? path : root * ".tsv"
    toml = extension == ".toml" ? path : root * ".toml"
    mkpath(dirname(tsv))
    _write_tsv(tsv, rows)
    document = Dict(
        "schema_version" => 1,
        "generated_at" => string(Dates.now()),
        "result" => [Dict(string(field) => _toml_value(getproperty(row, field))
                         for field in RESULT_COLUMNS) for row in rows],
    )
    open(toml, "w") do io
        TOML.print(io, document; sorted=true)
    end
    return (tsv=tsv, toml=toml)
end

function _selected_entries(suite, problem, arithmetic, provider)
    entries = suite_entries(suite)
    problem !== nothing && (entries = filter(e -> e.problem_id == problem, entries))
    arithmetic !== nothing && (entries = [SuiteEntry(e.problem_id, arithmetic,
                                                     e.provider) for e in entries])
    provider !== nothing && (entries = [SuiteEntry(e.problem_id, e.arithmetic,
                                                   provider) for e in entries])
    isempty(entries) && throw(ArgumentError("selection contains no benchmark entries"))
    return entries
end

function run_suite(
    suite::Symbol;
    problem=nothing,
    arithmetic=nothing,
    provider=nothing,
    output=joinpath(ROOT, "out", string(suite), "rows.toml"),
    verbose=false,
    warmup=true,
    strict_semantics=true,
    cache_dir=DEFAULT_CACHE,
)
    suite === :heavy && throw(ArgumentError(
        "the heavy suite is register-only in Round 2 and cannot be executed",
    ))
    entries = _selected_entries(suite, problem, arithmetic, provider)
    rows = NamedTuple[]
    for entry in entries
        spec = benchmark_spec(entry.problem_id)
        if spec.external !== nothing
            cache = external_cache_status(spec; cache_dir=cache_dir)
            push!(rows, _skip_row(
                spec, suite, entry.arithmetic, entry.provider,
                cache.reason, something(cache.checksum, missing),
            ))
            verbose && println("skip ", spec.id, ": ", cache.reason)
            continue
        end
        T = try
            _arithmetic_type(entry.arithmetic)
        catch exception
            push!(rows, _skip_row(
                spec, suite, entry.arithmetic, entry.provider,
                :arithmetic_unavailable,
            ))
            verbose && println("skip ", spec.id, ": ", exception)
            continue
        end
        try
            _load_requested_provider(entry.provider)
        catch exception
            push!(rows, _skip_row(
                spec, suite, entry.arithmetic, entry.provider,
                :provider_unavailable,
            ))
            verbose && println("skip ", spec.id, ": ", exception)
            continue
        end
        run = function ()
            built = build_problem(spec, T)
            result = _solve_built(
                built, T, entry.provider; verbose=verbose,
            )
            return built, result
        end
        local row
        try
            if T === BigFloat
                bits = parse(Int, replace(string(entry.arithmetic), "bigfloat" => ""))
                row = setprecision(BigFloat, bits) do
                    warmup && Base.invokelatest(run)
                    elapsed = @elapsed built, result = Base.invokelatest(run)
                    _result_row(spec, suite, entry.arithmetic, entry.provider,
                                built, result, elapsed)
                end
            else
                warmup && Base.invokelatest(run)
                elapsed = @elapsed built, result = Base.invokelatest(run)
                row = _result_row(spec, suite, entry.arithmetic, entry.provider,
                                  built, result, elapsed)
            end
        catch exception
            row = _error_row(
                spec, suite, entry.arithmetic, entry.provider, exception,
            )
        end
        push!(rows, row)
        if verbose
            elapsed_label = row.total_seconds isa Real ?
                string(round(row.total_seconds; digits=3), "s") : "unavailable"
            println(spec.id, " ", row.status, " ", elapsed_label)
        end
    end
    paths = write_results(output, rows)
    failed = filter(row -> row.status != :skipped && row.semantic_pass === false, rows)
    if strict_semantics && !isempty(failed)
        details = join(("$(row.problem_id):$(row.semantic_failures)" for row in failed), "; ")
        throw(ErrorException(
            "semantic regression in $(length(failed)) benchmark result(s): $details; " *
            "results were written to $(paths.toml)",
        ))
    end
    return (rows=rows, paths=paths)
end

function _parse_symbol(value)
    value === nothing && return nothing
    return Symbol(lowercase(strip(value)))
end

function _parse_cli(args)
    suite = :micro
    problem = nothing
    arithmetic = nothing
    provider = nothing
    output = nothing
    verbose = false
    prepare = false
    positional = String[]
    for argument in args
        if startswith(argument, "--problem=")
            problem = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--arithmetic=")
            arithmetic = _parse_symbol(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--provider=")
            provider = _parse_symbol(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--output=")
            output = split(argument, "="; limit=2)[2]
        elseif argument == "--verbose"
            verbose = true
        elseif argument == "--prepare"
            prepare = true
        elseif startswith(argument, "--")
            throw(ArgumentError("unknown option $argument"))
        else
            push!(positional, argument)
        end
    end
    !isempty(positional) && (suite = Symbol(lowercase(first(positional))))
    length(positional) > 1 && problem === nothing && (problem = positional[2])
    return (; suite, problem, arithmetic, provider, output, verbose, prepare)
end

function main(args=ARGS)
    options = _parse_cli(args)
    if options.prepare
        options.suite === :heavy && throw(ArgumentError(
            "the heavy suite is register-only; prepare selected non-heavy cases explicitly",
        ))
        ids = options.problem === nothing ?
              [entry.problem_id for entry in suite_entries(options.suite)
               if benchmark_spec(entry.problem_id).external !== nothing] :
              [options.problem]
        return prepare_external!(ids; verbose=options.verbose)
    end
    output = something(
        options.output,
        joinpath(ROOT, "out", string(options.suite), "rows.toml"),
    )
    result = run_suite(
        options.suite;
        problem=options.problem,
        arithmetic=options.arithmetic,
        provider=options.provider,
        output=output,
        verbose=options.verbose,
    )
    solved = count(row -> !(row.status in (:skipped, :error)), result.rows)
    errors = count(row -> row.status === :error, result.rows)
    skipped = count(row -> row.status === :skipped, result.rows)
    println("suite=", options.suite, " solved=", solved,
            " skipped=", skipped, " errors=", errors,
            " output=", result.paths.toml)
    return result
end
