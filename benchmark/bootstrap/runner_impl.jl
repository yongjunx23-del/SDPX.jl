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

_hostname() = try
    gethostname()
catch
    "unknown"
end

function _sha256_if_file(path)
    path === nothing && return missing
    isfile(path) || return missing
    return _sha256_file(path)
end

function _manifest_sha256()
    project = Base.active_project()
    project === nothing && return missing
    return _sha256_if_file(joinpath(dirname(project), "Manifest.toml"))
end

const _SOLVER_SOURCE_SHA256_CACHE = Ref{Union{Nothing,String}}(nothing)
const _SHA256_HEX64 = r"^[0-9a-f]{64}$"

_valid_sha256(value) = value isa AbstractString &&
    occursin(_SHA256_HEX64, strip(value))

_harness_source_sha256() = _sha256_file(joinpath(ROOT, "PhysicsBenchmarkHarness.jl"))
_schema_source_sha256() = _sha256_file(joinpath(ROOT, "result_schema.jl"))

function _callable_source_files(callable)
    files = String[]
    method_list = try
        methods(callable)
    catch
        nothing
    end
    method_list === nothing && return files
    for method in method_list
        file = String(method.file)
        (isempty(file) || file == "none") && continue
        absolute = isabspath(file) ? file : abspath(file)
        isfile(absolute) && push!(files, realpath(absolute))
    end
    return sort!(unique!(files))
end

function _catalog_source_sha256(catalog)
    injected = get(_CATALOG_FILE_CONTEXT, catalog, nothing)
    injected === nothing || return _catalog_manifest_sha256(injected)
    files = unique!(vcat(
        _callable_source_files(catalog.build),
        _callable_source_files(catalog.validate),
    ))
    payload = IOBuffer()
    write(payload, string(catalog.name), UInt8(0), catalog.version, UInt8(0))
    for id in sort!(collect(keys(catalog.specs)))
        spec = catalog.specs[id]
        write(payload, id, UInt8(0), spec.fingerprint, UInt8(0), repr(spec), UInt8(0))
    end
    for suite in sort!(collect(keys(catalog.suites)); by=string)
        write(payload, string(suite), UInt8(0), repr(catalog.suites[suite]), UInt8(0))
    end
    for path in sort!(files)
        write(payload, replace(path, '\\' => '/'), UInt8(0), read(path), UInt8(0xff))
    end
    return bytes2hex(SHA.sha256(take!(payload)))
end

function _callable_contract_sources(catalog, built)
    callables = Any[catalog.build]
    contract = _solve_contract(built)
    contract === nothing || push!(callables, contract)
    files = String[]
    for callable in callables
        append!(files, _callable_source_files(callable))
    end
    return sort!(unique!(files))
end

function _contract_source_payload(catalog, spec, built)
    files = _callable_contract_sources(catalog, built)
    isempty(files) && throw(ArgumentError(
        "contract fingerprint for $(spec.id) cannot verify callable source",
    ))
    context = get(_CATALOG_FILE_CONTEXT, catalog, nothing)
    if context !== nothing
        for file in files
            _path_under_root(file, context.root) || throw(ArgumentError(
                "contract callable source escapes catalog root: $file",
            ))
        end
    end
    payload = IOBuffer()
    for file in files
        write(payload, replace(file, '\\' => '/'), UInt8(0), read(file), UInt8(0xff))
    end
    return bytes2hex(SHA.sha256(take!(payload)))
end

function _derived_contract_fingerprint(catalog, spec, built, route)
    checksum = _built_value(built, :external_checksum, spec.fingerprint)
    declared = _built_value(built, :contract_fingerprint, nothing)
    declared === nothing || (_valid_sha256(declared) || throw(ArgumentError(
        "contract_fingerprint for $(spec.id) must be lowercase SHA-256",
    )))
    solve_reference = _built_value(built, :solve_reference, nothing)
    solve_settings = _solve_settings(built)
    source_payload = _contract_source_payload(catalog, spec, built)
    payload = join((
        "sdpx-benchmark-contract-v2",
        source_payload,
        _catalog_source_sha256(catalog),
        spec.id,
        spec.fingerprint,
        string(checksum),
        repr(solve_reference),
        repr(solve_settings),
        string(route),
        declared === nothing ? "" : strip(String(declared)),
    ), "|")
    return bytes2hex(SHA.sha256(payload))
end

function _contract_fingerprint(catalog, spec, built, route)
    return _derived_contract_fingerprint(catalog, spec, built, route)
end

function _safe_contract_fingerprint(catalog, spec, built, route)
    built === nothing && return missing
    return try
        _contract_fingerprint(catalog, spec, built, route)
    catch
        missing
    end
end

_safe_catalog_source_sha256(catalog) = try
    _catalog_source_sha256(catalog)
catch
    missing
end

"""Hash the exact Julia solver source tree used by this process.

`source_commit` plus `source_dirty` cannot distinguish two local optimization
candidates based on the same commit.  This content hash is stable across file
timestamps and directory traversal order, and deliberately excludes benchmark
drivers so solver and measurement identities remain separate.
"""
function _solver_source_sha256()
    cached = _SOLVER_SOURCE_SHA256_CACHE[]
    cached === nothing || return cached
    source_root = joinpath(REPOSITORY, "src")
    paths = String[]
    for (directory, _, files) in walkdir(source_root)
        for file in files
            endswith(file, ".jl") || continue
            push!(paths, joinpath(directory, file))
        end
    end
    sort!(paths)
    payload = IOBuffer()
    for path in paths
        relative = replace(relpath(path, source_root), '\\' => '/')
        write(payload, relative)
        write(payload, UInt8(0))
        write(payload, read(path))
        write(payload, UInt8(0xff))
    end
    digest = bytes2hex(SHA.sha256(take!(payload)))
    _SOLVER_SOURCE_SHA256_CACHE[] = digest
    return digest
end

function _package_commit(package::AbstractString)
    source = try
        Base.find_package(package)
    catch
        nothing
    end
    source === nothing && return :not_loaded
    root = normpath(joinpath(dirname(source), ".."))
    return try
        readchomp(`git -C $root rev-parse HEAD`)
    catch
        :not_a_git_checkout
    end
end

_built_value(built, name::Symbol, default) =
    hasproperty(built, name) ? getproperty(built, name) : default

function _solve_settings(built)
    return _built_value(built, :solve_settings, (;))
end

const _EXECUTION_MODES = (:build, :solve, :profile)

function _normalize_execution_mode(mode)
    value = Symbol(lowercase(replace(string(mode), '-' => '_', ' ' => '_')))
    value in _EXECUTION_MODES || throw(ArgumentError(
        "execution_mode must be one of $(_EXECUTION_MODES), got $(repr(mode))",
    ))
    return value
end

function _normalize_engine(engine)
    value = Symbol(lowercase(replace(string(engine), '-' => '_', ' ' => '_')))
    # Native HSD labels are accepted for construction-only rows so a future
    # catalog can reserve an engine identity without importing an unstable
    # public API.  Solve/profile reject those labels at the run boundary.
    value in (:auto, :legacy, :sdpx_legacy, :catalog_contract,
              :native, :native_hsd, :product_hsd) ||
        throw(ArgumentError("unsupported benchmark engine $(repr(engine))"))
    return value
end

_metadata_value(value, environment_key, default) =
    value === nothing ? get(ENV, environment_key, default) : value

function _execution_identity(; campaign_id=nothing, shard_id=nothing,
                             shard_index=nothing, shard_count=nothing)
    identity = (
        campaign_id=_metadata_value(campaign_id, "SDPX_CAMPAIGN_ID", "standalone"),
        shard_id=_metadata_value(shard_id, "SDPX_SHARD_ID", "local"),
        shard_index=_metadata_value(shard_index, "SDPX_SHARD_INDEX", "1"),
        shard_count=_metadata_value(shard_count, "SDPX_SHARD_COUNT", "1"),
        pbs_array_index=get(ENV, "PBS_ARRAY_INDEX", missing),
        pbs_queue=get(ENV, "PBS_QUEUE", missing),
        pbs_node=get(ENV, "PBS_NODENUM", get(ENV, "HOSTNAME", missing)),
    )
    for name in (:campaign_id, :shard_id, :shard_index, :shard_count)
        isempty(strip(string(getproperty(identity, name)))) && throw(ArgumentError(
            "$(name) must be non-empty for reproducible benchmark identity",
        ))
    end
    return identity
end

function _validate_shard_identity(identity)
    parse_positive(name) = try
        value = parse(Int, string(getproperty(identity, name)))
        value > 0 || throw(ArgumentError("$name must be a positive integer"))
        value
    catch exception
        exception isa ArgumentError && rethrow()
        throw(ArgumentError(
            "$name must be a positive integer, got $(repr(getproperty(identity, name)))",
        ))
    end
    index = parse_positive(:shard_index)
    count = parse_positive(:shard_count)
    index <= count || throw(ArgumentError(
        "shard_index must be <= shard_count (got $index > $count)",
    ))
    return (;
        identity...,
        shard_index=index,
        shard_count=count,
    )
end

function _solve_contract(built)
    contract = _built_value(built, :solve_contract, nothing)
    contract === nothing && return nothing
    return contract
end

_latest_applicable(callable, arguments...) =
    Base.invokelatest(applicable, callable, arguments...)

function _solve_contract_applicable(contract, built, ::Type{T}, provider, mode) where {T}
    return _latest_applicable(contract, built, T, provider, mode) ||
           _latest_applicable(contract, built, T, provider) ||
           _latest_applicable(contract, built, T)
end

function _invoke_solve_contract(contract, built, ::Type{T}, provider, mode) where {T}
    result = if _latest_applicable(contract, built, T, provider, mode)
        Base.invokelatest(contract, built, T, provider, mode)
    elseif _latest_applicable(contract, built, T, provider)
        Base.invokelatest(contract, built, T, provider)
    elseif _latest_applicable(contract, built, T)
        Base.invokelatest(contract, built, T)
    else
        throw(ArgumentError(
            "solve_contract must accept (built, T, provider, execution_mode), " *
            "(built, T, provider), or (built, T)",
        ))
    end
    # A contract may return `(result=..., metadata=...)` or `(built, result)`
    # so catalog authors can attach independent oracle metadata without
    # changing the harness route.  The common case is a solver result itself.
    if result isa Tuple && length(result) == 2
        return result[2]
    elseif hasproperty(result, :result)
        return getproperty(result, :result)
    end
    return result
end

function _solve_reference(built, spec)
    raw = _built_value(built, :solve_reference, nothing)
    raw === nothing && throw(ArgumentError(
        "catalog solve_contract for $(spec.id) must declare solve_reference",
    ))
    reference = if raw isa PhysicsBenchmarkReference
        raw
    elseif raw isa NamedTuple && hasproperty(raw, :status)
        PhysicsBenchmarkReference(
            status=raw.status,
            objective=get(raw, :objective, nothing),
            absolute_tolerance=Float64(get(raw, :absolute_tolerance, 1.0e-8)),
            relative_tolerance=Float64(get(raw, :relative_tolerance, 1.0e-8)),
            note=String(get(raw, :note, "")),
        )
    else
        throw(ArgumentError(
            "solve_reference for $(spec.id) must be PhysicsBenchmarkReference " *
            "or a named tuple containing status",
        ))
    end
    reference.status in (:build_only, :sampled_build_only) && throw(ArgumentError(
        "solve_reference for $(spec.id) must describe a solve result, not " *
        "$(reference.status)",
    ))
    return reference
end

function _validate_build_only_declaration(spec, built)
    tagged = :build_only in spec.tags
    referenced = spec.reference.status in (:build_only, :sampled_build_only)
    setting = get(_solve_settings(built), :build_only, false)
    setting isa Bool || throw(ArgumentError(
        "solve_settings.build_only for $(spec.id) must be Bool",
    ))
    tagged == referenced == setting || throw(ArgumentError(
        "benchmark $(spec.id) must declare build-only consistently in tags, " *
        "reference.status, and solve_settings.build_only",
    ))
    return tagged
end

function _resolve_engine_route(catalog, built, spec, execution_mode,
                               requested_engine, declared_build_only)
    execution_mode === :build && return :none
    contract = _solve_contract(built)
    has_contract = contract !== nothing
    route = if requested_engine === :auto
        has_contract ? :catalog_contract : :sdpx_legacy
    elseif requested_engine === :catalog_contract
        has_contract || throw(ArgumentError(
            "requested_engine=:catalog_contract requires solve_contract",
        ))
        :catalog_contract
    elseif requested_engine in (:legacy, :sdpx_legacy)
        declared_build_only && throw(ArgumentError(
            "build-only benchmark cannot execute the legacy solver route",
        ))
        :sdpx_legacy
    elseif requested_engine in (:native, :native_hsd, :product_hsd)
        throw(ArgumentError(
            "requested engine $(repr(requested_engine)) has no benchmark solve adapter yet",
        ))
    else
        throw(ArgumentError("unsupported benchmark engine $(repr(requested_engine))"))
    end
    if declared_build_only && route !== :catalog_contract
        throw(ArgumentError(
            "benchmark is build-only and has no catalog solve contract route",
        ))
    end
    if route === :catalog_contract
        _solve_reference(built, spec)
        _contract_fingerprint(catalog, spec, built, route)
    end
    return route
end

function _scaling_label(built)
    value = _built_value(built, :scaling, nothing)
    value === nothing && (value = _built_value(built, :benchmark_scale, missing))
    return value === nothing || value === missing ? :unspecified : value
end

function _layout_label(built)
    value = _built_value(built, :layout, nothing)
    value === nothing && (value = _built_value(built, :kind, missing))
    return value === nothing || value === missing ? :unspecified : value
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

function _options(provider::Symbol, ::Type{T}, built=nothing; verbose=false) where {T}
    settings = built === nothing ? (;) : _solve_settings(built)
    default_tolerance = T === Float64 ? T(1.0e-8) : T(1.0e-20)
    tolerance = hasproperty(settings, :tolerance) ?
                parse(T, string(settings.tolerance)) : default_tolerance
    maximum_iterations = get(settings, :maximum_iterations, 200)
    max_time = get(settings, :max_time, Inf)
    return SDPX.SolveOptions(
        duality_gap_threshold=tolerance,
        primal_error_threshold=tolerance,
        dual_error_threshold=tolerance,
        maximum_iterations=maximum_iterations,
        max_runtime=max_time,
        threads=Threads.nthreads(),
        verbosity=0,
        linear_algebra_backend=provider,
        diagnostics=true,
        timing=true,
        certification=true,
    )
end

function _solve_built(built, ::Type{T}, provider; verbose=false) where {T}
    options = _options(provider, T, built; verbose=verbose)
    # Native product-HSD production path: the harness builds a typed
    # Model/Settings/Outputs through the SDPX entrypoint bridge and calls the
    # public `optimize!` seam (engine=:native_hsd); the result is adapted back
    # to the legacy SDPResult/ConicResult schema in original coordinates.
    # No legacy solve!/interior_point/lp_solver route is reachable here, and
    # the executed route/provider is reported through the adapted diagnostics.
    if built.kind === :socp
        return SDPX._bridge_conic_solve(built.problem, options)
    end
    return SDPX._bridge_sdp_solve(built.problem, options)
end

function _safe_certificate(problem, result, ::Type{T}, built=nothing) where {T}
    settings = built === nothing ? (;) : _solve_settings(built)
    tolerance = hasproperty(settings, :tolerance) ?
                parse(T, string(settings.tolerance)) :
                (T === Float64 ? T(1e-8) : T(1e-20))
    try
        return SDPX.result_certificate(
            problem,
            result,
            SDPX.SolverOptions{T}(
                ϵ_gap=tolerance,
                ϵ_primal=tolerance,
                ϵ_dual=tolerance,
            ),
        )
    catch
        return nothing
    end
end

function _problem_facts(problem)
    if problem isa SDPX.SDPProblem
        return (
            variables=problem.dims.m,
            equalities=problem.dims.n,
            blocks=problem.dims.L,
            block_sizes=Tuple(problem.dims.k),
        )
    elseif problem isa SDPX.ConicProblem
        return (
            variables=length(problem.c),
            equalities=size(problem.Aeq, 1),
            blocks=length(problem.cones),
            block_sizes=Tuple(size(cone.A, 1) for cone in problem.cones),
        )
    elseif problem isa SDPX.CanonicalConicProgram
        barrier_blocks = filter(
            block -> !(SDPX.block_cone(block) in (:zero, :free)),
            SDPX.layout_blocks(problem.cone_layout),
        )
        equality_rows = sum(
            SDPX.block_length(block)
            for block in SDPX.layout_blocks(problem.cone_layout)
            if SDPX.block_cone(block) === :zero
        )
        return (
            variables=SDPX.canonical_num_variables(problem),
            equalities=equality_rows,
            blocks=length(barrier_blocks),
            block_sizes=Tuple(SDPX.block_length(block) for block in barrier_blocks),
        )
    end
    throw(ArgumentError(
        "benchmark facts require SDPProblem, ConicProblem, or " *
        "CanonicalConicProgram, got $(typeof(problem))",
    ))
end

_problem_eltype(problem::SDPX.CanonicalConicProgram) = eltype(problem.c)
_problem_eltype(problem) = eltype(problem)

function _problem_fingerprint(catalog, spec, built, arithmetic)
    facts = _problem_facts(built.problem)
    payload = join((
        string(catalog.name),
        catalog.version,
        spec.id,
        string(arithmetic),
        string(spec.seed),
        repr(spec.parameters),
        repr(facts),
        string(_built_value(built, :external_checksum, spec.fingerprint)),
        spec.fingerprint,
    ), "|")
    return bytes2hex(SHA.sha256(payload))
end

_trace_value(value) = SDPX.isavailable(value) ? value : missing
_trace_field(record, name::Symbol) = hasproperty(record, name) ?
    _trace_value(getproperty(record, name)) : missing
_sym(value) = value isa Symbol ? value : missing

"""Render a recorded accuracy value without narrowing its arithmetic type."""
function _string_metric(value)
    (value === missing || value === nothing) && return missing
    SDPX.isavailable(value) || return missing
    return string(value)
end

function _string_failures(value)
    (value === missing || value === nothing) && return missing
    SDPX.isavailable(value) || return missing
    if value isa AbstractVector || value isa Tuple
        return join(string.(value), ",")
    end
    return string(value)
end

function _certificate_field(certificate, name::Symbol)
    (certificate === nothing || certificate === missing) && return missing
    hasproperty(certificate, name) || return missing
    return getproperty(certificate, name)
end

function _first_recorded(values...)
    for value in values
        (value === missing || value === nothing) && continue
        SDPX.isavailable(value) || continue
        return value
    end
    return missing
end

function _solver_version()
    return try
        string(Base.pkgversion(SDPX))
    catch
        missing
    end
end

function _normalized_status(status)
    token = Symbol(lowercase(replace(
        string(status),
        '-' => '_',
        ' ' => '_',
    )))
    return get((
        notstarted=:not_started,
        feasiblecert=:feasible_certificate,
        infeasiblecert=:infeasible_certificate,
        iterlimit=:iteration_limit,
        timelimit=:time_limit,
        numericalbreakdown=:numerical_breakdown,
        maxrestartsexceeded=:max_restarts_exceeded,
        userstopped=:user_stopped,
        almostoptimal=:almost_optimal,
        insufficientprecision=:insufficient_precision,
        numericalfailure=:numerical_failure,
        primalinfeasible=:primal_infeasible,
        dualinfeasible=:dual_infeasible,
    ), token, token)
end

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
    reference,
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
    _normalized_status(status) === _normalized_status(reference.status) ||
        push!(failures, "status")
    normalized_status = _normalized_status(status)
    actual_optimal = normalized_status === :optimal
    actual_infeasible = normalized_status in (
        :primal_infeasible,
        :dual_infeasible,
    )
    if actual_optimal && expected !== nothing
        error = abs(objective - expected)
        allowed = reference.absolute_tolerance +
                  reference.relative_tolerance * max(one(error), abs(expected))
        (!isfinite(error) || error > allowed) && push!(failures, "objective")
    end
    residual_limit = 10 * max(
        reference.absolute_tolerance,
        reference.relative_tolerance,
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
    if (actual_optimal || actual_infeasible) &&
       spec.family in (:lp, :socp, :sdp)
        (certificate === nothing || !certificate.valid) &&
            push!(failures, "certificate")
    end
    if actual_infeasible && certificate !== nothing && certificate.valid
        expected_kind = normalized_status === :primal_infeasible ?
                        :primal_infeasibility : :dual_infeasibility
        hasproperty(certificate, :kind) &&
            certificate.kind !== expected_kind &&
            push!(failures, "certificate_kind")
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

function _specialization(result)
    diagnostics = result.diagnostics
    diagnostics === nothing && return missing
    selected = diagnostics.selected_algorithms
    return get(selected, :soc_specialization, missing)
end

function _physical_objective(built, result, default)
    objective_map = _built_value(built, :physical_objective, nothing)
    objective_map === nothing && return default
    return objective_map(result.x)
end

function _reference_interval(built, ::Type{T}) where {T}
    interval = _built_value(built, :objective_interval, nothing)
    interval === nothing && return nothing
    return (
        lower=parse(T, string(interval.lower)),
        upper=parse(T, string(interval.upper)),
    )
end

function _result_row(
    catalog, spec, suite, arithmetic, provider, built, result, elapsed;
    allocated_bytes=missing,
    gc_seconds=missing,
    execution_mode=:solve,
    requested_engine=:auto,
    executed_engine,
    reference,
    campaign_id="standalone",
    shard_id="local",
    shard_index="1",
    shard_count="1",
    pbs_array_index=missing,
    pbs_queue=missing,
    pbs_node=missing,
)
    execution_mode = _normalize_execution_mode(execution_mode)
    requested_engine = _normalize_engine(requested_engine)
    executed_engine in (:sdpx_legacy, :catalog_contract) || throw(ArgumentError(
        "solve/profile rows require an explicitly resolved executed_engine",
    ))
    trace = SDPX.performance_trace(result)
    T = eltype(built.problem)
    certificate = _safe_certificate(built.problem, result, T, built)
    native_objective = trace.final.primal_objective
    objective = _physical_objective(built, result, native_objective)
    trace_dual_objective = _trace_value(trace.final.dual_objective)
    certificate_dual_objective = _certificate_field(certificate, :dual_objective)
    dual_objective = _first_recorded(
        trace_dual_objective, certificate_dual_objective,
    )
    certificate_gap = _certificate_field(certificate, :gap)
    absolute_gap_value = if certificate_gap !== missing &&
                              certificate_gap !== nothing &&
                              SDPX.isavailable(certificate_gap)
        abs(certificate_gap)
    elseif native_objective !== missing &&
           trace_dual_objective !== missing &&
           SDPX.isavailable(native_objective) &&
           SDPX.isavailable(trace_dual_objective)
        abs(native_objective - trace_dual_objective)
    else
        missing
    end
    primal_tolerance = _certificate_field(certificate, :primal_residual_limit)
    dual_tolerance = _certificate_field(certificate, :dual_residual_limit)
    gap_tolerance = _certificate_field(certificate, :gap_limit)
    certificate_kind = _first_recorded(
        _certificate_field(certificate, :kind),
        _trace_value(trace.final.certificate_kind),
    )
    certificate_failures = _first_recorded(
        _certificate_field(certificate, :failures),
        _trace_value(trace.final.certificate_failures),
    )
    primal_affine_residual = _certificate_field(
        certificate, :primal_affine_residual,
    )
    dual_affine_residual = _certificate_field(
        certificate, :dual_affine_residual,
    )
    primal_cone_violation = _certificate_field(
        certificate, :primal_cone_violation,
    )
    dual_cone_violation = _certificate_field(
        certificate, :dual_cone_violation,
    )
    primal_residual_scaled = _certificate_field(
        certificate, :primal_residual_scaled,
    )
    dual_residual_scaled = _certificate_field(
        certificate, :dual_residual_scaled,
    )
    complementarity = _certificate_field(certificate, :complementarity)
    relative_complementarity = _certificate_field(
        certificate, :complementarity_relative,
    )
    interval = _reference_interval(built, T)
    interval_pass = interval === nothing ? missing :
                    interval.lower <= objective <= interval.upper
    expected = executed_engine === :catalog_contract ? reference.objective :
               (built.expected === nothing ? reference.objective : built.expected)
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
        reference,
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
    catalog_failures = validate_result(
        catalog, spec, built, result,
        (
            status=trace.final.status,
            objective,
            expected,
            primal_residual,
            dual_residual,
            relative_gap,
            certificate,
        ),
    )
    append!(failures, catalog_failures)
    specialization = _specialization(result)
    required_specialization = _built_value(
        built, :required_specialization, nothing,
    )
    required_specialization !== nothing &&
        specialization !== required_specialization &&
        push!(failures, "specialization")
    # NativeHSDDiagnostics is the positive evidence that a ConicResult came
    # from the product-HSD bridge. Reference PSD helpers retain SDP diagnostics,
    # so the benchmark gate can still detect a lift.
    psd_lift_used =
        result isa SDPX.ConicResult &&
        result.diagnostics !== nothing &&
        !(result.diagnostics isa SDPX.NativeHSDDiagnostics)
    _built_value(built, :forbid_psd_lift, false) && psd_lift_used &&
        push!(failures, "psd_lift")
    interval_pass === false && push!(failures, "objective_interval")
    maximum_relative_gap = _built_value(
        built, :maximum_relative_gap, nothing,
    )
    if maximum_relative_gap !== nothing
        limit = parse(T, string(maximum_relative_gap))
        (relative_gap === missing || !isfinite(relative_gap) ||
         abs(relative_gap) > limit) && push!(failures, "relative_gap_strict")
    end
    production_invariants =
        (required_specialization === nothing ||
         specialization === required_specialization) &&
        provider_match && !psd_lift_used && !unexpected_fallback
    if _built_value(built, :require_no_fallback, false)
        exact_no_fallback = fallback_reason === :none && la_fallback_reason === :none
        exact_no_fallback || push!(failures, "fallback")
        production_invariants &= exact_no_fallback
    end
    facts = _problem_facts(built.problem)
    contract_fingerprint = _contract_fingerprint(
        catalog, spec, built, executed_engine,
    )
    solve_seconds = begin
        predictor = _trace_value(trace.iteration.predictor_solve_seconds)
        corrector = _trace_value(trace.iteration.corrector_solve_seconds)
        predictor === missing || corrector === missing ? missing : predictor + corrector
    end
    return (
        schema_version=RESULT_SCHEMA_VERSION,
        source_commit=_source_commit(),
        source_dirty=_source_dirty(),
        julia_version=string(VERSION),
        os=string(Sys.KERNEL),
        cpu_name=_cpu_name(),
        hostname=_hostname(),
        pbs_job_id=get(ENV, "PBS_JOBID", missing),
        julia_threads=Threads.nthreads(),
        blas_threads=_blas_threads(),
        project_sha256=_sha256_if_file(Base.active_project()),
        manifest_sha256=_manifest_sha256(),
        benchmark_driver_sha256=_sha256_file(joinpath(ROOT, "runner_impl.jl")),
        solver_source_sha256=_solver_source_sha256(),
        catalog_source_sha256=_catalog_source_sha256(catalog),
        harness_source_sha256=_harness_source_sha256(),
        schema_source_sha256=_schema_source_sha256(),
        contract_fingerprint,
        mfla_commit=_package_commit("MultiFloatLinearAlgebra"),
        bfla_commit=_package_commit("BigFloatLinearAlgebra"),
        solver_name=_string_metric(_trace_value(trace.setup.solver)),
        solver_version=_solver_version(),
        catalog_name=catalog.name,
        catalog_version=catalog.version,
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
        campaign_id,
        shard_id,
        shard_index,
        shard_count,
        pbs_array_index,
        pbs_queue,
        pbs_node,
        execution_mode,
        requested_engine,
        executed_engine,
        scaling=_scaling_label(built),
        layout=_layout_label(built),
        status=trace.final.status,
        reference_status=reference.status,
        reference_absolute_tolerance=reference.absolute_tolerance,
        reference_relative_tolerance=reference.relative_tolerance,
        skip_reason=missing,
        termination_reason=_trace_value(trace.final.termination_reason),
        termination_stage=_trace_value(trace.final.termination_stage),
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
        executed_specialization=specialization,
        psd_lift_used,
        fallback_reason=fallback_reason,
        la_fallback_reason=la_fallback_reason,
        iterations=trace.counters.iterations,
        objective=string(objective),
        reference_objective=expected === nothing ? missing : string(expected),
        physical_objective=
            _built_value(built, :physical_objective, nothing) === nothing ?
            missing : string(objective),
        objective_interval_lower=interval === nothing ? missing : string(interval.lower),
        objective_interval_upper=interval === nothing ? missing : string(interval.upper),
        objective_in_reference_interval=interval_pass,
        benchmark_scale=_built_value(built, :benchmark_scale, missing),
        input_generation_precision_bits=_built_value(
            built, :input_generation_precision_bits, missing,
        ),
        original_equalities=_built_value(built, :original_equalities, missing),
        source_parameters=begin
            parameters = _built_value(built, :source_parameters, nothing)
            parameters === nothing ? missing : repr(parameters)
        end,
        objective_error=objective_error === missing ? missing : string(objective_error),
        dual_objective=_string_metric(dual_objective),
        absolute_gap=_string_metric(absolute_gap_value),
        primal_tolerance=_string_metric(primal_tolerance),
        dual_tolerance=_string_metric(dual_tolerance),
        gap_tolerance=_string_metric(gap_tolerance),
        certificate_kind=certificate_kind,
        certificate_failures=_string_failures(certificate_failures),
        primal_affine_residual=_string_metric(primal_affine_residual),
        dual_affine_residual=_string_metric(dual_affine_residual),
        primal_cone_violation=_string_metric(primal_cone_violation),
        dual_cone_violation=_string_metric(dual_cone_violation),
        primal_residual_scaled=_string_metric(primal_residual_scaled),
        dual_residual_scaled=_string_metric(dual_residual_scaled),
        complementarity=_string_metric(complementarity),
        relative_complementarity=_string_metric(relative_complementarity),
        primal_residual=_string_metric(primal_residual),
        dual_residual=_string_metric(dual_residual),
        relative_gap=_string_metric(relative_gap),
        certificate_policy=:original_coordinate_required,
        certificate_available=_trace_value(trace.final.certificate_available),
        certificate_valid=certificate === nothing ? missing : certificate.valid,
        provider_match=provider_match,
        unexpected_fallback=unexpected_fallback,
        production_invariants_valid=production_invariants,
        full_numerical_gate_valid=isempty(failures),
        catalog_validation_pass=isempty(catalog_failures),
        catalog_validation_failures=join(catalog_failures, ","),
        semantic_pass=isempty(failures),
        semantic_failures=join(failures, ","),
        total_seconds=elapsed,
        seconds_per_iteration=elapsed / max(trace.counters.iterations, 1),
        total_seconds_iqr=missing,
        allocated_bytes,
        gc_seconds,
        allocated_bytes_iqr=missing,
        setup_seconds=_trace_value(trace.setup.setup_seconds),
        frontend_seconds=_trace_value(trace.setup.frontend_seconds),
        preprocess_seconds=_trace_value(trace.setup.preprocess_seconds),
        presolve_seconds=_trace_value(trace.setup.presolve_seconds),
        core_seconds=_trace_value(trace.setup.core_seconds),
        certification_seconds=_trace_value(trace.final.certification_seconds),
        workspace_bytes=_trace_value(trace.final.workspace_bytes),
        process_peak_rss_bytes=_trace_value(trace.final.process_peak_rss_bytes),
        rss_bytes=_trace_value(trace.final.process_peak_rss_bytes),
        workspace_bytes_iqr=missing,
        process_peak_rss_bytes_iqr=missing,
        rss_iqr_bytes=missing,
        memory_budget_bytes=_trace_value(trace.final.memory_budget_bytes),
        restarts=_trace_field(trace.counters, :restarts),
        regularizations=_trace_field(trace.counters, :regularizations),
        refinement_solves=_trace_field(trace.counters, :refinement_solves),
        numeric_factorizations=_trace_field(
            trace.counters, :numeric_factorizations,
        ),
        factorization_attempts=_trace_field(
            trace.counters, :factorization_attempts,
        ),
        factorization_successes=_trace_field(
            trace.counters, :factorization_successes,
        ),
        factorization_failures=_trace_field(
            trace.counters, :factorization_failures,
        ),
        sample_count=1,
        sample_seconds=missing,
        sample_semantic_pass=missing,
        sample_status=missing,
        sample_iterations=missing,
        sample_objective=missing,
        sample_certificate_valid=missing,
        sample_route=missing,
        sample_semantic_parity=missing,
        sample_parity_failures=missing,
        sample_median_seconds=missing,
        sample_min_seconds=missing,
        sample_max_seconds=missing,
        sample_mad_seconds=missing,
        sample_spread_seconds=missing,
        assembly_seconds=_trace_value(trace.iteration.schur_assembly_seconds),
        factor_seconds=_trace_value(trace.iteration.kkt_factorization_seconds),
        solve_seconds=solve_seconds,
        refinement_seconds=_trace_value(trace.iteration.refinement_seconds),
        local_metric_seconds=_trace_value(trace.iteration.fixed_local_metric_seconds),
        local_factor_seconds=_trace_value(trace.iteration.fixed_local_factor_seconds),
        panel_transform_seconds=
            _trace_value(trace.iteration.equality_panel_transform_seconds),
        equality_gram_seconds=
            _trace_value(trace.iteration.equality_gram_syrk_seconds),
        equality_factor_seconds=
            _trace_value(trace.iteration.equality_factor_seconds),
        predictor_rhs_seconds=_trace_value(trace.iteration.predictor_rhs_seconds),
        corrector_rhs_seconds=_trace_value(trace.iteration.corrector_rhs_seconds),
        block_residual_seconds=
            _trace_value(trace.iteration.fixed_block_residual_seconds),
        block_recovery_seconds=
            _trace_value(trace.iteration.fixed_block_recovery_seconds),
        local_metric_preparations=
            _trace_field(trace.counters, :local_metric_preparations),
        equality_gram_assemblies=
            _trace_field(trace.counters, :equality_gram_assemblies),
        equality_factorizations=
            _trace_field(trace.counters, :equality_factorizations),
        rhs_solves=_trace_field(trace.counters, :rhs_solves),
        phase_cold_setup_seconds=_trace_value(trace.phases.cold_setup_seconds),
        phase_equality_rank_seconds=
            _trace_value(trace.phases.equality_rank_seconds),
        phase_symbolic_seconds=_trace_value(trace.phases.symbolic_seconds),
        phase_assembly_seconds=_trace_value(trace.phases.assembly_seconds),
        phase_numeric_factor_seconds=
            _trace_value(trace.phases.numeric_factor_seconds),
        phase_predictor_solve_seconds=
            _trace_value(trace.phases.predictor_solve_seconds),
        phase_corrector_rhs_seconds=
            _trace_value(trace.phases.corrector_rhs_seconds),
        phase_corrector_solve_seconds=
            _trace_value(trace.phases.corrector_solve_seconds),
        phase_refinement_seconds=
            _trace_value(trace.phases.refinement_seconds),
        phase_line_search_seconds=
            _trace_value(trace.phases.line_search_seconds),
        phase_state_update_seconds=
            _trace_value(trace.phases.state_update_seconds),
        phase_certificate_seconds=
            _trace_value(trace.phases.certificate_seconds),
        phase_reference_seconds=_trace_value(trace.phases.reference_seconds),
        phase_accounted_seconds=_trace_value(trace.phases.accounted_seconds),
        phase_unaccounted_seconds=
            _trace_value(trace.phases.unaccounted_seconds),
        phase_consistent=_trace_value(trace.phases.consistent),
        attempt_count=_trace_value(trace.attempts.count),
        attempt_elapsed_seconds=
            _trace_value(trace.attempts.elapsed_seconds_total),
        attempt_fallback_events=
            _trace_value(trace.attempts.fallback_events_total),
        input_fingerprint=_problem_fingerprint(catalog, spec, built, arithmetic),
        external_checksum=_built_value(built, :external_checksum, spec.fingerprint),
    )
end

function _json_escape(value)
    escaped = replace(string(value), "\\" => "\\\\", "\"" => "\\\"")
    return replace(escaped, "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
end

function _sample_json_list(values)
    isempty(values) && return missing
    payload = join((
        value isa AbstractString ? "\"" * _json_escape(value) * "\"" :
        value === true ? "true" :
        value === false ? "false" :
        value === nothing || value === missing ? "null" :
        repr(value)
        for value in values
    ), ",")
    return "[" * payload * "]"
end

function _median(values)
    ordered = sort!(collect(Float64, values))
    isempty(ordered) && return NaN
    midpoint = (length(ordered) + 1) ÷ 2
    return isodd(length(ordered)) ? ordered[midpoint] :
           (ordered[midpoint] + ordered[midpoint + 1]) / 2
end

function _sampling_summary(sample_seconds)
    seconds = [Float64(value) for value in sample_seconds]
    ordered = sort(seconds)
    center = _median(ordered)
    deviations = sort(abs.(ordered .- center))
    return (
        median_seconds=center,
        min_seconds=first(ordered),
        max_seconds=last(ordered),
        mad_seconds=_median(deviations),
        spread_seconds=last(ordered) - first(ordered),
        iqr_seconds=quantile(ordered, 0.75) - quantile(ordered, 0.25),
    )
end

function _sample_iqr(rows, field)
    values = Float64[]
    for row in rows
        value = getproperty(row, field)
        value isa Real && isfinite(value) && value >= 0 || return missing
        push!(values, Float64(value))
    end
    isempty(values) && return missing
    ordered = sort!(values)
    return quantile(ordered, 0.75) - quantile(ordered, 0.25)
end

function _sample_route_key(row)
    fields = (
        :execution_mode, :requested_engine, :executed_engine,
        :scaling, :layout,
        :conic_formulation, :planned_formulation, :executed_formulation,
        :planned_backend, :executed_backend, :planned_provider,
        :executed_provider, :executed_specialization, :psd_lift_used,
        :fallback_reason, :la_fallback_reason,
    )
    return join((_cell(getproperty(row, field)) for field in fields), "|")
end

const _SAMPLE_IDENTITY_FIELDS = (
    :input_fingerprint, :external_checksum,
    :catalog_name, :catalog_version,
    :project_sha256, :manifest_sha256,
    :benchmark_driver_sha256, :solver_source_sha256,
    :catalog_source_sha256, :harness_source_sha256,
    :schema_source_sha256, :contract_fingerprint,
    :execution_mode, :requested_engine, :executed_engine,
    :scaling, :layout,
    :campaign_id, :shard_id, :shard_index, :shard_count,
    :pbs_job_id, :pbs_array_index, :pbs_queue, :pbs_node,
)

const _SAMPLE_REQUIRED_NONEMPTY_FIELDS = (
    :input_fingerprint, :external_checksum,
    :catalog_name, :catalog_version,
    :project_sha256, :manifest_sha256,
    :benchmark_driver_sha256, :solver_source_sha256,
    :catalog_source_sha256, :harness_source_sha256,
    :schema_source_sha256, :contract_fingerprint,
    :execution_mode, :requested_engine, :executed_engine,
    :scaling, :layout, :campaign_id, :shard_id,
    :shard_index, :shard_count,
)

function _reference_tolerance(row)
    absolute = getproperty(row, :reference_absolute_tolerance)
    relative = getproperty(row, :reference_relative_tolerance)
    if absolute isa Real && isfinite(absolute) && absolute >= 0 &&
       relative isa Real && isfinite(relative) && relative >= 0
        return string(absolute), string(relative)
    end
    return nothing, nothing
end

function _parity_precision_bits(values...)
    lengths = Int[]
    for value in values
        value === nothing && continue
        push!(lengths, length(string(value)))
    end
    return max(256, ceil(Int, 4 * maximum(lengths; init=0)))
end

function _objective_parity(rows)
    # Semantic agreement, not bitwise string equality: exact equality is the
    # fast path, otherwise compare through BigFloat against the benchmark
    # row's reference absolute/relative tolerances. Missing/unusable
    # tolerances fall back to exact equality; unparseable or non-finite
    # objective values fail closed. Precision is derived from the decimal
    # string lengths (>=256 bits) and scoped locally, so long BigFloat
    # objectives are never collapsed at the ambient working precision.
    raw_objectives = [getproperty(row, :objective) for row in rows]
    unavailable(value) = value === missing || value === nothing
    all(unavailable, raw_objectives) && return (ok=true, message="")
    any(unavailable, raw_objectives) && return (
        ok=false, message="objective_missing",
    )
    try
        objective_strings = [
            string(value) for value in raw_objectives
        ]
        absolute, relative = _reference_tolerance(first(rows))
        bits = _parity_precision_bits(
            objective_strings..., absolute, relative,
        )
        return setprecision(BigFloat, bits) do
            numeric = [
                parse(BigFloat, objective)
                for objective in objective_strings
            ]
            all(isfinite, numeric) || return (
                ok=false, message="objective_nonfinite",
            )
            first_value = first(numeric)
            all(value -> value == first_value, numeric) && return (
                ok=true, message="",
            )
            if absolute === nothing || relative === nothing
                return (
                    ok=false,
                    message="objective_differ_no_reference_tolerance",
                )
            end
            absolute_tolerance = parse(BigFloat, absolute)
            relative_tolerance = parse(BigFloat, relative)
            if all(numeric) do value
                error = abs(value - first_value)
                allowed = absolute_tolerance + relative_tolerance *
                          max(one(BigFloat), abs(first_value))
                error <= allowed
            end
                return (ok=true, message="")
            end
            return (ok=false, message="objective")
        end
    catch exception
        return (
            ok=false,
            message="objective_unparseable",
        )
    end
end

function _sample_parity(rows)
    failures = String[]
    statuses = [_cell(getproperty(row, :status)) for row in rows]
    iterations = [getproperty(row, :iterations) for row in rows]
    objectives = [_cell(getproperty(row, :objective)) for row in rows]
    certificates = [_cell(getproperty(row, :certificate_valid)) for row in rows]
    routes = [_sample_route_key(row) for row in rows]
    length(unique(statuses)) == 1 || push!(failures, "status")
    length(unique(iterations)) == 1 || push!(failures, "iterations")
    objective_parity = _objective_parity(rows)
    objective_parity.ok || push!(failures, objective_parity.message)
    length(unique(certificates)) == 1 || push!(failures, "certificate")
    length(unique(routes)) == 1 || push!(failures, "route")
    for field in _SAMPLE_IDENTITY_FIELDS
        values = [_cell(getproperty(row, field)) for row in rows]
        length(unique(values)) == 1 || push!(failures, string(field))
    end
    for field in _SAMPLE_REQUIRED_NONEMPTY_FIELDS
        any(isempty, (_cell(getproperty(row, field)) for row in rows)) &&
            push!(failures, string(field) * "_missing")
    end
    for field in (
        :project_sha256, :manifest_sha256,
        :benchmark_driver_sha256, :solver_source_sha256,
        :catalog_source_sha256, :harness_source_sha256,
        :schema_source_sha256, :contract_fingerprint,
    )
        all(row -> _valid_sha256(_cell(getproperty(row, field))), rows) ||
            push!(failures, string(field) * "_invalid")
    end
    all(row -> row.semantic_pass, rows) || push!(failures, "semantic_pass")
    return (
        parity=isempty(failures),
        failures=join(failures, ","),
        statuses=statuses,
        iterations=iterations,
        objectives=objectives,
        certificates=certificates,
        routes=routes,
    )
end

function _sampling_row(rows, sample_seconds; sample_count=length(rows))
    summary = _sampling_summary(sample_seconds)
    selected = findmin(abs.(sample_seconds .- summary.median_seconds))[2]
    parity = _sample_parity(rows)
    base = rows[selected]
    values = Dict{Symbol,Any}(
        :sample_count => sample_count,
        :sample_seconds => _sample_json_list(sample_seconds),
        :sample_semantic_pass =>
            _sample_json_list([row.semantic_pass for row in rows]),
        :sample_status => _sample_json_list(parity.statuses),
        :sample_iterations => _sample_json_list(parity.iterations),
        :sample_objective => _sample_json_list(parity.objectives),
        :sample_certificate_valid =>
            _sample_json_list(parity.certificates),
        :sample_route => _sample_json_list(parity.routes),
        :sample_semantic_parity => parity.parity,
        :sample_parity_failures => parity.failures,
        :sample_median_seconds => summary.median_seconds,
        :sample_min_seconds => summary.min_seconds,
        :sample_max_seconds => summary.max_seconds,
        :sample_mad_seconds => summary.mad_seconds,
        :sample_spread_seconds => summary.spread_seconds,
        :total_seconds_iqr => summary.iqr_seconds,
        :allocated_bytes_iqr => _sample_iqr(rows, :allocated_bytes),
        :process_peak_rss_bytes_iqr =>
            _sample_iqr(rows, :process_peak_rss_bytes),
        :rss_iqr_bytes => _sample_iqr(rows, :rss_bytes),
        :workspace_bytes_iqr => _sample_iqr(rows, :workspace_bytes),
        # The aggregate row represents the sample distribution.  Keep the
        # legacy scalar timing field aligned with the reported median even
        # when an even number of samples requires averaging the two middle
        # observations (and therefore no individual sample has that value).
        :total_seconds => summary.median_seconds,
        :seconds_per_iteration => base.iterations isa Real &&
                                  isfinite(base.iterations) ?
            summary.median_seconds / max(base.iterations, 1) : missing,
    )
    return NamedTuple{RESULT_COLUMNS}(
        Tuple(
            haskey(values, field) ? values[field] : getproperty(base, field)
            for field in RESULT_COLUMNS
        ),
    )
end

function _selection_fingerprint(catalog, spec, arithmetic)
    return bytes2hex(SHA.sha256(join((
        string(catalog.name), catalog.version, spec.id, string(arithmetic),
        spec.fingerprint,
    ), "|")))
end

function _safe_problem_fingerprint(catalog, spec, built, arithmetic)
    built === nothing && return _selection_fingerprint(catalog, spec, arithmetic)
    return try
        _problem_fingerprint(catalog, spec, built, arithmetic)
    catch
        _selection_fingerprint(catalog, spec, arithmetic)
    end
end

function _skip_row(catalog, spec, suite, arithmetic, provider, reason;
                   execution_mode=:solve,
                   requested_engine=:auto,
                   campaign_id="standalone",
                   shard_id="local",
                   shard_index="1",
                   shard_count="1",
                   pbs_array_index=missing,
                   pbs_queue=missing,
                   pbs_node=missing,
                   executed_engine=:none,
                   built=nothing)
    execution_mode = _normalize_execution_mode(execution_mode)
    requested_engine = _normalize_engine(requested_engine)
    values = Dict{Symbol,Any}(field => missing for field in RESULT_COLUMNS)
    merge!(values, Dict(
        :schema_version => RESULT_SCHEMA_VERSION,
        :source_commit => _source_commit(),
        :source_dirty => _source_dirty(),
        :julia_version => string(VERSION),
        :os => string(Sys.KERNEL),
        :cpu_name => _cpu_name(),
        :hostname => _hostname(),
        :pbs_job_id => get(ENV, "PBS_JOBID", missing),
        :julia_threads => Threads.nthreads(),
        :blas_threads => _blas_threads(),
        :project_sha256 => _sha256_if_file(Base.active_project()),
        :manifest_sha256 => _manifest_sha256(),
        :benchmark_driver_sha256 => _sha256_file(joinpath(ROOT, "runner_impl.jl")),
        :solver_source_sha256 => _solver_source_sha256(),
        :catalog_source_sha256 => _safe_catalog_source_sha256(catalog),
        :harness_source_sha256 => _harness_source_sha256(),
        :schema_source_sha256 => _schema_source_sha256(),
        :contract_fingerprint => _safe_contract_fingerprint(
            catalog, spec, built, executed_engine,
        ),
        :mfla_commit => _package_commit("MultiFloatLinearAlgebra"),
        :bfla_commit => _package_commit("BigFloatLinearAlgebra"),
        :catalog_name => catalog.name,
        :catalog_version => catalog.version,
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
        :campaign_id => campaign_id,
        :shard_id => shard_id,
        :shard_index => shard_index,
        :shard_count => shard_count,
        :pbs_array_index => pbs_array_index,
        :pbs_queue => pbs_queue,
        :pbs_node => pbs_node,
        :execution_mode => execution_mode,
        :requested_engine => requested_engine,
        :executed_engine => executed_engine,
        :scaling => built === nothing ? missing : _scaling_label(built),
        :layout => built === nothing ? missing : _layout_label(built),
        :requested_provider => provider,
        :status => :skipped,
        :reference_status => spec.reference.status,
        :reference_absolute_tolerance => spec.reference.absolute_tolerance,
        :reference_relative_tolerance => spec.reference.relative_tolerance,
        :certificate_policy => :original_coordinate_required,
        :skip_reason => reason,
        :input_fingerprint => _safe_problem_fingerprint(
            catalog, spec, built, arithmetic,
        ),
        :external_checksum => built === nothing ? missing :
            _built_value(built, :external_checksum, spec.fingerprint),
    ))
    return NamedTuple{RESULT_COLUMNS}(Tuple(values[field] for field in RESULT_COLUMNS))
end

function _construction_row(
    catalog, spec, suite, arithmetic, provider, built, elapsed;
    allocated_bytes=missing,
    gc_seconds=missing,
    execution_mode=:build,
    requested_engine=:auto,
    campaign_id="standalone",
    shard_id="local",
    shard_index="1",
    shard_count="1",
    pbs_array_index=missing,
    pbs_queue=missing,
    pbs_node=missing,
    declared_build_only=false,
)
    execution_mode = _normalize_execution_mode(execution_mode)
    execution_mode === :build || throw(ArgumentError(
        "construction rows require execution_mode=:build",
    ))
    requested_engine = _normalize_engine(requested_engine)
    _contract_fingerprint(catalog, spec, built, :none)
    facts = _problem_facts(built.problem)
    catalog_failures = validate_result(
        catalog, spec, built, nothing,
        (status=spec.reference.status, build_only=declared_build_only,
         execution_mode=execution_mode),
    )
    base = _skip_row(
        catalog, spec, suite, arithmetic, provider, :build_only;
        execution_mode=execution_mode,
        requested_engine=requested_engine,
        campaign_id=campaign_id,
        shard_id=shard_id,
        shard_index=shard_index,
        shard_count=shard_count,
        pbs_array_index=pbs_array_index,
        pbs_queue=pbs_queue,
        pbs_node=pbs_node,
        executed_engine=:none,
        built=built,
    )
    formulation = built.kind === :socp ? :native_lorentz :
                  built.kind === :exp ? :native_exponential :
                  built.kind === :power ? :native_power :
                  built.kind === :product ? :native_product :
                  spec.family === :lp ? :lp_native : :sdp_native
    values = Dict{Symbol,Any}(
        :skip_reason => missing,
        :status => declared_build_only ? spec.reference.status : :built,
        :termination_reason => :model_built,
        :termination_stage => :construction,
        :precision_bits => _precision_bits(_problem_eltype(built.problem)),
        :conic_formulation => formulation,
        :variables => facts.variables,
        :equalities => facts.equalities,
        :blocks => facts.blocks,
        :block_sizes => facts.block_sizes,
        :certificate_policy => declared_build_only ?
                               :not_applicable_build_only :
                               :not_applicable_build,
        :catalog_validation_pass => isempty(catalog_failures),
        :catalog_validation_failures => join(catalog_failures, ","),
        :semantic_pass => isempty(catalog_failures),
        :semantic_failures => join(catalog_failures, ","),
        :full_numerical_gate_valid => missing,
        :production_invariants_valid => missing,
        :total_seconds => elapsed,
        :total_seconds_iqr => missing,
        :allocated_bytes => allocated_bytes,
        :gc_seconds => gc_seconds,
        :allocated_bytes_iqr => missing,
        :setup_seconds => elapsed,
        :sample_count => 1,
        :execution_mode => execution_mode,
        :requested_engine => requested_engine,
        :executed_engine => :none,
        :campaign_id => campaign_id,
        :shard_id => shard_id,
        :shard_index => shard_index,
        :shard_count => shard_count,
        :pbs_array_index => pbs_array_index,
        :pbs_queue => pbs_queue,
        :pbs_node => pbs_node,
        :scaling => _scaling_label(built),
        :layout => _layout_label(built),
        :rss_bytes => missing,
        :workspace_bytes_iqr => missing,
        :process_peak_rss_bytes_iqr => missing,
        :rss_iqr_bytes => missing,
        :input_fingerprint => _problem_fingerprint(
            catalog, spec, built, arithmetic,
        ),
        :external_checksum => _built_value(
            built, :external_checksum, spec.fingerprint,
        ),
    )
    return NamedTuple{RESULT_COLUMNS}(Tuple(
        haskey(values, field) ? values[field] : getproperty(base, field)
        for field in RESULT_COLUMNS
    ))
end

function _error_row(catalog, spec, suite, arithmetic, provider, exception;
                    execution_mode=:solve,
                    requested_engine=:auto,
                    campaign_id="standalone",
                    shard_id="local",
                    shard_index="1",
                    shard_count="1",
                    pbs_array_index=missing,
                    pbs_queue=missing,
                    pbs_node=missing,
                    executed_engine=:none,
                    built=nothing)
    skipped = _skip_row(
        catalog, spec, suite, arithmetic, provider,
        "execution_error: " * sprint(showerror, exception);
        execution_mode=execution_mode,
        requested_engine=requested_engine,
        campaign_id=campaign_id,
        shard_id=shard_id,
        shard_index=shard_index,
        shard_count=shard_count,
        pbs_array_index=pbs_array_index,
        pbs_queue=pbs_queue,
        pbs_node=pbs_node,
        executed_engine=executed_engine,
        built=built,
    )
    values = Dict{Symbol,Any}(
        field => getproperty(skipped, field) for field in RESULT_COLUMNS
    )
    values[:status] = :error
    values[:semantic_pass] = false
    values[:semantic_failures] = "execution_error"
    values[:sample_count] = 1
    return NamedTuple{RESULT_COLUMNS}(
        Tuple(values[field] for field in RESULT_COLUMNS),
    )
end

function _selected_entries(catalog, suite, problem, arithmetic, provider)
    entries = catalog_entries(catalog, suite)
    problem !== nothing && (entries = filter(e -> e.problem_id == problem, entries))
    arithmetic !== nothing && (entries = [PhysicsBenchmarkEntry(
        e.problem_id, arithmetic, e.provider,
    ) for e in entries])
    provider !== nothing && (entries = [PhysicsBenchmarkEntry(
        e.problem_id, e.arithmetic, provider,
    ) for e in entries])
    seen = Set{Tuple{String,Symbol,Symbol}}()
    entries = filter(entries) do entry
        key = (entry.problem_id, entry.arithmetic, entry.provider)
        key in seen && return false
        push!(seen, key)
        return true
    end
    isempty(entries) && throw(ArgumentError("selection contains no benchmark entries"))
    return entries
end

function run_suite(
    catalog::PhysicsBenchmarkCatalog,
    suite::Symbol;
    problem=nothing,
    arithmetic=nothing,
    provider=nothing,
    output=joinpath(ROOT, "out", string(suite), "rows.toml"),
    verbose=false,
    warmup=true,
    strict_semantics=true,
    samples=3,
    cache_dir=nothing,
    allow_large=false,
    execution_mode=:solve,
    requested_engine=:auto,
    campaign_id=nothing,
    shard_id=nothing,
    shard_index=nothing,
    shard_count=nothing,
)
    # `cache_dir` and `allow_large` remain accepted so the canonical child CLI
    # used by fresh-process campaigns stays stable. Catalogs own any related
    # policy and may capture storage locations in their injected build closure.
    samples isa Integer || throw(ArgumentError(
        "samples must be an integer count of timed solves, got $samples",
    ))
    samples == 1 || samples >= 3 || throw(ArgumentError(
        "samples must be 1 (explicit single run) or >= 3 timed solves; " *
        "got $samples; a two-run observation cannot support a timing statistic",
    ))
    execution_mode = _normalize_execution_mode(execution_mode)
    requested_engine = _normalize_engine(requested_engine)
    identity = _execution_identity(
        campaign_id=campaign_id,
        shard_id=shard_id,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    identity = _validate_shard_identity(identity)
    entries = _selected_entries(catalog, suite, problem, arithmetic, provider)
    rows = NamedTuple[]
    for entry in entries
        spec = catalog_spec(catalog, entry.problem_id)
        T = try
            _arithmetic_type(entry.arithmetic)
        catch exception
            push!(rows, _skip_row(
                catalog, spec, suite, entry.arithmetic, entry.provider,
                :arithmetic_unavailable;
                execution_mode=execution_mode,
                  requested_engine=requested_engine,
                  campaign_id=identity.campaign_id,
                  shard_id=identity.shard_id,
                  shard_index=identity.shard_index,
                  shard_count=identity.shard_count,
                  pbs_array_index=identity.pbs_array_index,
                  pbs_queue=identity.pbs_queue,
                  pbs_node=identity.pbs_node,
            ))
            verbose && println("skip ", spec.id, ": ", exception)
            continue
        end
        local last_built = nothing
        local last_route = :none
        if execution_mode === :build
            local rows_for_entry
            try
                measure = function ()
                    measurement = @timed build_problem(catalog, spec, T)
                    built = measurement.value
                    last_built = built
                    declared_build_only = _validate_build_only_declaration(
                        spec, built,
                    )
                    row = _construction_row(
                        catalog, spec, suite, entry.arithmetic, entry.provider,
                        built, measurement.time;
                        allocated_bytes=measurement.bytes,
                        gc_seconds=measurement.gctime,
                        execution_mode=execution_mode,
                        requested_engine=requested_engine,
                        campaign_id=identity.campaign_id,
                        shard_id=identity.shard_id,
                        shard_index=identity.shard_index,
                        shard_count=identity.shard_count,
                        pbs_array_index=identity.pbs_array_index,
                        pbs_queue=identity.pbs_queue,
                        pbs_node=identity.pbs_node,
                        declared_build_only=declared_build_only,
                    )
                    return row, measurement.time
                end
                execute = function ()
                    if warmup
                        warm_built = build_problem(catalog, spec, T)
                        _validate_build_only_declaration(spec, warm_built)
                    end
                    if samples >= 3
                        sample_rows = NamedTuple[]
                        timed = Float64[]
                        for _ in 1:samples
                            row, elapsed = measure()
                            push!(sample_rows, row)
                            push!(timed, elapsed)
                        end
                        return [_sampling_row(
                            sample_rows, timed; sample_count=samples,
                        )]
                    end
                    row, _ = measure()
                    return [row]
                end
                if T === BigFloat
                    bits = parse(Int, replace(
                        string(entry.arithmetic), "bigfloat" => "",
                    ))
                    rows_for_entry = setprecision(BigFloat, bits) do
                        execute()
                    end
                else
                    rows_for_entry = execute()
                end
            catch exception
                rows_for_entry = [_error_row(
                    catalog, spec, suite, entry.arithmetic, entry.provider,
                    exception;
                    execution_mode=execution_mode,
                      requested_engine=requested_engine,
                      campaign_id=identity.campaign_id,
                      shard_id=identity.shard_id,
                      shard_index=identity.shard_index,
                      shard_count=identity.shard_count,
                      pbs_array_index=identity.pbs_array_index,
                      pbs_queue=identity.pbs_queue,
                      pbs_node=identity.pbs_node,
                      executed_engine=last_route,
                      built=last_built,
                )]
            end
            append!(rows, rows_for_entry)
            verbose && println(
                spec.id, " status=", first(rows_for_entry).status,
                " build=", first(rows_for_entry).total_seconds, "s",
            )
            continue
        end
        try
            _load_requested_provider(entry.provider)
        catch exception
            push!(rows, _skip_row(
                catalog, spec, suite, entry.arithmetic, entry.provider,
                :provider_unavailable;
                execution_mode=execution_mode,
                  requested_engine=requested_engine,
                  campaign_id=identity.campaign_id,
                  shard_id=identity.shard_id,
                  shard_index=identity.shard_index,
                  shard_count=identity.shard_count,
                  pbs_array_index=identity.pbs_array_index,
                  pbs_queue=identity.pbs_queue,
                  pbs_node=identity.pbs_node,
            ))
            verbose && println("skip ", spec.id, ": ", exception)
            continue
        end
        solve_route = function (built, route)
            if route === :catalog_contract
                return _invoke_solve_contract(
                    _solve_contract(built), built, T, entry.provider,
                    execution_mode,
                )
            elseif route === :sdpx_legacy
                return _solve_built(built, T, entry.provider; verbose=verbose)
            end
            throw(ArgumentError("unsupported resolved solve route $route"))
        end
        prepare = function ()
            built = build_problem(catalog, spec, T)
            last_built = built
            declared_build_only = _validate_build_only_declaration(spec, built)
            route = _resolve_engine_route(
                catalog, built, spec, execution_mode, requested_engine,
                declared_build_only,
            )
            last_route = route
            reference = route === :catalog_contract ?
                        _solve_reference(built, spec) : spec.reference
            if route === :catalog_contract && !_solve_contract_applicable(
                _solve_contract(built), built, T, entry.provider, execution_mode,
            )
                throw(ArgumentError(
                    "solve_contract for $(spec.id) is not applicable in latest world",
                ))
            end
            return built, route, reference
        end
        warmup_once = function ()
            built, route, _ = prepare()
            Base.invokelatest(solve_route, built, route)
            return nothing
        end
        measure_sample = function ()
            # Canonical O0 timing boundary for every sample count: deterministic
            # model construction, route resolution, and reference validation
            # are outside; only the selected solver contract is timed.
            built, route, reference = prepare()
            measurement = @timed Base.invokelatest(solve_route, built, route)
            row = _result_row(
                catalog, spec, suite, entry.arithmetic, entry.provider,
                built, measurement.value, measurement.time;
                allocated_bytes=measurement.bytes,
                gc_seconds=measurement.gctime,
                execution_mode=execution_mode,
                requested_engine=requested_engine,
                executed_engine=route,
                reference=reference,
                campaign_id=identity.campaign_id,
                shard_id=identity.shard_id,
                shard_index=identity.shard_index,
                shard_count=identity.shard_count,
                pbs_array_index=identity.pbs_array_index,
                pbs_queue=identity.pbs_queue,
                pbs_node=identity.pbs_node,
            )
            return row, measurement.time
        end
        execute = function ()
            warmup && warmup_once()
            if samples >= 3
                sample_rows = NamedTuple[]
                timed = Float64[]
                for _ in 1:samples
                    row, elapsed = measure_sample()
                    push!(sample_rows, row)
                    push!(timed, elapsed)
                end
                return [_sampling_row(sample_rows, timed; sample_count=samples)]
            end
            row, _ = measure_sample()
            return [row]
        end
        local rows_for_entry
        try
            if T === BigFloat
                bits = parse(Int, replace(string(entry.arithmetic), "bigfloat" => ""))
                rows_for_entry = setprecision(BigFloat, bits) do
                    execute()
                end
            else
                rows_for_entry = execute()
            end
        catch exception
            rows_for_entry = [_error_row(
                catalog, spec, suite, entry.arithmetic, entry.provider, exception;
                execution_mode=execution_mode,
                requested_engine=requested_engine,
                campaign_id=identity.campaign_id,
                shard_id=identity.shard_id,
                shard_index=identity.shard_index,
                shard_count=identity.shard_count,
                pbs_array_index=identity.pbs_array_index,
                pbs_queue=identity.pbs_queue,
                pbs_node=identity.pbs_node,
                executed_engine=last_route,
                built=last_built,
            )]
        end
        append!(rows, rows_for_entry)
        if verbose
            row = first(rows_for_entry)
            if row.sample_count isa Integer && row.sample_count >= 3 &&
               row.sample_median_seconds isa Real
                elapsed_label = string(
                    "median=", round(row.sample_median_seconds; digits=3), "s ",
                    "spread=", round(row.sample_spread_seconds; digits=3), "s ",
                    "samples=", row.sample_count,
                )
            else
                elapsed_label = row.total_seconds isa Real ?
                    string(round(row.total_seconds; digits=3), "s") : "unavailable"
            end
            println(spec.id, " ", row.status, " ", elapsed_label)
        end
    end
    paths = write_results(output, rows)
    failed = filter(
        row -> row.status != :skipped &&
               (row.semantic_pass === false ||
                row.sample_semantic_parity === false),
        rows,
    )
    if strict_semantics && !isempty(failed)
        details = join((
            let failures = String[]
                isempty(row.semantic_failures) ||
                    push!(failures, row.semantic_failures)
                row.sample_semantic_parity === false &&
                    push!(failures, "sample_parity:$(row.sample_parity_failures)")
                join(failures, ",")
            end
            for row in failed
        ), "; ")
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
    cache_dir = nothing
    catalog_path = nothing
    verbose = false
    prepare = false
    allow_large = false
    samples = 3
    warmup = true
    execution_mode = :solve
    requested_engine = :auto
    campaign_id = nothing
    shard_id = nothing
    shard_index = nothing
    shard_count = nothing
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
        elseif startswith(argument, "--cache-dir=")
            cache_dir = abspath(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--catalog=")
            catalog_path = abspath(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--samples=")
            samples = parse(Int, split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--execution-mode=")
            execution_mode = _parse_symbol(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--engine=") ||
               startswith(argument, "--requested-engine=")
            requested_engine = _parse_symbol(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--campaign-id=")
            campaign_id = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--shard-id=")
            shard_id = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--shard-index=")
            shard_index = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--shard-count=")
            shard_count = split(argument, "="; limit=2)[2]
        elseif argument == "--verbose"
            verbose = true
        elseif argument == "--prepare"
            throw(ArgumentError(
                "--prepare belonged to removed external loaders; prepare physics inputs outside the harness",
            ))
        elseif argument == "--allow-large"
            allow_large = true
        elseif argument == "--no-warmup"
            warmup = false
        elseif startswith(argument, "--")
            throw(ArgumentError("unknown option $argument"))
        else
            push!(positional, argument)
        end
    end
    !isempty(positional) && (suite = Symbol(lowercase(first(positional))))
    length(positional) > 1 && problem === nothing && (problem = positional[2])
    return (; suite, problem, arithmetic, provider, output, cache_dir, catalog_path, verbose,
            prepare, allow_large, samples, warmup, execution_mode,
            requested_engine, campaign_id, shard_id, shard_index, shard_count)
end

function main(args=ARGS; catalog::PhysicsBenchmarkCatalog)
    options = _parse_cli(args)
    selected_catalog = options.catalog_path === nothing ? catalog :
                       load_catalog(options.catalog_path)
    output = something(
        options.output,
        joinpath(ROOT, "out", string(options.suite), "rows.toml"),
    )
    result = run_suite(
        selected_catalog,
        options.suite;
        problem=options.problem,
        arithmetic=options.arithmetic,
        provider=options.provider,
        output=output,
        verbose=options.verbose,
        samples=options.samples,
        warmup=options.warmup,
        cache_dir=options.cache_dir,
        allow_large=options.allow_large,
        execution_mode=options.execution_mode,
        requested_engine=options.requested_engine,
        campaign_id=options.campaign_id,
        shard_id=options.shard_id,
        shard_index=options.shard_index,
        shard_count=options.shard_count,
    )
    built = count(row -> row.status in (:built, :build_only, :sampled_build_only), result.rows)
    solved = count(row -> !(row.status in (
        :skipped, :error, :built, :build_only, :sampled_build_only,
    )), result.rows)
    errors = count(row -> row.status === :error, result.rows)
    skipped = count(row -> row.status === :skipped, result.rows)
    println("suite=", options.suite, " mode=", options.execution_mode,
            " built=", built, " solved=", solved,
            " skipped=", skipped, " errors=", errors,
            " output=", result.paths.toml)
    return result
end
