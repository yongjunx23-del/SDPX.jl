module FreshProcessCampaign

using TOML
using Dates
using LinearAlgebra
using SHA
using Statistics
using Printf

export ChildRecord, run_campaign, aggregate_campaign, write_summary,
       campaign_main

const ROOT = @__DIR__
const REPOSITORY = normpath(joinpath(ROOT, ".."))
const FRESH_SCHEMA_VERSION = 1
const CHILD_RESULT_SCHEMA_VERSION = 8
const _SHA256_HEX64_RE = r"^[0-9a-f]{64}$"
const _EXECUTION_MODES = (:build, :solve, :profile)
const _REQUESTED_ENGINES = (
    :auto, :legacy, :sdpx_legacy, :catalog_contract,
    :native, :native_hsd, :product_hsd,
)

_valid_sha256(value) = occursin(_SHA256_HEX64_RE, _text(value))

"""One independently spawned benchmark result and its process evidence."""
struct ChildRecord
    repetition::Int
    raw_toml::String
    raw_tsv::String
    log_path::String
    exit_code::Int
    row::Union{Nothing,Dict{String,Any}}
    failure::String
end

_text(value) = value === missing || value === nothing ? "" : strip(string(value))

function _parse_finite(value)
    text = _text(value)
    isempty(text) && return nothing
    parsed = try
        parse(Float64, text)
    catch
        nothing
    end
    parsed isa Real && isfinite(parsed) || return nothing
    return Float64(parsed)
end

function _parse_integer(value)
    text = _text(value)
    isempty(text) && return nothing
    try
        return parse(Int, text)
    catch
        parsed = try
            parse(Float64, text)
        catch
            nothing
        end
        parsed isa Real && isfinite(parsed) && isinteger(parsed) || return nothing
        return Int(parsed)
    end
end

function _parse_bool(value)
    value === true && return true
    value === false && return false
    text = lowercase(_text(value))
    text == "true" && return true
    text == "false" && return false
    return nothing
end

function _route_token(value)
    text = lowercase(replace(_text(value), '-' => '_', ' ' => '_'))
    isempty(text) && return nothing
    return Symbol(text)
end

function _canonical_requested_route(value)
    token = _route_token(value)
    token === :auto && return :auto
    token in (:legacy, :sdpx_legacy) && return :legacy
    token === :catalog_contract && return :contract
    token in (:native, :native_hsd, :product_hsd) && return :native
    return nothing
end

function _canonical_executed_route(value)
    token = _route_token(value)
    token === :none && return :none
    token in (:legacy, :sdpx_legacy) && return :legacy
    token === :catalog_contract && return :contract
    return nothing
end

function _route_failures(row, label)
    failures = String[]
    mode = _route_token(get(row, "execution_mode", missing))
    requested = _canonical_requested_route(
        get(row, "requested_engine", missing),
    )
    executed = _canonical_executed_route(
        get(row, "executed_engine", missing),
    )
    mode in _EXECUTION_MODES || push!(failures, "$(label):execution_mode_invalid")
    requested === nothing && push!(failures, "$(label):requested_engine_invalid")
    executed === nothing && push!(failures, "$(label):executed_engine_invalid")
    if mode === :build
        executed === :none || push!(failures, "$(label):build_executed_engine_not_none")
    elseif mode in (:solve, :profile)
        executed in (:legacy, :contract) ||
            push!(failures, "$(label):solve_executed_engine_invalid")
        requested === :native &&
            push!(failures, "$(label):native_requested_engine_invalid")
        requested === :legacy && executed !== :legacy &&
            push!(failures, "$(label):legacy_route_mismatch")
        requested === :contract && executed !== :contract &&
            push!(failures, "$(label):contract_route_mismatch")
    end
    return failures
end

const _OPTIONAL_TIMING_FIELDS = (
    "seconds_per_iteration", "total_seconds_iqr", "gc_seconds",
    "setup_seconds", "frontend_seconds", "preprocess_seconds",
    "presolve_seconds", "core_seconds", "certification_seconds",
    "sample_median_seconds", "sample_min_seconds", "sample_max_seconds",
    "sample_mad_seconds", "sample_spread_seconds", "assembly_seconds",
    "factor_seconds", "solve_seconds", "refinement_seconds",
    "local_metric_seconds", "local_factor_seconds",
    "panel_transform_seconds", "equality_gram_seconds",
    "equality_factor_seconds", "predictor_rhs_seconds",
    "corrector_rhs_seconds", "block_residual_seconds",
    "block_recovery_seconds",
)

function _timing_failures(row, label)
    failures = String[]
    total = _parse_finite(get(row, "total_seconds", ""))
    (total !== nothing && total > 0) ||
        push!(failures, "$(label):total_seconds_invalid")
    for field in _OPTIONAL_TIMING_FIELDS
        haskey(row, field) || continue
        text = _text(get(row, field, ""))
        isempty(text) && continue
        value = _parse_finite(get(row, field, ""))
        (value !== nothing && value >= 0) ||
            push!(failures, "$(label):$(field)_invalid")
    end
    return failures
end

function _objective_parity(rows)
    values = [_text(get(row, "objective", "")) for row in rows]
    any(isempty, values) && return false, "objective_missing"
    absolute = _text(get(first(rows), "reference_absolute_tolerance", ""))
    relative = _text(get(first(rows), "reference_relative_tolerance", ""))
    # Objectives are serialized as arbitrary precision decimal strings.  Use
    # a local high precision scope so a Float64 ambient context cannot turn a
    # tiny but meaningful difference into a false equality.
    bits = max(256, 4 * maximum(length.((values..., absolute, relative))))
    try
        setprecision(BigFloat, bits) do
            parsed = [parse(BigFloat, value) for value in values]
            all(isfinite, parsed) || return false, "objective_nonfinite"
            first_value = first(parsed)
            all(isequal(first_value), parsed) && return true, ""
            (isempty(absolute) || isempty(relative)) &&
                return false, "objective"
            absolute_tolerance = parse(BigFloat, absolute)
            relative_tolerance = parse(BigFloat, relative)
            (isfinite(absolute_tolerance) && absolute_tolerance >= 0 &&
             isfinite(relative_tolerance) && relative_tolerance >= 0) ||
                return false, "objective_tolerance_invalid"
            allowed = absolute_tolerance + relative_tolerance *
                      max(one(BigFloat), abs(first_value))
            all(value -> abs(value - first_value) <= allowed, parsed) &&
                return true, ""
            return false, "objective"
        end
    catch
        return false, "objective_unparseable"
    end
end

function _route(row)
    route = (
        _route_token(get(row, "execution_mode", missing)),
        _canonical_requested_route(get(row, "requested_engine", missing)),
        _canonical_executed_route(get(row, "executed_engine", missing)),
    )
    fields = (
        "conic_formulation", "planned_formulation", "executed_formulation",
        "planned_backend", "executed_backend", "planned_provider",
        "executed_provider", "executed_specialization", "psd_lift_used",
        "fallback_reason", "la_fallback_reason",
    )
    values = Any[string(value) for value in route]
    append!(values, (_text(get(row, field, "")) for field in fields))
    return join(values, "|")
end

const PAIRING_FIELDS = (
    # Problem identity and formulation/tolerance are part of the experiment,
    # not merely labels.  A mismatch means the rows cannot form a timing
    # sample, even when all processes happened to exit successfully.
    "suite", "catalog_name", "catalog_version",
    "problem_id", "name", "family", "problem_type", "source",
    "arithmetic", "precision_bits", "requested_provider",
    "execution_mode", "requested_engine", "executed_engine",
    "campaign_id", "shard_id",
    "pbs_job_id", "pbs_array_index", "pbs_queue", "pbs_node",
    "scaling", "layout",
    "reference_status", "reference_absolute_tolerance",
    "reference_relative_tolerance", "conic_formulation",
    "planned_formulation", "executed_formulation", "planned_backend",
    "executed_backend", "planned_provider", "executed_provider",
    "executed_specialization", "psd_lift_used", "fallback_reason",
    "la_fallback_reason", "input_fingerprint", "external_checksum",
    # Environment pairing keeps a sample attributable to one executable
    # environment.  Source hashes are included so no cross-build timings can
    # be silently mixed.
    "julia_version", "os", "cpu_name", "julia_threads", "blas_threads",
    "project_sha256", "manifest_sha256", "benchmark_driver_sha256",
    "solver_source_sha256", "catalog_source_sha256",
    "harness_source_sha256", "schema_source_sha256",
    "contract_fingerprint",
    "mfla_commit", "bfla_commit",
)

function _pairing_failures(rows)
    failures = String[]
    isempty(rows) && return ["no_rows"]
    first_row = first(rows)
    for field in (
        "catalog_name", "catalog_version", "source",
        "execution_mode", "requested_engine", "executed_engine",
        "campaign_id", "shard_id", "shard_index", "shard_count",
        "scaling", "layout", "input_fingerprint", "external_checksum",
        "catalog_source_sha256", "harness_source_sha256",
        "schema_source_sha256", "contract_fingerprint",
    )
        any(isempty(_text(get(row, field, ""))) for row in rows) &&
            push!(failures, "$(field)_missing")
    end
    for field in (
        "project_sha256", "manifest_sha256", "benchmark_driver_sha256",
        "solver_source_sha256", "catalog_source_sha256",
        "harness_source_sha256", "schema_source_sha256",
        "contract_fingerprint",
    )
        all(row -> _valid_sha256(get(row, field, "")), rows) ||
            push!(failures, "$(field)_invalid")
    end
    for (index, row) in enumerate(rows)
        get(row, "schema_version", nothing) == CHILD_RESULT_SCHEMA_VERSION ||
            push!(failures, "child_$(index):schema_version")
        _parse_integer(get(row, "sample_count", "")) == 1 ||
            push!(failures, "child_$(index):sample_count")
        append!(failures, _route_failures(row, "child_$(index)"))
        append!(failures, _timing_failures(row, "child_$(index)"))
        shard_index = _parse_integer(get(row, "shard_index", ""))
        shard_count = _parse_integer(get(row, "shard_count", ""))
        (shard_index !== nothing && shard_count !== nothing &&
         shard_index > 0 && shard_count > 0 && shard_index <= shard_count) ||
            push!(failures, "child_$(index):shard_topology")
    end
    for field in PAIRING_FIELDS
        values = if field == "execution_mode"
            [_route_token(get(row, field, "")) for row in rows]
        elseif field == "requested_engine"
            [_canonical_requested_route(get(row, field, "")) for row in rows]
        elseif field == "executed_engine"
            [_canonical_executed_route(get(row, field, "")) for row in rows]
        else
            [_text(get(row, field, "")) for row in rows]
        end
        length(unique(values)) == 1 || push!(failures, "pairing:$field")
    end
    for field in ("shard_index", "shard_count")
        values = [_parse_integer(get(row, field, "")) for row in rows]
        length(unique(values)) == 1 || push!(failures, "pairing:$field")
    end
    status = [_text(get(row, "status", "")) for row in rows]
    length(unique(status)) == 1 || push!(failures, "status")
    iterations = [_text(get(row, "iterations", "")) for row in rows]
    length(unique(iterations)) == 1 || push!(failures, "iterations")
    route = [_route(row) for row in rows]
    length(unique(route)) == 1 || push!(failures, "route")
    fingerprint = [_text(get(row, "input_fingerprint", "")) for row in rows]
    any(isempty, fingerprint) && push!(failures, "input_fingerprint_missing")
    length(unique(fingerprint)) == 1 || push!(failures, "input_fingerprint")
    execution_mode = _route_token(get(first_row, "execution_mode", ""))
    if execution_mode !== :build
        for field in (
            "conic_formulation", "planned_formulation", "executed_formulation",
            "planned_backend", "executed_backend", "planned_provider",
            "executed_provider", "fallback_reason", "la_fallback_reason",
        )
            any(isempty(_text(get(row, field, ""))) for row in rows) &&
                push!(failures, "$(field)_missing")
        end
        objective_ok, objective_failure = _objective_parity(rows)
        objective_ok || push!(failures, objective_failure)
    end
    return failures
end

function _summary(values)
    numbers = Float64[]
    for value in values
        parsed = _parse_finite(value)
        parsed === nothing && return nothing
        parsed >= 0 || return nothing
        push!(numbers, parsed)
    end
    isempty(numbers) && return nothing
    ordered = sort(numbers)
    center = median(ordered)
    mad = median(abs.(ordered .- center))
    return (
        median=center,
        min=first(ordered),
        max=last(ordered),
        mad=mad,
        spread=last(ordered) - first(ordered),
        iqr=quantile(ordered, 0.75) - quantile(ordered, 0.25),
    )
end

function _relpath_list(paths, campaign_dir)
    return [relpath(path, campaign_dir) for path in paths]
end

"""Aggregate already collected child rows, failing closed on semantic drift.

The input is a vector of `ChildRecord`s.  With `diagnostic=false` every child
must have exit code zero, a finite timing, and `semantic_pass=true`; solve and
profile children additionally require `certificate_valid=true`, while build
children require `catalog_validation_pass=true`.  All pairing fields and
semantic values must agree.
Diagnostic mode is intentionally labelled in the returned document and never
claims `aggregation_valid=true` when those gates fail.
"""
function aggregate_campaign(
    records::AbstractVector{<:ChildRecord};
    repetitions::Integer=length(records),
    diagnostic::Bool=false,
    campaign_dir::AbstractString=REPOSITORY,
    expected=nothing,
)
    repetitions >= 3 || throw(ArgumentError("repetitions must be >= 3"))
    length(records) == repetitions || throw(ArgumentError(
        "expected $repetitions child records, got $(length(records))",
    ))
    ordered = sort!(collect(records); by=record -> record.repetition)
    failures = String[]
    reps = [record.repetition for record in ordered]
    reps == collect(1:repetitions) || push!(failures, "repetition_sequence")
    for record in ordered
        record.exit_code == 0 || push!(failures, "child_$(record.repetition):exit_$(record.exit_code)")
        isempty(record.failure) || push!(failures, "child_$(record.repetition):$(record.failure)")
        record.row === nothing && push!(failures, "child_$(record.repetition):row_missing")
    end
    rows = [record.row for record in ordered if record.row !== nothing]
    pairing_failures = String[]
    execution_modes = [_route_token(get(row, "execution_mode", "")) for row in rows]
    build_mode = length(execution_modes) == repetitions &&
                 all(mode === :build for mode in execution_modes)
    if length(rows) == repetitions
        append!(pairing_failures, _pairing_failures(rows))
        append!(failures, pairing_failures)
        if expected !== nothing
            expected_fields = [
                (:suite, "suite"), (:problem_id, "problem_id"),
                (:arithmetic, "arithmetic"), (:provider, "requested_provider"),
            ]
            hasproperty(expected, :execution_mode) && push!(
                expected_fields, (:execution_mode, "execution_mode"),
            )
            hasproperty(expected, :requested_engine) && push!(
                expected_fields, (:requested_engine, "requested_engine"),
            )
            for (name, field) in (
                (:campaign_id, "campaign_id"), (:shard_id, "shard_id"),
                (:shard_index, "shard_index"), (:shard_count, "shard_count"),
            )
                hasproperty(expected, name) && push!(expected_fields, (name, field))
            end
            for (name, field) in expected_fields
                wanted = _text(getproperty(expected, name))
                matches = if field in ("shard_index", "shard_count")
                    wanted_integer = _parse_integer(wanted)
                    all(_parse_integer(get(row, field, "")) == wanted_integer
                        for row in rows)
                elseif field == "execution_mode"
                    wanted_route = _route_token(wanted)
                    all(_route_token(get(row, field, "")) == wanted_route
                        for row in rows)
                elseif field == "requested_engine"
                    wanted_route = _canonical_requested_route(wanted)
                    all(_canonical_requested_route(get(row, field, "")) == wanted_route
                        for row in rows)
                else
                    all(_text(get(row, field, "")) == wanted for row in rows)
                end
                if !matches
                    push!(failures, "selection:$field")
                    push!(pairing_failures, "selection:$field")
                end
            end
        end
        for (index, row) in enumerate(rows)
            semantic = _parse_bool(get(row, "semantic_pass", ""))
            certificate = _parse_bool(get(row, "certificate_valid", ""))
            semantic === true || push!(failures, "child_$(index):semantic_pass")
            if build_mode
                catalog = _parse_bool(get(row, "catalog_validation_pass", ""))
                catalog === true ||
                    push!(failures, "child_$(index):catalog_validation_pass")
            else
                certificate === true ||
                    push!(failures, "child_$(index):certificate_valid")
            end
        end
    end
    timing = [_parse_finite(get(row, "total_seconds", "")) for row in rows]
    any(value -> value === nothing || value <= 0, timing) &&
        push!(failures, "timing_missing_or_invalid")
    timing_summary = all(value -> value !== nothing && value > 0, timing) ?
        _summary(timing) : nothing
    memory_fields = (
        "allocated_bytes", "process_peak_rss_bytes", "rss_bytes", "workspace_bytes",
    )
    memory = Dict{String,Any}()
    for field in memory_fields
        summary = _summary([get(row, field, "") for row in rows])
        memory[field] = summary
    end
    iterations = _summary([get(row, "iterations", "") for row in rows])
    strict_gates = isempty(failures)
    aggregation_valid = strict_gates && !diagnostic
    diagnostic_label = diagnostic ? "diagnostic" : "strict"
    # Keep all raw evidence in the campaign artifact, but avoid absolute paths
    # in the deterministic summary so it can be copied between machines.
    raw_toml = _relpath_list([record.raw_toml for record in ordered], campaign_dir)
    raw_tsv = _relpath_list([record.raw_tsv for record in ordered], campaign_dir)
    logs = _relpath_list([record.log_path for record in ordered], campaign_dir)
    base = isempty(rows) ? Dict{String,Any}() : first(rows)
    result = Dict{String,Any}(
        "suite" => get(base, "suite", ""),
        "problem_id" => get(base, "problem_id", ""),
        "arithmetic" => get(base, "arithmetic", ""),
        "requested_provider" => get(base, "requested_provider", ""),
        "execution_mode" => get(base, "execution_mode", ""),
        "requested_engine" => get(base, "requested_engine", ""),
        "executed_engine" => get(base, "executed_engine", ""),
        "campaign_id" => get(base, "campaign_id", ""),
        "shard_id" => get(base, "shard_id", ""),
        "shard_index" => get(base, "shard_index", ""),
        "shard_count" => get(base, "shard_count", ""),
        "pbs_job_id" => get(base, "pbs_job_id", ""),
        "pbs_array_index" => get(base, "pbs_array_index", ""),
        "pbs_queue" => get(base, "pbs_queue", ""),
        "pbs_node" => get(base, "pbs_node", ""),
        "scaling" => get(base, "scaling", ""),
        "layout" => get(base, "layout", ""),
        "status" => get(base, "status", ""),
        "iterations" => get(base, "iterations", ""),
        "objective" => get(base, "objective", ""),
        "route" => isempty(rows) ? "" : _route(base),
        "input_fingerprint" => get(base, "input_fingerprint", ""),
        "external_checksum" => get(base, "external_checksum", ""),
        "catalog_source_sha256" => get(base, "catalog_source_sha256", ""),
        "harness_source_sha256" => get(base, "harness_source_sha256", ""),
        "schema_source_sha256" => get(base, "schema_source_sha256", ""),
        "contract_fingerprint" => get(base, "contract_fingerprint", ""),
        "semantic_pass" => length(rows) == repetitions &&
                           all(_parse_bool(get(row, "semantic_pass", "")) === true for row in rows),
        "certificate_valid" => build_mode ? "" :
                               (length(rows) == repetitions &&
                                all(_parse_bool(get(row, "certificate_valid", "")) === true for row in rows)),
        "repetitions" => repetitions,
        "child_exit_codes" => [record.exit_code for record in ordered],
        "child_raw_toml" => raw_toml,
        "child_raw_tsv" => raw_tsv,
        "child_logs" => logs,
        "timing_valid" => timing_summary !== nothing,
        "aggregation_mode" => diagnostic_label,
        "aggregation_valid" => aggregation_valid,
        "pairing_valid" => length(rows) == repetitions &&
                           isempty(pairing_failures),
        "failure_reasons" => join(unique(failures), ","),
    )
    if timing_summary !== nothing
        for (name, value) in pairs(timing_summary)
            result["total_seconds_$(name)"] = value
        end
    else
        for name in (:median, :min, :max, :mad, :spread, :iqr)
            result["total_seconds_$(name)"] = ""
        end
    end
    for (field, summary) in memory
        if summary === nothing
            for name in (:median, :min, :max, :mad, :spread, :iqr)
                result["$(field)_$(name)"] = ""
            end
        else
            for (name, value) in pairs(summary)
                result["$(field)_$(name)"] = value
            end
        end
    end
    # `rss_bytes` is the canonical row-level alias used by the v2 schema;
    # keep the human-readable `rss_iqr_bytes` spelling alongside the generic
    # field-derived `rss_bytes_iqr` summary key.
    result["rss_iqr_bytes"] = get(result, "rss_bytes_iqr", "")
    if iterations === nothing
        for name in (:median, :min, :max, :mad, :spread, :iqr)
            result["iterations_$(name)"] = ""
        end
    else
        for (name, value) in pairs(iterations)
            result["iterations_$(name)"] = value
        end
    end
    return Dict(
        "schema_version" => FRESH_SCHEMA_VERSION,
        "campaign" => Dict(
            "repetitions" => repetitions,
            "campaign_id" => get(base, "campaign_id", ""),
            "shard_id" => get(base, "shard_id", ""),
            "execution_mode" => get(base, "execution_mode", ""),
            "requested_engine" => get(base, "requested_engine", ""),
            "aggregation_mode" => diagnostic_label,
            "aggregation_valid" => aggregation_valid,
            "pairing_valid" => result["pairing_valid"],
            "timing_valid" => result["timing_valid"],
            "failure_reasons" => result["failure_reasons"],
        ),
        "result" => [result],
    )
end

function _tsv_cell(value)
    value === nothing && return ""
    return replace(string(value), '\t' => ' ', '\n' => ' ', '\r' => ' ')
end

function _summary_columns(row::Dict{String,Any})
    # Stable semantic order, followed by deterministic lexical ordering for
    # telemetry fields.  This keeps TSV diffs reviewable across Julia versions.
    preferred = [
        "suite", "problem_id", "arithmetic", "requested_provider",
        "execution_mode", "requested_engine", "executed_engine",
        "campaign_id", "shard_id", "shard_index", "shard_count",
        "status",
        "iterations", "objective", "route", "input_fingerprint",
        "external_checksum", "catalog_source_sha256",
        "harness_source_sha256", "schema_source_sha256",
        "contract_fingerprint",
        "semantic_pass", "certificate_valid", "repetitions", "child_exit_codes",
        "aggregation_mode", "aggregation_valid", "pairing_valid", "timing_valid",
        "failure_reasons",
    ]
    rest = sort!(setdiff(collect(keys(row)), preferred))
    return vcat(preferred, rest)
end

"""Write deterministic TOML and TSV summary artifacts."""
function write_summary(document::AbstractDict, path::AbstractString)
    root, extension = splitext(path)
    toml_path = extension == ".toml" ? path : root * ".toml"
    tsv_path = extension == ".tsv" ? path : root * ".tsv"
    mkpath(dirname(toml_path))
    open(toml_path, "w") do io
        TOML.print(io, document; sorted=true)
    end
    rows = get(document, "result", Any[])
    isempty(rows) && throw(ArgumentError("summary contains no result row"))
    row = Dict{String,Any}(string(key) => value for (key, value) in first(rows))
    columns = _summary_columns(row)
    open(tsv_path, "w") do io
        println(io, join(columns, '\t'))
        println(io, join((_tsv_cell(get(row, column, "")) for column in columns), '\t'))
    end
    return (toml=toml_path, tsv=tsv_path)
end

function _julia_cmd(arguments; project, threads, blas_threads)
    base = Base.julia_cmd()
    executable = vcat(base.exec, String["--project=$(project)", "--threads=$(threads)"], arguments)
    environment = Dict{String,String}(ENV)
    environment["OPENBLAS_NUM_THREADS"] = string(blas_threads)
    environment["MKL_NUM_THREADS"] = string(blas_threads)
    environment["BLAS_NUM_THREADS"] = string(blas_threads)
    return Cmd(Cmd(executable); dir=project, env=environment)
end

_default_blas_threads() = try
    LinearAlgebra.BLAS.get_num_threads()
catch
    1
end

function _child_record(
    repetition, campaign_dir, command;
    raw_base,
)
    raw_toml = raw_base * ".toml"
    raw_tsv = raw_base * ".tsv"
    log_path = raw_base * ".log"
    # A rerun in the same directory must not consume stale output if the new
    # child fails before producing its result artifact.
    rm(raw_toml; force=true)
    rm(raw_tsv; force=true)
    exit_code = -1
    failure = ""
    open(log_path, "w") do log
        try
            process = run(pipeline(command; stdout=log, stderr=log); wait=false)
            wait(process)
            exit_code = process.exitcode
            exit_code == 0 || (failure = "process_failed")
        catch exception
            failure = "process_exception:" * sprint(showerror, exception)
        end
    end
    row = nothing
    if isfile(raw_toml)
        try
            document = TOML.parsefile(raw_toml)
            get(document, "schema_version", nothing) == CHILD_RESULT_SCHEMA_VERSION ||
                (failure = isempty(failure) ? "schema_version" :
                           failure * ",schema_version")
            result = get(document, "result", Any[])
            length(result) == 1 || (failure = isempty(failure) ? "row_count" : failure * ",row_count")
            if length(result) == 1
                row = Dict{String,Any}(string(k) => v for (k, v) in first(result))
                get(row, "schema_version", nothing) == CHILD_RESULT_SCHEMA_VERSION ||
                    (failure = isempty(failure) ? "row_schema_version" :
                               failure * ",row_schema_version")
                _parse_integer(get(row, "sample_count", "")) == 1 ||
                    (failure = isempty(failure) ? "sample_count" :
                               failure * ",sample_count")
            end
        catch exception
            failure = isempty(failure) ? "result_parse:" * sprint(showerror, exception) :
                      failure * ",result_parse"
        end
    else
        failure = isempty(failure) ? "result_missing" : failure * ",result_missing"
    end
    return ChildRecord(repetition, raw_toml, raw_tsv, log_path, exit_code, row, failure)
end

"""Run one selected benchmark row in `repetitions` fresh Julia processes."""
function run_campaign(
    suite::Symbol;
    problem::AbstractString,
    arithmetic::Symbol=:float64,
    provider::Symbol=:auto,
    repetitions::Integer=3,
    campaign_dir::AbstractString=joinpath(ROOT, "out", "fresh_process"),
    project::AbstractString=REPOSITORY,
    cache_dir::Union{Nothing,AbstractString}=nothing,
    catalog::Union{Nothing,AbstractString}=nothing,
    threads::Integer=Threads.nthreads(),
    blas_threads::Integer=_default_blas_threads(),
    allow_large::Bool=false,
    diagnostic::Bool=false,
    verbose::Bool=false,
    execution_mode::Symbol=:solve,
    requested_engine::Symbol=:auto,
    campaign_id::Union{Nothing,AbstractString}=nothing,
    shard_id::Union{Nothing,AbstractString}=nothing,
    shard_index::Integer=1,
    shard_count::Integer=1,
)
    repetitions >= 3 || throw(ArgumentError("repetitions must be >= 3"))
    threads >= 1 || throw(ArgumentError("threads must be >= 1"))
    blas_threads >= 1 || throw(ArgumentError("blas_threads must be >= 1"))
    execution_mode in (:build, :solve, :profile) || throw(ArgumentError(
        "execution_mode must be :build, :solve, or :profile",
    ))
    requested_engine in (:auto, :legacy, :sdpx_legacy, :catalog_contract,
                         :native, :native_hsd, :product_hsd) ||
        throw(ArgumentError("unsupported benchmark engine $requested_engine"))
    execution_mode === :build ||
        !(requested_engine in (:native, :native_hsd, :product_hsd)) ||
        throw(ArgumentError(
            "requested engine $requested_engine has no benchmark solve adapter yet",
        ))
    shard_index >= 1 || throw(ArgumentError("shard_index must be >= 1"))
    shard_count >= shard_index || throw(ArgumentError(
        "shard_count must be >= shard_index",
    ))
    isempty(problem) && throw(ArgumentError("problem must be non-empty"))
    project = abspath(project)
    campaign_dir = abspath(campaign_dir)
    mkpath(campaign_dir)
    campaign_id = something(
        campaign_id,
        "fresh-" * bytes2hex(SHA.sha256(join((
            string(suite), problem, string(arithmetic), string(provider),
            string(execution_mode), string(requested_engine),
            string(catalog),
            string(threads), string(blas_threads),
            string(shard_index), string(shard_count),
        ), "|"))),
    )
    shard_id = something(
        shard_id,
        "shard-$(shard_index)-of-$(shard_count)",
    )
    isempty(strip(campaign_id)) && throw(ArgumentError(
        "campaign_id must be non-empty",
    ))
    isempty(strip(shard_id)) && throw(ArgumentError(
        "shard_id must be non-empty",
    ))
    runner = joinpath(ROOT, "runner.jl")
    records = ChildRecord[]
    for repetition in 1:repetitions
        base = joinpath(campaign_dir, @sprintf("child_%03d", repetition))
        arguments = String[
            runner,
            string(suite),
            "--problem=$(problem)",
            "--arithmetic=$(arithmetic)",
            "--provider=$(provider)",
            "--execution-mode=$(execution_mode)",
            "--engine=$(requested_engine)",
            "--campaign-id=$(campaign_id)",
            "--shard-id=$(shard_id)",
            "--shard-index=$(shard_index)",
            "--shard-count=$(shard_count)",
            "--samples=1",
            "--output=$(base).toml",
        ]
        cache_dir === nothing || push!(arguments, "--cache-dir=$(abspath(cache_dir))")
        catalog === nothing || push!(arguments, "--catalog=$(abspath(catalog))")
        allow_large && push!(arguments, "--allow-large")
        verbose && push!(arguments, "--verbose")
        # The runner's default is warmup=true.  We deliberately do not expose
        # --no-warmup here: a campaign without an untimed child warmup is not
        # a canonical fresh-process timing experiment.
        command = _julia_cmd(arguments; project, threads, blas_threads)
        push!(records, _child_record(repetition, campaign_dir, command; raw_base=base))
    end
    document = aggregate_campaign(
        records;
        repetitions,
        diagnostic,
        campaign_dir,
        expected=(suite=string(suite), problem_id=problem,
                  arithmetic=string(arithmetic), provider=string(provider),
                  execution_mode=string(execution_mode),
                  requested_engine=string(requested_engine),
                  campaign_id=campaign_id, shard_id=shard_id,
                  shard_index=shard_index, shard_count=shard_count),
    )
    summary = write_summary(document, joinpath(campaign_dir, "summary.toml"))
    return (; document, records, summary)
end

function _option_value(argument, prefix, default)
    startswith(argument, prefix) || return nothing
    return split(argument, "="; limit=2)[2]
end

function campaign_main(args=ARGS)
    suite = :micro
    problem = nothing
    arithmetic = :float64
    provider = :auto
    repetitions = 3
    campaign_dir = joinpath(ROOT, "out", "fresh_process")
    project = REPOSITORY
    cache_dir = nothing
    catalog = nothing
    threads = Threads.nthreads()
    blas_threads = _default_blas_threads()
    allow_large = false
    diagnostic = false
    verbose = false
    execution_mode = :solve
    requested_engine = :auto
    campaign_id = nothing
    shard_id = nothing
    shard_index = 1
    shard_count = 1
    positional = String[]
    for argument in args
        value = _option_value(argument, "--problem=", nothing)
        value !== nothing && (problem = value; continue)
        value = _option_value(argument, "--arithmetic=", nothing)
        value !== nothing && (arithmetic = Symbol(lowercase(value)); continue)
        value = _option_value(argument, "--provider=", nothing)
        value !== nothing && (provider = Symbol(lowercase(value)); continue)
        value = _option_value(argument, "--repetitions=", nothing)
        value !== nothing && (repetitions = parse(Int, value); continue)
        value = _option_value(argument, "--campaign-dir=", nothing)
        value !== nothing && (campaign_dir = value; continue)
        value = _option_value(argument, "--project=", nothing)
        value !== nothing && (project = value; continue)
        value = _option_value(argument, "--cache-dir=", nothing)
        value !== nothing && (cache_dir = value; continue)
        value = _option_value(argument, "--catalog=", nothing)
        value !== nothing && (catalog = value; continue)
        value = _option_value(argument, "--threads=", nothing)
        value !== nothing && (threads = parse(Int, value); continue)
        value = _option_value(argument, "--blas-threads=", nothing)
        value !== nothing && (blas_threads = parse(Int, value); continue)
        value = _option_value(argument, "--execution-mode=", nothing)
        value !== nothing && (execution_mode = Symbol(lowercase(value)); continue)
        value = _option_value(argument, "--engine=", nothing)
        value === nothing && (value = _option_value(
            argument, "--requested-engine=", nothing,
        ))
        value !== nothing && (requested_engine = Symbol(lowercase(value)); continue)
        value = _option_value(argument, "--campaign-id=", nothing)
        value !== nothing && (campaign_id = value; continue)
        value = _option_value(argument, "--shard-id=", nothing)
        value !== nothing && (shard_id = value; continue)
        value = _option_value(argument, "--shard-index=", nothing)
        value !== nothing && (shard_index = parse(Int, value); continue)
        value = _option_value(argument, "--shard-count=", nothing)
        value !== nothing && (shard_count = parse(Int, value); continue)
        argument == "--allow-large" && (allow_large = true; continue)
        argument == "--diagnostic" && (diagnostic = true; continue)
        argument == "--verbose" && (verbose = true; continue)
        startswith(argument, "--") && throw(ArgumentError("unknown option $argument"))
        push!(positional, argument)
    end
    !isempty(positional) && (suite = Symbol(lowercase(first(positional))))
    problem === nothing && length(positional) > 1 && (problem = positional[2])
    problem === nothing && throw(ArgumentError(
        "select exactly one problem with --problem=...",
    ))
    return run_campaign(
        suite;
        problem,
        arithmetic,
        provider,
        repetitions,
        campaign_dir,
        project,
        cache_dir,
        catalog,
        threads,
        blas_threads,
        allow_large,
        diagnostic,
        verbose,
        execution_mode,
        requested_engine,
        campaign_id,
        shard_id,
        shard_index,
        shard_count,
    )
end

end # module
