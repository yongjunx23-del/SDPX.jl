module FreshProcessCampaign

using TOML
using Dates
using LinearAlgebra
using Statistics
using Printf

export ChildRecord, run_campaign, aggregate_campaign, write_summary,
       campaign_main

const ROOT = @__DIR__
const REPOSITORY = normpath(joinpath(ROOT, ".."))
const FRESH_SCHEMA_VERSION = 1

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
    fields = (
        "conic_formulation", "planned_formulation", "executed_formulation",
        "planned_backend", "executed_backend", "planned_provider",
        "executed_provider", "executed_specialization", "psd_lift_used",
        "fallback_reason", "la_fallback_reason",
    )
    return join((_text(get(row, field, "")) for field in fields), "|")
end

const PAIRING_FIELDS = (
    # Problem identity and formulation/tolerance are part of the experiment,
    # not merely labels.  A mismatch means the rows cannot form a timing
    # sample, even when all processes happened to exit successfully.
    "suite", "catalog_name", "catalog_version",
    "problem_id", "name", "family", "problem_type", "source",
    "arithmetic", "precision_bits", "requested_provider",
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
    "solver_source_sha256",
    "mfla_commit", "bfla_commit",
)

function _pairing_failures(rows)
    failures = String[]
    isempty(rows) && return ["no_rows"]
    first_row = first(rows)
    for field in ("catalog_name", "catalog_version")
        any(isempty(_text(get(row, field, ""))) for row in rows) &&
            push!(failures, "$(field)_missing")
    end
    for field in PAIRING_FIELDS
        values = [_text(get(row, field, "")) for row in rows]
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
    objective_ok, objective_failure = _objective_parity(rows)
    objective_ok || push!(failures, objective_failure)
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
    )
end

function _relpath_list(paths, campaign_dir)
    return [relpath(path, campaign_dir) for path in paths]
end

"""Aggregate already collected child rows, failing closed on semantic drift.

The input is a vector of `ChildRecord`s.  With `diagnostic=false` every child
must have exit code zero, a finite timing, `semantic_pass=true`, and
`certificate_valid=true`; all pairing fields and semantic values must agree.
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
    if length(rows) == repetitions
        append!(failures, _pairing_failures(rows))
        if expected !== nothing
            expected_fields = (
                (:suite, "suite"), (:problem_id, "problem_id"),
                (:arithmetic, "arithmetic"), (:provider, "requested_provider"),
            )
            for (name, field) in expected_fields
                wanted = _text(getproperty(expected, name))
                all(_text(get(row, field, "")) == wanted for row in rows) ||
                    push!(failures, "selection:$field")
            end
        end
        for (index, row) in enumerate(rows)
            semantic = _parse_bool(get(row, "semantic_pass", ""))
            certificate = _parse_bool(get(row, "certificate_valid", ""))
            semantic === true || push!(failures, "child_$(index):semantic_pass")
            certificate === true || push!(failures, "child_$(index):certificate_valid")
        end
    end
    timing = [_parse_finite(get(row, "total_seconds", "")) for row in rows]
    any(value -> value === nothing || value <= 0, timing) &&
        push!(failures, "timing_missing_or_invalid")
    timing_summary = all(value -> value !== nothing && value > 0, timing) ?
        _summary(timing) : nothing
    memory_fields = ("allocated_bytes", "process_peak_rss_bytes", "workspace_bytes")
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
        "status" => get(base, "status", ""),
        "iterations" => get(base, "iterations", ""),
        "objective" => get(base, "objective", ""),
        "route" => isempty(rows) ? "" : _route(base),
        "input_fingerprint" => get(base, "input_fingerprint", ""),
        "semantic_pass" => length(rows) == repetitions &&
                           all(_parse_bool(get(row, "semantic_pass", "")) === true for row in rows),
        "certificate_valid" => length(rows) == repetitions &&
                               all(_parse_bool(get(row, "certificate_valid", "")) === true for row in rows),
        "repetitions" => repetitions,
        "child_exit_codes" => [record.exit_code for record in ordered],
        "child_raw_toml" => raw_toml,
        "child_raw_tsv" => raw_tsv,
        "child_logs" => logs,
        "timing_valid" => timing_summary !== nothing,
        "aggregation_mode" => diagnostic_label,
        "aggregation_valid" => aggregation_valid,
        "pairing_valid" => length(rows) == repetitions &&
                           !any(startswith(failure, "pairing:") ||
                                failure in ("status", "iterations", "route",
                                            "input_fingerprint", "input_fingerprint_missing")
                                for failure in failures),
        "failure_reasons" => join(unique(failures), ","),
    )
    if timing_summary !== nothing
        for (name, value) in pairs(timing_summary)
            result["total_seconds_$(name)"] = value
        end
    else
        for name in (:median, :min, :max, :mad, :spread)
            result["total_seconds_$(name)"] = ""
        end
    end
    for (field, summary) in memory
        if summary === nothing
            for name in (:median, :min, :max, :mad, :spread)
                result["$(field)_$(name)"] = ""
            end
        else
            for (name, value) in pairs(summary)
                result["$(field)_$(name)"] = value
            end
        end
    end
    if iterations === nothing
        for name in (:median, :min, :max, :mad, :spread)
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
        "suite", "problem_id", "arithmetic", "requested_provider", "status",
        "iterations", "objective", "route", "input_fingerprint",
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
            result = get(document, "result", Any[])
            length(result) == 1 || (failure = isempty(failure) ? "row_count" : failure * ",row_count")
            length(result) == 1 && (row = Dict{String,Any}(string(k) => v for (k, v) in first(result)))
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
)
    repetitions >= 3 || throw(ArgumentError("repetitions must be >= 3"))
    threads >= 1 || throw(ArgumentError("threads must be >= 1"))
    blas_threads >= 1 || throw(ArgumentError("blas_threads must be >= 1"))
    isempty(problem) && throw(ArgumentError("problem must be non-empty"))
    project = abspath(project)
    campaign_dir = abspath(campaign_dir)
    mkpath(campaign_dir)
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
                  arithmetic=string(arithmetic), provider=string(provider)),
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
    )
end

end # module
