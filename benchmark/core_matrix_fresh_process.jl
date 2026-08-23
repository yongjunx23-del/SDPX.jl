#!/usr/bin/env julia

module CoreMatrixFreshProcess

using Dates
using TOML

const ROOT = @__DIR__
const REPOSITORY = normpath(joinpath(ROOT, ".."))
const MATRIX_SCHEMA_VERSION = 1

if isdefined(Main, :SDPXBenchmarkRegistry)
    const Registry = Main.SDPXBenchmarkRegistry
else
    include(joinpath(ROOT, "SDPXBenchmarkRegistry.jl"))
    const Registry = SDPXBenchmarkRegistry
end

if isdefined(Main, :FreshProcessCampaign)
    const Campaign = Main.FreshProcessCampaign
else
    include(joinpath(ROOT, "fresh_process_campaign.jl"))
    const Campaign = FreshProcessCampaign
end

export core_matrix_entries, run_core_matrix_campaign, matrix_main

core_matrix_entries() = Registry.suite_entries(:core_matrix)

function _campaign_slug(entry)
    problem = replace(entry.problem_id, '/' => '_')
    return join((problem, string(entry.arithmetic), string(entry.provider)), "__")
end

_tsv_cell(value) = value === nothing ? "" :
    replace(string(value), '\t' => ' ', '\n' => ' ', '\r' => ' ')

function _matrix_columns(rows)
    preferred = [
        "case_id", "family", "problem_id", "arithmetic",
        "requested_provider", "status", "iterations", "objective",
        "semantic_pass", "certificate_valid", "aggregation_valid",
        "pairing_valid", "timing_valid", "total_seconds_median",
        "total_seconds_mad", "total_seconds_spread",
        "allocated_bytes_median", "process_peak_rss_bytes_median",
        "workspace_bytes_median", "route", "input_fingerprint",
        "failure_reasons", "campaign_dir",
    ]
    keys_union = Set{String}()
    for row in rows
        union!(keys_union, string.(keys(row)))
    end
    rest = sort!(collect(setdiff(keys_union, Set(preferred))))
    return vcat(filter(column -> column in keys_union, preferred), rest)
end

function _write_matrix_summary(document::AbstractDict, path::AbstractString)
    root, extension = splitext(path)
    toml_path = extension == ".toml" ? path : root * ".toml"
    tsv_path = extension == ".tsv" ? path : root * ".tsv"
    mkpath(dirname(toml_path))
    open(toml_path, "w") do io
        TOML.print(io, document; sorted=true)
    end
    rows = [
        Dict{String,Any}(string(key) => value for (key, value) in row)
        for row in get(document, "result", Any[])
    ]
    isempty(rows) && throw(ArgumentError("core matrix summary contains no rows"))
    columns = _matrix_columns(rows)
    open(tsv_path, "w") do io
        println(io, join(columns, '\t'))
        for row in rows
            println(io, join((_tsv_cell(get(row, column, "")) for column in columns), '\t'))
        end
    end
    return (toml=toml_path, tsv=tsv_path)
end

function _matrix_document(rows; diagnostic::Bool=false)
    selection_count = length(rows)
    valid_count = count(
        row -> get(row, "aggregation_valid", false) === true,
        rows,
    )
    semantic_count = count(
        row -> get(row, "semantic_pass", false) === true &&
               get(row, "certificate_valid", false) === true,
        rows,
    )
    matrix_valid = selection_count == 9 && valid_count == selection_count &&
                   semantic_count == selection_count && !diagnostic
    return Dict{String,Any}(
        "schema_version" => MATRIX_SCHEMA_VERSION,
        "generated_at" => string(Dates.now()),
        "campaign" => Dict{String,Any}(
            "name" => "core_matrix",
            "mode" => diagnostic ? "diagnostic" : "strict",
            "selection_count" => selection_count,
            "valid_count" => valid_count,
            "semantic_count" => semantic_count,
            "matrix_valid" => matrix_valid,
        ),
        "result" => rows,
    )
end

function _matrix_row(entry, campaign, matrix_dir)
    raw_rows = get(campaign.document, "result", Any[])
    length(raw_rows) == 1 || throw(ArgumentError(
        "fresh-process campaign for $(entry.problem_id) returned " *
        "$(length(raw_rows)) summary rows",
    ))
    row = Dict{String,Any}(
        string(key) => value for (key, value) in first(raw_rows)
    )
    row["case_id"] = _campaign_slug(entry)
    row["family"] = string(Registry.benchmark_spec(entry.problem_id).family)
    row["campaign_dir"] = relpath(dirname(campaign.summary.toml), matrix_dir)
    return row
end

"""
    run_core_matrix_campaign(; kwargs...)

Run every fixed LP/SOCP/SDP × Float64/Float64x4/BigFloat256 row through at
least three independent Julia processes. Each row keeps its raw child artifacts
and one strict aggregate; the top-level TOML/TSV records all nine aggregates.

Strict mode writes all evidence and then fails when any row is skipped,
semantically invalid, uncertified, unpaired, or unstable. `diagnostic=true`
keeps the same evidence but deliberately sets `matrix_valid=false` and does not
throw, which is useful when checking an environment before installing optional
providers.
"""
function run_core_matrix_campaign(
    ;
    repetitions::Integer=3,
    campaign_dir::AbstractString=joinpath(ROOT, "out", "core_matrix_fresh"),
    project::AbstractString=REPOSITORY,
    cache_dir::Union{Nothing,AbstractString}=nothing,
    threads::Integer=1,
    blas_threads::Integer=1,
    diagnostic::Bool=false,
    verbose::Bool=false,
)
    repetitions >= 3 || throw(ArgumentError("repetitions must be >= 3"))
    threads >= 1 || throw(ArgumentError("threads must be >= 1"))
    blas_threads >= 1 || throw(ArgumentError("blas_threads must be >= 1"))
    matrix_dir = abspath(campaign_dir)
    mkpath(matrix_dir)

    rows = Dict{String,Any}[]
    for entry in core_matrix_entries()
        row_dir = joinpath(matrix_dir, _campaign_slug(entry))
        verbose && println(
            "core_matrix ", entry.problem_id, " ", entry.arithmetic,
            " provider=", entry.provider,
        )
        campaign = Campaign.run_campaign(
            :core_matrix;
            problem=entry.problem_id,
            arithmetic=entry.arithmetic,
            provider=entry.provider,
            repetitions=repetitions,
            campaign_dir=row_dir,
            project=project,
            cache_dir=cache_dir,
            threads=threads,
            blas_threads=blas_threads,
            diagnostic=diagnostic,
            verbose=verbose,
        )
        push!(rows, _matrix_row(entry, campaign, matrix_dir))
    end

    document = _matrix_document(rows; diagnostic=diagnostic)
    paths = _write_matrix_summary(
        document,
        joinpath(matrix_dir, "core_matrix_summary.toml"),
    )
    matrix_valid = document["campaign"]["matrix_valid"] === true
    if !diagnostic && !matrix_valid
        invalid = [
            row["case_id"] for row in rows
            if get(row, "aggregation_valid", false) !== true ||
               get(row, "semantic_pass", false) !== true ||
               get(row, "certificate_valid", false) !== true
        ]
        throw(ErrorException(
            "core matrix fresh-process gate failed for $(join(invalid, ", ")); " *
            "evidence was written to $(paths.toml)",
        ))
    end
    return (; document, rows, paths)
end

function _option_value(argument, prefix)
    startswith(argument, prefix) || return nothing
    return split(argument, "="; limit=2)[2]
end

function matrix_main(args=ARGS)
    repetitions = 3
    campaign_dir = joinpath(ROOT, "out", "core_matrix_fresh")
    project = REPOSITORY
    cache_dir = nothing
    threads = 1
    blas_threads = 1
    diagnostic = false
    verbose = false

    for argument in args
        value = _option_value(argument, "--repetitions=")
        value !== nothing && (repetitions = parse(Int, value); continue)
        value = _option_value(argument, "--campaign-dir=")
        value !== nothing && (campaign_dir = value; continue)
        value = _option_value(argument, "--project=")
        value !== nothing && (project = value; continue)
        value = _option_value(argument, "--cache-dir=")
        value !== nothing && (cache_dir = value; continue)
        value = _option_value(argument, "--threads=")
        value !== nothing && (threads = parse(Int, value); continue)
        value = _option_value(argument, "--blas-threads=")
        value !== nothing && (blas_threads = parse(Int, value); continue)
        argument == "--diagnostic" && (diagnostic = true; continue)
        argument == "--verbose" && (verbose = true; continue)
        throw(ArgumentError("unknown option $argument"))
    end

    result = run_core_matrix_campaign(
        ;
        repetitions=repetitions,
        campaign_dir=campaign_dir,
        project=project,
        cache_dir=cache_dir,
        threads=threads,
        blas_threads=blas_threads,
        diagnostic=diagnostic,
        verbose=verbose,
    )
    campaign = result.document["campaign"]
    println(
        "core_matrix selection=", campaign["selection_count"],
        " valid=", campaign["valid_count"],
        " semantic=", campaign["semantic_count"],
        " matrix_valid=", campaign["matrix_valid"],
        " output=", result.paths.toml,
    )
    return result
end

end # module

if abspath(PROGRAM_FILE) == (@__FILE__)
    CoreMatrixFreshProcess.matrix_main(ARGS)
end
