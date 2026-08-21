#!/usr/bin/env julia
# Performance history: append benchmark results to a flat per-commit log and
# print per-problem trends.
#
#   julia --project=. benchmark/history_log.jl record RESULT.toml \
#       [--suite=ladder] [--history=benchmark/history]
#   julia --project=. benchmark/history_log.jl trend [problem_id] \
#       [--history=benchmark/history] [--last=10]
#
# `record` appends one CSV row per solved result row, stamped with the
# current commit, UTC time, and tree cleanliness. The log is append-only so
# git merges stay trivial; `trend` reads it back and prints, for each problem
# in insertion order, the median-time sequence across commits -- the "is this
# branch actually faster" view that raw per-commit artifact directories do
# not give you.

module HistoryLog

using TOML
using Dates
using Printf

mutable struct Options
    mode::String
    result::String
    problem::Union{Nothing,String}
    suite::String
    history::String
    last::Int
end

function _option(args, prefix)
    for argument in args
        startswith(argument, prefix) || continue
        return String(split(argument, "="; limit=2)[2])
    end
    return nothing
end

function parse_args(args)
    isempty(args) && throw(ArgumentError(
        "usage: history_log.jl record RESULT.toml [--suite=name] " *
        "[--history=dir] | trend [problem_id] [--history=dir] [--last=N]",
    ))
    options = Options(args[1], args[2], nothing, "", "benchmark/history", 12)
    for argument in args[3:end]
        value = _option((argument,), "--suite=")
        value !== nothing && (options.suite = value; continue)
        value = _option((argument,), "--history=")
        value !== nothing && (options.history = value; continue)
        value = _option((argument,), "--last=")
        value !== nothing && (options.last = parse(Int, value); continue)
        startswith(argument, "--") &&
            throw(ArgumentError("unknown option $argument"))
        options.problem = argument
    end
    options.mode in ("record", "trend") ||
        throw(ArgumentError("unknown mode $(options.mode)"))
    options.mode == "record" && isempty(options.result) &&
        throw(ArgumentError("record requires RESULT.toml"))
    return options
end

function _git(arguments...)
    try
        return readchomp(Cmd(["git", "-C", pwd(), arguments...]))
    catch
        return "unknown"
    end
end

_csv_cell(value) = replace(string(value), ',' => ';', '\n' => ' ', '\r' => ' ')

const COLUMNS = [
    "commit", "committed_at", "dirty", "suite", "problem_id",
    "arithmetic", "provider", "status", "semantic_pass",
    "sample_count", "total_seconds_median", "total_seconds_mad",
    "allocated_bytes_median", "iterations",
]

function _row_from_document(document, suite, commit, stamp, dirty)
    meta = get(document, "result", nothing)
    rows = document["result"]
    output = String[]
    for row in rows
        values = [
            commit, stamp, dirty, suite,
            get(row, "problem_id", ""), get(row, "arithmetic", ""),
            get(row, "executed_provider", get(row, "requested_provider", "")),
            get(row, "status", ""),
            string(get(row, "semantic_pass", "")),
            get(row, "sample_count", ""),
            get(row, "total_seconds", ""), get(row, "total_seconds_mad", ""),
            get(row, "allocated_bytes", ""), get(row, "iterations", ""),
        ]
        push!(output, join(_csv_cell.(values), ","))
    end
    return output
end

function record(options)
    document = TOML.parsefile(options.result)
    rows = get(document, "result", nothing)
    rows === nothing && throw(ArgumentError(
        "$(options.result) has no \"result\" table; is it a runner output?",
    ))
    commit = _git("rev-parse", "HEAD")
    dirty = isempty(_git("status", "--porcelain")) ? "clean" : "dirty"
    stamp = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ")
    path = joinpath(options.history, "performance-log.csv")
    mkpath(dirname(abspath(path)))
    header = !isfile(path)
    open(path, "a") do io
        header && println(io, join(COLUMNS, ","))
        for line in _row_from_document(document, options.suite, commit, stamp, dirty)
            println(io, line)
        end
    end
    println("appended ", length(rows), " rows to ", path,
            " (commit ", commit[1:min(12, end)], ", ", dirty, ")")
    return path
end

function trend(options)
    path = joinpath(options.history, "performance-log.csv")
    isfile(path) || error("no performance log at $path; run `record` first")
    lines = readlines(path)
    length(lines) >= 2 || error("performance log has no rows")
    header = split(lines[1], ',')
    columns = Dict{String,Int}(name => i for (i, name) in enumerate(header))
    entries = Dict{String,Vector{Tuple{String,String,Float64}}}()
    for line in lines[2:end]
        fields = split(line, ',')
        length(fields) < length(header) && continue
        problem = fields[columns["problem_id"]]
        suite = fields[columns["suite"]]
        # problem_id is already globally unique (it carries its own prefix);
        # the suite filter only narrows what gets recorded.
        key = options.problem !== nothing ? options.problem : problem
        median = tryparse(Float64, fields[columns["total_seconds_median"]])
        median === nothing && continue
        push!(get!(entries, key, Tuple{String,String,Float64}[]),
              (fields[columns["commit"]][1:min(12, end)],
               fields[columns["committed_at"]], median))
    end
    isempty(entries) && error("no matching rows for trend")
    for key in sort!(collect(keys(entries)))
        series = entries[key]
        options.last > 0 && length(series) > options.last &&
            (series = series[end-options.last+1:end])
        println(key)
        baseline = series[1][3]
        for (commit, stamp, median) in series
            ratio = baseline > 0 ? median / baseline : NaN
            @printf("  %s %s  %8.3fs  (%.2fx first)\n", commit, stamp, median, ratio)
        end
    end
    return nothing
end

function main(args=ARGS)
    options = parse_args(args)
    return options.mode == "record" ? record(options) : trend(options)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    HistoryLog.main(ARGS)
end
