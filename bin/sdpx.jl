#!/usr/bin/env julia
# User-facing SDPX command line.
#
# The numerical bridge remains in sdpx_solve.jl.  This file deliberately owns
# only command-line policy: SDPB-style option names, all-auto defaults, help,
# input/output paths and a compact human-readable summary.

isdefined(Main, :SDPXSolveCLI) || include(joinpath(@__DIR__, "sdpx_solve.jl"))

module SDPXUserCLI
using ..SDPXSolveCLI
using JSON

const HELP = raw"""
SDPX — LP / SOCP / SDP solver frontend

Usage:
  sdpx <problem.json> [result.json] [options]
  sdpx [options] <problem.json>

High-precision example (SDPB-style):
  sdpx problem.json result.json \
      --precision=840 \
      --dualityGapThreshold=1e-80 \
      --primalErrorThreshold=1e-80 \
      --dualErrorThreshold=1e-80

All user-facing policies default to auto.  Backend-specific IPM/KKT/kernel
parameters are intentionally not part of this interface; inspect the returned
`plan` and `resolved_options` fields to see what SDPX selected.

Core options:
  --precision=auto|BITS|float64|float64x2|float64x3|float64x4|bigfloat
      Integer BITS selects BigFloat and parses the model at that bit precision.

  --dualityGapThreshold=auto|VALUE
  --primalErrorThreshold=auto|VALUE
  --dualErrorThreshold=auto|VALUE
      Independent stopping/certification targets.  Numeric text is preserved
      as text until it is parsed in the selected arithmetic.

  --maxIterations=auto|N
  --maxRuntime=auto|SECONDS
  --threads=auto|N
  --verbosity=auto|0|1|2|3

Automatic pipeline controls:
  --algorithm=auto|lp|socp|sdp
  --presolve=auto|on|off
  --scaling=auto|none|equilibrate
  --sparse=auto|on|off
  --formulation=auto|primal|normal_equations|augmented
  --equalitySolver=auto|normal_equations|qr
  --workingPrecisionPolicy=auto|fixed
  --certificate=auto|on|off

Output:
  --output=PATH       Result JSON path (otherwise second positional argument or
                      <problem>.result.json).
  --quiet             Do not print the solve summary.
  --help, -h          Show this help.

Input format:
  JSON bridge schema v1 (`docs/src/bridge-schema.md`).  The canonical benchmark
  registry under `benchmark/` records generated and external conic workloads;
  native SDPA/CBF/MPS command-line loaders are not yet provided.
"""

const VALUE_OPTIONS = Set([
    "precision",
    "dualityGapThreshold",
    "primalErrorThreshold",
    "dualErrorThreshold",
    "maxIterations",
    "maxRuntime",
    "threads",
    "verbosity",
    "algorithm",
    "presolve",
    "scaling",
    "sparse",
    "formulation",
    "equalitySolver",
    "workingPrecisionPolicy",
    "certificate",
    "output",
])

function _take_option(arguments, index)
    token = arguments[index]
    startswith(token, "--") || error("internal CLI parser error")
    body = token[3:end]
    inline_value = nothing
    key = body
    if occursin('=', body)
        key, value = split(body, '='; limit=2)
        inline_value = value
    end
    key in ("help", "quiet") && begin
        inline_value === nothing || error(
            "flag option --$key does not accept a value",
        )
        return key, "true", index + 1
    end
    key in VALUE_OPTIONS || error("unknown option --$key")
    if inline_value !== nothing
        return key, inline_value, index + 1
    end
    index < length(arguments) || error("--$key requires a value")
    return key, arguments[index + 1], index + 2
end

function parse_cli(arguments)
    options = Dict{String,String}()
    positional = String[]
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        if token == "-h"
            options["help"] = "true"
            index += 1
        elseif startswith(token, "--")
            key, value, index = _take_option(arguments, index)
            options[key] = value
        else
            push!(positional, token)
            index += 1
        end
    end
    get(options, "help", "false") == "true" && return (; help=true, options, positional)
    isempty(positional) && error("missing problem path; use --help for usage")
    length(positional) <= 2 || error("expected at most two positional paths")
    return (; help=false, options, positional)
end

function _canonical_switch(value::AbstractString, label::AbstractString)
    normalized = lowercase(strip(value))
    normalized in ("auto", "on", "off") ||
        error("--$label must be auto, on, or off")
    return normalized
end

function _overlay!(spec, cli_options)
    settings = get!(spec, "settings", Dict{String,Any}())

    if haskey(cli_options, "precision")
        raw = strip(cli_options["precision"])
        lower = lowercase(raw)
        if lower == "auto"
            spec["precision"] = "auto"
            settings["precision_bits"] = "auto"
        elseif all(isdigit, raw)
            bits = parse(Int, raw)
            bits >= 2 || error("--precision must be at least 2 bits")
            spec["precision"] = "BigFloat"
            settings["precision_bits"] = bits
        elseif lower in ("float64", "float64x2", "float64x3", "float64x4", "bigfloat")
            spec["precision"] = lower
            lower == "bigfloat" || (settings["precision_bits"] = "auto")
        else
            error("unsupported --precision=$raw")
        end
    end

    for key in ("dualityGapThreshold", "primalErrorThreshold", "dualErrorThreshold")
        haskey(cli_options, key) || continue
        settings[key] = cli_options[key]  # keep numeric text exact
    end

    haskey(cli_options, "maxIterations") &&
        (settings["maximumIterations"] = cli_options["maxIterations"])
    haskey(cli_options, "maxRuntime") &&
        (settings["maxRuntime"] = cli_options["maxRuntime"])
    haskey(cli_options, "threads") &&
        (settings["threads"] = cli_options["threads"])
    haskey(cli_options, "verbosity") &&
        (settings["verbosity"] = cli_options["verbosity"])

    for key in ("algorithm", "scaling", "formulation", "equalitySolver", "workingPrecisionPolicy")
        haskey(cli_options, key) && (settings[key] = lowercase(strip(cli_options[key])))
    end
    for key in ("presolve", "sparse", "certificate")
        haskey(cli_options, key) && (settings[key] = _canonical_switch(cli_options[key], key))
    end
    return spec
end

function _default_output(input_path)
    base, _ = splitext(input_path)
    return base * ".result.json"
end

function _write_response(path, response)
    open(path, "w") do io
        JSON.print(io, response)
    end
end

function _print_summary(response, output_path)
    if get(response, "success", false) != true
        println(stderr, "SDPX error: ", get(response, "error", "unknown error"))
        println(stderr, "result: ", output_path)
        return
    end
    println("SDPX status:           ", response["status"])
    precision_suffix = haskey(response, "resolved_options") ?
        string(" / ", response["resolved_options"]["precision_bits"], " bits") : ""
    println("precision:             ", response["precision"], precision_suffix)
    println("objective:             ", response["objective"])
    println("duality gap:           ", response["relative_gap"])
    println("primal error:          ", response["primal_residual"])
    println("dual error:            ", response["dual_residual"])
    println("iterations:            ", response["iterations"])
    if haskey(response, "plan")
        plan = response["plan"]
        println("auto plan:             ", plan["algorithm"], " / ",
                plan["kkt_backend"], " / ", plan["gram_kernel"])
    end
    if haskey(response, "certificate")
        println("certificate:           ", response["certificate"]["valid"] ? "valid" : "FAILED")
    end
    println("result:                ", output_path)
end

function main(arguments=ARGS)
    parsed = try
        parse_cli(arguments)
    catch exception
        exception isa InterruptException && rethrow()
        println(stderr, "error: ", sprint(showerror, exception))
        println(stderr, "use --help for usage")
        return 2
    end
    if parsed.help
        print(HELP)
        return 0
    end

    input_path = parsed.positional[1]
    output_path = get(
        parsed.options,
        "output",
        length(parsed.positional) == 2 ? parsed.positional[2] : _default_output(input_path),
    )
    response, code = try
        isfile(input_path) || error("input file not found: $input_path")
        spec = JSON.parsefile(input_path)
        _overlay!(spec, parsed.options)
        SDPXSolveCLI.solve_specification(spec), 0
    catch exception
        exception isa InterruptException && rethrow()
        SDPXSolveCLI.failure_response(sprint(showerror, exception)), 1
    end
    _write_response(output_path, response)
    get(parsed.options, "quiet", "false") == "true" || _print_summary(response, output_path)
    return code
end

end # module SDPXUserCLI

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(SDPXUserCLI.main())
end
