#!/usr/bin/env julia
using TOML
using Printf
include(joinpath(@__DIR__, "profile_catalog.jl"))
using .ProfileCatalog

function _args(args)
    out = Dict{String,String}()
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || throw(ArgumentError("expected --key=value"))
        pair = split(args[i][3:end], "="; limit=2)
        length(pair) == 2 || throw(ArgumentError("expected --key=value"))
        out[pair[1]] = pair[2]; i += 1
    end
    out
end

function main(args=ARGS)
    opts = _args(args)
    manifest_path = get(opts, "manifest", get(ENV, "SDPX_HOTSPOT_MANIFEST", ""))
    isempty(manifest_path) && throw(ArgumentError("--manifest is required"))
    manifest = TOML.parsefile(manifest_path)
    haskey(manifest, "selected_case_key") || throw(ArgumentError("manifest has no selected_case_key"))
    selected = String(manifest["selected_case_key"])
    rows = haskey(ENV, "SDPX_PROFILE_FIXTURE") ? ProfileCatalog.fixture_rows() : ProfileCatalog.profile_catalog()
    case = findfirst(r -> r.case_key == selected, rows)
    case === nothing && throw(ArgumentError("selected case is absent from current catalog: $selected"))
    row = rows[case]
    row.solve_eligible && !row.build_only || throw(ArgumentError("selected case is build-only"))
    row.semantic_pass && row.certificate_valid || throw(ArgumentError("selected case failed semantic/certificate gate"))
    length(row.sample_seconds) == 3 || throw(ArgumentError("target requires exactly three timed samples"))
    length(row.sample_iterations) == 3 || throw(ArgumentError("missing per-sample iterations"))
    length(unique(row.sample_iterations)) == 1 || throw(ArgumentError("iteration nondeterminism"))
    metric = !isempty(row.sample_core_seconds) ? row.sample_core_seconds : row.sample_seconds
    median = ProfileCatalog._median(metric)
    solver_median = ProfileCatalog._median(row.sample_seconds)
    println("METRIC target=", row.case_key,
        " solver_seconds=", @sprintf("%.9f", solver_median),
        " core_seconds=", @sprintf("%.9f", median),
        " allocations=", isempty(row.allocation_bytes) ? 0 : maximum(row.allocation_bytes),
        " iterations=", row.iterations,
        " certificate_valid=true semantic_pass=true")
    out = get(opts, "output", get(ENV, "SDPX_METRIC_OUTPUT", ""))
    isempty(out) || open(out, "w") do io
        TOML.print(io, Dict("metric_schema"=>1, "case_key"=>row.case_key,
            "solver_median_seconds"=>solver_median, "core_median_seconds"=>median,
            "iterations"=>row.iterations, "certificate_valid"=>true,
            "semantic_pass"=>true, "sample_seconds"=>row.sample_seconds,
            "sample_core_seconds"=>row.sample_core_seconds,
            "allocation_bytes"=>row.allocation_bytes))
    end
end
main()
