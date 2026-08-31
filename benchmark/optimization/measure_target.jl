#!/usr/bin/env julia
using TOML
using Printf
include(joinpath(@__DIR__, "profile_catalog.jl"))
using .ProfileCatalog

function _args(args)
    out = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || throw(ArgumentError("expected --key=value"))
        pair = split(arg[3:end], "="; limit=2)
        length(pair) == 2 || throw(ArgumentError("expected --key=value"))
        out[pair[1]] = pair[2]
    end
    out
end

function _head()
    return get(ENV, "SDPX_EXPECTED_SOURCE_COMMIT", "") == "" ?
        readchomp(`git rev-parse HEAD`) : ENV["SDPX_EXPECTED_SOURCE_COMMIT"]
end

function _row(manifest, selected)
    key = haskey(manifest, "row") ? "row" : "case"
    rows = get(manifest, key, Any[])
    for row in rows
        String(get(row, "case_key", "")) == selected && return row
    end
    throw(ArgumentError("selected case is absent from manifest: $selected"))
end

function _require_live_identity(manifest, selected_row)
    source = String(get(manifest, "source_commit", ""))
    occursin(r"^[0-9a-fA-F]{40}$", source) ||
        throw(ArgumentError("live manifest source_commit must be an exact 40-hex SHA"))
    source == _head() || throw(ArgumentError(
        "hotspot manifest is stale: source=$source expected=$(_head())"))
    row_source = String(get(selected_row, "source_commit", source))
    row_source == source || throw(ArgumentError("row/source commit mismatch"))
    return nothing
end

function _validate_manifest_samples(row)
    for field in ("sample_seconds", "sample_iterations", "sample_status",
                  "sample_certificate_valid", "sample_semantic_pass", "sample_objective")
        haskey(row, field) || throw(ArgumentError("manifest missing $field"))
        length(row[field]) == 3 || throw(ArgumentError("$field must contain exactly three samples"))
    end
    all(x -> x isa Real && isfinite(x) && x > 0, row["sample_seconds"]) ||
        throw(ArgumentError("invalid sample seconds"))
    all(==(String(get(row, "status", ""))), String.(row["sample_status"])) ||
        throw(ArgumentError("sample status mismatch"))
    all(==(true), Bool.(row["sample_certificate_valid"])) ||
        throw(ArgumentError("certificate sample failed"))
    all(==(true), Bool.(row["sample_semantic_pass"])) ||
        throw(ArgumentError("semantic sample failed"))
    length(unique(Int.(row["sample_iterations"]))) == 1 ||
        throw(ArgumentError("iteration nondeterminism"))
    reference = get(row, "reference_objective", "")
    tol = get(row, "objective_tolerance", "")
    if reference != "" && tol != ""
        ref, delta = parse(Float64, string(reference)), parse(Float64, string(tol))
        all(x -> abs(Float64(x) - ref) <= delta, row["sample_objective"]) ||
            throw(ArgumentError("objective/reference interval failed"))
    end
    routes = [String(get(row, field, "")) for field in
        ("requested_route", "planned_route", "executed_route")]
    all(!isempty, routes) || throw(ArgumentError("route receipt incomplete"))
    return nothing
end

function main(args=ARGS)
    opts = _args(args)
    manifest_path = get(opts, "manifest", get(ENV, "SDPX_HOTSPOT_MANIFEST", ""))
    isempty(manifest_path) && throw(ArgumentError("--manifest is required"))
    manifest = TOML.parsefile(manifest_path)
    selected = String(get(manifest, "selected_case_key", ""))
    isempty(selected) && throw(ArgumentError("manifest has no selected_case_key"))
    fixture = get(ENV, "SDPX_PROFILE_FIXTURE", "0") == "1"
    fixture && get(ENV, "SDPX_OPTIMIZATION_TEST_MODE", "0") != "1" &&
        throw(ArgumentError("fixture mode requires explicit SDPX_OPTIMIZATION_TEST_MODE=1"))
    selected_row = _row(manifest, selected)
    fixture || _require_live_identity(manifest, selected_row)
    _validate_manifest_samples(selected_row)
    solve_eligible = Bool(get(selected_row, "solve_eligible", false))
    build_only = Bool(get(selected_row, "build_only", true))
    solve_eligible && !build_only || throw(ArgumentError("selected case is build-only"))
    metric_values = haskey(selected_row, "sample_core_seconds") &&
        length(selected_row["sample_core_seconds"]) == 3 ?
        Float64.(selected_row["sample_core_seconds"]) : Float64.(selected_row["sample_seconds"])
    all(isfinite, metric_values) || throw(ArgumentError("invalid core metric"))
    solver_values = Float64.(selected_row["sample_seconds"])
    median = ProfileCatalog._median(metric_values)
    solver_median = ProfileCatalog._median(solver_values)
    trajectory = get(selected_row, "trajectory_sha", "")
    println("METRIC target=", selected,
        " solver_seconds=", @sprintf("%.9f", solver_median),
        " core_seconds=", @sprintf("%.9f", median),
        " allocations=", maximum(Int.(get(selected_row, "allocation_bytes", [0]))),
        " iterations=", first(Int.(selected_row["sample_iterations"])),
        " certificate_valid=true semantic_pass=true",
        isempty(String(trajectory)) ? "" : " trajectory_sha=$(trajectory)")
    out = get(opts, "output", get(ENV, "SDPX_METRIC_OUTPUT", ""))
    isempty(out) || open(out, "w") do io
        TOML.print(io, Dict("metric_schema"=>2, "case_key"=>selected,
            "source_commit"=>String(get(manifest, "source_commit", "fixture")),
            "solver_median_seconds"=>solver_median, "core_median_seconds"=>median,
            "iterations"=>first(Int.(selected_row["sample_iterations"])),
            "certificate_valid"=>true, "semantic_pass"=>true,
            "sample_seconds"=>solver_values, "sample_core_seconds"=>metric_values,
            "sample_objective"=>Float64.(selected_row["sample_objective"]),
            "sample_trajectory_sha"=>get(selected_row, "sample_trajectory_sha", String[]),
            "objective_tolerance"=>get(selected_row, "objective_tolerance", ""),
            "allocation_bytes"=>Int.(get(selected_row, "allocation_bytes", [0]))))
    end
end
main()
