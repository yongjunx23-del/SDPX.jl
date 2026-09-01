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
    commit = readchomp(`git rev-parse HEAD`)
    occursin(r"^[0-9a-f]{40}$", commit) || throw(ArgumentError("invalid checked-out HEAD"))
    return commit
end

function _tree()
    tree = readchomp(`git rev-parse 'HEAD^{tree}'`)
    occursin(r"^[0-9a-f]{40}$", tree) || throw(ArgumentError("invalid checked-out tree"))
    return tree
end

const _TRAJECTORY_SHA256 = r"^[0-9a-f]{64}$"

function _valid_trajectory!(container, label)
    container isa AbstractDict || throw(ArgumentError("$label receipt must be an AbstractDict"))
    sem = String(get(container, "trajectory_semantics", ""))
    sha = String(get(container, "trajectory_sha", ""))
    reason = String(get(container, "trajectory_reason", ""))
    if sem == "sha256"
        occursin(_TRAJECTORY_SHA256, sha) && !isempty(reason) ||
            throw(ArgumentError("$label sha256 trajectory requires 64 lowercase hex and a reason"))
    elseif sem == "not_applicable"
        isempty(sha) && !isempty(reason) ||
            throw(ArgumentError("$label not_applicable trajectory requires empty SHA and reason"))
    else
        throw(ArgumentError("$label trajectory semantics must be sha256 or not_applicable"))
    end
    return nothing
end

function _require_receipt(row)
    receipt = get(row, "receipt", nothing)
    receipt isa AbstractDict || throw(ArgumentError("live selected row requires receipt::AbstractDict"))
    required = ("source_commit", "tree_fingerprint", "catalog", "family", "instance",
        "input_fingerprint", "project_sha256", "manifest_sha256", "catalog_run_id",
        "catalog_artifact_sha256", "environment_fingerprint", "provider_fingerprint",
        "provider_version", "cpu", "julia_threads", "blas_threads", "omp_threads",
        "gc_threads", "actual_objective", "objective_interval", "resolved_tolerances",
        "route_receipt", "requested_route", "planned_route", "executed_route", "requested_formulation",
        "planned_formulation", "executed_formulation", "requested_backend",
        "planned_backend", "executed_backend", "requested_provider", "planned_provider",
        "executed_provider", "requested_kernel", "planned_kernel", "executed_kernel",
        "reuse", "certificate_kind", "certificate_failures", "iterations",
        "trajectory_semantics", "trajectory_reason", "warmup_excluded", "sample_count")
    all(haskey(receipt, key) && receipt[key] !== nothing &&
        !(receipt[key] isa AbstractString && isempty(receipt[key])) for key in required) ||
        throw(ArgumentError("live selected row receipt is incomplete"))
    occursin(r"^[0-9a-f]{40}$", String(receipt["source_commit"])) ||
        throw(ArgumentError("receipt source_commit must be exact lowercase SHA-1"))
    for field in ("project_sha256", "manifest_sha256", "catalog_artifact_sha256", "input_fingerprint", "environment_fingerprint", "provider_fingerprint")
        occursin(r"^[0-9a-f]{64}$", String(receipt[field])) ||
            throw(ArgumentError("receipt $field must be exact lowercase SHA-256"))
    end
    receipt["warmup_excluded"] == 1 && receipt["sample_count"] == 3 ||
        throw(ArgumentError("receipt sample accounting invalid"))
    interval = receipt["objective_interval"]
    interval isa AbstractDict && Set(keys(interval)) == Set(("lower", "upper")) &&
        all(interval[key] !== nothing && !isempty(string(interval[key])) for key in ("lower", "upper")) ||
        throw(ArgumentError("objective interval must be a closed nonempty dictionary"))
    tolerances = receipt["resolved_tolerances"]
    tolerances isa AbstractDict && Set(keys(tolerances)) == Set(("primal", "dual", "gap")) &&
        all(tolerances[key] !== nothing && !isempty(string(tolerances[key])) for key in ("primal", "dual", "gap")) ||
        throw(ArgumentError("resolved tolerances must be a closed nonempty dictionary"))
    route = receipt["route_receipt"]
    route isa AbstractDict || throw(ArgumentError("route_receipt must be an AbstractDict"))
    route_keys = ("requested_route", "planned_route", "executed_route", "requested_formulation", "planned_formulation", "executed_formulation", "requested_backend", "planned_backend", "executed_backend", "requested_provider", "planned_provider", "executed_provider", "requested_kernel", "planned_kernel", "executed_kernel", "reuse")
    Set(keys(route)) == Set(route_keys) && all(!isempty(String(route[key])) for key in route_keys) ||
        throw(ArgumentError("route_receipt is not a closed complete receipt"))
    _valid_trajectory!(receipt, "live")
    return receipt
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
    tree = String(get(manifest, "tree_fingerprint", ""))
    tree == _tree() || throw(ArgumentError("hotspot manifest tree identity mismatch"))
    for field in ("catalog_run_id", "catalog_artifact_sha256", "catalog_fingerprint",
                  "environment_fingerprint", "provider_fingerprint")
        isempty(String(get(manifest, field, ""))) &&
            throw(ArgumentError("manifest missing $field"))
    end
    row_source = String(get(selected_row, "source_commit", source))
    row_source == source || throw(ArgumentError("row/source commit mismatch"))
    return nothing
end

function _validate_manifest_samples(row; fixture=false)
    # Live rows are never display-only: reject a missing/empty/nested-invalid
    # receipt before inspecting any convenience columns.
    live_receipt = fixture ? nothing : _require_receipt(row)
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
    isempty(string(reference)) && throw(ArgumentError("missing reference objective"))
    isempty(string(tol)) && throw(ArgumentError("missing objective tolerance"))
    if reference != "" && tol != ""
        ref, delta = parse(Float64, string(reference)), parse(Float64, string(tol))
        all(x -> abs(Float64(x) - ref) <= delta, row["sample_objective"]) ||
            throw(ArgumentError("objective/reference interval failed"))
    end
    routes = [String(get(row, field, "")) for field in
        ("requested_route", "planned_route", "executed_route")]
    all(!isempty, routes) || throw(ArgumentError("route receipt incomplete"))
    receipt = fixture ? get(row, "receipt", nothing) : live_receipt
    if receipt !== nothing
        semantics = string(get(receipt, "trajectory_semantics", get(row, "trajectory_semantics", "")))
        semantics in ("sha256", "not_applicable") ||
            throw(ArgumentError("trajectory semantics invalid"))
        _valid_trajectory!(receipt, fixture ? "fixture" : "live")
    end
    if !fixture
        required = ("source_commit", "tree_fingerprint", "catalog", "family",
            "instance", "input_fingerprint", "project_sha256", "manifest_sha256",
            "catalog_run_id", "catalog_artifact_sha256", "environment_fingerprint",
            "provider_fingerprint", "provider_version", "cpu", "julia_threads",
            "blas_threads", "omp_threads", "gc_threads", "objective_interval",
            "actual_objective", "resolved_tolerances", "requested_route",
            "planned_route", "executed_route", "requested_formulation",
            "planned_formulation", "executed_formulation", "requested_backend",
            "planned_backend", "executed_backend", "requested_provider",
            "planned_provider", "executed_provider", "requested_kernel",
            "planned_kernel", "executed_kernel", "reuse", "certificate_kind",
            "certificate_failures", "iterations", "trajectory_semantics",
            "warmup_excluded", "sample_count")
        all(haskey(receipt, key) && receipt[key] !== nothing &&
            !(receipt[key] isa AbstractString && isempty(receipt[key])) for key in required) ||
            throw(ArgumentError("complete receipt missing fields"))
        receipt["warmup_excluded"] == 1 && receipt["sample_count"] == 3 ||
            throw(ArgumentError("receipt sample accounting invalid"))
        interval = receipt["objective_interval"]
        interval isa AbstractDict && haskey(interval, "lower") && haskey(interval, "upper") ||
            throw(ArgumentError("objective interval must contain lower and upper"))
        tolerance = receipt["resolved_tolerances"]
        tolerance isa AbstractDict && all(haskey(tolerance, key) for key in ("primal", "dual", "gap")) ||
            throw(ArgumentError("resolved tolerances must contain primal, dual, gap"))
        semantics = String(receipt["trajectory_semantics"])
        semantics in ("sha256", "not_applicable") ||
            throw(ArgumentError("trajectory semantics invalid"))
        _valid_trajectory!(receipt, "live")
    end
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
    _validate_manifest_samples(selected_row; fixture=fixture)
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
            "tree_fingerprint"=>String(get(manifest, "tree_fingerprint", "")),
            "catalog_run_id"=>String(get(manifest, "catalog_run_id", "")),
            "catalog_artifact_sha256"=>String(get(manifest, "catalog_artifact_sha256", "")),
            "catalog_fingerprint"=>String(get(manifest, "catalog_fingerprint", "")),
            "environment_fingerprint"=>String(get(manifest, "environment_fingerprint", "")),
            "provider_fingerprint"=>String(get(manifest, "provider_fingerprint", "")),
            "solver_median_seconds"=>solver_median, "core_median_seconds"=>median,
            "iterations"=>first(Int.(selected_row["sample_iterations"])),
            "certificate_valid"=>true, "semantic_pass"=>true,
            "sample_seconds"=>solver_values, "sample_core_seconds"=>metric_values,
            "sample_objective"=>Float64.(selected_row["sample_objective"]),
            "sample_trajectory_sha"=>get(selected_row, "sample_trajectory_sha", String[]),
            "trajectory_semantics"=>get(selected_row, "trajectory_semantics", "not_applicable"),
            "objective_interval"=>Dict("lower"=>get(selected_row, "reference_lower", ""), "upper"=>get(selected_row, "reference_upper", "")),
            "resolved_tolerances"=>get(selected_row, "resolved_tolerances", ""),
            "requested_route"=>get(selected_row, "requested_route", ""),
            "planned_route"=>get(selected_row, "planned_route", ""),
            "executed_route"=>get(selected_row, "executed_route", ""),
            "certificate_kind"=>"summary", "certificate_failures"=>String[],
            "warmup_excluded"=>1, "sample_count"=>3,
            "receipt"=>get(selected_row, "receipt", Dict{String,Any}()),
            "objective_tolerance"=>get(selected_row, "objective_tolerance", ""),
            "allocation_bytes"=>Int.(get(selected_row, "allocation_bytes", [0]))))
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
