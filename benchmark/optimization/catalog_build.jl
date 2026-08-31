#!/usr/bin/env julia
# Build/validate every registered benchmark catalog without solving build-only
# physics artifacts. Optional external CFT lowering is represented explicitly;
# it is never silently treated as a successful required catalog.
using SHA
using TOML
using SDPX

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "profile_catalog.jl"))
using .ProfileCatalog

_sha(path) = bytes2hex(SHA.sha256(read(path)))
_optional_cft_required() = get(ENV, "SDPX_REQUIRE_OPTIONAL_CFT", "0") == "1"

function _unavailable_row(catalog_name, path, problem_id, family, reason)
    return Dict{String,Any}(
        "case_key" => join((catalog_name, String(family), String(problem_id), "Float64"), "|"),
        "catalog" => catalog_name, "catalog_version" => 1,
        "problem_id" => String(problem_id), "family" => String(family),
        "arithmetic" => "Float64", "build_status" => "optional_unavailable",
        "catalog_validation_pass" => false, "solve_eligible" => false,
        "optional_unavailable" => true, "reference_status" => "build_only",
        "source" => path, "input_fingerprint" => "",
        "build_seconds" => 0.0, "failure_taxonomy" => reason,
        "transform_exactness" => "catalog")
end

function _build_physics()
    harness_path = joinpath(ROOT, "benchmark", "bootstrap", "PhysicsBenchmarkHarness.jl")
    isfile(harness_path) || return Dict{String,Any}[]
    if !isdefined(Main, :PhysicsBenchmarkHarness)
        Base.include(Main, harness_path)
    end
    H = Main.PhysicsBenchmarkHarness
    root = joinpath(ROOT, "benchmark", "bootstrap", "physics")
    rows = Dict{String,Any}[]
    for dir in sort!(filter(isdir, readdir(root; join=true)))
        path = joinpath(dir, "catalog.jl")
        isfile(path) || continue
        catalog_name = basename(dir)
        try
            catalog = Base.invokelatest(H.load_catalog, path)
            catalog_name = String(catalog.name)
            for id in sort!(collect(keys(catalog.specs)))
                spec = Base.invokelatest(H.catalog_spec, catalog, id)
                entries = [e for es in values(catalog.suites) for e in es if e.problem_id == id]
                isempty(entries) && continue
                entry = first(sort!(entries; by=e -> (String(e.arithmetic), String(e.provider))))
                started = time_ns()
                try
                    built = Base.invokelatest(H.build_problem, catalog, spec, Float64)
                    failures = Base.invokelatest(H.validate_result, catalog, spec, built, nothing, (;))
                    push!(rows, Dict{String,Any}(
                        "case_key" => join((catalog_name, String(spec.family), id, String(entry.arithmetic)), "|"),
                        "catalog" => catalog_name, "catalog_version" => catalog.version,
                        "problem_id" => id, "family" => String(spec.family),
                        "arithmetic" => String(entry.arithmetic),
                        "build_status" => isempty(failures) ? "pass" : "failed",
                        "catalog_validation_pass" => isempty(failures), "solve_eligible" => false,
                        "optional_unavailable" => false, "reference_status" => String(spec.reference.status),
                        "source" => String(path), "input_fingerprint" => spec.fingerprint,
                        "build_seconds" => (time_ns() - started) * 1e-9,
                        "failure_taxonomy" => join(string.(failures), ","), "transform_exactness" => "catalog"))
                catch err
                    optional = catalog_name == "cft"
                    push!(rows, optional ? _unavailable_row(catalog_name, path, id, spec.family,
                        "optional_dependency_unavailable:" * string(nameof(typeof(err)))) :
                        Dict{String,Any}(
                            "case_key" => join((catalog_name, String(spec.family), id, String(entry.arithmetic)), "|"),
                            "catalog" => catalog_name, "catalog_version" => catalog.version,
                            "problem_id" => id, "family" => String(spec.family),
                            "arithmetic" => String(entry.arithmetic), "build_status" => "failed",
                            "catalog_validation_pass" => false, "solve_eligible" => false,
                            "optional_unavailable" => false, "reference_status" => String(spec.reference.status),
                            "source" => String(path), "input_fingerprint" => spec.fingerprint,
                            "build_seconds" => (time_ns() - started) * 1e-9,
                            "failure_taxonomy" => string(nameof(typeof(err))), "transform_exactness" => "catalog"))
                end
            end
        catch err
            if catalog_name == "cft"
                push!(rows, _unavailable_row(catalog_name, path, "catalog", :polynomial_matrix_program,
                    "optional_dependency_unavailable:" * string(nameof(typeof(err)))))
            else
                rethrow()
            end
        end
    end
    return rows
end

function main()
    out = get(ENV, "SDPX_CATALOG_OUTPUT", joinpath(pwd(), "catalog-manifest-v2.toml"))
    commit = get(ENV, "GITHUB_SHA", "local")
    rows = Dict{String,Any}[]
    for case in ProfileCatalog.enumerate_cases(; include_physics=false)
        status, failure = "pass", ""
        try
            g, spec = case.payload.mod, case.payload.spec
            Base.invokelatest(g.build, spec.problem, Float64, spec.params)
        catch err
            status, failure = "failed", string(nameof(typeof(err)))
        end
        push!(rows, Dict{String,Any}(
            "case_key" => case.key, "catalog" => String(case.catalog),
            "catalog_version" => 1, "problem_id" => String(case.id),
            "family" => String(case.family), "tier" => String(case.tier),
            "arithmetic" => String(case.arithmetic), "source" => case.source,
            "reference_status" => String(case.reference_status),
            "solve_eligible" => case.solve_eligible, "build_status" => status,
            "catalog_validation_pass" => status == "pass", "optional_unavailable" => false,
            "input_fingerprint" => case.transform.fingerprint,
            "transform_exactness" => case.transform.exactness,
            "transform_fingerprint" => case.transform.fingerprint,
            "failure_taxonomy" => failure))
    end
    append!(rows, _build_physics())
    sort!(rows; by=r -> r["case_key"])
    keys = [r["case_key"] for r in rows]
    length(unique(keys)) == length(keys) || error("duplicate catalog case key")
    doc = Dict("manifest_schema"=>2, "source_commit"=>commit,
        "catalog_protocol_version"=>2, "catalog_source_sha256"=>_sha(@__FILE__),
        "catalog_artifact_sha256"=>_sha(joinpath(ROOT, "benchmark", "optimization", "profile_catalog.jl")),
        "require_optional_cft"=>_optional_cft_required(), "case"=>rows)
    open(out, "w") do io; TOML.print(io, doc; sorted=true); end
    failed = any(r -> r["build_status"] == "failed", rows)
    optional_failed = any(r -> r["build_status"] == "optional_unavailable", rows)
    failed && error("required catalog build/validation failed")
    _optional_cft_required() && optional_failed && error("required optional CFT dependency unavailable")
    println("CATALOG_GATE_PASS cases=", length(rows), " optional_unavailable=", optional_failed,
        " output=", out)
end
main()
