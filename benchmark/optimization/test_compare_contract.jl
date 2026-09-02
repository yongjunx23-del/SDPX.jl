using Test
using TOML
using Statistics
include(joinpath(@__DIR__, "..", "bootstrap", "compare_impl.jl"))

function _v9_row(source)
    route = join(("solve", "catalog_contract", "catalog_contract", "auto", "dense", "sdp", "sdp",
        "sdp", "provider", "provider", "auto", "auto", "generic", "false", "none", "none"), "|")
    row = Dict{String,Any}(
        "schema_version"=>9, "problem_id"=>"fixture-v9", "arithmetic"=>"Float64", "requested_provider"=>"auto",
        "input_fingerprint"=>"a"^64, "catalog_validation_pass"=>true,
        "source_dirty"=>false, "fallback_reason"=>"none", "la_fallback_reason"=>"none", "shard_index"=>1, "shard_count"=>1,
        "execution_mode"=>"solve", "requested_engine"=>"catalog_contract", "executed_engine"=>"catalog_contract",
        "scaling"=>"auto", "layout"=>"dense", "conic_formulation"=>"sdp", "planned_formulation"=>"sdp",
        "executed_formulation"=>"sdp", "planned_backend"=>"provider", "executed_backend"=>"provider",
        "planned_provider"=>"auto", "executed_provider"=>"auto", "executed_specialization"=>"generic",
        "psd_lift_used"=>"false", "fallback_reason"=>"none", "la_fallback_reason"=>"none", "status"=>"optimal",
        "semantic_pass"=>true, "certificate_valid"=>true, "objective"=>"1.0", "reference_objective"=>"1.0",
        "reference_absolute_tolerance"=>"1e-8", "reference_relative_tolerance"=>"1e-8", "iterations"=>7,
        "source_commit"=>source, "source_dirty"=>false, "sample_count"=>3, "sample_seconds"=>"[1.0,1.1,1.0]",
        "sample_status"=>"[\"optimal\",\"optimal\",\"optimal\"]", "sample_semantic_pass"=>"[true,true,true]",
        "sample_certificate_valid"=>"[true,true,true]", "sample_iterations"=>"[7,7,7]", "sample_objective"=>"[1.0,1.0,1.0]",
        "sample_route"=>"[\"$route\",\"$route\",\"$route\"]", "sample_semantic_parity"=>true,
        "sample_parity_failures"=>"", "total_seconds"=>1.0, "sample_median_seconds"=>1.0,
        "sample_min_seconds"=>1.0, "sample_max_seconds"=>1.1, "sample_mad_seconds"=>0.0,
        "sample_spread_seconds"=>0.1, "total_seconds_iqr"=>0.05)
    for field in ("catalog_name", "catalog_version", "source", "suite", "name", "problem_type", "purpose", "seed", "julia_version", "os", "cpu_name", "mfla_commit", "bfla_commit", "benchmark_scale", "input_generation_precision_bits", "original_equalities", "source_parameters", "objective_interval_lower", "objective_interval_upper", "shard_id", "shard_index", "shard_count", "campaign_id", "external_checksum", "project_sha256", "manifest_sha256", "benchmark_driver_sha256", "solver_source_sha256", "catalog_source_sha256", "harness_source_sha256", "schema_source_sha256", "contract_fingerprint", "route_receipt", "pbs_job_id", "pbs_array_index", "pbs_queue", "pbs_node")
        row[field] = field == "route_receipt" ? "complete" : (field in ("shard_index", "shard_count") ? 1 : (field in ("objective_interval_lower", "objective_interval_upper") ? "1.0" : (field == "source_parameters" ? "fixture" : (field == "original_equalities" ? 0 : "a"^64))))
    end
    row["shard_index"] = 1; row["shard_count"] = 1
    row["route_receipt"] = Dict("requested_route"=>"auto", "planned_route"=>"catalog_contract", "executed_route"=>"catalog_contract")
    return row
end

@testset "schema-v9 pair contract" begin
    @test Main._CURRENT_RESULT_SCHEMA_VERSION == 9
    @test Main._schema_status(9) === :current
    @test Main._schema_status(8) === :legacy
    baseline = _v9_row("a"^40); candidate = _v9_row("b"^40)
    before = Main._sample_validation(baseline, "baseline"); after = Main._sample_validation(candidate, "candidate")
    @test before.valid; @test after.valid
    @test Main._sample_semantic_match(before, after, baseline, candidate)
    p1, p2, pout = tempname(), tempname(), tempname()
    for (path, row) in ((p1, baseline), (p2, candidate))
        open(path, "w") do io; TOML.print(io, Dict("schema_version"=>9, "result"=>[row])); end
    end
    compare_cli = normpath(joinpath(@__DIR__, "..", "bootstrap", "compare.jl"))
    # The CLI is tested in a disposable environment that explicitly declares
    # Dates (used by PhysicsBenchmarkHarness) and develops this exact source;
    # the package test environment intentionally does not need benchmark-only
    # stdlib dependencies.
    compare_project = mktempdir()
    write(joinpath(compare_project, "Project.toml"),
        "[deps]\nDates = \"ade2ca70-3891-5945-98fb-dc099432e06a\"\nPkg = \"44cfe95a-1eb2-52ea-b672-e2afdf69b78f\"\n")
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    withenv("SDPX_ROOT" => repo_root) do
        setup = `$(Base.julia_cmd()) --startup-file=no --project=$compare_project -e 'using Pkg; Pkg.develop(path=ENV["SDPX_ROOT"]); Pkg.instantiate()'`
        @test success(run(setup))
    end
    cli = `$(Base.julia_cmd()) --startup-file=no --project=$compare_project $compare_cli $p1 $p2 $pout`
    @test success(run(pipeline(cli, stdout=stdout, stderr=stderr)))
    @test isfile(pout)
    bad = tempname()
    open(bad, "w") do io; TOML.print(io, Dict("schema_version"=>8, "result"=>[baseline])); end
    @test !success(`$(Base.julia_cmd()) --startup-file=no --project=$compare_project $compare_cli $bad $p2`)
    rm.((p1, p2, pout, bad); force=true)
    rm(compare_project; force=true, recursive=true)
end

@testset "schema, route, identity, and malformed-sample negatives" begin
    good = _v9_row("a"^40)
    native_profile = merge(good, Dict("execution_mode"=>"profile",
        "requested_engine"=>"native_hsd", "executed_engine"=>"native_hsd"))
    @test Main._validate_route(native_profile; label="native").valid
    @test !Main._validate_route(merge(good, Dict("execution_mode"=>"bogus")); label="bad").valid
    @test !Main._validate_route(merge(good, Dict("executed_engine"=>"unknown")); label="bad").valid
    bad_sample = merge(good, Dict("sample_status"=>"[\"optimal\",\"iteration_limit\",\"optimal\"]"))
    @test !Main._sample_validation(bad_sample, "bad").valid
    bad_route = merge(good, Dict("sample_route"=>"[\"bad\",\"bad\",\"bad\"]"))
    @test !Main._sample_validation(bad_route, "bad").valid
    @test !Main._valid_solver_source_sha256("not-a-sha")
    @test Main._valid_solver_source_sha256("a"^64)
    p1, p2 = tempname(), tempname()
    for p in (p1, p2)
        open(p, "w") do io; TOML.print(io, Dict("schema_version"=>8, "result"=>Any[])); end
    end
    @test_throws ArgumentError Main.compare_result_files(p1, p2)
    rm(p1; force=true); rm(p2; force=true)
end
