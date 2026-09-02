using Test
using TOML
using SDPX

if !isdefined(Main, :V2FreshProcessProfile)
    include(joinpath(@__DIR__, "v2_fresh_process_profile.jl"))
end
using .V2FreshProcessProfile
using .GeneralBenchmarkV2
using .ProfileCatalog
using .V2Schema9Adapter

function _fresh_fixture(; pid=100, objective="-5.0",
                         status="optimal", core=true)
    route = Dict{String,Any}(
        "requested_route" => "bordered", "planned_route" => "bordered",
        "executed_route" => "bordered", "requested_formulation" => "auto",
        "planned_formulation" => "symmetric_augmented_hsd_core",
        "executed_formulation" => "symmetric_augmented_hsd_core",
        "requested_backend" => "native_hsd", "planned_backend" => "native_hsd",
        "executed_backend" => "native_hsd", "requested_provider" => "cholmod",
        "planned_provider" => "cholmod", "executed_provider" => "cholmod",
        "requested_kernel" => "symmetric_ldl", "planned_kernel" => "symmetric_ldl",
        "executed_kernel" => "symmetric_ldl", "reuse" => "none")
    r = Dict{String,Any}(
        "protocol_version" => 1, "pid" => pid,
        "source_commit" => "a"^40, "tree_fingerprint" => "b"^40,
        "catalog" => "general_v2_lp_tranche", "family" => "lp",
        "instance" => "v2_lp_box_small",
        "case_key" => "general_v2_lp_tranche|lp|small|v2_lp_box_small|Float64",
        "input_fingerprint" => "c"^64, "execution_fingerprint" => "3"^64,
        "catalog_artifact_sha256" => "d"^64,
        "project_sha256" => "e"^64, "manifest_sha256" => "f"^64,
        "benchmark_driver_sha256" => "4"^64,
        "solver_source_sha256" => "5"^64,
        "harness_source_sha256" => "6"^64,
        "schema_source_sha256" => "7"^64,
        "contract_fingerprint" => "8"^64,
        "campaign_id" => "9"^64, "shard_id" => "local",
        "shard_index" => 1, "shard_count" => 1,
        "environment" => Dict{String,Any}("cpu"=>"fixture", "julia_threads"=>4,
            "gc_threads"=>1, "blas_threads"=>1, "omp_threads"=>1,
            "provider"=>"cholmod"),
        "environment_fingerprint" => "1"^64, "provider_fingerprint" => "2"^64,
        "provider" => "cholmod", "provider_version" => "0.6.0",
        "precision_name" => "Float64", "precision_bits" => 53,
        "solver_tolerance" => "1e-8", "certificate_limit" => "5e-7",
        "route_receipt" => route,
        "status" => status, "certificate_valid" => status == "optimal",
        "validation_certificate" => status == "optimal",
        "validation_reference" => status == "optimal",
        "semantic_pass" => status == "optimal",
        "validation_failures" => status == "optimal" ? String[] : ["status"],
        "certificate_kind" => "optimal", "certificate_failures" => String[],
        "objective" => objective, "dual_objective" => objective,
        "primal_residual" => "0.0", "dual_residual" => "0.0",
        "relative_gap" => "0.0",
        "certificate_metrics" => Dict{String,Any}("primal_affine"=>0.0,
            "primal_cone"=>0.0, "dual_affine"=>0.0, "dual_cone"=>0.0,
            "relative_gap"=>0.0, "relative_complementarity"=>0.0),
        "objective_interval" => Dict("lower" => "-5.0", "upper" => "-5.0"),
        "reference_objective" => "-5.0", "iterations" => 9,
        "total_seconds" => 0.03, "maxrss_bytes" => 1000,
        "allocated_bytes" => 100,
    )
    core && (r["core_seconds"] = 0.001)
    r
end

@testset "fresh-process aggregator and protocol negatives" begin
    catalog = GeneralBenchmarkV2.lp_tranche_catalog()
    instance = only(filter(x -> x.id === :v2_lp_box_small, catalog.instances))
    precision = only(filter(x -> x.name === :Float64,
        GeneralBenchmarkV2.reviewed_precision_specs()))
    warm = _fresh_fixture(pid=1)
    measured = [_fresh_fixture(pid=i) for i in 2:4]
    mktempdir() do dir
        warmup_path = joinpath(dir, "warmup.toml")
        child_paths = [joinpath(dir, "sample_$i.toml") for i in 1:3]
        write(warmup_path, "warmup")
        foreach(path -> write(path, basename(path)), child_paths)
        warmup_hash = V2FreshProcessProfile._child_hash(warmup_path)
        child_hashes = V2FreshProcessProfile._child_hash.(child_paths)
        aggregate(rows) = V2FreshProcessProfile.aggregate_child_receipts(
            warm, rows, catalog, instance, precision;
            child_paths, child_hashes, warmup_path, warmup_hash)

        row = aggregate(measured)
        @test row.receipt["fresh_process"] === true
        @test row.receipt["warmup_excluded"] == 1
        @test row.receipt["sample_count"] == 3
        @test row.receipt["sample_pids"] == [2, 3, 4]
        @test row.peak_rss_bytes == 1000
        @test row.sample_core_seconds == [0.001, 0.001, 0.001]
        @test row.receipt["child_artifact_sha256"] == child_hashes
        @test row.receipt["warmup_artifact_sha256"] == warmup_hash
        @test row.receipt["sample_certificate_metrics"][1]["dual_cone"] == 0.0
        @test ProfileCatalog.validate_profile_row(row; live=true)
        @test row.receipt["process_isolation"] == "fresh_process_three_sample"
        @test V2Schema9Adapter.schema9_row(row).execution_mode == "profile"
        @test V2Schema9Adapter.schema9_row(row).process_peak_rss_bytes == 1000
        partial_core = [_fresh_fixture(pid=2, core=false),
            _fresh_fixture(pid=3), _fresh_fixture(pid=4)]
        partial_row = aggregate(partial_core)
        @test partial_row.sample_core_seconds[1] === nothing
        @test V2Schema9Adapter.schema9_row(partial_row).core_seconds === missing

        @test_throws ArgumentError aggregate(
            [_fresh_fixture(pid=2), _fresh_fixture(pid=2), _fresh_fixture(pid=4)])
        bad_objective = [_fresh_fixture(pid=2),
            _fresh_fixture(pid=3, objective="-4.0"), _fresh_fixture(pid=4)]
        @test_throws ArgumentError aggregate(bad_objective)
        @test_throws ArgumentError aggregate([_fresh_fixture(pid=2,
            status="numerical_breakdown"), _fresh_fixture(pid=3), _fresh_fixture(pid=4)])
        missing_identity = _fresh_fixture(pid=2)
        delete!(missing_identity, "tree_fingerprint")
        @test_throws ArgumentError aggregate(
            [missing_identity, _fresh_fixture(pid=3), _fresh_fixture(pid=4)])
        bad_hashes = copy(child_hashes); bad_hashes[1] = "0"^64
        @test_throws ArgumentError V2FreshProcessProfile.aggregate_child_receipts(
            warm, measured, catalog, instance, precision;
            child_paths, child_hashes=bad_hashes, warmup_path, warmup_hash)
    end
    dirty_probe = joinpath(V2FreshProcessProfile.ROOT,
        ".stageb_fresh_dirty_probe_$(getpid())")
    try
        write(dirty_probe, "untracked mutation")
        @test_throws ArgumentError V2FreshProcessProfile._require_clean_source(
            "test_mutation")
    finally
        rm(dirty_probe; force=true)
    end
    @test V2FreshProcessProfile._require_clean_source("test_clean")
    mktempdir() do dir
        artifact = joinpath(dir, "receipt.toml")
        V2FreshProcessProfile._atomic_toml(artifact, Dict{String,Any}("ok"=>true))
        @test TOML.parsefile(artifact)["ok"] === true
        @test_throws ArgumentError V2FreshProcessProfile._atomic_toml(
            artifact, Dict{String,Any}("ok"=>false))
        race = joinpath(dir, "race.bin")
        contenders = [Threads.@spawn(try
            V2FreshProcessProfile._atomic_bytes(race, Vector{UInt8}(codeunits(value)))
            true
        catch
            false
        end) for value in ("first", "second")]
        @test count(fetch, contenders) == 1
        @test String(read(race)) in ("first", "second")
        if !Sys.iswindows()
            link = joinpath(dir, "source-link")
            symlink(V2FreshProcessProfile.ROOT, link)
            @test_throws ArgumentError V2FreshProcessProfile._canonical_destination(
                joinpath(link, ".ignored-stageb", "escape.toml"))
            dangling = joinpath(dir, "dangling.toml")
            symlink(joinpath(dir, "does-not-exist"), dangling)
            @test_throws ArgumentError V2FreshProcessProfile._canonical_destination(dangling)
        end

        bundle = mktempdir(dir; prefix="bundle.", cleanup=false)
        mkpath(joinpath(bundle, "children"))
        fixed = Dict("receipt.toml"=>"receipt", "schema.tsv"=>"tsv",
            "schema.toml"=>"schema")
        for (name, value) in fixed
            write(joinpath(bundle, name), value)
        end
        write(joinpath(bundle, "children", "warmup.toml"), "warmup")
        for i in 1:3
            write(joinpath(bundle, "children", "sample_$i.toml"), "sample$i")
        end
        completion = Dict{String,Any}("completion_protocol"=>1, "complete"=>true,
            "bundle"=>bundle, "source_commit"=>"a"^40,
            "tree_fingerprint"=>"b"^40, "case_key"=>"fixture",
            "files"=>Dict(
                "receipt_sha256"=>V2FreshProcessProfile._child_hash(joinpath(bundle,"receipt.toml")),
                "schema_tsv_sha256"=>V2FreshProcessProfile._child_hash(joinpath(bundle,"schema.tsv")),
                "schema_toml_sha256"=>V2FreshProcessProfile._child_hash(joinpath(bundle,"schema.toml")),
                "warmup_sha256"=>V2FreshProcessProfile._child_hash(joinpath(bundle,"children","warmup.toml")),
                "sample_sha256"=>[V2FreshProcessProfile._child_hash(joinpath(bundle,"children","sample_$i.toml")) for i in 1:3]))
        marker = joinpath(dir, "bundle.complete.toml")
        V2FreshProcessProfile._atomic_toml(marker, completion)
        @test V2FreshProcessProfile._validate_completion(marker)["complete"] === true
        write(joinpath(bundle, "children", "sample_2.toml"), "tampered")
        @test_throws ArgumentError V2FreshProcessProfile._validate_completion(marker)
        rm(bundle; recursive=true, force=true)
    end
end
