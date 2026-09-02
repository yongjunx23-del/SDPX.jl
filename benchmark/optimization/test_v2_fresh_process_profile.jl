using Test
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
        "input_fingerprint" => "c"^64, "catalog_artifact_sha256" => "d"^64,
        "project_sha256" => "e"^64, "manifest_sha256" => "f"^64,
        "environment_fingerprint" => "1"^64, "provider_fingerprint" => "2"^64,
        "provider" => "cholmod", "precision_name" => "Float64",
        "precision_bits" => 53, "solver_tolerance" => "1e-8",
        "certificate_limit" => "5e-7", "route_receipt" => route,
        "status" => status, "certificate_valid" => status == "optimal",
        "validation_certificate" => status == "optimal",
        "validation_reference" => status == "optimal",
        "semantic_pass" => status == "optimal",
        "validation_failures" => status == "optimal" ? String[] : ["status"],
        "objective" => objective, "objective_interval" => Dict("lower" => "-5.0", "upper" => "-5.0"),
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
    precision = only(filter(x -> x.name === :Float64, GeneralBenchmarkV2.reviewed_precision_specs()))
    warm = _fresh_fixture(pid=1)
    measured = [_fresh_fixture(pid=i) for i in 2:4]
    row = V2FreshProcessProfile.aggregate_child_receipts(warm, measured, catalog, instance, precision;
        child_paths=["s2.toml", "s3.toml", "s4.toml"], child_hashes=["a"^64, "b"^64, "c"^64])
    @test row.receipt["fresh_process"] === true
    @test row.receipt["warmup_excluded"] == 1
    @test row.receipt["sample_count"] == 3
    @test row.receipt["sample_pids"] == [2, 3, 4]
    @test row.peak_rss_bytes == 1000
    @test row.sample_core_seconds == [0.001, 0.001, 0.001]
    @test row.receipt["child_artifact_sha256"] == ["a"^64, "b"^64, "c"^64]
    @test V2Schema9Adapter.schema9_row(row).execution_mode == "fresh_process_three_sample"
    @test V2Schema9Adapter.schema9_row(row).process_peak_rss_bytes == 1000

    @test_throws ArgumentError V2FreshProcessProfile.aggregate_child_receipts(warm,
        [_fresh_fixture(pid=2), _fresh_fixture(pid=2), _fresh_fixture(pid=4)],
        catalog, instance, precision)
    bad_objective = [_fresh_fixture(pid=2), _fresh_fixture(pid=3, objective="-4.0"), _fresh_fixture(pid=4)]
    @test_throws ArgumentError V2FreshProcessProfile.aggregate_child_receipts(warm, bad_objective,
        catalog, instance, precision)
    @test_throws ArgumentError V2FreshProcessProfile.aggregate_child_receipts(warm,
        [_fresh_fixture(pid=2, status="numerical_breakdown"), _fresh_fixture(pid=3), _fresh_fixture(pid=4)],
        catalog, instance, precision)
    missing = _fresh_fixture(pid=2); delete!(missing, "tree_fingerprint")
    @test_throws ArgumentError V2FreshProcessProfile.aggregate_child_receipts(warm, [missing, _fresh_fixture(pid=3), _fresh_fixture(pid=4)],
        catalog, instance, precision)
end
