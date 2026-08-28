using Test
using TOML

if !isdefined(Main, :FreshProcessCampaign)
    include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "fresh_process_campaign.jl"))
end
using .FreshProcessCampaign

function _fresh_fixture_row(; seconds="1.0", iterations="8",
                            objective="6.0", status="optimal",
                            semantic_pass=true, certificate_valid=true,
                            fingerprint="fixture-fingerprint", route="sdp_native")
    return Dict{String,Any}(
        "schema_version" => 8,
        "suite" => "micro",
        "catalog_name" => "fixture",
        "catalog_version" => "1",
        "problem_id" => "synthetic/lp_box",
        "name" => "fixture",
        "family" => "lp",
        "problem_type" => "linear_program",
        "source" => "synthetic",
        "arithmetic" => "float64",
        "precision_bits" => "53",
        "requested_provider" => "auto",
        "execution_mode" => "solve",
        "requested_engine" => "auto",
        "executed_engine" => "sdpx_legacy",
        "campaign_id" => "fixture-campaign",
        "shard_id" => "shard-1-of-1",
        "shard_index" => "1",
        "shard_count" => "1",
        "pbs_job_id" => "",
        "pbs_array_index" => "",
        "pbs_queue" => "",
        "pbs_node" => "fixture-node",
        "scaling" => "tiny",
        "layout" => "lp",
        "reference_status" => "optimal",
        "reference_absolute_tolerance" => "1e-8",
        "reference_relative_tolerance" => "1e-8",
        "conic_formulation" => route,
        "planned_formulation" => "dense_augmented",
        "executed_formulation" => "dense_augmented",
        "planned_backend" => "dense",
        "executed_backend" => "dense",
        "planned_provider" => "blas_lapack",
        "executed_provider" => "blas_lapack",
        "executed_specialization" => "none",
        "psd_lift_used" => false,
        "fallback_reason" => "none",
        "la_fallback_reason" => "none",
        "input_fingerprint" => fingerprint,
        "external_checksum" => "fixture-checksum",
        "julia_version" => "1.12.6",
        "os" => "macos",
        "cpu_name" => "fixture-cpu",
        "julia_threads" => "1",
        "blas_threads" => "1",
        "project_sha256" => repeat("1", 64),
        "manifest_sha256" => repeat("2", 64),
        "benchmark_driver_sha256" => repeat("3", 64),
        "solver_source_sha256" => repeat("4", 64),
        "catalog_source_sha256" => repeat("5", 64),
        "harness_source_sha256" => repeat("6", 64),
        "schema_source_sha256" => repeat("7", 64),
        "contract_fingerprint" => repeat("8", 64),
        "mfla_commit" => "",
        "bfla_commit" => "",
        "status" => status,
        "iterations" => iterations,
        "objective" => objective,
        "semantic_pass" => semantic_pass,
        "certificate_valid" => certificate_valid,
        "total_seconds" => seconds,
        "allocated_bytes" => "1000",
        "process_peak_rss_bytes" => "2000",
        "rss_bytes" => "2000",
        "workspace_bytes" => "3000",
        "sample_count" => 1,
    )
end

function _fresh_records(rows; exit_codes=fill(0, length(rows)), failures=fill("", length(rows)))
    return [ChildRecord(
        index,
        "/tmp/child_$(index).toml",
        "/tmp/child_$(index).tsv",
        "/tmp/child_$(index).log",
        exit_codes[index],
        rows[index],
        failures[index],
    ) for index in eachindex(rows)]
end

@testset "fresh process campaign aggregation" begin
    rows = [_fresh_fixture_row(seconds=string(value)) for value in (1.0, 1.2, 0.9)]
    document = aggregate_campaign(
        _fresh_records(rows);
        repetitions=3,
        campaign_dir="/tmp",
    )
    result = only(document["result"])
    @test document["schema_version"] == 1
    @test document["campaign"]["aggregation_valid"]
    @test result["pairing_valid"]
    @test result["timing_valid"]
    @test result["total_seconds_median"] == 1.0
    @test result["total_seconds_min"] == 0.9
    @test result["total_seconds_max"] == 1.2
    @test isapprox(result["total_seconds_mad"], 0.1)
    @test isapprox(result["total_seconds_spread"], 0.3)
    @test result["iterations_median"] == 8.0
    @test result["allocated_bytes_median"] == 1000.0
    @test result["rss_bytes_median"] == 2000.0
    @test result["rss_iqr_bytes"] == 0.0
    @test result["total_seconds_iqr"] > 0

    prefix = repeat("1", 160)
    precise = [_fresh_fixture_row(
        objective="0." * prefix * digit,
    ) for digit in ("5", "5", "6")]
    for row in precise
        row["reference_absolute_tolerance"] = "1e-160"
        row["reference_relative_tolerance"] = "1e-160"
    end
    ambient = precision(BigFloat)
    precise_document = aggregate_campaign(
        _fresh_records(precise);
        repetitions=3,
    )
    @test precise_document["campaign"]["aggregation_valid"]
    @test precision(BigFloat) == ambient

    summary_path = tempname() * ".toml"
    paths = write_summary(document, summary_path)
    @test isfile(paths.toml)
    @test isfile(paths.tsv)
    @test !occursin("generated_at", read(paths.toml, String))
    @test startswith(read(paths.tsv, String), "suite\tproblem_id")
end

@testset "fresh process pairing and fail closed gates" begin
    base = [_fresh_fixture_row() for _ in 1:3]
    mismatch = deepcopy(base)
    mismatch[2]["iterations"] = "9"
    mismatch[3]["input_fingerprint"] = "different"
    mismatch_document = aggregate_campaign(
        _fresh_records(mismatch);
        repetitions=3,
    )
    mismatch_result = only(mismatch_document["result"])
    @test !mismatch_document["campaign"]["aggregation_valid"]
    @test !mismatch_result["pairing_valid"]
    @test occursin("iterations", mismatch_result["failure_reasons"])
    @test occursin("input_fingerprint", mismatch_result["failure_reasons"])

    selection_document = aggregate_campaign(
        _fresh_records(base);
        repetitions=3,
        expected=(suite="micro", problem_id="synthetic/not_selected",
                  arithmetic="float64", provider="auto",
                  campaign_id="fixture-campaign", shard_id="shard-1-of-1",
                  shard_index=1, shard_count=1),
    )
    selection_result = only(selection_document["result"])
    @test !selection_document["campaign"]["aggregation_valid"]
    @test occursin("selection:problem_id", selection_result["failure_reasons"])

    wrong_identity = aggregate_campaign(
        _fresh_records(base);
        repetitions=3,
        expected=(suite="micro", problem_id="synthetic/lp_box",
                  arithmetic="float64", provider="auto",
                  campaign_id="other-campaign", shard_id="other-shard",
                  shard_index=2, shard_count=2),
    )
    @test !wrong_identity["campaign"]["aggregation_valid"]
    @test occursin("selection:campaign_id", only(wrong_identity["result"])["failure_reasons"])
    @test occursin("selection:shard_index", only(wrong_identity["result"])["failure_reasons"])

    failed = [_fresh_fixture_row() for _ in 1:3]
    failed[2]["semantic_pass"] = false
    failed[3]["certificate_valid"] = false
    failed_document = aggregate_campaign(
        _fresh_records(failed; exit_codes=[0, 17, 0]);
        repetitions=3,
    )
    failed_result = only(failed_document["result"])
    @test !failed_document["campaign"]["aggregation_valid"]
    @test occursin("exit_17", failed_result["failure_reasons"])
    @test occursin("semantic_pass", failed_result["failure_reasons"])
    @test occursin("certificate_valid", failed_result["failure_reasons"])

    diagnostic_document = aggregate_campaign(
        _fresh_records(failed; exit_codes=[0, 17, 0]);
        repetitions=3,
        diagnostic=true,
    )
    diagnostic_result = only(diagnostic_document["result"])
    @test diagnostic_document["campaign"]["aggregation_mode"] == "diagnostic"
    @test !diagnostic_document["campaign"]["aggregation_valid"]
    @test diagnostic_result["timing_valid"]

    missing_timing = [_fresh_fixture_row() for _ in 1:3]
    missing_timing[2]["total_seconds"] = ""
    missing_document = aggregate_campaign(
        _fresh_records(missing_timing);
        repetitions=3,
    )
    missing_result = only(missing_document["result"])
    @test !missing_result["timing_valid"]
    @test missing_result["total_seconds_median"] == ""
    @test occursin("timing_missing_or_invalid", missing_result["failure_reasons"])

    @test_throws ArgumentError aggregate_campaign(
        _fresh_records(base);
        repetitions=2,
    )

    invalid_contract = deepcopy(base)
    invalid_contract[2]["contract_fingerprint"] = "not-a-sha"
    invalid_contract_document = aggregate_campaign(
        _fresh_records(invalid_contract); repetitions=3,
    )
    @test !invalid_contract_document["campaign"]["aggregation_valid"]
    @test occursin(
        "contract_fingerprint_invalid",
        only(invalid_contract_document["result"])["failure_reasons"],
    )

    missing_route = deepcopy(base)
    missing_route[1]["executed_engine"] = ""
    missing_route_document = aggregate_campaign(
        _fresh_records(missing_route); repetitions=3,
    )
    @test !missing_route_document["campaign"]["aggregation_valid"]
    @test occursin(
        "executed_engine_missing",
        only(missing_route_document["result"])["failure_reasons"],
    )

    invalid_source = deepcopy(base)
    invalid_source[2]["project_sha256"] = "not-a-sha"
    invalid_source_document = aggregate_campaign(
        _fresh_records(invalid_source); repetitions=3,
    )
    @test !invalid_source_document["campaign"]["aggregation_valid"]
    @test occursin(
        "project_sha256_invalid",
        only(invalid_source_document["result"])["failure_reasons"],
    )

    wrong_schema = deepcopy(base)
    wrong_schema[1]["schema_version"] = 7
    wrong_schema_document = aggregate_campaign(
        _fresh_records(wrong_schema); repetitions=3,
    )
    @test !wrong_schema_document["campaign"]["aggregation_valid"]
    @test occursin("schema_version", only(wrong_schema_document["result"])["failure_reasons"])

    wrong_samples = deepcopy(base)
    wrong_samples[3]["sample_count"] = 3
    wrong_samples_document = aggregate_campaign(
        _fresh_records(wrong_samples); repetitions=3,
    )
    @test !wrong_samples_document["campaign"]["aggregation_valid"]
    @test occursin("sample_count", only(wrong_samples_document["result"])["failure_reasons"])

    canonical_shards = deepcopy(base)
    canonical_shards[2]["shard_index"] = "01"
    canonical_shards[2]["shard_count"] = "001"
    canonical_document = aggregate_campaign(
        _fresh_records(canonical_shards); repetitions=3,
    )
    @test canonical_document["campaign"]["aggregation_valid"]
end

@testset "build campaign aggregation uses catalog validation" begin
    rows = [_fresh_fixture_row(seconds=string(value)) for value in (1.0, 1.2, 0.9)]
    for row in rows
        row["execution_mode"] = "build"
        row["requested_engine"] = "auto"
        row["executed_engine"] = "none"
        row["objective"] = ""
        row["certificate_valid"] = ""
        row["catalog_validation_pass"] = true
    end
    document = aggregate_campaign(
        _fresh_records(rows);
        repetitions=3,
        expected=(suite="micro", problem_id="synthetic/lp_box",
                  arithmetic="float64", provider="auto",
                  execution_mode="build", requested_engine="auto"),
    )
    result = only(document["result"])
    @test document["campaign"]["aggregation_valid"]
    @test result["execution_mode"] == "build"
    @test result["certificate_valid"] == ""
    @test result["semantic_pass"]
end

@testset "fresh aggregation validates canonical route matrix and timing" begin
    aliases = [_fresh_fixture_row() for _ in 1:3]
    for row in aliases
        row["requested_engine"] = "sdpx_legacy"
    end
    alias_document = aggregate_campaign(
        _fresh_records(aliases); repetitions=3, campaign_dir="/tmp",
        expected=(suite="micro", problem_id="synthetic/lp_box",
                  arithmetic="float64", provider="auto",
                  execution_mode="solve", requested_engine="legacy"),
    )
    @test alias_document["campaign"]["aggregation_valid"]

    invalid = [_fresh_fixture_row() for _ in 1:3]
    invalid[2]["requested_engine"] = "native_hsd"
    invalid_document = aggregate_campaign(
        _fresh_records(invalid); repetitions=3, campaign_dir="/tmp",
    )
    @test !invalid_document["campaign"]["aggregation_valid"]
    @test occursin(
        "native_requested_engine_invalid",
        only(invalid_document["result"])["failure_reasons"],
    )

    invalid = [_fresh_fixture_row() for _ in 1:3]
    invalid[1]["executed_engine"] = "none"
    invalid_document = aggregate_campaign(
        _fresh_records(invalid); repetitions=3, campaign_dir="/tmp",
    )
    @test !invalid_document["campaign"]["aggregation_valid"]
    @test occursin(
        "solve_executed_engine_invalid",
        only(invalid_document["result"])["failure_reasons"],
    )

    invalid = [_fresh_fixture_row() for _ in 1:3]
    invalid[3]["total_seconds"] = "Inf"
    invalid[2]["setup_seconds"] = "-0.1"
    invalid_document = aggregate_campaign(
        _fresh_records(invalid); repetitions=3, campaign_dir="/tmp",
    )
    @test !invalid_document["campaign"]["aggregation_valid"]
    failures = only(invalid_document["result"])["failure_reasons"]
    @test occursin("total_seconds_invalid", failures)
    @test occursin("setup_seconds_invalid", failures)
end
