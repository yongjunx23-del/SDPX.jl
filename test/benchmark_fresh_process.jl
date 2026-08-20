using Test
using TOML

if !isdefined(Main, :FreshProcessCampaign)
    include(joinpath(@__DIR__, "..", "benchmark", "fresh_process_campaign.jl"))
end
using .FreshProcessCampaign

function _fresh_fixture_row(; seconds="1.0", iterations="8",
                            objective="6.0", status="optimal",
                            semantic_pass=true, certificate_valid=true,
                            fingerprint="fixture-fingerprint", route="sdp_native")
    return Dict{String,Any}(
        "suite" => "micro",
        "problem_id" => "synthetic/lp_box",
        "name" => "fixture",
        "family" => "lp",
        "problem_type" => "linear_program",
        "source" => "synthetic",
        "arithmetic" => "float64",
        "precision_bits" => "53",
        "requested_provider" => "auto",
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
        "external_checksum" => "",
        "julia_version" => "1.12.6",
        "os" => "macos",
        "cpu_name" => "fixture-cpu",
        "julia_threads" => "1",
        "blas_threads" => "1",
        "project_sha256" => "project",
        "manifest_sha256" => "manifest",
        "benchmark_driver_sha256" => "driver",
        "solver_source_sha256" => "solver-source",
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
        "workspace_bytes" => "3000",
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
                  arithmetic="float64", provider="auto"),
    )
    selection_result = only(selection_document["result"])
    @test !selection_document["campaign"]["aggregation_valid"]
    @test occursin("selection:problem_id", selection_result["failure_reasons"])

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
end
