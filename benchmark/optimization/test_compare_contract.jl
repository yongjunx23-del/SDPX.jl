using Test
using TOML
using Statistics
include(joinpath(@__DIR__, "..", "bootstrap", "compare_impl.jl"))

function _v9_row(source)
    route = join(("solve", "catalog_contract", "catalog_contract", "auto", "dense", "sdp", "sdp",
        "sdp", "provider", "provider", "auto", "auto", "generic", "false",
        "none", "none"), "|")
    return Dict{String,Any}(
        "schema_version" => 9, "problem_id" => "fixture-v9",
        "arithmetic" => "Float64", "requested_provider" => "auto",
        "execution_mode" => "solve", "requested_engine" => "catalog_contract",
        "executed_engine" => "catalog_contract", "scaling" => "auto", "layout" => "dense",
        "conic_formulation" => "sdp", "planned_formulation" => "sdp",
        "executed_formulation" => "sdp", "planned_backend" => "provider",
        "executed_backend" => "provider", "planned_provider" => "auto",
        "executed_provider" => "auto", "executed_specialization" => "generic",
        "psd_lift_used" => "false", "fallback_reason" => "none", "la_fallback_reason" => "none",
        "status" => "optimal", "semantic_pass" => true, "certificate_valid" => true,
        "objective" => "1.0", "reference_objective" => "1.0",
        "reference_absolute_tolerance" => "1e-8", "reference_relative_tolerance" => "1e-8",
        "iterations" => 7, "source_commit" => source, "source_dirty" => false,
        "sample_count" => 3, "sample_seconds" => "[1.0,1.1,1.0]",
        "sample_status" => "[\"optimal\",\"optimal\",\"optimal\"]",
        "sample_semantic_pass" => "[true,true,true]",
        "sample_certificate_valid" => "[true,true,true]",
        "sample_iterations" => "[7,7,7]", "sample_objective" => "[1.0,1.0,1.0]",
        "sample_route" => "[\"$route\",\"$route\",\"$route\"]",
        "sample_semantic_parity" => true, "sample_parity_failures" => "",
        "total_seconds" => 1.0, "sample_median_seconds" => 1.0,
        "sample_min_seconds" => 1.0, "sample_max_seconds" => 1.1,
        "sample_mad_seconds" => 0.0, "sample_spread_seconds" => 0.1,
        "total_seconds_iqr" => 0.05,
    )
end

@testset "schema-v9 pair contract" begin
    @test Main._CURRENT_RESULT_SCHEMA_VERSION == 9
    @test Main._schema_status(9) === :current
    @test Main._schema_status(8) === :legacy
    baseline = _v9_row("a"^40)
    candidate = _v9_row("b"^40)
    before = Main._sample_validation(baseline, "baseline")
    after = Main._sample_validation(candidate, "candidate")
    @test before.valid
    @test after.valid
    @test Main._sample_semantic_match(before, after, baseline, candidate)
end
