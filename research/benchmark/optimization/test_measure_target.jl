using Test
include(joinpath(@__DIR__, "measure_target.jl"))

@testset "measure target receipt and trajectory negatives" begin
    @test_throws ArgumentError Main._require_receipt(Dict{String,Any}())
    route_keys = ("requested_route", "planned_route", "executed_route", "requested_formulation", "planned_formulation", "executed_formulation", "requested_backend", "planned_backend", "executed_backend", "requested_provider", "planned_provider", "executed_provider", "requested_kernel", "planned_kernel", "executed_kernel", "reuse")
    valid = Dict{String,Any}(key => "value" for key in ("source_commit", "tree_fingerprint", "catalog", "family", "instance", "input_fingerprint", "project_sha256", "manifest_sha256", "catalog_run_id", "catalog_artifact_sha256", "environment_fingerprint", "provider_fingerprint", "provider_version", "cpu", "julia_threads", "blas_threads", "omp_threads", "gc_threads", "actual_objective", "requested_route", "planned_route", "executed_route", "requested_formulation", "planned_formulation", "executed_formulation", "requested_backend", "planned_backend", "executed_backend", "requested_provider", "planned_provider", "executed_provider", "requested_kernel", "planned_kernel", "executed_kernel", "reuse", "certificate_kind", "certificate_failures", "iterations", "trajectory_semantics", "trajectory_reason", "warmup_excluded", "sample_count"))
    for field in ("source_commit", "tree_fingerprint")
        valid[field] = "a"^40
    end
    for field in ("input_fingerprint", "project_sha256", "manifest_sha256", "catalog_artifact_sha256", "environment_fingerprint", "provider_fingerprint")
        valid[field] = "a"^64
    end
    valid["objective_interval"] = Dict("lower" => 0.0, "upper" => 1.0)
    valid["resolved_tolerances"] = Dict("primal" => 1e-8, "dual" => 1e-8, "gap" => 1e-8)
    valid["route_receipt"] = Dict(key => "value" for key in route_keys)
    valid["trajectory_semantics"] = "not_applicable"
    valid["trajectory_sha"] = ""
    valid["warmup_excluded"] = 1
    valid["sample_count"] = 3
    valid_row = Dict{String,Any}("receipt" => valid)
    @test Main._require_receipt(valid_row) === valid
    missing_nested = deepcopy(valid_row); delete!(missing_nested["receipt"]["route_receipt"], "reuse")
    @test_throws ArgumentError Main._require_receipt(missing_nested)
    empty_nested = deepcopy(valid_row); empty_nested["receipt"]["objective_interval"] = Dict("lower" => "", "upper" => 1.0)
    @test_throws ArgumentError Main._require_receipt(empty_nested)
    @test Main._valid_trajectory!(Dict("trajectory_semantics" => "not_applicable",
        "trajectory_sha" => "", "trajectory_reason" => "no published trace"), "fixture") === nothing
    @test Main._valid_trajectory!(Dict("trajectory_semantics" => "sha256",
        "trajectory_sha" => "a"^64, "trajectory_reason" => "published trace"), "fixture") === nothing
    @test_throws ArgumentError Main._valid_trajectory!(Dict(
        "trajectory_semantics" => "validated", "trajectory_sha" => "a"^64,
        "trajectory_reason" => "legacy"), "fixture")
    @test_throws ArgumentError Main._valid_trajectory!(Dict(
        "trajectory_semantics" => "not_applicable", "trajectory_sha" => "a",
        "trajectory_reason" => "missing"), "fixture")
    @test_throws ArgumentError Main._valid_trajectory!(Dict(
        "trajectory_semantics" => "sha256", "trajectory_sha" => "A"^64,
        "trajectory_reason" => "uppercase"), "fixture")
end
