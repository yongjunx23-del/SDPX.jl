using Test
using TOML

# Keep the focused file useful on its own as well as through runtests.jl.
if !isdefined(Main, :PhysicsBenchmarkHarness)
    include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "PhysicsBenchmarkHarness.jl"))
end
using .PhysicsBenchmarkHarness

function _compare_fixture(path; objective, total_seconds=1.0,
                          iterations=8, allocated_bytes=nothing,
                          process_peak_rss_bytes=nothing,
                          workspace_bytes=nothing, semantic_pass=true,
                          certificate_valid=true, input_fingerprint="fixture",
                          sample_count=1, sample_semantic_parity=nothing,
                          schema_version=8,
                          solver_source_sha256=repeat("a", 64),
                          contract_fingerprint=repeat("b", 64))
    row = Dict{String,Any}(
        "suite" => "fixture",
        "catalog_name" => "fixture",
        "catalog_version" => "1",
        "source" => "fixture",
        "problem_id" => "fixture/case",
        "arithmetic" => "float64",
        "requested_provider" => "auto",
        "execution_mode" => "solve",
        "requested_engine" => "auto",
        "executed_engine" => "sdpx_legacy",
        "scaling" => "tiny",
        "layout" => "lp",
        "campaign_id" => "fixture-campaign",
        "shard_id" => "shard-1-of-1",
        "shard_index" => "1",
        "shard_count" => "1",
        "pbs_job_id" => "fixture-job",
        "pbs_array_index" => "1",
        "pbs_queue" => "fixture-queue",
        "pbs_node" => "fixture-node",
        "input_fingerprint" => input_fingerprint,
        "external_checksum" => "fixture-checksum",
        "source_dirty" => false,
        "schema_version" => schema_version,
        "solver_source_sha256" => solver_source_sha256,
        "project_sha256" => repeat("1", 64),
        "manifest_sha256" => repeat("2", 64),
        "benchmark_driver_sha256" => repeat("3", 64),
        "catalog_source_sha256" => repeat("4", 64),
        "harness_source_sha256" => repeat("5", 64),
        "schema_source_sha256" => repeat("6", 64),
        "contract_fingerprint" => contract_fingerprint,
        "conic_formulation" => "lp_native",
        "planned_formulation" => "dense_augmented",
        "executed_formulation" => "dense_augmented",
        "planned_backend" => "dense",
        "executed_backend" => "dense",
        "planned_provider" => "blas_lapack",
        "executed_provider" => "blas_lapack",
        "fallback_reason" => "none",
        "la_fallback_reason" => "none",
        "status" => "Optimal",
        "reference_absolute_tolerance" => "1e-8",
        "reference_relative_tolerance" => "1e-8",
        "semantic_pass" => semantic_pass,
        "certificate_valid" => certificate_valid,
        "catalog_validation_pass" => true,
        "objective" => objective,
        "total_seconds" => total_seconds,
        "iterations" => iterations,
        "sample_count" => sample_count,
    )
    sample_semantic_parity === nothing ||
        (row["sample_semantic_parity"] = sample_semantic_parity)
    allocated_bytes === nothing || (row["allocated_bytes"] = allocated_bytes)
    process_peak_rss_bytes === nothing ||
        (row["process_peak_rss_bytes"] = process_peak_rss_bytes)
    workspace_bytes === nothing || (row["workspace_bytes"] = workspace_bytes)
    document = Dict("schema_version" => schema_version, "result" => [row])
    if sample_count >= 3
        row["sample_seconds"] = "[0.8,1.0,1.2]"
        row["sample_semantic_pass"] = "[true,true,true]"
        row["sample_status"] = "[\"Optimal\",\"Optimal\",\"Optimal\"]"
        row["sample_iterations"] = "[8,8,8]"
        row["sample_objective"] = "[\"$(objective)\",\"$(objective)\",\"$(objective)\"]"
        row["sample_certificate_valid"] = "[true,true,true]"
        row["sample_route"] = "[\"solve|auto|sdpx_legacy|tiny|lp|lp_native|dense_augmented|dense_augmented|dense|dense|blas_lapack|blas_lapack|||none|none\",\"solve|auto|sdpx_legacy|tiny|lp|lp_native|dense_augmented|dense_augmented|dense|dense|blas_lapack|blas_lapack|||none|none\",\"solve|auto|sdpx_legacy|tiny|lp|lp_native|dense_augmented|dense_augmented|dense|dense|blas_lapack|blas_lapack|||none|none\"]"
        row["sample_parity_failures"] = ""
        row["sample_median_seconds"] = 1.0
        row["sample_min_seconds"] = 0.8
        row["sample_max_seconds"] = 1.2
        row["sample_mad_seconds"] = 0.2
        row["sample_spread_seconds"] = 0.4
        row["total_seconds_iqr"] = 0.2
    end
    open(path, "w") do io
        TOML.print(io, document; sorted=true)
    end
    return path
end

@testset "sample parity is part of comparison validity" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    _compare_fixture(
        baseline;
        objective="1.0",
        sample_count=3,
        sample_semantic_parity=false,
    )
    _compare_fixture(
        candidate;
        objective="1.0",
        sample_count=3,
        sample_semantic_parity=false,
    )
    row = only(compare_result_files(baseline, candidate; allow_dirty=true))
    @test !row.samples_parity_match
    @test !row.comparison_valid
    @test occursin("baseline_samples_not_valid", row.comparison_evidence)
    @test occursin("candidate_samples_not_valid", row.comparison_evidence)
end

@testset "serialized multi-sample evidence is independently revalidated" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    _compare_fixture(baseline; objective="1.0", sample_count=3)
    _compare_fixture(candidate; objective="1.0", sample_count=3)
    valid = only(compare_result_files(baseline, candidate; allow_dirty=true))
    @test valid.comparison_valid
    @test valid.samples_parity_match

    tampered = TOML.parsefile(candidate)
    tampered["result"][1]["sample_seconds"] = "[0.8,1.0]"
    open(candidate, "w") do io
        TOML.print(io, tampered; sorted=true)
    end
    invalid = only(compare_result_files(baseline, candidate; allow_dirty=true))
    @test !invalid.comparison_valid
    @test occursin("candidate_sample_seconds_length", invalid.comparison_evidence)

    _compare_fixture(candidate; objective="1.0", sample_count=3)
    tampered = TOML.parsefile(candidate)
    tampered["result"][1]["sample_median_seconds"] = 2.0
    open(candidate, "w") do io
        TOML.print(io, tampered; sorted=true)
    end
    invalid = only(compare_result_files(baseline, candidate; allow_dirty=true))
    @test !invalid.comparison_valid
    @test occursin("candidate_sample_median_seconds_mismatch", invalid.comparison_evidence)

    _compare_fixture(candidate; objective="1.0", sample_count=3)
    tampered = TOML.parsefile(candidate)
    tampered["result"][1]["sample_status"] = "[\"Optimal\",\"Infeasible\",\"Optimal\"]"
    tampered["result"][1]["sample_semantic_parity"] = true
    open(candidate, "w") do io
        TOML.print(io, tampered; sorted=true)
    end
    invalid = only(compare_result_files(baseline, candidate; allow_dirty=true))
    @test !invalid.comparison_valid
    @test occursin("candidate_sample_status_parity", invalid.comparison_evidence)
end

@testset "route matrix and timing fields fail closed" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    _compare_fixture(baseline; objective="1.0")
    _compare_fixture(candidate; objective="1.0")
    document = TOML.parsefile(candidate)
    document["result"][1]["requested_engine"] = "native_hsd"
    open(candidate, "w") do io
        TOML.print(io, document; sorted=true)
    end
    invalid = only(compare_result_files(baseline, candidate; allow_dirty=true))
    @test !invalid.comparison_valid
    @test occursin("candidate_native_requested_engine_invalid", invalid.comparison_evidence)

    _compare_fixture(candidate; objective="1.0")
    document = TOML.parsefile(candidate)
    document["result"][1]["total_seconds"] = "NaN"
    document["result"][1]["setup_seconds"] = "-1"
    open(candidate, "w") do io
        TOML.print(io, document; sorted=true)
    end
    invalid = only(compare_result_files(baseline, candidate; allow_dirty=true))
    @test !invalid.comparison_valid
    @test occursin("candidate_total_seconds_invalid", invalid.comparison_evidence)
    @test occursin("candidate_setup_seconds_invalid", invalid.comparison_evidence)

    empty_document = tempname() * ".toml"
    open(empty_document, "w") do io
        TOML.print(io, Dict("schema_version" => 8, "result" => Any[]); sorted=true)
    end
    @test_throws ArgumentError compare_result_files(empty_document, baseline; allow_dirty=true)
end

@testset "comparison CLI preserves diagnostics and returns failure status" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    _compare_fixture(baseline; objective="1.0")
    _compare_fixture(candidate; objective="1.0", semantic_pass=false)
    project = normpath(joinpath(@__DIR__, ".."))
    compare_script = joinpath(project, "benchmark", "bootstrap", "compare.jl")
    julia = Base.julia_cmd()
    command(arguments) = Cmd(vcat(
        julia.exec,
        ["--project=$(project)", compare_script, arguments...],
    ))
    @test success(pipeline(
        command([baseline, baseline]); stdout=devnull, stderr=devnull,
    ))
    @test !success(pipeline(
        command([baseline, candidate]); stdout=devnull, stderr=devnull,
    ))
    diagnostic = tempname() * ".tsv"
    @test !success(pipeline(
        command([baseline, candidate, diagnostic]);
        stdout=devnull, stderr=devnull,
    ))
    @test isfile(diagnostic)
end

@testset "high-precision result comparison" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    # Float64 rounds both values to one, while the last decimal digit remains
    # observable by the scoped BigFloat subtraction.
    prefix = "1." * repeat("0", 140)
    _compare_fixture(
        baseline;
        objective=prefix * "1",
        total_seconds="2.0",
        iterations="8",
        allocated_bytes="1000",
        process_peak_rss_bytes="4000",
        workspace_bytes="8000",
    )
    _compare_fixture(
        candidate;
        objective=prefix * "2",
        total_seconds="3.0",
        iterations="11.0",
        allocated_bytes="2500",
        process_peak_rss_bytes="6000",
        workspace_bytes="12000",
    )
    ambient = precision(BigFloat)
    output = tempname() * ".tsv"
    rows = compare_result_files(baseline, candidate; output=output)
    @test precision(BigFloat) == ambient
    row = only(rows)
    @test row.objective_delta isa BigFloat
    @test row.objective_delta != 0
    @test abs(row.objective_delta) < BigFloat("1e-100")
    @test occursin("objective_delta", readline(output))
    @test occursin("e-", lowercase(read(output, String)))
    @test row.iteration_delta == 3
    @test row.total_seconds_ratio == 1.5
    @test row.allocated_bytes_delta == 1500
    @test row.allocated_bytes_ratio == 2.5
    @test row.process_peak_rss_bytes_delta == 2000
    @test row.process_peak_rss_bytes_ratio == 1.5
    @test row.workspace_bytes_delta == 4000
    @test row.workspace_bytes_ratio == 1.5
    @test row.comparison_valid
    @test row.comparison_evidence == ""

    # Missing and zero memory telemetry is a valid compatibility case, not a
    # parser error and not an artificial zero-speedup claim.
    missing_baseline = tempname() * ".toml"
    missing_candidate = tempname() * ".toml"
    _compare_fixture(missing_baseline; objective="1.0", allocated_bytes=0)
    _compare_fixture(missing_candidate; objective="1.0", allocated_bytes=100)
    missing_row = only(compare_result_files(missing_baseline, missing_candidate))
    @test missing_row.allocated_bytes_delta === missing
    @test missing_row.allocated_bytes_ratio === missing
    @test missing_row.process_peak_rss_bytes_delta === missing
    @test missing_row.process_peak_rss_bytes_ratio === missing
    @test missing_row.workspace_bytes_delta === missing
    @test missing_row.workspace_bytes_ratio === missing
end

@testset "comparison pairing and semantic gates" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    _compare_fixture(baseline; objective="1.0")
    _compare_fixture(candidate; objective="1.0", input_fingerprint="other")
    @test_throws ArgumentError compare_result_files(baseline, candidate)

    dirty = tempname() * ".toml"
    _compare_fixture(dirty; objective="1.0")
    document = TOML.parsefile(dirty)
    document["result"][1]["source_dirty"] = true
    open(dirty, "w") do io
        TOML.print(io, document; sorted=true)
    end
    @test_throws ArgumentError compare_result_files(dirty, dirty)
    @test length(compare_result_files(dirty, dirty; allow_dirty=true)) == 1

    semantic_baseline = tempname() * ".toml"
    semantic_candidate = tempname() * ".toml"
    _compare_fixture(semantic_baseline; objective="1.0")
    _compare_fixture(
        semantic_candidate;
        objective="1.0",
        semantic_pass=false,
        certificate_valid=false,
    )
    semantic_row = only(compare_result_files(semantic_baseline, semantic_candidate))
    @test !semantic_row.comparison_valid
    @test occursin("semantic_pass_mismatch", semantic_row.comparison_evidence)
    @test occursin("certificate_mismatch", semantic_row.comparison_evidence)

    changed_solver = tempname() * ".toml"
    _compare_fixture(
        changed_solver;
        objective="1.0",
        solver_source_sha256=repeat("b", 64),
    )
    changed_row = only(compare_result_files(semantic_baseline, changed_solver))
    @test changed_row.baseline_solver_source_sha256 == repeat("a", 64)
    @test changed_row.candidate_solver_source_sha256 ==
          repeat("b", 64)
    @test changed_row.baseline_solver_source_sha256_valid
    @test changed_row.candidate_solver_source_sha256_valid
    @test changed_row.solver_source_sha256_valid
    @test changed_row.comparison_valid
end

@testset "schema-v8 route/topology pairing and cross-job identity" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    _compare_fixture(baseline; objective="1.0")
    _compare_fixture(candidate; objective="1.0")

    document = TOML.parsefile(candidate)
    document["result"][1]["execution_mode"] = "profile"
    open(candidate, "w") do io
        TOML.print(io, document; sorted=true)
    end
    route_row = only(compare_result_files(baseline, candidate))
    @test !route_row.execution_mode_match
    @test !route_row.comparison_valid
    @test occursin("pairing:execution_mode", route_row.comparison_evidence)

    # Campaign/PBS identities describe where a row came from, but are not
    # semantic route fields.  A baseline and candidate may therefore come
    # from different jobs while remaining comparable.
    document["result"][1]["execution_mode"] = "solve"
    document["result"][1]["campaign_id"] = "candidate-job"
    document["result"][1]["pbs_job_id"] = "9876.server"
    open(candidate, "w") do io
        TOML.print(io, document; sorted=true)
    end
    cross_job = only(compare_result_files(baseline, candidate))
    @test cross_job.comparison_valid
    @test cross_job.baseline_campaign_id == "fixture-campaign"
    @test cross_job.candidate_campaign_id == "candidate-job"
    @test cross_job.baseline_pbs_job_id == "fixture-job"
    @test cross_job.candidate_pbs_job_id == "9876.server"
    @test cross_job.audit_identity_recorded

    document["result"][1]["shard_count"] = "2"
    open(candidate, "w") do io
        TOML.print(io, document; sorted=true)
    end
    topology_row = only(compare_result_files(baseline, candidate))
    @test !topology_row.shard_topology_match
    @test !topology_row.comparison_valid
    @test occursin("pairing:shard_count", topology_row.comparison_evidence)
end

@testset "schema-v8 solver source hash gate" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    _compare_fixture(baseline; objective="1.0")
    _compare_fixture(candidate; objective="1.0", solver_source_sha256=repeat("c", 64))
    valid_row = only(compare_result_files(baseline, candidate))
    @test valid_row.solver_source_sha256_valid
    @test valid_row.comparison_valid

    missing_hash = tempname() * ".toml"
    _compare_fixture(missing_hash; objective="1.0", solver_source_sha256="")
    missing_row = only(compare_result_files(baseline, missing_hash))
    @test !missing_row.candidate_solver_source_sha256_valid
    @test !missing_row.solver_source_sha256_valid
    @test !missing_row.comparison_valid
    @test occursin("solver_source_sha256_invalid", missing_row.comparison_evidence)

    missing_row_schema = tempname() * ".toml"
    _compare_fixture(missing_row_schema; objective="1.0")
    missing_schema_document = TOML.parsefile(missing_row_schema)
    delete!(missing_schema_document["result"][1], "schema_version")
    open(missing_row_schema, "w") do io
        TOML.print(io, missing_schema_document; sorted=true)
    end
    missing_schema_row = only(compare_result_files(baseline, missing_row_schema))
    @test !missing_schema_row.solver_source_sha256_valid
    @test !missing_schema_row.comparison_valid
    @test occursin("legacy_schema_version", missing_schema_row.comparison_evidence)

    invalid_hash = tempname() * ".toml"
    _compare_fixture(invalid_hash; objective="1.0", solver_source_sha256=repeat("g", 64))
    invalid_row = only(compare_result_files(invalid_hash, candidate))
    @test !invalid_row.baseline_solver_source_sha256_valid
    @test !invalid_row.solver_source_sha256_valid
    @test !invalid_row.comparison_valid

    legacy = tempname() * ".toml"
    _compare_fixture(
        legacy;
        objective="1.0",
        schema_version=4,
        solver_source_sha256=repeat("d", 64),
    )
    legacy_row = only(compare_result_files(legacy, legacy))
    @test !legacy_row.solver_source_sha256_valid
    @test !legacy_row.comparison_valid
    @test occursin("legacy_schema_version", legacy_row.comparison_evidence)
end

@testset "schema-v7 is an explicit non-claim diagnostic" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    _compare_fixture(baseline; objective="1.0", schema_version=7)
    _compare_fixture(candidate; objective="1.0", schema_version=7)
    row = only(compare_result_files(baseline, candidate))
    @test !row.comparison_valid
    @test occursin("legacy_schema_version", row.comparison_evidence)
end

@testset "schema-v8 duplicate keys, canonical shards, and content identity" begin
    baseline = tempname() * ".toml"
    candidate = tempname() * ".toml"
    _compare_fixture(baseline; objective="1.0")
    _compare_fixture(candidate; objective="1.0")

    duplicate_document = TOML.parsefile(baseline)
    push!(duplicate_document["result"], deepcopy(only(duplicate_document["result"])))
    open(baseline, "w") do io
        TOML.print(io, duplicate_document; sorted=true)
    end
    @test_throws ArgumentError compare_result_files(baseline, candidate)

    _compare_fixture(baseline; objective="1.0")
    canonical_document = TOML.parsefile(candidate)
    canonical_document["result"][1]["shard_index"] = "01"
    canonical_document["result"][1]["shard_count"] = "001"
    open(candidate, "w") do io
        TOML.print(io, canonical_document; sorted=true)
    end
    canonical = only(compare_result_files(baseline, candidate))
    @test canonical.comparison_valid
    @test canonical.shard_index_match
    @test canonical.shard_count_match

    invalid_hash = TOML.parsefile(candidate)
    invalid_hash["result"][1]["catalog_source_sha256"] = "invalid"
    open(candidate, "w") do io
        TOML.print(io, invalid_hash; sorted=true)
    end
    invalid_baseline_hash = TOML.parsefile(baseline)
    invalid_baseline_hash["result"][1]["catalog_source_sha256"] = "invalid"
    open(baseline, "w") do io
        TOML.print(io, invalid_baseline_hash; sorted=true)
    end
    invalid = only(compare_result_files(baseline, candidate))
    @test !invalid.comparison_valid
    @test occursin(
        "candidate_catalog_source_sha256_invalid",
        invalid.comparison_evidence,
    )

    _compare_fixture(baseline; objective="1.0")
    _compare_fixture(candidate; objective="1.0")
    for path in (baseline, candidate)
        missing_route = TOML.parsefile(path)
        missing_route["result"][1]["executed_provider"] = ""
        open(path, "w") do io
            TOML.print(io, missing_route; sorted=true)
        end
    end
    missing_route = only(compare_result_files(baseline, candidate))
    @test !missing_route.comparison_valid
    @test occursin("baseline_executed_provider_empty", missing_route.comparison_evidence)

    _compare_fixture(baseline; objective="1.0")
    _compare_fixture(candidate; objective="1.0")
    changed_contract = TOML.parsefile(candidate)
    changed_contract["result"][1]["contract_fingerprint"] = repeat("c", 64)
    open(candidate, "w") do io
        TOML.print(io, changed_contract; sorted=true)
    end
    @test_throws ArgumentError compare_result_files(baseline, candidate)
end
