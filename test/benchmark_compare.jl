using Test
using TOML

# The normal test runner includes benchmark_registry.jl first.  Keeping this
# fallback makes the focused file useful on its own as well.
if !isdefined(Main, :SDPXBenchmarkRegistry)
    include(joinpath(@__DIR__, "..", "benchmark", "SDPXBenchmarkRegistry.jl"))
end
using .SDPXBenchmarkRegistry

function _compare_fixture(path; objective, total_seconds=1.0,
                          iterations=8, allocated_bytes=nothing,
                          process_peak_rss_bytes=nothing,
                          workspace_bytes=nothing, semantic_pass=true,
                          certificate_valid=true, input_fingerprint="fixture",
                          sample_count=1, sample_semantic_parity=nothing,
                          schema_version=6,
                          solver_source_sha256=repeat("a", 64))
    row = Dict{String,Any}(
        "suite" => "fixture",
        "problem_id" => "fixture/case",
        "arithmetic" => "float64",
        "requested_provider" => "auto",
        "input_fingerprint" => input_fingerprint,
        "source_dirty" => false,
        "schema_version" => schema_version,
        "solver_source_sha256" => solver_source_sha256,
        "status" => "Optimal",
        "semantic_pass" => semantic_pass,
        "certificate_valid" => certificate_valid,
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
    @test row.samples_parity_match
    @test !row.comparison_valid
    @test occursin("baseline_samples_not_valid", row.comparison_evidence)
    @test occursin("candidate_samples_not_valid", row.comparison_evidence)
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

@testset "schema-v6 solver source hash gate" begin
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
