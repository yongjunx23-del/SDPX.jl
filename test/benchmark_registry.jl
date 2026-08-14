using Test
using TOML

include(joinpath(@__DIR__, "..", "benchmark", "SDPXBenchmarkRegistry.jl"))
using .SDPXBenchmarkRegistry

include(joinpath(
    @__DIR__, "..", "benchmark", "round4_scoreboard_contracts.jl",
))

@testset "formulation scoreboard correctness gates" begin
    @test _round4_objective_valid(6.0 + 1.0e-9, 6.0, 1.0e-8, 1.0e-8)
    @test !_round4_objective_valid(6.1, 6.0, 1.0e-8, 1.0e-8)
    @test _round4_severe_false_negative(false, true, false)
    @test _round4_severe_false_negative(false, false, true)
    @test !_round4_severe_false_negative(true, true, true)
    @test !_round4_severe_false_negative(false, false, false)
end

@testset "benchmark registry contracts" begin
    registry = benchmark_registry()
    @test length(registry) >= 60
    @test length(unique(spec.id for spec in registry)) == length(registry)
    @test all(!isempty(string(spec.purpose)) for spec in registry)

    for source in (:netlib, :sdplib, :dimacs, :cblib)
        specs = filter(spec -> spec.source === source, registry)
        @test !isempty(specs)
        @test all(spec.external !== nothing for spec in specs)
        @test all(!isempty(spec.external.authoritative_url) for spec in specs)
        @test all(!isempty(spec.external.filename) for spec in specs)
        @test all(spec.external.format isa Symbol for spec in specs)
        @test all(!isempty(spec.external.license_note) for spec in specs)
    end

    @test suite_names() == (:micro, :representative, :local_full, :heavy)
    @test 6 <= length(suite_entries(:micro)) <= 12
    @test 20 <= length(suite_entries(:representative)) <= 30
    @test 50 <= length(suite_entries(:local_full)) <= 100
    @test all(entry.arithmetic === :registered_only for entry in
              suite_entries(:heavy))
    @test_throws ArgumentError run_suite(:heavy; output=tempname())
    @test_throws ArgumentError SDPXBenchmarkRegistry.main(["heavy", "--prepare"])

    micro_ids = Set(entry.problem_id for entry in suite_entries(:micro))
    @test "synthetic/lp_eq_exact_deficient" in micro_ids
    @test "synthetic/sdp_small_eig_1e8" in micro_ids
    @test all(benchmark_spec(id).source === :synthetic for id in micro_ids)

    seed_spec = benchmark_spec("synthetic/sdp_small_eig_1e8")
    first_problem = build_problem(seed_spec, Float64)
    second_problem = build_problem(seed_spec, Float64)
    @test first_problem.expected == second_problem.expected
    @test first_problem.problem.c == second_problem.problem.c
    @test first_problem.problem.C == second_problem.problem.C

    external = benchmark_spec("netlib/afiro")
    status = external_cache_status(external; cache_dir=mktempdir())
    @test !status.available
    @test status.reason === :not_cached

    corrupt_cache = mktempdir()
    corrupt_path = joinpath(
        corrupt_cache, string(external.source), external.external.filename,
    )
    mkpath(dirname(corrupt_path))
    write(corrupt_path, "not the authoritative benchmark")
    corrupt_status = external_cache_status(external; cache_dir=corrupt_cache)
    @test !corrupt_status.available
    @test corrupt_status.reason === :checksum_mismatch

    rows = run_suite(
        :representative;
        problem="netlib/afiro",
        output=tempname() * ".toml",
        warmup=false,
    ).rows
    @test length(rows) == 1
    @test rows[1].status === :skipped
    @test rows[1].skip_reason === :not_cached

    local_output = tempname() * ".toml"
    local_result = run_suite(
        :micro;
        problem="synthetic/lp_box",
        output=local_output,
        warmup=false,
    )
    @test length(local_result.rows) == 1
    @test local_result.rows[1].status === :Optimal
    @test local_result.rows[1].semantic_pass
    @test local_result.rows[1].conic_formulation === :lp_native
    @test local_result.rows[1].certificate_policy === :original_coordinate_required
    @test local_result.rows[1].provider_match
    @test isempty(local_result.rows[1].semantic_failures)
    @test isfile(local_result.paths.toml)
    @test isfile(local_result.paths.tsv)

    clean_document = TOML.parsefile(local_result.paths.toml)
    dirty_document = deepcopy(clean_document)
    for row in clean_document["result"]
        row["source_dirty"] = false
    end
    for row in dirty_document["result"]
        row["source_dirty"] = true
    end
    clean_output = tempname() * ".toml"
    dirty_output = tempname() * ".toml"
    open(clean_output, "w") do io
        TOML.print(io, clean_document; sorted=true)
    end
    open(dirty_output, "w") do io
        TOML.print(io, dirty_document; sorted=true)
    end
    @test_throws ArgumentError compare_result_files(
        dirty_output,
        dirty_output,
    )
    @test length(compare_result_files(
        clean_output,
        clean_output,
    )) == 1
end
