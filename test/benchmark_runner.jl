using Test
using TOML

if !isdefined(Main, :PhysicsBenchmarkHarness)
    include(joinpath(@__DIR__, "..", "benchmark", "PhysicsBenchmarkHarness.jl"))
end
if !isdefined(Main, :physics_benchmark_catalog)
    include(joinpath(
        @__DIR__, "..", "benchmark", "fixtures", "smoke_catalog.jl",
    ))
end
using .PhysicsBenchmarkHarness

@testset "physics benchmark catalog contract" begin
    catalog = physics_benchmark_catalog()
    @test catalog.name === :smoke
    @test catalog.version == "1"
    @test only(catalog_entries(catalog, :smoke)).problem_id == "smoke/lp_box"
    @test catalog_spec(catalog, "smoke/lp_box").fingerprint != ""
    @test_throws ArgumentError catalog_entries(catalog, :unknown)
    @test validate_result(catalog, nothing, nothing, nothing, (;)) == String[]
    loaded = load_catalog(joinpath(
        @__DIR__, "..", "benchmark", "fixtures", "smoke_catalog.jl",
    ))
    @test loaded.name === :smoke
    @test only(catalog_entries(loaded, :smoke)).problem_id == "smoke/lp_box"
end

@testset "schema-v7 runner one-row contract" begin
    @test RESULT_SCHEMA_VERSION == 7
    @test length(unique(RESULT_COLUMNS)) == length(RESULT_COLUMNS)
    @test :catalog_name in RESULT_COLUMNS
    @test :catalog_validation_pass in RESULT_COLUMNS

    output = tempname() * ".toml"
    run = run_suite(
        physics_benchmark_catalog(),
        :smoke;
        problem="smoke/lp_box",
        samples=1,
        warmup=false,
        output,
    )
    @test length(run.rows) == 1
    row = only(run.rows)
    @test row.schema_version == 7
    @test row.catalog_name === :smoke
    @test string(row.status) == "Optimal"
    @test row.catalog_validation_pass
    @test row.semantic_pass
    @test isfile(run.paths.toml)
    @test isfile(run.paths.tsv)

    document = TOML.parsefile(run.paths.toml)
    @test document["schema_version"] == 7
    @test length(document["result"]) == 1
    @test only(document["result"])["problem_id"] == "smoke/lp_box"
end
