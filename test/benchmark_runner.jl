using Test
using TOML
using SDPX

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

@testset "build-only catalog never calls the solver" begin
    spec = PhysicsBenchmarkSpec(
        id="build-only/lp",
        name="build-only LP fixture",
        family=:lp,
        problem_type=:linear_program,
        tags=(:build_only,),
        reference=PhysicsBenchmarkReference(status=:build_only),
        fingerprint="build-only-fixture-v1",
    )
    builder = function (_, ::Type{T}) where {T}
        problem = SDPX.linear_program(
            zeros(T, 1), ones(T, 1, 1), zeros(T, 1); T,
        )
        return (
            problem,
            expected=nothing,
            kind=:lp,
            external_checksum="build-only-fixture-v1",
            solve_settings=(build_only=true,),
        )
    end
    catalog = PhysicsBenchmarkCatalog(
        :build_only_fixture, "1", [spec],
        Dict(:smoke => [PhysicsBenchmarkEntry(
            spec.id, :float64, :auto,
        )]),
        builder,
    )
    output = tempname() * ".toml"
    run = run_suite(
        catalog, :smoke; samples=3, warmup=false, output,
    )
    row = only(run.rows)
    @test row.status == :build_only
    @test row.termination_stage == :construction
    @test row.termination_reason == :model_built
    @test row.semantic_pass
    @test row.iterations === missing
    @test row.sample_count == 3
    @test row.sample_semantic_parity
    @test row.certificate_policy == :not_applicable_build_only
    @test row.external_checksum == "build-only-fixture-v1"
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
