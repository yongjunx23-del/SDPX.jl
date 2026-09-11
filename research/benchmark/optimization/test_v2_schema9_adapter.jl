using Test
using SDPX

if !isdefined(Main, :GeneralBenchmarkV2)
    include(joinpath(@__DIR__, "..", "general", "v2", "GeneralBenchmarkV2.jl"))
end
if !isdefined(Main, :ProfileCatalog)
    include(joinpath(@__DIR__, "profile_catalog.jl"))
end
include(joinpath(@__DIR__, "v2_schema9_adapter.jl"))

using .GeneralBenchmarkV2
using .ProfileCatalog
using .V2Schema9Adapter

@testset "V2 schema-v9 adapter validates its own optimizer contract" begin
    catalog = GeneralBenchmarkV2.lp_tranche_catalog()
    instance = only(filter(x -> x.id === :v2_lp_box_small, catalog.instances))
    precision = GeneralBenchmarkV2.V2Precision(
        :Float64, Float64, 53, "1e-8", "5e-7", :cholmod,
    )
    row = profile_v2_target(catalog, instance, precision)
    @test row.solve_eligible
    @test !row.build_only
    @test row.warmup_count == 1
    @test length(row.sample_seconds) == 3
    @test length(row.sample_iterations) == 3
    @test length(unique(row.sample_iterations)) == 1
    @test length(unique(row.sample_objective)) == 1
    @test row.objective == row.sample_objective[1]
    @test row.reference_objective == -5.0
    @test row.objective != row.reference_objective
    @test all(row.sample_certificate_valid)
    @test all(row.sample_semantic_pass)
    @test length(row.sample_core_seconds) == 3
    @test validate_profile_row(row; live=true)
    @test row.receipt["warmup_excluded"] == 1
    @test row.receipt["sample_count"] == 3
    @test row.receipt["fresh_process"] === false
    @test row.receipt["precision_bits"] == 53
    @test length(row.receipt["sample_total_seconds"]) == 3
    @test length(row.receipt["sample_core_seconds"]) == 3
    @test row.receipt["phase_accounting_complete"] === false
    @test row.receipt["reference_objective"] == -5.0
    schema = schema9_row(row)
    @test schema.precision_bits == 53
    @test schema.total_seconds == ProfileCatalog._median(row.sample_seconds)
    @test schema.core_seconds == ProfileCatalog._median(row.sample_core_seconds)
    @test schema.setup_seconds === missing
    @test schema.frontend_seconds == row.setup_seconds
    doctored = ProfileRow(; merge(NamedTuple{fieldnames(ProfileRow)}(
        Tuple(getfield(row, name) for name in fieldnames(ProfileRow))),
        (; objective=123.0, sample_objective=[123.0, 123.0, 123.0]))...)
    @test schema9_row(doctored).objective == "123.0"
    @test schema9_row(doctored).reference_objective == "-5.0"
    @test schema9_row(doctored).objective_error == abs(123.0 - row.reference_objective)
    missing_core = ProfileRow(; merge(NamedTuple{fieldnames(ProfileRow)}(
        Tuple(getfield(row, name) for name in fieldnames(ProfileRow))),
        (; sample_core_seconds=Union{Nothing,Float64}[nothing, nothing, nothing]))...)
    @test validate_profile_row(missing_core)
    @test schema9_row(missing_core).core_seconds === missing
    @test schema9_row(missing_core).phase_accounted_seconds === missing
    @test schema9_row(missing_core).phase_unaccounted_seconds === missing
    partial_core = ProfileRow(; merge(NamedTuple{fieldnames(ProfileRow)}(
        Tuple(getfield(row, name) for name in fieldnames(ProfileRow))),
        (; sample_core_seconds=Union{Nothing,Float64}[nothing, row.sample_core_seconds[2],
            row.sample_core_seconds[3]]))...)
    @test validate_profile_row(partial_core)
    @test schema9_row(partial_core).core_seconds === missing
    @test schema9_row(partial_core).phase_accounted_seconds === missing
    @test Set(keys(row.receipt["route_receipt"])) == Set((
        "requested_route", "planned_route", "executed_route",
        "requested_formulation", "planned_formulation", "executed_formulation",
        "requested_backend", "planned_backend", "executed_backend",
        "requested_provider", "planned_provider", "executed_provider",
        "requested_kernel", "planned_kernel", "executed_kernel", "reuse"))

    output = tempname()
    paths = write_schema9(output, [row])
    @test isfile(paths.tsv)
    @test isfile(paths.toml)
    text = read(paths.tsv, String)
    @test occursin("schema_version", first(split(text, '\n')))
    @test occursin("v2_lp_box_small", text)
    rm(paths.tsv; force=true)
    rm(paths.toml; force=true)
end

@testset "V2 schema-v9 adapter rejects holdout targets" begin
    catalog = GeneralBenchmarkV2.lp_tranche_catalog()
    original = only(filter(x -> x.id === :v2_lp_box_small, catalog.instances))
    holdout = GeneralBenchmarkV2.V2Instance(original.id, original.family, original.tier,
        original.axis_values, :holdout, original.source, original.provenance,
        original.checksum, original.resource, original.reference, original.payload)
    badcatalog = GeneralBenchmarkV2.V2Catalog(:holdout_probe, catalog.version,
        catalog.families, [holdout], (train=Symbol[], holdout=[holdout.id], sentinel=Symbol[]))
    precision = GeneralBenchmarkV2.V2Precision(:Float64, Float64, 53,
        "1e-8", "5e-7", :cholmod)
    @test_throws ArgumentError profile_v2_target(badcatalog, holdout, precision)

    # A forged holdout with the same ID as a real training row must not be
    # accepted merely because an ID search finds a training instance.
    forged_holdout = GeneralBenchmarkV2.V2Instance(original.id, original.family,
        original.tier, original.axis_values, :holdout, original.source,
        original.provenance, original.checksum, original.resource,
        original.reference, original.payload)
    @test_throws ArgumentError profile_v2_target(catalog, forged_holdout, precision)
end

@testset "V2 schema-v9 adapter rejects malformed samples" begin
    catalog = GeneralBenchmarkV2.lp_tranche_catalog()
    instance = only(filter(x -> x.id === :v2_lp_box_small, catalog.instances))
    precision = GeneralBenchmarkV2.V2Precision(
        :Float64, Float64, 53, "1e-8", "5e-7", :cholmod,
    )
    row = profile_v2_target(catalog, instance, precision)
    names = fieldnames(ProfileRow)
    values = NamedTuple{names}(Tuple(getfield(row, name) for name in names))
    bad = ProfileRow(; merge(values, (; sample_iterations=[1, 2, 1]))...)
    @test !validate_profile_row(bad)
end
