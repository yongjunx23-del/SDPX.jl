using Test
include(joinpath(@__DIR__, "v2", "GeneralBenchmarkV2.jl"))
using .GeneralBenchmarkV2

@testset "general benchmark V2 schema and deterministic identity" begin
    axes = [V2Axis(:z, [2, 1]), V2Axis(:a, ["b", "a"])]
    expanded = expand(axes)
    @test [x.a for x in expanded] == ["a", "a", "b", "b"]
    @test [x.z for x in expanded] == [1, 2, 1, 2]

    exact = V2Transform(:scalar_polynomial, :sdpx_sdp, :halfline_sos,
        1, :exact_univariate_halfline;
        positive_prefactor_factored=true,
        positive_prefactor_proof="positive factor is exp(-2pi*Delta)>0",
        lifting_dimensions=(source=4, target=10, gram_blocks=2),
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    @test length(exact.fingerprint) == 64
    @test exact.lifting_dimensions.target == 10
    @test_throws ArgumentError V2Transform(:x, :y, :bad, 1, :sos_relaxation;
        positive_prefactor_factored=true)

    tier = only(filter(t -> t.name === :small, resource_tiers()))
    ref = V2Reference(:optimal, :optimal, ("0", "1"), nothing, "unit test")
    instance = V2Instance(:unit, :unit, tier, (dimension=1,), :train,
        "unit-test", (equations=("test",),), "unit-checksum",
        (wall_seconds=1, memory_bytes=1024), ref, nothing)
    @test length(input_fingerprint(instance)) == 64
    mutated = V2Instance(:unit, :unit, tier, (dimension=2,), :train,
        "unit-test", (equations=("test",),), "unit-checksum",
        (wall_seconds=1, memory_bytes=1024), ref, nothing)
    @test input_fingerprint(instance) != input_fingerprint(mutated)

    family = V2Family(:unit, V2Axis[],
        (i, p) -> V2Built(nothing, nothing, nothing, "source",
            exact, (source_dimension=1, target_dimension=1),
            (setup_seconds=nothing,)), nothing,
        (i, r) -> V2Validation(:optimal, true, true, Symbol[]), (:halfline_sos,))
    catalog = V2Catalog(:unit_catalog, 1, [family], [instance],
        (train=[:unit], holdout=Symbol[], sentinel=Symbol[]))
    @test validate_catalog(catalog)
    @test length(catalog_fingerprint(catalog)) == 64
    @test build_instance(catalog, instance, V2Precision(:Float64, Float64, 53,
        "1e-8", "5e-7", :test))[2] >= 0

    @test_throws ArgumentError V2Reference(:build_only, :optimal, nothing, nothing)
    @test_throws ArgumentError V2Catalog(:bad, 1, [family],
        [instance, instance], (train=[:unit], holdout=Symbol[], sentinel=Symbol[]))
end

@testset "general benchmark V2 adapter preserves V1 inventory" begin
    # The adapter is intentionally additive: it consumes V1 specs but does not
    # mutate the include-time V1 registry or rename any existing IDs.
    v1 = GenericConicBenchmark.inventory()
    catalog = adapt_generic_specs(v1; generic_module=GenericConicBenchmark)
    families = Set(f.name for f in catalog.families)
    @test Set((:lp, :socp, :sdp, :exp, :power)) ⊆ families
    @test length(catalog.instances) == length(v1)
    @test Set(i.id for i in catalog.instances) == Set(s.id for s in v1)
    @test all(length(i.checksum) == 64 for i in catalog.instances)
    @test all(length(input_fingerprint(i)) == 64 for i in catalog.instances)
    @test all(length(execution_fingerprint(i,
        V2Precision(:Float64, Float64, 53, "1e-8", "5e-7", :test))) == 64
        for i in catalog.instances)
    @test all(i.reference.status !== :known_solver_finding for i in catalog.instances)
    @test count(i -> i.reference.status === :xfail, catalog.instances) > 0

    # Build one representative instance through the additive adapter for each
    # public scalar/conic family; V1 builders and IDs remain the source of truth.
    precision = V2Precision(:Float64, Float64, 53, "1e-8", "5e-7", :test)
    for family in (:lp, :socp, :sdp, :exp, :power)
        instance = first(filter(i -> i.family === family, catalog.instances))
        built, elapsed = build_instance(catalog, instance, precision)
        @test built.problem !== nothing
        @test built.transform.transform_id === :identity
        @test elapsed >= 0
    end
end
