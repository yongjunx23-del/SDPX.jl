using Test
using LinearAlgebra
using SDPX
isdefined(Main, :GenericConicBenchmark) ||
    include(joinpath(@__DIR__, "GenericConicBenchmark.jl"))
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
    ref = V2Reference(:optimal, :optimal, ("0", "1"),
        (built, cert) -> cert.valid, "unit test")
    instance = V2Instance(:unit, :unit, tier, (dimension=1,), :train,
        "unit-test", (equations=("test",), transform=exact), "unit-checksum",
        (wall_seconds=1, memory_bytes=1024), ref, nothing)
    @test length(input_fingerprint(instance)) == 64
    mutated = V2Instance(:unit, :unit, tier, (dimension=2,), :train,
        "unit-test", (equations=("test",), transform=exact), "unit-checksum",
        (wall_seconds=1, memory_bytes=1024), ref, nothing)
    @test input_fingerprint(instance) != input_fingerprint(mutated)

    family = V2Family(:unit, V2Axis[],
        (i, p) -> V2Built(nothing, nothing, nothing, input_fingerprint(i),
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
    @test_throws ArgumentError V2Reference(:xfail, :interval_or_bound, nothing,
        (built, cert) -> true)
    @test_throws ArgumentError V2Reference(:optimal, :optimal, nothing, nothing)
    @test_throws ArgumentError V2Catalog(:bad, 1, [family],
        [instance, instance], (train=[:unit], holdout=Symbol[], sentinel=Symbol[]))

    # Every disposition transition is explicit and certificate-kind aware.
    @test classify_disposition(:optimal, false, :optimal, true, true, true, true) === :PASS
    @test classify_disposition(:optimal, true, :optimal, true, true, true, true) === :RESOLVED
    @test classify_disposition(:optimal, false, :numerical_breakdown, true, true, true, true) === :FAIL
    @test classify_disposition(:numerical_breakdown, false, :numerical_breakdown, true, true, false, true, [:certificate]) === :XFAIL
    @test classify_disposition(:numerical_breakdown, false, :numerical_breakdown, true, true, false, false, [:certificate]) === :FAIL
    @test classify_disposition(:numerical_breakdown, false, :optimal, true, true, true, true) === :XPASS
    @test classify_disposition(:numerical_breakdown, true, :optimal, true, true, true, true) === :RESOLVED
    @test classify_disposition(:numerical_breakdown, false, :numerical_breakdown, true, true, true, false) === :FAIL
    @test classify_disposition(:build_only, false, :build_only, true, true, false, true) === :PASS
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
    @test all(i.reference.status === :build_only for i in catalog.instances)
    @test all(i -> get(i.provenance, :compatibility_only, false) === true &&
                   get(i.provenance, :solve_eligible, true) === false &&
                   haskey(i.provenance, :v1_expected_status),
              catalog.instances)
    @test_throws ArgumentError run_instance(catalog, first(catalog.instances),
        V2Precision(:Float64, Float64, 53, "1e-8", "5e-7", :test))

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

@testset "native V2 corpus owns typed artifacts and disjoint suites" begin
    catalog = native_v2_catalog()
    @test !isempty(catalog.suites.train)
    @test !isempty(catalog.suites.holdout)
    @test !isempty(catalog.suites.sentinel)
    @test isempty(intersect(Set(catalog.suites.train), Set(catalog.suites.holdout)))
    @test isempty(intersect(Set(catalog.suites.train), Set(catalog.suites.sentinel)))
    @test all(i -> i.payload isa V2ConicArtifact, catalog.instances)
    @test all(i -> i.payload isa V2ConicArtifact && i.payload.family == i.family,
              catalog.instances)
    @test all(i -> i.reference.oracle !== nothing, catalog.instances)
    @test all(i -> i.split === :train && i.reference.status !== :build_only &&
                   i.reference.status !== :xfail &&
                   get(i.provenance, :solve_eligible, false) === true,
              training_instances(catalog))
    sent = only(filter(i -> i.split === :sentinel && i.family === :lp, catalog.instances))
    @test sent.reference.status === :xfail
    @test sent.reference.expected_status === :primal_infeasible
    @test sent.reference.prior_observed_status === :numerical_breakdown
    @test sent.reference.disposition === :XFAIL
    @test sent.reference.oracle.dual_ray == Rational{Int}[1, -1]
    @test sent.reference.certificate_kind === :farkas
    train_lp = only(filter(i -> i.family === :lp && i.split === :train, catalog.instances))
    train_soc = only(filter(i -> i.family === :soc && i.split === :train, catalog.instances))
    train_rsoc = only(filter(i -> i.family === :rsoc && i.split === :train, catalog.instances))
    train_sdp = only(filter(i -> i.family === :sdp && i.split === :train, catalog.instances))
    @test train_lp.reference.oracle.primal_witness == Rational{Int}[1//2]
    @test train_lp.reference.oracle.dual_multipliers == Rational{Int}[-1]
    @test train_lp.reference.oracle.cone_dual_slacks == Rational{Int}[0]
    @test train_lp.reference.oracle.dual_bound == train_lp.reference.oracle.objective == 1//2
    @test train_lp.reference.oracle.dual_multipliers == Rational{Int}[-1]
    @test train_rsoc.reference.oracle.primal_witness == Rational{Int}[1, 1]
    @test train_sdp.reference.oracle.primal_witness == Rational{Int}[1, 0, 1]
    @test train_sdp.reference.oracle.dual_multipliers == Rational{Int}[-1, 0, 0, -1]
    @test length(train_sdp.reference.oracle.cone_dual_slacks) == 3
    @test_throws ArgumentError V2ConicArtifact(:sdp, :asymmetric,
        Rational{Int}[1, 1//2, 1//3, 1], 2, 1//2, false, :x, 1)
    # Pure mathematical identity excludes ID/split/provenance metadata.
    same_math_a = V2Instance(:a, :soc, train_soc.tier, train_soc.axis_values,
        :train, "source-a", train_soc.provenance, train_soc.checksum,
        train_soc.resource, train_soc.reference, train_soc.payload)
    same_math_b = V2Instance(:b, :soc, train_soc.tier, train_soc.axis_values,
        :holdout, "source-b", train_soc.provenance, train_soc.checksum,
        train_soc.resource, train_soc.reference, train_soc.payload)
    @test mathematical_fingerprint(same_math_a) == mathematical_fingerprint(same_math_b)
    @test input_fingerprint(same_math_a) == input_fingerprint(same_math_b)
    bad_id = V2Instance(:wrong_id, :soc, train_soc.tier, train_soc.axis_values,
        :train, train_soc.source, train_soc.provenance, train_soc.checksum,
        train_soc.resource, train_soc.reference, train_soc.payload)
    @test_throws ArgumentError V2Catalog(:bad_id, 2, catalog.families, [bad_id],
        (train=[:wrong_id], holdout=Symbol[], sentinel=Symbol[]))
    bad_family = V2Instance(:wrong_family, :lp, train_soc.tier, train_soc.axis_values,
        :train, train_soc.source, train_soc.provenance, train_soc.checksum,
        train_soc.resource, train_soc.reference, train_soc.payload)
    @test_throws ArgumentError V2Catalog(:bad_family, 2, catalog.families, [bad_family],
        (train=[:wrong_family], holdout=Symbol[], sentinel=Symbol[]))
    @test all(i -> i.split == :train ? i.id in catalog.suites.train :
                   (i.split == :holdout ? i.id in catalog.suites.holdout : i.id in catalog.suites.sentinel),
              catalog.instances)
    training = training_instances(catalog)
    @test length(training) == 8
    @test all(i -> i.split === :train &&
                   get(i.provenance, :solve_eligible, false) === true &&
                   i.reference.status !== :xfail &&
                   i.reference.status !== :build_only,
              training)
    @test all(family -> begin
        train = only(filter(i -> i.family === family && i.split === :train, catalog.instances))
        holdout = only(filter(i -> i.family === family && i.split === :holdout, catalog.instances))
        train.checksum != holdout.checksum &&
            GeneralBenchmarkV2._hex((train.payload.coefficients, train.payload.dimension,
                train.payload.cone_parameter, train.payload.infeasible)) !=
            GeneralBenchmarkV2._hex((holdout.payload.coefficients, holdout.payload.dimension,
                holdout.payload.cone_parameter, holdout.payload.infeasible))
    end, (:lp, :nonpositive, :soc, :rsoc, :sdp, :exp, :power, :mixed))
    precision = V2Precision(:Float64, Float64, 53, "1e-8", "1e-8", :standard)
    for family in (:lp, :nonpositive, :soc, :rsoc, :sdp, :exp, :power, :mixed)
        instance = only(filter(i -> i.family === family && i.split === :train, catalog.instances))
        result = run_instance(catalog, instance, precision)
        @test result.status === :optimal
        @test result.certificate_valid
        @test result.validation.reference
        @test result.validation.failures == Symbol[]
        @test result.setup_seconds !== nothing
        @test result.core_seconds !== nothing
    end

    # Oracle checks actual packed model coordinates and actual model records,
    # not just source-artifact formulas. Mutating objective/row/cone data is
    # rejected even when the fake certificate reports the expected objective.
    soc_original = only(filter(i -> i.family === :soc && i.split === :train, catalog.instances))
    soc_built, _ = build_instance(catalog, soc_original, precision)
    fake_certificate = (primal_objective=BigFloat(0),)
    @test soc_built.oracle(soc_built, fake_certificate)
    soc_built.problem.objective.expression.coefficients[1] += 1.0
    @test !soc_built.oracle(soc_built, fake_certificate)
    soc_built, _ = build_instance(catalog, soc_original, precision)
    old_expr = soc_built.problem.constraint_blocks[1].expressions[1]
    soc_built.problem.constraint_blocks[1].expressions[1] =
        SDPX.ScalarAffine(old_expr.model, old_expr.precision_bits,
            old_expr.indices, old_expr.coefficients, old_expr.constant + 1.0)
    @test !soc_built.oracle(soc_built, fake_certificate)
    soc_built, _ = build_instance(catalog, soc_original, precision)
    soc_built.problem.constraint_blocks[1].expressions[2].coefficients[1] += 1.0
    @test !soc_built.oracle(soc_built, fake_certificate)
    soc_built, _ = build_instance(catalog, soc_original, precision)
    soc_built.oracle.primal_witness[1] += 1//1
    @test !soc_built.oracle(soc_built, fake_certificate)
    soc_built, _ = build_instance(catalog, soc_original, precision)
    soc_built.oracle.dual_multipliers[1] += 1//1
    @test !soc_built.oracle(soc_built, fake_certificate)
    sdp_built, _ = build_instance(catalog, train_sdp, precision)
    sdp_built.oracle.primal_witness[2] += 1//1
    @test !sdp_built.oracle(sdp_built, (primal_objective=BigFloat(2),))
    sdp_built, _ = build_instance(catalog, train_sdp, precision)
    sdp_built.oracle.dual_multipliers[2] += 1//1
    @test !sdp_built.oracle(sdp_built, (primal_objective=BigFloat(2),))

    # Every exact coefficient is semantic input, not decorative metadata.
    original = only(filter(i -> i.family === :soc && i.split === :train, catalog.instances))
    changed_artifact = V2ConicArtifact(:soc, original.id,
        Rational{Int}[1//3, 2//3], 2, 1//2, false, :changed, 1)
    changed = V2Instance(original.id, original.family, original.tier,
        original.axis_values, original.split, original.source, original.provenance,
        GeneralBenchmarkV2._hex(changed_artifact), original.resource, original.reference, changed_artifact)
    @test input_fingerprint(changed) != input_fingerprint(original)
    changed_built, _ = build_instance(catalog, changed, precision)
    @test changed_built.facts.coefficients == changed_artifact.coefficients
    @test changed_built.facts.cone_parameter == changed_artifact.cone_parameter
    @test changed_built.input_fingerprint == input_fingerprint(changed)
    @test changed_built.input_fingerprint != input_fingerprint(original)
    @test changed_built.facts.model_fingerprint !=
          build_instance(catalog, original, precision)[1].facts.model_fingerprint
    @test_throws ArgumentError V2ConicArtifact(:soc, :extra,
        Rational{Int}[1//2, 1//2, 1//3], 2, 1//2, false, :x, 1)
    sentinel = only(filter(i -> i.family === :soc && i.split === :sentinel, catalog.instances))
    sentinel_result = run_instance(catalog, sentinel, precision)
    @test sentinel_result.status === :numerical_breakdown
    @test sentinel_result.validation.status === :XFAIL
    @test sentinel_result.validation.observed_status == sentinel_result.status
    @test sentinel_result.validation.disposition === :XFAIL
    @test sentinel_result.validation.reference
    sentinel_built, _ = build_instance(catalog, sentinel, precision)
    @test GeneralBenchmarkV2._farkas_valid(sentinel.payload, sentinel_built)
    # The canonical encoder is explicit for exact rational coefficients and
    # does not depend on host-endian or struct string formatting.
    metadata_variant = V2ConicArtifact(:soc, train_soc.id,
        train_soc.payload.coefficients, train_soc.payload.dimension,
        train_soc.payload.cone_parameter, train_soc.payload.infeasible,
        :different_generator, 99)
    metadata_instance = V2Instance(train_soc.id, train_soc.family, train_soc.tier,
        train_soc.axis_values, train_soc.split, train_soc.source, train_soc.provenance,
        GeneralBenchmarkV2._hex(metadata_variant), train_soc.resource,
        train_soc.reference, metadata_variant)
    @test mathematical_fingerprint(metadata_instance) == mathematical_fingerprint(train_soc)
    @test input_fingerprint(metadata_instance) == input_fingerprint(train_soc)
    @test catalog_fingerprint(catalog) != catalog_fingerprint(V2Catalog(
        :general_v2_native, 2, catalog.families,
        [metadata_instance; filter(i -> i !== train_soc, catalog.instances)], catalog.suites))
    @test GeneralBenchmarkV2._hex(Rational{Int}[1//2, 1//3]) !=
          GeneralBenchmarkV2._hex(Rational{Int}[1//2, 1//4])
    @test GeneralBenchmarkV2._canonical_bytes(:x) != GeneralBenchmarkV2._canonical_bytes("x")
    @test GeneralBenchmarkV2._canonical_bytes(Float32(1)) != GeneralBenchmarkV2._canonical_bytes(Float64(1))
    @test GeneralBenchmarkV2._canonical_bytes(Int8(1)) != GeneralBenchmarkV2._canonical_bytes(Int16(1))
    @test GeneralBenchmarkV2._canonical_bytes(UInt8(1)) != GeneralBenchmarkV2._canonical_bytes(Int8(1))
    @test GeneralBenchmarkV2._canonical_bytes(Int[]) != GeneralBenchmarkV2._canonical_bytes(Array{Int,2}(undef, 0, 0))
    @test GeneralBenchmarkV2._canonical_bytes(String) != GeneralBenchmarkV2._canonical_bytes("String")
    # Exact interval bytes do not depend on ambient BigFloat precision.
    ref_at_128 = setprecision(BigFloat, 128) do
        GeneralBenchmarkV2._native_reference(original.payload).objective_interval
    end
    ref_at_512 = setprecision(BigFloat, 512) do
        GeneralBenchmarkV2._native_reference(original.payload).objective_interval
    end
    @test ref_at_128 == ref_at_512
    for bits in (256, 512)
        built_big = setprecision(BigFloat, 79) do
            build_instance(catalog, train_lp,
                V2Precision(:BigFloat, BigFloat, bits, "1e-8", "1e-8", :standard))[1]
        end
        @test built_big.facts.model_precision_bits == bits
        @test built_big.facts.model_fingerprint ==
              GeneralBenchmarkV2._native_model_fingerprint(built_big.problem)
    end
    @test bytes2hex(GeneralBenchmarkV2._canonical_bytes(Rational{Int}(1, 2))) ==
          "0a4300000000000000013143000000000000000132"
    @test GeneralBenchmarkV2._hex(Rational{Int}(1, 2)) ==
          "e6f6e48d1b9bcedbb4d603dbeffe70afff478f3da15f1109f62057b587b844e3"
    # Direct BigFloat builds are scoped by V2Precision.bits and their model
    # fingerprints are intentionally distinct even when rational coefficients
    # happen to be exactly representable at both precisions.
    bf256 = build_instance(catalog, train_sdp,
        V2Precision(:BigFloat256, BigFloat, 256, "1e-8", "1e-8", :generic))[1]
    bf512 = build_instance(catalog, train_sdp,
        V2Precision(:BigFloat512, BigFloat, 512, "1e-8", "1e-8", :generic))[1]
    @test bf256.facts.model_precision_bits == 256
    @test bf512.facts.model_precision_bits == 512
    @test bf256.problem.arithmetic.precision_bits == 256
    @test bf512.problem.arithmetic.precision_bits == 512
    @test bf256.facts.model_fingerprint != bf512.facts.model_fingerprint
    @test bf256.transform == train_sdp.provenance.transform
end
