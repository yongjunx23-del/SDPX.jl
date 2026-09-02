using Test
using LinearAlgebra
using SparseArrays
using SDPX

if !isdefined(Main, :PhysicsBenchmarkHarness)
    include(joinpath(@__DIR__, "..", "..", "PhysicsBenchmarkHarness.jl"))
end
include(joinpath(@__DIR__, "MasslessEFT.jl"))
using .MasslessEFT

@testset "massless EFT deterministic reviewed slice" begin
    specs = massless_eft_specs(Float64)
    @test specs.smoke.id != specs.train.id != specs.production.id
    @test specs.smoke.maxN == 2
    @test specs.train.maxN == 6
    @test specs.production.maxN == 14
    @test length(pair_basis(specs.smoke.maxN)) == 4
    @test length(pair_basis(specs.train.maxN)) == 16
    @test length(pair_basis(specs.production.maxN)) == 64

    # The canonical source endpoint and cheap production identity must not
    # inherit the caller's ambient BigFloat precision.
    low_precision_spec = setprecision(BigFloat, 64) do
        massless_eft_specs(BigFloat).production
    end
    high_precision_spec = setprecision(BigFloat, 512) do
        massless_eft_specs(BigFloat).production
    end
    @test precision(low_precision_spec.phi_star) == 1024
    @test low_precision_spec.phi_star == high_precision_spec.phi_star
    @test spec_fingerprint(low_precision_spec) == spec_fingerprint(high_precision_spec)

    manifest_contract = MasslessEFT._manifest_contract()
    @test manifest_contract.source_generator_sha256 == MasslessEFT.SOURCE_GENERATOR_SHA256
    @test manifest_contract.source_auditor_sha256 == MasslessEFT.SOURCE_AUDITOR_SHA256
    @test manifest_contract.external_receipt_json_sha256 == MasslessEFT.SOURCE_RESULT_SHA256
    @test manifest_contract.sdpx_import_base == MasslessEFT.SDPX_IMPORT_BASE
    @test manifest_contract.source_guard_precision_bits == MasslessEFT.SOURCE_GUARD_PRECISION_BITS

    for scale in (:smoke, :train)
        first_artifact = build_massless_eft(scale)
        second_artifact = build_massless_eft(scale)
        @test first_artifact.fingerprint == second_artifact.fingerprint
        @test canonical_text(first_artifact) == canonical_text(second_artifact)
        @test validate_artifact(first_artifact).valid
        @test size(first_artifact.real_rows, 2) == 1 + length(first_artifact.pairs)
        @test size(first_artifact.real_rows, 1) == first_artifact.counts.original_cones
        @test first_artifact.counts.cone_dimension == 3
        @test first_artifact.pairs == pair_basis(first_artifact.spec.maxN)
        @test first_artifact.cone_rhs == [1.0, 0.0, 1.0]
        @test first_artifact.g0_map[1] == -3.0
        @test first_artifact.g0_map[2] == 3.0
        @test first_artifact.g2_map[1] == -3.0 / 8
        @test first_artifact.g2_map[3] == 1.0 / 2
        @test first_artifact.g2_map[4] == 1.0 / 4
        @test first_artifact.g2_map[5] == -1.0 / 32
        @test first_artifact.provenance.source_generator_sha256 == MasslessEFT.SOURCE_GENERATOR_SHA256
        @test first_artifact.provenance.source_auditor_sha256 == MasslessEFT.SOURCE_AUDITOR_SHA256
        @test first_artifact.provenance.external_receipt_json_sha256 == MasslessEFT.SOURCE_RESULT_SHA256
        @test first_artifact.provenance.manifest_sha256 == manifest_sha256()
        @test !first_artifact.witness_certified
        @test first_artifact.spec.precision_bits == 1024
        @test first_artifact.spec.heldout_ngrid == 2 * first_artifact.spec.ngrid - 1
        @test first_artifact.spec.normalization == :physical_factor_four_unresolved
        parity = generator_parity_gate(first_artifact.spec, first_artifact.phis)
        @test parity.passed

        sample_coefficients = [0.2, -0.1, 0.03, 0.07, -0.04]
        sample_coefficients = vcat(sample_coefficients,
            zeros(Float64, first_artifact.counts.variables - length(sample_coefficients)))
        row = 1
        tau = evaluate_amplitude(first_artifact, sample_coefficients)[row]
        direct_disk = 1 - (real(tau)^2 + (1 - imag(tau))^2)
        psd_determinant = (2 - imag(tau)) * imag(tau) - real(tau)^2
        @test direct_disk == psd_determinant

        # Mutating a persisted coefficient without updating its fingerprint is
        # fail-closed; this also guards the row payload's fingerprint coverage.
        mutated_rows = copy(first_artifact.real_rows)
        mutated_rows[1, 1] += 1.0
        mutated = MasslessEFTArtifact(first_artifact.schema_version, first_artifact.spec,
            first_artifact.phis, first_artifact.heldout_phis, first_artifact.spins,
            first_artifact.pairs, mutated_rows, first_artifact.imag_rows,
            first_artifact.cone_rhs, first_artifact.g0_map, first_artifact.g2_map,
            first_artifact.objective_maps, first_artifact.witness_candidate,
            first_artifact.witness_certified, first_artifact.provenance,
            first_artifact.counts, first_artifact.fingerprint)
        @test !validate_artifact(mutated; rebuild=false).valid

        # Enforced and independently regenerated held-out audits use the exact
        # same disk excess expression and report endpoints as uncertified.
        zero_coefficients = zeros(Float64, first_artifact.counts.variables)
        @test audit_enforced(first_artifact, zero_coefficients).max_positive_excess == 0.0
        heldout = audit_heldout(first_artifact, zero_coefficients)
        @test heldout.heldout_indices == collect(2:2:(first_artifact.spec.heldout_ngrid - 1))
        @test heldout.endpoint_limits.phi_zero == :not_represented
        @test heldout.policy == :diagnostic_only_no_declared_threshold
    end
end

@testset "massless EFT Model and compact SOC parity" begin
    artifact = build_massless_eft(:smoke)
    compact = build_soc_problem(artifact, :none)
    model = build_model(artifact, :none)
    sdp_model = build_sdp_model(artifact, :none)
    @test prove_representation_parity(artifact).valid
    @test length(sdp_model.constraint_blocks) == length(compact.cones)
    for objective in (:none, :min_g0, :max_g0)
        @test build_model(artifact, objective).objective !== nothing
        @test build_sdp_model(artifact, objective).objective !== nothing
    end
    @test_throws ArgumentError build_soc_problem(artifact, :g2)
    @test_throws ArgumentError build_model(artifact, :g2)
    @test compact.variables == length(artifact.g0_map)
    @test length(compact.cones) == size(artifact.real_rows, 1)
    @test length(model.variables) == compact.variables
    @test length(model.constraint_blocks) == length(compact.cones)
    for row in axes(artifact.real_rows, 1)
        block = model.constraint_blocks[row]
        @test block.shape == 3
        for coordinate in 1:3
            expression = block.expressions[coordinate]
            expected = coordinate == 1 ? zeros(Float64, compact.variables) :
                (coordinate == 2 ? artifact.real_rows[row, :] : -artifact.imag_rows[row, :])
            actual = zeros(Float64, compact.variables)
            actual[expression.indices] .= expression.coefficients
            @test actual == expected
            @test expression.constant == artifact.cone_rhs[coordinate]
            @test compact.cones[row].A[coordinate, :] == expected
            @test compact.cones[row].b[coordinate] == expression.constant
        end
    end

    # Test-only inspection of private canonical IR: shared EFT tails make the
    # local two-active-variable fixed-trace specialization inapplicable.
    # Add a mathematically inert ZeroCone row solely to exercise the
    # specialization's applicability gate; every SOC tail still shares the
    # global EFT variables, so disjoint-active-variable detection rejects it.
    SDPX.constraint!(model, :test_zero_row, 0.0, SDPX.ZeroCone())
    native = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(native)
    @test SDPX.fixed_trace_q3_canonical_plan(canonical) === nothing
end

@testset "massless EFT claim boundary and catalog" begin
    # The catalog loader itself is exercised by the harness test process; this
    # direct include keeps this focused test independent of external arrays.
    include(joinpath(@__DIR__, "catalog.jl"))
    catalog = physics_benchmark_catalog()
    @test Set(keys(catalog.suites)) == Set((:smoke, :train))
    @test all(spec.reference.status === :sampled_build_only for spec in values(catalog.specs))
    @test all(:build_only in spec.tags for spec in values(catalog.specs))
    @test all(spec.reference.objective === nothing for spec in values(catalog.specs) if !endswith(spec.id, "production_N14_L60_grid300"))
    @test catalog.specs["massless_eft/production_N14_L60_grid300"].reference.objective.independent === false
    @test !haskey(catalog.suites, :production)
    @test all(entry.provider === :auto for entries in values(catalog.suites) for entry in entries)
    @test all(spec.fingerprint == spec_fingerprint(getproperty(massless_eft_specs(Float64), Main._spec_for_id(spec.id)), manifest_sha256()) for spec in values(catalog.specs))
    smoke_spec = catalog.specs["massless_eft/smoke_N2_L4_grid9"]
    built = Main.PhysicsBenchmarkHarness.build_problem(catalog, smoke_spec, Float64)
    @test built.solve_settings.build_only === true
    @test Main.PhysicsBenchmarkHarness.validate_result(catalog, smoke_spec, built, nothing, nothing) == []

    loaded_one = Main.PhysicsBenchmarkHarness.load_catalog(joinpath(@__DIR__, "catalog.jl"))
    loaded_two = Main.PhysicsBenchmarkHarness.load_catalog(joinpath(@__DIR__, "catalog.jl"))
    @test loaded_one.name == loaded_two.name == :massless_eft_pole_augmented
    @test sort(collect(keys(loaded_one.specs))) == sort(collect(keys(loaded_two.specs)))
    @test [s.fingerprint for s in values(loaded_one.specs)] == [s.fingerprint for s in values(loaded_two.specs)]
    @test all(s.parameters.manifest_sha256 == loaded_one.specs["massless_eft/smoke_N2_L4_grid9"].parameters.manifest_sha256 for s in values(loaded_one.specs))
end
