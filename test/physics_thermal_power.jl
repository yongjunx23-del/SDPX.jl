using Test
using LinearAlgebra
using MultiFloats
using SDPX

include(joinpath(
    @__DIR__, "..", "benchmark", "bootstrap", "applications", "thermal_power",
    "GiudiceRenyiPower.jl",
))
using .GiudiceRenyiPower

@testset "Giudice maximum-Renyi power artifacts" begin
    artifacts = Dict{Symbol,Any}()
    for scale in (:tiny, :small, :medium, :stress)
        artifact = build_renyi_power(scale)
        artifacts[scale] = artifact
        verdict = validate_artifact(artifact)
        @test verdict.valid
        @test isempty(verdict.failures)
        @test artifact.spec.reference_status == :build_only
        @test !artifact.spec.paper_equivalent
        @test artifact.provenance.source_version == "2012.12848v2"
        @test artifact.counts.renyi_order == 4
        @test artifact.counts.power_alpha == "1/4"
        @test artifact.counts.power_cones == artifact.spec.energy_levels
        @test artifact.fingerprint == stable_fingerprint(artifact)

        # Independent normalization, energy and Jensen optimum checks.
        levels = artifact.spec.energy_levels
        p = artifact.optimal_probabilities
        @test sum(p) == 1.0
        @test sum(artifact.energies .* p) == artifact.target_mean_energy
        @test artifact.optimal_epigraph == p .^ 4
        @test sum(artifact.optimal_epigraph) == artifact.expected_objective
        @test artifact.expected_objective == 1.0 / levels^3
        @test all(iszero, power_cone_margins(
            artifact, artifact.optimal_epigraph, p,
        ))
        @test all(>(0.0), power_cone_margins(
            artifact, artifact.strict_epigraph, p,
        ))
    end

    @test [artifacts[scale].spec.energy_levels for
           scale in (:tiny, :small, :medium, :stress)] == [8, 32, 128, 512]
    @test [artifacts[scale].counts.canonical_rows for
           scale in (:tiny, :small, :medium, :stress)] == [34, 130, 514, 2050]

    fingerprints = (
        tiny="1b48e5af4d8fcbd33d3fec02cb55efa228a48098a26105a0319da3c6ceae72f3",
        small="8fee8d9cbab80a6c13daf837a4823a7b499f1e2fc2dd5de79935b1d8f569f8f9",
        medium="bae424bda9994b8c2e5b74a4efa6962c9744db47e6b5e9986cadbdc4c5f9d823",
        stress="fb92243239f16b71037f684d750c3dedb70d855d862c3f6b3841100232ea2c6a",
    )
    for scale in keys(fingerprints)
        @test artifacts[scale].fingerprint == getproperty(fingerprints, scale)
    end
    repeated = build_renyi_power(:tiny)
    @test repeated.fingerprint == artifacts[:tiny].fingerprint
    @test canonical_text(repeated) == canonical_text(artifacts[:tiny])

    corrupt = deepcopy(artifacts[:tiny])
    corrupt.energies[2] += 0.5
    corrupt_verdict = validate_artifact(corrupt)
    @test !corrupt_verdict.valid
    @test "energy_semantics" in corrupt_verdict.failures
    @test "fingerprint" in corrupt_verdict.failures

    base = renyi_power_specs().tiny
    @test_throws ArgumentError build_renyi_power(RenyiPowerSpec{Float64}(
        id=base.id,
        scale=base.scale,
        energy_levels=base.energy_levels,
        energy_maximum=base.energy_maximum,
        renyi_order=2,
    ))
    @test_throws ArgumentError build_renyi_power(RenyiPowerSpec{Float64}(
        id=base.id,
        scale=base.scale,
        energy_levels=base.energy_levels,
        energy_maximum=base.energy_maximum,
        source_version="2012.12848v1",
    ))
    @test_throws ArgumentError build_renyi_power(RenyiPowerSpec{Float64}(
        id=base.id,
        scale=base.scale,
        energy_levels=base.energy_levels,
        energy_maximum=base.energy_maximum,
        paper_equivalent=true,
    ))
end

@testset "Giudice native power-cone lowering" begin
    artifact = build_renyi_power(:tiny)
    problem = build_power_problem(artifact)
    levels = artifact.spec.energy_levels
    @test problem isa SDPX.CanonicalConicProgram{Float64}
    @test SDPX.canonical_num_variables(problem) == 2 * levels
    @test SDPX.canonical_num_slack(problem) == 4 * levels + 2
    @test SDPX.layout_barrier_degree(problem.cone_layout) == 4 * levels
    @test problem.c == vcat(zeros(levels), ones(levels))

    blocks = SDPX.layout_blocks(problem.cone_layout)
    @test SDPX.block_cone(first(blocks)) == :nonnegative
    @test SDPX.block_length(first(blocks)) == levels
    power_blocks = filter(block -> SDPX.block_cone(block) == :power, blocks)
    zero_blocks = filter(block -> SDPX.block_cone(block) == :zero, blocks)
    @test length(power_blocks) == levels
    @test length(zero_blocks) == 2
    @test all(block -> SDPX.block_length(block) == 3, power_blocks)
    @test all(block -> SDPX.block_parameter(block) == 0.25, power_blocks)

    # Independently reconstruct the analytic primal point in canonical form.
    x = vcat(artifact.optimal_probabilities, artifact.optimal_epigraph)
    slack = problem.b - problem.A * x
    @test maximum(abs, problem.A * x + slack - problem.b) == 0.0
    @test all(iszero, view(slack, (length(slack) - 1):length(slack)))
    @test SDPX.in_canonical_cone(problem, slack; dual=false)
    @test dot(problem.c, x) == artifact.expected_objective

    # Inspect one cone row triple: slack=(t_i,1,p_i).
    first_power = first(power_blocks)
    rows = SDPX.block_offset(first_power):(
        SDPX.block_offset(first_power) + SDPX.block_length(first_power) - 1
    )
    @test slack[rows] == [
        artifact.optimal_epigraph[1], 1.0, artifact.optimal_probabilities[1],
    ]

    setprecision(BigFloat, 256) do
        big_artifact = build_renyi_power(:tiny, BigFloat)
        @test validate_artifact(big_artifact).valid
        big_problem = build_power_problem(big_artifact)
        @test eltype(big_problem.c) == BigFloat
        @test big_problem.precision_bits == 256
        @test all(block -> SDPX.block_parameter(block) == BigFloat("0.25"),
                  filter(block -> SDPX.block_cone(block) == :power,
                         SDPX.layout_blocks(big_problem.cone_layout)))
    end

    multi_artifact = build_renyi_power(:tiny, MultiFloats.Float64x2)
    @test validate_artifact(multi_artifact).valid
    multi_problem = build_power_problem(multi_artifact)
    @test eltype(multi_problem.c) == MultiFloats.Float64x2
    @test SDPX.canonical_num_slack(multi_problem) == 34
end

@testset "Giudice injected build-only catalog" begin
    if !isdefined(Main, :PhysicsBenchmarkHarness)
        include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "PhysicsBenchmarkHarness.jl"))
    end
    harness = Main.PhysicsBenchmarkHarness
    catalog = harness.load_catalog(joinpath(
        @__DIR__, "..", "benchmark", "bootstrap", "applications", "thermal_power", "catalog.jl",
    ))
    @test catalog.name == :giudice21_renyi_power
    @test length(harness.catalog_entries(catalog, :scaling)) == 3
    entry = only(harness.catalog_entries(catalog, :smoke))
    spec = harness.catalog_spec(catalog, entry.problem_id)
    @test spec.reference.status == :build_only
    @test :native_power in spec.tags
    built = harness.build_problem(catalog, spec, Float64)
    @test built.kind == :power
    @test built.problem isa SDPX.CanonicalConicProgram
    @test built.external_checksum == built.artifact.fingerprint == spec.fingerprint
    @test built.solve_settings.build_only
end
