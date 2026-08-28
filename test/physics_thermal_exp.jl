using Test
using LinearAlgebra
using MultiFloats
using SDPX

include(joinpath(
    @__DIR__, "..", "benchmark", "bootstrap", "applications", "thermal_exp",
    "GibbsRelativeEntropyEXP.jl",
))
import .GibbsRelativeEntropyEXP
using .GibbsRelativeEntropyEXP: GibbsRelativeEntropySpec
using .GibbsRelativeEntropyEXP: gibbs_relative_entropy_specs
using .GibbsRelativeEntropyEXP: build_gibbs_relative_entropy, build_exp_problem
using .GibbsRelativeEntropyEXP: exp_cone_margins

function _left_fold_sum(values)
    total = zero(eltype(values))
    for value in values
        total += value
    end
    return total
end

function _independent_gibbs_oracle(::Type{T}, levels::Int) where {T}
    energies = T[T(index - 1) / T(levels - 1) for index in 1:levels]
    weights = T[exp(-T(2) * energy) for energy in energies]
    partition = _left_fold_sum(weights)
    raw_probabilities = weights ./ partition
    return energies, weights, partition, raw_probabilities
end

function _relative_entropy(p, q)
    total = zero(eltype(p))
    for index in eachindex(p)
        total += p[index] * log(p[index] / q[index])
    end
    return total
end

@testset "finite-level Gibbs EXP artifacts" begin
    artifacts = Dict{Symbol,Any}()
    expected_fingerprints = (
        tiny="34ecf0cfea71bc94c6e48eb0c574281b415465c08b527584c89a5098557ee110",
        small="1b4916ab808151295ecf292bbd7314b1408d5b5e85f0c82335624948128bb980",
        medium="7e3927d7598bf4a489b91e75c225ad4e84f50640e87ecee10f37a7c5588b1745",
        stress="76a9a00f8714734ed05e8f75d391f3d7ca81ba4ce2196a17f6411bc7558d8b7d",
    )
    for scale in (:tiny, :small, :medium, :stress)
        artifact = build_gibbs_relative_entropy(scale)
        artifacts[scale] = artifact
        verdict = GibbsRelativeEntropyEXP.validate_artifact(artifact)
        @test verdict.valid
        @test isempty(verdict.failures)
        @test artifact.spec.reference_status === :build_only
        @test artifact.spec.source === :jaynes_kullback_leibler_primary
        @test artifact.spec.source_version == "PhysRev.106.620+AnnMathStat.22.79"
        @test !artifact.spec.paper_equivalent
        @test !artifact.provenance.paper_equivalent
        @test artifact.provenance.reference_status === :build_only
        @test artifact.counts.exponential_cones == artifact.spec.energy_levels
        @test artifact.counts.equality_rows == 1
        @test artifact.counts.canonical_rows == 3 * artifact.spec.energy_levels + 1
        @test artifact.counts.barrier_degree == 3 * artifact.spec.energy_levels
        @test artifact.fingerprint == GibbsRelativeEntropyEXP.stable_fingerprint(artifact)
        @test artifact.fingerprint == getproperty(expected_fingerprints, scale)

        # Independent finite-level Gibbs construction. The final coordinate
        # is deliberately closed in model arithmetic; all earlier coordinates
        # are the direct normalized weights and the stored correction accounts
        # exactly for the last one.
        levels = artifact.spec.energy_levels
        energies, weights, partition, raw_q =
            _independent_gibbs_oracle(Float64, levels)
        @test artifact.energies == energies
        @test artifact.boltzmann_weights == weights
        @test artifact.partition_function == partition
        @test artifact.gibbs_probabilities[1:(end - 1)] == raw_q[1:(end - 1)]
        @test artifact.gibbs_probabilities[end] - raw_q[end] ==
              artifact.normalization_correction
        @test _left_fold_sum(artifact.gibbs_probabilities) == 1.0
        @test all(>(0.0), artifact.gibbs_probabilities)

        # Boundary optimum and a scale-independent strict interior witness.
        boundary_margin = exp_cone_margins(
            artifact,
            artifact.optimal_epigraph,
            artifact.gibbs_probabilities,
        )
        strict_margin = exp_cone_margins(
            artifact,
            artifact.strict_epigraph,
            artifact.gibbs_probabilities,
        )
        @test all(iszero, boundary_margin)
        @test all(>(0.0), strict_margin)
        @test strict_margin ≈
              artifact.gibbs_probabilities .* (1.0 - exp(-1.0)) rtol=2e-15
        @test all(iszero, artifact.optimal_epigraph)
        @test artifact.strict_epigraph == artifact.gibbs_probabilities
        @test iszero(artifact.expected_objective)

        # A separate non-Gibbs normalized state has positive relative entropy;
        # its exact epigraph values land back on the exponential-cone boundary.
        perturbed = copy(artifact.gibbs_probabilities)
        delta = min(perturbed[1], perturbed[2]) / 4
        perturbed[1] += delta
        perturbed[2] -= delta
        t_perturbed = perturbed .* log.(perturbed ./ artifact.gibbs_probabilities)
        @test _left_fold_sum(perturbed) == 1.0
        @test _relative_entropy(perturbed, artifact.gibbs_probabilities) > 0.0
        @test maximum(abs, exp_cone_margins(
            artifact, t_perturbed, perturbed,
        )) <= 4eps(Float64)
    end

    @test [artifacts[scale].spec.energy_levels for
           scale in (:tiny, :small, :medium, :stress)] == [8, 32, 128, 512]
    @test [artifacts[scale].counts.canonical_rows for
           scale in (:tiny, :small, :medium, :stress)] == [25, 97, 385, 1537]

    repeated = build_gibbs_relative_entropy(:tiny)
    @test GibbsRelativeEntropyEXP.canonical_text(repeated) ==
          GibbsRelativeEntropyEXP.canonical_text(artifacts[:tiny])
    @test repeated.fingerprint == artifacts[:tiny].fingerprint
    @test occursin(
        "10.1103/PhysRev.106.620",
        GibbsRelativeEntropyEXP.canonical_text(repeated),
    )
    @test occursin(
        "10.1214/aoms/1177729694",
        GibbsRelativeEntropyEXP.canonical_text(repeated),
    )

    # Fingerprinting alone is not the validator: an independently rebuilt
    # semantic field must also catch a mutation.
    corrupt = deepcopy(artifacts[:tiny])
    corrupt.gibbs_probabilities[1] += 0.125
    corrupt_verdict = GibbsRelativeEntropyEXP.validate_artifact(corrupt)
    @test !corrupt_verdict.valid
    @test "probability_semantics" in corrupt_verdict.failures
    @test "fingerprint" in corrupt_verdict.failures

    truncated = deepcopy(artifacts[:tiny])
    pop!(truncated.strict_epigraph)
    truncated_verdict = GibbsRelativeEntropyEXP.validate_artifact(truncated)
    @test !truncated_verdict.valid
    @test "strict_epigraph_length" in truncated_verdict.failures
    @test "strict_epigraph_semantics" in truncated_verdict.failures
    @test "fingerprint_encoding" in truncated_verdict.failures

    base = gibbs_relative_entropy_specs().tiny
    _thermal_exp_replace_spec(; id=base.id, scale=base.scale,
                              energy_levels=base.energy_levels,
                              energy_minimum=base.energy_minimum,
                              energy_maximum=base.energy_maximum,
                              inverse_temperature=base.inverse_temperature,
                              source=base.source, source_version=base.source_version,
                              reference_status=base.reference_status,
                              paper_equivalent=base.paper_equivalent) =
        GibbsRelativeEntropySpec{Float64}(
            id=id,
            scale=scale,
            energy_levels=energy_levels,
            energy_minimum=energy_minimum,
            energy_maximum=energy_maximum,
            inverse_temperature=inverse_temperature,
            source=source,
            source_version=source_version,
            reference_status=reference_status,
            paper_equivalent=paper_equivalent,
        )
    @test_throws ArgumentError build_gibbs_relative_entropy(_thermal_exp_replace_spec(
        source=:secondary_summary,
    ))
    @test_throws ArgumentError build_gibbs_relative_entropy(_thermal_exp_replace_spec(
        source_version="unreviewed",
    ))
    @test_throws ArgumentError build_gibbs_relative_entropy(_thermal_exp_replace_spec(
        reference_status=:optimal,
    ))
    @test_throws ArgumentError build_gibbs_relative_entropy(_thermal_exp_replace_spec(
        paper_equivalent=true,
    ))
    @test_throws ArgumentError build_gibbs_relative_entropy(_thermal_exp_replace_spec(
        inverse_temperature=1.0,
    ))
    @test_throws ArgumentError build_gibbs_relative_entropy(_thermal_exp_replace_spec(
        energy_levels=7,
    ))
    @test_throws ArgumentError build_gibbs_relative_entropy(_thermal_exp_replace_spec(
        id="wrong/id",
    ))
end

@testset "native exponential-cone canonical lowering" begin
    artifact = build_gibbs_relative_entropy(:tiny)
    problem = build_exp_problem(artifact)
    levels = artifact.spec.energy_levels
    @test problem isa SDPX.CanonicalConicProgram{Float64}
    @test SDPX.canonical_num_variables(problem) == 2 * levels
    @test SDPX.canonical_num_slack(problem) == 3 * levels + 1
    @test SDPX.layout_barrier_degree(problem.cone_layout) == 3 * levels
    @test problem.c == vcat(zeros(levels), ones(levels))

    blocks = SDPX.layout_blocks(problem.cone_layout)
    exp_blocks = filter(block -> SDPX.block_cone(block) == :exp, blocks)
    zero_blocks = filter(block -> SDPX.block_cone(block) == :zero, blocks)
    @test length(exp_blocks) == levels
    @test length(zero_blocks) == 1
    @test all(block -> SDPX.block_length(block) == 3, exp_blocks)
    @test SDPX.block_length(only(zero_blocks)) == 1

    # Independent coefficient/sign audit of b-A*x=(-t_i,p_i,q_i).
    dense_A = Matrix(problem.A)
    for (index, block) in enumerate(exp_blocks)
        row = SDPX.block_offset(block)
        expected_first = zeros(2 * levels)
        expected_second = zeros(2 * levels)
        expected_first[levels + index] = 1.0
        expected_second[index] = -1.0
        @test dense_A[row, :] == expected_first
        @test dense_A[row + 1, :] == expected_second
        @test all(iszero, dense_A[row + 2, :])
        @test problem.b[row] == 0.0
        @test problem.b[row + 1] == 0.0
        @test problem.b[row + 2] == artifact.gibbs_probabilities[index]
    end
    zero_row = SDPX.block_offset(only(zero_blocks))
    @test dense_A[zero_row, 1:levels] == fill(-1.0, levels)
    @test all(iszero, dense_A[zero_row, (levels + 1):(2 * levels)])
    @test problem.b[zero_row] == -1.0

    # Reconstruct both analytic primal witnesses in the frozen canonical form.
    boundary_x = vcat(
        artifact.gibbs_probabilities,
        artifact.optimal_epigraph,
    )
    boundary_slack = problem.b - problem.A * boundary_x
    @test maximum(abs, problem.A * boundary_x + boundary_slack - problem.b) == 0.0
    @test boundary_slack[zero_row] == 0.0
    @test SDPX.in_canonical_cone(problem, boundary_slack; dual=false)
    @test dot(problem.c, boundary_x) == artifact.expected_objective == 0.0
    for (index, block) in enumerate(exp_blocks)
        row = SDPX.block_offset(block)
        @test boundary_slack[row:(row + 2)] == [
            0.0,
            artifact.gibbs_probabilities[index],
            artifact.gibbs_probabilities[index],
        ]
    end

    strict_x = vcat(
        artifact.gibbs_probabilities,
        artifact.strict_epigraph,
    )
    strict_slack = problem.b - problem.A * strict_x
    @test maximum(abs, problem.A * strict_x + strict_slack - problem.b) == 0.0
    @test strict_slack[zero_row] == 0.0
    @test SDPX.in_canonical_cone(problem, strict_slack; dual=false)
    for block in exp_blocks
        row = SDPX.block_offset(block)
        @test SDPX.exp_primal_interior(
            strict_slack[row], strict_slack[row + 1], strict_slack[row + 2],
        )
    end

    @test_throws DimensionMismatch exp_cone_margins(
        artifact, zeros(levels - 1), artifact.gibbs_probabilities,
    )
    bad_probability = copy(artifact.gibbs_probabilities)
    bad_probability[1] = 0.0
    @test_throws ArgumentError exp_cone_margins(
        artifact, artifact.optimal_epigraph, bad_probability,
    )
end

@testset "Gibbs EXP arithmetic preservation" begin
    setprecision(BigFloat, 256) do
        artifact = build_gibbs_relative_entropy(:tiny, BigFloat)
        verdict = GibbsRelativeEntropyEXP.validate_artifact(artifact)
        @test verdict.valid
        @test isempty(verdict.failures)
        @test precision(artifact.partition_function) == 256
        @test _left_fold_sum(artifact.gibbs_probabilities) == one(BigFloat)
        @test all(>(zero(BigFloat)), exp_cone_margins(
            artifact, artifact.strict_epigraph, artifact.gibbs_probabilities,
        ))
        problem = build_exp_problem(artifact)
        @test eltype(problem.c) == BigFloat
        @test problem.precision_bits == 256
        @test SDPX.canonical_num_slack(problem) == 25
        boundary_x = vcat(
            artifact.gibbs_probabilities, artifact.optimal_epigraph,
        )
        boundary_slack = problem.b - problem.A * boundary_x
        @test iszero(boundary_slack[end])
        @test SDPX.in_canonical_cone(problem, boundary_slack; dual=false)
    end

    T = MultiFloats.Float64x2
    artifact = build_gibbs_relative_entropy(:tiny, T)
    verdict = GibbsRelativeEntropyEXP.validate_artifact(artifact)
    @test verdict.valid
    @test isempty(verdict.failures)
    @test eltype(artifact.gibbs_probabilities) == T
    @test _left_fold_sum(artifact.gibbs_probabilities) == one(T)
    @test all(>(zero(T)), exp_cone_margins(
        artifact, artifact.strict_epigraph, artifact.gibbs_probabilities,
    ))
    problem = build_exp_problem(artifact)
    @test eltype(problem.c) == T
    @test SDPX.canonical_num_slack(problem) == 25
    @test length(filter(
        block -> SDPX.block_cone(block) == :exp,
        SDPX.layout_blocks(problem.cone_layout),
    )) == 8
end

@testset "injected Gibbs EXP build-only catalog" begin
    if !isdefined(Main, :PhysicsBenchmarkHarness)
        include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "PhysicsBenchmarkHarness.jl"))
    end
    harness = Main.PhysicsBenchmarkHarness
    catalog_path = joinpath(
        @__DIR__, "..", "benchmark", "bootstrap", "applications", "thermal_exp", "catalog.jl",
    )
    catalog = harness.load_catalog(catalog_path)
    @test catalog.name === :finite_gibbs_relative_entropy_exp
    @test catalog.version == "1"
    @test length(harness.catalog_entries(catalog, :smoke)) == 1
    @test length(harness.catalog_entries(catalog, :scaling)) == 3
    @test length(harness.catalog_entries(catalog, :stress)) == 1

    all_entries = vcat(
        harness.catalog_entries(catalog, :scaling),
        harness.catalog_entries(catalog, :stress),
    )
    @test [harness.catalog_spec(catalog, entry.problem_id).parameters.energy_levels
           for entry in all_entries] == [8, 32, 128, 512]
    catalog_builds = Dict{String,Any}()
    for entry in all_entries
        spec = harness.catalog_spec(catalog, entry.problem_id)
        @test entry.arithmetic === :float64
        @test spec.family === :exp
        @test spec.problem_type === :exponential_cone_program
        @test spec.reference.status === :build_only
        @test spec.reference.objective === nothing
        @test spec.parameters.paper_equivalent === false
        @test :native_exp in spec.tags
        @test :build_only in spec.tags
        built_scale = harness.build_problem(catalog, spec, Float64)
        catalog_builds[entry.problem_id] = built_scale
        @test built_scale.problem isa SDPX.CanonicalConicProgram{Float64}
        @test built_scale.kind === :exp
        @test SDPX.canonical_num_slack(built_scale.problem) ==
              3 * spec.parameters.energy_levels + 1
        @test length(filter(
            block -> SDPX.block_cone(block) == :exp,
            SDPX.layout_blocks(built_scale.problem.cone_layout),
        )) == spec.parameters.energy_levels
        @test isempty(harness.validate_result(
            catalog, spec, built_scale, nothing, (;),
        ))
    end

    entry = only(harness.catalog_entries(catalog, :smoke))
    spec = harness.catalog_spec(catalog, entry.problem_id)
    built = catalog_builds[entry.problem_id]
    @test built.kind === :exp
    @test built.problem isa SDPX.CanonicalConicProgram{Float64}
    @test built.expected === nothing
    @test built.solve_settings == (build_only=true,)
    @test built.artifact.spec.reference_status === :build_only
    @test !built.artifact.spec.paper_equivalent
    @test built.external_checksum == built.artifact.fingerprint == spec.fingerprint
    @test isempty(harness.validate_result(
        catalog, spec, built, nothing, (;),
    ))

    # Build-path audit: neither the artifact implementation nor its injected
    # catalog contains a public solve call or a provider-dispatch call.
    implementation_text = read(joinpath(
        dirname(catalog_path), "GibbsRelativeEntropyEXP.jl",
    ), String)
    catalog_text = read(catalog_path, String)
    for forbidden in ("SDPX.optimize!", "SDPX.solve", "run_provider", "provider_solve")
        @test !occursin(forbidden, implementation_text)
        @test !occursin(forbidden, catalog_text)
    end
end
