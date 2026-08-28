using Test
using SHA
using SDPX

include(joinpath(
    @__DIR__, "..", "benchmark", "bootstrap", "applications", "lattice_bootstrap",
    "KZFiniteNLatticeBootstrap.jl",
))
using .KZFiniteNLatticeBootstrap

function _resigned_artifact(
    artifact::LatticeBootstrapArtifact{T};
    variables=artifact.variables,
    equations=artifact.equations,
    gram_blocks=artifact.gram_blocks,
    objective=artifact.objective,
    operator_basis=artifact.operator_basis,
    oracle=artifact.oracle,
    provenance=artifact.provenance,
    counts=artifact.counts,
) where {T}
    provisional = LatticeBootstrapArtifact{T}(
        artifact.schema_version,
        artifact.spec,
        variables,
        equations,
        gram_blocks,
        objective,
        operator_basis,
        oracle,
        provenance,
        counts,
        "",
    )
    return LatticeBootstrapArtifact{T}(
        provisional.schema_version,
        provisional.spec,
        provisional.variables,
        provisional.equations,
        provisional.gram_blocks,
        provisional.objective,
        provisional.operator_basis,
        provisional.oracle,
        provisional.provenance,
        provisional.counts,
        stable_fingerprint(provisional),
    )
end

@testset "KZ finite-N lattice word algebra" begin
    @test isempty(reduce_word(Int8[1, -1, 2, -2]))
    @test isempty(reduce_word(Int8[1, 2, -2, -1]))
    @test isempty(reduce_word(Int8[1, 2, -1, -2]; cyclic=false)) == false
    @test canonical_loop(Int8[1, 2, -1, -2]) ==
          canonical_loop(Int8[2, -1, -2, 1])
    @test canonical_loop(Int8[1, 2, -1, -2]) ==
          canonical_loop(Int8[1, -2, -1, 2])
    @test canonical_loop(Int8[1, 2, -1, -2]) ==
          canonical_loop(inverse_word(LinkWord(Int8[1, 2, -1, -2])))
    @test_throws ArgumentError canonical_loop(Int8[1, 2])
    @test_throws ArgumentError LinkWord(Int8[3])

    plaquette = LinkWord(Int8[1, 2, -1, -2])
    doubled_edge = LinkWord(Int8[1, 2, -1, -2, 1, 2, -1, -2])
    @test is_edge_simple(plaquette)
    @test !is_edge_simple(doubled_edge)
end

@testset "KZ exact D=2 SU(2) oracle" begin
    exact_lambda2 = 0.6580472673593595855628314264880248331
    @test exact_su2_plaquette(2.0) ≈ exact_lambda2 atol=2e-15 rtol=2e-15
    setprecision(BigFloat, 256) do
        value = exact_su2_plaquette(BigFloat(2))
        reference = parse(BigFloat,
            "0.65804726735935958556283142648802483309977015865607")
        @test abs(value - reference) < big"1e-48"
    end
    @test_throws ArgumentError exact_su2_plaquette(0.0)
end

@testset "KZ build-first affine SDP artifacts" begin
    build_times = Dict{Symbol,Float64}()
    artifacts = Dict{Symbol,Any}()
    for scale in (:tiny, :small, :medium)
        elapsed = @elapsed artifacts[scale] = build_lattice_bootstrap(scale)
        build_times[scale] = elapsed
        artifact = artifacts[scale]
        verdict = validate_artifact(artifact)
        @test verdict.valid
        @test isempty(verdict.failures)
        @test artifact.fingerprint == stable_fingerprint(artifact)
        @test length(artifact.fingerprint) == 64
        @test artifact.spec.reference_status === :build_only
        @test artifact.spec.paper_equivalent === false
        @test artifact.spec.publication_claim === :none
        @test artifact.spec.scope === :based_length_edge_simple_subset
        @test artifact.provenance.reference_status === :build_only
        @test artifact.provenance.paper_equivalent === false
        @test artifact.provenance.publication_claim === :none
        @test artifact.provenance.scope === :based_length_edge_simple_subset
        @test all(isfinite, (artifact.oracle.exact_value,))
        @test all(row -> all(isfinite, row.coefficients), artifact.equations)
        @test all(block -> block.entries == permutedims(block.entries), artifact.gram_blocks)
        @test artifact.counts.equation_scope == :edge_simple_Aid_Avar
        @test artifact.counts.hierarchy == :based_length_not_paper_Lambda
    end

    tiny = artifacts[:tiny]
    @test tiny.counts == (
        variables=6,
        equations=1,
        psd_blocks=1,
        block_sizes=(9,),
        operator_basis=9,
        gram_moments=6,
        maximum_moment_length=8,
        equation_scope=:edge_simple_Aid_Avar,
        hierarchy=:based_length_not_paper_Lambda,
        scope=:based_length_edge_simple_subset_not_paper_lambda,
    )
    @test artifacts[:small].counts.variables == 55
    @test artifacts[:small].counts.equations == 3
    @test artifacts[:small].counts.block_sizes == (49,)
    @test artifacts[:medium].counts.variables == 1865
    @test artifacts[:medium].counts.equations == 15
    @test artifacts[:medium].counts.block_sizes == (361,)

    # Independent transcription of KZ Eq. (2.37) at SU(2), D=2, λ=2:
    # w3 - 1 + w4 - w2 + (3λ/4)w1 = 0. Compare each canonical loop word,
    # not merely the coefficient vector emitted by the implementation.
    level_one_row = only(tiny.equations)
    actual_row = Dict(
        tiny.variables[index] => 2 * coefficient
        for (index, coefficient) in
            zip(level_one_row.indices, level_one_row.coefficients)
    )
    expected_eq_2_37 = Dict(
        LinkWord(Int8[-2, -1, -2, 1, 2, -1, 2, 1]) => 1.0, # w4
        LinkWord(Int8[-2, -1, 2, 1]) => 1.5,                # (3λ/4)w1
        LinkWord(Int8[-2, -1, 2, 1, -2, -1, 2, 1]) => 1.0, # w3
        LinkWord(Int8[-2, -2, -1, 2, 2, 1]) => -1.0,       # -w2
    )
    @test actual_row == expected_eq_2_37
    @test 2 * level_one_row.constant == -1.0
    @test occursin("Eq. (2.37)", level_one_row.source)

    # Explicit O₂†O₃ word oracle for one Eq. (3.3) Gram entry. The raw word
    # and both reduced representatives are transcribed independently.
    @test tiny.operator_basis[2] == LinkWord(Int8[-1, -2, 1, 2])
    @test tiny.operator_basis[3] == LinkWord(Int8[-1, 2, 1, -2])
    raw_o2_dagger_o3 = Int8[-2, -1, 2, 1, -1, 2, 1, -2]
    reduced_o2_dagger_o3 = reduce_word(raw_o2_dagger_o3; cyclic=true)
    @test reduced_o2_dagger_o3 == LinkWord(Int8[-2, -1, 2, 2, 1, -2])
    rectangle = LinkWord(Int8[-2, -2, -1, 2, 2, 1])
    @test canonical_loop(reduced_o2_dagger_o3) == rectangle
    rectangle_index = only(findall(==(rectangle), tiny.variables))
    @test tiny.gram_blocks[1].entries[2, 3] ==
          AffineEntry{Float64}(0.0, rectangle_index, 1.0)

    @test artifacts[:tiny].counts.operator_basis < artifacts[:small].counts.operator_basis
    @test artifacts[:small].counts.operator_basis < artifacts[:medium].counts.operator_basis
    @test artifacts[:tiny].counts.variables < artifacts[:small].counts.variables
    @test artifacts[:small].counts.variables < artifacts[:medium].counts.variables

    repeated = build_lattice_bootstrap(:tiny)
    @test repeated.fingerprint == tiny.fingerprint
    @test canonical_text(repeated) == canonical_text(tiny)
    @test tiny.fingerprint ==
          "ca58c9d39cb6cd237325f58df65ec44eab762ae4d2903fa188f7079a2637b855"
    @test artifacts[:small].fingerprint ==
          "4c20e1f0b4da93110e390719f0efc36b7f9187971ae1fa0fd0e750ada7aaa235"
    @test artifacts[:medium].fingerprint ==
          "310af7bf2976e1665a967a5da5190474ef7e153196e2cd30fccd9d06cd211abd"
    @test build_lattice_bootstrap(LatticeBenchmarkSpec{Float64}(
        id="lambda3",
        scale=:tiny,
        coupling=3.0,
        operator_max_length=4,
        equation_max_length=4,
    )).fingerprint != tiny.fingerprint

    corrupt = deepcopy(tiny)
    corrupt.equations[1].indices[1] = 0
    corrupt_verdict = validate_artifact(corrupt)
    @test !corrupt_verdict.valid
    @test "equation_index" in corrupt_verdict.failures
    @test "fingerprint" in corrupt_verdict.failures

    # Re-signing altered metadata must not bypass semantic validation. These
    # checks also prove that counts, provenance, and operator basis are part
    # of the fingerprint rather than unaudited side metadata.
    bad_counts = _resigned_artifact(
        tiny;
        counts=merge(tiny.counts, (variables=tiny.counts.variables + 1,)),
    )
    @test bad_counts.fingerprint != tiny.fingerprint
    @test "counts_semantics" in validate_artifact(bad_counts).failures

    bad_provenance = _resigned_artifact(
        tiny;
        provenance=merge(tiny.provenance, (doi="altered",)),
    )
    @test bad_provenance.fingerprint != tiny.fingerprint
    @test "provenance_semantics" in validate_artifact(bad_provenance).failures

    bad_basis = _resigned_artifact(tiny; operator_basis=reverse(tiny.operator_basis))
    @test bad_basis.fingerprint != tiny.fingerprint
    @test "operator_basis_semantics" in validate_artifact(bad_basis).failures

    problem = build_sdpx_problem(tiny)
    @test problem.dims.m == tiny.counts.variables
    @test problem.dims.n == tiny.counts.equations
    @test problem.dims.k == collect(tiny.counts.block_sizes)

    # SDPX stores each PSD constraint as A(x)-C ⪰ 0. The identity Gram
    # entry therefore requires C₁₁=-1, while G₁₂=w1 has A[w1]₁₂=+1.
    plaquette = LinkWord(Int8[-2, -1, 2, 1])
    plaquette_index = only(findall(==(plaquette), tiny.variables))
    @test problem.C[1][1, 1] == -1.0
    @test -problem.C[1][1, 1] == 1.0
    @test problem.C[1][1, 2] == 0.0
    @test problem.cons.Asp[1][plaquette_index][1, 2] == 1.0
    @test problem.b[1] == 0.5
    @test problem.B[plaquette_index, 1] == 0.75

    setprecision(BigFloat, 256) do
        big_artifact = build_lattice_bootstrap(:tiny, BigFloat)
        @test validate_artifact(big_artifact).valid
        big_problem = build_sdpx_problem(big_artifact)
        @test big_problem.dims == (L=1, m=6, n=1, k=[9])
    end

    @test_throws ArgumentError build_lattice_bootstrap(LatticeBenchmarkSpec{Float64}(
        id="invalid-lambda",
        scale=:invalid,
        coupling=-1.0,
        operator_max_length=4,
        equation_max_length=4,
    ))
    @test_throws ArgumentError build_lattice_bootstrap(LatticeBenchmarkSpec{Float64}(
        id="fake-paper-hierarchy",
        scale=:invalid,
        coupling=2.0,
        operator_max_length=4,
        equation_max_length=4,
        hierarchy=:paper_lambda,
    ))
    @test_throws ArgumentError build_lattice_bootstrap(LatticeBenchmarkSpec{Float64}(
        id="claimed-optimal",
        scale=:tiny,
        coupling=2.0,
        operator_max_length=4,
        equation_max_length=4,
        reference_status=:optimal,
    ))
    @test_throws ArgumentError build_lattice_bootstrap(LatticeBenchmarkSpec{Float64}(
        id="claimed-paper-equivalent",
        scale=:tiny,
        coupling=2.0,
        operator_max_length=4,
        equation_max_length=4,
        paper_equivalent=true,
    ))
    @test_throws ArgumentError build_lattice_bootstrap(LatticeBenchmarkSpec{Float64}(
        id="claimed-publication",
        scale=:tiny,
        coupling=2.0,
        operator_max_length=4,
        equation_max_length=4,
        publication_claim=:reproduced,
    ))

    @info "KZ lattice build evidence" build_times counts=Dict(
        scale => artifacts[scale].counts for scale in keys(artifacts)
    ) fingerprints=Dict(
        scale => artifacts[scale].fingerprint for scale in keys(artifacts)
    )
end

@testset "KZ paper Lambda=3 census fails closed" begin
    census = paper_lambda3_census()
    @test census.loops == 8335
    @test census.equations == 14591
    @test census.free_variables == 1044
    @test census.block_sizes ==
          (54, 52, 46, 45, 98, 45, 46, 52, 53, 98, 30, 27, 21, 24, 16, 11, 8, 12)
    @test !census.independently_reproduced
    @test_throws ArgumentError assert_paper_lambda3_reproduced!()
end

@testset "KZ injected build-only physics catalog" begin
    if !isdefined(Main, :PhysicsBenchmarkHarness)
        include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "PhysicsBenchmarkHarness.jl"))
    end
    harness = Main.PhysicsBenchmarkHarness
    catalog = harness.load_catalog(joinpath(
        @__DIR__, "..", "benchmark", "bootstrap", "applications", "lattice_bootstrap", "catalog.jl",
    ))
    @test catalog.name == :kz25_finite_n_lattice
    @test length(harness.catalog_entries(catalog, :scaling)) == 3
    entry = only(harness.catalog_entries(catalog, :smoke))
    spec = harness.catalog_spec(catalog, entry.problem_id)
    @test spec.reference.status == :build_only
    @test spec.parameters.doi == "10.1007/JHEP03(2025)099"
    @test occursin("based-length edge-simple subset, not paper Lambda", spec.name)
    @test spec.parameters.scope == :based_length_edge_simple_subset_not_paper_lambda
    @test spec.parameters.equation_scope == :edge_simple_Aid_Avar
    @test spec.parameters.reference_status == :build_only
    @test spec.parameters.paper_equivalent === false
    @test spec.parameters.publication_claim == :none
    built = harness.build_problem(catalog, spec, Float64)
    @test built.kind == :sdp
    @test built.problem isa SDPX.SDPProblem
    @test built.external_checksum == built.artifact.fingerprint == spec.fingerprint
    @test built.solve_settings.build_only
end
