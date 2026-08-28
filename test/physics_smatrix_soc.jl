using Test
using SHA
using MultiFloats
using SDPX

include(joinpath(
    @__DIR__, "..", "benchmark", "bootstrap", "applications", "smatrix_soc",
    "PaulosSMatrixSOC.jl",
))
using .PaulosSMatrixSOC

@testset "Paulos rapidity crossing oracle" begin
    theta = asinh(1.0)
    @test crossing_coordinate(theta) ≈ 0.5 - 0.5im atol=2e-15 rtol=2e-15
    complex_theta = complex(theta, 0.23)
    @test crossing_coordinate(complex_theta) ≈
          crossing_coordinate(im * pi - complex_theta) atol=3e-15 rtol=3e-15
    @test crossing_coordinate(-theta) ≈ conj(crossing_coordinate(theta))
end

@testset "Paulos sampled native-Q3 artifacts" begin
    artifacts = Dict{Symbol,Any}()
    for scale in (:tiny, :small, :medium, :stress)
        artifact = build_smatrix_soc(scale)
        artifacts[scale] = artifact
        verdict = validate_artifact(artifact)
        @test verdict.valid
        @test isempty(verdict.failures)
        @test artifact.spec.reference_status == :sampled_build_only
        @test !artifact.spec.paper_equivalent
        @test artifact.provenance.formula_oracle_status == :verified
        @test artifact.counts.unitarity == :sampled_not_continuous
        @test artifact.counts.lorentz_cones == artifact.spec.energy_samples
        @test artifact.counts.cone_dimension == 3
        @test artifact.fingerprint == stable_fingerprint(artifact)
    end

    @test [artifacts[scale].counts.ansatz_degree for
           scale in (:tiny, :small, :medium, :stress)] == [2, 4, 8, 12]
    @test [artifacts[scale].counts.energy_samples for
           scale in (:tiny, :small, :medium, :stress)] == [16, 64, 256, 1024]

    tiny = artifacts[:tiny]
    # Independent reconstruction of every sampled basis value, Re/Im tail,
    # and both analytic witness margins.
    for row in eachindex(tiny.rapidities)
        theta = tiny.rapidities[row]
        sh = (exp(theta) - exp(-theta)) / 2
        z = complex(sh, 0.0) / complex(sh, 1.0)
        for degree in 0:tiny.spec.ansatz_degree
            expected = z^degree
            @test tiny.real_basis[row, degree + 1] ≈ real(expected) atol=2e-15 rtol=2e-15
            @test tiny.imaginary_basis[row, degree + 1] ≈ imag(expected) atol=2e-15 rtol=2e-15
        end
        expected_s = 2 * tiny.spec.external_mass^2 * (1 + (exp(theta) + exp(-theta)) / 2)
        @test tiny.energies_squared[row] ≈ expected_s atol=3e-14 rtol=3e-15
    end
    @test evaluate_amplitude(tiny, tiny.strict_witness) == zeros(ComplexF64, 16)
    @test cone_margins(tiny, tiny.strict_witness) == ones(16)
    @test evaluate_amplitude(tiny, tiny.free_plus_witness) == ones(ComplexF64, 16)
    @test evaluate_amplitude(tiny, tiny.free_minus_witness) == -ones(ComplexF64, 16)
    @test cone_margins(tiny, tiny.free_plus_witness) == zeros(16)
    @test cone_margins(tiny, tiny.free_minus_witness) == zeros(16)

    repeated = build_smatrix_soc(:tiny)
    @test repeated.fingerprint == tiny.fingerprint
    @test canonical_text(repeated) == canonical_text(tiny)
    @test tiny.fingerprint ==
          "124ad9620585b15e75831b3b80a382c0d523af555af00ba3e63ee603287a569a"
    @test artifacts[:small].fingerprint ==
          "372b798bb45ad7240aacec475d42b437fd835c3bc32200f4c9713b74b711283c"
    @test artifacts[:medium].fingerprint ==
          "e8602fae3fdecc0917a5d020a8e45c426446ec2e82492cfe11cbfb938b0e6cee"
    @test artifacts[:stress].fingerprint ==
          "226865a9749e8a5d3851cef2bc4aeaa4a8a263f90dc51cd956fe0f925faf9f36"

    corrupt = deepcopy(tiny)
    corrupt.real_basis[1, 1] = 2.0
    corrupt_verdict = validate_artifact(corrupt)
    @test !corrupt_verdict.valid
    @test "real_basis_semantics" in corrupt_verdict.failures
    @test "fingerprint" in corrupt_verdict.failures

    problem = build_soc_problem(tiny)
    @test problem isa SDPX.ConicProblem{Float64}
    @test problem.variables == tiny.counts.variables
    @test length(problem.cones) == tiny.counts.energy_samples
    probe = [0.2, -0.1, 0.05]
    amplitudes = evaluate_amplitude(tiny, probe)
    for row in eachindex(problem.cones)
        cone = problem.cones[row]
        @test size(cone.A) == (3, 3)
        @test cone.b == [1.0, 0.0, 0.0]
        @test iszero(cone.A[1, :])
        @test cone.A[2, :] == tiny.real_basis[row, :]
        @test cone.A[3, :] == tiny.imaginary_basis[row, :]
        vector = cone.A * probe + cone.b
        @test vector[2] ≈ real(amplitudes[row]) atol=2e-15 rtol=2e-15
        @test vector[3] ≈ imag(amplitudes[row]) atol=2e-15 rtol=2e-15
        @test vector[1] - hypot(vector[2], vector[3]) ≈
              cone_margins(tiny, probe)[row] atol=2e-15 rtol=2e-15
    end

    setprecision(BigFloat, 256) do
        artifact = build_smatrix_soc(:tiny, BigFloat)
        @test validate_artifact(artifact).valid
        problem_big = build_soc_problem(artifact)
        @test eltype(problem_big) == BigFloat
        @test length(problem_big.cones) == 16
    end

    multi_artifact = build_smatrix_soc(:tiny, MultiFloats.Float64x2)
    @test validate_artifact(multi_artifact).valid
    multi_problem = build_soc_problem(multi_artifact)
    @test eltype(multi_problem) == MultiFloats.Float64x2
    @test length(multi_problem.cones) == 16

    base = smatrix_soc_specs().tiny
    @test_throws ArgumentError build_smatrix_soc(SMatrixSOCSpec{Float64}(
        id=base.id,
        scale=base.scale,
        ansatz_degree=base.ansatz_degree,
        energy_samples=base.energy_samples,
        source_version="1607.06110v1",
    ))
    @test_throws ArgumentError build_smatrix_soc(SMatrixSOCSpec{Float64}(
        id=base.id,
        scale=base.scale,
        ansatz_degree=base.ansatz_degree,
        energy_samples=base.energy_samples,
        reference_status=:optimal,
    ))
    @test_throws ArgumentError build_smatrix_soc(SMatrixSOCSpec{Float64}(
        id=base.id,
        scale=base.scale,
        ansatz_degree=base.ansatz_degree,
        energy_samples=base.energy_samples,
        paper_equivalent=true,
    ))
end

@testset "Paulos injected sampled-build-only catalog" begin
    if !isdefined(Main, :PhysicsBenchmarkHarness)
        include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "PhysicsBenchmarkHarness.jl"))
    end
    harness = Main.PhysicsBenchmarkHarness
    catalog = harness.load_catalog(joinpath(
        @__DIR__, "..", "benchmark", "bootstrap", "applications", "smatrix_soc", "catalog.jl",
    ))
    @test catalog.name == :paulos16_sampled_smatrix_soc
    @test length(harness.catalog_entries(catalog, :scaling)) == 3
    entry = only(harness.catalog_entries(catalog, :smoke))
    spec = harness.catalog_spec(catalog, entry.problem_id)
    @test spec.reference.status == :sampled_build_only
    built = harness.build_problem(catalog, spec, Float64)
    @test built.kind == :socp
    @test built.problem isa SDPX.ConicProblem
    @test built.external_checksum == built.artifact.fingerprint == spec.fingerprint
    @test built.solve_settings.build_only
end
