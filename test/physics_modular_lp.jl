using Test
using SHA
using SparseArrays
using MultiFloats
using SDPX

include(joinpath(
    @__DIR__, "..", "benchmark", "physics", "modular_lp",
    "HellermanModularLP.jl",
))
using .HellermanModularLP

@testset "Hellerman fixed-point polynomial oracle" begin
    # Independent transcription of paper Eqs. (3.23)-(3.24).
    setprecision(BigFloat, 256) do
        z = BigFloat("0.375")
        pi_b = BigFloat(pi)
        r20 = -inv(BigFloat(16)) + sum(
            pi_b^2 * BigFloat(n)^2 / sinh(pi_b * BigFloat(n))^2
            for n in 1:64
        )
        paper_f1(x) = BigFloat(2) * BigFloat(pi) * x - inv(BigFloat(2))
        function paper_f3(x)
            y = BigFloat(2) * BigFloat(pi) * x
            return y^3 - BigFloat(9) / BigFloat(2) * y^2 +
                   (BigFloat(41) / BigFloat(8) + BigFloat(6) * r20) * y -
                   (BigFloat(17) / BigFloat(16) + BigFloat(3) * r20)
        end
        @test abs(derivative_polynomial(1, z) - paper_f1(z)) < big"1e-70"
        @test abs(derivative_polynomial(3, z) - paper_f3(z)) < big"1e-70"

        # Independent Eq. (3.28) vacuum combination at c_total=4.
        ehat0 = -inv(BigFloat(12))
        q = exp(-BigFloat(2) * BigFloat(pi))
        expected_b1 = paper_f1(ehat0) -
                      BigFloat(2) * q * paper_f1(ehat0 + 1) +
                      q^2 * paper_f1(ehat0 + 2)
        expected_b3 = paper_f3(ehat0) -
                      BigFloat(2) * q * paper_f3(ehat0 + 1) +
                      q^2 * paper_f3(ehat0 + 2)
        @test abs(vacuum_polynomial(1, ehat0) - expected_b1) < big"1e-70"
        @test abs(vacuum_polynomial(3, ehat0) - expected_b3) < big"1e-70"
    end
end

@testset "Hellerman finite-grid LP artifacts" begin
    artifacts = Dict{Symbol,Any}()
    for scale in (:tiny, :small, :medium, :stress)
        artifact = build_modular_lp(scale)
        artifacts[scale] = artifact
        verdict = validate_artifact(artifact)
        @test verdict.valid
        @test isempty(verdict.failures)
        @test artifact.spec.reference_status == :build_only
        @test !artifact.spec.paper_equivalent
        @test artifact.provenance.formula_oracle_status == :verified
        @test artifact.counts.discretization == :finite_dimension_grid_not_continuum
        @test artifact.fingerprint == stable_fingerprint(artifact)
        @test all(isfinite, artifact.coefficients)
        @test all(isfinite, artifact.rhs)
    end

    @test artifacts[:tiny].counts == (
        variables=16,
        nonnegative_rows=16,
        equality_rows=1,
        derivative_orders=(1,),
        dimension_points=16,
        eta_product_terms=64,
        discretization=:finite_dimension_grid_not_continuum,
        paper_equivalent=false,
    )
    @test artifacts[:small].counts.derivative_orders == (1, 3)
    @test artifacts[:medium].counts.derivative_orders == (1, 3, 5)
    @test artifacts[:stress].counts.derivative_orders == (1, 3, 5, 7)
    @test [artifacts[scale].counts.variables for scale in (:tiny, :small, :medium, :stress)] ==
          [16, 32, 64, 128]

    tiny = artifacts[:tiny]
    ehat0 = -1 / 12
    for (column, dimension) in pairs(tiny.dimensions)
        # Independent p=1 coefficient from Eq. (3.23), including Boltzmann weight.
        expected = (2pi * (dimension + ehat0) - 0.5) * exp(-2pi * dimension)
        @test tiny.coefficients[1, column] ≈ expected atol=3e-15 rtol=3e-15
    end
    q = exp(-2pi)
    f1(x) = 2pi * x - 0.5
    expected_rhs = -(f1(ehat0) - 2q * f1(ehat0 + 1) + q^2 * f1(ehat0 + 2))
    @test tiny.rhs[1] ≈ expected_rhs atol=3e-15 rtol=3e-15

    repeated = build_modular_lp(:tiny)
    @test repeated.fingerprint == tiny.fingerprint
    @test canonical_text(repeated) == canonical_text(tiny)
    @test tiny.fingerprint ==
          "bcef79cea6d8373be66f550ea3abd0a97e440c41adedea48f18abbaf6cb0e77c"
    @test artifacts[:small].fingerprint ==
          "bb2d3b576671580b97510c35dc3f0210211bef7b0f0d576cb4ed6189fda81e0c"
    @test artifacts[:medium].fingerprint ==
          "b6a9c1df9dd09ab19100d5157f4030cf05da46b614dc3e2c7a5b01c134a4ffdd"
    @test artifacts[:stress].fingerprint ==
          "a19af30e56a56a9f849a61c678a717ab795f53bda7ed0e036ebbb0de585d8002"

    corrupt = deepcopy(tiny)
    corrupt.coefficients[1, 1] += 1.0
    corrupt_verdict = validate_artifact(corrupt)
    @test !corrupt_verdict.valid
    @test "coefficient_semantics" in corrupt_verdict.failures
    @test "fingerprint" in corrupt_verdict.failures

    problem = build_lp_problem(tiny)
    @test SDPX.classify_problem(problem).cone == :lp
    @test problem.dims.m == 16
    @test problem.dims.n == 1
    @test problem.dims.k == ones(Int, 16)
    # Public LP convention is G*x >= h. Here G=I and h=0, so each scalar
    # A(x)-C block is exactly x_j >= 0; this sign is checked explicitly.
    for variable in 1:16
        @test problem.C[variable][1, 1] == 0.0
        @test problem.cons.Asp[variable][variable][1, 1] == 1.0
    end
    @test Matrix(problem.B) == permutedims(tiny.coefficients)
    @test problem.b == tiny.rhs
    @test iszero(problem.c)

    setprecision(BigFloat, 256) do
        artifact = build_modular_lp(:tiny, BigFloat)
        @test validate_artifact(artifact).valid
        problem_big = build_lp_problem(artifact)
        @test eltype(problem_big) == BigFloat
        @test SDPX.classify_problem(problem_big).cone == :lp
    end

    multi_artifact = build_modular_lp(:tiny, MultiFloats.Float64x2)
    @test validate_artifact(multi_artifact).valid
    multi_problem = build_lp_problem(multi_artifact)
    @test eltype(multi_problem) == MultiFloats.Float64x2
    @test SDPX.classify_problem(multi_problem).cone == :lp

    base = modular_lp_specs().tiny
    @test_throws ArgumentError build_modular_lp(ModularLPSpec{Float64}(
        id=base.id,
        scale=base.scale,
        maximum_derivative_order=base.maximum_derivative_order,
        dimension_points=base.dimension_points,
        dimension_minimum=base.dimension_minimum,
        dimension_maximum=base.dimension_maximum,
        source_version="0902.2790v1",
    ))
    @test_throws ArgumentError build_modular_lp(ModularLPSpec{Float64}(
        id=base.id,
        scale=base.scale,
        maximum_derivative_order=base.maximum_derivative_order,
        dimension_points=base.dimension_points,
        dimension_minimum=base.dimension_minimum,
        dimension_maximum=base.dimension_maximum,
        reference_status=:optimal,
    ))
    @test_throws ArgumentError build_modular_lp(ModularLPSpec{Float64}(
        id=base.id,
        scale=base.scale,
        maximum_derivative_order=base.maximum_derivative_order,
        dimension_points=base.dimension_points,
        dimension_minimum=base.dimension_minimum,
        dimension_maximum=base.dimension_maximum,
        paper_equivalent=true,
    ))
end

@testset "Hellerman injected build-only catalog" begin
    if !isdefined(Main, :PhysicsBenchmarkHarness)
        include(joinpath(@__DIR__, "..", "benchmark", "PhysicsBenchmarkHarness.jl"))
    end
    harness = Main.PhysicsBenchmarkHarness
    catalog = harness.load_catalog(joinpath(
        @__DIR__, "..", "benchmark", "physics", "modular_lp", "catalog.jl",
    ))
    @test catalog.name == :hellerman09_modular_lp
    @test length(harness.catalog_entries(catalog, :scaling)) == 3
    entry = only(harness.catalog_entries(catalog, :smoke))
    spec = harness.catalog_spec(catalog, entry.problem_id)
    @test spec.reference.status == :build_only
    built = harness.build_problem(catalog, spec, Float64)
    @test built.kind == :lp
    @test built.problem isa SDPX.SDPProblem
    @test built.external_checksum == built.artifact.fingerprint == spec.fingerprint
    @test built.solve_settings.build_only
end
