# Exact Hellerman modular functional PMP front-end and fail-closed contract tests.

@testset "Original Hellerman modular functional PMP front end" begin
    modular_path = joinpath(@__DIR__, "..", "benchmark", "bootstrap",
        "physics", "modular_pmp", "ModularPMP.jl")
    host = Module(:ModularPMPContract)
    Base.include(host, modular_path)
    mp = getfield(host, :ModularPMP)
    hellerman = getfield(mp, :HellermanModularLP)
    spec = mp.modular_pmp_specs().fixed_gap

    # The basis is generated from the literal f_p equations in
    # HellermanModularLP, not an injected monomial table.  It has one column
    # per odd derivative p and the vacuum normalization is the literal b_p.
    @test spec.derivative_orders == collect(1:2:15)
    @test size(spec.basis) == (16, 8)
    @test length(spec.normalization) == 8
    @test spec.source_version == "0902.2790v2"
    @test spec.eta_product_terms == 64
    @test all(isfinite, spec.basis)
    @test all(isfinite, spec.normalization)
    @test rank(Float64.(spec.basis)) == 8
    # Independent interpolation checks against the literal Hellerman module.
    setprecision(256) do
        eh = (BigFloat(2) - BigFloat(2) - BigFloat(2)) / BigFloat(24)
        shift = spec.gap + eh
        for (column, order) in pairs(spec.derivative_orders)
            for x in BigFloat.(0:order)
                expected = hellerman.derivative_polynomial(order, x + shift)
                reconstructed = sum(spec.basis[k + 1, column] * x^k for k in 0:spec.degree)
                @test isapprox(reconstructed, expected; atol=big"1e-45", rtol=big"1e-45")
            end
            @test spec.normalization[column] == hellerman.vacuum_polynomial(
                order, eh; eta_product_terms=64)
        end
    end
    @test mp.chi_positive_on_halfline(spec.gap)

    # No strict functional witness was independently certified at this gap;
    # construction must fail instead of relabeling synthetic data.
    @test_throws ArgumentError mp.build_modular_pmp(spec)
    @test_throws ArgumentError mp.build_modular_pmp_sdp(spec, Float64)

    @test mp.halfline_gram_dimensions(4) ==
          (parity=:even, q=3, r=2, q_blocks=3, r_blocks=2)
    @test mp.halfline_gram_dimensions(3) ==
          (parity=:odd, q=2, r=2, q_blocks=2, r_blocks=2)
    @test mp.halfline_gram_dimensions(0) ==
          (parity=:even, q=1, r=0, q_blocks=1, r_blocks=0)

    even = mp.reconstruct_even_coefficients
    odd = mp.reconstruct_odd_coefficients
    @test even([1 2 3; 2 4 5; 3 5 6], [7 8; 8 9]) ==
          [1, 4 + 7, 10 + 16, 10 + 9, 6]
    @test even(reshape([4], 1, 1), zeros(Int, 0, 0)) == [4]
    @test odd([2 3; 3 5], [7 11; 11 13]) == [7, 24, 19, 5]
    @test_throws DimensionMismatch even(ones(Int, 2, 2), ones(Int, 2, 2))
    @test_throws DimensionMismatch odd(ones(Int, 1, 1), ones(Int, 2, 2))

    # Validator mutation probes use a directly constructed artifact shell and
    # verify every lowered-data integrity gate fails closed.
    dims = mp.halfline_gram_dimensions(spec.degree)
    alpha = zeros(BigFloat, length(spec.derivative_orders))
    polynomial = zeros(BigFloat, spec.degree + 1)
    polynomial[1] = 1 # deliberate basis*alpha mismatch mutation
    Q = Matrix{BigFloat}(I, dims.q_blocks, dims.q_blocks)
    R = Matrix{BigFloat}(I, dims.r_blocks, dims.r_blocks)
    provenance = (test=true,)
    artifact = mp.ModularPMPArtifact(2, spec, dims, alpha, polynomial, Q, R,
        true, provenance, "bad")
    verdict = mp.validate_artifact(artifact)
    @test !verdict.valid
    @test "basis_alpha_reconstruction" in verdict.failures
    @test "vacuum_normalization" in verdict.failures
    @test "fingerprint" in verdict.failures
    @test !mp.validate_artifact(mp.ModularPMPArtifact(
        2, spec, dims, alpha, polynomial, Q + transpose(Q), R, true,
        provenance, "bad")).valid
end
