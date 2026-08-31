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

    # Gram checks preserve the element type and distinguish PSD from PD.
    @test mp._validate_gram(Matrix{BigFloat}(I, 2, 2), false)
    @test mp._validate_gram(zeros(BigFloat, 2, 2), false)
    @test !mp._validate_gram(zeros(BigFloat, 2, 2), true)
    tiny_negative = BigFloat[1 0; 0 big"-1e-1000"]
    @test !mp._validate_gram(tiny_negative, false)
    @test !mp._validate_gram(BigFloat[1 2; 0 1], false)
    @test !mp._validate_gram(Matrix{Int}(I, 2, 2), false)

    # Every lowered-data mutation is rejected without throwing.
    @test !mp.validate_artifact(mp.ModularPMPArtifact(
        2, spec, (parity=:even, q=1, r=1, q_blocks=1, r_blocks=1),
        alpha, polynomial, Q, R, false, mp._provenance(spec), "bad")).valid
    @test "provenance" in mp.validate_artifact(mp.ModularPMPArtifact(
        2, spec, dims, alpha, polynomial, Q, R, false,
        (wrong=true,), "bad")).failures
    @test mp.stable_fingerprint(artifact) != mp.stable_fingerprint(
        mp.ModularPMPArtifact(2, spec, (parity=:even, q=99, r=2,
            q_blocks=3, r_blocks=2), alpha, polynomial, Q, R, true,
            provenance, "bad"))
    @test mp.stable_fingerprint(artifact) != mp.stable_fingerprint(
        mp.ModularPMPArtifact(2, spec, dims, alpha, polynomial,
            Q + Matrix{BigFloat}(I, dims.q_blocks, dims.q_blocks), R,
            true, provenance, "bad"))
    @test occursin("artifact.gram_dimensions", mp.canonical_text(artifact))

    # Non-finite vectors, asymmetric/tiny-negative Gram data, and malformed
    # dimensions all fail as explicit verdicts rather than validator throws.
    nan_alpha = copy(alpha); nan_alpha[1] = BigFloat("NaN")
    @test "alpha_finite" in mp.validate_artifact(mp.ModularPMPArtifact(
        2, spec, dims, nan_alpha, polynomial, Q, R, false, provenance, "bad")).failures
    nan_poly = copy(polynomial); nan_poly[1] = BigFloat("NaN")
    @test "polynomial_finite" in mp.validate_artifact(mp.ModularPMPArtifact(
        2, spec, dims, alpha, nan_poly, Q, R, false, provenance, "bad")).failures
    asym = copy(Q); asym[1, 2] = one(BigFloat)
    @test "Q_psd" in mp.validate_artifact(mp.ModularPMPArtifact(
        2, spec, dims, alpha, polynomial, asym, R, false, provenance, "bad")).failures
    negative = copy(Q); negative[1, 1] = big"-1e-1000"
    @test "Q_psd" in mp.validate_artifact(mp.ModularPMPArtifact(
        2, spec, dims, alpha, polynomial, negative, R, false, provenance, "bad")).failures
    malformed_Q = ones(BigFloat, 1, 2)
    malformed = mp.validate_artifact(mp.ModularPMPArtifact(
        2, spec, dims, alpha, polynomial, malformed_Q, R, false, provenance, "bad"))
    @test !malformed.valid
    @test "Q_dimensions" in malformed.failures
end
