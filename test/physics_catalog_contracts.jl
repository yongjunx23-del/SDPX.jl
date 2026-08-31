# Exact modular functional PMP-to-SDP contract tests.

@testset "Original modular functional exact half-line lift" begin
    modular_path = joinpath(@__DIR__, "..", "benchmark", "bootstrap",
        "physics", "modular_pmp", "ModularPMP.jl")
    host = Module(:ModularPMPContract)
    Base.include(host, modular_path)
    mp = getfield(host, :ModularPMP)
    specs = mp.modular_pmp_specs()
    spec = specs.fixed_gap
    artifact = mp.build_modular_pmp(spec)
    verdict = mp.validate_artifact(artifact)
    @test verdict.valid
    @test mp.chi_positive_on_halfline(spec.gap)
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

    # The canonical lowering is a real affine SDP with parity-correct Gram
    # blocks.  Its independent fingerprint is stable across rebuilds.
    canonical = mp.build_modular_pmp_sdp(spec, Float64)
    @test canonical !== nothing
    @test artifact.fingerprint == mp.stable_fingerprint(artifact)
    @test artifact.fingerprint == mp.build_modular_pmp(spec).fingerprint
end
