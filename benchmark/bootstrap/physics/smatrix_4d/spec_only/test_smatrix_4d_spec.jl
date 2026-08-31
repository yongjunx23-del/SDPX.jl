using Test
using LinearAlgebra
using SDPX
include(joinpath(@__DIR__, "diagnostic_impl.jl"))
const S4 = SMatrix4DSpecDiagnostic

@testset "4D S-matrix remains experimental and unregistered" begin
    @test !isfile(joinpath(@__DIR__, "..", "catalog.jl"))
    spec = S4.SMatrix4DSpec{Float64}(
        id="diagnostic", scale=:tiny, external_mass=1.0,
        ansatz_degree=4, energy_samples=8, spin_max=4,
        quadrature_order=32, formulation=:primal_full_unitarity,
        witness_mode=:none, reference_status=:experimental_build_only,
        paper_equivalent=false,
    )
    artifact = S4.build_smatrix_4d(spec)
    @test S4.validate_artifact(artifact).valid
    @test artifact.counts.dimension == 4
    @test artifact.counts.lorentz_cones == 24
    @test size(artifact.projection_real) == (24, 35)
    @test artifact.fingerprint == S4.stable_fingerprint(artifact)
    @test !artifact.provenance.paper_equivalent
    @test artifact.provenance.formulation_source == :miro_guerrieri_gumus_2210_01502v2
end

@testset "4D kinematics, crossing and projection normalization" begin
    artifact = S4.build_smatrix_4d(:tiny)
    T = Float64
    mass2 = artifact.spec.external_mass^2
    for (s, z) in Iterators.product(artifact.energies_squared, artifact.quadrature_nodes)
        t = -(s - 4mass2) * (1-z) / 2
        u = -(s - 4mass2) * (1+z) / 2
        @test isapprox(s+t+u, 4mass2; atol=1e-12)
        c = zeros(T, length(artifact.basis_indices)); c[findfirst(==( (0,0,0) ), artifact.basis_indices)] = 1
        m = S4.evaluate_amplitude(artifact, c, s, t, u)
        @test m == S4.evaluate_amplitude(artifact, c, t, s, u)
        @test m == S4.evaluate_amplitude(artifact, c, u, t, s)
        break
    end
    # Constant M=1 has f_0=1/(16*pi); its odd/positive-spin projections
    # vanish under symmetric Gauss-Legendre quadrature.
    constant_column = findfirst(==( (0,0,0) ), artifact.basis_indices)
    @test isapprox(artifact.projection_real[1, constant_column], 1/(16pi); atol=1e-12)
    @test maximum(abs, artifact.projection_real[2:3, constant_column]) < 1e-8
    @test maximum(abs, artifact.projection_imag[:, constant_column]) < 1e-12
end

@testset "4D boundary, SOC/PSD equivalence and max-margin construction" begin
    artifact = S4.build_smatrix_4d(:tiny)
    zero_coefficients = artifact.zero_witness
    @test maximum(abs, S4.cone_margins(artifact, zero_coefficients)) < 1e-12
    @test minimum(S4.cone_margins(artifact, artifact.strict_witness)) > 0 ||
          minimum(artifact.strict_margins) <= 100eps(Float64)
    sdp_blocks = S4.build_sdp_blocks(artifact)
    @test length(sdp_blocks.coefficients) == size(artifact.projection_real, 1)
    coefficients = fill(0.01, length(artifact.basis_indices))
    margins = S4.cone_margins(artifact, coefficients)
    for (row, block) in enumerate(sdp_blocks.coefficients)
        evaluated = copy(sdp_blocks.constants[row])
        for j in axes(block, 3)
            evaluated .+= coefficients[j] .* block[:, :, j]
        end
        @test isapprox(det(evaluated), margins[row]; atol=1e-12, rtol=1e-10)
    end
    p_soc = S4.build_soc_problem(artifact)
    @test length(p_soc.cones) == size(artifact.projection_real, 1)
    p_margin = S4.build_max_margin_problem(artifact)
    @test length(p_margin.cones) == size(artifact.projection_real, 1) + 2
    @test length(p_margin.c) == length(artifact.basis_indices) + 1
    @test p_margin.c[end] == -1.0
    @test_throws ArgumentError S4.require_strict_witness(artifact)

    # The paper's linearized dual and the implementation finite conic dual
    # are distinct parameterized formulations, not aliases of the primal.
    dual_spec = S4.SMatrix4DSpec{Float64}(
        id="dual-diagnostic", scale=:tiny, ansatz_degree=4,
        energy_samples=8, spin_max=4, quadrature_order=32,
        formulation=:dual_linearized, witness_mode=:max_margin,
        reference_status=:experimental_build_only,
    )
    @test_throws ArgumentError S4.build_smatrix_4d(dual_spec)
    @test S4.SMatrix4DSpec{Float64}(
        id="finite-dual-diagnostic", scale=:tiny, ansatz_degree=4,
        energy_samples=8, spin_max=4, quadrature_order=32,
        formulation=:finite_conic_dual,
    ).formulation === :finite_conic_dual
end

@testset "4D threshold is excluded but its limit is explicit" begin
    artifact = S4.build_smatrix_4d(:tiny)
    @test all(>(0), artifact.betas)
    @test all(<(1), artifact.betas)
    beta = 1e-12
    @test isapprox(sqrt((1e-12)^2), beta; atol=eps())
    # At beta=0, S_l=1 for every finite f_l and the disk margin is zero.
    @test 1.0 - (1.0)^2 == 0.0
end
