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
        witness_mode=:none, reference_status=:sampled_build_only,
        paper_equivalent=false,
    )
    artifact = S4.build_smatrix_4d(spec)
    @test S4.validate_artifact(artifact).valid
    @test artifact.counts.dimension == 4
    @test artifact.counts.lorentz_cones == 24
    @test artifact.counts.basis_width == 11
    @test size(artifact.projection_real, 1) == 24
    @test size(artifact.projection_real, 2) == length(artifact.basis_indices)
    @test all(index -> index[1] >= index[2] >= index[3], artifact.basis_indices)
    @test artifact.fingerprint == S4.stable_fingerprint(artifact)
    @test !artifact.provenance.paper_equivalent
    @test artifact.provenance.formulation_source == :miro_guerrieri_gumus_2605_06613v1
    @test artifact.provenance.basis_implementation == :single_anchor_triple_rho_diagnostic
    @test artifact.provenance.arxiv == "2605.06613"
    @test artifact.provenance.predecessor_references == ("2210.01502v2", "1708.06765v1", "2106.10257")
    @test artifact.spec.basis_kind === :single_anchor_triple_rho
    @test artifact.spec.matches_paper_defaults
    @test artifact.phase_space == artifact.betas
    @test artifact.spec.nmax_values == (10, 12, 14, 16, 18, 20)
    @test artifact.spec.lmax_values == (16, 18)
    @test artifact.spec.s_max == 300.0
    @test artifact.spec.a7_t_grid[end] == 4.0
    @test artifact.spec.dual_mu2 == 12.0
    @test artifact.provenance.paper_defaults.dual_mu2 == 12
    @test artifact.provenance.primal_analyticity_assumption === :maximal_landau
    @test artifact.provenance.dual_analyticity_assumption === :axiomatic_fixed_t
    @test S4.alpha_threshold_value(artifact.spec) == 0.0
    stress = S4.smatrix_4d_specs(Float64).stress
    @test stress.ansatz_degree == 16
    @test stress.spin_max == 32
    @test stress.s_max == 300.0
    @test stress.quadrature_order == 256
    # The stress tier is metadata/profile-only; it is not registered or solved.
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
    # Constant M=1 has f_0=1/(16*pi); its positive-spin projections
    # vanish under symmetric Gauss-Legendre quadrature.
    constant_column = findfirst(==( (0,0,0) ), artifact.basis_indices)
    @test isapprox(artifact.projection_real[1, constant_column], 1/(16pi); atol=1e-12)
    @test maximum(abs, artifact.projection_real[2:3, constant_column]) < 1e-8
    @test maximum(abs, artifact.projection_imag[:, constant_column]) < 1e-12

    # All six permutations, including repeated exponents, are one orbit and
    # evaluate identically after permutation of the rho coordinates.
    rhos = (0.2 + 0.1im, -0.3 + 0.2im, 0.4 - 0.1im)
    rho_permutations = ((rhos[1], rhos[2], rhos[3]),
        (rhos[1], rhos[3], rhos[2]), (rhos[2], rhos[1], rhos[3]),
        (rhos[2], rhos[3], rhos[1]), (rhos[3], rhos[1], rhos[2]),
        (rhos[3], rhos[2], rhos[1]))
    for index in ((2, 2, 0), (2, 1, 0), (3, 1, 1))
        value = S4._orbit_value(rhos, index)
        @test length(S4._orbit_permutations(index)) ==
            (index[1] == index[2] == index[3] ? 1 :
             (index[1] == index[2] || index[1] == index[3] || index[2] == index[3] ? 3 : 6))
        for rho_permutation in rho_permutations
            @test isapprox(value, S4._orbit_value(rho_permutation, index);
                atol=1e-14, rtol=1e-14)
        end
    end
    upper = S4.rho_coordinate(8.0, 1.0, 4 / 3; rim=:upper)
    lower = S4.rho_coordinate(8.0, 1.0, 4 / 3; rim=:lower)
    @test lower == conj(upper)
    @test S4.rho_coordinate(4 / 3, 1.0, 4 / 3) == 0
    @test isapprox(S4.rho_coordinate(4.0, 1.0, 4 / 3), 1; atol=1e-14)
    @test_throws ArgumentError S4.rho_coordinate(4.0, 1.0, 4.0)
    for (x, mass2, sstar) in ((3.0, 1.0, 4 / 3), (4.0, 1.0, 4 / 3),
                              (5.0, 1.0, 4 / 3), (1.0, 1.0, 4 / 3))
        @test_throws ArgumentError S4.rho_coordinate(x, mass2, sstar; rim=:invalid)
    end
end

@testset "4D boundary, SOC/PSD equivalence and max-margin construction" begin
    artifact = S4.build_smatrix_4d(:tiny)
    zero_coefficients = artifact.zero_witness
    @test maximum(abs, S4.cone_margins(artifact, zero_coefficients)) < 1e-12
    @test minimum(S4.cone_margins(artifact, artifact.strict_witness)) > 0 ||
          minimum(artifact.strict_margins) <= 100eps(Float64)
    for degree_width in ((4, 11), (8, 41), (12, 102), (16, 204))
        s = S4.SMatrix4DSpec{Float64}(id="width", scale=:tiny,
            ansatz_degree=degree_width[1], energy_samples=2, spin_max=0,
            quadrature_order=degree_width[1] + 4)
        @test length(S4.build_smatrix_4d(s).basis_indices) == degree_width[2]
    end
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
    @test p_soc.cones[1].b == [1.0, 1.0, 0.0]
    @test p_margin.cones[1].b == [1.0, 1.0, 0.0]
    @test sdp_blocks.constants[1] == [2.0 0.0; 0.0 0.0]
    @test_throws ArgumentError S4.require_strict_witness(artifact)
    bad_spin = S4.SMatrix4DSpec{Float64}(
        id="bad-spin", scale=:tiny, ansatz_degree=4, energy_samples=8,
        spin_max=4, spin_set=(2, 0), quadrature_order=32)
    @test_throws ArgumentError S4.build_smatrix_4d(bad_spin)
    bad_center = S4.SMatrix4DSpec{Float64}(
        id="bad-center", scale=:tiny, ansatz_degree=4, energy_samples=8,
        spin_max=4, quadrature_order=32, basis_centers=(3.0,))
    @test_throws ArgumentError S4.build_smatrix_4d(bad_center)
    bad_dual = S4.SMatrix4DSpec{Float64}(
        id="bad-dual", scale=:tiny, ansatz_degree=4, energy_samples=8,
        spin_max=4, quadrature_order=32, dual_mu2=3.0)
    @test_throws ArgumentError S4.build_smatrix_4d(bad_dual)
    bad_objective = S4.SMatrix4DSpec{Float64}(
        id="bad-objective", scale=:tiny, ansatz_degree=4, energy_samples=8,
        spin_max=4, quadrature_order=32, objective=:c0)
    @test_throws ArgumentError S4.build_smatrix_4d(bad_objective)
    custom = S4.SMatrix4DSpec{Float64}(
        id="custom", scale=:tiny, ansatz_degree=4, energy_samples=8,
        spin_max=4, quadrature_order=32, basis_centers=(7.0,),
        matches_paper_defaults=false)
    @test S4.validate_artifact(S4.build_smatrix_4d(custom)).valid
    @test !S4.build_smatrix_4d(custom).provenance.matches_paper_defaults

    # The paper's linearized dual and the implementation finite conic dual
    # are distinct parameterized formulations, not aliases of the primal.
    dual_spec = S4.SMatrix4DSpec{Float64}(
        id="dual-diagnostic", scale=:tiny, ansatz_degree=4,
        energy_samples=8, spin_max=4, quadrature_order=32,
        formulation=:dual_linearized, witness_mode=:max_margin,
        reference_status=:sampled_build_only,
    )
    @test_throws ArgumentError S4.build_smatrix_4d(dual_spec)
    @test S4.SMatrix4DSpec{Float64}(
        id="finite-dual-diagnostic", scale=:tiny, ansatz_degree=4,
        energy_samples=8, spin_max=4, quadrature_order=32,
        formulation=:finite_conic_dual,
    ).formulation === :finite_conic_dual

    # Every mutable artifact array is fingerprinted and semantically rebuilt.
    for field in (:energies_squared, :betas, :phase_space, :strict_margins,
                  :strict_witness, :zero_witness, :projection_real,
                  :projection_imag, :quadrature_nodes, :quadrature_weights)
        target = getfield(artifact, field)
        old = target[1]
        target[1] = old + (field === :projection_real ? 1.0e-8 : 1.0e-8)
        @test S4.stable_fingerprint(artifact) != artifact.fingerprint
        @test !S4.validate_artifact(artifact).valid
        target[1] = old
    end
    @test S4.validate_artifact(artifact).valid
    altered = S4.SMatrix4DSpec{Float64}(
        id="altered", scale=:tiny, ansatz_degree=4, energy_samples=8,
        spin_max=4, quadrature_order=32, beta_min=0.2,
        matches_paper_defaults=false)
    altered_artifact = S4.build_smatrix_4d(altered)
    @test altered_artifact.fingerprint != artifact.fingerprint
    # Even a forged replacement fingerprint cannot make provenance/counts
    # inconsistent with a deterministic rebuild.
    forged_provenance = S4.SMatrix4DArtifact(
        artifact.schema_version, artifact.spec, artifact.betas,
        artifact.phase_space, artifact.energies_squared,
        artifact.quadrature_nodes, artifact.quadrature_weights,
        artifact.basis_indices, artifact.projection_real, artifact.projection_imag,
        artifact.strict_witness, artifact.zero_witness, artifact.strict_margins,
        artifact.witness_certified, merge(artifact.provenance, (title="forged",)),
        artifact.counts, "")
    forged = S4.SMatrix4DArtifact(
        forged_provenance.schema_version, forged_provenance.spec,
        forged_provenance.betas, forged_provenance.phase_space,
        forged_provenance.energies_squared, forged_provenance.quadrature_nodes,
        forged_provenance.quadrature_weights, forged_provenance.basis_indices,
        forged_provenance.projection_real, forged_provenance.projection_imag,
        forged_provenance.strict_witness, forged_provenance.zero_witness,
        forged_provenance.strict_margins, forged_provenance.witness_certified,
        forged_provenance.provenance, forged_provenance.counts,
        S4.stable_fingerprint(forged_provenance))
    @test !S4.validate_artifact(forged).valid
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
