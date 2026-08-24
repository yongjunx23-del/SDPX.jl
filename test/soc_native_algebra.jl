using LinearAlgebra

@testset "Native Lorentz algebra" begin
    @testset "alias safety and cone sheet" begin
        left = [3.0, 1.0, 0.5]
        right = [2.0, -0.25, 1.0]
        reference = zeros(3)
        SDPX._soc_jordan!(reference, left, right)

        left_alias = copy(left)
        SDPX._soc_jordan!(left_alias, left_alias, right)
        @test left_alias ≈ reference
        right_alias = copy(right)
        SDPX._soc_jordan!(right_alias, left, right_alias)
        @test right_alias ≈ reference

    end

    @testset "scaled stable boundary root" begin
        @test SDPX._soc_fraction_to_boundary(
            [1.0e150, 0.0, 0.0],
            [-1.0e150, 1.0e150, 0.0],
        ) ≈ 0.5
        @test SDPX._soc_fraction_to_boundary(
            [1.0, 0.0], [-0.5, 0.5],
        ) == 1.0
        @test SDPX._soc_fraction_to_boundary(
            [1.0, 0.0], [-1.0, 1.0],
        ) ≈ 0.5
    end

    @testset "strict full-step policy" begin
        state = [[2.0, 0.0]]
        direction = [[0.0, 0.5]]
        trial = [[0.0, 0.0]]
        @test SDPX._native_soc_strict_step(
            state, direction, trial, 1.0; allow_full_step=true,
        ) == 1.0
        @test SDPX._native_soc_strict_step(
            state, direction, trial, 1.0,
        ) == 0.99

        boundary_state = [[1.0, 0.0]]
        boundary_direction = [[-0.5, 0.5]]
        boundary_trial = [[0.0, 0.0]]
        boundary_step = SDPX._native_soc_strict_step(
            boundary_state, boundary_direction, boundary_trial, 1.0;
            allow_full_step=true,
        )
        @test boundary_step == 0.99
        @test SDPX._soc_is_interior(
            boundary_state[1] .+ boundary_step .* boundary_direction[1],
        )
        @test SDPX._native_soc_strict_step(
            state, direction, trial, 0.5,
        ) == 0.495
    end

    @testset "general NT scaling and its inverse roundtrip" begin
        primal = [3.0, 0.3, -0.2]
        dual = [2.5, -0.1, 0.25]
        w = zeros(3)
        lambda = zeros(3)
        ok, eta, eta_squared =
            SDPX._soc_nt_scaling!(w, lambda, primal, dual)
        @test ok

        for dimension in (2, 3, 5)
            s = [3.0; fill(0.15, dimension - 1)]
            z = [2.0; fill(-0.1, dimension - 1)]
            scaling = zeros(dimension)
            central = zeros(dimension)
            scaling_ok, scale, _ =
                SDPX._soc_nt_scaling!(scaling, central, s, z)
            @test scaling_ok
            source = collect(range(0.1, 0.1 * dimension; length=dimension))
            transformed = copy(source)
            SDPX._soc_nt_apply_w!(transformed, scaling, scale, transformed)
            SDPX._soc_nt_apply_winv!(
                transformed, scaling, scale, transformed,
            )
            @test transformed ≈ source atol=1e-11 rtol=1e-11
        end
    end

    @testset "Lorentz coordinate/matrix identity" begin
        coordinate = [2.0, 0.4, -0.3]
        matrix = [coordinate[1] + coordinate[2] coordinate[3];
                  coordinate[3] coordinate[1] - coordinate[2]]
        @test det(matrix) ≈ SDPX._soc_determinant(coordinate)
        @test eigmin(Symmetric(matrix)) ≈ SDPX._soc_margin(coordinate)
    end

    @testset "fixed-trace local kernels match explicit affine algebra" begin
        A = [0.0 0.0; 1.25 -0.5; 0.75 2.0]
        x = [0.4, -0.3]
        slack = [1.1, -0.2, 0.7]
        offset = [1.5, -0.6, 0.25]
        dual = [0.8, -0.4, 0.3]

        residual = zeros(3)
        SDPX._soc_fixed_trace_primal_residual!(
            residual, x, slack, 1, 2,
            A[2, 1], A[2, 2], A[3, 1], A[3, 2],
            offset[1], offset[2], offset[3],
        )
        @test residual ≈ A * x + offset - slack

        scattered = [0.2, -0.1]
        reference = scattered - transpose(A) * dual
        SDPX._soc_fixed_trace_dual_scatter!(
            scattered, dual, 1, 2,
            A[2, 1], A[2, 2], A[3, 1], A[3, 2],
        )
        @test scattered ≈ reference

        contracted = [0.15, -0.25]
        source = [0.9, -0.2, 0.45]
        contraction_reference = contracted + transpose(A) * source
        SDPX._soc_fixed_trace_transpose_scatter!(
            contracted, source, 1, 2,
            A[2, 1], A[2, 2], A[3, 1], A[3, 2],
        )
        @test contracted ≈ contraction_reference

        direction = [0.7, -0.35]
        recovered = [0.6, -0.8, 0.4]
        recovery_reference = recovered + A * direction
        SDPX._soc_fixed_trace_primal_map!(
            recovered, direction, 1, 2,
            A[2, 1], A[2, 2], A[3, 1], A[3, 2],
        )
        @test recovered ≈ recovery_reference
        @test recovered[1] == 0.6
    end
end

@testset "fixed-trace HKM metric interiority contract" begin
    destination = zeros(3)
    dual = [2.0, 0.25, -0.5]
    @test SDPX._soc_fixed_trace_hkm_metric!(
        destination, [3.0, 1.0, 0.5], dual,
    )
    @test destination ≈ [
        (3 * 2 - 1 * 0.25 + 0.5 * -0.5) / (9 - 1 - 0.25),
        -(1 * -0.5 + 0.5 * 0.25) / (9 - 1 - 0.25),
        (3 * 2 + 1 * 0.25 - 0.5 * -0.5) / (9 - 1 - 0.25),
    ]
    # A reflected head has a positive determinant but is outside the cone;
    # the boundary itself is non-interior in either representation.
    @test !SDPX._soc_fixed_trace_hkm_metric!(
        zeros(3), [-3.0, 1.0, 0.5], dual,
    )
    @test !SDPX._soc_fixed_trace_hkm_metric!(
        zeros(3), [1.0, 1.0, 0.0], dual,
    )
end
