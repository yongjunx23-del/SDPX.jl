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

        @test_throws ArgumentError SDPX._soc_inverse!(zeros(2), [-2.0, 0.0])
        @test_throws DimensionMismatch SDPX._soc_inverse!(zeros(3), [2.0, 0.0])
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

    @testset "general NT agrees with Q3 and its inverse" begin
        primal = [3.0, 0.3, -0.2]
        dual = [2.5, -0.1, 0.25]
        w = zeros(3)
        lambda = zeros(3)
        ok, eta, eta_squared =
            SDPX._soc_nt_scaling!(w, lambda, primal, dual)
        q3_w = zeros(3)
        q3_lambda = zeros(3)
        q3_ok, q3_eta, q3_eta_squared =
            SDPX._q3_nt_scaling!(q3_w, q3_lambda, primal, dual)
        @test ok && q3_ok
        @test w ≈ q3_w
        @test lambda ≈ q3_lambda
        @test eta ≈ q3_eta
        @test eta_squared ≈ q3_eta_squared

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

    @testset "Q3 coordinate/matrix identity" begin
        coordinate = [2.0, 0.4, -0.3]
        matrix = SDPX._q3_to_sym2(coordinate)
        @test det(matrix) ≈ SDPX._soc_determinant(coordinate)
        @test eigmin(Symmetric(matrix)) ≈ SDPX._soc_margin(coordinate)
        @test SDPX._sym2_to_q3(matrix) ≈ coordinate
    end
end
