# ExpCone / PowerCone primitives (Subagent E).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra

@testset "exponential cone membership" begin
    @test SDPX.exp_membership(0, 1, 1)
    @test SDPX.exp_membership(0.0, 1.0, 1.0)
    @test SDPX.exp_membership(1.0, 1.0, exp(1.0))
    @test SDPX.exp_membership(1.0, 1.0, exp(1.0) + 1.0)
    @test !SDPX.exp_membership(1.0, 1.0, exp(1.0) - 1.0)
    # limit face (x, 0, z) with x <= 0, z >= 0
    @test SDPX.exp_membership(0.0, 0.0, 1.0)
    @test SDPX.exp_membership(-1.0, 0.0, 1.0)
    @test SDPX.exp_membership(-2.5, 0.0, 0.0)
    @test !SDPX.exp_membership(1.0, 0.0, 1.0)
    @test !SDPX.exp_membership(-1.0, 0.0, -1.0)
    @test !SDPX.exp_membership(0.0, 1.0, 0.0)  # 1*exp(0)=1 <= 0 is false
    @test !SDPX.exp_membership(0.0, -1.0, 1.0)
end

@testset "exponential cone barrier + gradient + hessian" begin
    x, y, z = 0.5, 1.0, 3.0
    f = SDPX.exp_barrier(x, y, z)
    @test isfinite(f)
    g = SDPX.exp_barrier_gradient(x, y, z)
    H = SDPX.exp_barrier_hessian(x, y, z)
    @test size(H) == (3, 3)
    @test H ≈ H'
    # finite-difference gradient check
    h = 1e-6
    gx_num = (SDPX.exp_barrier(x + h, y, z) - SDPX.exp_barrier(x - h, y, z)) / (2h)
    gy_num = (SDPX.exp_barrier(x, y + h, z) - SDPX.exp_barrier(x, y - h, z)) / (2h)
    gz_num = (SDPX.exp_barrier(x, y, z + h) - SDPX.exp_barrier(x, y, z - h)) / (2h)
    @test g[1] ≈ gx_num rtol=1e-4
    @test g[2] ≈ gy_num rtol=1e-4
    @test g[3] ≈ gz_num rtol=1e-4
    # finite-difference Hessian check (diagonal)
    hxx = (SDPX.exp_barrier_gradient(x + h, y, z)[1] - SDPX.exp_barrier_gradient(x - h, y, z)[1]) / (2h)
    @test H[1, 1] ≈ hxx rtol=1e-3
    # BigFloat
    setprecision(BigFloat, 256) do
        xb, yb, zb = BigFloat(0.5), BigFloat(1.0), BigFloat(3.0)
        fb = SDPX.exp_barrier(xb, yb, zb)
        @test fb isa BigFloat
        @test Float64(fb) ≈ f rtol=1e-12
    end
end

@testset "power cone membership" begin
    alpha = 0.5
    @test SDPX.power_membership(1, 1, 1, alpha)
    # x^0.5 y^0.5 >= |z|
    @test SDPX.power_membership(1.0, 1.0, 1.0, alpha)
    @test SDPX.power_membership(4.0, 1.0, 2.0, alpha)
    @test !SDPX.power_membership(1.0, 1.0, 1.5, alpha)
    @test SDPX.power_membership(1.0, 1.0, -1.0, alpha)
    @test !SDPX.power_membership(-1.0, 1.0, 0.0, alpha)
    # alpha near 0 and 1
    @test SDPX.power_membership(1.0, 1.0, 1.0, 0.01)
    @test SDPX.power_membership(1.0, 1.0, 1.0, 0.99)
end

@testset "power cone barrier + gradient + hessian" begin
    alpha = 0.5
    x, y, z = 2.0, 2.0, 1.0
    f = SDPX.power_barrier(x, y, z, alpha)
    @test isfinite(f)
    g = SDPX.power_barrier_gradient(x, y, z, alpha)
    H = SDPX.power_barrier_hessian(x, y, z, alpha)
    @test size(H) == (3, 3)
    @test H ≈ H'
    h = 1e-6
    gx_num = (SDPX.power_barrier(x + h, y, z, alpha) - SDPX.power_barrier(x - h, y, z, alpha)) / (2h)
    gy_num = (SDPX.power_barrier(x, y + h, z, alpha) - SDPX.power_barrier(x, y - h, z, alpha)) / (2h)
    gz_num = (SDPX.power_barrier(x, y, z + h, alpha) - SDPX.power_barrier(x, y, z - h, alpha)) / (2h)
    @test g[1] ≈ gx_num rtol=1e-4
    @test g[2] ≈ gy_num rtol=1e-4
    @test g[3] ≈ gz_num rtol=1e-4
    hxx = (SDPX.power_barrier_gradient(x + h, y, z, alpha)[1] - SDPX.power_barrier_gradient(x - h, y, z, alpha)[1]) / (2h)
    hxz = (SDPX.power_barrier_gradient(x, y, z + h, alpha)[1] - SDPX.power_barrier_gradient(x, y, z - h, alpha)[1]) / (2h)
    hyz = (SDPX.power_barrier_gradient(x, y, z + h, alpha)[2] - SDPX.power_barrier_gradient(x, y, z - h, alpha)[2]) / (2h)
    @test H[1, 1] ≈ hxx rtol=1e-3
    @test H[1, 3] ≈ hxz rtol=1e-3
    @test H[2, 3] ≈ hyz rtol=1e-3
    @test abs(H[1, 3]) > 1e-4  # Ensure non-zero
    @test abs(H[2, 3]) > 1e-4
    # BigFloat
    setprecision(BigFloat, 256) do
        xb, yb, zb = BigFloat(2.0), BigFloat(2.0), BigFloat(1.0)
        fb = SDPX.power_barrier(xb, yb, zb, BigFloat(0.5))
        @test fb isa BigFloat
        @test Float64(fb) ≈ f rtol=1e-12
    end
end

@testset "dual exponential cone membership" begin
    # Dual: -u * exp(v/u - 1) <= w, u < 0
    # For (u, v, w) = (-1.0, 1.0, exp(0) = 1.0): 1 * exp(0) = 1 <= 1.0 (boundary)
    @test SDPX.exp_dual_membership(-1.0, 1.0, 1.0)
    @test SDPX.exp_dual_membership(-1.0, 1.0, 0.5)
    @test !SDPX.exp_dual_membership(-1.0, 1.0, 0.05)
    # limit face u == 0: v >= 0, w >= 0
    @test SDPX.exp_dual_membership(0.0, 1.0, 1.0)
    @test SDPX.exp_dual_membership(0.0, 0.0, 0.0)
    @test !SDPX.exp_dual_membership(0.0, -1.0, 1.0)
    @test !SDPX.exp_dual_membership(1.0, 1.0, 1.0)
end

@testset "dual power cone membership" begin
    # Dual: (u/alpha)^alpha * (v/(1-alpha))^(1-alpha) >= |w|, u >= 0, v >= 0
    alpha = 0.5
    # For u=0.5, v=0.5: (1)^0.5 * (1)^0.5 = 1 >= |w|
    @test SDPX.power_dual_membership(0.5, 0.5, 1.0, alpha)
    @test SDPX.power_dual_membership(0.5, 0.5, 0.5, alpha)
    @test !SDPX.power_dual_membership(0.5, 0.5, 1.5, alpha)
    @test !SDPX.power_dual_membership(-0.5, 0.5, 0.0, alpha)
    @test SDPX.power_dual_membership(0.0, 0.0, 0.0, alpha)
end

@testset "in-place barrier Hessians" begin
    # Exp cone in-place
    x, y, z = 0.5, 1.0, 3.0
    H_exp = zeros(Float64, 3, 3)
    SDPX.exp_barrier_hessian!(H_exp, x, y, z)
    @test H_exp ≈ SDPX.exp_barrier_hessian(x, y, z)

    # Power cone in-place
    H_pow = zeros(Float64, 3, 3)
    SDPX.power_barrier_hessian!(H_pow, 2.0, 2.0, 1.0, 0.5)
    @test H_pow ≈ SDPX.power_barrier_hessian(2.0, 2.0, 1.0, 0.5)
end
