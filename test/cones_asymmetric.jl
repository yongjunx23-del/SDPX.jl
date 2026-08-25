# ExpCone / PowerCone primitives (Subagent E).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra

@testset "exponential cone membership" begin
    @test SDPX.exp_membership(0.0, 1.0, 1.0)
    @test SDPX.exp_membership(1.0, 1.0, exp(1.0))
    @test SDPX.exp_membership(1.0, 1.0, exp(1.0) + 1.0)
    @test !SDPX.exp_membership(1.0, 1.0, exp(1.0) - 1.0)
    # limit face (0, y, z) with y, z >= 0
    @test SDPX.exp_membership(0.0, 0.0, 1.0)
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
    @test H[1, 1] ≈ hxx rtol=1e-3
    # BigFloat
    setprecision(BigFloat, 256) do
        xb, yb, zb = BigFloat(2.0), BigFloat(2.0), BigFloat(1.0)
        fb = SDPX.power_barrier(xb, yb, zb, BigFloat(0.5))
        @test fb isa BigFloat
        @test Float64(fb) ≈ f rtol=1e-12
    end
end
