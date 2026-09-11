using SDPX, Test, LinearAlgebra

@testset "PSD acceptance certifies actual stored inverse orientation" begin
    delta = 2.0^-50
    s = [1.0, sqrt(2.0) * delta, 2delta^2]
    y = [1.0, 0.0, 1.0]
    state = SDPX.SymmetricCones.PSDNTScaling{Float64}(2)
    accepted = false
    try
        SDPX.SymmetricCones.nt_scaling!(
            SDPX.SymmetricCones.PSDTriangleCone{Float64}(2), state, s, y)
        accepted = true
    catch err
        err isa DomainError || rethrow()
    end
    # A future accurate reconstruction may accept this interior point. Merely
    # deleting the inverse gate must not: inspect the actual stored matrices,
    # never a new Cholesky or inv(P), in the independent diagnostic arithmetic.
    accurate_inverse = setprecision(BigFloat, 512) do
        K = BigFloat.(state.Pinv)
        X = BigFloat.(state.S)
        Y = BigFloat.(state.Y)
        error = maximum(abs, K * X * K - Y)
        allowance = BigFloat(eps(Float64)) * max(BigFloat(1), maximum(abs, Y)) * 20000
        isfinite(error) && error <= allowance
    end
    @test !accepted || accurate_inverse
    @test accepted == state.valid[1]
end

@testset "Power algebraic pairing is not root or geometry certification" begin
    Q = Rational{BigInt}
    c = Q(1, BigInt(2)^44)
    a = b = Q(1, 2)
    u = v = Q(1)
    w = 2 * (1 - 5c / 4)
    A = 2a + b*c
    B = 2b + a*c
    x = A / (c*u)
    y = B / (c*v)
    z = -2 * (1-c) / (c*w)
    gap = (x*y - z*z) / (x*y)
    @test u*x + v*y + w*z == 3
    @test gap > 0
    # For alpha=1/2, the true third negative barrier-gradient component is
    # rational: -2z/(xy-z²). No production gap/root/gradient helper is reused.
    negative_gradient3 = -2z / (x*y-z*z)
    @test abs(negative_gradient3/w - 1) > Q(1, 2)
    setprecision(BigFloat, 512) do
        cc = BigFloat(c)
        ww = BigFloat(w)
        phi = log(ww/2) + log1p(cc/2) - log1p(-cc)/2
        true_gap = cc - (1-cc)*expm1(-2phi)
        @test isapprox(true_gap, BigFloat(gap); rtol=BigFloat("1e-130"), atol=0)
        @test abs(expm1(-phi)) < BigFloat("1e-10")
        @test isapprox(expm1(-2phi)/true_gap,
            BigFloat(negative_gradient3/w - 1); rtol=BigFloat("1e-130"), atol=0)
    end
    # This proves insufficiency of the algebraic identity, NOT that all current
    # runtime gates accept this deliberately nonroot parameter.
end
