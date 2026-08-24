# Phase 6 HSD embedding + certificate tests.
using SDPX
using Test
using LinearAlgebra

const _MF = try
    @eval import MultiFloats
    true
catch
    false
end

@testset "Phase 6 HSD embedding + certificates" begin
    # Optimal LP: min x1+x2 s.t. x1+x2=1, x>=0.
    A = reshape([1.0, 1.0], 1, 2)
    b = [1.0]
    c = [1.0, 1.0]
    M = SDPX.hsd_skew_embedding(A, b, c)
    @test SDPX.is_skew_symmetric(M)
    @test size(M) == (1 + 1 + 2, 1 + 1 + 2)

    # Primal-infeasible LP: x1+x2=1 and x1+x2=2 (impossible).
    A2 = reshape([1.0, 1.0], 2, 1)
    b2 = [1.0, 2.0]
    c2 = [0.0]
    # Farkas certificate y=(-1,1): A'y = 0 <= 0, b'y = 1 > 0 -> valid.
    fy = SDPX.primal_infeasibility_certificate(A2, b2, [-1.0, 1.0])
    @test fy.valid
    @test fy.farkas_value > 0
    # A non-certificate y=(1,-1): b'y = -1 < 0 -> invalid.
    @test !SDPX.primal_infeasibility_certificate(A2, b2, [1.0, -1.0]).valid

    # Dual-infeasible / unbounded: min -x s.t. 0*x=0, x>=0 (unbounded below).
    A3 = zeros(1, 1)
    b3 = [0.0]
    c3 = [-1.0]
    ray = SDPX.dual_infeasibility_certificate(A3, c3, [1.0])
    @test ray.valid
    @test ray.objective_value < 0

    # hsd_status mapping.
    @test SDPX.hsd_status([0.5, 0.5], [1.0], [0.0, 0.0], 1.0, 1e-12) === :optimal
    @test SDPX.hsd_status([0.0, 0.0], [1.0], [1.0, 1.0], 0.0, 1.0) === :infeasible
end

@testset "Phase 6 HSD bordered-system prototype" begin
    A = reshape([1.0, 1.0], 1, 2)
    b = [1.0]
    c = [1.0, 1.0]
    system = SDPX.hsd_bordered_system(A, b, c)
    @test system.dim == 1 + 1 + 2
    @test system.m == 1
    @test system.n == 2
    # Skew part is skew-symmetric; the diagonal barrier term mu is added.
    M = system.matrix
    skew = M - Diagonal(M)
    @test SDPX.is_skew_symmetric(skew)
    @test all(isapprox(diag(M), ones(system.dim); atol=1e-12))
end

@testset "Phase 6 HSD classify" begin
    # Optimal LP: min x1+x2 s.t. x1+x2=1, x>=0. x=(1,0), y=0, s=(0,0), tau=1.
    A = reshape([1.0, 1.0], 1, 2)
    b = [1.0]
    c = [1.0, 1.0]
    r = SDPX.hsd_classify(A, b, c, [1.0, 0.0], [1.0], [0.0, 0.0], 1.0, 1e-12)
    @test r.status === :optimal
    @test r.valid

    # Primal-infeasible LP: x1+x2=1 and x1+x2=2. Farkas y=(-1,1).
    A2 = reshape([1.0, 1.0], 2, 1)
    b2 = [1.0, 2.0]
    c2 = [0.0]
    r2 = SDPX.hsd_classify(A2, b2, c2, [0.0], [-1.0, 1.0], [0.0], 0.0, 1.0)
    @test r2.status === :primal_infeasible
    @test r2.valid

    # Dual-infeasible / unbounded: min -x s, 0*x=0, x>=0 (ray x=1).
    A3 = zeros(1, 1)
    b3 = [0.0]
    c3 = [-1.0]
    r3 = SDPX.hsd_classify(A3, b3, c3, [1.0], [0.0], [0.0], 0.0, 1.0)
    @test r3.status === :dual_infeasible
    @test r3.valid
end


@testset "Phase 6 HSD multi-precision classify + bordered" begin
    for T in vcat([Float64, BigFloat], _MF ? [MultiFloats.Float64x2, MultiFloats.Float64x3, MultiFloats.Float64x4] : Type[])
        T === BigFloat && setprecision(BigFloat, 256)
        A = reshape(T[1, 1], 1, 2)
        b = T[1]
        c = T[1, 1]
        r = SDPX.hsd_classify(A, b, c, T[1, 0], T[1], T[0, 0], T(1), T(1e-12))
        @test r.status === :optimal
        sys = SDPX.hsd_bordered_system(A, b, c)
        @test sys.dim == 4
        @test SDPX.is_skew_symmetric(sys.matrix - Diagonal(sys.matrix))
    end
end

@testset "Phase 6 HSD path-following solve" begin
    A = reshape([1.0, 1.0], 1, 2)
    b = [1.0]
    c = [1.0, 1.0]
    r = SDPX.hsd_lp_solve(A, b, c)
    @test r.status === :optimal
    @test r.valid
    @test isapprox(r.pobj, 1.0; atol=1e-6)
    @test r.tau > 0
    xp = r.x ./ r.tau
    @test isapprox(sum(A[1, :] .* xp), 1.0; atol=1e-6)
end

@testset "Phase 6 HSD path-following infeasibility detection" begin
    # Infeasible LP: x1+x2=1 and x1+x2=2. HSD drives tau -> 0, kappa > 0.
    A = reshape([1.0, 1.0], 2, 1)
    b = [1.0, 2.0]
    c = [0.0]
    r = SDPX.hsd_lp_solve(A, b, c)
    @test r.status === :primal_infeasible
    @test r.valid
    @test r.tau < 1e-6
    @test r.kappa > 1e-6
end

@testset "Phase 6 HSD path-following non-symmetric optimum" begin
    # min 1*x1+3*x2 s.t. x1+x2=1, x>=0 => optimum 1 (x1=1,x2=0).
    A = reshape([1.0, 1.0], 1, 2)
    b = [1.0]
    c = [1.0, 3.0]
    r = SDPX.hsd_lp_solve(A, b, c)
    @test r.status === :optimal
    @test isapprox(r.pobj, 1.0; atol=1e-6)
    xp = r.x ./ r.tau
    @test isapprox(xp[2], 0.0; atol=1e-5)
end

@testset "Phase 6 HSD path-following multi-precision" begin
    for T in vcat([Float64, BigFloat], _MF ? [MultiFloats.Float64x2, MultiFloats.Float64x3, MultiFloats.Float64x4] : Type[])
        T === BigFloat && setprecision(BigFloat, 256)
        A = reshape(T[1, 1], 1, 2)
        b = T[1]
        c = T[1, 1]
        r = SDPX.hsd_lp_solve(A, b, c)
        @test r.status === :optimal
        @test isapprox(Float64(r.pobj), 1.0; atol=1e-6)
    end
end

@testset "Phase 6 HSD path-following 3-variable LP" begin
    # min x1+x2+x3 s.t. x1+x2+x3=1, x>=0 => optimum 1.0.
    A = reshape([1.0, 1.0, 1.0], 1, 3)
    b = [1.0]
    c = [1.0, 1.0, 1.0]
    r = SDPX.hsd_lp_solve(A, b, c)
    @test r.status === :optimal
    @test isapprox(r.pobj, 1.0; atol=1e-6)
    xp = r.x ./ r.tau
    @test isapprox(sum(xp), 1.0; atol=1e-6)
end

@testset "Phase 6 HSD degenerate A=0 unbounded detection" begin
    A = zeros(1, 1)
    b = [0.0]
    c = [-1.0]
    r = SDPX.hsd_lp_solve(A, b, c)
    @test r.status === :dual_infeasible
    @test r.valid
    @test r.pobj == -Inf
end

@testset "Phase 6 HSD path-following 2-constraint LP" begin
    # min x1+x2 s.t. x1=1, x2=1, x>=0 => optimum 2.0.
    A = [1.0 0.0; 0.0 1.0]
    b = [1.0, 1.0]
    c = [1.0, 1.0]
    r = SDPX.hsd_lp_solve(A, b, c)
    @test r.status === :optimal
    @test isapprox(r.pobj, 2.0; atol=1e-6)
end

@testset "Phase 6 HSD negative-RHS infeasibility detection" begin
    # Infeasible: x2 = -1 violates x2>=0.
    A = [1.0 0.0; 0.0 1.0]
    b = [1.0, -1.0]
    c = [1.0, 1.0]
    r = SDPX.hsd_lp_solve(A, b, c)
    @test r.status === :primal_infeasible
    @test r.valid
    @test r.tau < 1e-6
end

@testset "Phase 6 HSD path-following general constraint matrix" begin
    # min x1+x2 s.t. x1+2*x2=3, x>=0 => optimum 1.5 (x1=0,x2=1.5).
    A = reshape([1.0, 2.0], 1, 2)
    b = [3.0]
    c = [1.0, 1.0]
    r = SDPX.hsd_lp_solve(A, b, c)
    @test r.status === :optimal
    @test isapprox(r.pobj, 1.5; atol=1e-6)
end

@testset "Phase 6 HSD path-following 4-variable LP" begin
    # min sum(x) s.t. sum(x)=1, x>=0 => optimum 1.0.
    A = reshape([1.0, 1.0, 1.0, 1.0], 1, 4)
    b = [1.0]
    c = [1.0, 1.0, 1.0, 1.0]
    r = SDPX.hsd_lp_solve(A, b, c)
    @test r.status === :optimal
    @test isapprox(r.pobj, 1.0; atol=1e-6)
end
