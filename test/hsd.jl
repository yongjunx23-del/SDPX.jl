# Phase 6 HSD embedding + certificate tests.
using SDPX
using Test
using LinearAlgebra

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
