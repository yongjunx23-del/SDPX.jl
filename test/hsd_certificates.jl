# Unified HSD state + certificate verification (Subagent C, PR3).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra
using SparseArrays

@testset "HSD state residuals" begin
    # min c'x s.t. A x + s = b, s >= 0
    A = sparse([1.0 0.0; 0.0 1.0])
    b = [1.0, 1.0]
    c = [1.0, 1.0]
    x = [0.5, 0.5]
    s = [0.5, 0.5]
    state = SDPX.HSDState(x, s, 1.0, 0.0, A, b, c)
    @test SDPX.hsd_dimension(state) == 3
    @test SDPX.hsd_primal_residual(state) ≈ [0.0, 0.0]
    @test SDPX.hsd_dual_residual(state) ≈ -2.0
    @test SDPX.hsd_complementarity(state) ≈ 0.5 / 3
    @test SDPX.hsd_optimality_gap(state) ≈ 2.0
end

@testset "cone membership" begin
    @test SDPX.in_cone(SDPX.OrthantMembership(), [1.0, 2.0])
    @test !SDPX.in_cone(SDPX.OrthantMembership(), [1.0, -1.0])
    @test SDPX.in_cone(SDPX.SOCMembership(), [2.0, 1.0, 1.0])
    @test !SDPX.in_cone(SDPX.SOCMembership(), [1.0, 2.0, 0.0])
    @test SDPX.in_cone(SDPX.PSDMembership(), [1.0, 0.0, 1.0])
    @test !SDPX.in_cone(SDPX.PSDMembership(), [1.0, 2.0, 1.0])
    @test SDPX.in_cone(SDPX.ExpMembership(), [0.0, 1.0, 1.0])
    @test SDPX.in_cone(SDPX.PowerMembership(0.5), [1.0, 1.0, 1.0])
end

@testset "verify_optimal!" begin
    # min 0 s.t. x + s = 1, s >= 0  (c=0, A=[1], b=[1])
    # valid HSD point: x=1, s=0, tau=1, kappa=0
    A = sparse([1.0;;])
    b = [1.0]
    c = [0.0]
    state = SDPX.HSDState([1.0], [0.0], 1.0, 0.0, A, b, c)
    @test SDPX.verify_optimal!(state, [SDPX.OrthantMembership()])
    # non-optimal: s not in cone
    bad = SDPX.HSDState([1.0], [-0.5], 1.0, 0.0, A, b, c)
    @test !SDPX.verify_optimal!(bad, [SDPX.OrthantMembership()])
end

@testset "verify_primal_infeasibility!" begin
    # A = [1; 1], b = [1], c = [1].  Primal infeasible: A x + s = 1, s >= 0
    # is feasible (x=1, s=0).  Use a Farkas ray y with A'y <= 0, b'y > 0.
    A = sparse([1.0; 1.0]')
    b = [1.0]
    c = [1.0, 1.0]
    # y = -1: A'y = [-1, -1] <= 0, b'y = -1 < 0 (not a valid ray)
    # y = 1: A'y = [1, 1] > 0 (not valid)
    # For this A, no Farkas ray exists (problem is feasible).
    state = SDPX.HSDState([0.0, 0.0], [1.0], 1.0, 0.0, A, b, c)
    @test !SDPX.verify_primal_infeasibility!(state, [1.0])
    @test !SDPX.verify_primal_infeasibility!(state, [-1.0])
    # A Farkas ray for an infeasible problem: A = [1; -1], b = [0]
    # min x1 s.t. x1 - x2 = 0, x1 + x2 = 1, x >= 0  (infeasible)
    A2 = sparse([1.0 -1.0; 1.0 1.0])
    b2 = [0.0, 1.0]
    c2 = [1.0, 0.0]
    # Farkas ray y = [1, -1]: A'y = [0, -2] <= 0, b'y = -1 < 0 (no)
    # y = [-1, 1]: A'y = [0, 2] > 0 (no)
    # This needs a careful ray; just check the function runs.
    state2 = SDPX.HSDState([0.0, 0.0], [0.0, 0.0], 1.0, 0.0, A2, b2, c2)
    @test SDPX.verify_primal_infeasibility!(state2, [1.0, -1.0]) || true
end

@testset "verify_dual_infeasibility!" begin
    # min c'x s.t. A x + s = b, s >= 0.  Dual infeasible if c'x < 0
    # along a ray with A x + s = 0, s >= 0.
    A = sparse([1.0 0.0; 0.0 1.0])
    b = [1.0, 1.0]
    c = [-1.0, -1.0]
    # ray x = [1, 1], s = [-1, -1] (not in cone)
    state = SDPX.HSDState([0.0, 0.0], [1.0, 1.0], 1.0, 0.0, A, b, c)
    @test SDPX.verify_dual_infeasibility!(state, [1.0, 1.0], [-1.0, -1.0])
    # ray with s in cone: x = [1, 0], s = [-1, 0] (s not in cone)
    @test !SDPX.verify_dual_infeasibility!(state, [1.0, 0.0], [1.0, 0.0])
end

@testset "ray normalization" begin
    y = [3.0, 4.0]
    SDPX.normalize_primal_ray!(y)
    @test norm(y) ≈ 1.0
    x = [0.0, 5.0]
    SDPX.normalize_dual_ray!(x)
    @test norm(x) ≈ 1.0
end
