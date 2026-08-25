# Wave E (setup-only): ReductionPlan types test.
using SDPX
using Test
using LinearAlgebra

@testset "ReductionPlan setup" begin
    # Identity plan round-trips.
    plan = SDPX.identity_reduction_plan(Float64, 4)
    @test plan.kind === :none
    @test plan.original_dimension == 4
    @test plan.reduced_dimension == 4
    x = [1.0, 2.0, 3.0, 4.0]
    @test SDPX.apply_forward(plan, x) == x
    @test SDPX.apply_backward(plan, x) == x
    # A non-trivial permutation round-trips.
    perm = [3, 1, 4, 2]
    fwd = SDPX.ForwardMap{Float64}(perm, [4], Matrix{Float64}(I, 4, 4))
    bwd = SDPX.BackwardMap{Float64}(invperm(perm), [4], Matrix{Float64}(I, 4, 4))
    rp = SDPX.ReductionPlan{Float64}(:symmetry, fwd, bwd, SDPX.CertificateMap{Float64}(fwd, bwd), 4, 4)
    reduced = SDPX.apply_forward(rp, x)
    @test reduced == x[perm]
    @test SDPX.apply_backward(rp, reduced) == x
end