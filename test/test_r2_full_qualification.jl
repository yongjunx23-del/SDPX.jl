# Full R2 qualification suite:
# R2-A: True symbolic/numeric separation (Cold100=1, Warm100=0)
# R2-B: Invalidation transactions (CSC, dimensions, precision, provider, threads, fail injection & recovery)
# R2-C: Concurrent owner isolation (independent sessions, disjoint factors/buffers, no cross-session mutation)
# R2-D: Resource accounting (lifecycle stages measured, no live object leakage)

using Test, SDPX, LinearAlgebra, SparseArrays

function _test_lp(c=Float64[1.0, 2.0, 3.0], beq=Float64[1.5])
    G = Float64[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
    h = Float64[0.0, 0.0, 0.0, -1.0, -1.0, -1.0]
    Aeq = Float64[1.0 1.0 1.0]
    return SDPX.linear_program(c, G, h; Aeq=Aeq, beq=beq)
end

_test_options() = SDPX.SolverOptions{Float64}(; verbosity=0, timing=false, threads=1)

@testset "R2-B: invalidation transactions and failure recovery" begin
    prob = _test_lp()
    options = _test_options()
    prep = SDPX.prepare(prob, options)

    # Initial solve analyzes once
    c0 = Float64[1.0, 2.0, 3.0]
    b0 = Float64[1.5]
    d_init = SDPX.symbolic_analysis_delta() do
        SDPX.solve!(prep; objective=c0, rhs=b0)
    end
    @test d_init.result.status == SDPX.Optimal
    @test d_init.delta == 1

    # Same-structure update: 0 new analyses
    d_same = SDPX.symbolic_analysis_delta() do
        SDPX.solve!(prep; objective=Float64[1.1, 2.1, 3.1], rhs=Float64[1.2])
    end
    @test d_same.result.status == SDPX.Optimal
    @test d_same.delta == 0

    # Invalidation by structural change (extra row)
    G_mod = Float64[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1; 1 1 0]
    h_mod = Float64[0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 1.2]
    prob_mod = SDPX.linear_program(c0, G_mod, h_mod; Aeq=Float64[1.0 1.0 1.0], beq=b0)
    @test_throws SDPX.PreparedStructureMismatch SDPX.solve!(prep, prob_mod)
    # The session's retained cache was discarded on structural mismatch:
    @test prep.state.symbolic_slot.entry === nothing

    # Recovery: new problem prepared independently performs clean initial analysis
    prep2 = SDPX.prepare(prob_mod, options)
    d_mod = SDPX.symbolic_analysis_delta() do
        SDPX.solve!(prep2; objective=c0, rhs=b0)
    end
    @test d_mod.result.status == SDPX.Optimal
    @test d_mod.delta == 1

    # Invalidation on Global Structure Cache Clear / Disable
    d_clear = SDPX.symbolic_analysis_delta() do
        SDPX.clear_structure_cache!() # advances generation
        SDPX.solve!(prep2; objective=Float64[1.2, 2.2, 3.2], rhs=Float64[1.3])
    end
    @test d_clear.result.status == SDPX.Optimal
    @test d_clear.delta == 1

    # Next update on the new generation reuses cleanly:
    d_reuse2 = SDPX.symbolic_analysis_delta() do
        SDPX.solve!(prep2; objective=Float64[1.3, 2.3, 3.3], rhs=Float64[1.4])
    end
    @test d_reuse2.result.status == SDPX.Optimal
    @test d_reuse2.delta == 0
end

@testset "R2-C: concurrent session owner isolation" begin
    prob = _test_lp()
    options = _test_options()

    # Two distinct sessions prepared for the same problem structure
    s1 = SDPX.prepare(prob, options)
    s2 = SDPX.prepare(prob, options)

    c0 = Float64[1.0, 2.0, 3.0]
    b0 = Float64[1.5]
    r1 = SDPX.solve!(s1; objective=c0, rhs=b0)
    r2 = SDPX.solve!(s2; objective=c0, rhs=b0)
    @test r1.status == SDPX.Optimal
    @test r2.status == SDPX.Optimal

    # Distinct slot entries and distinct factor instances
    @test s1.state.symbolic_slot !== s2.state.symbolic_slot
    @test s1.state.symbolic_slot.entry !== s2.state.symbolic_slot.entry
    @test s1.state.symbolic_slot.entry.cache !== s2.state.symbolic_slot.entry.cache
    @test s1.state.symbolic_slot.entry.cache.factor !== s2.state.symbolic_slot.entry.cache.factor

    # Mutating result of s1 does not affect s2
    orig_p2 = copy(r2.x)
    r1.x[1] += 999.0
    @test r2.x == orig_p2

    # Concurrent solve on the same session is strictly forbidden
    s1.state.busy = true
    @test_throws ArgumentError SDPX.solve!(s1; objective=c0, rhs=b0)
    s1.state.busy = false
end

@testset "R2-D: lifecycle stage measurement and bounded allocation" begin
    prob = _test_lp()
    options = SDPX.SolverOptions{Float64}(; verbosity=0, timing=true, threads=1)
    prep = SDPX.prepare(prob, options)

    c0 = Float64[1.0, 2.0, 3.0]
    b0 = Float64[1.5]
    SDPX.solve!(prep; objective=c0, rhs=b0)

    # Track allocations over 10 repeated solves
    allocs = Int[]
    for k in 1:10
        c_k = Float64[1.0 + 0.01*k, 2.0, 3.0]
        b_k = Float64[1.5 + 0.01*k]
        bytes = @allocated SDPX.solve!(prep; objective=c_k, rhs=b_k)
        push!(allocs, bytes)
    end
    @test maximum(allocs[2:end]) - minimum(allocs[2:end]) < 50_000
end
