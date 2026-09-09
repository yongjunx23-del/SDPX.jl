# R2 narrow qualification suite (NOT full R2 closure):
# R2-A: symbolic/numeric separation gate is exercised in
#       validation/scientific_core/test_r2a_symbolic_numeric_separation.jl
#       (Warm100=0, Cold100=1).
# R2-B: invalidation transactions (CSC/dimensions/precision/provider/threads,
#       fail injection and recovery).
# R2-C: sequential session owner isolation (NOT multithread qualification).
# R2-D: allocation variation across repeated solves (NOT a retained-live/peak bound).

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

@testset "R2-C: sequential session owner isolation (NOT multithread qualification)" begin
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

    # NOTE: this testset runs sessions sequentially and flips `busy`
    # manually. It verifies disjoint ownership and same-session rejection
    # only; it does NOT establish multithread/task-level concurrency
    # qualification (that remains an open R2 gate).
    # Concurrent solve on the same session is strictly forbidden
    s1.state.busy = true
    @test_throws ArgumentError SDPX.solve!(s1; objective=c0, rhs=b0)
    s1.state.busy = false
end

@testset "R2-D: allocation variation across repeated solves (NOT a peak bound)" begin
    prob = _test_lp()
    options = SDPX.SolverOptions{Float64}(; verbosity=0, timing=true, threads=1)
    prep = SDPX.prepare(prob, options)

    c0 = Float64[1.0, 2.0, 3.0]
    b0 = Float64[1.5]
    SDPX.solve!(prep; objective=c0, rhs=b0)

    # Track allocations over 10 repeated solves. This bounds allocation
    # VARIATION only; it is not a retained-live-object/peak/RSS bound and
    # does not establish a complete phase accounting.
    allocs = Int[]
    for k in 1:10
        c_k = Float64[1.0 + 0.01*k, 2.0, 3.0]
        b_k = Float64[1.5 + 0.01*k]
        bytes = @allocated SDPX.solve!(prep; objective=c_k, rhs=b_k)
        push!(allocs, bytes)
    end
    @test maximum(allocs[2:end]) - minimum(allocs[2:end]) < 50_000
end

@testset "R2: primary exception survives a failing check-in (dual failure)" begin
    # Approved lease protocol: cleanup must not strand `busy`, retain an active
    # owner, or mask the primary exception.  This injects a check-in failure at
    # the real call site while the solve body is already failing (NaN objective)
    # and requires the primary ArgumentError to propagate.
    prob = _test_lp()
    options = _test_options()
    prep = SDPX.prepare(prob, options)
    c0 = Float64[1.0, 2.0, 3.0]
    b0 = Float64[1.5]
    first = SDPX.solve!(prep; objective=c0, rhs=b0)
    @test first.status == SDPX.Optimal
    @test prep.state.symbolic_slot.entry !== nothing

    SDPX._LEASE_CHECKIN_FAULT[] = () -> error("injected check-in failure")
    try
        @test_throws ArgumentError SDPX.solve!(
            prep; objective=Float64[NaN, 2.0, 3.0], rhs=b0,
        )
    finally
        SDPX._LEASE_CHECKIN_FAULT[] = nothing
    end
    # Session must not be stranded, and the failed update must not retain a
    # stale factor.
    @test prep.state.busy == false
    free = trylock(prep.state.lock)
    free && unlock(prep.state.lock)
    @test free
    @test prep.state.symbolic_slot.entry === nothing
    # Recovery still works.
    recovered = SDPX.solve!(prep; objective=c0, rhs=b0)
    @test recovered.status == SDPX.Optimal
end

@testset "R2-C: multi-task concurrent session isolation and collision" begin
    prob = _test_lp()
    options = _test_options()
    c0 = Float64[1.0, 2.0, 3.0]
    b0 = Float64[1.5]

    # Part 1 (independent sessions, concurrent): two tasks solve two distinct
    # prepared sessions from the same problem. Neither task may interfere
    # with the other.
    s1 = SDPX.prepare(prob, options)
    s2 = SDPX.prepare(prob, options)
    r1_box = Ref{Any}(nothing)
    r2_box = Ref{Any}(nothing)
    Base.@sync begin
        Base.Threads.@spawn begin
            try
                r1_box[] = SDPX.solve!(s1; objective=copy(c0), rhs=copy(b0))
            catch e
                r1_box[] = e
            end
        end
        Base.Threads.@spawn begin
            try
                r2_box[] = SDPX.solve!(s2; objective=copy(c0), rhs=copy(b0))
            catch e
                r2_box[] = e
            end
        end
    end
    @test !(r1_box[] isa Exception)
    @test !(r2_box[] isa Exception)
    @test r1_box[].status == SDPX.Optimal
    @test r2_box[].status == SDPX.Optimal
    # Bit-identical solutions across independent concurrent sessions.
    @test r1_box[].x == r2_box[].x
    # Distinct slot entries: no shared mutable factor state.
    @test s1.state.symbolic_slot !== s2.state.symbolic_slot
    @test s1.state.symbolic_slot.entry !== s2.state.symbolic_slot.entry
    @test s1.state.symbolic_slot.entry.cache.factor !== s2.state.symbolic_slot.entry.cache.factor

    # Part 2 (same session collision): two tasks race on the SAME prepared
    # session. Reaching this point without hanging proves no deadlock. One
    # task must succeed with `Optimal` and the loser must be rejected with
    # `ArgumentError` ("PreparedSolver is sequential"), unless the runtime
    # schedules the tasks strictly one after another, in which case both
    # report `Optimal`.
    s_shared = SDPX.prepare(prob, options)
    outcomes = Vector{Any}(undef, 2)
    Base.@sync begin
        Base.Threads.@spawn begin
            try
                outcomes[1] = SDPX.solve!(s_shared; objective=copy(c0), rhs=copy(b0))
            catch e
                outcomes[1] = e
            end
        end
        Base.Threads.@spawn begin
            try
                outcomes[2] = SDPX.solve!(s_shared; objective=copy(c0), rhs=copy(b0))
            catch e
                outcomes[2] = e
            end
        end
    end
    _is_optimal(x) = !(x isa Exception) && hasproperty(x, :status) && x.status == SDPX.Optimal
    _is_sequential_rejection(x) = (x isa ArgumentError) && occursin("PreparedSolver is sequential", x.msg)
    n_optimal = count(_is_optimal, outcomes)
    n_rejected = count(_is_sequential_rejection, outcomes)
    @test (n_optimal == 1 && n_rejected == 1) || (n_optimal == 2)

    # Part 3 (post-collision recovery): the session must not be stranded and
    # must remain fully functional.
    @test s_shared.state.busy == false
    r_after = SDPX.solve!(s_shared; objective=c0, rhs=b0)
    @test r_after.status == SDPX.Optimal
end
