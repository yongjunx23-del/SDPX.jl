using SDPX, MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra
using Test, Serialization
function lp_state(T,gauge)
    model=SDPX.Model(T)
    x=SDPX.variable!(model,:x,1;domain=SDPX.Reals())
    SDPX.constraint!(model,:lower,x[1]-one(T),SDPX.Nonnegative())
    SDPX.objective!(model,SDPX.Minimize(),x[1])
    canonical=SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    b=SDPX.HSDState(canonical)
    @assert b.n==b.m==1
    SDPX._store_owned_scalar!(b.x,1,gauge)
    SDPX._store_owned_scalar!(b.y,1,gauge)
    SDPX._store_owned_scalar!(b.s,1,zero(T))
    b.tau=gauge;b.kappa=zero(T)
    SDPX._cert_residual!(b)
    return b
end
function verify_candidate(canonical,candidate)
    T=eltype(candidate.x)
    SDPX.verify_optimal!(canonical,candidate,
        SDPX.alloc_zeros(T,candidate.n),SDPX.alloc_zeros(T,candidate.m),SDPX.alloc_zeros(T,candidate.m);tol=T(1e-8))
end
function snapshot(b)
    deepcopy((b.x,b.y,b.s,b.rP,b.rD,b.xt,b.yt,b.st,b.tau,b.kappa,b.mu,b.A,b.b,b.c))
end
@testset "Narrow normalized optimality candidate" begin
    for T in (Float64,Float64x4,BigFloat)
        b=lp_state(T,T(2)^(-80));before=snapshot(b)
        c=SDPX._normalized_optimality_candidate(b)
        @test c !== nothing
        @test c.A===b.A && c.b===b.b && c.c===b.c
        @test verify_candidate(b.canonical,c)
        @test snapshot(b)==before
        @test !hasproperty(c,:workspace)
        @test c.x!==b.x && c.s!==b.s && c.y!==b.y
        if T===BigFloat
            entries=vcat(c.x,c.y,c.s,c.rP,c.rD,c.xt,c.yt,c.st)
            @test length(unique(objectid.(entries)))==length(entries)
        end
        SDPX._store_owned_scalar!(c.x,1,T(7))
        @test snapshot(b)==before
        @test !verify_candidate(b.canonical,c)
        c=SDPX._normalized_optimality_candidate(b);c.kappa=one(T)
        @test !verify_candidate(b.canonical,c)
        c=SDPX._normalized_optimality_candidate(b);SDPX._store_owned_scalar!(c.s,1,-one(T))
        @test !verify_candidate(b.canonical,c)
        for invalid in (zero(T),-one(T),T(Inf),T(NaN))
            bad=lp_state(T,T(2)^(-80));bad.tau=invalid
            @test SDPX._normalized_optimality_candidate(bad)===nothing
        end
        bad=lp_state(T,T(2)^(-80));bad.kappa=-one(T)
        @test SDPX._normalized_optimality_candidate(bad)===nothing
        bad=lp_state(T,T(2)^(-80));SDPX._store_owned_scalar!(bad.x,1,T(NaN))
        @test SDPX._normalized_optimality_candidate(bad)===nothing
    end
    # A reciprocal-overflow gauge can still have finite recovered coordinates.
    b=lp_state(Float64,nextfloat(0.0));before=snapshot(b)
    @test isinf(inv(b.tau))
    c=SDPX._normalized_optimality_candidate(b)
    @test c!==nothing && verify_candidate(b.canonical,c)
    @test snapshot(b)==before
    b=lp_state(Float64,2.0^-80);b.x[1]=floatmax(Float64)
    @test SDPX._normalized_optimality_candidate(b)===nothing
end
@testset "Normalized result retains a replayable owned gauge" begin
    for T in (Float64,Float64x4,BigFloat)
        gauges=T===Float64 ? (T(2)^(-80),nextfloat(0.0)) : (T(2)^(-80),)
        for gauge in gauges
            b=lp_state(T,gauge)
            state=SDPX._product_cone_hsd_state(b)
            before=snapshot(b)
            x=SDPX.alloc_zeros(T,b.n);s=SDPX.alloc_zeros(T,b.m);y=SDPX.alloc_zeros(T,b.m)
            r=SDPX._product_hsd_verified_result(state,x,s,y,T(1e-8),
                SDPX.ProductHSDVerifiedAcceptedStep,SDPX.HSDStepOK)
            @test r!==nothing && r.status===SDPX.ProductHSDOptimal
            @test isfinite(r.normalized_residual) && r.normalized_residual<=T(1e-8)
            @test r.tau==one(T)
            @test snapshot(b)==before
            replay=lp_state(T,one(T))
            SDPX.copy_owned!(replay.x,r.hsd_x);SDPX.copy_owned!(replay.s,r.hsd_s);SDPX.copy_owned!(replay.y,r.hsd_y)
            replay.tau=r.tau;replay.kappa=r.kappa;replay.mu=r.mu
            @test verify_candidate(b.canonical,replay)
            SDPX._store_owned_scalar!(x,1,T(7))
            @test r.x==T[1]
            SDPX._store_owned_scalar!(b.x,1,T(9))
            @test r.hsd_x==T[1]
        end
    end
end
@testset "Saved N6 replay without copying Newton workspace" begin
    b=deserialize(ENV["N6_REPLAY_INPUT"])
    before=snapshot(b)
    c=SDPX._normalized_optimality_candidate(b)
    @test c!==nothing
    @test verify_candidate(b.canonical,c)
    @test snapshot(b)==before
    @test c.A===b.A && c.b===b.b && c.c===b.c
    candidate_bytes=@allocated SDPX._normalized_optimality_candidate(b)
    deepcopy(b) # Warm up the comparison; cumulative allocation, not RSS.
    deepcopy_bytes=@allocated deepcopy(b)
    println("CANDIDATE_ALLOCATED=",candidate_bytes," DEEPCOPY_ALLOCATED=",deepcopy_bytes)
end
