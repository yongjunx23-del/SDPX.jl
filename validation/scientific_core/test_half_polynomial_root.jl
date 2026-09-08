using Test, TOML, SDPX
include("power_half_root_geometry.jl")
include("half_power_polynomial_root.jl")
const HR=HalfPowerPolynomialRoot
const RG=PowerHalfRootGeometry
const Q=Rational{BigInt}
const POLYNOMIAL_ROOT_RESULTS=Any[]
poly(u,v,w,c)=w*w*(1+c/2)^2-4u*v*(1-c)
function reference_root(u,v,w)
    iszero(w) && return Q(1),Q(1)
    a=Q(0);b=Q(1)
    @assert poly(u,v,w,a)<0<poly(u,v,w,b)
    for _ in 1:256
        c=(a+b)/2
        if poly(u,v,w,c)<0;a=c;else;b=c;end
    end
    a,b
end
function check_root(u,v,w;warm=nothing)
    out=HR.root(u,v,w;warm)
    @test out.status===:qualified
    U,V,W=Q.((u,v,w));a,b=reference_root(U,V,W)
    @test Q(out.lower)<=a<=b<=Q(out.upper)
    @test out.lower<=out.candidate<=out.upper
    @test Q(out.radius)>=max(Q(out.candidate)-Q(out.lower),Q(out.upper)-Q(out.candidate))
    @test Q(out.radius)<=Q(out.tolerance)*Q(out.lower)
    @test out.iterations<=64 && out.midpoint_probes<=512 && out.endpoint_evaluations<=2
    for step in out.trace
        @test Q(step.lower)<=a<=b<=Q(step.upper)
        @test Q(step.new_lower)<=a<=b<=Q(step.new_upper)
        exact=poly(U,V,W,Q(step.probe))
        @test Q(step.value_lower)<=exact<=Q(step.value_upper)
        for c in (Q(step.lower),Q(step.upper))
            derivative=W*W+4U*V+W*W*c/2
            @test Q(step.derivative_lower)<=derivative<=Q(step.derivative_upper)
        end
    end
    push!(POLYNOMIAL_ROOT_RESULTS,(;u,v,w,warm,out))
    out
end
@testset "full-gap half-Power polynomial interval root" begin
    @test !isdefined(SDPX,:HalfPowerPolynomialRoot)
    for u in (0.25,1.,8.),v in (0.25,1.,8.),fraction in (0.,0.125,0.5,0.9,1-0x1p-20),sign in (-1.,1.)
        w=sign*2sqrt(u*v)*fraction
        cold=check_root(u,v,w)
        if !iszero(w)
            warm=check_root(u,v,w;warm=cold.candidate)
            @test max(cold.lower,warm.lower)<=min(cold.upper,warm.upper)
        end
    end
    fixtures=TOML.parsefile(joinpath(@__DIR__,"fixtures/half_root_geometry.toml"))["points"]
    floatword(s)=reinterpret(Float64,parse(UInt64,s;base=16))
    for f in fixtures
        u,v,w=floatword.(f["dual_bits"]);gap=floatword(f["accepted_gap_bits"])
        check_root(u,v,w);check_root(u,v,w;warm=gap)
    end
    # Direct exact-polynomial enclosure controls include both conceptual
    # endpoints, not the narrower log-series evaluator's c-domain.
    for c in (0.,HR.MIN_GAP,0.01,0.5,prevfloat(1.),1.)
        out=HR.polynomial(1.,2.,-1.,c)
        @test Q(out.lo)<=poly(Q(1),Q(2),Q(-1),Q(c))<=Q(out.hi)
    end
    @test HR.root(1.,1.,2.).status===:unsupported
    @test HR.root(1.,1.,2.1).status===:unsupported
    @test HR.root(1.,1.,prevfloat(2.)).status===:unsupported
    @test HR.root(1.,1.,0.5;alpha=0.4).status===:unsupported
    @test HR.root(1.,1.,0.5;max_iterations=0).status===:unsupported
    @test HR.root(1.,1.,0.5;max_iterations=1).status===:unsupported
    @test HR.root(1.,1.,0.5;max_bisections=0).status===:unsupported
    @test HR.root(1.,1.,0.5;warm=0.).status===:unsupported
    @test HR.root(1.,1.,0.5;tolerance=NaN).status===:unsupported
    @test HR.root(Float32(1),1.,0.5).status===:unsupported
end
