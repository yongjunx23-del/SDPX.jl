using Test, TOML, InteractiveUtils, SDPX
include(joinpath(@__DIR__,"power_half_phi_reference.jl"))
const HP=PowerHalfPhiReference
_half_float(s)=reinterpret(Float64,parse(UInt64,s;base=16))
_half_sum(xs)=sum(Rational{BigInt}(x) for x in xs)
function _half_exact(u,v,w,c)
    U,V,W,C=Rational{BigInt}.((u,v,w,c))
    D=4U*V*(1-C)
    Q=W*W*(1+C/2)^2-D # independent factored expression
    return Q,D
end
function _half_reference(u,v,w,c;p=512)
    U,V,W,C=Rational{BigInt}.((u,v,w,c))
    arguments=(W*W/(4U*V),1+C/2,1-C)
    setprecision(BigFloat,p) do
        lower=[setrounding(BigFloat,RoundDown) do
                   log(BigFloat(x))
               end for x in arguments]
        upper=[setrounding(BigFloat,RoundUp) do
                   log(BigFloat(x))
               end for x in arguments]
        lo=setrounding(BigFloat,RoundDown) do
            lower[1]/2+lower[2]-upper[3]/2
        end
        hi=setrounding(BigFloat,RoundUp) do
            upper[1]/2+upper[2]-lower[3]/2
        end
        return lo,hi
    end
end
function _half_check(u,v,w,c)
    out=HP.evaluate(u,v,w,c)
    @test out.status===:ok
    out.status===:ok || return out
    Q,D=_half_exact(u,v,w,c)
    @test _half_sum(out.q_expansion)==Q
    @test _half_sum(out.d_expansion)==D
    @test Rational{BigInt}(out.ledger.Q.lo)<=Q<=Rational{BigInt}(out.ledger.Q.hi)
    @test Rational{BigInt}(out.ledger.D.lo)<=D<=Rational{BigInt}(out.ledger.D.hi)
    @test Rational{BigInt}(out.ledger.Z.lo)<=Q/D<=Rational{BigInt}(out.ledger.Z.hi)
    @test length(out.q_expansion)==20 && length(out.d_expansion)==6
    @test out.ledger.two_prod_calls==13 && out.ledger.two_sum_calls==205
    @test out.ledger.series_degree==12
    @test all(x -> iszero(x) || !issubnormal(x),out.q_expansion)
    @test all(x -> denominator(Rational{BigInt}(x)*(BigInt(1)<<306))==1,out.q_expansion)
    lo,hi=_half_reference(u,v,w,c)
    @test BigFloat(out.lower)<=lo<=hi<=BigFloat(out.upper)
    # Independent mixed relative/absolute quality goal, not production work.
    @test BigFloat(out.radius)<BigFloat(2)^(-44)*max(abs(lo),abs(hi))+BigFloat(2)^(-98)
    @test out.lower<=out.estimate<=out.upper
    @test Rational{BigInt}(out.radius)>=max(Rational{BigInt}(out.estimate)-Rational{BigInt}(out.lower),
                                          Rational{BigInt}(out.upper)-Rational{BigInt}(out.estimate))
    return out
end

@testset "native EFT and execution contract" begin
    @test HP._runtime_ok()
    @test !isdefined(SDPX,:PowerHalfPhiReference)
    x=1.0+0x1p-27;y=1.0-0x1p-27
    hi,lo=HP._two_prod(x,y)
    @test hi==1.0 && lo==-0x1p-54
    @test hi-1.0==0.0 && fma(x,y,-1.0)==-0x1p-54
    for a in (0.0,0x1p-8,nextfloat(0x1p-8),prevfloat(1.0),nextfloat(1.0),0x1p8),
        b in (-0x1p8,-1.0,0.0,0x1p-40,nextfloat(0.5),0x1p8)
        h,l=HP._two_prod(a,b)
        @test Rational{BigInt}(h)+Rational{BigInt}(l)==Rational{BigInt}(a)*Rational{BigInt}(b)
        s,e=HP._two_sum(a,b)
        @test Rational{BigInt}(s)+Rational{BigInt}(e)==Rational{BigInt}(a)+Rational{BigInt}(b)
    end
    ir=sprint(io -> code_llvm(io,HP._two_prod,Tuple{Float64,Float64};debuginfo=:none))
    native=sprint(io -> code_native(io,HP._two_prod,Tuple{Float64,Float64};debuginfo=:none))
    sum_ir=sprint(io -> code_llvm(io,HP._two_sum,Tuple{Float64,Float64};debuginfo=:none))
    @test occursin("llvm.fma.f64",ir)
    @test occursin(r"\b(fmadd|fmsub|fnmadd|fnmsub)\b",native)
    @test !occursin(r"\b(fadd|fsub|fmul|fdiv)\s+(fast|reassoc|contract|nnan|ninf|afn|nsz|arcp)\b",ir*sum_ir)
    if haskey(ENV,"HALF_PHI_OUT")
        write(joinpath(ENV["HALF_PHI_OUT"],"two-prod.ll"),ir)
        write(joinpath(ENV["HALF_PHI_OUT"],"two-prod.asm"),native)
        write(joinpath(ENV["HALF_PHI_OUT"],"two-sum.ll"),sum_ir)
    end
end

const _HALF_FIXTURES=TOML.parsefile(joinpath(@__DIR__,"fixtures/half_phi_native.toml"))["points"]
@testset "all retained root currents: exact expansions and directed Phi containment" begin
    for f in _HALF_FIXTURES
        u,v,w=_half_float.(f["dual_bits"])
        @test length(f["current_bits"])==64
        for current in f["current_bits"]
            c=_half_float(current)
            _half_check(u,v,w,c)
        end
        c=_half_float(last(f["current_bits"]))
        a,b=_half_reference(u,v,w,c;p=512)
        c1,d=_half_reference(u,v,w,c;p=1024)
        @test max(a,c1)<=min(b,d)
        out=HP.evaluate(u,v,w,c)
        @test out.radius<0x1p-98 # fixed absolute goal at the retained cycle
        println("HALF_PHI ",f["id"]," estimate=",out.estimate," radius=",out.radius)
    end
end

@testset "exact cancellation, sign and domain-edge controls" begin
    c=1.0-(1.0-0x1p-10)^2
    u=1.0+c/2;v=u;w=2.0*(1.0-0x1p-10)
    @test first(_half_exact(u,v,w,c))==0
    _half_check(u,v,w,c)
    for W in (prevfloat(w),nextfloat(w),-w,-prevfloat(w),-nextfloat(w))
        _half_check(u,v,W,c)
    end
    @test first(_half_exact(u,v,prevfloat(w),c))<0
    @test first(_half_exact(u,v,nextfloat(w),c))>0
    for C in (0x1p-40,nextfloat(0x1p-40),prevfloat(0x1p-8),0x1p-8)
        _half_check(1.0,1.0,2.0-2C,C)
    end
    _half_check(0x1p-8,0x1p-8,0x1p-7,0x1p-40)
    _half_check(0x1p8,0x1p-8,2.0,0x1p-40)
    _half_check(0x1p7,0x1p7,0x1p8,0x1p-40)
    # Deliberately discard product residuals: a native interval around the
    # already-corrupted Q cannot enclose the exact-input target.
    f=first(_HALF_FIXTURES);u,v,w=_half_float.(f["dual_bits"]);c=_half_float(last(f["current_bits"]))
    p=w*w;cc=4*(u*v);q=(p-cc)+(p+cc)*c+p*c*c/4;d=cc*(1-c)
    bad=HP._series(HP._divide(HP._enclose_sum([q]),HP._enclose_sum([d])))
    lo,hi=_half_reference(u,v,w,c)
    @test !(BigFloat(bad.lo)<=lo<=hi<=BigFloat(bad.hi))
    good=HP.evaluate(u,v,w,c)
    @test !(BigFloat(good.estimate)<=lo<=hi<=BigFloat(good.estimate)) # understated zero radius
end

@testset "fail-closed unsupported contexts and inputs" begin
    for args in ((1.0,1.0,0.0,0x1p-20),(0.0,1.0,1.0,0x1p-20),
                 (-1.0,1.0,1.0,0x1p-20),(1.0,1.0,2.0,0.0),
                 (prevfloat(0x1p-8),1.0,2.0,0x1p-20),(nextfloat(0x1p8),1.0,2.0,0x1p-20),
                 (1.0,1.0,nextfloat(0.0),0x1p-20),(1.0,1.0,2.0,prevfloat(0x1p-40)),
                 (1.0,1.0,2.0,nextfloat(0x1p-8)),(Inf,1.0,2.0,0x1p-20),
                 (1.0,NaN,2.0,0x1p-20),(1.0,1.0,-Inf,0x1p-20),
                 (Float32(1),1.0,2.0,0x1p-20),(BigFloat(1),1.0,2.0,0x1p-20))
        @test HP.evaluate(args...).status===:unsupported
    end
    @test HP.evaluate(1.0,1.0,1.0,0x1p-20).reason===:series_domain
    @test HP.evaluate(1.0,1.0,2.0,0x1p-20;alpha=nextfloat(0.5)).reason===:alpha
    @test_throws HP.ArithmeticDomainError HP._divide(HP.Interval(1,1),HP.Interval(-1,0))
    raw=Base.Rounding.rounding_raw(Float64)
    try
        Base.Rounding.setrounding_raw(Float64,Base.Rounding.JL_FE_UPWARD)
        @test HP.evaluate(1.0,1.0,2.0,0x1p-20).reason===:runtime_context
        @test Base.Rounding.rounding_raw(Float64)==Base.Rounding.JL_FE_UPWARD
    finally
        Base.Rounding.setrounding_raw(Float64,raw)
    end
    old=get_zero_subnormals()
    try
        @test set_zero_subnormals(true)
        @test HP.evaluate(1.0,1.0,2.0,0x1p-20).reason===:runtime_context
        @test get_zero_subnormals()
    finally
        set_zero_subnormals(old)
    end
    @test HP._runtime_ok()
    c=0x1p-20
    first=HP.evaluate(1.0,1.0,2.0-2c,c)
    saved_q=copy(first.q_expansion);saved_d=copy(first.d_expansion)
    first.q_expansion[1]=999.0;first.d_expansion[1]=-999.0
    second=HP.evaluate(1.0,1.0,2.0-2c,c)
    @test second.q_expansion==saved_q && second.d_expansion==saved_d
    @test second.q_expansion!==first.q_expansion && second.d_expansion!==first.d_expansion
end
