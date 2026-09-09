module PowerHalfPhiReference
# DISCONNECTED arithmetic qualification. Not loaded by SDPX; no root decisions.
export evaluate, Interval

struct Interval
    lo::Float64
    hi::Float64
end
struct ArithmeticDomainError <: Exception end
_normalzero(x) = isfinite(x) && (iszero(x) || !issubnormal(x))
function _eft_check(xs)
    all(_normalzero,xs) || throw(ArithmeticDomainError())
end
# Private transforms used only on the proved construction domain (see companion
# proof). Final finiteness alone does not establish their error-free identity.
@noinline function _two_prod(a::Float64,b::Float64)
    p=a*b
    e=fma(a,b,-p) # explicit single-rounding FMA, NOT muladd
    _eft_check((a,b,p,e))
    return p,e
end
@noinline function _two_sum(a::Float64,b::Float64)
    s=a+b
    bv=s-a
    av=s-bv
    br=b-bv
    ar=a-av
    e=ar+br
    _eft_check((a,b,s,bv,av,br,ar,e))
    return s,e
end
function _grow(terms)
    expansion=Float64[]
    calls=0
    for term in terms
        q=term
        for i in eachindex(expansion)
            q,error=_two_sum(q,expansion[i])
            expansion[i]=error;calls+=1
        end
        push!(expansion,q) # retain EVERY component, including signed zeros
    end
    return expansion,calls
end

# Every rounded primitive gets its own outward step. FTZ must be disabled.
_add(x::Interval,y::Interval)=Interval(prevfloat(x.lo+y.lo),nextfloat(x.hi+y.hi))
function _mul(x::Interval,y::Interval)
    p=(x.lo*y.lo,x.lo*y.hi,x.hi*y.lo,x.hi*y.hi)
    return Interval(prevfloat(minimum(p)),nextfloat(maximum(p)))
end
function _divide(x::Interval,y::Interval)
    y.lo>0 || throw(ArithmeticDomainError())
    p=(x.lo/y.lo,x.lo/y.hi,x.hi/y.lo,x.hi/y.hi)
    return Interval(prevfloat(minimum(p)),nextfloat(maximum(p)))
end
function _enclose_sum(expansion)
    result=Interval(0.0,0.0)
    for x in expansion
        result=_add(result,Interval(x,x))
    end
    return result
end
function _coefficient(k)
    value=(isodd(k) ? 1.0 : -1.0)/Float64(k)
    return Interval(prevfloat(value),nextfloat(value))
end
function _series(z::Interval)
    p=_coefficient(12)
    for k in 11:-1:1
        p=_add(_mul(p,z),_coefficient(k))
    end
    p=_mul(p,z)
    # |z|<=2^-8: tail <= |z|^13/(13(1-|z|)) < 2^-107.
    p=_add(p,Interval(-0x1p-107,0x1p-107))
    return _mul(p,Interval(0.5,0.5))
end

function _construct(u,v,w,c)
    P=_two_prod(w,w)
    uv=_two_prod(u,v)
    C=(ldexp(uv[1],2),ldexp(uv[2],2))
    cc=_two_prod(c,c)
    q=Float64[P[1],P[2],-C[1],-C[2]]
    for x in (P...,C...)
        pair=_two_prod(x,c);append!(q,pair)
    end
    for x in P,y in cc
        pair=_two_prod(x,y)
        push!(q,ldexp(pair[1],-2),ldexp(pair[2],-2))
    end
    d=Float64[C[1],C[2]]
    for x in C
        pair=_two_prod(x,c);push!(d,-pair[1],-pair[2])
    end
    @assert length(q)==20 && length(d)==6
    qe,qcalls=_grow(q);de,dcalls=_grow(d)
    @assert qcalls==190 && dcalls==15
    return qe,de
end

_runtime_ok() = VERSION==v"1.12.6" && Sys.ARCH===:aarch64 && Sys.KERNEL===:Darwin &&
    Base.GIT_VERSION_INFO.commit=="15346901f0039751c5488744f1f62de7d87510a8" &&
    Base.JLOptions().fast_math==0 && rounding(Float64)==RoundNearest &&
    !get_zero_subnormals() && Core.Intrinsics.have_fma(Float64)
unsupported(reason)=(status=:unsupported,reason=reason)

"""Bounded Float64 alpha=1/2 Phi enclosure. Never a solver/root acceptance.
No log/log1p, BigFloat, BigInt, wider scalar type, or precision/context mutation.
The explicit expansion network and runtime assumptions are qualification scope.
"""
function evaluate(u,v,w,c;alpha=0.5)
    all(x -> x isa Float64,(u,v,w,c,alpha)) || return unsupported(:type)
    _runtime_ok() || return unsupported(:runtime_context)
    reinterpret(UInt64,alpha)==0x3fe0000000000000 || return unsupported(:alpha)
    all(isfinite,(u,v,w,c)) || return unsupported(:nonfinite)
    0x1p-8<=u<=0x1p8 && 0x1p-8<=v<=0x1p8 && 0x1p-8<=abs(w)<=0x1p8 &&
        0x1p-40<=c<=0x1p-8 || return unsupported(:input_domain)
    try
        qe,de=_construct(u,v,w,c)
        Q=_enclose_sum(qe);D=_enclose_sum(de)
        D.lo>0 || return unsupported(:denominator)
        Z=_divide(Q,D)
        -0x1p-8<=Z.lo<=Z.hi<=0x1p-8 || return unsupported(:series_domain)
        phi=_series(Z)
        all(isfinite,(phi.lo,phi.hi)) && phi.lo<=phi.hi || return unsupported(:interval)
        # Bounded representative, not an analytic root or a reconstructed entry.
        estimate=clamp(phi.lo/2+phi.hi/2,phi.lo,phi.hi)
        radius=nextfloat(max(estimate-phi.lo,phi.hi-estimate))
        return (status=:ok,reason=:qualified_domain,estimate=estimate,lower=phi.lo,upper=phi.hi,
            radius=radius,ledger=(Q=Q,D=D,Z=Z,remainder=0x1p-107,
                two_prod_calls=13,two_sum_calls=205,series_degree=12),
            q_expansion=qe,d_expansion=de)
    catch error
        error isa ArithmeticDomainError || rethrow()
        return unsupported(:arithmetic_domain)
    end
end
end
