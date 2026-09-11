module ExactInputs
using MultiFloats, Random, SHA, TOML
const MF = Float64x4
const GRID = 1074
const PRODUCT_GRID = 2GRID
const UNIT = big(1) << PRODUCT_GRID
const MIXED_TOL = big(31) // big(10)^62 # exact 3.1e-61; not floor-free relative error

# Exact value of binary64 * 2^1074. Signed-zero bits remain in the fixture,
# while its mathematical integer value is zero. No MPFR/MFA/MFLA oracle call.
function scaled(x::Float64)
    bits=reinterpret(UInt64,x); exponent=Int((bits>>52)&0x7ff)
    exponent==0x7ff && throw(ArgumentError("nonfinite binary64 limb"))
    mantissa=bits&0x000fffffffffffff
    value=exponent==0 ? BigInt(mantissa) : BigInt(mantissa|0x0010000000000000)<<(exponent-1)
    return bits>>63==0 ? value : -value
end
scaled(x::MF)=sum(scaled, x._limbs;init=big(0))
words(x::MF)=[string(reinterpret(UInt64,v);base=16,pad=16) for v in x._limbs]
function fromwords(data)
    length(data)==4 || throw(ArgumentError("four limbs required"))
    for s in data
        length(s)==16 && all(c->c in "0123456789abcdef",s) || throw(ArgumentError("invalid limb word"))
    end
    x=MF(ntuple(i->reinterpret(Float64,parse(UInt64,data[i];base=16)),4))
    all(isfinite,x._limbs) || throw(ArgumentError("nonfinite fixture"))
    words(x)==data || error("tuple construction changed stored words")
    return x
end
encode(A::Matrix{MF})=Dict("shape"=>collect(size(A)),"words"=>reduce(vcat,words.(vec(A));init=String[]))
function decode(data)
    dims=Tuple(Int.(data["shape"]))
    length(dims)==2 && all(d->1<=d<=256,dims) || throw(ArgumentError("fixture dimension outside 1:256"))
    raw=data["words"];length(raw)==4prod(dims) || throw(ArgumentError("limb count mismatch"))
    A=Matrix{MF}(undef,dims...)
    for i in eachindex(A);A[i]=fromwords(raw[4i-3:4i]);end
    return A
end
function digest(A,B,C0,alpha,beta)
    io=IOBuffer()
    for X in (A,B,C0)
        print(io,size(X),';')
        for x in X,word in words(x);print(io,word,';');end
    end
    for x in (alpha,beta),word in words(x);print(io,word,';');end
    bytes2hex(sha256(take!(io)))
end
function fullwidth(rng,shift=0)
    x=MF(ntuple(i->ldexp((rand(rng,Bool) ? 1.0 : -1.0)*(0.5+rand(rng)/2),shift-55(i-1)),4))
    MultiFloats.isnormalized(x) || error("generator produced unnormalized input")
    x
end
function build(m,k,n,family,beta,input_id;seed=UInt32(20260908))
    all(d->1<=d<=256,(m,k,n)) || throw(ArgumentError("bounded fixture dimensions required"))
    family in ("fullwidth","paircancel","zero") || throw(ArgumentError("unknown input family"))
    beta in (0.0,0.5) || throw(ArgumentError("beta must be 0 or 0.5"))
    rng=MersenneTwister(seed)
    A=[fullwidth(rng) for i in 1:m,t in 1:k]
    B=[fullwidth(rng) for t in 1:k,j in 1:n]
    C0=[fullwidth(rng,family=="paircancel" ? -12 : 0) for i in 1:m,j in 1:n]
    if family=="paircancel"
        for t in 1:div(k,2)
            for i in 1:m;A[i,2t]=-A[i,2t-1]+fullwidth(rng,-12);end
            for j in 1:n;B[2t,j]=B[2t-1,j];end
        end
        # Odd-k tail is genuinely unpaired and at the residual scale.
        isodd(k) && foreach(i->A[i,k]=fullwidth(rng,-12),1:m)
    elseif family=="zero"
        fill!(A,zero(MF));fill!(B,zero(MF));fill!(C0,zero(MF))
    end
    all(MultiFloats.isnormalized,A) && all(MultiFloats.isnormalized,B) && all(MultiFloats.isnormalized,C0) || error("input normalization")
    alpha=one(MF);bb=MF(beta)
    fixture=Dict("schema"=>1,"type"=>"Float64x4","precision_bits"=>209,
        "encoding"=>"IEEE754-binary64 hexadecimal words; column-major elements; limbs 1:4",
        "host_endian_bom"=>string(ENDIAN_BOM;base=16),"shape"=>[m,k,n],"input_id"=>input_id,
        "family"=>family,"seed"=>string(seed),"alpha"=>words(alpha),"beta"=>words(bb),
        "A"=>encode(A),"B"=>encode(B),"C0"=>encode(C0),"input_sha256"=>digest(A,B,C0,alpha,bb))
    fixture
end
function materialize(fixture)
    fixture["schema"]==1 && fixture["type"]=="Float64x4" && fixture["precision_bits"]==209 || error("fixture schema/type")
    A=decode(fixture["A"]);B=decode(fixture["B"]);C0=decode(fixture["C0"])
    alpha=fromwords(fixture["alpha"]);beta=fromwords(fixture["beta"])
    words(alpha)==words(one(MF)) || error("alpha scope")
    words(beta) in (words(zero(MF)),words(MF(0.5))) || error("beta scope")
    m,k=size(A);size(B,1)==k && size(C0)==(m,size(B,2)) || error("fixture shapes")
    fixture["shape"]==[m,k,size(B,2)] || error("shape metadata")
    all(X->all(MultiFloats.isnormalized,X),(A,B,C0)) || throw(ArgumentError("unnormalized fixture"))
    digest(A,B,C0,alpha,beta)==fixture["input_sha256"] || error("fixture word hash")
    return A,B,C0,alpha,beta
end
# One scalar exact-integer accumulation per output, prepared once per cell.
function reference(fixture)
    A,B,C0,alpha,beta=materialize(fixture);m,k=size(A);n=size(B,2)
    a=scaled.(A);b=scaled.(B);c=scaled.(C0)
    product=Matrix{BigInt}(undef,m,n);absolute=similar(product);result=similar(product);operand=similar(product)
    for j in 1:n,i in 1:m
        value=big(0);bound=big(0)
        for t in 1:k
            term=a[i,t]*b[t,j];value+=term;bound+=abs(term)
        end
        cb=iszero(beta) ? big(0) : c[i,j]<<(GRID-1)
        product[i,j]=value;absolute[i,j]=bound;result[i,j]=value+cb;operand[i,j]=bound+abs(cb)
    end
    Dict("schema"=>1,"algorithm"=>"scalar exact dyadic integer accumulation",
        "scale_exponent"=>PRODUCT_GRID,"shape"=>[m,n],"input_sha256"=>fixture["input_sha256"],
        "result"=>string.(vec(result);base=16),"product"=>string.(vec(product);base=16),
        "product_absolute"=>string.(vec(absolute);base=16),"operand"=>string.(vec(operand);base=16))
end
function unpack_reference(ref)
    ref["schema"]==1 && ref["scale_exponent"]==PRODUCT_GRID || error("reference schema")
    dims=Tuple(Int.(ref["shape"]));all(d->1<=d<=256,dims) || error("reference dimensions")
    Dict(k=>reshape(parse.(BigInt,ref[k];base=16),dims) for k in ("result","product","product_absolute","operand"))
end
ratio(n,d)=iszero(d) ? "undefined_zero_denominator" : string(n//d)
function distribution(nums,dens)
    ratios=Rational{BigInt}[a//abs(b) for (a,b) in zip(nums,dens) if !iszero(b)]
    isempty(ratios) && return Dict("nonzero_count"=>0,"zero_count"=>length(dens),"max"=>"undefined","median"=>"undefined")
    sort!(ratios);n=length(ratios)
    med=isodd(n) ? ratios[div(n+1,2)] : (ratios[div(n,2)]+ratios[div(n,2)+1])/2
    Dict("nonzero_count"=>n,"zero_count"=>length(dens)-n,"max"=>string(last(ratios)),"median"=>string(med))
end
function metrics(C,ref)
    finite=all(x->all(isfinite,x._limbs),C)
    finite || return Dict{String,Any}("pass"=>false,"finite"=>false,"reason"=>"nonfinite output")
    r=ref["result"];size(C)==size(r) || error("output shape")
    values=scaled.(C) .<< GRID;error=abs.(values-r)
    maxerr=maximum(error);normref=maximum(abs,r);normal=all(MultiFloats.isnormalized,C)
    zeroerrors=count(i->iszero(r[i])&&!iszero(error[i]),eachindex(r))
    mixed=maxerr//max(UNIT,normref)
    Dict{String,Any}("pass"=>(normal && mixed<=MIXED_TOL),"finite"=>true,"normalized"=>normal,
        "max_absolute_error"=>ratio(maxerr,UNIT),"mixed_max1_norm_error"=>string(mixed),
        "mixed_max1_tolerance"=>string(MIXED_TOL),"normwise_relative_error"=>ratio(maxerr,normref),
        "componentwise_relative"=>distribution(error,r),"zero_reference_nonzero_errors"=>zeroerrors,
        "operand_scaled_error"=>ratio(maxerr,maximum(ref["operand"])),
        "zero_reference_exact"=>(iszero(normref)&&iszero(maxerr)))
end
end
