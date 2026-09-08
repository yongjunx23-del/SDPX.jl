module FactorPreservingAffine
# Explicit unpromoted replay. No default solver route or existing success flag.
using SDPX, LinearAlgebra, SparseArrays, TOML
include("power_half_root_geometry.jl")
include("power_half_root_geometry_capture.jl")
include("half_power_compensated_factor.jl")
const RG=PowerHalfRootGeometry
const CAP=HalfRootGeometryCapture
const word=CAP.floatword
const PHYSICAL_FORCING=0x1p-17 # existing Float64 512*sqrt(eps) Newton ceiling
words(A::AbstractArray{Float64})=Tuple(reinterpret(UInt64,vec(A)))
words(x::Float64)=(reinterpret(UInt64,x),)
fingerprint(arrays...)=Tuple(Iterators.flatten(words(x) for x in arrays))
function lower_solve(L,b)
    n=length(b);x=Vector{Float64}(undef,n)
    for i in 1:n
        v=b[i]
        for j in 1:i-1;v-=L[i,j]*x[j];end
        x[i]=v/L[i,i]
    end
    all(isfinite,x) || error("nonfinite triangular action")
    x
end
function upper_solve(L,b)
    n=length(b);x=Vector{Float64}(undef,n)
    for i in n:-1:1
        v=b[i]
        for j in i+1:n;v-=L[j,i]*x[j];end
        x[i]=v/L[i,i]
    end
    all(isfinite,x) || error("nonfinite adjoint triangular action")
    x
end
function lower_multiply(L,v)
    [sum(L[i,j]*v[j] for j in 1:i) for i in eachindex(v)]
end
function upper_multiply(L,v)
    [sum(L[j,i]*v[j] for j in i:length(v)) for i in eachindex(v)]
end
struct BlockMetric
    offset::Int
    L::Matrix{Float64}
    R::Matrix{Float64}
    scale::Float64
    mu::Float64
    primal::Vector{Float64}
    dual::Vector{Float64}
    shadow::Vector{Float64}
    frozen::Tuple{Vararg{UInt64}}
end
function verify(block::BlockMetric)
    fingerprint(block.L,block.R,block.scale,block.mu,block.primal,block.dual,block.shadow)==block.frozen ||
        error("factor/point/scale epoch drift")
    true
end
function block_metric(offset,L,s,y,shadow,mu)
    size(L)==(3,3) && length(s)==length(y)==length(shadow)==3 || throw(DimensionMismatch())
    all(A->all(isfinite,A),(L,s,y,shadow)) && isfinite(mu) && mu>0 || error("nonfinite metric inputs")
    all(i->L[i,i]>0,1:3) && all(iszero,(L[1,2],L[1,3],L[2,3])) || error("invalid lower factor")
    sx,sy,sz=RG.point.(s)
    s[1]>0 && s[2]>0 && (sx*sy-sz*sz).lo>0 || error("current half-Power primal interior unresolved")
    pairing=sum((RG.point(s[i])*RG.point(y[i]) for i in 1:3);init=RG.point(0))
    pairing.lo>0 || error("exact pairing positivity unresolved")
    scale=sqrt(mu)
    a=scale.*lower_solve(L,y)
    b=upper_multiply(L,s)./scale
    aa=sum(abs2,a);p=s[1]*y[1]+s[2]*y[2]+s[3]*y[3]
    isfinite(aa) && aa>0 && isfinite(p) && p>0 || error("unresolved BFGS denominator")
    M=Matrix{Float64}(undef,3,3)
    for j in 1:3,i in j:3
        v=(i==j ? 1.0 : 0.0)-a[i]*a[j]/aa+b[i]*b[j]/p
        M[i,j]=v;M[j,i]=v
    end
    all(isfinite,M) || error("nonfinite whitened metric")
    R=Matrix(cholesky(Symmetric(M);check=true).L) # no shift, clipping or fallback
    lc,sc,yc,tc=copy(L),copy(s),copy(y),copy(shadow)
    block=BlockMetric(offset,lc,R,scale,mu,sc,yc,tc,fingerprint(lc,R,scale,mu,sc,yc,tc))
    block,(a=copy(a),b=copy(b),p=p,aa=aa,M=M)
end
function transform(block::BlockMetric,v,kind::Symbol)
    verify(block)
    kind===:S && return block.scale.*upper_solve(block.L,lower_multiply(block.R,v))
    kind===:St && return block.scale.*upper_multiply(block.R,lower_solve(block.L,v))
    kind===:W && return lower_solve(block.R,upper_multiply(block.L,v))./block.scale
    kind===:Wt && return lower_multiply(block.L,upper_solve(block.R,v))./block.scale
    throw(ArgumentError("unknown transform"))
end
struct FactorCone <: SDPX.AbstractConeLinearization{Float64}
    dimension::Int
    lp_scales::Vector{Float64}
    blocks::Vector{BlockMetric}
    frozen_lp::Tuple{Vararg{UInt64}}
end
SDPX.cone_dimension(cone::FactorCone)=cone.dimension
function SDPX.validate_cone_linearization(cone::FactorCone)
    words(cone.lp_scales)==cone.frozen_lp || error("LP factor drift")
    all(x->isfinite(x)&&x>0,cone.lp_scales) || error("LP factor domain")
    expected=length(cone.lp_scales)+1
    for b in cone.blocks
        b.offset==expected || error("factor cone coverage drift");verify(b);expected+=3
    end
    expected==cone.dimension+1 || error("factor cone dimension drift")
    true
end
function transform(cone::FactorCone,v,kind::Symbol)
    SDPX.validate_cone_linearization(cone);length(v)==cone.dimension || throw(DimensionMismatch())
    kind in (:S,:St,:W,:Wt) || throw(ArgumentError("unknown transform"))
    out=Vector{Float64}(undef,cone.dimension)
    for i in eachindex(cone.lp_scales)
        out[i]=kind in (:S,:St) ? cone.lp_scales[i]*v[i] : v[i]/cone.lp_scales[i]
    end
    for b in cone.blocks
        rows=b.offset:b.offset+2;out[rows]=transform(b,view(v,rows),kind)
    end
    all(isfinite,out) || error("nonfinite product transform")
    out
end
function SDPX.apply_cone_linearization!(out::AbstractVector{Float64},cone::FactorCone,v::AbstractVector{Float64})
    length(out)==cone.dimension || throw(DimensionMismatch())
    copyto!(out,transform(cone,transform(cone,v,:St),:S))
end
function inverse_action(cone,v)
    transform(cone,transform(cone,v,:W),:Wt)
end
struct AffineEpoch{F}
    source_record::Int
    factor_mode::Symbol
    A::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    c::Vector{Float64}
    x::Vector{Float64}
    s::Vector{Float64}
    y::Vector{Float64}
    tau::Float64
    kappa::Float64
    mu::Float64
    cone::FactorCone
    Ahat::Matrix{Float64}
    bhat::Vector{Float64}
    core::Matrix{Float64}
    factor::F
    frozen::Tuple{Vararg{UInt64}}
    frozen_structure::Tuple
    root_reports::Vector{Any}
    construction::Vector{Any}
end
function epoch_fingerprint(e)
    fingerprint(e.A.nzval,e.b,e.c,e.x,e.s,e.y,e.tau,e.kappa,e.mu,e.Ahat,e.bhat,e.core,e.factor.factors)
end
function verify(e::AffineEpoch)
    SDPX.validate_cone_linearization(e.cone)
    epoch_fingerprint(e)==e.frozen || error("affine numerical epoch drift")
    (size(e.A),Tuple(e.A.colptr),Tuple(e.A.rowval),Tuple(e.factor.ipiv),
        Tuple((b.offset,b.frozen) for b in e.cone.blocks),e.cone.frozen_lp,e.factor_mode)==e.frozen_structure || error("affine structure/factor drift")
    true
end
function build(row;factor_mode::Symbol=:stored_native)
    factor_mode in (:stored_native,:compensated_half_candidate) || error("unsupported factor experiment")
    RG.Phi._runtime_ok() || error("unsupported Float64 arithmetic context")
    row["schema"]==1 && row["provider"]=="explicit_research_dual_hessian_one_secant" || error("provider policy")
    m,n=row["A_shape"];1<=n<=16 && 1<=m<=32 || error("bounded affine shape")
    ptr=copy(row["A_colptr"]);rows=copy(row["A_rowval"]);values=word.(row["A_bits"])
    length(ptr)==n+1 && length(rows)==length(values)<=m*n || error("CSC lengths")
    ptr[1]==1 && ptr[end]==length(values)+1 || error("CSC endpoints")
    for j in 1:n
        1<=ptr[j]<=ptr[j+1]<=length(values)+1 || error("CSC pointers")
        previous=0
        for k in ptr[j]:ptr[j+1]-1
            previous<rows[k]<=m || error("CSC rows");previous=rows[k]
        end
    end
    A=SparseMatrixCSC{Float64,Int}(m,n,ptr,rows,values)
    b,c,x,s,y=(word.(row[k*"_bits"]) for k in ("b","c","x","s","y"))
    tau,kappa,mu=(word(row[k*"_bits"]) for k in ("tau","kappa","mu"))
    length(b)==length(s)==length(y)==m && length(c)==length(x)==n || throw(DimensionMismatch())
    all(v->all(isfinite,v),(A.nzval,b,c,x,s,y)) || error("nonfinite epoch input")
    all(x->isfinite(x)&&x>0,(tau,kappa,mu)) || error("epoch scalar domain")
    lp=row["lp_rows"];lp==collect(1:length(lp)) || error("LP coverage")
    scales=Float64[]
    for i in lp
        s[i]>0 && y[i]>0 || error("LP interior")
        push!(scales,sqrt(s[i]/y[i]))
    end
    blocks=BlockMetric[];reports=Any[];construction=Any[]
    for p in row["power"]
        offset=p["offset"];rows=offset:offset+2
        localrow=Dict("id"=>"trial-$(row["source_record"])-offset-$offset",
            "settings"=>p["settings"],"alpha_bits"=>p["alpha_bits"],
            "accepted_gap_bits"=>p["accepted_gap_bits"],"accepted_valid"=>p["accepted_valid"],
            "dual_bits"=>[string(reinterpret(UInt64,v);base=16,pad=16) for v in y[rows]])
        replay=CAP.replay(localrow);push!(reports,replay)
        replay.root.status===:qualified && get(replay.native,"factor_certificate",false) || error("root/factor replay unsupported")
        tag=SDPX.PowerConjugateTag{Float64}(word(p["alpha_bits"]))
        SDPX._ns_conjugate_primal_interior(tag,s[rows]...) || error("native current-primal domain gate failed")
        L=replay.snapshots["L"]
        factor_info=nothing
        if factor_mode===:compensated_half_candidate
            factor_info=HalfPowerCompensatedFactor.factor(replay.snapshots["shadow"])
            factor_info.status===:formed || error("compensated factor domain unsupported")
            L=factor_info.L
        end
        legacy_ok,legacy_error=SDPX._ns_structural_hessian_factor_certificate!(L,tag,replay.snapshots["shadow"]...)
        metric,info=block_metric(offset,L,s[rows],y[rows],replay.snapshots["shadow"],mu)
        push!(blocks,metric);push!(construction,(;info...,factor_mode,factor_info,legacy_ok,legacy_error))
    end
    cone=FactorCone(m,scales,blocks,words(scales));SDPX.validate_cone_linearization(cone)
    Ahat=hcat([transform(cone,Vector(A[:,j]),:W) for j in 1:n]...)
    bhat=transform(cone,b,:W)
    # Full augmented core and homogeneous border, not normal equations.
    K=zeros(n+m+2,n+m+2)
    K[1:n,n+1:n+m]=transpose(Ahat);K[1:n,n+m+1]=c
    K[n+1:n+m,1:n]=Ahat
    for i in 1:m;K[n+i,n+i]=-1;end
    K[n+1:n+m,n+m+1]=-bhat
    K[n+m+1,1:n]=c;K[n+m+1,n+1:n+m]=bhat;K[n+m+1,n+m+2]=1
    K[n+m+2,n+m+1]=kappa;K[n+m+2,n+m+2]=tau
    all(isfinite,K) || error("nonfinite affine core")
    factor=lu(K;check=true)
    frozen=fingerprint(A.nzval,b,c,x,s,y,tau,kappa,mu,Ahat,bhat,K,factor.factors)
    structural=(size(A),Tuple(A.colptr),Tuple(A.rowval),Tuple(factor.ipiv),
        Tuple((b.offset,b.frozen) for b in cone.blocks),cone.frozen_lp,factor_mode)
    AffineEpoch(row["source_record"],factor_mode,A,b,c,x,s,y,tau,kappa,mu,cone,Ahat,bhat,K,factor,frozen,structural,reports,construction)
end
function affine_rhs(e::AffineEpoch)
    verify(e)
    SDPX.residual_newton_rhs(e.A*e.x+e.s-e.b*e.tau,
        transpose(e.A)*e.y+e.c*e.tau,dot(e.c,e.x)+dot(e.b,e.y)+e.kappa,-e.s,-e.tau*e.kappa)
end
function solve(e::AffineEpoch,rhs::SDPX.HSDNewtonRHS{Float64}=affine_rhs(e))
    verify(e);m,n=size(e.A)
    rhs.cone_corrector == -e.s || error("first experiment is affine-only; combined corrector unsupported")
    length(rhs.primal_affine)==m && length(rhs.dual_affine)==n || throw(DimensionMismatch())
    all(v->all(isfinite,v),(rhs.primal_affine,rhs.dual_affine,rhs.cone_corrector)) &&
        isfinite(rhs.homogeneous_gap) && isfinite(rhs.tau_kappa) || error("nonfinite RHS")
    rhs=SDPX.HSDNewtonRHS(copy(rhs.primal_affine),copy(rhs.dual_affine),rhs.homogeneous_gap,
        copy(rhs.cone_corrector),rhs.tau_kappa)
    rp=transform(e.cone,rhs.primal_affine,:W);h=transform(e.cone,rhs.cone_corrector,:W)
    right=vcat(rhs.dual_affine,rp-h,rhs.homogeneous_gap,rhs.tau_kappa)
    solution=e.factor\right
    all(isfinite,solution) || error("nonfinite affine solution")
    dx=copy(solution[1:n]);dyhat=copy(solution[n+1:n+m]);dshat=h-dyhat
    dy=transform(e.cone,dyhat,:Wt);ds=transform(e.cone,dshat,:S)
    direction=SDPX.NewtonDirection(dx,dy,ds,solution[end-1],solution[end])
    system=SDPX.NewtonSystem(e.A,e.b,e.c,e.cone,e.tau,e.kappa,rhs)
    residual=SDPX.NewtonResidual(system);SDPX.newton_residual!(residual,system,direction)
    verify(e)
    (;direction,residual,rhs,transformed_solution=copy(solution),transformed_rhs=right,
        transformed_residual=e.core*solution-right,production_admitted=false)
end
end
