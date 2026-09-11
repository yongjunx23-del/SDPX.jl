module NativeHalfPair
using SDPX,LinearAlgebra,SparseArrays
import ..FactorPreservingAffine
import ..NativeFactorAffineCertificate
const FA=FactorPreservingAffine
const NC=NativeFactorAffineCertificate
const PowerHalfRootGeometry=FA.RG
const RG=PowerHalfRootGeometry
include("half_power_polynomial_root.jl")
const HR=HalfPowerPolynomialRoot
const EF=FA.HalfPowerFactorCertificate
const HF=FA.HalfPowerCompensatedFactor
const POLICY=:experimental_dual_hessian_one_secant
struct RootSettings
    tolerance::Float64
    max_iterations::Int
    max_bisections::Int
    function RootSettings(t,i,b)
        t isa Float64 && i isa Int && b isa Int || throw(ArgumentError("exact native settings types required"))
        new(t,i,b)
    end
end
RootSettings()=RootSettings(256eps(Float64),64,512)
settings_key(s::RootSettings)=(reinterpret(UInt64,s.tolerance),s.max_iterations,s.max_bisections)
valid(s::RootSettings)=s.tolerance==256eps(Float64) && 1<=s.max_iterations<=64 && 0<=s.max_bisections<=512
struct Layout
    orthant::Int
    alphas::Tuple
end
layout_key(l::Layout)=(l.orthant,l.alphas)
valid(l::Layout,m)=0<=l.orthant<=m && l.orthant+3length(l.alphas)==m &&
    all(a->a isa Float64 && reinterpret(UInt64,a)==0x3fe0000000000000,l.alphas)
mutable struct Owner
    anchor::Any
    tokens::Tuple
    generation::Int
end
Owner()=Owner(nothing,(),0)
struct PairRefusal
    status::Symbol
    stage::Symbol
    reason::Symbol
    reports::Vector{Any}
    production_admitted::Bool
end
refuse(stage,reason,reports)=PairRefusal(:unsupported,stage,reason,deepcopy(reports),false)
struct PairReceipt
    status::Symbol
    s::Vector{Float64}
    y::Vector{Float64}
    mu::Float64
    layout::Layout
    settings::RootSettings
    policy::Symbol
    owner::Owner
    generation::Int
    cone::FA.FactorCone
    reports::Vector{Any}
    frozen::Tuple
    production_admitted::Bool
end
struct WarmToken
    owner::Owner
    anchor::PairReceipt
    generation::Int
    offset::Int
    candidate::Float64
    settings::Tuple
    layout::Tuple
    policy::Symbol
end
key(x::Float64)=reinterpret(UInt64,x)
key(x::Union{Int,Bool,Symbol,Nothing})=x
key(x::RG.I)=(key(x.lo),key(x.hi))
key(x::Tuple)=map(key,x)
key(x::NamedTuple)=(keys(x),map(key,values(x)))
key(x::AbstractArray)=(size(x),Tuple(key(v) for v in x))
function pair_key(p::PairReceipt)
    (key(p.s),key(p.y),key(p.mu),layout_key(p.layout),settings_key(p.settings),p.policy,objectid(p.owner),p.generation,
        key(p.cone.lp_scales),Tuple((b.offset,key(b.L),key(b.R),key(b.scale),key(b.mu),key(b.primal),key(b.dual),key(b.shadow)) for b in p.cone.blocks),
        key(p.reports),VERSION,Sys.ARCH,Sys.KERNEL)
end
function verify(p::PairReceipt)
    RG.Phi._runtime_ok() || error("unsupported pair arithmetic context")
    p.status===:certified && !p.production_admitted && p.policy===POLICY && valid(p.settings) || error("pair policy")
    length(p.s)==length(p.y)==p.cone.dimension && valid(p.layout,length(p.s)) || error("pair dimensions")
    p.layout.orthant==length(p.cone.lp_scales) && length(p.layout.alphas)==length(p.cone.blocks)==length(p.reports) ||
        error("pair coverage")
    isfinite(p.mu)&&p.mu>0 && all(isfinite,p.s)&&all(isfinite,p.y) || error("pair finite domain")
    all(i->p.s[i]>0 && p.y[i]>0,1:p.layout.orthant) || error("orthant pair interior")
    pair_key(p)==p.frozen || error("pair receipt drift")
    SDPX.validate_cone_linearization(p.cone)
    for (b,r) in zip(p.cone.blocks,p.reports)
        rows=b.offset:b.offset+2
        size(b.L)==size(b.R)==(3,3) && length(b.primal)==length(b.dual)==length(b.shadow)==3 || error("pair factor dimensions")
        b.offset==r.offset && b.primal==p.s[rows] && b.dual==p.y[rows] && b.mu==p.mu || error("pair metric point drift")
        x,y,z=RG.point.(b.primal)
        b.primal[1]>0 && b.primal[2]>0 && (x*y-z*z).lo>0 || error("primal pair interior")
        SDPX._ns_conjugate_primal_interior(SDPX.PowerConjugateTag{Float64}(0.5),b.primal...) || error("native primal interior")
    end
    true
end
function root_call(y,s,probe)
    HR.root(y...;warm=probe,tolerance=s.tolerance,max_iterations=s.max_iterations,max_bisections=s.max_bisections)
end
function reconstruct(y,c)
    # Fixed ordinary Float64 order. Neither this d nor c is the determinant or
    # gap authority for the rounded stored shadow.
    numerator=1.0+0.5*c
    uc=y[1]*c;vc=y[2]*c
    x=numerator/uc;v=numerator/vc
    d=(c*x)*v
    z=(-y[3]*d)/2.0
    (;shadow=Float64[x,v,z],numerator,uc,vc,d_candidate=d)
end
function certify(p::PairReceipt)
    verify(p);records=Any[]
    for (b,r) in zip(p.cone.blocks,p.reports)
        fresh=root_call(b.dual,p.settings,r.probe)
        fresh.status===:qualified && key(fresh)==key(r.root) || return refuse(:root,:replay,records)
        reconstruction=reconstruct(b.dual,r.root.candidate)
        key(reconstruction)==key(r.reconstruction) && key(reconstruction.shadow)==key(b.shadow) ||
            return refuse(:shadow,:stored_words,records)
        poly=NC.polynomials(b);metric=NC.bfgs_bound(b,poly)
        push!(records,(;offset=b.offset,metric,known_polynomial_products=poly.budget.products,known_polynomial_sums=poly.budget.sums))
        metric.status===:certified || return refuse(:metric,:true_bfgs,records)
    end
    for i in eachindex(p.cone.lp_scales)
        ell=RG.point(p.cone.lp_scales[i])
        error=(RG.point(p.s[i])/RG.point(p.y[i]))/(ell*ell)-RG.point(1)
        RG.absupper(error)<=RG.KAPPA || return refuse(:orthant,:metric,records)
    end
    (;status=:certified,records,work_scope=:stage_receipts_only,production_admitted=false)
end
function anchor!(owner::Owner,pair::PairReceipt)
    owner.anchor===nothing && isempty(owner.tokens) && owner.generation==0 || error("anchor already initialized")
    pair.owner===owner && pair.generation==0 || error("not this owner's initial pair")
    certify(pair).status===:certified || error("uncertified initial anchor")
    # Explicit initial-state binding, NOT accepted line-search progress. There
    # is intentionally no trial-commit/acceptance operation in this module.
    tokens=Tuple(WarmToken(owner,pair,0,r.offset,r.root.candidate,settings_key(pair.settings),layout_key(pair.layout),pair.policy) for r in pair.reports)
    owner.anchor=pair;owner.tokens=tokens
    tokens
end
function warm_valid(owner,warm,layout,settings,policy)
    owner.anchor isa PairReceipt && warm isa Tuple && length(warm)==length(layout.alphas) || return false
    anchor=owner.anchor
    try verify(anchor) catch err
        err isa ErrorException || rethrow();return false
    end
    for (i,t) in enumerate(warm)
        t isa WarmToken && i<=length(owner.tokens) && t===owner.tokens[i] && t.owner===owner && t.anchor===anchor &&
            t.generation==owner.generation==anchor.generation && t.offset==layout.orthant+3(i-1)+1 &&
            t.settings==settings_key(settings)==settings_key(anchor.settings) &&
            t.layout==layout_key(layout)==layout_key(anchor.layout) && t.policy===policy===anchor.policy || return false
    end
    length(owner.tokens)==length(warm)
end
function build(s,y,mu,layout;policy,settings,owner=Owner(),warm=nothing)
    reports=Any[]
    s isa Vector{Float64} && y isa Vector{Float64} && mu isa Float64 && layout isa Layout &&
        settings isa RootSettings && owner isa Owner || return refuse(:input,:type,reports)
    policy===POLICY && valid(settings) || return refuse(:input,:policy_settings,reports)
    m=length(s);1<=m<=32 && length(y)==m && valid(layout,m) || return refuse(:input,:layout,reports)
    RG.Phi._runtime_ok() || return refuse(:input,:runtime,reports)
    all(isfinite,s) && all(isfinite,y) && isfinite(mu) && mu>0 || return refuse(:input,:finite_mu,reports)
    warm===nothing || isempty(layout.alphas) && return refuse(:warm,:no_power_blocks,reports)
    warm===nothing || warm_valid(owner,warm,layout,settings,policy) || return refuse(:warm,:lineage,reports)
    sc=copy(s);yc=copy(y);scales=Float64[];blocks=FA.BlockMetric[]
    for i in 1:layout.orthant
        sc[i]>0 && yc[i]>0 || return refuse(:orthant,:interior,reports)
        scale=sqrt(sc[i]/yc[i]);isfinite(scale) && scale>0 || return refuse(:orthant,:factor_range,reports)
        push!(scales,scale)
    end
    stage=:root
    try
        for index in eachindex(layout.alphas)
            offset=layout.orthant+3(index-1)+1;rows=offset:offset+2;dual=yc[rows];primal=sc[rows]
            tag=SDPX.PowerConjugateTag{Float64}(0.5)
            SDPX._ns_conjugate_primal_interior(tag,primal...) || return refuse(:primal,:interior,reports)
            probe=warm===nothing || warm[index].candidate==1.0 || iszero(dual[3]) ? nothing : warm[index].candidate
            mode=warm===nothing ? :cold : probe===nothing ? :cold_endpoint_receipt : :accepted_anchor_probe
            stage=:root;root=root_call(dual,settings,probe)
            if root.status!==:qualified
                push!(reports,(;offset,mode,probe,root));return refuse(stage,root.reason,reports)
            end
            stage=:shadow;reconstruction=reconstruct(dual,root.candidate)
            if !all(isfinite,reconstruction.shadow)
                push!(reports,(;offset,mode,probe,root,reconstruction));return refuse(stage,:nonfinite,reports)
            end
            stage=:factor;formed=HF.factor(reconstruction.shadow)
            if formed.status!==:formed
                push!(reports,(;offset,mode,probe,root,reconstruction,formed));return refuse(stage,formed.reason,reports)
            end
            point=EF.verify(reconstruction.shadow,formed.L,dual)
            if point.status!==:certified
                push!(reports,(;offset,mode,probe,root,reconstruction,formed,point));return refuse(:stored_geometry,point.reason,reports)
            end
            stage=:metric;block,construction=FA.block_metric(offset,formed.L,primal,dual,reconstruction.shadow,mu)
            poly=NC.polynomials(block);metric=NC.bfgs_bound(block,poly)
            push!(reports,(;offset,mode,probe,root,reconstruction,formed,point,construction,metric,
                known_polynomial_products=poly.budget.products,known_polynomial_sums=poly.budget.sums,legacy_evaluated=false))
            metric.status===:certified || return refuse(stage,metric.reason,reports)
            push!(blocks,block)
        end
        cone=FA.FactorCone(m,scales,blocks,FA.words(scales));SDPX.validate_cone_linearization(cone)
        generation=owner.anchor===nothing ? 0 : owner.generation+1
        provisional=PairReceipt(:certified,sc,yc,mu,layout,settings,policy,owner,generation,cone,reports,(),false)
        pair=PairReceipt(:certified,sc,yc,mu,layout,settings,policy,owner,generation,cone,reports,pair_key(provisional),false)
        stage=:pair_certificate;certificate=certify(pair)
        certificate.status===:certified ? pair : certificate
    catch err
        if err isa RG.EnclosureFailure || err isa RG.Phi.ArithmeticDomainError || err isa DomainError ||
            err isa PosDefException || err isa ErrorException
            push!(reports,(;exception=string(typeof(err)),message=sprint(showerror,err)))
            return refuse(stage,:native_refusal,reports)
        end
        rethrow()
    end
end
function epoch(pair::PairReceipt,A,b,c,x,tau,kappa;source_record::Int=0)
    verify(pair);reports=Any[]
    A isa SparseMatrixCSC{Float64,Int} && all(v->v isa Vector{Float64},(b,c,x)) &&
        tau isa Float64 && kappa isa Float64 || return refuse(:epoch_input,:type,reports)
    m,n=size(A);1<=n<=16 && m==length(pair.s) && length(b)==m && length(c)==length(x)==n ||
        return refuse(:epoch_input,:shape,reports)
    all(v->all(isfinite,v),(A.nzval,b,c,x)) && isfinite(tau)&&isfinite(kappa)&&tau>0&&kappa>0 ||
        return refuse(:epoch_input,:finite,reports)
    length(A.colptr)==n+1 && length(A.rowval)==length(A.nzval)<=m*n && A.colptr[1]==1 && A.colptr[end]==length(A.nzval)+1 ||
        return refuse(:epoch_input,:csc,reports)
    for j in 1:n
        1<=A.colptr[j]<=A.colptr[j+1]<=length(A.nzval)+1 || return refuse(:epoch_input,:csc,reports)
        previous=0
        for k in A.colptr[j]:A.colptr[j+1]-1
            previous<A.rowval[k]<=m || return refuse(:epoch_input,:csc,reports);previous=A.rowval[k]
        end
    end
    certificate=certify(pair);certificate.status===:certified || return certificate
    try
        e=FA._assemble_epoch(copy(A),copy(b),copy(c),copy(x),copy(pair.s),copy(pair.y),tau,kappa,pair.mu,
            deepcopy(pair.cone),source_record,:native_half_pair,deepcopy(pair.reports),deepcopy(pair.reports))
        (;status=:formed_epoch,epoch=e,production_admitted=false)
    catch err
        if err isa SingularException || err isa DomainError || err isa PosDefException ||
            err isa FA.FactorSeamNumericalFailure
            push!(reports,(;exception=string(typeof(err)),message=sprint(showerror,err)))
            return refuse(:epoch_factor,:singular_or_domain,reports)
        end
        rethrow()
    end
end
function trial(anchor::PairReceipt,tau,kappa,ds,dy,dtau,dkappa,alpha;warm=nothing)
    verify(anchor);m=length(anchor.s);reports=Any[]
    anchor.owner.anchor===anchor && anchor.owner.generation==anchor.generation || return refuse(:trial_input,:anchor_lineage,reports)
    ds isa Vector{Float64} && dy isa Vector{Float64} && length(ds)==length(dy)==m &&
        all(x->x isa Float64 && isfinite(x),(tau,kappa,dtau,dkappa,alpha)) &&
        all(isfinite,ds) && all(isfinite,dy) && 0<=alpha<=1 || return refuse(:trial_input,:domain,reports)
    st=[anchor.s[i]+alpha*ds[i] for i in 1:m];yt=[anchor.y[i]+alpha*dy[i] for i in 1:m]
    tt=tau+alpha*dtau;kt=kappa+alpha*dkappa
    if !(isfinite(tt)&&isfinite(kt)&&tt>0&&kt>0)
        push!(reports,(;st,yt,tau=tt,kappa=kt,alpha))
        return refuse(:trial_scalar,:positive,reports)
    end
    mu=(dot(st,yt)+tt*kt)/(m+1) # nu=orthant+3*Power=m, same native arithmetic
    pair=build(st,yt,mu,anchor.layout;policy=anchor.policy,settings=anchor.settings,owner=anchor.owner,warm)
    (;status=pair.status,pair,st,yt,tau=tt,kappa=kt,mu,alpha,construction_only=true,production_admitted=false)
end
end
