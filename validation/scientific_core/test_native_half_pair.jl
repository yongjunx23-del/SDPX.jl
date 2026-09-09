using Test,TOML,LinearAlgebra,SparseArrays,SDPX
include("factor_preserving_affine.jl")
include("factor_affine_reference.jl")
include("native_factor_affine_certificate.jl")
include("half_power_native_corrector.jl")
include("factor_combined_epoch.jl")
include("native_half_pair.jl")
const FA=FactorPreservingAffine
const FAR=FactorAffineReference
const NC=NativeFactorAffineCertificate
const NP=NativeHalfPair
const Q=Rational{BigInt}
const NATIVE_PAIR_RESULTS=Any[]
const SETTINGS=NP.RootSettings()
function psd3(M)
    all(i->M[i,i]>=0,1:3) && all(p->M[p[1],p[1]]*M[p[2],p[2]]-M[p[1],p[2]]^2>=0,((1,2),(1,3),(2,3))) && det(M)>=0
end
function check_pair(p)
    @test p isa NP.PairReceipt
    p isa NP.PairReceipt || return
    @test NP.verify(p)
    @test NP.certify(p).status===:certified
    @test !p.production_admitted
    @test all(b->b.mu==p.mu,p.cone.blocks)
    _,_,metrics=FAR.transforms(p.cone)
    for (r,b,metric) in zip(p.reports,p.cone.blocks,metrics)
        x,y,z=Q.(b.shadow);d=x*y-z*z;H=FAR.true_hessian(b.shadow);Li=FAR.inverse_lower(Q.(b.L));I3=Matrix{Q}(I,3,3)
        @test d>0
        @test d/(x*y)>=Q(0x1p-40)
        @test Q(r.point.eta)^2>=sum(abs2,Li*H*Li'-I3)
        gradient=Q[-y/d-1/(2x),-x/d-1/(2y),2z/d]
        error=-gradient-Q.(b.dual);sol=FAR.exact_solve(H,reshape(error,3,1))[:,1]
        @test Q(r.point.decrement)^2>=dot(error,sol)
        E=metric["true_hessian_formula_error"];eta=Q(r.metric.true_bound)
        @test psd3(eta*I3-E) && psd3(eta*I3+E)
        @test r.root.radius<=r.root.tolerance*r.root.lower
        @test r.reconstruction.shadow==b.shadow
        @test r.formed.L==b.L
        @test !r.legacy_evaluated
    end
end
function receipt_fields(p)
    p isa NP.PairReceipt ? (;status=p.status,s=p.s,y=p.y,mu=p.mu,generation=p.generation,
        reports=p.reports,lp_scales=p.cone.lp_scales,blocks=[(;offset=b.offset,L=b.L,R=b.R,scale=b.scale,mu=b.mu,
            primal=b.primal,dual=b.dual,shadow=b.shadow) for b in p.cone.blocks],production_admitted=p.production_admitted) :
        (;status=p.status,stage=p.stage,reason=p.reason,reports=p.reports,production_admitted=p.production_admitted)
end
@testset "native half-Power pair and trial construction" begin
    @test !isdefined(SDPX,:NativeHalfPair)
    for sign in (0.,-1.,1.)
        s=[2.,3.,0.25];y=[2.,1.,sign*0.5];mu=(dot(s,y)+1.)/4
        owner=NP.Owner();pair=NP.build(s,y,mu,NP.Layout(0,(0.5,));policy=NP.POLICY,settings=SETTINGS,owner)
        pair isa NP.PairReceipt || println("ORDINARY_REFUSAL ",receipt_fields(pair))
        check_pair(pair);pair isa NP.PairReceipt || continue
        tokens=NP.anchor!(owner,pair)
        @test owner.anchor===pair
        @test_throws ErrorException NP.anchor!(owner,pair)
        before=NP.pair_key(pair)
        warm=NP.build(copy(s),copy(y),mu,pair.layout;policy=NP.POLICY,settings=SETTINGS,owner,warm=tokens)
        check_pair(warm)
        @test warm.reports[1].mode== (sign==0 ? :cold_endpoint_receipt : :accepted_anchor_probe)
        bad=copy(y);bad[1]=0.
        rejected=NP.build(s,bad,mu,pair.layout;policy=NP.POLICY,settings=SETTINGS,owner,warm=tokens)
        @test rejected isa NP.PairRefusal
        @test owner.anchor===pair && NP.pair_key(pair)==before
        @test NP.build(s,y,mu,pair.layout;policy=NP.POLICY,settings=SETTINGS,owner=NP.Owner(),warm=tokens).stage===:warm
        @test NP.build(s,y,mu,pair.layout;policy=NP.POLICY,settings=NP.RootSettings(256eps(Float64),63,512),owner,warm=tokens).stage===:warm
        for alpha in (0.,0.25,0.5)
            ds=-0.01.*s;dy=0.02.*y
            trial=NP.trial(pair,1.,1.,ds,dy,-0.01,-0.005,alpha;warm=tokens)
            @test trial.status===:certified
            @test trial.construction_only && !trial.production_admitted
            @test trial.mu==(dot(trial.st,trial.yt)+trial.tau*trial.kappa)/4
            @test all(b->b.mu==trial.mu,trial.pair.cone.blocks)
            @test owner.anchor===pair && NP.pair_key(pair)==before
            push!(NATIVE_PAIR_RESULTS,(;kind=:ordinary_trial,sign,alpha,tau=trial.tau,kappa=trial.kappa,mu=trial.mu,pair=receipt_fields(trial.pair)))
        end
        original=copy(pair.s);s[1]+=1.
        @test pair.s==original
        push!(NATIVE_PAIR_RESULTS,(;kind=:ordinary,sign,pair=receipt_fields(pair),warm=receipt_fields(warm)))
    end
    s=[1.,2.,2.,3.,0.25];y=[2.,1.5,2.,1.,-0.5];mu=(dot(s,y)+1.)/6
    mixed=NP.build(s,y,mu,NP.Layout(2,(0.5,));policy=NP.POLICY,settings=SETTINGS)
    check_pair(mixed)
    push!(NATIVE_PAIR_RESULTS,(;kind=:mixed,pair=receipt_fields(mixed)))
    for id in (17,19)
        row=TOML.parsefile(joinpath(@__DIR__,"fixtures/factor_affine_trial_$id.toml"))
        s,y=(FA.word.(row[k*"_bits"]) for k in ("s","y"));mu=FA.word(row["mu_bits"])
        layout=NP.Layout(length(row["lp_rows"]),Tuple(FA.word(p["alpha_bits"]) for p in row["power"]))
        owner=NP.Owner();pair=NP.build(s,y,mu,layout;policy=NP.POLICY,settings=SETTINGS,owner)
        println("CAPTURE_NATIVE_PAIR ",id," ",pair.status,pair isa NP.PairRefusal ? " $(pair.stage)/$(pair.reason)" : "")
        @test pair isa Union{NP.PairReceipt,NP.PairRefusal}
        if pair isa NP.PairReceipt
            check_pair(pair);tokens=NP.anchor!(owner,pair)
            warm=NP.build(s,y,mu,layout;policy=NP.POLICY,settings=SETTINGS,owner,warm=tokens)
            @test warm isa Union{NP.PairReceipt,NP.PairRefusal}
            m,n=row["A_shape"];A=SparseMatrixCSC(m,n,copy(row["A_colptr"]),copy(row["A_rowval"]),FA.word.(row["A_bits"]))
            b,c,x=(FA.word.(row[k*"_bits"]) for k in ("b","c","x"));tau=FA.word(row["tau_bits"]);kappa=FA.word(row["kappa_bits"])
            built=NP.epoch(pair,A,b,c,x,tau,kappa;source_record=id)
            @test built.status===:formed_epoch
            e=built.epoch;affine=FA.solve(e);certificate=NC.certify(e,affine)
            @test certificate.status===:certified
            @test all(v->v<=Q(FA.PHYSICAL_FORCING),FAR.physical(e,affine).errors)
            combined=Any[]
            for sigma in (0.,0.25,0.75)
                co=FactorCombinedEpoch.build(e,affine;sigma_mu=sigma*e.mu);result=FactorCombinedEpoch.solve(co)
                cc=FactorCombinedEpoch.certify(co,result)
                @test cc.status===:certified
                @test all(v->v<=Q(FA.PHYSICAL_FORCING),FAR.physical(e,result).errors)
                push!(combined,(;sigma,certificate=cc,direction=(;dx=result.direction.dx,dy=result.direction.dy,ds=result.direction.ds,dtau=result.direction.dtau,dkappa=result.direction.dkappa)))
            end
            oldA=copy(e.A.nzval);A.nzval[1]+=1.;b[1]+=1.;x[1]+=1.
            @test FA.verify(e) && e.A.nzval==oldA
            old=pair.s[1];pair.s[1]+=1.
            @test_throws ErrorException NP.verify(pair)
            @test FA.verify(e)
            @test NP.build(s,y,mu,layout;policy=NP.POLICY,settings=SETTINGS,owner,warm=tokens).stage===:warm
            pair.s[1]=old
            push!(NATIVE_PAIR_RESULTS,(;kind=:capture,id,pair=receipt_fields(pair),warm=receipt_fields(warm),affine_certificate=certificate,combined))
        else
            @test pair.stage in (:root,:factor,:stored_geometry,:metric,:primal)
            @test all(r->!hasproperty(r,:exception),pair.reports)
            push!(NATIVE_PAIR_RESULTS,(;kind=:capture,id,pair=receipt_fields(pair)))
        end
    end
    for (s,y,mu,layout,settings) in (([1.,1.,0.],[1.,1.,2.],1.,NP.Layout(0,(0.5,)),SETTINGS),
        ([1.,1.,2.],[1.,1.,0.],1.,NP.Layout(0,(0.5,)),SETTINGS),
        ([1.,1.,0.],[1.,1.,prevfloat(2.)],1.,NP.Layout(0,(0.5,)),SETTINGS),
        ([1.,1.,0.],[1.,1.,0.],NaN,NP.Layout(0,(0.5,)),SETTINGS),
        ([1.,1.,0.],[1.,1.,0.],1.,NP.Layout(0,(0.4,)),SETTINGS),
        ([1.,1.,0.],[1.,1.,0.],1.,NP.Layout(1,(0.5,)),SETTINGS),
        ([1.,1.,0.],[1.,1.,0.5],1.,NP.Layout(0,(0.5,)),NP.RootSettings(256eps(Float64),1,512)),
        ([1.,1.,0.],[1.,1.,0.5],1.,NP.Layout(0,(0.5,)),NP.RootSettings(512eps(Float64),64,512)))
        @test NP.build(s,y,mu,layout;policy=NP.POLICY,settings) isa NP.PairRefusal
    end
    @test NP.build(Float32[1,1,0],[1.,1.,0.],1.,NP.Layout(0,(0.5,));policy=NP.POLICY,settings=SETTINGS).stage===:input
end
