module FactorCombinedEpoch
# Frozen combined-direction experiment. No production routing or readiness flag.
using SDPX,LinearAlgebra
import ..FactorPreservingAffine
import ..HalfPowerNativeCorrector
import ..NativeFactorAffineCertificate
const FA=FactorPreservingAffine
const HC=HalfPowerNativeCorrector
const NC=NativeFactorAffineCertificate
const RG=FA.RG
const EF=FA.HalfPowerFactorCertificate
rhs_words(r)=FA.fingerprint(r.primal_affine,r.dual_affine,r.homogeneous_gap,r.cone_corrector,r.tau_kappa)
copy_rhs(r)=SDPX.HSDNewtonRHS(copy(r.primal_affine),copy(r.dual_affine),r.homogeneous_gap,copy(r.cone_corrector),r.tau_kappa)
copy_direction(d)=SDPX.NewtonDirection(copy(d.dx),copy(d.dy),copy(d.ds),d.dtau,d.dkappa)
function correction_words(c)
    d=c.data
    arrays=(c.point,c.ds,c.dy,d.L,d.u,d.raw,d.swap,d.averaged,d.chi,d.ytilde,
        d.selected.original,(attempt.L for attempt in d.selected.history)...)
    (FA.fingerprint(c.alpha,arrays...),map(size,arrays),Tuple(attempt.shifts for attempt in d.selected.history))
end
struct CombinedEpoch{E,D,R}
    epoch::E
    affine_direction::D
    sigma_mu::Float64
    corrections::Vector{Any}
    rho::Vector{Float64}
    hhat::Vector{Float64}
    z::Vector{Float64}
    rhs::R
    frozen::Tuple
end
function fingerprint(c)
    d=c.affine_direction
    (FA.fingerprint(d.dx,d.dy,d.ds,d.dtau,d.dkappa,c.sigma_mu,c.rho,c.hhat,c.z),
        rhs_words(c.rhs),Tuple((v.offset,correction_words(v)) for v in c.corrections))
end
function verify(c::CombinedEpoch)
    FA.verify(c.epoch)
    fingerprint(c)==c.frozen || error("combined numerical/structural epoch drift")
    m,n=size(c.epoch.A);d=c.affine_direction
    map(length,(c.rho,c.hhat,c.z,d.dx,d.dy,d.ds,c.rhs.primal_affine,c.rhs.dual_affine,c.rhs.cone_corrector))==
        (m,m,m,n,m,m,m,n,m) || error("combined dimensions")
    all(i->iszero(c.rho[i]),eachindex(c.epoch.cone.lp_scales)) || error("unused orthant rho policy")
    length(c.corrections)==length(c.epoch.cone.blocks) || error("combined block coverage")
    for (v,b) in zip(c.corrections,c.epoch.cone.blocks)
        v.offset==b.offset && v.alpha==0.5 || error("combined block policy")
    end
    true
end
function build(epoch,affine;sigma_mu::Float64)
    FA.verify(epoch);canonical=FA.affine_rhs(epoch)
    rhs_words(affine.rhs)==rhs_words(canonical) || error("noncanonical affine RHS")
    prerequisite=NC.certify(epoch,affine)
    prerequisite.status===:certified || error("uncertified affine prerequisite")
    isfinite(sigma_mu) && sigma_mu>=0 || error("invalid frozen sigma_mu")
    d=copy_direction(affine.direction);m=length(epoch.s)
    h=zeros(m);hhat=zeros(m);rho=zeros(m);corrections=Any[]
    for i in eachindex(epoch.cone.lp_scales)
        h[i]=(sigma_mu-epoch.s[i]*epoch.y[i]-d.ds[i]*d.dy[i])/epoch.y[i]
        hhat[i]=h[i]/epoch.cone.lp_scales[i]
    end
    for block in epoch.cone.blocks
        rows=block.offset:block.offset+2;point=copy(epoch.s[rows]);ds=copy(d.ds[rows]);dy=copy(d.dy[rows])
        data=HC.compute(point,ds,dy)
        data.status===:certified || error("current-point corrector unsupported: $(data.reason)")
        push!(corrections,(;offset=block.offset,alpha=0.5,point,ds,dy,data))
        rho[rows]=sigma_mu.*data.ytilde-epoch.y[rows]-data.chi
        hhat[rows]=FA.transform(block,rho[rows],:St)
        h[rows]=FA.transform(block,hhat[rows],:S)
    end
    rt=sigma_mu-epoch.tau*epoch.kappa-d.dtau*d.dkappa
    rhs=SDPX.HSDNewtonRHS(copy(canonical.primal_affine),copy(canonical.dual_affine),canonical.homogeneous_gap,h,rt)
    z=FA.inverse_action(epoch.cone,h)
    all(A->all(isfinite,A),(rho,hhat,z,h)) && isfinite(rt) || error("nonfinite combined RHS")
    provisional=CombinedEpoch(epoch,d,sigma_mu,corrections,rho,hhat,z,rhs,())
    result=CombinedEpoch(epoch,d,sigma_mu,corrections,rho,hhat,z,rhs,fingerprint(provisional))
    verify(result);result
end
function solve(c::CombinedEpoch)
    verify(c);e=c.epoch;m,n=size(e.A);rhs=copy_rhs(c.rhs)
    right=vcat(rhs.dual_affine,FA.transform(e.cone,rhs.primal_affine,:W)-c.hhat,rhs.homogeneous_gap,rhs.tau_kappa)
    solution=e.factor\right
    all(isfinite,solution) || error("nonfinite combined direction")
    dyhat=copy(solution[n+1:n+m]);dshat=c.hhat-dyhat
    direction=SDPX.NewtonDirection(copy(solution[1:n]),FA.transform(e.cone,dyhat,:Wt),
        FA.transform(e.cone,dshat,:S),solution[end-1],solution[end])
    system=SDPX.NewtonSystem(e.A,e.b,e.c,e.cone,e.tau,e.kappa,rhs)
    residual=SDPX.NewtonResidual(system);SDPX.newton_residual!(residual,system,direction)
    verify(c)
    (;direction,residual,rhs,transformed_solution=copy(solution),transformed_rhs=right,
        transformed_residual=e.core*solution-right,production_admitted=false)
end
function action_error(expected,actual,M,source)
    worst=0.0
    for i in eachindex(actual)
        work=RG.point(abs(actual[i]))
        for j in eachindex(source);work=work+RG.abs_interval(M[i,j])*RG.point(abs(source[j]));end
        worst=max(worst,NC.ratio_bound(RG.point(actual[i])-expected[i],work))
    end
    worst
end
function certify(c::CombinedEpoch,result)
    verify(c);e=c.epoch;d=c.affine_direction
    rhs_words(result.rhs)==rhs_words(c.rhs) || return (status=:unsupported,reason=:result_rhs,production_admitted=false)
    canonical=FA.affine_rhs(e)
    FA.fingerprint(c.rhs.primal_affine,c.rhs.dual_affine,c.rhs.homogeneous_gap)==
        FA.fingerprint(canonical.primal_affine,canonical.dual_affine,canonical.homogeneous_gap) ||
        return (status=:unsupported,reason=:semantic_rhs,production_admitted=false)
    prerequisite=NC.certify(e,(;direction=d,rhs=canonical))
    prerequisite.status===:certified || return (status=:unsupported,reason=:affine_prerequisite,production_admitted=false)
    isfinite(c.sigma_mu) && c.sigma_mu>=0 && (iszero(c.sigma_mu)||0x1p-80<=c.sigma_mu<=0x1p80) ||
        return (status=:unsupported,reason=:sigma_domain,production_admitted=false)
    scalar_domain(x)=isfinite(x)&&(iszero(x)||0x1p-80<=abs(x)<=0x1p80)
    all(scalar_domain,(e.tau,e.kappa,d.dtau,d.dkappa)) &&
        all(v->all(scalar_domain,v),(e.s,e.y,d.ds,d.dy)) ||
        return (status=:unsupported,reason=:scalar_polynomial_domain,production_admitted=false)
    try
        gamma3=(RG.point(3)*RG.point(eps(Float64)))/(RG.point(1)-RG.point(3)*RG.point(eps(Float64)))
        limit=(RG.point(128)*gamma3).lo
        block_reports=Any[];total_products=0;total_sums=0
        for (record,block) in zip(c.corrections,e.cone.blocks)
            rows=block.offset:block.offset+2
            record.point==e.s[rows] && record.ds==d.ds[rows] && record.dy==d.dy[rows] ||
                return (status=:unsupported,reason=:corrector_inputs,production_admitted=false)
            # Fresh native certified replay is a conservative policy/provenance
            # guard: its stored numerical words must equal the actual retained
            # words. Its independent EFT gates therefore certify those SAME
            # values, not a substituted reconstruction. Reference arithmetic is
            # never used here and no replay value is written into the epoch.
            replay=HC.compute(record.point,record.ds,record.dy)
            replay.status===:certified && replay.counter_scope===:complete ||
                return (status=:unsupported,reason=:corrector_replay,production_admitted=false)
            correction_words((;record...,data=replay))==correction_words(record) ||
                return (status=:unsupported,reason=:corrector_words,production_admitted=false)
            p=NC.polynomials(block);S=NC.intervals(p.Sn,p.Sd);Theta=NC.intervals(p.Tn,p.Td)
            Gn=NC.pmul(permutedims(p.Wn),p.Wn,p.budget);Gd=EF.mul(p.Wd,p.Wd,p.budget)
            G=NC.intervals(Gn,Gd);rho=c.rho[rows];h=c.rhs.cone_corrector[rows];hh=c.hhat[rows];z=c.z[rows]
            rho_error=0.0
            for i in 1:3
                expected=RG.point(c.sigma_mu)*replay.gradient[i]-RG.point(block.dual[i])-RG.point(replay.chi[i])
                work=RG.point(abs(rho[i]))+RG.point(c.sigma_mu)*RG.abs_interval(replay.gradient[i])+
                    RG.point(abs(block.dual[i]))+RG.point(abs(replay.chi[i]))
                rho_error=max(rho_error,NC.ratio_bound(RG.point(rho[i])-expected,work))
            end
            st=NC.paction(permutedims(p.Sn),p.Sd,rho,p.budget)
            sh=NC.paction(p.Sn,p.Sd,hh,p.budget)
            tr=NC.paction(p.Tn,p.Td,rho,p.budget)
            tz=NC.paction(p.Tn,p.Td,z,p.budget)
            gh=NC.paction(Gn,Gd,h,p.budget)
            errors=(;rho=rho_error,adjoint=action_error(st,hh,permutedims(S),rho),
                recovery=action_error(sh,h,S,hh),forward=action_error(tr,h,Theta,rho),
                inverse_posterior=action_error(tz,h,Theta,z),inverse_action=action_error(gh,z,G,h))
            composed_error=0.0
            for i in 1:3
                work=RG.point(abs(rho[i]))
                for j in 1:3
                    work=work+RG.abs_interval(G[i,j])*RG.point(abs(h[j]))
                    for k in 1:3
                        work=work+RG.abs_interval(G[i,j])*RG.abs_interval(Theta[j,k])*RG.point(abs(rho[k]))
                    end
                end
                composed_error=max(composed_error,NC.ratio_bound(RG.point(z[i])-RG.point(rho[i]),work),
                    NC.ratio_bound(gh[i]-RG.point(rho[i]),work))
            end
            passed=all(v->v<=limit,values(errors)) && composed_error<=limit
            push!(block_reports,(;passed,errors,composed_error,limit,
                intervals=(;st,sh,tr,tz,gh),replay_counter_scope=replay.counter_scope))
            total_products+=p.budget.products+replay.products;total_sums+=p.budget.sums+replay.sums
        end
        scalar_budget=EF.Budget(0,0)
        scalar_numerator=EF.sub(EF.sub(EF.constant(c.sigma_mu),EF.product(scalar_budget,EF.constant(e.tau),EF.constant(e.kappa)),scalar_budget),
            EF.product(scalar_budget,EF.constant(d.dtau),EF.constant(d.dkappa)),scalar_budget)
        scalar_expected=EF.enclose(scalar_numerator)
        scalar_work=RG.point(abs(c.rhs.tau_kappa))+RG.point(c.sigma_mu)+RG.point(abs(e.tau))*RG.point(abs(e.kappa))+
            RG.point(abs(d.dtau))*RG.point(abs(d.dkappa))
        scalar_error=NC.ratio_bound(RG.point(c.rhs.tau_kappa)-scalar_expected,scalar_work)
        orthant_reports=Any[]
        for i in eachindex(e.cone.lp_scales)
            numerator=EF.sub(EF.sub(EF.constant(c.sigma_mu),EF.product(scalar_budget,EF.constant(e.s[i]),EF.constant(e.y[i])),scalar_budget),
                EF.product(scalar_budget,EF.constant(d.ds[i]),EF.constant(d.dy[i])),scalar_budget)
            expected=EF.enclose(numerator)/RG.point(e.y[i]);h=c.rhs.cone_corrector[i];hh=c.hhat[i];ell=e.cone.lp_scales[i]
            work=RG.point(abs(h))+(RG.point(c.sigma_mu)+RG.point(abs(e.s[i]))*RG.point(abs(e.y[i]))+
                RG.point(abs(d.ds[i]))*RG.point(abs(d.dy[i])))/RG.point(e.y[i])
            error=NC.ratio_bound(RG.point(h)-expected,work)
            transformed=NC.ratio_bound(RG.point(hh)-RG.point(h)/RG.point(ell),RG.point(abs(hh))+RG.point(abs(h))/RG.point(ell))
            push!(orthant_reports,(;passed=error<=limit && transformed<=limit,error,transformed,expected))
        end
        equations=NC._certify_equations(e,result)
        passed=all(x->x.passed,block_reports) && all(x->x.passed,orthant_reports) && scalar_error<=limit && equations.status===:certified
        complete=prerequisite.counter_scope===:complete && get(equations,:counter_scope,:unavailable)===:complete
        counts=complete ? (;counter_scope=:complete,
            products=total_products+scalar_budget.products+prerequisite.products+equations.products,
            sums=total_sums+scalar_budget.sums+prerequisite.sums+equations.sums) : (;counter_scope=:partial_equation_counts)
        (;status=passed ? :certified : :unsupported,reason=:combined_native_bounds,block_reports,orthant_reports,
            scalar_error,scalar_expected,equations,counts...,production_admitted=false)
    catch err
        err isa RG.EnclosureFailure || err isa RG.Phi.ArithmeticDomainError || rethrow()
        (;status=:unsupported,reason=err isa RG.EnclosureFailure ? err.reason : :eft_domain,
            counter_scope=:unavailable,production_admitted=false)
    end
end
end
