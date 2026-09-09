using Test,TOML,LinearAlgebra,SDPX
include("factor_preserving_affine.jl")
include("factor_affine_reference.jl")
include("half_power_native_corrector.jl")
include("native_factor_affine_certificate.jl")
include("factor_combined_epoch.jl")
const FA=FactorPreservingAffine
const FAR=FactorAffineReference
const FC=FactorCombinedEpoch
const Q=Rational{BigInt}
const COMBINED_RESULTS=Any[]
inside(I,x)=Q(I.lo)<=x<=Q(I.hi)
function refreeze(c;rho=copy(c.rho),hhat=copy(c.hhat),z=copy(c.z),rhs=FC.copy_rhs(c.rhs),corrections=c.corrections,direction=c.affine_direction,sigma_mu=c.sigma_mu)
    p=FC.CombinedEpoch(c.epoch,direction,sigma_mu,corrections,rho,hhat,z,rhs,())
    FC.CombinedEpoch(c.epoch,direction,sigma_mu,corrections,rho,hhat,z,rhs,FC.fingerprint(p))
end
function exact_action_error(expected,actual,M,source)
    residual=Q.(actual)-expected
    work=abs.(Q.(actual))+abs.(M)*abs.(Q.(source))
    maximum(iszero(work[i]) ? (iszero(residual[i]) ? Q(0) : error("zero work")) : abs(residual[i])/work[i] for i in eachindex(work))
end
function refreeze_epoch(e)
    values=Tuple(k===:frozen ? FA.epoch_fingerprint(e) : getproperty(e,k) for k in fieldnames(typeof(e)))
    typeof(e)(values...)
end
direction_fields(d)=(;dx=copy(d.dx),dy=copy(d.dy),ds=copy(d.ds),dtau=d.dtau,dkappa=d.dkappa)
rhs_fields(r)=(;primal=copy(r.primal_affine),dual=copy(r.dual_affine),gap=r.homogeneous_gap,h=copy(r.cone_corrector),tau_kappa=r.tau_kappa)
@testset "frozen combined factor epochs" begin
    for id in (17,19)
        row=TOML.parsefile(joinpath(@__DIR__,"fixtures/factor_affine_trial_$id.toml"))
        e=FA.build(row;factor_mode=:compensated_half_candidate);affine=FA.solve(e)
        frozen=FA.epoch_fingerprint(e);affine_bits=FA.fingerprint(affine.direction.dx,affine.direction.ds,affine.direction.dy)
        for sigma in (0.,0.25,0.75)
            combined=FC.build(e,affine;sigma_mu=sigma*e.mu);result=FC.solve(combined);certificate=FC.certify(combined,result)
            println("COMBINED_CERTIFICATE ",id," sigma=",sigma," status=",certificate.status," reason=",certificate.reason)
            if certificate.status!==:certified;println(certificate);end
            @test certificate.status===:certified
            certificate.status===:certified || continue
            @test certificate.counter_scope===:complete
            @test !result.production_admitted && !certificate.production_admitted
            @test FA.epoch_fingerprint(e)==frozen
            @test FA.fingerprint(affine.direction.dx,affine.direction.ds,affine.direction.dy)==affine_bits
            @test FC.rhs_words(result.rhs)==FC.rhs_words(combined.rhs)
            @test result.rhs.cone_corrector !== combined.rhs.cone_corrector
            @test combined.affine_direction.ds !== affine.direction.ds
            @test combined.affine_direction.dy !== affine.direction.dy
            ref=FAR.physical(e,result)
            @test all(x->x<=Q(FA.PHYSICAL_FORCING),ref.errors)
            for group in 1:5
                @test all(i->inside(certificate.equations.bounds[group][i],ref.residuals[group][i]),eachindex(ref.residuals[group]))
                @test Q(certificate.equations.errors[group])>=ref.errors[group]
            end
            scalar=Q(combined.sigma_mu)-Q(e.tau)*Q(e.kappa)-Q(affine.direction.dtau)*Q(affine.direction.dkappa)
            @test inside(certificate.scalar_expected,scalar)
            scalar_work=abs(Q(combined.rhs.tau_kappa))+Q(combined.sigma_mu)+abs(Q(e.tau)*Q(e.kappa))+
                abs(Q(affine.direction.dtau)*Q(affine.direction.dkappa))
            @test abs(Q(combined.rhs.tau_kappa)-scalar)/scalar_work<=Q(certificate.scalar_error)
            for i in eachindex(e.cone.lp_scales)
                expected=(Q(combined.sigma_mu)-Q(e.s[i])*Q(e.y[i])-Q(affine.direction.ds[i])*Q(affine.direction.dy[i]))/Q(e.y[i])
                @test inside(certificate.orthant_reports[i].expected,expected)
                work=abs(Q(combined.rhs.cone_corrector[i]))+(Q(combined.sigma_mu)+abs(Q(e.s[i])*Q(e.y[i]))+
                    abs(Q(affine.direction.ds[i])*Q(affine.direction.dy[i])))/Q(e.y[i])
                @test abs(Q(combined.rhs.cone_corrector[i])-expected)/work<=Q(certificate.orthant_reports[i].error)
            end
            G=ref.W'*ref.W
            for (k,b) in enumerate(e.cone.blocks)
                rows=b.offset:b.offset+2;report=certificate.block_reports[k];ri=Q.(combined.rho[rows]);hi=Q.(combined.rhs.cone_corrector[rows])
                hhi=Q.(combined.hhat[rows]);zi=Q.(combined.z[rows]);Si=ref.S[rows,rows];Ti=ref.Theta[rows,rows];Gi=G[rows,rows]
                for (intervals,exact) in ((report.intervals.st,Si'*ri),(report.intervals.sh,Si*hhi),
                    (report.intervals.tr,Ti*ri),(report.intervals.tz,Ti*zi),(report.intervals.gh,Gi*hi))
                    @test all(i->inside(intervals[i],exact[i]),1:3)
                end
                @test exact_action_error(Si'*ri,hhi,Si',ri)<=Q(report.errors.adjoint)
                @test exact_action_error(Si*hhi,hi,Si,hhi)<=Q(report.errors.recovery)
                @test exact_action_error(Ti*ri,hi,Ti,ri)<=Q(report.errors.forward)
                @test exact_action_error(Ti*zi,hi,Ti,zi)<=Q(report.errors.inverse_posterior)
                @test exact_action_error(Gi*hi,zi,Gi,hi)<=Q(report.errors.inverse_action)
                work=abs.(ri)+abs.(Gi)*abs.(hi)+abs.(Gi)*abs.(Ti)*abs.(ri)
                @test maximum(abs.(zi-ri)./work)<=Q(report.composed_error)
                @test maximum(abs.(Gi*hi-ri)./work)<=Q(report.composed_error)
                data=combined.corrections[k].data
                x,y,z=Q.(e.s[rows]);gap=x*y-z*z;gradient=Q[y/gap+1/(2x),x/gap+1/(2y),-2z/gap]
                expected=Q(combined.sigma_mu)*gradient-Q.(e.y[rows])-Q.(data.chi)
                work=abs.(ri)+Q(combined.sigma_mu)*abs.(gradient)+abs.(Q.(e.y[rows]))+abs.(Q.(data.chi))
                @test maximum(abs.(ri-expected)./work)<=Q(report.errors.rho)
                @test data.L !== b.L
                @test combined.corrections[k].point==e.s[rows]
            end
            if sigma==0.25
                mismatched=deepcopy(e);mismatched.s[4]+=0.125;mismatched=refreeze_epoch(mismatched)
                @test FA.verify(mismatched)
                @test NativeFactorAffineCertificate.certify(mismatched,
                    (;direction=affine.direction,rhs=FA.affine_rhs(mismatched))).reason===:metric_epoch_point
                @test FC.certify(refreeze(combined;sigma_mu=combined.sigma_mu+1e-3),result).status===:unsupported
                @test FC.certify(refreeze(combined;sigma_mu=NaN),result).reason===:sigma_domain
                omitted=SDPX.HSDNewtonRHS(copy(combined.rhs.primal_affine),copy(combined.rhs.dual_affine),
                    combined.rhs.homogeneous_gap,copy(combined.rhs.cone_corrector),combined.sigma_mu-e.tau*e.kappa)
                @test FC.certify(refreeze(combined;rhs=omitted),(;result...,rhs=FC.copy_rhs(omitted))).status===:unsupported
                wrongrhs=FC.copy_rhs(combined.rhs);wrongrhs.primal_affine[1]+=1e-3
                @test FC.certify(refreeze(combined;rhs=wrongrhs),(;result...,rhs=FC.copy_rhs(wrongrhs))).reason===:semantic_rhs
                extra=(;direction_fields(affine.direction)...,dx=vcat(affine.direction.dx,0.))
                @test NativeFactorAffineCertificate.certify(e,(;direction=extra,rhs=FA.affine_rhs(e))).reason===:direction_rhs_shape
                narrower=(;direction_fields(affine.direction)...,dx=Float32.(affine.direction.dx))
                @test NativeFactorAffineCertificate.certify(e,(;direction=narrower,rhs=FA.affine_rhs(e))).reason===:direction_rhs_type
                shaped=deepcopy(combined.corrections);old=shaped[1]
                shaped[1]=(;old...,data=(;old.data...,L=reshape(copy(vec(old.data.L)),1,9)))
                @test FC.certify(refreeze(combined;corrections=shaped),result).reason===:corrector_words
                @test_throws ErrorException FC.certify(refreeze(combined;hhat=vcat(combined.hhat,0.)),result)
            end
            @test NativeFactorAffineCertificate.certify(e,result).reason===:non_affine
            @test_throws ErrorException FA.solve(e,result.rhs)
            # Refingerprinted corruptions must fail numerical/provenance gates,
            # not merely the immutable-word integrity check.
            badrho=copy(combined.rho);badrho[4]+=1e-3;bad=refreeze(combined;rho=badrho)
            @test FC.verify(bad)
            @test FC.certify(bad,result).status===:unsupported
            badhh=copy(combined.hhat);badhh[4]+=1.;bad=refreeze(combined;hhat=badhh)
            @test FC.certify(bad,result).status===:unsupported
            badrhs=FC.copy_rhs(combined.rhs);badrhs.cone_corrector[4]+=1e-3;bad=refreeze(combined;rhs=badrhs)
            @test FC.certify(bad,(;result...,rhs=FC.copy_rhs(badrhs))).status===:unsupported
            badrhs=SDPX.HSDNewtonRHS(copy(combined.rhs.primal_affine),copy(combined.rhs.dual_affine),
                combined.rhs.homogeneous_gap,copy(combined.rhs.cone_corrector),combined.rhs.tau_kappa+1e-3)
            bad=refreeze(combined;rhs=badrhs)
            @test FC.certify(bad,(;result...,rhs=FC.copy_rhs(badrhs))).status===:unsupported
            badrecords=deepcopy(combined.corrections);badrecords[1].data.ytilde[1]+=1.
            @test FC.certify(refreeze(combined;corrections=badrecords),result).reason===:corrector_words
            stale=deepcopy(combined);stale.rho[4]+=1.
            @test_throws ErrorException FC.solve(stale)
            @test_throws ErrorException FC.certify(stale,result)
            wrong=FC.copy_direction(result.direction);wrong.dx[1]+=1.
            @test FC.certify(combined,(;result...,direction=wrong)).status===:unsupported
            buffers=Any[combined.rho,combined.hhat,combined.z,combined.rhs.primal_affine,
                combined.rhs.dual_affine,combined.rhs.cone_corrector,combined.affine_direction.dx,
                combined.affine_direction.dy,combined.affine_direction.ds,result.direction.dx,result.direction.dy,result.direction.ds]
            for record in combined.corrections
                append!(buffers,[record.point,record.ds,record.dy,record.data.L,record.data.u,
                    record.data.raw,record.data.swap,record.data.averaged,record.data.chi,record.data.ytilde])
            end
            @test all(i->all(j->!Base.mightalias(buffers[i],buffers[j]),i+1:length(buffers)),eachindex(buffers))
            push!(COMBINED_RESULTS,(;source_record=id,sigma,epoch=(;A=Matrix(e.A),b=e.b,c=e.c,x=e.x,s=e.s,y=e.y,tau=e.tau,kappa=e.kappa,mu=e.mu,
                Ahat=e.Ahat,bhat=e.bhat,core=e.core,lu=e.factor.factors,pivots=e.factor.ipiv,lp_scales=e.cone.lp_scales,
                blocks=[(;offset=b.offset,L=b.L,R=b.R,scale=b.scale,mu=b.mu,primal=b.primal,dual=b.dual,shadow=b.shadow) for b in e.cone.blocks]),
                affine=direction_fields(combined.affine_direction),sigma_mu=combined.sigma_mu,corrections=combined.corrections,
                rho=combined.rho,hhat=combined.hhat,z=combined.z,rhs=rhs_fields(combined.rhs),
                direction=direction_fields(result.direction),transformed_rhs=result.transformed_rhs,transformed_solution=result.transformed_solution,
                certificate,exact_errors=string.(ref.errors)))
            println("COMBINED_VERIFIED ",id," sigma=",sigma," physical=",certificate.equations.errors,
                " shifts=",[x.errors.forward for x in certificate.block_reports])
        end
    end
end
