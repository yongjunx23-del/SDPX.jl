using Test, TOML, LinearAlgebra, SDPX
include("factor_preserving_affine.jl")
include("factor_affine_reference.jl")
const FA=FactorPreservingAffine
const FAR=FactorAffineReference
const HF=FA.HalfPowerCompensatedFactor
const Q=Rational{BigInt}
const FACTOR_AFFINE_RESULTS=Any[]
@testset "compensated actual-shadow half-Power factor candidate" begin
    for id in (17,19)
        row=TOML.parsefile(joinpath(@__DIR__,"fixtures/factor_affine_trial_$id.toml"))
        epoch=FA.build(row;factor_mode=:compensated_half_candidate)
        candidate=FA.solve(epoch);reference=FAR.physical(epoch,candidate)
        @test !candidate.production_admitted
        @test epoch.factor_mode===:compensated_half_candidate
        @test all(x->x<=Q(FA.PHYSICAL_FORCING),reference.errors)
        for (block,construction) in zip(epoch.cone.blocks,epoch.construction)
            result=construction.factor_info;shadow=block.shadow;x,y,z=Q.(shadow)
            p=x*y;d=p-z*z;delta=d/p
            @test result.status===:formed
            @test sum(Q.(result.expansion))==d
            @test Q(result.determinant.lo)<=d<=Q(result.determinant.hi)
            @test Q(result.product.lo)<=p<=Q(result.product.hi)
            @test Q(result.delta_interval.lo)<=delta<=Q(result.delta_interval.hi)
            @test result.two_prod_calls==2 && result.two_sum_calls==6
            @test all(v->iszero(v)||!issubnormal(v),result.expansion)
            @test all(v->denominator(Q(v)*big(2)^168)==1,result.expansion)
            Li=FAR.inverse_lower(Q.(block.L));H=FAR.true_hessian(shadow)
            error=Li*H*Li'-Matrix{Q}(I,3,3)
            @test sum(abs2,error)<=Q(FA.RG.KAPPA)^2
            certificate=construction.runtime_geometry
            @test certificate.status===:certified
            @test Q(certificate.eta)^2>=sum(abs2,error)
            xq,yq,zq=Q.(shadow);dq=xq*yq-zq*zq
            gradient=Q[-yq/dq-1/(2xq),-xq/dq-1/(2yq),2zq/dq]
            residual=-gradient-Q.(block.dual)
            exact_v=Li*residual
            @test all(i->Q(certificate.v[i].lo)<=exact_v[i]<=Q(certificate.v[i].hi),1:3)
            decrement2=dot(residual,vec(FAR.exact_solve(H,reshape(residual,3,1))))
            @test Q(certificate.decrement)^2>=decrement2
            @test certificate.eta<=FA.RG.KAPPA && certificate.decrement<=FA.RG.KAPPA
            @test certificate.products<=1<<16 && certificate.sums<=1<<20
            gram=Q.(block.L)*Q.(block.L)'
            for j in 1:3,i in j:3
                work=abs(gram[i,j])+abs(H[i,j])
                @test iszero(work) ? gram[i,j]==H[i,j] : Q(certificate.backward)>=abs(gram[i,j]-H[i,j])/work
            end
            for shift in (-4,4)
                gauged=[ldexp(shadow[1],shift),ldexp(shadow[2],-shift),shadow[3]]
                alternate=HF.factor(gauged)
                @test alternate.status===:formed
                scale=Diagonal([ldexp(1.0,-shift),ldexp(1.0,shift),1.0])
                @test FA.words(alternate.L)==FA.words(scale*block.L)
            end
        end
        for metric in reference.metrics
            @test metric["factor_formula_frobenius_squared"]<=Q(FA.RG.KAPPA)^2
            @test metric["true_hessian_formula_frobenius_squared"]<=Q(FA.RG.KAPPA)^2
        end
        info=FAR.rounded_diagnostics(reference)
        info["candidate_legacy_factor_gates"]=[c.legacy_ok for c in epoch.construction]
        info["native_true_geometry"]=[Dict("status"=>string(c.runtime_geometry.status),
            "eta"=>c.runtime_geometry.eta,"decrement"=>c.runtime_geometry.decrement,
            "backward"=>c.runtime_geometry.backward,"products"=>c.runtime_geometry.products,
            "sums"=>c.runtime_geometry.sums,
            "F_bounds"=>[[v.lo,v.hi] for v in vec(c.runtime_geometry.F)],
            "v_bounds"=>[[v.lo,v.hi] for v in c.runtime_geometry.v]) for c in epoch.construction]
        push!(FACTOR_AFFINE_RESULTS,(;epoch,candidate,reference,info))
        println("COMPENSATED_HALF_FACTOR ",id," ",info)
    end
    control=HF.factor([1.,1.,0.])
    @test control.status===:formed
    @test maximum(abs,control.L*control.L'-Diagonal([1.5,1.5,2.]))<=16eps(Float64)
    @test HF.factor([1.,1.,2.]).status===:unsupported
    @test HF.factor([0.,1.,0.]).status===:unsupported
    @test HF.factor([Inf,1.,0.]).status===:unsupported
    @test HF.factor([0x1p40,1.,0.]).status===:unsupported
    @test HF.factor(Float32[1,1,0]).status===:unsupported
    verifier=FA.HalfPowerFactorCertificate
    central=verifier.verify([1.,1.,0.],control.L,[1.5,1.5,0.])
    @test central.status===:certified
    corrupt=copy(control.L);corrupt[1,1]*=1.01
    @test verifier.verify([1.,1.,0.],corrupt,[1.5,1.5,0.]).status===:unsupported
    @test verifier.verify([1.,1.,0.],control.L,[1.6,1.5,0.]).status===:unsupported
    @test verifier.verify([0x1p-32,1.,0.],control.L,[1.5,1.5,0.]).status===:unsupported
end
