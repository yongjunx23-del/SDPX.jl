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
end
