using Test, TOML, LinearAlgebra, SDPX
include("factor_preserving_affine.jl")
include("factor_affine_reference.jl")
const FA=FactorPreservingAffine
const FAR=FactorAffineReference
const Q=Rational{BigInt}
const FACTOR_AFFINE_RESULTS=Any[]
function action_ok(actual,M,v)
    reference=M*Q.(v);error=maximum(abs,Q.(actual)-reference)
    work=maximum(abs.(M)*abs.(Q.(v)))
    iszero(work) ? iszero(error) : error<=Q(FA.RG.KAPPA)*work
end
@testset "unpromoted factor-preserving affine epochs" begin
    for id in (17,19)
        row=TOML.parsefile(joinpath(@__DIR__,"fixtures/factor_affine_trial_$id.toml"))
        epoch=FA.build(row);candidate=FA.solve(epoch)
        reference=FAR.physical(epoch,candidate)
        @test !candidate.production_admitted
        @test !all(x->x.old_result[1],epoch.root_reports)
        @test all(x->x.root.status===:qualified,epoch.root_reports)
        @test all(x->get(x.native,"factor_certificate",false),epoch.root_reports)
        @test all(x->!x.native["public_valid"],epoch.root_reports)
        # This reference certificate is for the unpromoted, factor-defined
        # unregularized metric; it never erases failed old production gates.
        @test all(e->e<=Q(FA.PHYSICAL_FORCING),reference.errors)
        for metric in reference.metrics
            @test metric["exact_target_secant"]
            @test metric["scale_relative_error"]<=8Q(eps(Float64))
            @test metric["factor_formula_frobenius_squared"]<=Q(FA.RG.KAPPA)^2
        end
        if id==17
            @test all(x->x["true_hessian_formula_frobenius_squared"]<=Q(FA.RG.KAPPA)^2,reference.metrics)
        else
            # ||E||2 >= ||E||F/sqrt(3): this is a real true-metric violation,
            # not merely failure of an overly conservative upper bound.
            @test reference.metrics[3]["true_hessian_formula_frobenius_squared"]>3Q(FA.RG.KAPPA)^2
            @test !epoch.root_reports[3].native["inverse"]
        end
        sources=vcat([collect(Matrix{Float64}(I,epoch.cone.dimension,epoch.cone.dimension)[:,i]) for i in 1:epoch.cone.dimension],
            [Vector(epoch.A[:,i]) for i in axes(epoch.A,2)],
            [epoch.s,epoch.y,epoch.b,candidate.rhs.primal_affine,candidate.rhs.cone_corrector,
             candidate.direction.dy,candidate.direction.ds])
        for v in sources,(kind,M) in ((:S,reference.S),(:St,reference.S'),(:W,reference.W),(:Wt,reference.W'))
            @test action_ok(FA.transform(epoch.cone,v,kind),M,v)
        end
        @test !action_ok(FA.transform(epoch.cone,epoch.b,:Wt),reference.S',epoch.b)
        # Actual physical recovery is retained, not replaced by a roundtrip.
        @test all(isfinite,candidate.direction.dy) && all(isfinite,candidate.direction.ds)
        frozen=FA.fingerprint(epoch.A.nzval,epoch.b,epoch.c,epoch.s,epoch.y)
        row["s_bits"][1]="3ff0000000000000"
        @test FA.fingerprint(epoch.A.nzval,epoch.b,epoch.c,epoch.s,epoch.y)==frozen
        @test FA.verify(epoch)
        for change in (:L,:R,:lp,:A,:core,:factor)
            bad=deepcopy(epoch)
            if change==:L;bad.cone.blocks[1].L[1,1]=nextfloat(bad.cone.blocks[1].L[1,1])
            elseif change==:R;bad.cone.blocks[1].R[1,1]=nextfloat(bad.cone.blocks[1].R[1,1])
            elseif change==:lp;bad.cone.lp_scales[1]=nextfloat(bad.cone.lp_scales[1])
            elseif change==:A;bad.A.nzval[1]=nextfloat(bad.A.nzval[1])
            elseif change==:core;bad.core[1,1]=1.0
            else;bad.factor.factors[1,1]=nextfloat(bad.factor.factors[1,1])
            end
            @test_throws ErrorException FA.solve(bad,candidate.rhs)
        end
        rhs=candidate.rhs
        other=SDPX.HSDNewtonRHS(rhs.primal_affine.*0.75,rhs.dual_affine.*1.25,
            rhs.homogeneous_gap*0.875,copy(rhs.cone_corrector),rhs.tau_kappa*0.75)
        factor_bits=FA.words(epoch.factor.factors)
        second=FA.solve(epoch,other);second_reference=FAR.physical(epoch,second)
        @test FA.words(epoch.factor.factors)==factor_bits
        @test all(e->e<=Q(FA.PHYSICAL_FORCING),second_reference.errors)
        rhs_snapshot=copy(second.rhs.primal_affine)
        other.primal_affine[1]+=1
        @test second.rhs.primal_affine==rhs_snapshot
        @test !Base.mightalias(second.rhs.primal_affine,other.primal_affine)
        non_affine=SDPX.HSDNewtonRHS(copy(rhs.primal_affine),copy(rhs.dual_affine),rhs.homogeneous_gap,
            zeros(length(rhs.cone_corrector)),rhs.tau_kappa)
        @test_throws ErrorException FA.solve(epoch,non_affine)
        info=FAR.rounded_diagnostics(reference)
        push!(FACTOR_AFFINE_RESULTS,(;epoch,candidate,reference,info))
        println("FACTOR_AFFINE_REFERENCE ",id," ",info)
    end
end
