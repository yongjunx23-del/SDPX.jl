using Test,TOML,LinearAlgebra,SDPX
include("factor_preserving_affine.jl")
include("factor_affine_reference.jl")
include("half_power_native_corrector.jl")
const FA=FactorPreservingAffine
const FAR=FactorAffineReference
const HC=HalfPowerNativeCorrector
const Q=Rational{BigInt}
const CORRECTOR_RESULTS=Any[]
function exact_chi(s,a,b)
    x,y,z=Q.(s);a=Q.(a);b=Q.(b);d=x*y-z*z;q=Q[y,x,-2z];J=Q.([0 1 0;1 0 0;0 0 -2])
    A=dot(q,a);B=dot(q,b);C=B*(J*a)+A*(J*b)+q*dot(a,J*b)
    -C/(2d^2)+q*(A*B/d^3)+Q[a[1]*b[1]/(2x^3),a[2]*b[2]/(2y^3),0]
end
inside(I,x)=Q(I.lo)<=x<=Q(I.hi)
@testset "native current-point half-Power corrector" begin
    for s in ([1.,1.,0.],[2.,3.,0.5],[2.,3.,-0.5]),
        (a,b) in (([1.,0.,0.],[1.,0.,0.]),([0.25,0.125,-0.5],[0.5,-0.25,0.125]))
        budget=FA.HalfPowerFactorCertificate.Budget(0,0)
        value=HC.third_enclosure(s,a,b,budget);swapped=HC.third_enclosure(s,b,a,budget);ref=exact_chi(s,a,b)
        @test all(i->inside(value[i],ref[i]),1:3)
        @test all(i->inside(swapped[i],ref[i]),1:3)
        @test exact_chi(s,a,b)==exact_chi(s,b,a)
        H=FAR.true_hessian(s)
        @test dot(Q.(s),ref)==dot(Q.(a),H*Q.(b))
        @test exact_chi(s,s,b)==H*Q.(b)
    end
    central=HC.compute([1.,1.,0.],[1.,0.,0.],[1.5,0.,0.])
    central.status===:certified || println("CENTRAL_CORRECTOR ",central)
    @test central.status===:certified
    @test central.chi==[1.5,0.,0.]
    @test !HC.raw_accuracy([1.75,-0.25,0.],central.first) # symmetric/Euler-compatible wrong vector
    @test !HC.raw_accuracy([-0.5,0.,0.],central.first) # rejected oracle sign error
    for id in (17,19)
        row=TOML.parsefile(joinpath(@__DIR__,"fixtures/factor_affine_trial_$id.toml"))
        epoch=FA.build(row;factor_mode=:compensated_half_candidate);affine=FA.solve(epoch)
        for block in epoch.cone.blocks
            rows=block.offset:block.offset+2;s=epoch.s[rows];ds=affine.direction.ds[rows];dy=affine.direction.dy[rows]
            result=HC.compute(s,ds,dy)
            result.status===:certified || println("UNSUPPORTED_CORRECTOR ",id,"/",block.offset," ",result)
            @test result.status===:certified
            @test result.factor.reason===:true_stored_hessian_only
            @test !hasproperty(result.factor,:decrement)
            @test !result.production_admitted
            H=FAR.true_hessian(s);Li=FAR.inverse_lower(Q.(result.L));E=Li*H*Li'-Matrix{Q}(I,3,3)
            @test Q(result.factor.eta)^2>=sum(abs2,E)
            residual=H*Q.(result.u)-Q.(dy)
            @test all(i->inside(result.posterior.intervals[i],residual[i]),1:3)
            ref=exact_chi(s,ds,result.u)
            @test all(i->inside(result.first[i],ref[i])&&inside(result.swapped[i],ref[i]),1:3)
            x,y,z=Q.(s);d=x*y-z*z;grad=Q[y/d+1/(2x),x/d+1/(2y),-2z/d]
            @test all(i->inside(result.gradient[i],grad[i]),1:3)
            post=dot(Q.(s),Q.(result.chi))-dot(Q.(ds),Q.(dy))
            work=sum(abs.(Q.(s).*Q.(result.chi)))+sum(abs.(Q.(ds).*Q.(dy)))
            @test iszero(work) ? iszero(post) : abs(post)/work<=Q(result.final_error)
            @test result.raw_error<=0x1p-16
            @test result.projection_error<=0x1p-16
            push!(CORRECTOR_RESULTS,(;id,offset=block.offset,s,ds,dy,result))
            println("CORRECTOR_CERTIFIED ",id,"/",block.offset," solve=",result.posterior.worst,
                " raw_euler=",result.raw_error," post_euler=",result.final_error,
                " products=",result.products," sums=",result.sums)
        end
    end
end
