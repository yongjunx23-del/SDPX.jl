using Test, SDPX, LinearAlgebra
isdefined(@__MODULE__,:StandardConicMath) || include(joinpath(@__DIR__,"..","validation","scientific_core","StandardConicMath.jl"))

@testset "logarithmic exponential barrier kernel" begin
    # Avoid avoidable p²/p³ overflow; fail closed when positive curvature
    # itself cannot be represented at the requested arithmetic.
    Hwide=zeros(3,3);qwide=zeros(3)
    SDPX.exp_logarithmic_hessian!(Hwide,[-1e155,1.0,1.0])
    expected=setprecision(BigFloat,512) do;Float64(inv(BigFloat(1e155))^2);end
    @test abs(Hwide[1,1]-expected)<=eps(0.0)
    SDPX.exp_logarithmic_third!(qwide,[-1e105,1.0,1.0],[1.0,0.0,0.0],[1.0,0.0,0.0])
    expected3=setprecision(BigFloat,512) do;Float64(2inv(BigFloat(1e105))^3);end
    @test abs(qwide[1]-expected3)<=2eps(0.0)
    @test_throws DomainError SDPX.exp_logarithmic_hessian!(Hwide,[-1e200,1.0,1.0])
    arithmetic=Tuple{DataType,Int}[(Float64,53),(BigFloat,256),(BigFloat,512)]
    isdefined(Main,:MultiFloats) && push!(arithmetic,(Main.MultiFloats.Float64x4,212))
    for (T,bits) in arithmetic
        setprecision(BigFloat,max(256,bits)) do
            tol=T(20000)*eps(T)
            for p in (T[-2,1,1],T[-1,2,3],T[0,1,2],T[1,2,4])
                before=deepcopy(p)
                g=SDPX.alloc_zeros(T,3);H=SDPX.alloc_zeros(T,3,3);q=SDPX.alloc_zeros(T,3)
                gr,Hr=StandardConicMath.exp_gradient_hessian(p)
                SDPX.exp_logarithmic_gradient!(g,p);SDPX.exp_logarithmic_hessian!(H,p)
                @test norm(g-gr,Inf)<=tol*max(one(T),norm(gr,Inf))
                @test norm(H-Hr,Inf)<=tol*max(one(T),norm(Hr,Inf))
                @test abs(SDPX.exp_logarithmic_barrier(p)-StandardConicMath.exp_barrier(p))<=tol
                @test abs(dot(p,g)+3)<=tol
                for h in (T[1,0,0],T[0,1,0],T[0,0,1],T[1/8,-1/4,1/2])
                    qr=StandardConicMath.exp_third(p,h,p)
                    SDPX.exp_logarithmic_third!(q,p,h,p)
                    @test norm(q-qr,Inf)<=tol*max(one(T),norm(qr,Inf))
                    SDPX.exp_logarithmic_third!(q,p,h,h)
                    @test dot(h,q)^2<=4dot(h,H*h)^3+tol*max(one(T),4dot(h,H*h)^3)
                end
                @test p==before
                if T===BigFloat
                    snapshot=deepcopy(H)
                    SDPX._store_owned_scalar!(H,2,T(17))
                    @test H[1,2]==snapshot[1,2] # independently owned symmetric entries
                    @test p==before
                end
            end
            for bad in (T[0,0,1],T[0,1,1],T[2,1,1],T[NaN,1,2],T[0,1,Inf])
                @test_throws DomainError SDPX.exp_logarithmic_barrier(bad)
            end
            # Negative control: independently confirm the old exp-gap Hessian's
            # third derivative violates self-concordance at (-log4,1,1).
            p=T[-log(T(4)),1,1];h=T[1,0,0]
            step=eps(T)^(one(T)/T(4))
            oldH=SDPX.exp_barrier_hessian(p...)
            Hp=SDPX.exp_barrier_hessian((p+step*h)...)
            Hm=SDPX.exp_barrier_hessian((p-step*h)...)
            d3=(Hp[1,1]-Hm[1,1])/(2step)
            @test abs(d3)>2oldH[1,1]^(T(3)/2)
        end
    end
end
