using Test, SDPX, LinearAlgebra
isdefined(@__MODULE__,:StandardConicMath) || include(joinpath(@__DIR__,"..","validation","scientific_core","StandardConicMath.jl"))

@testset "logarithmic Exp actual Fenchel inverse" begin
    arithmetic=Tuple{DataType,Int}[(Float64,53),(BigFloat,256),(BigFloat,512)]
    isdefined(Main,:MultiFloats) && push!(arithmetic,(Main.MultiFloats.Float64x4,212))
    for (T,bits) in arithmetic
        setprecision(BigFloat,max(256,bits)) do
            tol=T(20000)*eps(T)
            out=SDPX.alloc_zeros(T,3)
            for p in (T[-1,1,1],T[-2,1,1],T[-1,2,3],T[0,1,2],T[1,2,4])
                g,H=StandardConicMath.exp_gradient_hessian(p)
                d=-g;before=deepcopy(d)
                result=SDPX.exp_logarithmic_conjugate!(out,d)
                @test norm(out-p,Inf)<=tol*max(one(T),norm(p,Inf))
                @test abs(result.value+StandardConicMath.exp_barrier(p)+3)<=tol
                @test abs(dot(out,d)-3)<=tol
                @test abs(result.root_residual)<=T(16)*eps(T)*result.dual_margin
                @test 1<=result.iterations<=64
                @test d==before
                if T===BigFloat
                    step=exp2(-T(bits÷4));bound=sqrt(step)*step
                    Hi=inv(H)
                    for h in (T[1,0,0],T[0,1,0],T[0,0,1])
                        plus=SDPX.alloc_zeros(T,3);minus=SDPX.alloc_zeros(T,3)
                        fp=SDPX.exp_logarithmic_conjugate!(plus,d+step*h).value
                        fm=SDPX.exp_logarithmic_conjugate!(minus,d-step*h).value
                        @test abs((fp-fm)/(2step)+dot(p,h))<=bound*max(one(T),abs(dot(p,h)))
                        @test norm(-(plus-minus)/(2step)-Hi*h,Inf)<=bound*max(one(T),norm(Hi*h,Inf))
                        _,Hp=StandardConicMath.exp_gradient_hessian(plus)
                        _,Hm=StandardConicMath.exp_gradient_hessian(minus)
                        expected=Hi*StandardConicMath.exp_third(p,Hi*h,Hi*h)
                        @test norm((inv(Hp)-inv(Hm))*h/(2step)-expected,Inf)<=bound*max(one(T),norm(expected,Inf))
                    end
                end
            end
            snapshot=deepcopy(out)
            for bad in (T[0,0,1],T[-1,-2,1],T[-1,-1,1],T[-1,0,0],T[-1,NaN,1])
                @test_throws DomainError SDPX.exp_logarithmic_conjugate!(out,bad)
                @test out==snapshot
            end
            @test_throws DomainError SDPX.exp_logarithmic_conjugate!(out,T[-1,0,2];max_iterations=1)
            @test out==snapshot
        end
    end
end
