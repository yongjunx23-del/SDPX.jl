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

@testset "Fenchel replay work is homogeneous and finite" begin
    setprecision(BigFloat, 256) do
        p = BigFloat[-1, 1, 1]
        d = .-collect(SDPX.exp_barrier_gradient(p...))
        scale = BigFloat("1e40")
        out = Vector{BigFloat}(undef, 3)
        result = SDPX.exp_logarithmic_conjugate!(out, d / scale)
        @test norm(out - scale .* p, Inf) <=
              BigFloat(20000) * eps(BigFloat) * norm(scale .* p, Inf)
        @test result.root ==
              SDPX.exp_logarithmic_conjugate!(Vector{BigFloat}(undef, 3), d).root

        untouched = fill(BigFloat(-1), 3)
        @test_throws DomainError SDPX.exp_logarithmic_conjugate!(
            untouched, Float64[-1e308, 0, 1e308],
        )
        @test untouched == fill(BigFloat(-1), 3)
    end
end

@testset "Fenchel replay allowances are gradient-homogeneous and bounded" begin
    setprecision(BigFloat, 256) do
        T = BigFloat
        primal = T[-1, 1, 1]
        gradient = SDPX._exp_logarithmic_gradient_values(primal)
        dual = .-collect(gradient)
        shadow = Vector{T}(undef, 3)
        result = SDPX.exp_logarithmic_conjugate!(shadow, dual)
        y, z, l, psi = SDPX._exp_logarithmic_terms(shadow)

        replay_allowances = (root_residual; rho_value=result.root) ->
            SDPX._exp_logarithmic_replay_allowances(
                dual[1], dual[2], dual[3], rho_value, y, z, l,
                shadow[1], psi, gradient, root_residual,
            )
        base = replay_allowances(result.root_residual)
        @test all(isfinite, base)
        @test all(value -> value >= zero(T), base)

        # The helper returns gradient-unit bounds, so every component must
        # scale as lambda^-1 when primal and dual coordinates are scaled
        # inversely.  Call the helper directly rather than testing only the
        # public inverse, which could mask a defect in one allowance term.
        for lambda in (T("1e-20"), T("1e-8"), T("1e8"), T("1e20"))
            scaled = SDPX._exp_logarithmic_replay_allowances(
                dual[1] / lambda, dual[2] / lambda, dual[3] / lambda,
                result.root, y * lambda, z * lambda, l,
                shadow[1] * lambda, psi * lambda,
                ntuple(i -> gradient[i] / lambda, 3),
                result.root_residual,
            )
            @test all(isfinite, scaled)
            @test all(i -> isapprox(scaled[i] * lambda, base[i];
                                    rtol=T("1e-35"), atol=T("1e-100")), 1:3)
        end

        # A finite but excessive root error is unresolved by the half-coordinate
        # guard; a nonfinite root error must be rejected before any bound is
        # accepted as a usable allowance.
        @test_throws DomainError replay_allowances(T(10))
        @test_throws DomainError replay_allowances(T(Inf))
        @test_throws DomainError replay_allowances(result.root_residual;
                                                   rho_value=T(Inf))
    end
end
