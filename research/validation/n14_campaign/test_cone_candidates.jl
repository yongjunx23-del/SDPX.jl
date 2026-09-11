using SDPX, MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra, Test
@testset "Normalized feasibility candidates across cone families" begin
    for T in (Float64,Float64x4,BigFloat), family in (:soc,:psd,:exp,:power)
        model=SDPX.Model(T)
        SDPX.variable!(model,:unused,1;domain=SDPX.Reals())
        if family===:soc
            SDPX.constraint!(model,:cone,T[2,0.25,0.5],SDPX.LorentzCone())
        elseif family===:psd
            SDPX.constraint!(model,:cone,T[2 0.25;0.25 1],SDPX.PSDCone())
        elseif family===:exp
            SDPX.constraint!(model,:cone,T[0,1,2],SDPX.ExponentialCone())
        else
            SDPX.constraint!(model,:cone,T[2,1,0.25],SDPX.PowerCone(T(0.35)))
        end
        SDPX.objective!(model,SDPX.Minimize(),zero(T))
        canonical=SDPX.canonicalize(SDPX.compile_product_cone_model(model))
        b=SDPX.HSDState(canonical)
        gauge=T(2)^(-80)
        for i in eachindex(b.s)
            SDPX._store_owned_scalar!(b.s,i,b.b[i]*gauge)
        end
        b.tau=gauge;b.kappa=zero(T)
        SDPX._cert_residual!(b)
        before=deepcopy((b.x,b.y,b.s,b.A,b.b,b.c,b.tau,b.kappa))
        c=SDPX._normalized_optimality_candidate(b)
        @test c!==nothing
        @test SDPX.verify_optimal!(canonical,c,SDPX.alloc_zeros(T,b.n),
            SDPX.alloc_zeros(T,b.m),SDPX.alloc_zeros(T,b.m);tol=T(1e-8))
        @test before==(b.x,b.y,b.s,b.A,b.b,b.c,b.tau,b.kappa)
    end
end
