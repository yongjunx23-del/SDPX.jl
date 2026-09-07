using SDPX, Test

@testset "recovered candidate cannot hide affine errors in embedding scale" begin
    for (T,bits) in ((Float64,53),(BigFloat,256),(BigFloat,512))
        setprecision(BigFloat,max(bits,256)) do
            tol=parse(T,"1e-10")
            model=SDPX.Model(T)
            variable=SDPX.variable!(model,:x,1;domain=SDPX.Reals())
            SDPX.constraint!(model,:upper,one(T)-variable[1],SDPX.Nonnegative())
            SDPX.objective!(model,SDPX.Minimize(),-variable[1])
            canonical=SDPX.canonicalize(SDPX.compile_product_cone_model(model))
            state=SDPX.HSDState(canonical)
            state.tau=one(T);state.kappa=tol/8
            SDPX._store_owned_scalar!(state.x,1,one(T))
            SDPX._store_owned_scalar!(state.s,1,zero(T))
            SDPX._store_owned_scalar!(state.y,1,one(T)+tol*T(101)/100)
            xo=T[13];so=T[13];yo=T[13]
            @test !SDPX.verify_optimal!(canonical,state,xo,so,yo;tol)
            @test xo==T[13] && so==T[13] && yo==T[13]
            SDPX._store_owned_scalar!(state.y,1,one(T)+tol/4)
            @test SDPX.verify_optimal!(canonical,state,xo,so,yo;tol)
            @test xo==T[1]
            @test SDPX._certificate_objective_scale(T(-1),T(-1))==1
            @test SDPX._certificate_objective_scale(T(-5),T(-3))==4
        end
    end
end
