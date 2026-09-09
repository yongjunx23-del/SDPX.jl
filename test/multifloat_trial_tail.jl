using Test, SDPX

# This kernel belongs to the optional MFLA extension. Validation loads it
# explicitly; package environments without that provider do not exercise SIMD.
if Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt) !== nothing
    @testset "MultiFloat SIMD trial update respects logical tails" begin
        MF = Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt).MultiFloats
        for T in (MF.Float64x2, MF.Float64x4)
            capacity = 68
            model = SDPX.Model(T)
            x = SDPX.variable!(model, :x, capacity; domain=SDPX.Reals())
            SDPX.constraint!(model, :ball, Any[one(T); collect(x)], SDPX.LorentzCone())
            SDPX.objective!(model, SDPX.Minimize(), -x[1])
            canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
            state = SDPX.ProductConeHSDState(canonical)
            base = state.base
            n0, m0 = base.n, base.m
            @test n0 == capacity && m0 == capacity + 1
            for (storage, denominator) in ((base.x,16),(base.dx,-32),
                    (base.s,8),(base.ds,-16),(base.y,4),(base.dy,-64))
                for i in eachindex(storage)
                    storage[i] = T(i) / T(denominator)
                end
            end
            # Padded backing storage makes a pre-fix failure deterministic and
            # safe: unchecked extra stores hit canaries, not another GC object.
            # Only this direct kernel sees shortened logical n/m; no solve runs.
            ns = get(ENV,"SDPX_TAIL_TEST_MINIMAL","0") == "1" ? (2,) :
                 (0,1,2,3,4,5,6,7,8,9,63,64,65,66,67,68)
            ms = get(ENV,"SDPX_TAIL_TEST_MINIMAL","0") == "1" ? (3,) :
                 (0,1,2,3,4,5,6,7,67,68,69)
            sentinel = -T(53)/T(4)
            try
                for n in ns, m in ms, alpha in (zero(T),T(1)/T(2),one(T))
                    base.n=n; base.m=m
                    fill!(base.xt,sentinel);fill!(base.st,sentinel);fill!(base.yt,sentinel)
                    @test SDPX._trial_point_vec4!(state,alpha)
                    for (out,current,direction,count_) in ((base.xt,base.x,base.dx,n),
                            (base.st,base.s,base.ds,m),(base.yt,base.y,base.dy,m))
                        expected=[i<=count_ ? current[i]+alpha*direction[i] : sentinel for i in eachindex(out)]
                        @test all(out[i]._limbs == expected[i]._limbs for i in eachindex(out))
                    end
                end
            finally
                base.n=n0; base.m=m0
            end
        end
    end
end
