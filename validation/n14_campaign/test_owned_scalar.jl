using SDPX, Test
@testset "BigFloat owned arithmetic isolates MPFR limbs" begin
    for ambient in (128,256,512), bits in (128,256,512)
        setprecision(BigFloat,ambient) do
            x=BigFloat("0.1";precision=bits)
            expected=deepcopy(x)
            y=SDPX.owned_arithmetic_copy(BigFloat,x;precision_bits=bits)
            v=SDPX.owned_vector_copy(BigFloat,BigFloat[x,x];precision_bits=bits)
            @test precision(y)==bits && all(z->precision(z)==bits,v)
            @test y==expected && all(==(expected),v)
            SDPX._store_owned_scalar!(BigFloat[x],1,BigFloat(7;precision=bits))
            @test y==expected && all(==(expected),v)
            SDPX._store_owned_scalar!(v,1,BigFloat(9;precision=bits))
            @test v[2]==expected && y==expected && x==7
            low=SDPX.owned_arithmetic_copy(BigFloat,BigFloat("0.1";precision=512);precision_bits=128)
            @test precision(low)==128 && low==BigFloat("0.1";precision=128)
        end
    end
end
