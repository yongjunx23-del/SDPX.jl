using Test, SDPX, LinearAlgebra, SparseArrays
@testset "owned compatibility constraint contractions" begin
    for (T,p) in ((Float64,53),(BigFloat,256),(BigFloat,512))
        setprecision(BigFloat,p) do
            for k in (1,2,3),sparse_mode in (false,true)
                A1=[T(i+j) for i in 1:k,j in 1:k]
                A2=[T(i-2j) for i in 1:k,j in 1:k] # preserve nonsymmetric raw coefficients
                A3=zeros(T,k,k)
                source=permutedims(cat(A1,A2,A3;dims=3),(3,1,2))
                problem=SDPX.ingest(ones(T,3),[source],[zeros(T,k,k)],zeros(T,3,0),T[];
                    sparse=sparse_mode,validate=false,verbosity=0)
                x=T[0.5,-2,7];M=[T(2i-j) for i in 1:k,j in 1:k]
                original_x=Rational{BigInt}.(x);original_source=Rational{BigInt}.(source)
                P=SDPX.alloc_zeros(T,k,k)
                @test SDPX.buildP_owned!(P,problem.cons,1,x)===P
                reference=Rational{BigInt}.(A1)/2-2 .* Rational{BigInt}.(A2)
                @test Rational{BigInt}.(P)==reference
                v=SDPX.alloc_zeros(T,3);v[1]=T(7);v[2]=T(-3);v[3]=T(2)
                expected=Rational{BigInt}[7,-3,2]
                for i in 1:3
                    Ai=(A1,A2,A3)[i]
                    expected[i]-=sum(Rational{BigInt}(Ai[j])*Rational{BigInt}(M[j]) for j in eachindex(M))/2
                end
                @test SDPX.accumulate_v_owned!(v,problem.cons,1,M,T(-0.5))===v
                @test Rational{BigInt}.(v)==expected
                @test Rational{BigInt}.(x)==original_x
                @test Rational{BigInt}.(source)==original_source
                before=Rational{BigInt}.(P)
                SDPX._store_owned_scalar!(P,1,P[1]+one(T))
                @test Rational{BigInt}.(P)[2:end]==before[2:end]
                @test Rational{BigInt}.(source)==original_source
                @test_throws DimensionMismatch SDPX.buildP_owned!(SDPX.alloc_zeros(T,k+1,k+1),problem.cons,1,x)
                @test_throws DimensionMismatch SDPX.accumulate_v_owned!(SDPX.alloc_zeros(T,4),problem.cons,1,M,one(T))
            end
            Av=reshape(T[1],1,1);cons=SDPX.DenseCons{T}([Av])
            @test_throws ArgumentError SDPX.buildP_owned!(Av,cons,1,T[1])
            @test Av==reshape(T[1],1,1)
            Ai=sparse(ones(T,2,2))
            sparse_cons=SDPX.SparseCons{T}([[Ai]],[[1]],[[1]],[ones(T,3,1)])
            alias=reshape(nonzeros(Ai),2,2)
            @test_throws ArgumentError SDPX.buildP_owned!(alias,sparse_cons,1,T[1])
            @test Matrix(Ai)==ones(T,2,2)
            inactive=SDPX.SparseCons{T}([[spzeros(T,2,2)]],[Int[]],[Int[]],[zeros(T,3,0)])
            wrong=SDPX.alloc_zeros(T,3,3)
            for i in eachindex(wrong);SDPX._store_owned_scalar!(wrong,i,T(7));end
            @test_throws DimensionMismatch SDPX.buildP_owned!(wrong,inactive,1,T[1])
            @test all(==(T(7)),wrong)
            v=SDPX.alloc_zeros(T,1);SDPX._store_owned_scalar!(v,1,T(11))
            @test_throws DimensionMismatch SDPX.accumulate_v_owned!(v,inactive,1,wrong,one(T))
            @test v==T[11]
            proper=SDPX.alloc_zeros(T,2,2)
            @test SDPX.buildP_owned!(proper,inactive,1,T[1])===proper
            @test all(iszero,proper)
            @test SDPX.accumulate_v_owned!(v,inactive,1,proper,one(T))===v
            @test v==T[11]
        end
    end
end
