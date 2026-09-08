using Test, SDPX, LinearAlgebra, SparseArrays
function _private_pattern_fixture(m,n;block=true)
    T=BigFloat
    # Preserve every structural slot, including explicit numerical zeros.
    ptr=Int[1+(j-1)*m for j in 1:n+1]
    rows=Int[i for j in 1:n for i in 1:m]
    values=T[i==j ? 1 : (iszero(mod(i+j,3)) ? (i-j)/8 : 0) for j in 1:n for i in 1:m]
    A=SparseMatrixCSC{T,Int}(m,n,ptr,rows,values)
    ranges=UnitRange{Int}[i:i for i in 1:m]
    theta=T[2i+1 for i in 1:m]
    cone=block ? SDPX.BlockProductConeLinearization{T}(
        Matrix{T}[reshape(T[t],1,1) for t in theta],zeros(T,m),ranges) :
        SDPX.ProductConeLinearization{T}(Matrix(Diagonal(theta)),zeros(T,m),ranges)
    rhs=SDPX.HSDNewtonRHS(zeros(T,m),zeros(T,n),one(T),zeros(T,m),one(T))
    SDPX.NewtonSystem(A,ones(T,m),ones(T,n),cone,one(T),one(T),rhs)
end
@testset "private scalar-LP pattern allocation and parity" begin
    saved_enabled=SDPX.structure_cache_stats().enabled
    try
        for bits in (256,512),enabled in (false,true),block in (false,true),
            (m,n) in ((0,0),(3,0),(0,2),(1,1),(3,2),(32,32))
            setprecision(BigFloat,bits) do
                SDPX.set_structure_cache_enabled!(enabled)
                system=_private_pattern_fixture(m,n;block)
                A=system.A;counts=SDPX._research_private_lp_counts(A)
                @test counts==(n=n,m=m,d=n+m,a=m*n,q=m*n+n+m)
                before=SDPX.structure_cache_stats()
                private=SDPX._research_private_lp_pattern(system)
                second=SDPX._research_private_lp_pattern(system)
                @test SDPX.structure_cache_stats()==before
                shared=SDPX._symmetric_core_pattern_from_validated(system,SDPX.IdentityRankBasis(BigFloat,n))
                for name in fieldnames(typeof(private))
                    @test getfield(private,name)==getfield(shared,name)
                    @test getfield(private,name)==getfield(second,name)
                end
                lengths=Dict(:ar_colptr=>n+1,:ar_rowval=>m*n,:colptr=>n+m+1,
                    :rowval=>counts.q,:ar_slots=>m*n,:theta_slots=>m,:x_diag_slots=>n,
                    :nzval=>counts.q,:block_ranges=>m,:block_shapes=>m)
                for (name,length_) in lengths
                    @test length(getfield(private,name))==length_
                    @test getfield(private,name)!==getfield(second,name)
                end
                @test !Base.mightalias(private.ar_colptr,A.colptr)
                @test !Base.mightalias(private.ar_rowval,A.rowval)
                @test all(x->precision(x)==bits,private.nzval)
                @test length(private.ar_slots)==length(A.nzval) # explicit zeros retained
                if !isempty(private.nzval)
                    second_values=Rational{BigInt}.(second.nzval)
                    Avalues=Rational{BigInt}.(A.nzval)
                    SDPX._core_store_owned!(private.nzval,1,BigFloat(99))
                    @test Rational{BigInt}.(second.nzval)==second_values
                    @test Rational{BigInt}.(A.nzval)==Avalues
                end
                if !isempty(private.ar_rowval)
                    oldrows=copy(A.rowval);private.ar_rowval[1]=m+1
                    @test A.rowval==oldrows
                    @test second.ar_rowval==oldrows
                end
            end
        end
    finally
        SDPX.set_structure_cache_enabled!(saved_enabled)
    end
    @test_throws ArgumentError SDPX._research_private_lp_counts(-1,1,0)
    @test_throws ArgumentError SDPX._research_private_lp_counts(32,33,0)
    @test_throws ArgumentError SDPX._research_private_lp_counts(1,1,2)
    @test_throws OverflowError SDPX._research_private_lp_counts(typemax(Int),1,0)
    @test_throws InexactError SDPX._research_private_lp_counts(big(typemax(Int))+1,0,0)
    for kind in (:pointer_length,:row_length,:value_length,:endpoint,:unordered,:duplicate,:row_bound)
        system=_private_pattern_fixture(3,2)
        A=system.A
        if kind==:pointer_length;pop!(A.colptr)
        elseif kind==:row_length;pop!(A.rowval)
        elseif kind==:value_length;pop!(A.nzval)
        elseif kind==:endpoint;A.colptr[end]-=1
        elseif kind==:unordered;A.rowval[1],A.rowval[2]=A.rowval[2],A.rowval[1]
        elseif kind==:duplicate;A.rowval[2]=A.rowval[1]
        else;A.rowval[1]=4
        end
        before=SDPX.structure_cache_stats()
        @test_throws Exception SDPX._research_private_lp_pattern(system)
        @test SDPX.structure_cache_stats()==before
    end
end
