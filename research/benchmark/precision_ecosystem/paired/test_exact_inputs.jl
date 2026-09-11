using Test, Random
include("provenance.jl")
include("ExactInputs.jl")
const EI=ExactInputs
const MF=MultiFloats.Float64x4
const OUT=ENV["PAIRED_OUT"]
LinearAlgebra.BLAS.set_num_threads(1)
Q(x::MF)=sum(Rational{BigInt},x._limbs;init=big(0)//big(1))
before=provenance();save_toml(joinpath(OUT,"before.toml"),before)
try
    @testset "exact stored-input paired reference" begin
        @testset "binary64 integer decoding" begin
            for x in (0.,-0.,nextfloat(0.),-nextfloat(0.),floatmin(Float64),
                      floatmax(Float64),1.,-1.,2.0^-165,nextfloat(1.))
                @test EI.scaled(x)//(big(1)<<1074)==Rational{BigInt}(x)
            end
            @test_throws ArgumentError EI.scaled(Inf)
            @test_throws ArgumentError EI.scaled(NaN)
            x=EI.fullwidth(MersenneTwister(4))
            @test EI.words(EI.fromwords(EI.words(x)))==EI.words(x)
            @test EI.scaled(x)//(big(1)<<1074)==Q(x)
            @test_throws ArgumentError EI.fromwords(["0","0","0","0"])
            @test_throws ArgumentError EI.fromwords(["7ff0000000000000","0000000000000000","0000000000000000","0000000000000000"])
        end
        for family in ("fullwidth","paircancel","zero"),beta in (0.,0.5),k in (4,5)
            fixture=EI.build(3,k,2,family,beta,"test-$family-$beta-$k")
            A,B,C0,alpha,bb=EI.materialize(fixture)
            reference=EI.reference(fixture);ref=EI.unpack_reference(reference)
            # Independent rational scalar oracle, using Julia's rational limb
            # conversion rather than the IEEE word decoder under test.
            for j in 1:2,i in 1:3
                exact=sum((Q(A[i,t])*Q(B[t,j]) for t in 1:k);init=big(0)//big(1))+Q(bb)*Q(C0[i,j])
                @test ref["result"][i,j]//EI.UNIT==exact
            end
            encoded=joinpath(OUT,"fixture-$family-$beta-$k.toml")
            save_toml(encoded,fixture)
            reloaded=TOML.parsefile(encoded)
            @test EI.digest(EI.materialize(reloaded)...)==fixture["input_sha256"]
            @test EI.reference(reloaded)==reference
            occupancy=[count(x->!iszero(x._limbs[4]),X) for X in (A,B,C0)]
            @test family=="zero" ? all(iszero,occupancy) : occupancy==[length(A),length(B),length(C0)]
            if family=="paircancel"
                for t in 1:div(k,2),j in 1:2
                    @test EI.words(B[2t,j])==EI.words(B[2t-1,j])
                end
            end
            for strategy in (:direct,:experimental_mfa)
                C=copy(C0)
                config=MultiFloatLinearAlgebra.KernelConfig(;gemm_strategy=strategy,thread_count=1)
                MultiFloatLinearAlgebra.gemm!(C,A,B,alpha,bb;config)
                @test EI.metrics(C,ref)["pass"]
            end
            bad=deepcopy(fixture);bad["A"]["words"][1]="0000000000000000"
            if family!="zero";@test_throws Exception EI.materialize(bad);end
        end
        # Positive row/column tails plus odd-k, on both sides of the direct
        # default sixteen-column panel boundary. Not performance evidence.
        for (m,k,n) in ((7,5,1),(8,9,15),(9,9,16),(9,9,17)),beta in (0.,0.5)
            fixture=EI.build(m,k,n,"paircancel",beta,"tails-$m-$k-$n-$beta")
            A,B,C0,alpha,bb=EI.materialize(fixture);ref=EI.unpack_reference(EI.reference(fixture))
            for strategy in (:direct,:experimental_mfa)
                C=copy(C0)
                MultiFloatLinearAlgebra.gemm!(C,A,B,alpha,bb;config=MultiFloatLinearAlgebra.KernelConfig(;gemm_strategy=strategy,thread_count=Threads.nthreads()))
                @test EI.metrics(C,ref)["pass"]
            end
        end
        zero_fixture=EI.build(2,2,2,"zero",0.,"zero")
        _,_,Z,_,_=EI.materialize(zero_fixture);ref=EI.unpack_reference(EI.reference(zero_fixture))
        metric=EI.metrics(Z,ref)
        @test metric["pass"] && metric["zero_reference_exact"]
        @test metric["normwise_relative_error"]=="undefined_zero_denominator"
        @test metric["componentwise_relative"]["nonzero_count"]==0
        @test EI.distribution(ref["product_absolute"],ref["product"])["zero_count"]==4
        bad=copy(Z);bad[1,1]=MF(1e-60)
        @test !EI.metrics(bad,ref)["pass"]
        @test EI.metrics(bad,ref)["zero_reference_nonzero_errors"]==1
        overlap=copy(Z);overlap[1,1]=MF((1.,-1.,0.,0.))
        @test !EI.metrics(overlap,ref)["normalized"]
        @test !EI.metrics(overlap,ref)["pass"]
        nonfinite=copy(Z);nonfinite[1,1]=MF(Inf)
        @test !EI.metrics(nonfinite,ref)["pass"]
        @test_throws ArgumentError EI.build(257,2,2,"fullwidth",0.,"oversized")
        @test_throws ArgumentError EI.build(2,2,2,"fullwidth",1.,"beta")
    end
finally
    after=provenance();save_toml(joinpath(OUT,"after.toml"),after);@assert before==after
end
println("SOURCE_UNCHANGED_EXACT_REFERENCE_PASSED")
